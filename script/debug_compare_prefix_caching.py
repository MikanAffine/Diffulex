#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass
from typing import Any


# Enable request-level mapping traces before importing the engine stack.
os.environ.setdefault("DIFFULEX_TRACE_STEP_DETAILS", "1")

from diffulex import Diffulex, SamplingParams  # noqa: E402


@dataclass
class StepTrace:
    step_id: int
    status: str
    is_prefill: bool
    new_tokens: int
    num_cached_tokens: int
    contiguous_cached_prefix_len: int
    cache_len: int
    truncated_response: list[int]
    full_response: list[int]
    kv_mapping_trace: dict[str, Any] | None
    sampler_trace: dict[str, Any] | None


@dataclass
class RequestTrace:
    req_id: int
    completion_reason: str | None
    truncated_response: list[int]
    full_response: list[int]
    steps: list[StepTrace]


def _capture_req(req) -> StepTrace:
    return StepTrace(
        step_id=int(getattr(req, "nfe", 0)),
        status=str(getattr(req.status, "name", req.status)),
        is_prefill=bool(getattr(req, "is_prefilling", False)),
        new_tokens=int(getattr(req, "new_tokens", 0) or 0),
        num_cached_tokens=int(getattr(req, "num_cached_tokens", 0) or 0),
        contiguous_cached_prefix_len=int(getattr(req, "contiguous_in_cache_prefix_len", 0) or 0),
        cache_len=int(getattr(req, "cache_len", 0) or 0),
        truncated_response=list(getattr(req, "truncated_response", []) or []),
        full_response=list(getattr(req, "full_response", []) or []),
        kv_mapping_trace=copy.deepcopy(getattr(req, "_kv_mapping_trace", None)),
        sampler_trace=copy.deepcopy(getattr(req, "_sampler_trace", None)),
    )


def _run_one_request(llm, prompt: str | list[int], sampling_params: SamplingParams) -> RequestTrace:
    req_id = llm.add_request(prompt, sampling_params)
    traces: list[StepTrace] = []
    last_req = None

    while not llm.is_finished():
        reqs, _ = llm.step()
        for req in reqs:
            if req.req_id != req_id:
                continue
            last_req = req
            traces.append(_capture_req(req))

    if last_req is None:
        raise RuntimeError(f"Request {req_id} never appeared in step() output")

    return RequestTrace(
        req_id=req_id,
        completion_reason=getattr(last_req, "completion_reason", None),
        truncated_response=list(getattr(last_req, "truncated_response", []) or []),
        full_response=list(getattr(last_req, "full_response", []) or []),
        steps=traces,
    )


def _build_prompt(base_prompt: str, repeat_words: int) -> str:
    if repeat_words <= 0:
        return base_prompt
    filler = " ".join(["cacheprobe"] * repeat_words)
    return f"{base_prompt}\nContext: {filler}\nAnswer:"


def _extract_step_signature(step: StepTrace) -> dict[str, Any]:
    sampler_trace = step.sampler_trace or {}
    return {
        "accepted_ids_map": sampler_trace.get("accepted_ids_map", {}),
        "sampled_tokens_map": sampler_trace.get("sampled_tokens_map", {}),
        "true_local_ids_map": sampler_trace.get("true_local_ids_map", {}),
        "completion_prefix": step.truncated_response,
    }


def _compare_request_traces(
    no_cache_trace: RequestTrace,
    cache_trace: RequestTrace,
) -> dict[str, Any]:
    no_cache_steps = [_extract_step_signature(step) for step in no_cache_trace.steps]
    cache_steps = [_extract_step_signature(step) for step in cache_trace.steps]

    mismatches = []
    max_steps = max(len(no_cache_steps), len(cache_steps))
    for idx in range(max_steps):
        lhs = no_cache_steps[idx] if idx < len(no_cache_steps) else None
        rhs = cache_steps[idx] if idx < len(cache_steps) else None
        if lhs != rhs:
            mismatches.append(
                {
                    "step_index": idx,
                    "no_cache": lhs,
                    "cache": rhs,
                }
            )

    cache_hit_steps = [
        {
            "step_index": idx,
            "num_cached_tokens": step.num_cached_tokens,
            "contiguous_cached_prefix_len": step.contiguous_cached_prefix_len,
            "context_len": (step.kv_mapping_trace or {}).get("context_len", 0),
        }
        for idx, step in enumerate(cache_trace.steps)
        if step.num_cached_tokens > 0 or step.contiguous_cached_prefix_len > 0
    ]

    return {
        "completion_reason_equal": no_cache_trace.completion_reason == cache_trace.completion_reason,
        "truncated_response_equal": no_cache_trace.truncated_response == cache_trace.truncated_response,
        "full_response_equal": no_cache_trace.full_response == cache_trace.full_response,
        "step_signatures_equal": not mismatches,
        "mismatches": mismatches,
        "cache_hit_steps": cache_hit_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare multi_bd outputs with and without prefix caching.")
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--model-name", default="sdar", help="Diffulex model_name")
    parser.add_argument("--prompt", default="Question: What is 1+1?", help="Prompt prefix")
    parser.add_argument(
        "--repeat-words",
        type=int,
        default=96,
        help="Repeat a filler token this many times to ensure a cacheable prompt prefix.",
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-nfe", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-num-reqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force eager execution instead of allowing CUDA graph capture on decode steps.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full trace payload as JSON.")
    args = parser.parse_args()

    prompt = _build_prompt(args.prompt, args.repeat_words)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_nfe=args.max_nfe,
    )
    common = dict(
        model_name=args.model_name,
        decoding_strategy="multi_bd",
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        master_port=args.master_port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_reqs=args.max_num_reqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        buffer_size=args.buffer_size,
        block_size=args.block_size,
        enforce_eager=args.enforce_eager,
    )

    runs: dict[str, list[RequestTrace]] = {}
    for enable_prefix_caching in (False, True):
        llm = Diffulex(args.model, enable_prefix_caching=enable_prefix_caching, **common)
        try:
            run_key = "cache_on" if enable_prefix_caching else "cache_off"
            runs[run_key] = [
                _run_one_request(llm, prompt, sampling_params),
                _run_one_request(llm, prompt, sampling_params),
            ]
        finally:
            llm.exit()

    baseline_second = runs["cache_off"][1]
    cached_second = runs["cache_on"][1]
    comparison = _compare_request_traces(baseline_second, cached_second)

    payload = {
        "prompt": prompt,
        "cache_off_second": asdict(baseline_second),
        "cache_on_second": asdict(cached_second),
        "comparison": comparison,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("comparison:")
        print(json.dumps(comparison, ensure_ascii=False, indent=2))

    return 0 if comparison["truncated_response_equal"] and comparison["step_signatures_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

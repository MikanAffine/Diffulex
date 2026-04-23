from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig

import diffulex.distributed.parallel_state as parallel_state
from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata
from diffulex.config import Config, DecodingThresholds
from diffulex.model.auto_model import AutoModelForDiffusionLM
from diffulex.strategy.dmax.attention.metadata import DMaxAttnMetaData, fetch_dmax_attn_metadata


def _mock_single_rank() -> None:
    parallel_state.reset_parallel_state()
    parallel_state.PARALLEL_STATE = parallel_state.build_parallel_state_for_test(
        tp_size=1,
        ep_size=1,
        dp_size=1,
        world_size=1,
        global_rank=0,
    )


def _load_naive_iter(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["blocks"][0]["iterations"][0]


def _make_runtime_config(model_path: str, hf_config) -> Config:
    cfg = Config(
        model=model_path,
        model_name="llada2_mini",
        decoding_strategy="dmax",
        sampling_mode="edit",
        mask_token_id=156895,
        block_size=32,
        buffer_size=1,
        multi_block_prefix_full=False,
        token_merge_mode="dmax_topk",
        token_merge_top_k=1,
        token_merge_renormalize=True,
        token_merge_weight=1.0,
        decoding_thresholds=DecodingThresholds(
            add_block_threshold=0.1,
            semi_complete_threshold=0.9,
            accept_threshold=0.5,
            remask_threshold=0.4,
        ),
        use_lora=False,
        pre_merge_lora=False,
        max_num_batched_tokens=4096,
        max_num_reqs=1,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        data_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=1,
        enforce_eager=True,
        hf_config=hf_config,
        tokenizer_vocab_size=getattr(hf_config, "vocab_size", None),
        kv_cache_layout="unified",
        enable_prefix_caching=True,
    )
    return cfg


def _build_metadata(seq_len: int, *, device: torch.device, block_size: int = 32) -> DMaxAttnMetaData:
    metadata = DMaxAttnMetaData(
        is_prefill=[True],
        cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        slot_mapping=torch.full((seq_len,), -1, dtype=torch.int32, device=device),
        context_lens=torch.tensor([0], dtype=torch.int32, device=device),
        page_tables=torch.full((1, 1), -1, dtype=torch.int32, device=device),
        page_size=32,
        block_size=block_size,
        kv_cache_layout="unified",
    )
    metadata.init_multi_block(
        valid_slices=torch.tensor([seq_len], dtype=torch.int32, device=device),
        buffer_size=1,
        is_prefix_full=False,
        status_table=torch.tensor([0], dtype=torch.int32, device=device),
        prefix_lens=torch.tensor([0], dtype=torch.int32, device=device),
        padded_prefix_lens=torch.tensor([0], dtype=torch.int32, device=device),
    )
    metadata.init_token_merging(mask_token_id=156895)
    return metadata


def _tensor_row(x: torch.Tensor, idx: int) -> torch.Tensor:
    if x.dim() == 3:
        return x[0, idx]
    if x.dim() == 2:
        return x[idx]
    raise ValueError(f"Unsupported tensor rank: {x.dim()}")


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = (a.float() - b.float()).abs()
    return {"max": float(d.max()), "mean": float(d.mean()), "norm": float(d.norm())}


def run_harness(model_path: str, naive_trace_path: str, rel_pos: int, include_qkv: bool) -> dict:
    _mock_single_rank()
    set_fetch_fn_for_attn_metadata(fetch_dmax_attn_metadata)

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg = _make_runtime_config(model_path, hf_config)
    model = AutoModelForDiffusionLM.from_config(cfg).cuda().eval()
    model.model.enable_debug_dump(include_qkv=include_qkv)

    naive_iter = _load_naive_iter(naive_trace_path)
    input_ids = naive_iter["current_window_tokens"].to(torch.int64).cuda()
    positions = naive_iter["current_window_positions"].to(torch.int64).cuda()

    metadata = _build_metadata(len(input_ids), device=input_ids.device)
    # Refresh the global DMax metadata instance to point to our tensors.
    import diffulex.strategy.dmax.attention.metadata as dmax_meta_mod
    dmax_meta_mod.DMAX_ATTN_METADATA = metadata

    with torch.inference_mode():
        hidden = model(input_ids, positions)
        logits = model.compute_logits(hidden)
    debug = model.model.pop_last_debug_dump()

    abs_pos = 128 + rel_pos
    naive_logits = _tensor_row(naive_iter["lm_head_logits"], rel_pos).float()
    engine_logits = logits[abs_pos].detach().cpu().float()

    summary = {
        "rel_pos": rel_pos,
        "abs_pos": abs_pos,
        "top1_token": int(torch.argmax(engine_logits).item()),
        "candidate_logits": {},
        "layer_diffs": {},
    }
    for token_id in [300, 56592, 38825, 31073, 2251, 3310, 1788]:
        summary["candidate_logits"][str(token_id)] = {
            "naive": float(naive_logits[token_id]),
            "engine": float(engine_logits[token_id]),
            "delta": float(engine_logits[token_id] - naive_logits[token_id]),
        }

    naive_hidden = naive_iter["hidden_states"]
    for layer_idx in range(20):
        key = f"layer_{layer_idx}_out"
        if key not in debug or key not in naive_hidden:
            break
        summary["layer_diffs"][key] = _diff(debug[key][abs_pos].detach().cpu(), _tensor_row(naive_hidden[key], rel_pos))

    if "moe" in debug:
        summary["moe_keys"] = list(debug["moe"].keys())[:20]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-forward DMax step1 harness on one GPU.")
    parser.add_argument("--model-path", default="/data1/ckpts/Zigeng/DMax-Math-16B")
    parser.add_argument("--naive-trace", default="benchmark_results/sample_00_tensor_trace.pt")
    parser.add_argument("--rel-pos", type=int, default=30)
    parser.add_argument("--include-qkv", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    summary = run_harness(
        model_path=args.model_path,
        naive_trace_path=args.naive_trace,
        rel_pos=args.rel_pos,
        include_qkv=args.include_qkv,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

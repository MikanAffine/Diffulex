from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_pt(path: str) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _as_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    if torch.is_tensor(x):
        return x.tolist()
    return list(x)


def _tail(x: Any, n: int = 6) -> list[Any]:
    seq = _as_list(x)
    return seq[-n:]


def _tensor_row(x: torch.Tensor, idx: int) -> torch.Tensor:
    if x.dim() == 3:
        return x[0, idx]
    if x.dim() == 2:
        return x[idx]
    raise ValueError(f"Unsupported tensor rank: {x.dim()}")


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = (a.float() - b.float()).abs()
    return {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "norm": float(d.norm()),
    }


def build_summary(
    naive_trace_path: str,
    engine_dump_path: str,
    *,
    rel_pos: int,
    candidate_ids: list[int],
) -> dict[str, Any]:
    naive = _load_pt(naive_trace_path)
    engine = _load_pt(engine_dump_path)

    naive_block = naive["blocks"][0]
    naive_iter = naive_block["iterations"][0]
    engine_meta = engine["meta"]
    engine_inputs = engine["inputs"]
    engine_hidden = engine["hidden_states"]

    active_block = engine["blocks"][0]
    block_start = int(active_block["start"])
    abs_pos = block_start + rel_pos

    summary: dict[str, Any] = {
        "naive_trace_path": str(Path(naive_trace_path).resolve()),
        "engine_dump_path": str(Path(engine_dump_path).resolve()),
        "block_start": block_start,
        "block_end": int(active_block["end"]),
        "rel_pos": rel_pos,
        "abs_pos": abs_pos,
        "meta": {
            "engine_req_id": int(engine_meta["req_id"]),
            "engine_step_nfe": int(engine_meta["step_nfe"]),
            "engine_req_status": str(engine_meta["req_status"]),
            "engine_running_len": int(engine_meta["running_len"]),
            "naive_window_len": int(len(naive_iter["current_window_tokens"])),
            "engine_input_len": int(len(engine_inputs["input_ids"])),
        },
        "window_check": {
            "naive_positions_head": _as_list(naive_iter["current_window_positions"][:8]),
            "naive_positions_tail": _tail(naive_iter["current_window_positions"], 8),
            "engine_positions_head": _as_list(engine_inputs["positions"][:8]),
            "engine_positions_tail": _tail(engine_inputs["positions"], 8),
            "naive_tokens_tail": _tail(naive_iter["current_window_tokens"], 8),
            "engine_tokens_tail": _tail(engine_inputs["input_ids"], 8),
        },
        "block_after": {
            "naive_tail": _tail(naive_iter["block_after"], 6),
            "engine_tail": _tail(active_block["token_ids"], 6),
        },
        "top1": {
            "naive_tail_tokens": _tail(naive_iter["top1_tokens"], 6),
            "engine_tail_tokens": _as_list(engine["logits"]["topk_ids"][block_start + 26 : block_start + 32, 0]),
            "naive_tail_conf": _tail(naive_iter["top1_confidence"], 6),
            "engine_tail_conf": _as_list(engine["logits"]["topk_probs"][block_start + 26 : block_start + 32, 0]),
            "naive_rel_pos_token": int(_as_list(naive_iter["top1_tokens"])[rel_pos]),
            "naive_rel_pos_conf": float(_as_list(naive_iter["top1_confidence"])[rel_pos]),
            "engine_abs_pos_token": int(engine["logits"]["topk_ids"][abs_pos, 0].item()),
            "engine_abs_pos_conf": float(engine["logits"]["topk_probs"][abs_pos, 0].item()),
        },
    }

    naive_logits = _tensor_row(naive_iter["lm_head_logits"], rel_pos).float()
    engine_logits = engine["logits"]["full"][abs_pos].float()
    candidate_summary = []
    for token_id in candidate_ids:
        candidate_summary.append(
            {
                "token_id": int(token_id),
                "naive_logit": float(naive_logits[token_id]),
                "engine_logit": float(engine_logits[token_id]),
                "delta": float(engine_logits[token_id] - naive_logits[token_id]),
            }
        )
    summary["candidate_logits"] = candidate_summary

    per_layer = []
    naive_hidden = naive_iter["hidden_states"]
    for layer_idx in range(64):
        key = f"layer_{layer_idx}_out"
        if key not in naive_hidden or key not in engine_hidden:
            break
        per_layer.append(
            {
                "layer": layer_idx,
                **_diff_stats(_tensor_row(naive_hidden[key], rel_pos), engine_hidden[key][abs_pos]),
            }
        )
    summary["per_layer_hidden_diff"] = per_layer
    if "final_norm" in naive_hidden and "final_norm" in engine_hidden:
        summary["final_norm_diff"] = _diff_stats(
            _tensor_row(naive_hidden["final_norm"], rel_pos),
            engine_hidden["final_norm"][abs_pos],
        )

    decoder_diff = []
    naive_decoder = naive_iter.get("decoder_debug", {})
    engine_decoder = engine_hidden.get("decoder", {})
    for layer_idx in range(8):
        row = {"layer": layer_idx}
        found = False
        for suffix in (
            "post_input_layernorm",
            "post_attention_residual",
            "post_attention_layernorm",
            "layer_out",
        ):
            key = f"layer_{layer_idx}_{suffix}"
            if key not in naive_decoder or key not in engine_decoder:
                continue
            row[suffix] = _diff_stats(_tensor_row(naive_decoder[key], rel_pos), engine_decoder[key][abs_pos])
            found = True
        if found:
            decoder_diff.append(row)
    summary["decoder_diff"] = decoder_diff
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DMax block4 step1 naive vs engine dumps.")
    parser.add_argument(
        "--naive-trace",
        default="benchmark_results/sample_00_tensor_trace.pt",
        help="Path to naive/reference tensor trace.",
    )
    parser.add_argument(
        "--engine-dump",
        default="benchmark_results/diffulex_engine_debug_dump/req_1_step_001_rank_0.pt",
        help="Path to engine step1 dump.",
    )
    parser.add_argument("--rel-pos", type=int, default=30, help="Relative position inside active block to inspect.")
    parser.add_argument(
        "--candidate-ids",
        default="300,56592,38825,31073,2251,3310,1788",
        help="Comma-separated token ids to compare in logits.",
    )
    parser.add_argument("--output-json", default="", help="Optional path to write the summary JSON.")
    args = parser.parse_args()

    candidate_ids = [int(x) for x in args.candidate_ids.split(",") if x.strip()]
    summary = build_summary(
        args.naive_trace,
        args.engine_dump,
        rel_pos=args.rel_pos,
        candidate_ids=candidate_ids,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

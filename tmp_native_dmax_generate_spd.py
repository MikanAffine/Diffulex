from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HF generate_spd_trace and dump per-sample traces.")
    parser.add_argument("--model-path", default="/data1/ckpts/Zigeng/DMax-Math-16B")
    parser.add_argument("--docs-path", default="diffulex_bench/data/GSM8K.json")
    parser.add_argument("--output-dir", default="benchmark_results/native_dmax_generate_spd_trace")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--minimal-topk", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--trace-topk", type=int, default=5)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--tensor-trace", action="store_true", help="Save per-step tensor trace as .pt files.")
    parser.add_argument(
        "--dump-hidden-states",
        action="store_true",
        help="Include active-block hidden states for every layer in tensor traces.",
    )
    parser.add_argument(
        "--dump-router-logits",
        action="store_true",
        help="Include per-layer router logits in tensor traces.",
    )
    parser.add_argument(
        "--dump-full-logits",
        action="store_true",
        help="Include full active-block logits in tensor traces. Files can become large.",
    )
    parser.add_argument(
        "--dump-attention-debug",
        action="store_true",
        help="Include attention-path debug tensors such as qkv, rotary q/k, and attention outputs.",
    )
    parser.add_argument(
        "--dump-decoder-debug",
        action="store_true",
        help="Include decoder residual/layernorm intermediate tensors.",
    )
    parser.add_argument(
        "--dump-moe-debug",
        action="store_true",
        help="Include MoE gate/router/topk/output debug tensors.",
    )
    parser.add_argument(
        "--comprehensive-dump",
        action="store_true",
        help="Enable tensor trace plus hidden/router/logits/attention/decoder/moe debug tensors together.",
    )
    return parser


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = build_parser().parse_args()

    model_path = Path(args.model_path)
    docs_path = Path(args.docs_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = json.loads(docs_path.read_text())[: args.limit]
    dtype = resolve_dtype(args.torch_dtype)
    if args.comprehensive_dump:
        args.tensor_trace = True
        args.dump_hidden_states = True
        args.dump_router_logits = True
        args.dump_full_logits = True
        args.dump_attention_debug = True
        args.dump_decoder_debug = True
        args.dump_moe_debug = True

    print("loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)

    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device,
        local_files_only=True,
    )
    model = model.to(dtype)
    model.eval()

    if not hasattr(model, "generate_spd_trace"):
        raise AttributeError(
            "Model is missing generate_spd_trace(). Please add that function to modeling_llada2_moe.py first."
        )

    results = []
    for i, doc in enumerate(docs):
        prompt = doc["question"] + "\nLet's think step by step\n"
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)

        trace_path = output_dir / f"sample_{i:02d}_trace.json"
        tensor_trace_path = output_dir / f"sample_{i:02d}_tensor_trace.pt" if args.tensor_trace else None
        with torch.inference_mode():
            nfe, generated_tokens = model.generate_spd_trace(
                inputs=input_ids,
                gen_length=args.gen_length,
                block_length=args.block_length,
                steps=args.steps,
                minimal_topk=args.minimal_topk,
                threshold=args.threshold,
                trace_path=str(trace_path),
                tensor_trace_path=str(tensor_trace_path) if tensor_trace_path is not None else None,
                trace_topk=args.trace_topk,
                dump_hidden_states=args.dump_hidden_states,
                dump_router_logits=args.dump_router_logits,
                dump_full_logits=args.dump_full_logits,
                dump_attention_debug=args.dump_attention_debug,
                dump_decoder_debug=args.dump_decoder_debug,
                dump_moe_debug=args.dump_moe_debug,
                return_trace=False,
            )

        text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        item = {
            "index": i,
            "ground_truth_answer": doc.get("ground_truth_answer"),
            "nfe": int(nfe) if not isinstance(nfe, int) else nfe,
            "output": text,
            "trace_path": str(trace_path),
            "tensor_trace_path": str(tensor_trace_path) if tensor_trace_path is not None else None,
        }
        results.append(item)
        print(f"=== sample {i} ===", flush=True)
        print(f"nfe: {item['nfe']}", flush=True)
        print(f"trace: {trace_path}", flush=True)
        if tensor_trace_path is not None:
            print(f"tensor trace: {tensor_trace_path}", flush=True)
        print(text, flush=True)

    out_path = output_dir / "native_dmax_generate_spd_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

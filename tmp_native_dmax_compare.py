from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/home/jyj/workspace/denglab-diffulex/DMax/dInfer/python")
from dinfer.decoding_llada_origin.serving import DiffusionLLMServing, SamplingParams


def main() -> None:
    model_path = "/data1/ckpts/Zigeng/DMax-Math-16B"
    docs = json.loads(Path("diffulex_bench/data/GSM8K.json").read_text())[:3]
    print("loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": doc["question"] + "\nLet's think step by step\n"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for doc in docs
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    print("input shape", tuple(input_ids.shape), flush=True)

    sample_params = SamplingParams(
        threshold=0.5,
        cache="prefix",
        temperature=0.0,
        early_stop=True,
        cont_weight=0,
        prefix_look=0,
        after_look=0,
        warmup_steps=0,
        enable_torch_compile=True,
        mask_id=156895,
        eos_id=156892,
        parallel_decoding="threshold",
        use_credit=False,
        use_bd=True,
        max_length=2048,
    )

    server = DiffusionLLMServing(
        model_path,
        model_type="llada2-mini",
        sample_params=sample_params,
        num_gpus=4,
        dp_size=1,
        tpep_size=4,
        backend="sglang",
        timeout=600,
    )

    try:
        out = server.generate(input_ids, gen_length=256, block_length=32)
        nfe = server.num_forwards
        results = []
        pad_id = tokenizer.pad_token_id
        for i in range(out.shape[0]):
            full_ids = out[i].tolist()
            while full_ids and full_ids[-1] == sample_params.eos_id:
                full_ids.pop()
            prompt_len = int(attention_mask[i].sum().item()) if pad_id is not None else len(input_ids[i])
            gen_ids = full_ids[prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            results.append(
                {
                    "index": i,
                    "ground_truth_answer": docs[i].get("ground_truth_answer"),
                    "nfe": nfe,
                    "output": text,
                }
            )
            print(f"=== sample {i} ===", flush=True)
            print(f"nfe: {nfe}", flush=True)
            print(text, flush=True)
        out_path = Path("benchmark_results/native_dmax_serving_tp4_limit3.json")
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"saved to {out_path}", flush=True)
    finally:
        server.stop_serving()


if __name__ == "__main__":
    main()

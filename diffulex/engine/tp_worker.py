import atexit

import torch.multiprocessing as mp

from tqdm.auto import tqdm
from time import perf_counter
from dataclasses import fields
from transformers import AutoTokenizer

from diffulex.config import Config
from diffulex.sampling_params import SamplingParams
from diffulex.engine.request import AutoReq
from diffulex.engine.scheduler import AutoScheduler, SchedulerBase
from diffulex.engine.model_runner import AutoModelRunner
from diffulex.mixin.quantization.engine.tp_worker import TPWorkerQuantizationMixin
from diffulex.mixin.async_engine.engine.tp_worker import DiffulexTPWorkerAsyncMixin
from diffulex.logger import get_logger

logger = get_logger(__name__)


class DiffulexTPWorker(TPWorkerQuantizationMixin, DiffulexTPWorkerAsyncMixin):
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=AutoModelRunner.from_config, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        self.model_runner = AutoModelRunner.from_config(config, 0, self.events)
        self.scheduler: SchedulerBase = AutoScheduler.from_config(config)
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if getattr(self, "_exited", False):
            return
        self._exited = True
        if hasattr(self, "model_runner") and self.model_runner is not None:
            try:
                self.model_runner.call("exit")
            except Exception:
                pass
            try:
                del self.model_runner
            except Exception:
                pass
        for p in getattr(self, "ps", []):
            try:
                p.join()
            except Exception:
                pass

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        req = AutoReq.create(self.config, prompt, sampling_params)
        req.page_size = self.config.kv_cache_page_size
        self.scheduler.add(req)
        # Return req_id so caller can build a stable mapping
        return req.req_id

    def step(self):
        self.clear_step_act_quant_cache()
        reqs, is_prefill = self.scheduler.schedule()
        sample_output = self.model_runner.call("run", reqs, is_prefill)
        n_diff_steps = self.scheduler.postprocess(reqs, sample_output)
        outputs = [(req.req_id, req.completion_token_ids) for req in reqs if req.is_finished]
        num_tokens = sum(req.num_tokens for req in reqs) if is_prefill else sum(req.new_tokens for req in reqs)
        # Diffusion decoding modifies tokens in-place; we currently don't stream intermediate edits
        deltas = []
        return outputs, num_tokens, is_prefill, n_diff_steps, deltas

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # Map internal req_id -> input index to keep output order stable
        reqid_to_idx = {}
        for idx, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            rid = self.add_request(prompt, sp)
            reqid_to_idx[rid] = idx
        outputs = [None] * len(prompts)
        # Track token/time totals for correct average throughput reporting.
        prefill_total_tokens = 0
        decode_total_tokens = 0
        prefill_total_time = 0.0
        decode_total_time = 0.0
        prefill_steps = 0
        decode_steps = 0
        n_steps = 0
        n_diff_steps = [-1] * len(prompts)
        while not self.is_finished():
            n_steps += 1
            t = perf_counter()
            output, num_tokens, is_prefill, cur_n_diff_steps, _ = self.step()
            dt = perf_counter() - t

            # Accumulate totals to compute average throughput correctly.
            if is_prefill:
                prefill_steps += 1
                prefill_total_tokens += int(num_tokens)
                prefill_total_time += float(dt)
            else:
                decode_steps += 1
                decode_total_tokens += int(num_tokens)
                decode_total_time += float(dt)

            if use_tqdm:
                avg_prefill_throughput = (
                    prefill_total_tokens / prefill_total_time if prefill_total_time > 0 else 0.0
                )
                avg_decode_throughput = (
                    decode_total_tokens / decode_total_time if decode_total_time > 0 else 0.0
                )
                pbar.set_postfix({
                    "Prefill(avg)": f"{int(avg_prefill_throughput)}tok/s",
                    "Decode(avg)": f"{int(avg_decode_throughput)}tok/s",
                })
            if cur_n_diff_steps:
                for req_id, n_step in cur_n_diff_steps.items():
                    if req_id in reqid_to_idx and n_step >= 0:
                        n_diff_steps[reqid_to_idx[req_id]] = n_step
            for req_id, token_ids in output:
                if req_id in reqid_to_idx:
                    outputs[reqid_to_idx[req_id]] = token_ids
                if use_tqdm:
                    pbar.update(1)
                    
        avg_prefill_throughput = (
            prefill_total_tokens / prefill_total_time if prefill_total_time > 0 else 0.0
        )
        avg_decode_throughput = (
            decode_total_tokens / decode_total_time if decode_total_time > 0 else 0.0
        )
        avg_prefill_step_ms = (
            (prefill_total_time / prefill_steps) * 1000.0 if prefill_steps > 0 else 0.0
        )
        avg_decode_step_ms = (
            (decode_total_time / decode_steps) * 1000.0 if decode_steps > 0 else 0.0
        )
        logger.info(
            "Finished in %d steps (prefill=%d, decode=%d). "
            "Prefill: %d tok in %.2fs (avg %.2f tok/s, %.2f ms/step). "
            "Decode: %d tok in %.2fs (avg %.2f tok/s, %.2f ms/step).",
            n_steps,
            prefill_steps,
            decode_steps,
            prefill_total_tokens,
            prefill_total_time,
            avg_prefill_throughput,
            avg_prefill_step_ms,
            decode_total_tokens,
            decode_total_time,
            avg_decode_throughput,
            avg_decode_step_ms,
        )
        # Ensure all outputs are present
        assert all(toks is not None for toks in outputs), "Some reqs did not produce outputs"
        outputs = [{
            "text": self.tokenizer.decode(token_ids).split(self.tokenizer.eos_token)[0],
            "token_ids": token_ids[:token_ids.index(self.config.eos)] if self.config.eos in token_ids else token_ids,
            "n_diff_steps": n_diff_step,
        } for token_ids, n_diff_step in zip(outputs, n_diff_steps)]
        if use_tqdm:
            pbar.close()
        return outputs

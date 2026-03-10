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
from diffulex.mixin.quantization.engine.tp_worker import (
    DiffulexTPWorkerQuantizationMixin,
)
from diffulex.mixin.async_engine.engine.tp_worker import DiffulexTPWorkerAsyncMixin
from diffulex.utils.stats import GenerationStats
from diffulex.logger import get_logger

logger = get_logger(__name__)


class DiffulexTPWorker(DiffulexTPWorkerQuantizationMixin, DiffulexTPWorkerAsyncMixin):
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
        return req.req_id

    def step(self):
        self.clear_step_act_quant_cache()
        reqs, is_prefill = self.scheduler.schedule()
        sample_output = self.model_runner.call("run", reqs)
        num_nfes = self.scheduler.postprocess(reqs, sample_output)
        outputs = [(req.req_id, req.completion_token_ids) for req in reqs if req.is_finished]
        num_tokens = sum(req.num_tokens for req in reqs) if is_prefill else sum(req.new_tokens for req in reqs)
        deltas = []
        return outputs, num_tokens, is_prefill, num_nfes, deltas

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Diffulex Generating", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        req_id_to_prompt_id = {}
        for prompt_id, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            req_id = self.add_request(prompt, sp)
            req_id_to_prompt_id[req_id] = prompt_id

        outputs = [None] * len(prompts)
        stats = GenerationStats(len(prompts))
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, is_prefill, num_nfes, _ = self.step()
            dt = perf_counter() - t

            stats.record_step(num_tokens, is_prefill, dt)
            stats.update_num_nfes(req_id_to_prompt_id, num_nfes)

            if use_tqdm:
                pbar.set_postfix(stats.postfix())

            for req_id, token_ids in output:
                if req_id in req_id_to_prompt_id:
                    outputs[req_id_to_prompt_id[req_id]] = token_ids

                if use_tqdm:
                    pbar.update(1)

        if use_tqdm:
            pbar.close()

        stats.log_summary()
        if not all(toks is not None for toks in outputs):
            logger.warning("Some reqs did not produce outputs")

        outputs = [
            dict(
                text=self.tokenizer.decode(token_ids).split(self.tokenizer.eos_token)[0],
                token_ids=token_ids[: token_ids.index(self.config.eos)] if self.config.eos in token_ids else token_ids,
                num_nfes=cur_req_num_nfes,
            )
            for token_ids, cur_req_num_nfes in zip(outputs, stats.num_nfes)
        ]

        return outputs

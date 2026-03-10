from diffulex.logger import get_logger

logger = get_logger(__name__)


class GenerationStats:
    """Accumulates generation throughput and timing statistics."""

    def __init__(self, num_prompts: int):
        self.n_steps = 0
        self.prefill_tokens = 0
        self.prefill_time = 0.0
        self.prefill_steps = 0
        self.decode_tokens = 0
        self.decode_time = 0.0
        self.decode_steps = 0
        self.num_nfes = [-1] * num_prompts

    def record_step(self, num_tokens: int, is_prefill: bool, dt: float):
        self.n_steps += 1
        if is_prefill:
            self.prefill_steps += 1
            self.prefill_tokens += int(num_tokens)
            self.prefill_time += float(dt)
        else:
            self.decode_steps += 1
            self.decode_tokens += int(num_tokens)
            self.decode_time += float(dt)

    def update_num_nfes(self, reqid_to_idx: dict, num_nfes: dict):
        if not num_nfes:
            return
        for req_id, num_nfe in num_nfes.items():
            if req_id in reqid_to_idx and num_nfe >= 0:
                self.num_nfes[reqid_to_idx[req_id]] = num_nfe

    @property
    def avg_prefill_throughput(self) -> float:
        return self.prefill_tokens / self.prefill_time if self.prefill_time > 0 else 0.0

    @property
    def avg_decode_throughput(self) -> float:
        return self.decode_tokens / self.decode_time if self.decode_time > 0 else 0.0

    @property
    def avg_prefill_step_ms(self) -> float:
        return (self.prefill_time / self.prefill_steps) * 1000.0 if self.prefill_steps > 0 else 0.0

    @property
    def avg_decode_step_ms(self) -> float:
        return (self.decode_time / self.decode_steps) * 1000.0 if self.decode_steps > 0 else 0.0

    def postfix(self) -> dict:
        return {
            "Prefill(avg)": f"{int(self.avg_prefill_throughput)}tok/s",
            "Decode(avg)": f"{int(self.avg_decode_throughput)}tok/s",
        }

    def log_summary(self):
        logger.info(
            "Finished in %d steps (prefill=%d, decode=%d). "
            "Prefill: %d tok in %.2fs (avg %.2f tok/s, %.2f ms/step). "
            "Decode: %d tok in %.2fs (avg %.2f tok/s, %.2f ms/step).",
            self.n_steps,
            self.prefill_steps,
            self.decode_steps,
            self.prefill_tokens,
            self.prefill_time,
            self.avg_prefill_throughput,
            self.avg_prefill_step_ms,
            self.decode_tokens,
            self.decode_time,
            self.avg_decode_throughput,
            self.avg_decode_step_ms,
        )

from diffulex.config import Config
from diffulex.engine.request import DllmReq, AutoReq
from diffulex.sampling_params import SamplingParams


@AutoReq.register("multi_block_diffusion", is_default=True)
class MultiBDReq(DllmReq):
    """Req for Multi-Block Diffusion strategy. Accepts config for AutoReq.create()."""

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        config: Config | None = None,
    ):
        super().__init__(token_ids, sampling_params)
        # config is passed by AutoReq.create(); req attributes are set in init_multi_block()
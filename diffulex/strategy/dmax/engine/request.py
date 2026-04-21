from diffulex.config import Config
from diffulex.engine.request import AutoReq
from diffulex.sampling_params import SamplingParams
from diffulex.attention.metadata import is_warming_up
from diffulex.strategy_template.token_merging_multi_block.engine.request import (
    TokenMergingMultiBlockReqTemplate,
)


@AutoReq.register("dmax")
class DMaxReq(TokenMergingMultiBlockReqTemplate):
    """Req for DMax-style block diffusion with token merging."""

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        config: Config | None = None,
    ):
        super().__init__(token_ids, sampling_params)
        if config is None:
            raise ValueError("DMaxReq requires config to initialize token-merge state.")
        if is_warming_up():
            # Used for warming up token merging
            self.init_token_merging_multi_block(config)

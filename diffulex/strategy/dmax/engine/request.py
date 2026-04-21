from diffulex.config import Config
from diffulex.engine.request import AutoReq
from diffulex.sampling_params import SamplingParams
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
        self.init_token_merging_multi_block(config)

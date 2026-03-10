from __future__ import annotations

from typing import TYPE_CHECKING

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager, KVCacheManagerBase

if TYPE_CHECKING:
    from diffulex.engine.request import DllmReq


@AutoKVCacheManager.register("d2f", is_default=True)
class D2fKVCacheManager(KVCacheManagerBase):
    def __init__(self, config: Config):
        super().__init__(config)

    def can_append(self, req: "DllmReq") -> bool:
        return self.can_append_multi_block(req)

    def may_append(self, req: "DllmReq") -> None:
        self.may_append_multi_block(req)

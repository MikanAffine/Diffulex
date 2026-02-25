from __future__ import annotations

from typing import TYPE_CHECKING

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager, KVCacheManagerBase

if TYPE_CHECKING:
    from .request import FDV2Req


@AutoKVCacheManager.register("fast_dllm_v2", is_default=True)
class FDV2KVCacheManager(KVCacheManagerBase):
    def __init__(self, config: Config):
        super().__init__(config)

    def can_append(self, req: "FDV2Req") -> bool:
        return len(self.free_page_ids) >= (req.cached_or_caching_num_tokens % self.page_size == 1)

    def may_append(self, req: "FDV2Req") -> None:
        if req.cached_or_caching_num_tokens == 0:
            return
        page_table = req.page_table
        if not page_table:
            return
        last_page = self.pages[page_table[-1]]
        if req.cached_or_caching_num_tokens // self.page_size == len(req.page_table):
            if last_page.hash == -1:
                prev_end_token = req.cached_or_caching_num_tokens - req.caching_num_tokens - 1
                prev_page_id = prev_end_token // self.page_size
                if 0 <= prev_page_id < req.num_pages:
                    token_ids: list[int] = req.page(prev_page_id)
                    prefix = self.pages[page_table[-2]].hash if len(page_table) > 1 else -1
                    h = self.compute_hash(token_ids, prefix)
                    last_page.update(h, token_ids)
                    self.hash_to_page_id[h] = last_page.page_id
            page_id = self.free_page_ids[0]
            self._allocate_page(page_id)
            page_table.append(page_id)
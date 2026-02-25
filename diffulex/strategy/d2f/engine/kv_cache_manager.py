from __future__ import annotations

from typing import TYPE_CHECKING

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager, KVCacheManagerBase

if TYPE_CHECKING:
    from .request import D2FReq


@AutoKVCacheManager.register("d2f", is_default=True)
class D2FKVCacheManager(KVCacheManagerBase):
    def __init__(self, config: Config):
        super().__init__(config)

    def _required_kv_blocks(self, req: "D2FReq") -> int:
        """How many KV-cache blocks this sequence needs *now* for cached+to-cache tokens.

        NOTE: In diffusion decoding, a single decode step may move multiple tokens into
        "to_cache", which can cross multiple KV blocks. So we must ensure block_table
        is large enough for all cached_or_caching tokens, not just append one block.
        """
        n = req.cached_or_caching_num_tokens
        if n <= 0:
            return 0
        # Need enough blocks to cover token indices [0, n-1].
        return (n + self.page_size - 1) // self.page_size

    def can_append(self, req: "D2FReq") -> bool:
        # We may need to allocate multiple blocks in one step (cached_or_caching can jump).
        required = self._required_kv_blocks(req)
        missing = max(0, required - len(req.block_table))
        return len(self.free_block_ids) >= missing

    def may_append(self, req: "D2FReq") -> None:
        if req.cached_or_caching_num_tokens == 0:
            return
        block_table = req.block_table
        if not block_table:
            # Defensive: allocate() should have populated it for prefill/prompt, but don't crash here.
            return

        required = self._required_kv_blocks(req)
        # Allocate enough KV blocks to cover all cached_or_caching tokens.
        while len(block_table) < required:
            last_block = self.blocks[block_table[-1]]
            # Preserve the existing "finalize previous block hash" behavior before moving on.
            if last_block.hash == -1:
                prev_end_token = req.cached_or_caching_num_tokens - req.caching_num_tokens - 1
                prev_block_idx = prev_end_token // self.page_size
                if 0 <= prev_block_idx < req.num_pages:
                    token_ids: list[int] = req.page(prev_block_idx)
                    prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                    h = self.compute_hash(token_ids, prefix)
                    last_block.update(h, token_ids)
                    self.hash_to_block_id[h] = last_block.block_id

            if not self.free_block_ids:
                raise RuntimeError(
                    "D2FKVCacheManager: insufficient free KV cache blocks to append: "
                    f"required={required}, current_len={len(block_table)}, "
                    f"cached_or_caching_num_tokens={req.cached_or_caching_num_tokens}, "
                    f"block_size={self.page_size}."
                )

            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
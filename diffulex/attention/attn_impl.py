import os
import torch
import torch.nn as nn

from einops import rearrange

from diffulex_kernel import (
    store_kv_cache_distinct_layout,
    store_kv_cache_unified_layout,
    chunked_prefill_attn_unified,
)
from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.attention.metadata import is_warming_up
from diffulex.logger import get_logger


USE_REFERENCE_MULTI_BLOCK_ATTN = os.environ.get("DIFFULEX_USE_REFERENCE_MULTI_BLOCK_ATTN", "0") == "1"
USE_REFERENCE_CACHED_CONTEXT_ATTN = os.environ.get("DIFFULEX_USE_REFERENCE_CACHED_CONTEXT_ATTN", "0") == "1"
USE_REFERENCE_FOR_MULTI_BD = os.environ.get("DIFFULEX_USE_REFERENCE_FOR_MULTI_BD", "0") == "1"
logger = get_logger(__name__)
_ATTN_DUMP_CALL_ID = 0


def reference_multi_block_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    attn_metadata: AttnMetaDataBase,
    scale: float,
) -> torch.Tensor:
    """Reference multi-block attention.

    Debug-only reference path for multi-block attention. Normal execution should keep
    requests batched and use the Triton kernel; set
    ``DIFFULEX_USE_REFERENCE_MULTI_BLOCK_ATTN=1`` only when validating/debugging the
    kernel path.
    """
    page_tables = attn_metadata.page_tables
    context_lens = attn_metadata.context_lens
    cu_seqlens_q = attn_metadata.cu_seqlens_q
    valid_slices = getattr(attn_metadata, "valid_slices", None)
    status_table = getattr(attn_metadata, "status_table", None)
    prefix_lens = getattr(attn_metadata, "prefix_lens", None)
    padded_prefix_lens = getattr(attn_metadata, "padded_prefix_lens", None)
    page_size = attn_metadata.page_size
    block_size = attn_metadata.block_size
    is_prefix_full = bool(getattr(attn_metadata, "is_prefix_full", False))

    num_seqs = len(cu_seqlens_q) - 1
    output = torch.zeros_like(q)

    for seq_id in range(num_seqs):
        q_start = int(cu_seqlens_q[seq_id].item())
        q_end = int(cu_seqlens_q[seq_id + 1].item())
        if valid_slices is not None:
            valid_end = int(valid_slices[seq_id].item())
            valid_q_len = valid_end - q_start
        else:
            valid_q_len = q_end - q_start

        ctx_len = int(context_lens[seq_id].item())
        if valid_q_len <= 0:
            continue

        q_seq = q[q_start : q_start + valid_q_len]

        k_parts, v_parts = [], []
        if k_cache.numel() > 0 and ctx_len > 0:
            for rel_page_id in range(page_tables.shape[1]):
                abs_page_id = int(page_tables[seq_id, rel_page_id].item())
                if abs_page_id < 0:
                    continue
                page_start = rel_page_id * page_size
                if page_start >= ctx_len:
                    break
                n = min(page_start + page_size, ctx_len) - page_start
                k_parts.append(k_cache[abs_page_id, :n])
                v_parts.append(v_cache[abs_page_id, :n])

        cached_ctx_len = sum(part.shape[0] for part in k_parts)
        if cached_ctx_len != ctx_len:
            logger.warning(
                "reference_multi_block_attn: context/page-table mismatch for seq %s: "
                "context_len=%s, reconstructed_ctx_len=%s",
                seq_id,
                ctx_len,
                cached_ctx_len,
            )

        k_new = k[q_start : q_start + valid_q_len]
        v_new = v[q_start : q_start + valid_q_len]
        if k_parts:
            k_full = torch.cat(k_parts + [k_new], dim=0)
            v_full = torch.cat(v_parts + [v_new], dim=0)
        else:
            k_full = k_new
            v_full = v_new

        mask = None
        if block_size > 0:
            qi = torch.arange(valid_q_len, device=q.device)
            kj = torch.arange(valid_q_len, device=q.device)
            abs_q = ctx_len + qi
            abs_k = ctx_len + kj

            if is_prefix_full:
                status = int(status_table[seq_id].item()) if status_table is not None else 0
                prefix_len = int(prefix_lens[seq_id].item()) if prefix_lens is not None else 0
                padded_prefix_len = int(padded_prefix_lens[seq_id].item()) if padded_prefix_lens is not None else 0

                if status == 0:
                    pure_prefix = (qi[:, None] < prefix_len) & (kj[None, :] < prefix_len)
                    padded_causal = (
                        (qi[:, None] >= prefix_len)
                        & (qi[:, None] < padded_prefix_len)
                        & (kj[None, :] < padded_prefix_len)
                    )
                    block_ends = ((abs_q // block_size) + 1) * block_size
                    block_mask_extend = (abs_k[None, :] < block_ends[:, None]) & (qi[:, None] >= padded_prefix_len)
                    new_kv_mask = pure_prefix | padded_causal | block_mask_extend
                else:
                    block_ends = ((abs_q // block_size) + 1) * block_size
                    new_kv_mask = abs_k[None, :] < block_ends[:, None]
            else:
                block_ends = ((abs_q // block_size) + 1) * block_size
                new_kv_mask = abs_k[None, :] < block_ends[:, None]

            if cached_ctx_len > 0:
                cache_mask = torch.ones(valid_q_len, cached_ctx_len, dtype=torch.bool, device=q.device)
                mask = torch.cat([cache_mask, new_kv_mask], dim=1)
            else:
                mask = new_kv_mask
            mask = mask.unsqueeze(0).unsqueeze(0)

        q_ref = q_seq.float().transpose(0, 1)
        k_ref = k_full.float().transpose(0, 1)
        v_ref = v_full.float().transpose(0, 1)
        if q_ref.shape[0] != k_ref.shape[0]:
            group_size = q_ref.shape[0] // k_ref.shape[0]
            k_ref = k_ref.repeat_interleave(group_size, dim=0)
            v_ref = v_ref.repeat_interleave(group_size, dim=0)

        scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * float(scale)
        if mask is not None:
            score_mask = mask.squeeze(0).squeeze(0)
            scores = scores.masked_fill(~score_mask.unsqueeze(0), torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_ref).transpose(0, 1).to(q.dtype)
        output[q_start : q_start + valid_q_len] = attn_out

    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

        self.q_shape = {
            "nh": self.num_heads,
            "hd": self.head_dim,
        }
        self.kv_shape = {
            "nkvh": self.num_kv_heads,
            "hd": self.head_dim,
        }
        # Import the specified fetch function
        from diffulex.attention import fetch_attn_metadata

        self.fetch_attn_metadata = fetch_attn_metadata

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        global _ATTN_DUMP_CALL_ID
        # Reshape
        q = rearrange(q, "s (nh hd) -> s nh hd", **self.q_shape)
        k = rearrange(k, "s (nkvh hd) -> s nkvh hd", **self.kv_shape)
        v = rearrange(v, "s (nkvh hd) -> s nkvh hd", **self.kv_shape)

        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
        k_cache, v_cache = self.k_cache, self.v_cache
        dump_enabled = os.environ.get("DIFFULEX_DUMP_ATTN_TENSORS", "0") == "1" and not is_warming_up()
        dump_payload = None
        if dump_enabled:
            skip_calls = int(os.environ.get("DIFFULEX_DUMP_ATTN_SKIP_CALLS", "0"))
            max_calls = int(os.environ.get("DIFFULEX_DUMP_ATTN_MAX_CALLS", "1"))
            if _ATTN_DUMP_CALL_ID >= skip_calls and (_ATTN_DUMP_CALL_ID - skip_calls) < max_calls:
                dump_payload = {
                    "q": q.detach().cpu(),
                    "k": k.detach().cpu(),
                    "v": v.detach().cpu(),
                    "is_prefill": list(getattr(attn_metadata, "is_prefill", [])),
                    "cu_seqlens_q": getattr(attn_metadata, "cu_seqlens_q", None).detach().cpu()
                    if getattr(attn_metadata, "cu_seqlens_q", None) is not None
                    else None,
                    "cu_seqlens_k": getattr(attn_metadata, "cu_seqlens_k", None).detach().cpu()
                    if getattr(attn_metadata, "cu_seqlens_k", None) is not None
                    else None,
                    "context_lens": getattr(attn_metadata, "context_lens", None).detach().cpu()
                    if getattr(attn_metadata, "context_lens", None) is not None
                    else None,
                    "page_tables": getattr(attn_metadata, "page_tables", None).detach().cpu()
                    if getattr(attn_metadata, "page_tables", None) is not None
                    else None,
                    "slot_mapping": getattr(attn_metadata, "slot_mapping", None).detach().cpu()
                    if getattr(attn_metadata, "slot_mapping", None) is not None
                    else None,
                    "valid_slices": getattr(attn_metadata, "valid_slices", None).detach().cpu()
                    if getattr(attn_metadata, "valid_slices", None) is not None
                    else None,
                    "status_table": getattr(attn_metadata, "status_table", None).detach().cpu()
                    if getattr(attn_metadata, "status_table", None) is not None
                    else None,
                    "prefix_lens": getattr(attn_metadata, "prefix_lens", None).detach().cpu()
                    if getattr(attn_metadata, "prefix_lens", None) is not None
                    else None,
                    "padded_prefix_lens": getattr(attn_metadata, "padded_prefix_lens", None).detach().cpu()
                    if getattr(attn_metadata, "padded_prefix_lens", None) is not None
                    else None,
                    "page_size": int(getattr(attn_metadata, "page_size", -1)),
                    "block_size": int(getattr(attn_metadata, "block_size", -1)),
                }
                if os.environ.get("DIFFULEX_DUMP_KV_PAGES", "0") == "1" and k_cache.numel() and v_cache.numel():
                    page_tables = getattr(attn_metadata, "page_tables", None)
                    if page_tables is not None:
                        used_pages = torch.unique(page_tables[page_tables >= 0]).detach().cpu().tolist()
                        max_pages = int(os.environ.get("DIFFULEX_DUMP_KV_MAX_PAGES", "64"))
                        used_pages = [int(x) for x in used_pages[:max_pages]]
                        dump_payload["used_page_ids"] = used_pages
                        if used_pages:
                            page_index = torch.tensor(used_pages, dtype=torch.long, device=k_cache.device)
                            dump_payload["k_cache_pages"] = k_cache.index_select(0, page_index).detach().cpu()
                            dump_payload["v_cache_pages"] = v_cache.index_select(0, page_index).detach().cpu()
            _ATTN_DUMP_CALL_ID += 1
        is_unified_layout = attn_metadata.kv_cache_layout == "unified"

        # Fast Store KV cache
        if k_cache.numel() and v_cache.numel():
            if attn_metadata.need_kv_cache_store:
                store_kv_cache = store_kv_cache_unified_layout if is_unified_layout else store_kv_cache_distinct_layout
                store_kv_cache(k, v, k_cache, v_cache, attn_metadata.slot_mapping, attn_metadata)

        context_lens = getattr(attn_metadata, "context_lens", None)

        use_reference_cached_context = (
            getattr(attn_metadata, "use_multi_block", False)
            and USE_REFERENCE_CACHED_CONTEXT_ATTN
            and context_lens is not None
            and bool((context_lens > 0).any().item())
        )

        force_reference_multi_bd = (
            getattr(attn_metadata, "use_multi_block", False)
            and USE_REFERENCE_FOR_MULTI_BD
        )

        if getattr(attn_metadata, "use_multi_block", False) and (
            USE_REFERENCE_MULTI_BLOCK_ATTN
            or use_reference_cached_context
            or force_reference_multi_bd
        ):
            o = reference_multi_block_attn(q, k, v, k_cache, v_cache, attn_metadata, self.scale)
        else:
            o = chunked_prefill_attn_unified(q, k, v, k_cache, v_cache, attn_metadata)
        if dump_payload is not None:
            dump_payload["o"] = o.detach().cpu()
            dump_dir = os.environ.get("DIFFULEX_DUMP_ATTN_DIR", "/tmp/diffulex_attn_dump")
            os.makedirs(dump_dir, exist_ok=True)
            dump_idx = _ATTN_DUMP_CALL_ID - int(os.environ.get("DIFFULEX_DUMP_ATTN_SKIP_CALLS", "0")) - 1
            dump_path = os.path.join(dump_dir, f"attn_call_{dump_idx:03d}.pt")
            torch.save(dump_payload, dump_path)
            logger.info("DIFFULEX_DUMP_ATTN_TENSORS: saved %s", dump_path)

        # Final reshape
        return rearrange(o, "s nh hd -> s (nh hd)").contiguous()

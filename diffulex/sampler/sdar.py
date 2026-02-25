import torch

from dataclasses import dataclass

from diffulex.sampler.auto_sampler import AutoSampler
from diffulex.sampler.base import SamplerNoShiftLogits, SampleOutputBase
from diffulex.engine.request import DllmReq


@dataclass
class SDARSampleOutputForDiffusionLM(SampleOutputBase):
    pass


@AutoSampler.register("sdar")
class SDARSamplerForDiffusionLM(SamplerNoShiftLogits):
    def forward(self, reqs: list[DllmReq], logits: torch.Tensor, temperatures: torch.Tensor,
                top_p=None, top_k=None, margin_confidence=False, neg_entropy=False, threshold=0.95):
        attn_metadata = self.fetch_attn_metadata()
        split_logits = torch.split(
            logits, [len(req.running_sequence) for req in reqs] if attn_metadata.is_prefill 
            else [attn_metadata.chunk_size] * len(reqs), dim=0
        )
        
        accepted_ids_map = {}
        sampled_tokens_map = {}
        true_local_ids_map = {}
        for temperature, req, req_logits in zip(temperatures, reqs, split_logits):
            true_local_ids_sub_map = {}
            accepted_ids_sub_map = {}
            sampled_tokens_sub_map = {}
            
            for block_id, block in enumerate(req.dllm_blocks):
                if not block.is_active or (block.num_mask_tokens == 0):
                    continue
                
                if len(block.mask_token_global_ids) == 0:
                    continue
                
                if attn_metadata.is_prefill:
                    mask_token_logits = req_logits[block.mask_token_global_ids, ...]
                else:
                    mask_token_logits = req_logits[block.mask_token_relative_ids, ...]
                
                confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                    mask_token_logits, 
                    temperature, 
                    top_p=top_p, 
                    top_k=top_k, 
                    neg_entropy=(neg_entropy == "neg_entropy"),
                    margin_confidence=(margin_confidence == "margin_confidence")
                )
                
                high_conf_indices = torch.where(initial_confidence > threshold)[0]
                
                if len(high_conf_indices) == 0:
                    max_prob_idx = initial_confidence.argmax()
                    accepted_ids = max_prob_idx.view(1)
                else:
                    max_prob_idx = initial_confidence.argmax()
                    accepted_ids = torch.unique(torch.cat([high_conf_indices, max_prob_idx.view(1)]))
                
                accepted_ids_list = accepted_ids.to(device="cpu").tolist()
                true_local_ids_sub_map[str(block_id)] = [block.mask_token_relative_ids[i] for i in accepted_ids_list]
                accepted_ids_sub_map[str(block_id)] = accepted_ids_list
                sampled_tokens_sub_map[str(block_id)] = sampled_tokens.to(device="cpu").tolist()
            
            seq_idx = str(req.req_id)
            true_local_ids_map[seq_idx] = true_local_ids_sub_map
            accepted_ids_map[seq_idx] = accepted_ids_sub_map
            sampled_tokens_map[seq_idx] = sampled_tokens_sub_map

        return SDARSampleOutputForDiffusionLM(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map
        )
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from sparse_attn.token_sparse.kernel import triton_get_attn_cache

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def set_model(model, args): 
    num_layers = len(model.model.layers)
    
    if args.sparse_layer is not None:
        sparse_layers = args.sparse_layer
    else:
        # From profile_sparse_layer.py
        if args.model_path == "meta-llama/Llama-3.1-8B-Instruct":
            sparse_layers = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        elif args.model_path == "mistralai/Mistral-Nemo-Instruct-2407":
            sparse_layers = [14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        else:
            sparse_layers = list(range(num_layers//2, num_layers))

    print(f"Sparse layers: {sparse_layers}")
    
    for i in range(num_layers):
        model.model.layers[i].self_attn.sparse_cluster.layer_idx = i
        model.model.layers[i].self_attn.sparse_cluster.min_tokens = args.min_tokens
        model.model.layers[i].self_attn.sparse_cluster.window_size = args.window_size
        model.model.layers[i].self_attn.sparse_cluster.kernel_size = args.kernel_size
        
        if i in sparse_layers:
            model.model.layers[i].self_attn.sparse_cluster.coverage = args.coverage
        else:
            model.model.layers[i].self_attn.sparse_cluster.coverage = 0.0

class SparseCluster():
    def __init__(self):
        self.window_size = None
        self.kernel_size = None
        self.min_tokens = None
        self.coverage = None
        
        self.layer_idx = None
        self.q_len = None
        self.bsz = None
        self.head_dim = None
        self.num_heads = None

    def get_index(self, query_states, key_states):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        self.bsz, self.num_heads, self.q_len, self.head_dim = query_states.shape

        # if self.q_len <= self.window_size or self.sparse_ratio == 0.0:
        #     return None
        if self.q_len <= (self.min_tokens + self.window_size) or self.coverage == 0.0:
            return None
        else:
            # Get approximated score
            # attn_cache: (bsz, num_heads, q_len - window_size)
            attn_cache = triton_get_attn_cache(
                query_states=query_states,
                key_states=key_states,
                window_size=self.window_size,
                out_dtype=query_states.dtype)
            
            attn_cache = F.avg_pool1d(attn_cache, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            
            attn_cache_global = attn_cache.mean(dim=1)
            attn_cache_normalized = attn_cache / (attn_cache.sum(dim=-1, keepdim=True) + 1e-8)
            attn_cache_normalized_global = attn_cache_global / (attn_cache_global.sum(dim=-1, keepdim=True) + 1e-8)

            # Calculate cumulative sum and find number of tokens needed to reach coverage threshold
            sorted_attn_cache, _ = torch.sort(attn_cache_normalized_global, dim=-1, descending=False)
            cumulative_coverage = torch.cumsum(sorted_attn_cache.float(), dim=-1)
            
            num_sparse_tokens = (cumulative_coverage < self.coverage).sum(dim=-1).item() 
            
            if num_sparse_tokens == 0:
                return None
            else:
                num_sparse_tokens += 1
                
            max_capacity_prompt = max(self.min_tokens, (self.q_len - self.window_size) - num_sparse_tokens)
            indices = attn_cache_normalized.topk(max_capacity_prompt, dim=-1).indices
            window_indices = torch.arange(self.q_len - self.window_size, self.q_len, device=indices.device)
            window_indices = window_indices.unsqueeze(0).unsqueeze(0).expand(self.bsz, indices.shape[1], -1)
            indices = torch.cat([indices, window_indices], dim=-1).sort(dim=-1, descending=False)[0]

            # Return indices: (bsz, num_heads, compressed_len)
            return indices


    def compress_qkv(self, query_states, key_states, value_states, indices):
        """
        Compress Q, K, V tensors using the provided indices.
        
        Args:
            query_states: (bsz, num_heads, q_len, head_dim) - e.g., (1, 32, q_len, head_dim)
            key_states: (bsz, num_heads, q_len, head_dim) - e.g., (1, 32, q_len, head_dim)
            value_states: (bsz, num_heads, q_len, head_dim) - e.g., (1, 32, q_len, head_dim)
            indices: (bsz, num_heads, compressed_sequence_length) - e.g., (1, 32, compressed_len)
        
        Returns:
            compressed_query_states: (bsz, num_heads, compressed_sequence_length, head_dim)
            compressed_key_states: (bsz, num_heads, compressed_sequence_length, head_dim)
            compressed_value_states: (bsz, num_heads, compressed_sequence_length, head_dim)
        """
        if indices is None:
            return query_states, key_states, value_states
        
        # Compress K and V: expand indices to include head_dim dimension
        # key_states: (bsz, num_heads, q_len, head_dim)
        # indices: (bsz, num_heads, compressed_len) -> (bsz, num_heads, compressed_len, head_dim)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        compressed_key_states = torch.gather(
            key_states, 
            dim=2, 
            index=indices_expanded
        )  # (bsz, num_heads, compressed_len, head_dim)
        
        compressed_value_states = torch.gather(
            value_states,
            dim=2,
            index=indices_expanded
        )  # (bsz, num_heads, compressed_len, head_dim)
        
        compressed_query_states = torch.gather(
            query_states,
            dim=2,
            index=indices_expanded
        )
        
        return compressed_query_states, compressed_key_states, compressed_value_states

    def decompress_output(self, attn_output, indices):
        """
        Decompress attention output from compressed size back to original size.
        
        Args:
            attn_output: (bsz, compressed_len, num_heads, head_dim) - compressed attention output
            indices: (bsz, num_heads, compressed_sequence_length) - compression indices
        
        Returns:
            decompressed_output: (bsz, q_len, num_heads, head_dim)
        """
        if indices is None:
            return attn_output
        
        # indices has shape (bsz, num_heads, compressed_len)
        # attn_output: (bsz, compressed_len, num_heads, head_dim) -> (bsz, num_heads, compressed_len, head_dim)
        attn_output = attn_output.transpose(1, 2)  # (bsz, num_heads, compressed_len, head_dim)
        
        # Create output tensor with original size
        decompressed_output = torch.zeros(
            self.bsz, self.num_heads, self.q_len, self.head_dim,
            dtype=attn_output.dtype, device=attn_output.device
        )
        
        # Expand indices to match the grouped shape: (bsz, num_heads, compressed_len, head_dim)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        
        # Use scatter to place compressed values at original positions
        # scatter_(dim, index, src) places src values at positions specified by index
        decompressed_output.scatter_(
            dim=2,  # sequence length dimension
            index=indices_expanded,
            src=attn_output
        )

        # Transpose back to (bsz, q_len, num_heads, head_dim)
        decompressed_output = decompressed_output.transpose(1, 2).reshape(self.bsz, self.q_len, -1).contiguous()
        
        return decompressed_output


def init_cluster(self):
    self.sparse_cluster = SparseCluster()
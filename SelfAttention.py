import math, torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, attention_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.attention_heads = attention_heads
        self.head_dim = self.embed_size // self.attention_heads
        # Linear layers for applying linear transformations to input
        self.values = nn.Linear(self.embed_size, self.embed_size)
        self.keys = nn.Linear(self.embed_size, self.embed_size)
        self.queries = nn.Linear(self.embed_size, self.embed_size)
        # Linear layers for applying linear transformations to output
        self.ouput_layer = nn.Linear(self.embed_size, self.embed_size)


    def forward(self, values, keys, queries, mask):
        sample_count = queries.shape[0] # Number of training samples
        # Value, Key & Query Individual Sample Length
        v_len, k_len, q_len = values.shape[1], keys.shape[1], queries.shape[1]
        # Passing through linear transformation
        # (No. Samples, Single Sample Length, Embedding Size)
        values = self.values(values)  
        keys = self.keys(keys) 
        queries = self.queries(queries)
        # Split into multiple attention heads
        values = values.reshape(sample_count, v_len, self.attention_heads, self.head_dim)
        keys = keys.reshape(sample_count, k_len, self.attention_heads, self.head_dim)
        queries = queries.reshape(sample_count, q_len, self.attention_heads, self.head_dim)
        # Multiply Q(queries) with transpose of K(keys) in multi-head fashion
        # Formula = [Q.K(t)]
        # Weight dimesions: (No. Samples, Attention Heads, Query Length, Key Length)
        weights = torch.einsum('nqhd,nkhd->nhqk', queries, keys)
        # Apply mask to [Q.K(t)] if it exists
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float("-1e19")) # Masking with negative inf
        # Computing attention by scaling weights, multiplying by values & applying softmax
        attention = F.softmax(weights/math.sqrt(self.embed_size), dim=3) # Softmax is applied row-wise to Query x Keys Matrix
        # Multiplying attention with Values Matirx and combining Multi-Head attention into a Single Head
        attention = torch.einsum('nhql,nlhd->nqhd', attention, values).reshape(sample_count, q_len, self.embed_size)
        # Finally passing our attention through output linear layer
        return self.ouput_layer(attention)
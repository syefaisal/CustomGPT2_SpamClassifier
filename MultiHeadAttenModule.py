import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, 
    dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads

        # Reduces the projection dimension to match 
        # the desired output dimension
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Use a linear layer to combine multiple head outputs
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Create one large matrix for all heads
        keys = self.W_key(x) # Tensor of shape (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)


        # We implicitly split the matrix by adding a 'num_heads' dimension and using PyTorch's view method
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # Reshaping to represent Multiple Heads
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        # Step1: Compute Scaled dot product attention (aka slef attention) witha casual mask
        attnention_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Step 2: Use the mask to fill the attn_scores with -ve infinity
        attnention_scores.masked_fill_(mask_bool, -torch.inf)


        # Step 3: Compute Attention Weights by normalizing the attn_scores 
        # using softmax
        attention_weights = torch.softmax(attnention_scores / keys.shape[-1]**0.5, dim=-1)

        # Step 4: Applying Dropout to prevent overfitting
        attention_weights = self.dropout(attention_weights)

        # Tensor Shape: (b, num_tokens, num_heads, head_dim)
        # Step 5: Compute Context Vector
        context_vector = (attention_weights @ values).transpose(1, 2)

        # Step 6: For Multi-Head Attention, Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector) # Adds an optional linear projection

        return context_vector

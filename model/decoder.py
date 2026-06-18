import math
from torch import nn
import torch 
    
def get_alibi_slopes(n_heads): 
    """
    return tnsor shape(1, n_heads, max_len, max_len)
    """

    def get_slopes(n):
        start = 2 * (-8 / n)
        return [start ** (i + 1) for i in range(n)]
    
    if math.log2(n_heads).is_integer():
        return get_slopes(n_heads)
    
    closest_pow2 = 2 ** math.floor(math.log2(n_heads))
    base = get_slopes(closest_pow2)
    extra = get_slopes(2 * closest_pow2)[0::2]
    return (base + extra)[:n_heads]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # Q = self.Q(x) produces a (seq_len, d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        slopes = torch.tensor(get_alibi_slopes(n_heads), dtype = torch.float32)
        self.register_buffer("alibi_slopes", slopes, persistent = False)

    def forward(self, x, pad_mask = None):
        Bs, seq_len, _ = x.shape
        d_head = self.d_model // self.n_heads

        def split_heads(t):
            return t.view(Bs, seq_len, self.n_heads, d_head).transpose(1, 2)

        q = split_heads(self.W_q(x))
        k = split_heads(self.W_k(x))
        v = split_heads(self.W_v(x))

        scores = q @ k.transpose(-2, -1)  # (Bs, n_heads, seq, seq)
        scores = scores / (d_head ** 0.5)

        # Causal masked scores
        # ALiBi: bias[i, j] = slope_h * (j - i), <= 0 for j <= i, so keys
        positions = torch.arange(seq_len, device = x.device)
        relative_pos = positions.view(1, -1) - positions.view(-1, 1) # (seq, seq): j - i
        alibi_bias = self.alibi_slopes.view(self.n_heads, 1, 1) * relative_pos.unsqueeze(0)
        scores = scores + alibi_bias.unsqueeze(0)

        mask = torch.tril(torch.ones(seq_len, seq_len, 
                                    device=scores.device, dtype = torch.bool))

        causal_mask = mask.view(1, 1, seq_len, seq_len)

        if pad_mask is not None: # prevent leakage
            key_invalid = pad_mask.view(Bs, 1, 1, seq_len)
            causal_mask = causal_mask & ~key_invalid

        scores = scores.masked_fill(~causal_mask, float("-inf"))

        weights = torch.softmax(scores, dim = -1)
        weights = torch.nan_to_num(weights, nan = 0.0)
        output = weights @ v  # (B, n_heads, T, d_head)
        output = output.transpose(1, 2).contiguous().view(Bs, seq_len, self.d_model)
        output = self.W_o(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.hid_dim = 4 * d_model
        self.ln1 = nn.Linear(self.d_model, self.hid_dim)
        self.GELU = nn.GELU()
        self.ln2 = nn.Linear(self.hid_dim, self.d_model)
    
    def forward(self, x):
        output = self.ln1(x)
        output = self.GELU(output)
        output = self.ln2(output)

        return output
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.multi_head = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, pad_mask = None):
        x = x + self.multi_head(self.norm1(x), pad_mask = pad_mask)
        x = x + self.ff(self.norm2(x))

        return x

import math
from torch import nn
import torch 

def get_slopes(n):
    return [1 / (2 ** (i + 1)) for i in range(n)]
    
def build_alibi_tensor(max_len, n_heads): 
    """
    return tnsor shape(1, n_heads, max_len, max_len)
    """
    slopes = torch.tenseor(get_slopes(n_heads)).view(1, n_heads, 1, 1)
    context_pos = torch.arange(max_len)[:, None]
    memory_pos = torch.arange(max_len)[None, :]

    relative_distance = memory_pos - context_pos
    relative_distance = relative_distance.unsqueeze(0).unsqueeze(0)
    alibi_bias = slopes + relative_distance
    return alibi_bias

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

    def forward(self, x, alibi_bias = None, pad_mask = None):
        B, seq_len, _ = x.shape
        seq_len = x.shape[0]
        d_head = self.d_model//self.n_heads
        # q = self.W_q(x).view([seq_len, self.n_heads,
        #                     d_head]) # per head dim
        # q = q.transpose(0, 1) # (n_heads: batch-like, seq_len, d_head)

        # k = self.W_k(x).view([seq_len, self.n_heads,
        #                     d_head]) # per head dim
        # k = k.transpose(0, 1)

        # v = self.W_v(x).view([seq_len, self.n_heads,
        #                     d_head]) # per head dim
        # v = v.transpose(0, 1)

        q = self.W_q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # scores = q @ k.transpose(-2, -1) # (n_heads, seq_len, seq_len)
        # scores = scores / (d_head ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if alibi_bias is not None:
            scores = scores + alibi_bias
        # Causal masked scores
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))

        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == True, float("-inf"))

        attention = torch.softmax(scores, dim = -1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        # softmax on j's every i
        # weights = torch.softmax(scores, dim = -1)

        # Final Attention (softmax) @ V
        # output = weights @ v # (n_heads, seq_len, d_head)
        # output = output.transpose(0, 1) # original shape
        # output = output.contiguous().view(seq_len, self.d_model) # (seq_len, d_model)
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
        self.multi_head = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        x = x + self.multi_head(x)
        x = x + self.ff(x)

        return x
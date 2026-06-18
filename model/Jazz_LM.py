from torch import nn

from .embeddings import MyModule
from .decoder import DecoderLayer

class JazzLanguageModel(nn.Module):
    def __init__(self, vocabsize_table, special_tk_size,
                d_model, n_heads, n_layers):
        super().__init__()
        self.emb = MyModule(vocabsize_table, special_tk_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads) for n in range(n_layers)])
        heads = [nn.Linear(d_model, size) for size in vocabsize_table]
        heads = heads + [nn.Linear(d_model, 4)]
        self.out_head = nn.ModuleList(heads)

    def forward(self, tokens, pad_mask = None):
        """
        return:
        list of 23 logit tensors
        (seq_len, vocab_size) for a single seq
        (Bs, seq_len, vocab_size) for batches
        => used by the data pipeline (training)
        => or a future generation loop (inference) to read out just the slice it needs.
        """
        x = self.emb(tokens) # (seq_len, d_model)
        single_head = x.dim() == 2
        if single_head:
            x = x.unsqueeze(0)

        for layer in self.layers:
            x = layer(x, pad_mask = pad_mask)
        
        head_logits = [head(x) for head in self.out_head] # each (B, T, vocab_size_k)

        if single_head:
            head_logits = [logit.squeeze(0) for logit in head_logits] # (T, vocab_size_k)

        
        return head_logits

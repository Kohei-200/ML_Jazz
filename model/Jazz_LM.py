from torch import nn

from embeddings import MyModule
from decoder import DecoderLayer

class JazzLanguageModel(nn.Module):
    def __init__(self, vocabsize_table, special_tk_size, 
                d_model, n_heads, n_layers):
        super().__init__()
        self.emb = MyModule(vocabsize_table, special_tk_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads) for n in range(n_layers)])
        heads = [nn.Linear(d_model, size) for size in vocabsize_table]
        heads = heads + [nn.Linear(d_model, 4)]
        self.out_head = nn.ModuleList(heads)

    def forward(self, tokens):
        x = self.emb(tokens) # (seq_len, d_model)
        for layer in self.layers:
            x = layer(x) # (seq_len, d_model)
        slot_idx = 0
        logits = [] # used to compute the loss
        for i, tok in enumerate(tokens):
            token_val = tok.item()
            if token_val >= 1000:
                if token_val == 1101:
                    slot_idx = 0
                elif token_val == 1102:
                    slot_idx = 4
                elif token_val == 1103:
                    slot_idx = 12
                continue
            elif slot_idx in [3, 11, 21]: # instrument, ten13, or noteshape
                logits.append(self.out_head[22](x[i]))
                slot_idx += 1
            else:
                logits.append(self.out_head[slot_idx](x[i]))
                slot_idx += 1

        logits = [logit.squeeze() for logit in logits]
        return logits
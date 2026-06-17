from torch import nn
import torch 
from tokenizer.vocab import SPECIAL_TOKEN_IDX

class MyModule(nn.Module):
    # initialize table with ModuleList and refer it back in forward
    def __init__(self, vocabsize_table, special_tk_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.slot_embeddings = nn.ModuleList( # :list : loop for embedding different sizes
            [nn.Embedding(size, d_model) for size in vocabsize_table]
        )
        self.special_embeddings = nn.Embedding(special_tk_size, d_model)

    def forward(self, x):
        embeddings = []
        slot_idx = 0
        for tok in x:
            if tok.item() in [1000, 1001, 1002]:
                tok = torch.zeros(self.d_model)
                embeddings.append(tok)
                slot_idx = 0
            elif tok.item() in [1101, 1102, 1103]:
                token_val = tok.item()
                idx_val = torch.tensor(SPECIAL_TOKEN_IDX[token_val])
                tok = self.special_embeddings(idx_val.unsqueeze(0)).squeeze(0) # shape: (d_model, )
                embeddings.append(tok)

                if token_val == 1101:
                    slot_idx = 0
                elif token_val == 1102:
                    slot_idx = 4
                elif token_val == 1103:
                    slot_idx = 12
            else:
                # print(slot_idx, tok.item())
                tok = self.slot_embeddings[slot_idx](tok.unsqueeze(0))
                embeddings.append(tok)
                slot_idx += 1
        embeddings = [e.squeeze() for e in embeddings]
        return torch.stack(embeddings) # 2D tensor
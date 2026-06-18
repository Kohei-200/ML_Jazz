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
    
    def _emb_one(self, x):
        embeddings = []
        slot_idx = 0

        for tok in x:
            if tok.item() in [1000, 1001, 1002]:
                tok_vec = torch.zeros(self.d_model, device = x.device)
                embeddings.append(tok_vec)
                slot_idx = 0
            elif tok.item() in [1101, 1102, 1103]:
                token_val = tok.item()
                idx_val = torch.tensor(SPECIAL_TOKEN_IDX[token_val], device = x.device)
                tok_vec = self.special_embeddings(idx_val.unsqueeze(0)).squeeze(0)  # (d_model,)
                embeddings.append(tok_vec)
                if token_val == 1101:
                    slot_idx = 0
                elif token_val == 1102:
                    slot_idx = 4
                elif token_val == 1103:
                    slot_idx = 12
            else:
                tok_vec = self.slot_embeddings[slot_idx](tok.unsqueeze(0))
                embeddings.append(tok_vec)
                slot_idx += 1
        embeddings = [emb.squeeze() for emb in embeddings]
        return torch.stack(embeddings) # seq_len, d_model

    def forward(self, x):
        if x.dim() == 1:
            return self._emb_one(x)
        
        return torch.stack(
            [self._emb_one(row) for row in x]
        ) # 2D tensor
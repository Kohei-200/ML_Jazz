import os
import torch
from torch import nn
from torch.utils.data import Dataset

from tokenizer.vocab import SPECIAL_TOKENS, get_prediction_events

PAD_ID = SPECIAL_TOKENS["PAD"]
IGNORE_INDEX = -100   # standard CrossEntropyLoss ignore value
IGNORE_SLOT = -1       # sentinel: "no out_head applies here"

class JazzSoloDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted(f for f in os.listdir(folder_path) if f.endswith(".pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt file in {folder_path}")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.folder_path, self.files[idx]))
        return data
    
def collate_fn(batch):
    """
    return: (B, T-1)
    inputs: what model sees
    targets : local class id to be predicted at each input
    slot_ids: which out_head each target belongs to 
    pad_mask: True at PAD
    """
    padded = nn.utils.rnn.pad_sequence(
        batch, batch_first = True, padding_value = PAD_ID
    )
    inputs = padded[:, :-1]
    #Batchsize, Time steps minus 1
    Bs, Tm1 = inputs.shape 
    targets = torch.full((Bs, Tm1), IGNORE_INDEX, dtype = torch.long)
    slot_ids = torch.full((Bs, Tm1), IGNORE_SLOT, dtype = torch.long)

    for b, seq in enumerate(batch):
        for predictor_pos, slot_id, target_id in get_prediction_events(seq):
            targets[b, predictor_pos] = target_id
            slot_ids[b, predictor_pos] = slot_id

        pad_mask = inputs == PAD_ID

        return inputs, targets, slot_ids, pad_mask


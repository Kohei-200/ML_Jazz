import torch
import torch.nn.functional as F

def compute_loss(head_logits, targets, slot_ids):
    """
    gather every pair <= scored against that head into one batch
    => run a single CE loss call
    Positions equally weighted overall
    """
    total_loss = targets.new_zeros((), dtype = torch.float32)
    total_count = 0

    for slot_id, logits in enumerate(head_logits):
        mask = slot_ids == slot_id
        if not mask.any():
            continue

        slot_logits = logits[mask] # (n, vocab_size)
        slot_targets = targets[mask]

        total_loss = total_loss + F.cross_entropy(
            slot_logits, slot_targets, reduction = "sum")
        total_count += slot_targets.numel()
    
    if total_count == 0:
        raise ValueError("No input")
    
    return total_loss / total_count
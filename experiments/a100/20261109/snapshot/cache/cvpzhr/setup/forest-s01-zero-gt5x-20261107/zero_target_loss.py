"""Training-only weights for exact zero progress targets."""
import torch


def zero_target_weights(target_pt, multiplier):
    # GT remains a loss-only target. No thresholding of small positive labels.
    return torch.where(target_pt == 0, multiplier, 1.).to(dtype=torch.float32)

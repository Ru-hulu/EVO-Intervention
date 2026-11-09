"""Loss-only native failure metadata and detached hard-example weighting."""
import torch


class FailureCollator:
    def __init__(self, collator, failure_frames):
        self.collator=collator
        self.failure_frames=failure_frames

    def __call__(self, samples):
        batch=self.collator(samples)
        batch['native_failure']=torch.tensor([
            self.failure_frames[s['episode_id']][str(s['frame_index'])] for s in samples],dtype=torch.bool)
        return batch


def hard_failure_weights(target, prediction, native_failure, multiplier=5., threshold=.2):
    hard=native_failure & (target==0) & (prediction.detach()>threshold)
    return torch.where(hard,multiplier,1.).float(),hard

"""Official linspace/rint prefix sampling, then right-pad the current frame.

Temporal sampling follows Robometer upstream 352d160389daa964788de1ec933d1925f3a6de4f:
robometer/data/datasets/helpers.py::linspace_subsample_frames and
pad_trajectory_to_max_frames_np(pad_from='right'). Keep the existing 10-image
adaptation: seven sampled main-view slots followed by current main/left/right.
"""
import numpy as np

POLICY = 'uniform-prefix8-rint-rightpad-current-v1'


def prefix_indices(current, size=8):
    if current < 0 or size != 8:
        raise ValueError('This experiment requires a nonempty prefix and eight time slots')
    if current + 1 <= size:
        indices = list(range(current + 1))
        return indices + [current] * (size - len(indices))
    indices = np.rint(np.linspace(0, current, size)).astype(int).tolist()
    indices[0], indices[-1] = 0, current
    return indices

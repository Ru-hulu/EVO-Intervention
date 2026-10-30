"""One episode at a time; sample at 5 FPS before building the eight-image window."""
from collections import deque
from fractions import Fraction
import math


def sample_frame_index(sample_number, source_fps, sample_fps=5):
    return math.ceil(sample_number * Fraction(str(source_fps)) / Fraction(str(sample_fps)))


class ObservationWindow:
    def __init__(self, size=8):
        self.frames = deque(maxlen=size)
        self.episode_id = None
        self.size = size

    def push(self, episode_id, frame):
        if episode_id != self.episode_id:
            self.frames.clear()
            self.episode_id = episode_id
        self.frames.append(frame)
        return [self.frames[0]] * (self.size - len(self.frames)) + list(self.frames)


def history_window(frames, index, size=8):
    """Offline equivalent of ObservationWindow, including current but never future frames."""
    return [frames[max(0, i)] for i in range(index - size + 1, index + 1)]

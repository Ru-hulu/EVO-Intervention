"""Incremental CPU inputs for one fixed-resolution, append-only video."""

from itertools import pairwise

import numpy as np
import torch

from .reward import build_reward_messages


class SimpleMemRewardPreprocessor:
    """Cache pixels and tokens; append only new complete temporal patches.

    Frames are CPU uint8 RGB [T,H,W,3], already at the desired resolution,
    with H/W divisible by the processor's patch_size * merge_size. For multiple
    cameras, concatenate only the NEW synchronized frames before append().
    Frame indices refer to the source video, whose fps is fixed at construction.
    Keep processor settings fixed. Reset both this object and the model stream
    between episodes; create a new object when changing the task or fps.
    Returned pixel tensors are views of an append-only CPU buffer; do not mutate.
    """

    def __init__(self, processor, instruction, fps):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.processor, self.fps = processor, fps
        text = processor.apply_chat_template(
            build_reward_messages(instruction), tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        placeholder = processor.vision_start_token + processor.video_token + processor.vision_end_token
        self.prefix, self.suffix = text.split(placeholder)
        self.suffix_ids = self._tokenize(self.suffix)
        self.reset()

    def _tokenize(self, text):
        return self.processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]

    def reset(self):
        self.pixels = self.ids = self.shape = None
        self.rows = self.temporal = 0
        self.last_index = None
        self.patch_keys = []
        self.stats = {}

    def append(self, frames, frame_indices):
        indices = list(frame_indices)
        video = self.processor.video_processor
        patch = video.temporal_patch_size
        factor = video.patch_size * video.merge_size
        if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("Expected CPU uint8 RGB frames [T,H,W,3]")
        if len(frames) == 0 or len(frames) % patch or len(indices) != len(frames):
            raise ValueError("Append complete temporal patches with one source index per frame")
        if any(b <= a for a, b in pairwise(indices)) or (
            self.last_index is not None and indices[0] <= self.last_index
        ):
            raise ValueError("Frame indices must increase; reset before replay or window replacement")
        shape = frames.shape[1:3]
        if any(size % factor for size in shape) or (self.shape is not None and shape != self.shape):
            raise ValueError("Use a fixed resolution divisible by patch_size * merge_size")
        encoded = video(videos=[frames], do_resize=False, do_sample_frames=False, return_tensors="pt")
        temporal, height, width = encoded["video_grid_thw"][0].tolist()
        new_pixels = encoded["pixel_values_videos"]
        times = self.processor._calculate_timestamps(indices.copy(), self.fps, patch)
        visual = (
            self.processor.vision_start_token
            + self.processor.video_token * (height * width // video.merge_size**2)
            + self.processor.vision_end_token
        )
        text = "".join(f"<{time:.1f} seconds>" + visual for time in times)
        # A special vision_end token separates chunks, preventing BPE merges
        # across append boundaries. Include the prompt with the first chunk.
        new_ids = self._tokenize((self.prefix if self.ids is None else "") + text)
        required = self.rows + len(new_pixels)
        if self.pixels is None or required > len(self.pixels):
            capacity = max(required, 2 * len(self.pixels) if self.pixels is not None else required)
            buffer = new_pixels.new_empty(capacity, new_pixels.shape[1])
            if self.rows:
                buffer[: self.rows].copy_(self.pixels[: self.rows])
            self.pixels = buffer
        self.pixels[self.rows : required].copy_(new_pixels)
        self.ids = new_ids if self.ids is None else torch.cat([self.ids, new_ids], dim=1)
        self.rows, self.temporal = required, self.temporal + temporal
        self.shape, self.last_index = shape, indices[-1]
        self.patch_keys.extend(tuple(indices[i : i + patch]) for i in range(0, len(indices), patch))
        self.stats = {"processed_frames": len(frames), "cached_frames": self.temporal * patch}
        ids = torch.cat([self.ids, self.suffix_ids], dim=1)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": (ids == self.processor.video_token_id).long() * 2,
            "pixel_values_videos": self.pixels[: self.rows],
            "video_grid_thw": torch.tensor([[self.temporal, height, width]]),
        }

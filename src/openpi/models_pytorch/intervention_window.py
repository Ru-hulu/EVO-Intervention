"""Sliding-window storage for per-layer Pi05 prefix features."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch
from torch import Tensor


class VLMFeatureSlidingWindow:
    """Store a fixed number of Pi05 prefix feature frames.

    The hidden-state sequence is ordered as all image tokens followed by all
    prompt tokens, matching ``PI05Pytorch.embed_prefix``.

    Inputs to :meth:`append_prefix`:
      hidden_states: tuple of [B, image_tokens + prompt_tokens, D], including
        the embedding output at index 0 and one output per VLM layer after it.
      image_token_counts: number of image tokens contributed by each camera.
      prompt_mask: [B, prompt_tokens] validity mask for the prompt sequence.

    Once full, :meth:`get` returns image features with shape
    ``[B, N, L, K, D]`` and prompt features with shape ``[B, N, L, S, D]``.
    This first version intentionally supports batch size 1 only.
    """

    def __init__(
        self,
        window_size: int = 11,
        selected_layers: Sequence[int] = (1, 3, 5, 7, 9, 11, 13, 15, 17),
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not selected_layers or any(layer < 1 for layer in selected_layers):
            raise ValueError("selected_layers must contain positive post-layer indices")
        if tuple(sorted(selected_layers)) != tuple(selected_layers):
            raise ValueError("selected_layers must be sorted")

        self.window_size = window_size
        self.selected_layers = tuple(selected_layers)
        self._frames: deque[tuple[Tensor, Tensor, Tensor]] = deque(maxlen=window_size)

    def clear(self) -> None:
        self._frames.clear()

    @property
    def ready(self) -> bool:
        return len(self._frames) == self.window_size

    @property
    def length(self) -> int:
        return len(self._frames)

    def append_prefix(
        self,
        hidden_states: Sequence[Tensor],
        image_token_counts: Sequence[int],
        prompt_mask: Tensor,
    ) -> None:
        if not hidden_states:
            raise ValueError("hidden_states cannot be empty")
        if prompt_mask.ndim != 2 or prompt_mask.shape[0] != 1:
            raise ValueError("prompt_mask must have shape [1, prompt_tokens]")
        if len(image_token_counts) == 0 or any(count <= 0 for count in image_token_counts):
            raise ValueError("image_token_counts must contain positive counts")

        selected_hidden_states = []
        for layer_index in self.selected_layers:
            if layer_index >= len(hidden_states):
                raise ValueError(
                    f"Requested hidden-state index {layer_index}, but only {len(hidden_states)} states are available"
                )
            selected_hidden_states.append(hidden_states[layer_index])

        image_token_count = sum(image_token_counts)
        image = torch.stack([state[:, :image_token_count] for state in selected_hidden_states], dim=1)
        prompt = torch.stack([state[:, image_token_count:] for state in selected_hidden_states], dim=1)
        self._frames.append((image.detach(), prompt.detach(), prompt_mask.detach()))

    def get(self) -> dict[str, Tensor] | None:
        """Return a full window, or ``None`` while it is warming up."""
        if not self.ready:
            return None

        images, prompts, prompt_masks = zip(*self._frames, strict=True)
        return {
            "image_tokens": torch.stack(images, dim=1),
            "prompt_tokens": torch.stack(prompts, dim=1),
            "prompt_mask": torch.stack(prompt_masks, dim=1),
        }

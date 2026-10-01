"""Sample K action-chunk candidates from a flow-matching VLA with a single Qwen forward pass.

`framework.predict_action([example] * K)` would run Qwen on K identical copies of the input.
Here Qwen runs once (batch 1) and its hidden states are expanded to K rows; the action head's
own `torch.randn` then draws a different initial noise per row, so the K candidates differ.

Supported (by registry name, `config.framework.name`): QwenGR00T (flow matching),
QwenPI / QwenFM / QwenPI_v3 (layer-wise flow matching).

`seed` makes the candidates reproducible without touching the global RNG; reuse the same seed
to compare methods or checkpoints on the same noise (common random numbers).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from evo_intervention.training.trainer_utils.trainer_tools import resize_images


@torch.no_grad()
def sample_vla_action_candidates(framework, example: dict, num_candidates: int, seed: Optional[int] = None):
    """example: one observation dict (`image`, `lang`, optional `state`), as for `predict_action`.

    Returns normalized candidate chunks [K, H_action, A].
    """
    name = framework.config.framework.name
    if name not in ("QwenGR00T", "QwenPI", "QwenFM", "QwenPI_v3"):
        raise NotImplementedError(f"No single-pass candidate sampler for framework '{name}'")
    K = num_candidates

    images, instructions = [to_pil_preserve(example["image"])], [example["lang"]]
    train_obs_image_size = getattr(framework.config.datasets.vla_data, "obs_image_size", None)
    if train_obs_image_size:
        images = resize_images(images, target_size=train_obs_image_size)
    state = example.get("state")

    if name == "QwenGR00T":
        qwen_inputs = framework.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = framework.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        reference = outputs.hidden_states[-1]  # [1, L, H]
        hidden = reference.expand(K, -1, -1)
    else:
        if name == "QwenPI_v3" and state is not None:
            # QwenPI_v3 feeds the state as discretized text tokens, not to the action head.
            instructions = framework.add_discretized_state_to_instruction(instructions, [state])
            state = None
        layers, attention_mask = framework._encode_vl_hidden_states(images, instructions)
        reference = layers[-1]
        hidden = [h.expand(K, -1, -1) for h in layers]

    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=torch.bool).expand(K, -1)
    if state is not None:
        state = torch.from_numpy(np.asarray(state)[None]).to(reference.device, dtype=reference.dtype)
        state = state.expand(K, *state.shape[1:])

    devices = [reference.device] if reference.is_cuda else []
    with torch.random.fork_rng(devices=devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.float32):
            actions = framework.action_model.predict_action(hidden, state, encoder_attention_mask=attention_mask)
    return actions.float()

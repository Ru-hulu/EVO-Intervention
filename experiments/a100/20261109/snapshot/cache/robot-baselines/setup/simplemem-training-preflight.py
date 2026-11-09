"""Validate installed training kernels and a synthetic multimodal backward pass."""
import json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(8)
from simplememvla.compat import apply_fla_torch_compat
apply_fla_torch_compat()
import deepspeed
import flash_attn
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata
from simplememvla.data.collator import SimpleMemVLADataCollator
from simplememvla.data.messages import build_simplememvla_messages
from simplememvla.benchmarks.rmbench import CAMERA_LABELS
from rmbench_sim.policy import _build_vla

assert qwen.is_fast_path_available
assert all(getattr(qwen, name) is not None for name in (
    'causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
    'fused_recurrent_gated_delta_rule'))
q = torch.randn(1, 128, 4, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
flash_attn.flash_attn_func(q, q, q, causal=True).float().square().mean().backward()
assert torch.isfinite(q.grad).all()
print('FLASH_ATTENTION_BACKWARD_OK', flush=True)
checkpoint = '/cache/robot-baselines/models/SimpleMemVLA/RMBench'
processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
feature = dict(
    messages=build_simplememvla_messages(
        camera_tags=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        history_camera_tags=['cam_high'], instruction='pick up the cube',
        robot_tag='Aloha-AgileX dual-arm robot', control_frequency_hz=17,
        action_horizon=30, camera_labels=CAMERA_LABELS, subtask='Reach for the cube.'),
    videos=[np.zeros((2, 64, 64, 3), dtype=np.uint8)],
    video_metadata=[VideoMetadata(total_num_frames=2, fps=2.0, frames_indices=[0, 1], duration=1.0, video_backend='lerobot')],
    images=[np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)],
    actions=torch.zeros(30, 14), action_mask=torch.ones(30, 14, dtype=torch.bool), state=torch.zeros(14),
)
batch = SimpleMemVLADataCollator(processor, max_length=8192)([feature])
batch = {key: value.to('cuda') for key, value in batch.items()}
print('COLLATOR_OK', batch['input_ids'].shape, flush=True)
from transformers.initialization import no_init_weights
with no_init_weights():
    model = _build_vla(checkpoint, torch.bfloat16, 'flash_attention_2').to('cuda').train()
print('MODEL_LOADED_OK', flush=True)
model.gradient_checkpointing_enable()
output = model(**batch)
assert torch.isfinite(output.loss)
print('FORWARD_OK', float(output.loss), flush=True)
output.loss.backward()
counts = {}
for prefix in ('backbone', 'action_head'):
    grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), prefix
    counts[prefix] = len(grads)
report = dict(torch=torch.__version__, deepspeed=deepspeed.__version__, flash_attn=flash_attn.__version__,
              device=torch.cuda.get_device_name(), checkpoint=checkpoint, synthetic_data=True,
              loss=float(output.loss), action_loss=float(output.action_loss), vl_loss=float(output.vl_loss),
              finite_gradient_tensors=counts, peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
Path('/cache/robot-baselines/manifests/simplemem-training-preflight.json').write_text(json.dumps(report, indent=2)+'\n')
print('TRAINING_BACKWARD_OK', json.dumps(report), flush=True)

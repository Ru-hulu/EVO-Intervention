"""加载进度检查点及持续的三视角在线预测接口。"""

import torch
from transformers import AutoProcessor

from mem_lite.datasets.m3_inputs import M3Collator
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder, verify_vision_source
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.train_stage3 import CHECKPOINT_FORMAT, load_trainable_state


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_checkpoint(path, device="cuda", model_path=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != CHECKPOINT_FORMAT:
        raise ValueError("需要 StreamForest 进度模型检查点")
    config = checkpoint["config"]
    base = model_path or config["model_path"]
    verify_vision_source(base, config["history_source"])
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    model = M3ProgressPredictor(base, dtype=dtype, **{**config["model"], "gradient_checkpointing": False}).to(device)
    load_trainable_state(model, checkpoint["model"])
    processor = AutoProcessor.from_pretrained(base, max_pixels=config["max_pixels"], local_files_only=True)
    return model.eval(), processor, config


class StreamingProgressPredictor:
    """逐次调用保留记忆；episode_id 或任务指令改变时自动重置。"""

    def __init__(self, model, processor, history_source, max_sequence_length=8192):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.stream = StreamMemory(
            QwenFrameEncoder(model.qwen.model.visual), processor.image_processor,
            history_source["memory_config"], history_source["max_pixels"], history_source["min_pixels"],
        )
        if model.history_projection.in_features != self.stream.encoder.hidden_size:
            raise ValueError("历史投影维度与原生视觉特征不一致")
        self.collator = M3Collator(processor, self.stream.encoder.hidden_size, max_sequence_length,
                                  self.stream.memory.token_budget)

    def reset(self):
        self.stream.reset()

    @torch.no_grad()
    def __call__(self, images, frame_index, episode_id, task_prompt):
        if len(images) != 3:
            raise ValueError("在线预测需要固定顺序的3个视角")
        history = self.stream.step(images[0], frame_index, episode_id, task_prompt)
        inputs = self.collator([{"images": images, "task_prompt": task_prompt, "history_tokens": history}])
        result = self.model(to_device(inputs, self.device))
        result["history_length"] = 0 if history is None else len(history)
        return result

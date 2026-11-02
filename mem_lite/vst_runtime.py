"""Online one-frame inference using the same VST model as windowed training."""

import torch
from transformers import AutoProcessor

from mem_lite.datasets.vst_dataset import VSTCollator
from mem_lite.models.vst_predictor import VSTProgressPredictor
from mem_lite.train_vst import CHECKPOINT_FORMAT, device_batch, load_trainable_state


class VSTStreamingPredictor:
    def __init__(self, checkpoint_path, device="cuda", model_path=None):
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["format"] != CHECKPOINT_FORMAT:
            raise ValueError("Expected a VST KV-cache checkpoint")
        self.config = saved["config"]
        path = model_path or self.config["model_path"]
        self.device = torch.device(device)
        self.model = VSTProgressPredictor(path, **self.config["model"]).to(self.device).eval()
        load_trainable_state(self.model, saved["model"])
        processor = AutoProcessor.from_pretrained(path, max_pixels=self.config["max_pixels"], local_files_only=True)
        self.collator = VSTCollator(processor, self.config["compression_ratio"], self.config["max_sequence_length"])
        self.reset()

    def reset(self):
        self.cache, self.stream, self.last_frame = None, None, None

    @torch.inference_mode()
    def predict(self, images, task_prompt, episode_id, frame_index):
        identity = (episode_id, task_prompt)
        if identity != self.stream:
            self.reset()
            self.stream = identity
        if self.last_frame is not None and frame_index <= self.last_frame:
            raise ValueError("Frames must increase within an episode; reset before replay")
        batch = device_batch(self.collator([{"images": images, "task_prompt": task_prompt}]), self.device)
        result = self.model(batch, cache=self.cache)
        self.cache, self.last_frame = result["cache"], frame_index
        return {"pt": float(result["pt"][0]), "history_tokens": int(self.cache.lengths[0])}

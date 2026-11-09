"""Frozen causal FOREST reward; policy features come independently from IMPRINT."""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from mem_lite.runtime import load_checkpoint, StreamingProgressPredictor, to_device
from serve_robomme_tcp import TCPPolicyServer


class Forest:
    def __init__(self, checkpoint):
        model, processor, cfg = load_checkpoint(checkpoint)
        model.requires_grad_(False)
        self.predictor = StreamingProgressPredictor(model, processor, cfg['history_source'], cfg['max_sequence_length'])
        self.reset_model()

    def reset_model(self):
        self.predictor.reset()
        self.last_frame = -1
        return {'reset': True}

    @torch.inference_mode()
    def observe(self, payload):
        frames = payload['frames']
        if not frames:
            raise ValueError('Cannot score an empty observation sequence')
        prompt = payload['prompt']
        stream = self.predictor.stream
        history = None
        for item in frames:
            index = int(item['frame'])
            if index <= self.last_frame:
                raise ValueError('Frame indices must increase within an episode')
            # Match the IMPRINT training-video convention: 20 frames/s -> 5 Hz.
            # Raw camera channel ordering matches capture.py -> training videos.
            if index % 4 == 0:
                history = stream.step(Image.fromarray(item['head']), index, payload['episode'], prompt)
            else:
                history = stream.memory.get_memory_tokens()
            self.last_frame = index
        images = [Image.fromarray(np.asarray(x, dtype=np.uint8)) for x in payload['images']]
        inputs = self.predictor.collator([dict(images=images, task_prompt=prompt, history_tokens=history)])
        result = self.predictor.model(to_device(inputs, self.predictor.device))
        progress = float(result['pt'][0])
        if not np.isfinite(progress):
            raise ValueError('Non-finite FOREST output')
        return dict(progress=progress, frame=self.last_frame,
                    history_tokens=0 if history is None else len(history))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', required=True, type=int)
    a = p.parse_args()
    torch.set_num_threads(4)
    model = Forest(a.checkpoint)
    print(json.dumps({'event': 'forest_ready', 'checkpoint': a.checkpoint}), flush=True)
    TCPPolicyServer(model, host='127.0.0.1', port=a.port).serve_forever()

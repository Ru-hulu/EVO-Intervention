"""Optional real-weight GPU check; writes only a new small scratch cache.

No optimizer, training, label generation or existing-output overwrite.
Use --sample-file to read images/PT from an existing episode without its history.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.datasets.prepare_stream import DEFAULT_MEMORY_CONFIG, build_episode
from mem_lite.memory.splitting import override_memory_config
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder, load_vision_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.frames < 12:
        parser.error("至少 12 帧以覆盖默认窗口的事件切分")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source = M3Dataset([args.sample_file], load_history=False)
    if len(source.episodes) != 1 or len(source) < args.frames:
        raise ValueError("需要包含足够帧的一条源 episode")
    indices = [i * (len(source) - 1) // (args.frames - 1) for i in range(args.frames)]
    observations = []
    for index in indices:
        row = source[index]
        observations.append({"frame_index": row["frame_index"], "images": row["images"], "pt": row["target_pt"]})
    visual, processor, vision_hash = load_vision_encoder(args.model_path, args.device)
    config = override_memory_config(json.loads(DEFAULT_MEMORY_CONFIG.read_text()), "max_depth", args.clip_model_path)
    stream = StreamMemory(QwenFrameEncoder(visual), processor, config)
    origin = {"method": "streamforest-v1", "vision_sha256": vision_hash, "memory_config": config,
              "max_pixels": 65536, "min_pixels": 4096, "sampling": "evenly_spaced_smoke_only"}
    started = perf_counter()
    summary = build_episode(observations, stream, output / "episode", "smoke check", "smoke", origin)
    if stream.memory.frames_in_long == 0:
        raise AssertionError("未触发事件切分")
    committed = stream.memory.frames_in_long
    clip_hashes = dict(stream.boundary_encoder.provenance)
    cached = M3Dataset([output / "episode"])
    stream.bind_history_source(summary["history_source"], require_provenance=True)
    stream.reset()
    # Recreate the online path from the saved images under the inference autocast.
    for row in cached:
        with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16):
            actual = stream.step(row["images"][0], row["frame_index"], row["episode_id"], row["task_prompt"])
        if actual is None:
            assert row["history_tokens"] is None
        else:
            torch.testing.assert_close(actual.cpu(), row["history_tokens"], rtol=0, atol=0)
            assert len(actual) <= stream.memory.token_budget
    assert stream.boundary_encoder.provenance == clip_hashes
    result = {"passed": True, "frames": len(cached), "frames_committed_to_events": committed,
              "cache_online_history_bit_exact": True, "device": args.device,
              "history_capacity": stream.memory.token_budget,
              "history_dim": stream.encoder.hidden_size, "clip_dim": stream.memory.last_frame.numel(),
              "seconds_excluding_model_load": perf_counter() - started,
              "clip": clip_hashes, "training_performed": False}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

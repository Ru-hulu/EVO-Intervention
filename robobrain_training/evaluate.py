"""用同一输入模板生成奖励分数，保留解析失败并计算 PT 的 MAE/RMSE。"""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from .data import RewardCollator, RewardDataset, parse_score
from .model import load_model, to_device


@torch.no_grad()
def predict(model, collator, rows):
    inputs = to_device(collator.encode(rows, supervised=False), next(model.parameters()).device)
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=32, use_cache=True)
    outputs = generated[:, inputs["input_ids"].shape[1]:]
    texts = collator.processor.batch_decode(outputs, skip_special_tokens=True)
    return [{"id": row["id"], "episode_id": row["episode_id"], "frame_index": row["frame_index"],
             "target_pt": row["pt"], "predicted_pt": parse_score(text), "response": text}
            for row, text in zip(rows, texts, strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", help="Omit to evaluate the unmodified base checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    rows = RewardDataset(config["validation_data"]).rows
    if args.limit:
        rows = rows[:args.limit]
    model, processor = load_model(config, adapter=args.adapter)
    model.eval()
    collator = RewardCollator(processor, config["max_length"])
    records = []
    started = time.monotonic()
    with (args.output_dir / "predictions.jsonl").open("w") as stream:
        for start in range(0, len(rows), args.batch_size):
            batch = predict(model, collator, rows[start:start + args.batch_size])
            records.extend(batch)
            for record in batch:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    errors = [100 * (row["predicted_pt"] - row["target_pt"]) for row in records if row["predicted_pt"] is not None]
    report = {"samples": len(records), "valid_predictions": len(errors), "parse_failures": len(records) - len(errors),
              "mae_pp_on_valid": sum(abs(x) for x in errors) / len(errors) if errors else None,
              "rmse_pp_on_valid": math.sqrt(sum(x*x for x in errors) / len(errors)) if errors else None,
              "conditioning": "same-video start/current/end", "adapter": args.adapter,
              "elapsed_seconds": time.monotonic() - started}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()

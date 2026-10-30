"""检查全部数据；可选真实 LoRA 前向及 Trainer.evaluate，不进行训练。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from torch.utils.data import Subset

from .data import RewardCollator, RewardDataset, parse_score, score_text
from .evaluate import predict
from .model import load_model, to_device
from .prepare import sample_indices
from .train import make_trainer


def check_data(config):
    datasets = {split: RewardDataset(config[key]) for split, key in
                (("train", "train_data"), ("validation", "validation_data"))}
    root = Path(config["train_data"]).parent
    manifest = json.loads((root / "manifest.json").read_text())
    source = json.loads(Path(manifest["source_manifest"]).read_text())
    if hashlib.sha256(Path(manifest["source_manifest"]).read_bytes()).hexdigest() != manifest["source_manifest_sha256"]:
        raise ValueError("Original split manifest changed")
    episode_sets = {}
    for split, dataset in datasets.items():
        expected = {row["video_id"] for row in source["videos"] if row["split"] == split}
        actual = {row["episode_id"] for row in dataset.rows}
        if actual != expected or len({row["id"] for row in dataset.rows}) != len(dataset):
            raise ValueError("Dataset split or sample IDs differ from source")
        episode_sets[split] = actual
        for episode in (entry for entry in manifest["episodes"] if entry["split"] == split):
            rows = [row for row in dataset.rows if row["episode_id"] == episode["episode_id"]]
            if [row["frame_index"] for row in rows] != sample_indices(episode["raw_frame_count"], episode["fps"]):
                raise ValueError("Sample grid differs from 5 FPS")
            with Path(episode["labels_source"]).open() as stream:
                labels = {int(row["frame_index"]): float(row["pt"]) for row in csv.DictReader(stream)}
            for row in rows:
                if row["pt"] != labels[row["frame_index"]] or row["conversations"][1]["value"] != score_text(row["pt"]):
                    raise ValueError("Reward differs from original PT")
                if row["image"][0] != row["image"][2] or row["image"][2:5] != rows[0]["image"][5:8]:
                    raise ValueError("BEFORE must be the same episode's starting observations")
                if row["image"][1] != rows[0]["image"][1] or len(row["image"]) != 8:
                    raise ValueError("Reference endpoint or image order changed")
                if any(not Path(path).is_file() for path in row["image"]):
                    raise ValueError("Missing sample image")
                if abs(parse_score(row["conversations"][1]["value"]) - row["pt"]) > 0.000501:
                    raise ValueError("Target rounding exceeds 0.05 percentage points")
    if episode_sets["train"] & episode_sets["validation"]:
        raise ValueError("Video split leakage")
    return datasets, {"data_checks": "passed", "splits": manifest["splits"],
                      "same_video_endpoint_conditioning": True, "expert_videos_excluded": manifest["expert_videos_excluded"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    datasets, report = check_data(config)
    report.update(training_started=False, backward_calls=0, optimizer_steps=0)
    if args.forward:
        torch.set_num_threads(8)
        torch.manual_seed(config["seed"])
        model, processor = load_model(config, trainable=True)
        model.eval()
        collator = RewardCollator(processor, config["max_length"])
        rows = [dataset[index] for dataset in datasets.values() for index in (0, len(dataset)//2, len(dataset)-1)]
        checks = []
        for start in range(0, len(rows), 2):
            selected = rows[start:start+2]
            batch = collator(selected)
            prompts = collator.encode(selected, supervised=False)
            for row, item in enumerate(selected):
                labels = batch["labels"][row]
                answer = processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=True)
                if answer != item["conversations"][1]["value"]:
                    raise ValueError("Loss mask includes non-answer tokens")
                sequence = batch["input_ids"][row][batch["attention_mask"][row].bool()]
                prompt = prompts["input_ids"][row][prompts["attention_mask"][row].bool()]
                if not torch.equal(sequence[:len(prompt)], prompt):
                    raise ValueError("Training and generation use different prompts")
            with torch.no_grad():
                loss = model(**to_device(batch, "cuda")).loss.item()
            if not torch.isfinite(torch.tensor(loss)):
                raise ValueError("Non-finite forward loss")
            checks.append({"ids": [row["id"] for row in selected], "loss": loss,
                           "sequence_tokens": batch["input_ids"].shape[1]})
        # 使用实际 Trainer 验证 API 和评估入口；不调用 train 或创建优化器。
        with tempfile.TemporaryDirectory(prefix="robobrain-trainer-check-") as directory:
            trainer = make_trainer(config, model, processor, output_dir=directory)
            evaluation = trainer.evaluate(eval_dataset=Subset(datasets["validation"], [0, len(datasets["validation"])//2]))
            if trainer.state.global_step != 0 or trainer.optimizer is not None:
                raise ValueError("Check unexpectedly entered training")
        model.eval()
        predictions = predict(model, collator, [datasets["validation"][len(datasets["validation"])//2]])
        report.update(forward_checks=checks, trainer_evaluation=evaluation,
                      sample_generation=predictions, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      gradients_absent=all(p.grad is None for p in model.parameters()))
    report["status"] = "passed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

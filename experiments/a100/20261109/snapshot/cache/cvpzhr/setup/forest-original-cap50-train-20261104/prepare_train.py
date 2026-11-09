"""Materialize the confirmed run only from completed, published caches."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PREP = Path("/cache/cvpzhr/setup/forest-proprimem-cap50-20261103")
BASE = Path("/cache/cvpzhr/datasets/forest-proprimem-cap50-5fps-20261103")
AUG = Path("/cache/cvpzhr/datasets/forest-proprimem-cap50-5fps-static-jitter-seed42-20261104")
sys.path.insert(0, str(ROOT / "code"))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_code():
    for name, digest in read(ROOT / "implementation-sha256.json").items():
        require(sha(ROOT / name) == digest, f"Training implementation changed: {name}")


def cache_gate():
    require(read(PREP / "pipeline-status.json")["phase"] == "completed",
            "Cache preparation is not completed; training was not started")
    base, aug = read(BASE / "report.json"), read(AUG / "report.json")
    require(base["status"] == aug["status"] == "completed_with_review_exclusion",
            "Cache reports are incomplete")
    require(base["episodes"] == {"train": 1656, "validation": 417}, "Wrong base episode counts")
    require(base["samples"] == {"train": 272091, "validation": 66891}, "Wrong base sample counts")
    require(aug["episodes"] == 1656 and aug["augmented_samples"] == 318931,
            "Wrong augmented dataset")
    for flag in ("dense_pt_files_unchanged", "original_observation_pt_unchanged",
                 "inserted_pt_fixed_to_anchor", "views_synchronized", "histories_causal",
                 "validation_unchanged"):
        require(aug[flag] is True, f"Augmentation validation failed: {flag}")
    return read(PREP / "cache-plan.json")


def materialize():
    check_code()
    plan = cache_gate()
    config = read(ROOT / "config.template.json")
    inputs = [PREP / "cache-plan.json", PREP / "augmentation-policy.json",
              BASE / "report.json", AUG / "report.json",
              AUG / "train.sample_files.json", BASE / "validation.sample_files.json"]
    paths = {
        "train": read(AUG / "train.sample_files.json"),
        "validation": read(BASE / "validation.sample_files.json"),
    }
    expected = {
        split: [str((AUG if split == "train" else BASE) / "cache" / split / row["source_id"])
                for row in plan["sources"] if row["split"] == split]
        for split in paths
    }
    require(paths == expected, "Published paths differ from the agreed source split")
    require(read(AUG / "validation.sample_files.json") == paths["validation"],
            "Validation must use the unchanged base cache")
    config["sample_files"] = paths["train"]
    config["validation_sample_files"] = paths["validation"]
    config["validation_suite_by_episode"] = {
        str(BASE / "cache/validation" / row["source_id"]): row["suite"]
        for row in plan["sources"] if row["split"] == "validation"
    }
    config["preparation_hashes"] = {str(path): sha(path) for path in inputs}
    from mem_lite.datasets.m3_dataset import M3Dataset
    signatures, ids, sizes = {}, {}, {}
    for split, files in paths.items():
        data = M3Dataset(files, config["model"]["history_dim"])
        require(len(data) == (318931 if split == "train" else 66891), "Unexpected sample count")
        require(data.history_source == config["history_source"] == plan["history_source"],
                "History source differs from the original Forest setup")
        require(data.history_capacity == config["history_capacity"], "History budget changed")
        ids[split] = {summary["episode_id"] for _, summary in data.episodes}
        require(len(ids[split]) == len(files), "Duplicate source episode")
        sizes[split] = {}
        for directory, summary in data.episodes:
            require(summary["source"]["split"] == split, "Wrong episode split")
            require(("augmentation" in summary) == (split == "train"), "Wrong augmentation split")
            sizes[split][str(directory)] = summary["num_samples"]
        signatures[split] = data.signature
        del data
    require(not ids["train"] & ids["validation"], "Source episode leakage between splits")
    config["data_signature"] = signatures["train"]
    config["validation_data_signature"] = signatures["validation"]
    target = ROOT / "config.json"
    if target.exists() and read(target) != config:
        require(not Path(config["output_dir"]).exists(), "Do not change a started experiment")
    write(target, config)

    smoke = json.loads(json.dumps(config))
    # Long training episodes exercise history capacity; short validation episodes keep this bounded.
    for split, key in (("train", "sample_files"), ("validation", "validation_sample_files")):
        smoke[key] = [
            (max if split == "train" else min)(
                [p for p in paths[split] if Path(p).parts[-3] == suite],
                key=lambda p: sizes[split][p],
            )
            for suite in ("Count", "Order", "State")
        ]
    for key in ("data_signature", "validation_data_signature"):
        smoke.pop(key)
    smoke.update(output_dir=str(ROOT / "smoke-output"), max_steps=2, num_workers=0,
                 validation_num_workers=0, persistent_workers=False, log_every_steps=1)
    write(ROOT / "smoke.config.json", smoke)
    print(json.dumps({"status": "configured", "training_started": False,
                      "config": str(target), "train_samples": 318931, "validation_samples": 66891}))


def receipt_identity(world_size):
    return {
        "config_sha256": sha(ROOT / "config.json"),
        "implementation_sha256": sha(ROOT / "implementation-sha256.json"),
        "world_size": world_size,
    }


def finish_smoke(world_size):
    import torch
    smoke = read(ROOT / "smoke.config.json")
    out = Path(smoke["output_dir"])
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    logs = [json.loads(line) for line in (out / "validation.jsonl").read_text().splitlines()]
    require(last["step"] == 3, "Smoke resume must complete the third optimizer step")
    for state in (last, best):
        require(state["format"] == "mem-lite-streamforest-v1", "Wrong model architecture")
        require(len(state["rng_by_rank"]) == world_size, "Wrong smoke world size")
        require(set(state["model"]) == {"m3_lora", "history_projection", "progress_head"},
                "Wrong trainable modules")
        for part in state["model"].values():
            require(all(torch.isfinite(value).all().item() for value in part.values()),
                    "Non-finite checkpoint tensors")
    require(best["step"] == last["config"]["best_validation_step"], "Wrong best checkpoint")
    require(last["config"]["best_validation_mae_pp"] == min(row["mae_pp"] for row in logs),
            "Best metric did not survive resume")
    from mem_lite.datasets.m3_dataset import M3Dataset
    count = len(M3Dataset(smoke["validation_sample_files"]))
    require(all(row["num_samples"] == count for row in logs), "Validation duplicate or missing frames")
    require(all(math.isfinite(row["mae_pp"]) for row in logs), "Invalid validation metric")
    write(ROOT / "smoke-passed.json", dict(status="passed", **receipt_identity(world_size),
                                          checkpoint_readback=True, resume_step=3,
                                          validation_samples=count))
    print("Real-cache GPU smoke, checkpoint readback and one-step resume passed")


def check_launch(world_size):
    check_code()
    cache_gate()
    receipt = read(ROOT / "smoke-passed.json")
    require(receipt.get("status") == "passed" and
            all(receipt.get(key) == value for key, value in receipt_identity(world_size).items()),
            "Smoke receipt does not match config, code or world size")
    config = read(ROOT / "config.json")
    for path, digest in config["preparation_hashes"].items():
        require(sha(Path(path)) == digest, f"Prepared input changed: {path}")
    require(not Path(config["output_dir"]).exists(), "Output directory already exists; do not overwrite")
    print("Full training launch checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "finish-smoke", "check-launch"))
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        materialize()
    elif args.command == "finish-smoke":
        finish_smoke(args.world_size)
    else:
        check_launch(args.world_size)

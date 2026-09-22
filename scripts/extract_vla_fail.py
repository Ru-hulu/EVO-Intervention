"""Extract LLMD features from a JSONL manifest of raw policy observations.

Each line: {"episode_id": "unique-id", "observations": ["frame0.npz", ...],
            "success": true}. NPZ paths are relative to the manifest. Fields
must match Policy.infer's raw inputs (not LeRobot's storage schema). No pickle.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from vla_fail.llmd import make_fixed_noise
from vla_fail.runtime import FEATURE_METADATA

from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-noise", type=Path, help="Saved .npy; reuse for all data splits")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = get_config(args.config)
    policy = create_trained_policy(config, args.checkpoint, pytorch_device=args.device)
    noise = (
        make_fixed_noise((config.model.action_horizon, config.model.action_dim), args.seed)
        if args.fixed_noise is None
        else np.load(args.fixed_noise, allow_pickle=False)
    )
    metadata = {**FEATURE_METADATA, "config": args.config, "checkpoint": args.checkpoint}
    args.output.mkdir(parents=True, exist_ok=True)
    noise_path = args.output / "fixed_noise.npy"
    if noise_path.exists() and not np.array_equal(np.load(noise_path, allow_pickle=False), noise):
        raise ValueError("Output folder contains a different fixed prior")
    np.save(noise_path, noise)
    entries = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [entry["episode_id"] for entry in entries]
    if any(not isinstance(eid, str) or not eid for eid in ids) or len(set(ids)) != len(ids):
        raise ValueError("episode_id values must be unique nonempty strings")
    for index, entry in enumerate(entries):
        destination = args.output / f"episode_{index:06d}.npz"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        policy.reset()
        features = []
        for filename in entry["observations"]:
            with np.load(args.manifest.parent / filename, allow_pickle=False) as frame:
                obs = {key: frame[key] for key in frame.files}
            if "prompt" in obs:
                obs["prompt"] = str(obs["prompt"].item())
            features.append(policy.extract_llmd_features(obs, fixed_noise=noise))
        if not features:
            raise ValueError("Each episode needs at least one observation")
        labels = {}
        if "success" in entry:
            if type(entry["success"]) is not bool:
                raise ValueError("success must be a boolean")
            labels["success"] = np.array(entry["success"])
        np.savez_compressed(
            destination,
            features=np.stack(features),
            fixed_noise=noise,
            metadata=np.array(json.dumps(metadata)),
            episode_id=np.array(entry["episode_id"]),
            **labels,
        )
        print(f"{destination}: {len(features)} observations")


if __name__ == "__main__":
    main()

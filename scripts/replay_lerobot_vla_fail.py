"""Offline pi05 replay for LLMD/ACC; never connects to robot hardware.

This adapter uses an installed LeRobot runtime and its checkpoint processors.
It captures a fixed-prior pass before normal denoising, sharing prefix features.
It checks its first action chunk against the unmodified sampler with identical
action noise. No trained weights or installed LeRobot files are changed.
"""

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
import numpy as np
from safetensors.torch import load_file
import torch
import transformers
from vla_fail.llmd import make_fixed_noise


def capture_features(policy, batch, fixed_noise, *, actions=True, noise=None):
    """Single-threaded instance-local hook, restored even when inference fails."""
    core = policy.model
    original = core.denoise_step
    captured = []
    sampled_times = []

    def hook(_module, inputs):
        captured.append(inputs[0].detach().float().cpu().numpy()[0].copy())

    def monitored(**kwargs):
        if not sampled_times:
            timestep = kwargs["timestep"]
            if not torch.all(timestep == 1):
                raise ValueError("Expected pi05 to start at the pure-noise time=1 endpoint")
            sampled_times.append(1.0)
            diagnostic = {**kwargs, "x_t": fixed_noise, "timestep": torch.ones_like(timestep)}
            handle = core.action_out_proj.register_forward_pre_hook(hook)
            try:
                diagnostic_velocity = original(**diagnostic)
            finally:
                handle.remove()
            if not actions:
                return diagnostic_velocity
        return original(**kwargs)

    core.denoise_step = monitored
    try:
        result = policy.predict_action_chunk(
            batch,
            noise=noise if actions else fixed_noise,
            num_steps=policy.config.num_inference_steps if actions else 1,
        )
    finally:
        core.denoise_step = original
    if len(captured) != 1 or not np.isfinite(captured[0]).all():
        raise RuntimeError("Expected exactly one finite fixed-prior action-head feature sample")
    return captured[0], result if actions else None


def xyz_from_joint_degrees(actions, fk):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
        raise ValueError("Piper bimanual conversion expects finite [H,14] absolute joint actions in degrees")
    result = {}
    for arm, start in (("left", 0), ("right", 7)):
        xyz = [fk.CalFK(np.deg2rad(row[start : start + 6]).tolist())[-1][:3] for row in actions]
        result[f"positions_{arm}"] = np.asarray(xyz, dtype=np.float32) / 1000  # SDK mm -> metres.
    return result


def load_policy(checkpoint, device, tokenizer=None):
    config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
    if config.type != "pi05" or config.use_relative_actions:
        raise ValueError("This adapter requires pi05 with absolute actions")
    expected_names = [
        f"{arm}_{name}.pos" for arm in ("left", "right") for name in [*[f"joint_{i}" for i in range(1, 7)], "gripper"]
    ]
    if config.action_feature_names != expected_names:
        raise ValueError("Checkpoint joint ordering does not match the bimanual Piper adapter")
    config.device = device
    config.dtype = "float32"
    config.compile_model = False
    config.gradient_checkpointing = False
    config.rtc_config = None
    policy = PI05Policy(config)
    # The runtime's convenience from_pretrained catches weight errors and may
    # return a random model. Load explicitly and fail closed on every mismatch.
    weights = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    weights = policy._fix_pytorch_state_dict_keys(weights, config)  # noqa: SLF001 - runtime checkpoint compatibility
    weights = {key if key.startswith("model.") else f"model.{key}": value for key, value in weights.items()}
    policy.load_state_dict(weights, strict=True)
    del weights
    gc.collect()
    policy.eval()
    processors = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer or checkpoint / "tokenizer")},
        },
    )
    return policy, processors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", nargs="*", type=Path, default=[])
    parser.add_argument("--episodes", nargs="*", type=Path, default=[])
    parser.add_argument("--episode-group", choices=["failure", "success", "evaluation"], default="failure")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer", type=Path, help="Local tokenizer path if not bundled with the checkpoint")
    parser.add_argument("--reference-note", default="Independent reference collection; see bundle provenance.")
    parser.add_argument(
        "--wait-bundle-seconds",
        type=float,
        default=0,
        help="Wait for each explicitly named bundle to arrive by atomic file transfer",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--action-seed", type=int, default=20260922)
    parser.add_argument("--dh-offset", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max-queries", type=int, help="Explicit smoke-test limit; omit for complete episodes")
    args = parser.parse_args()
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

    torch.set_num_threads(8)
    policy, (preprocess, postprocess) = load_policy(args.checkpoint, args.device, args.tokenizer)
    print("WEIGHTS_LOADED_STRICTLY", flush=True)
    fixed = make_fixed_noise((policy.config.chunk_size, policy.config.max_action_dim), args.noise_seed)
    fixed_tensor = torch.from_numpy(fixed).to(args.device)[None]
    fk = C_PiperForwardKinematics(dh_is_offset=args.dh_offset)
    source = Path(inspect.getfile(type(policy.model))).read_bytes()
    metadata = {
        "feature_source": "lerobot.pi05.action_out_proj.input",
        "flow_time": 1.0,
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": (args.checkpoint / "model.safetensors").stat().st_size,
        "model_source_sha256": hashlib.sha256(source).hexdigest(),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    camera_keys = {
        "base": "observation.images.base_0_rgb",
        "left_wrist": "observation.images.left_wrist_0_rgb",
        "right_wrist": "observation.images.right_wrist_0_rgb",
    }
    verified = False
    for kind, paths in (("reference", args.reference), ("failure", args.episodes)):
        for path in paths:
            deadline = time.monotonic() + args.wait_bundle_seconds
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(2)
            group = "reference" if kind == "reference" else args.episode_group
            destination = args.output / group / path.name
            if destination.exists():
                raise FileExistsError(destination)
            with np.load(path, allow_pickle=False) as archive:
                bundle = {key: archive[key] for key in archive.files}
            count = len(bundle["frame_indices"])
            if args.max_queries is not None:
                count = min(count, args.max_queries)
            if kind == "failure" and (
                np.any(bundle["executed_steps"][1:count] >= policy.config.chunk_size)
                or np.any(bundle["executed_steps"][1:count] <= 0)
            ):
                raise ValueError("Replay query spacing must satisfy 0 < R < chunk_size for ACC")
            policy.reset()
            feature_rows, action_rows, left, right, latency = [], [], [], [], []
            for index in range(count):
                observation = {
                    "observation.state": torch.from_numpy(bundle["state"][index]),
                    "task": str(bundle["task"].item()),
                }
                for name, key in camera_keys.items():
                    observation[key] = torch.from_numpy(bundle[f"images_{name}"][index]).permute(2, 0, 1)
                batch = preprocess(observation)
                # Per-frame noise is reproducible and separate from the LLMD fixed prior.
                seed = args.action_seed + int(bundle["frame_indices"][index])
                generator = torch.Generator(device=args.device).manual_seed(seed)
                noise = torch.randn(fixed_tensor.shape, generator=generator, device=args.device)
                started = time.monotonic()
                with torch.inference_mode():
                    features, actions = capture_features(
                        policy, batch, fixed_tensor, actions=kind == "failure", noise=noise
                    )
                    if kind == "failure":
                        if not verified:
                            baseline = policy.predict_action_chunk(batch, noise=noise)
                            torch.testing.assert_close(actions, baseline, rtol=1e-5, atol=1e-5)
                            print("SAMPLER_ACTION_EQUIVALENCE_PASSED", flush=True)
                            verified = True
                        actions = postprocess(actions)[0].detach().float().cpu().numpy()
                        xyz = xyz_from_joint_degrees(actions, fk)
                        action_rows.append(actions)
                        left.append(xyz["positions_left"])
                        right.append(xyz["positions_right"])
                feature_rows.append(features)
                latency.append(time.monotonic() - started)
                if index % 50 == 0 or index == count - 1:
                    print(f"{group}/{path.name} query={index+1}/{count} seconds={np.mean(latency):.3f}", flush=True)
            extra = {}
            if kind == "failure":
                extra = {
                    "predicted_actions": np.stack(action_rows),
                    "positions_left": np.stack(left),
                    "positions_right": np.stack(right),
                }
            if "success" in bundle:
                extra["success"] = bundle["success"]
            notes = (
                "Offline counterfactual replay; RTC disabled. Fresh model-predicted chunks, not original runtime chunks. "
                f"Piper SDK FK: joint degrees to radians; DH offset={args.dh_offset}; millimetres to metres. "
                f"Reference: {args.reference_note} "
                "No success calibration or verified failure-onset annotation."
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                features=np.stack(feature_rows),
                fixed_noise=fixed,
                metadata=np.array(json.dumps(metadata)),
                episode_id=bundle["episode_id"],
                timestamps=bundle["timestamps"][:count],
                frame_indices=bundle["frame_indices"][:count],
                executed_steps=bundle["executed_steps"][:count],
                query_latency_s=np.array(latency),
                provenance=bundle["provenance"],
                evaluation_notes=np.array(notes),
                **extra,
            )
            print(f"SAVED {destination}", flush=True)
            del bundle, feature_rows, action_rows, left, right
            gc.collect()


if __name__ == "__main__":
    main()

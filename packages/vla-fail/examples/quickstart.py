"""Reproduce the hand-calculated examples; these are NOT MuJoCo experiments."""

import json
from pathlib import Path

import numpy as np
from vla_fail import ACCDetector
from vla_fail import LLMDDetector
from vla_fail import VLAFAILDetector
from vla_fail import calibrate_threshold
from vla_fail import make_fixed_noise


def main():
    # Toy H=2, F=2 feature data. Real features must come from the selected model.
    train = np.array(
        [
            [[1, 8], [4, 3]],
            [[1, 12], [4, 7]],
            [[3, 8], [6, 3]],
            [[3, 12], [6, 7]],
        ],
        dtype=float,
    )
    llmd = LLMDDetector(
        fixed_noise=make_fixed_noise((2, 7)), metadata={"checkpoint": "toy-only", "feature_source": "action_head_input"}
    )
    llmd.partial_fit(train[:2]).partial_fit(train[2:]).finalize()
    llmd.threshold = 10.0  # Illustration only, NOT a calibrated real threshold.
    token_only = VLAFAILDetector(llmd=llmd)
    print("LLMD example (expected score approximately 13):")
    print(json.dumps(token_only.step(tokens=[[3, 12], [8, 9]]).to_dict(), indent=2))

    old = np.array([[0.1, 0.2, 0.1], [0.2, 0.2, 0.1], [0.3, 0.2, 0.1], [0.4, 0.2, 0.1], [0.5, 0.2, 0.1]])
    new = np.array([[0.1, 0.24, 0.1], [0.2, 0.24, 0.1], [0.3, 0.24, 0.1], [0.4, 0.24, 0.1], [0.5, 0.24, 0.1]])
    acc = ACCDetector(alpha=0.9, v_min=0.01, threshold=0.5)
    pair = acc.score_pair(old, new, executed_steps=2, previous_smoothed_score=0.02)
    print("ACC example (raw=1.666667, smoothed=0.184667):")
    print(json.dumps(pair.to_dict(), indent=2))

    # Synthetic score series only to illustrate the calibration API.
    # In reality, collect these from held-out SUCCESSFUL simulator episodes.
    fake_success_scores = [[0.5, 1.0 + index / 100, 0.8] for index in range(40)]
    calibration = calibrate_threshold(fake_success_scores, alpha=0.025)
    print("Synthetic calibration threshold:", calibration.threshold)

    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    llmd.save(output / "toy_llmd.npz")
    restored = LLMDDetector.load(output / "toy_llmd.npz", expected_metadata={"checkpoint": "toy-only"})
    print("Reloaded LLMD score:", restored.score([[3, 12], [8, 9]]).score)


if __name__ == "__main__":
    main()

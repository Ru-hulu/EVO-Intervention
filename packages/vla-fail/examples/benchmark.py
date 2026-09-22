"""Measure numerical kernels only; no VLA forward, MuJoCo, HTTP or GPU transfer."""

import argparse
import json
import time

import numpy as np
from vla_fail import ACCDetector
from vla_fail import LLMDDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.horizon < 2 or args.features < 1 or args.samples < 2 or args.iterations < 1:
        parser.error("Need horizon>=2, features>=1, samples>=2 and iterations>=1")
    rng = np.random.default_rng(42)
    training = rng.normal(size=(args.samples, args.horizon, args.features))
    start = time.perf_counter()
    llmd = LLMDDetector().fit(training)
    fit_seconds = time.perf_counter() - start
    query = rng.normal(size=(args.horizon, args.features))
    positions = rng.normal(size=(args.horizon, 3))
    acc = ACCDetector()
    for _ in range(3):
        llmd.score(query)
    llmd_ms, acc_ms = [], []
    for _ in range(args.iterations):
        start = time.perf_counter()
        llmd.score(query)
        llmd_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        acc.score_pair(positions, positions, args.horizon // 2)
        acc_ms.append((time.perf_counter() - start) * 1000)
    result = {
        "input": vars(args),
        "fit_seconds": fit_seconds,
        "factor_mib": args.horizon * args.features**2 * 8 / 2**20,
        "llmd_median_ms": float(np.median(llmd_ms)),
        "llmd_p95_ms": float(np.percentile(llmd_ms, 95)),
        "acc_median_ms": float(np.median(acc_ms)),
        "acc_p95_ms": float(np.percentile(acc_ms, 95)),
        "scope": "Synthetic CPU scoring only; excludes model inference and integration.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Audit every indexed video: size/hash, timing, and first/middle/last RGB decode.

This samples decoded frames in every file; it is not an exhaustive frame decode.
The result is a read-only audit and does not modify the curated dataset.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def check_video(job):
    import av

    path, view, fps, total = job
    result = {"path": str(path)}
    try:
        if path.stat().st_size != view["video_size_bytes"]:
            raise ValueError("file size mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(2**20), b""):
                digest.update(block)
        if digest.hexdigest() != view["video_sha256"]:
            raise ValueError("sha256 mismatch")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            result.update(frames=stream.frames, fps=float(stream.average_rate),
                          width=stream.width, height=stream.height)
            if stream.frames and stream.frames != total:
                raise ValueError(f"frame count {stream.frames} != annotation {total}")
            if abs(float(stream.average_rate) - fps) > 0.05:
                raise ValueError(f"fps {float(stream.average_rate)} != annotation {fps}")
            for index in [0, total // 2, total - 1]:
                timestamp = int(index / fps / stream.time_base) + (stream.start_time or 0)
                container.seek(timestamp, stream=stream, backward=True, any_frame=False)
                found = False
                for frame in container.decode(video=0):
                    source_index = round((frame.pts - (stream.start_time or 0)) * stream.time_base * fps)
                    if source_index >= index:
                        if source_index != index:
                            raise ValueError(f"seek skipped requested frame {index}, got {source_index}")
                        frame.to_ndarray(format="rgb24")
                        found = True
                        break
                if not found:
                    raise ValueError(f"cannot decode requested frame {index}")
        result["status"] = "ok"
    except Exception as error:
        result.update(status="error", reason=f"{type(error).__name__}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    jobs = []
    for file in sorted(root.glob("*/*/metadata.jsonl")):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            jobs += [(file.parent / view["video_path"], view, row["fps"], row["total_frames"])
                     for view in row["video_views"].values()]
    results = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.with_suffix(".jsonl").open("w") as log:
        futures = [pool.submit(check_video, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            log.write(json.dumps(result) + "\n")
            log.flush()
            if len(results) % 100 == 0:
                print(f"checked {len(results)}/{len(jobs)}", flush=True)
    failures = [r for r in results if r["status"] != "ok"]
    report = dict(total=len(jobs), passed=len(jobs) - len(failures), failures=failures,
                  coverage="every indexed view, full-file sha256, first/middle/last RGB decode")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(total=report["total"], passed=report["passed"], failed=len(failures))))


if __name__ == "__main__":
    main()

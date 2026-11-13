"""Download the original Qwen3.5-4B snapshot directly on A100, with official hashes.

This only downloads and verifies model files. It does not modify dataset labels
or start label generation, training, or evaluation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import threading
import time
from urllib.parse import quote

import requests

from experiments.zeroshot_voc_20261113.rebuild_official_weights import (
    atomic_json, fetch_one, safe_error, verify,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).with_name("qwen35-download-catalog.json"))
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    spec = json.loads(args.catalog.read_text())
    assert spec["repo_id"] == "Qwen/Qwen3.5-4B"
    assert len(spec["revision"]) == 40
    destination = Path(spec["local_dir"])
    assert destination == Path("/cache/cvpzhr/models/Qwen3.5-4B")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / "download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    destination.mkdir(parents=True, exist_ok=True)
    for row in spec["files"]:
        path = PurePosixPath(row["path"])
        assert not path.is_absolute() and ".." not in path.parts
        assert isinstance(row["size"], int) and row["size"] >= 0
        assert row.get("lfs") or row.get("blob_id")
    state = dict(status="downloading", pid=os.getpid(), started_at=time.time(),
                 repo_id=spec["repo_id"], revision=spec["revision"], destination=str(destination),
                 total_files=len(spec["files"]), total_bytes=sum(x["size"] for x in spec["files"]),
                 files={}, workers=args.workers, model_bytes_transit_local_computer=False,
                 label_generation_started=False, evaluation_started=False,
                 catalog_sha256=hashlib.sha256(args.catalog.read_bytes()).hexdigest())
    mutex, last_write = threading.Lock(), [0.0]

    def update(name, force=False, **fields):
        with mutex:
            state["files"].setdefault(name, {}).update(fields)
            now = time.time()
            if force or now - last_write[0] >= 3:
                state.update(updated_at=now,
                             downloaded_bytes=sum(x.get("bytes", 0) for x in state["files"].values()),
                             verified_files=sum(x.get("status") == "verified" for x in state["files"].values()))
                atomic_json(args.state_dir / "status.json", state)
                last_write[0] = now

    def download(row):
        path = destination / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if digest := verify(path, row):
            update(row["path"], True, status="verified", bytes=row["size"], sha256=digest, reused=True)
            return
        if path.exists():
            raise ValueError("Existing file differs from official snapshot: " + row["path"])
        part = path.with_name(path.name + ".download-part")
        filename = quote(row["path"], safe="/")
        repo, revision = spec["repo_id"], spec["revision"]
        urls = [f"https://hf-mirror.com/{repo}/resolve/{revision}/{filename}"]
        if not row.get("lfs"):
            urls.insert(0, f"https://hf-mirror.com/api/resolve-cache/models/{repo}/{revision}/{filename}")
        # The author's ModelScope mirror is accepted only with the pinned HF hash.
        modelscope = f"https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={filename}"
        # Prefer the faster local CDN for large blobs; the pinned HF SHA256
        # still determines whether the downloaded bytes are accepted.
        if row.get("lfs"):
            urls.insert(0, modelscope)
        else:
            urls.append(modelscope)
        with requests.Session() as session:
            session.trust_env = False
            for attempt in range(1, 9):
                update(row["path"], True, status="downloading", attempt=attempt,
                       bytes=part.stat().st_size if part.exists() else 0, expected_bytes=row["size"])
                try:
                    fetch_one(session, urls, part, row["size"], lambda n: update(row["path"], bytes=n))
                    update(row["path"], True, status="verifying", bytes=row["size"])
                    digest = verify(part, row)
                    if not digest:
                        part.replace(part.with_name(part.name + f".invalid-{time.time_ns()}"))
                        raise ValueError("Official hash mismatch: " + row["path"])
                    part.replace(path)
                    update(row["path"], True, status="verified", bytes=row["size"], sha256=digest)
                    return
                except Exception as error:
                    update(row["path"], True, status="failed" if attempt == 8 else "retrying", error=safe_error(error))
                    if attempt == 8:
                        raise
                    time.sleep(min(5 * attempt, 30))

    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(download, row): row for row in sorted(spec["files"], key=lambda x: -x["size"])}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(dict(file=futures[future]["path"], error=safe_error(error)))
    if not failures:
        index = json.loads((destination / "model.safetensors.index.json").read_text())
        declared = {x["path"] for x in spec["files"]}
        for shard in set(index["weight_map"].values()):
            if shard not in declared or state["files"].get(shard, {}).get("status") != "verified":
                failures.append(dict(file=shard, error="Missing verified weight shard"))
        config = json.loads((destination / "config.json").read_text())
        if config.get("model_type") != "qwen3_5":
            failures.append(dict(file="config.json", error="Unexpected model type"))
    state.update(status="failed" if failures else "completed", failures=failures, finished_at=time.time(),
                 all_official_hashes_verified=not failures)
    atomic_json(args.state_dir / "status.json", state)
    if not failures:
        atomic_json(args.state_dir / "verification.json", state)
    print(json.dumps({k: v for k, v in state.items() if k != "files"}), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

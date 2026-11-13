"""Restore pinned public checkpoints on the remote host, without launching evaluation.

The catalog comes from the official Hugging Face model API at each pinned
revision. A mirror may supply download redirects, but every resulting file is
checked against the official LFS SHA256 or Git blob SHA1 before being published.
No account token is sent to the mirror. All model bytes stay on the remote host.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time
from urllib.parse import quote, urlsplit, urlunsplit

import requests


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def safe_error(error):
    # Signed CDN query strings are unnecessary in logs and status reports.
    return re.sub(r"https?://[^\s'\"]+", lambda m: urlunsplit(
        (*urlsplit(m.group(0))[:3], "", "")), str(error))[:700]


def verify(path, row, extra_sha256=None):
    if not path.is_file() or path.stat().st_size != row["size"]:
        return None
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {row['size']}\0".encode())
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            sha256.update(block)
            blob.update(block)
    digest = sha256.hexdigest()
    lfs = row.get("lfs")
    if lfs:
        if digest != lfs["sha256"]:
            return None
    elif blob.hexdigest() != row["blob_id"]:
        return None
    if extra_sha256 and digest != extra_sha256:
        return None
    return digest


def fetch_one(session, urls, part, size, progress):
    """Resume an incomplete file; HTTP 200 at a nonzero offset restarts safely."""
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError("Partial file exceeds the official size")
    if offset == size:
        return
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    last_error = None
    for url in urls:
        try:
            with session.get(url, headers=headers, timeout=(12, 45), stream=True) as r:
                r.raise_for_status()
                if r.status_code == 206:
                    content_range = r.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected HTTP Content-Range")
                elif r.status_code == 200:
                    offset = 0
                else:
                    raise ValueError(f"Unexpected download status {r.status_code}")
                mode = "ab" if offset else "wb"
                with part.open(mode) as sink:
                    for block in r.iter_content(1024 * 1024):
                        if not block:
                            continue
                        offset += len(block)
                        if offset > size:
                            raise ValueError("Download exceeds the official size")
                        sink.write(block)
                        progress(offset)
                    sink.flush()
                if offset != size:
                    raise IOError(f"Incomplete response: {offset}/{size}")
                return
        except Exception as error:
            last_error = error
            # A response may already have extended the partial file. Let the
            # next outer attempt recalculate its exact offset before resuming.
            if part.exists() and part.stat().st_size != offset:
                break
            if part.exists() and part.stat().st_size:
                break
    raise last_error or IOError("No download URL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (args.state_dir / "download.lock").open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    catalog = json.loads(args.catalog.read_text())
    provenance = json.loads((Path(__file__).parent / "model-provenance.json").read_text())
    assert set(catalog) == set(provenance) and len(catalog) == 9
    rows = []
    for model, spec in catalog.items():
        assert spec["revision"] == provenance[model]["revision"]
        assert spec["local_dir"] == provenance[model]["path"]
        for row in spec["files"]:
            relative = PurePosixPath(row["path"])
            assert not relative.is_absolute() and ".." not in relative.parts
            assert isinstance(row["size"], int) and row["size"] >= 0
            assert row.get("lfs") or row.get("blob_id")
            rows.append((model, spec, row))
    state = {
        "status": "downloading", "pid": os.getpid(), "started_at": time.time(),
        "official_revisions": {k: v["revision"] for k, v in catalog.items()},
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "endpoint": args.endpoint, "model_bytes_transit_local_computer": False,
        "evaluation_started": False, "workers": args.workers,
        "total_files": len(rows), "total_bytes": sum(r["size"] for _, _, r in rows),
        "files": {},
    }
    mutex = threading.Lock()
    last_write = [0.0]

    def update(key, force=False, **values):
        with mutex:
            entry = state["files"].setdefault(key, {})
            entry.update(values)
            now = time.time()
            if force or now - last_write[0] >= 3:
                state["updated_at"] = now
                state["verified_files"] = sum(f.get("status") == "verified" for f in state["files"].values())
                state["downloaded_bytes"] = sum(f.get("bytes", 0) for f in state["files"].values())
                atomic_json(args.state_dir / "download-status.json", state)
                last_write[0] = now

    def download(item):
        model, spec, row = item
        key = model + "/" + row["path"]
        dest = Path(spec["local_dir"]) / row["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        extra = provenance[model]["small_file_sha256"].get(row["path"])
        digest = verify(dest, row, extra)
        if digest:
            update(key, True, status="verified", bytes=row["size"], sha256=digest, reused=True)
            return
        if dest.exists():
            raise RuntimeError("Existing final file failed verification: " + str(dest))
        part = dest.with_name(dest.name + ".download-part")
        path = quote(row["path"], safe="/")
        stem = f"{args.endpoint.rstrip('/')}/{spec['repo_id']}"
        urls = [f"{stem}/resolve/{spec['revision']}/{path}"]
        if not row.get("lfs"):
            urls += [f"{stem}/raw/{spec['revision']}/{path}"]
        session = requests.Session()
        session.trust_env = False
        session.headers["User-Agent"] = "zeroshot-official-checkpoint-restore/1"
        for attempt in range(1, 13):
            update(key, True, status="downloading", attempt=attempt,
                   bytes=part.stat().st_size if part.exists() else 0, expected_bytes=row["size"])
            try:
                fetch_one(session, urls, part, row["size"], lambda n: update(key, bytes=n))
                update(key, True, status="verifying", bytes=row["size"])
                digest = verify(part, row, extra)
                if not digest:
                    quarantine = part.with_name(part.name + f".invalid-{int(time.time())}")
                    part.replace(quarantine)
                    raise ValueError("Downloaded file failed official hash verification")
                part.replace(dest)
                update(key, True, status="verified", sha256=digest, bytes=row["size"])
                return
            except Exception as error:
                update(key, True, status="retrying", error=safe_error(error))
                if attempt == 12:
                    update(key, True, status="failed")
                    raise
                time.sleep(min(5 * attempt, 30))
        session.close()

    # Start the weight shards immediately; smaller configuration files follow.
    rows.sort(key=lambda item: -item[2]["size"])
    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(download, row): row for row in rows}
        for future in as_completed(futures):
            model, _, row = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append({"model": model, "file": row["path"], "error": safe_error(error)})
    state["status"] = "failed" if failures else "completed"
    state["failures"] = failures
    state["finished_at"] = time.time()
    state["models"] = {
        model: {"complete": all(state["files"].get(model + "/" + f["path"], {}).get("status") == "verified"
                                for f in spec["files"]), "revision": spec["revision"], "path": spec["local_dir"]}
        for model, spec in catalog.items()
    }
    atomic_json(args.state_dir / "download-status.json", state)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

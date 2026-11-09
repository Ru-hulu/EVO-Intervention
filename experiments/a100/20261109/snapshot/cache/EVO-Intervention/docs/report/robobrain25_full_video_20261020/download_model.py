"""Download the pinned public checkpoint directly on the remote machine."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit
import fcntl
import hashlib
import json
import os
import re
import shutil
import threading
import time

import requests

RUN = Path("/root/autodl-tmp/EvalProgress/outputs/full_video_20261020")
TARGET = Path("/root/models/RoboBrain2.5-4B")
META_PATH = Path("/root/autodl-tmp/EvalProgress/_setup_logs/robobrain25_4b_latency10_20261016/model_metadata.json")
META = json.loads(META_PATH.read_text())
CHUNK = 32 * 1024**2
WORKERS = 32
LOCK = threading.Lock()
STARTED = time.time()
STATE = {"phase": "starting", "model": META["id"], "revision": META["sha"],
         "model_path": str(TARGET), "expected_bytes": sum(x["size"] for x in META["siblings"]),
         "downloaded_bytes": 0, "completed_chunks": [], "workers": WORKERS,
         "transport": "HF metadata via platform network acceleration; weight bytes directly from official HF CDN to this server"}


def save(**updates):
    with LOCK:
        STATE.update(updates)
        STATE["elapsed_s"] = round(time.time() - STARTED, 2)
        STATE["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = RUN / "download_status.json.tmp"
        tmp.write_text(json.dumps(STATE, indent=2))
        tmp.replace(RUN / "download_status.json")


def verify(path, item):
    if path.stat().st_size != item["size"]:
        raise RuntimeError("size mismatch: " + item["rfilename"])
    expected = (item.get("lfs") or {}).get("sha256")
    if expected:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for data in iter(lambda: f.read(8 * 1024**2), b""):
                h.update(data)
        if h.hexdigest() != expected:
            raise RuntimeError("SHA256 mismatch: " + item["rfilename"])
    return expected


RUN.mkdir(parents=True, exist_ok=True)
TARGET.mkdir(parents=True, exist_ok=True)
lockfile = (RUN / "download_model.lock").open("a")
fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
try:
    expected = STATE["expected_bytes"]
    existing_bytes = sum(p.stat().st_blocks * 512 for p in TARGET.rglob("*") if p.is_file())
    if shutil.disk_usage(TARGET).free < max(0, expected - existing_bytes) + 2 * 1024**3:
        raise RuntimeError("Insufficient model-disk space; need model size plus 2 GiB reserve")
    session = requests.Session()
    large = next(x for x in META["siblings"] if x["rfilename"] == "model.safetensors")
    small_bytes = 0
    for item in META["siblings"]:
        if item == large:
            continue
        url = f"https://huggingface.co/{META['id']}/resolve/{META['sha']}/{item['rfilename']}"
        final = TARGET / item["rfilename"]
        if not final.exists():
            for attempt in range(5):
                try:
                    response = session.get(url, timeout=(10, 45))
                    response.raise_for_status()
                    final.parent.mkdir(parents=True, exist_ok=True)
                    part = final.with_name(final.name + ".part")
                    part.write_bytes(response.content)
                    verify(part, item)
                    part.replace(final)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2)
        verify(final, item)
        small_bytes += item["size"]
    final = TARGET / large["rfilename"]
    if not final.exists():
        url = f"https://huggingface.co/{META['id']}/resolve/{META['sha']}/model.safetensors?download=true"
        response = session.get(url, stream=True, allow_redirects=False, timeout=(10, 30))
        response.raise_for_status()
        signed = response.headers["Location"]
        response.close()
        host = urlsplit(signed).hostname or ""
        if urlsplit(signed).scheme != "https" or not host.endswith((".hf.co", ".huggingface.co", ".amazonaws.com")):
            raise RuntimeError("Unexpected weight download host")
        part = TARGET / "model.safetensors.part"
        fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, large["size"])
        count = (large["size"] + CHUNK - 1) // CHUNK
        journal = RUN / "download_chunks.json"
        completed = set(json.loads(journal.read_text()) if journal.exists() else [])
        save(phase="downloading", downloaded_bytes=small_bytes, total_chunks=count)
        print("DOWNLOAD", host, count, "chunks", flush=True)

        def chunk(i):
            if i in completed:
                return i
            begin, end = i * CHUNK, min((i + 1) * CHUNK, large["size"]) - 1
            direct = requests.Session()
            direct.trust_env = False
            for attempt in range(6):
                try:
                    with direct.get(signed, headers={"Range": f"bytes={begin}-{end}"},
                                    stream=True, timeout=(12, 60)) as reply:
                        if reply.status_code != 206:
                            raise RuntimeError("CDN HTTP " + str(reply.status_code))
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", reply.headers.get("Content-Range", ""))
                        if not match or tuple(map(int, match.groups())) != (begin, end, large["size"]):
                            raise RuntimeError("Range mismatch")
                        offset = begin
                        for data in reply.iter_content(1024 * 1024):
                            if offset + len(data) > end + 1:
                                raise RuntimeError("Range overflow")
                            written = os.pwrite(fd, data, offset)
                            if written != len(data):
                                raise RuntimeError("Short disk write")
                            offset += written
                        if offset != end + 1:
                            raise RuntimeError("Incomplete range")
                    return i
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(2 + attempt)

        try:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                jobs = [pool.submit(chunk, i) for i in range(count)]
                for future in as_completed(jobs):
                    completed.add(future.result())
                    journal_tmp = journal.with_suffix(".json.tmp")
                    journal_tmp.write_text(json.dumps(sorted(completed)))
                    journal_tmp.replace(journal)
                    downloaded = sum(min(CHUNK, large["size"] - i * CHUNK) for i in completed)
                    save(downloaded_bytes=small_bytes + downloaded, completed_chunks=sorted(completed))
            os.fsync(fd)
        finally:
            os.close(fd)
        save(phase="verifying")
        digest = verify(part, large)
        part.replace(final)
    else:
        digest = verify(final, large)
    save(phase="complete", downloaded_bytes=expected, verified_sha256=digest,
         free_bytes_after=shutil.disk_usage(TARGET).free)
    (RUN / "model_metadata.json").write_text(json.dumps(META, ensure_ascii=False, indent=2))
    print("COMPLETE", TARGET, digest, flush=True)
except Exception as exc:
    # Do not include signed CDN URLs or proxy credentials in logs.
    save(phase="failed", error_type=type(exc).__name__)
    print("FAILED", type(exc).__name__, flush=True)
    raise SystemExit(1)

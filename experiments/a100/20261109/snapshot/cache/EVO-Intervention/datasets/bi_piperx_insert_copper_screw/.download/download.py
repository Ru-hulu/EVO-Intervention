import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import ssl
import time
import urllib.request
import urllib.error
import threading

ROOT = Path("/cache/EVO-Intervention/datasets/bi_piperx_insert_copper_screw")
CONTROL = ROOT / ".download"
REPO = "MINT-SJTU/RW-RL-Dataset"
TASK = "bi_piperx_insert_copper_screw"
REVISION = "e18a20263435640c96f5f995f769a621302fd5f2"
ENDPOINT = "https://hf-mirror.com"
EXPECTED_MANIFEST = "9481d9f6f85673b0009c4916ec5522c5677d8eeaabee28dddaf00018fb524573"
CONTEXT = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
START = time.time()
STATE = {"phase": "listing", "started_at": START, "repository": REPO, "revision": REVISION,
         "destination": str(ROOT), "endpoint": ENDPOINT, "workers": 16,
         "completed_files": 0, "completed_bytes": 0}
def emit(event, **fields):
    print(json.dumps({"time": time.time(), "event": event, **fields}), flush=True)
def status(**fields):
    STATE.update(fields)
    STATE.update(updated_at=time.time(), elapsed_seconds=round(time.time()-START, 2))
    temp = CONTROL / "status.tmp"
    temp.write_text(json.dumps(STATE, indent=2))
    temp.replace(CONTROL / "status.json")
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST = 0.0
PAUSE_UNTIL = 0.0
def open_url(url):
    global NEXT_REQUEST, PAUSE_UNTIL
    while True:
        with REQUEST_LOCK:
            now = time.monotonic()
            wait = max(NEXT_REQUEST, PAUSE_UNTIL) - now
            if wait <= 0:
                NEXT_REQUEST = now + 0.55
                break
        time.sleep(min(wait, 1.0))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 RW-RL dataset downloader"})
    try:
        return urllib.request.urlopen(req, context=CONTEXT, timeout=90)
    except urllib.error.HTTPError as error:
        if error.code == 429:
            retry_after = error.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else 60.0
            with REQUEST_LOCK:
                PAUSE_UNTIL = max(PAUSE_UNTIL, time.monotonic() + wait)
            emit("rate_limit_pause", seconds=wait)
        raise
def digest(path, item):
    h = hashlib.sha256() if item["algorithm"] == "sha256" else hashlib.sha1()
    if item["algorithm"] == "git-sha1":
        h.update(("blob " + str(item["size"]) + "\0").encode())
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()
def download(item):
    rel = item["path"][len(TASK)+1:]
    if ".." in PurePosixPath(rel).parts or rel.startswith("/"):
        raise ValueError("Invalid dataset path")
    final = ROOT / rel
    if final.is_file() and final.stat().st_size == item["size"] and digest(final, item) == item["hash"]:
        return item
    final.parent.mkdir(parents=True, exist_ok=True)
    part = CONTROL / "parts" / (hashlib.sha256(rel.encode()).hexdigest() + ".part")
    mode = "resolve" if item["algorithm"] == "sha256" else "raw"
    base = f"{ENDPOINT}/datasets/{REPO}/{mode}/{REVISION}/{item['path']}"
    for attempt in range(12):
        url = base + "?download=true"
        if attempt:
            url += "&retry=" + str(time.time_ns())
        try:
            h = hashlib.sha256() if item["algorithm"] == "sha256" else hashlib.sha1()
            if item["algorithm"] == "git-sha1":
                h.update(("blob " + str(item["size"]) + "\0").encode())
            count = 0
            with open_url(url) as response, part.open("wb") as output:
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    h.update(chunk)
                    count += len(chunk)
            if count != item["size"] or h.hexdigest() != item["hash"]:
                raise ValueError(f"Size/hash mismatch: received {count}, expected {item['size']}")
            part.replace(final)
            return item
        except Exception as error:
            emit("retry", path=rel, attempt=attempt+1, error=str(error)[:250])
            if attempt == 11:
                raise
            time.sleep(min(2**attempt, 16))
def main():
    status()
    if (CONTROL / "inventory.json").is_file():
        items = json.loads((CONTROL / "inventory.json").read_text())["files"]
    else:
        listing_url = f"{ENDPOINT}/api/datasets/{REPO}/tree/{REVISION}/{TASK}?recursive=true&expand=false&limit=1000"
        files = []
        while listing_url:
            with open_url(listing_url) as response:
                entries = json.load(response)
                link = response.headers.get("Link", "")
            files.extend(item for item in entries if item["type"] == "file")
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            listing_url = match.group(1).replace("https://huggingface.co/", ENDPOINT+"/") if match else None
        items = [{"path": f["path"], "size": f["size"],
                  "hash": f.get("lfs", {}).get("oid", f["oid"]),
                  "algorithm": "sha256" if f.get("lfs") else "git-sha1"}
                 for f in sorted(files, key=lambda f: f["path"])]
    manifest_digest = hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest_digest != EXPECTED_MANIFEST:
        raise RuntimeError("Mirror inventory differs from the previously inspected official inventory")
    (CONTROL / "inventory.json").write_text(json.dumps(
        {"repo": REPO, "revision": REVISION, "task": TASK, "manifest_sha256": manifest_digest, "files": items}, indent=2))
    total_bytes = sum(f["size"] for f in items)
    assert len(items) == 2236 and total_bytes == 21508733725
    status(phase="downloading", total_files=len(items), total_bytes=total_bytes)
    emit("download_start", files=len(items), bytes=total_bytes, workers=16)
    # Check metadata and an actual video early, then process the remaining files.
    items.sort(key=lambda f: (0 if "/meta/" in f["path"] else
                             1 if "/videos/" in f["path"] and f["path"].endswith("file-000.mp4") else 2,
                             f["path"]))
    failures = []
    last_report = 0
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(download, item): item for item in items}
        for future in cf.as_completed(futures):
            item = futures[future]
            try:
                future.result()
                STATE["completed_files"] += 1
                STATE["completed_bytes"] += item["size"]
            except Exception as error:
                failures.append({"path": item["path"], "error": str(error)[:300]})
            if time.time()-last_report >= 10:
                status(failed_files=len(failures))
                emit("progress", completed_files=STATE["completed_files"],
                     completed_bytes=STATE["completed_bytes"], failed_files=len(failures))
                last_report = time.time()
    if failures:
        (CONTROL / "failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError(f"{len(failures)} files failed; see failures.json")
    status(phase="verifying", completed_files=len(items), completed_bytes=total_bytes)
    emit("verification_start")
    def verify(item):
        path = ROOT / item["path"][len(TASK)+1:]
        if path.stat().st_size != item["size"] or digest(path, item) != item["hash"]:
            raise RuntimeError("Final checksum mismatch: " + item["path"])
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, items))
    import pyarrow.parquet as pq
    info = json.loads((ROOT/"meta/info.json").read_text())
    episodes = pq.read_table(ROOT/"meta/episodes/chunk-000/file-000.parquet")
    data_files = list((ROOT/"data").rglob("*.parquet"))
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in data_files)
    video_counts = {path.name: len(list(path.rglob("*.mp4"))) for path in (ROOT/"videos").iterdir() if path.is_dir()}
    assert episodes.num_rows == info["total_episodes"] == len(data_files) == 558
    assert rows == info["total_frames"] == 1255729
    assert len(video_counts) == 3 and all(n == 558 for n in video_counts.values())
    report = {"complete": True, "revision": REVISION, "endpoint": ENDPOINT,
              "verified_files": len(items), "verified_bytes": total_bytes,
              "manifest_sha256": manifest_digest, "episodes": episodes.num_rows,
              "frames": rows, "fps": info["fps"], "video_counts": video_counts,
              "completed_at": time.time()}
    (CONTROL/"verification.json").write_text(json.dumps(report, indent=2))
    status(phase="completed", verification=report)
    emit("completed", **report)
if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status(phase="failed", error=str(error)[:500])
        emit("failed", error=str(error)[:500])
        raise

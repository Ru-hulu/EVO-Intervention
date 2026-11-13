"""Fetch small files from authors' ModelScope mirrors on the remote machine.

Accept only bytes matching the original pinned Hugging Face Git blob and any
additional provenance SHA256. The main weight downloader can reuse these blobs.
This never downloads weight shards or launches inference.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests

from rebuild_official_weights import atomic_json, safe_error, verify


# These are the same model authors' official mirrors, verified via their API.
OFFICIAL_MIRRORS = {
    "robobrain25-4b": "BAAI/RoboBrain2.5-4B",
    "robobrain25-8b": "BAAI/RoboBrain2.5-8B-NV",
    "vlac-2b": "InternRobotics/VLAC",
    "vlac-8b": "InternRobotics/VLAC-8b",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--blob-dir", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    args.blob_dir.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for model, mirror in OFFICIAL_MIRRORS.items():
        spec = catalog[model]
        assert spec["repo_id"] == mirror
        for row in spec["files"]:
            if row.get("lfs"):
                continue
            tasks.setdefault(row["blob_id"], (model, mirror, spec, row))

    def fetch(item):
        model, mirror, spec, row = item
        dest = args.blob_dir / row["blob_id"]
        extra = spec["small_file_sha256"].get(row["path"])
        result = {"model": model, "file": row["path"], "blob_id": row["blob_id"],
                  "hf_revision": spec["revision"], "model_bytes_transit_local_computer": False}
        if digest := verify(dest, row, extra):
            return dict(result, status="verified", sha256=digest, reused=True)
        url = (f"https://modelscope.cn/api/v1/models/{mirror}/repo?Revision=master"
               f"&FilePath={quote(row['path'], safe='')}")
        result["source"] = url
        session = requests.Session()
        session.trust_env = False
        try:
            with session.get(url, stream=True, timeout=(10, 30)) as response:
                response.raise_for_status()
                part = dest.with_suffix(".part")
                size = 0
                with part.open("wb") as target:
                    for block in response.iter_content(1024 * 1024):
                        size += len(block)
                        if size > row["size"]:
                            raise ValueError("Mirror file differs from pinned HF size")
                        target.write(block)
                digest = verify(part, row, extra)
                if not digest:
                    raise ValueError("Mirror file differs from pinned HF hash")
                part.replace(dest)
                return dict(result, status="verified", sha256=digest)
        except Exception as error:
            return dict(result, status="unavailable", error=safe_error(error))
        finally:
            session.close()

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(fetch, tasks.values()))
    atomic_json(args.blob_dir / "manifest.json", {
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "model_bytes_transit_local_computer": False,
        "accepted_only_matching_pinned_hf_blobs": True,
        "files": results,
    })
    print(json.dumps({"verified": sum(r["status"] == "verified" for r in results),
                      "unavailable": sum(r["status"] != "verified" for r in results)}))


if __name__ == "__main__":
    main()

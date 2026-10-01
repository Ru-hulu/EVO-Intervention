"""在远程运行并行计算，将完成状态写入文件，SSH 断开不影响计算。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

root = Path(__file__).resolve().parent
status = root / "compute_status.json"
status.write_text(json.dumps({"status": "running", "pid": os.getpid()}))
try:
    manifest = json.loads((root / "manifest.json").read_text())
    for item in manifest["items"]:
        video = Path("/root/video") / f"{item['video_id']}.mp4"
        assert video.stat().st_size == item["video_bytes"], f"Upload incomplete: {video}"
        assert (Path("/root/lable") / f"{item['video_id']}.JSON").exists()
    subprocess.run([
        sys.executable, str(root / "video_to_delta_parallel.py"),
        "--file-list", str(root / "video_files.txt"), "--workers", "8",
    ], check=True)
    result = {"status": "complete", "videos": len(manifest["items"])}
except Exception:
    result = {"status": "failed", "error": traceback.format_exc()}
status.write_text(json.dumps(result))
print(result, flush=True)

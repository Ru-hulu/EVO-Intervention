"""后台等待上传，启动远程计算，完成后自动取回并核对 delta 文件。"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from scripts.delta_to_progress import frame_descriptions

SSH = ["ssh", "-o", "ControlPath=/tmp/codex-qwen-%C", "-o", "ConnectTimeout=10", "-p", "50101", "root@connect.westc.seetacloud.com"]
RSH = "ssh -o ControlPath=/tmp/codex-qwen-%C -o ServerAliveInterval=30 -p 50101"
PYTHON = "/root/autodl-tmp/qwen3-vl-ft/.venv/bin/python"
REMOTE = "/root/autodl-tmp/qwen3-vl-ft/jobs/delta_20261001"
manifest = json.loads((ROOT / "manifest.json").read_text())
output = Path(manifest["output_dir"])
label_dir = Path(manifest["items"][0]["label"]).parent
video_dir = Path(manifest["items"][0]["video"]).parent


def status(value):
    (ROOT / "pipeline_status.json").write_text(json.dumps(value, ensure_ascii=False, indent=2))
    print(json.dumps(value, ensure_ascii=False), flush=True)


def remote_python(code):
    command = PYTHON + " -c " + shlex.quote(code)
    return subprocess.run([*SSH, command], capture_output=True, text=True, check=True).stdout.strip()


def verify_file(label):
    stem = label.stem
    path = output / f"{stem}-delta.txt"
    rows = path.read_text(encoding="utf-8").splitlines()
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames,r_frame_rate",
        "-of", "json", str(video_dir / f"{stem}.mp4"),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert len(rows) == int(stream["nb_frames"]), stem
    assert stream["r_frame_rate"] == "30/1", stem
    descriptions = frame_descriptions(json.loads(label.read_text()), len(rows))
    for row, expected in zip(rows, descriptions, strict=True):
        value, description = row.split(maxsplit=1)
        assert math.isfinite(float(value)) and float(value) >= 0, stem
        assert description == expected, (stem, description, expected)
    assert float(rows[0].split()[0]) == 0, stem
    return {"file": stem, "frames": len(rows)}


try:
    status({"status": "waiting_for_upload"})
    while not (ROOT / "upload_status.json").exists():
        time.sleep(30)
    upload = json.loads((ROOT / "upload_status.json").read_text())
    if upload["status"] != "complete":
        raise RuntimeError(upload)
    launch = f"""from pathlib import Path
import subprocess, json
root = Path({REMOTE!r})
with (root / 'compute.log').open('w') as log:
    process = subprocess.Popen([{PYTHON!r}, str(root / 'remote_job.py')], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({{'pid': process.pid}}))
"""
    started = json.loads(remote_python(launch))
    status({"status": "computing", "remote_pid": started["pid"], "videos": manifest["pending_count"]})
    while True:
        time.sleep(60)
        try:
            raw = remote_python(f"from pathlib import Path; print(Path({REMOTE + '/compute_status.json'!r}).read_text())")
            remote_status = json.loads(raw)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        if remote_status["status"] == "failed":
            raise RuntimeError(remote_status)
        if remote_status["status"] == "complete":
            break
    status({"status": "downloading", "videos": manifest["pending_count"]})
    file_list = ROOT / "delta_files.txt"
    file_list.write_text("".join(f"{item['video_id']}-delta.txt\n" for item in manifest["items"]))
    subprocess.run([
        "rsync", "-a", "--partial", "--files-from=" + str(file_list), "-e", RSH,
        "root@connect.westc.seetacloud.com:/root/pt/", str(output) + "/",
    ], check=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        verified = list(executor.map(verify_file, sorted(label_dir.glob("*.JSON"))))
    (ROOT / "verification.json").write_text(json.dumps(verified, indent=2))
    status({"status": "complete", "files": len(verified), "frames": sum(item["frames"] for item in verified), "output_dir": str(output)})
except Exception:
    status({"status": "failed", "error": traceback.format_exc()})

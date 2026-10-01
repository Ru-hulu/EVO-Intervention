import json
from pathlib import Path
import subprocess

folder = Path(__file__).resolve().parent
manifest = json.loads((folder / "manifest.json").read_text())
ssh = ["ssh", "-o", "ControlPath=/tmp/codex-qwen-%C", "-o", "ServerAliveInterval=30", "-p", "50101", "root@connect.westc.seetacloud.com"]
rsh = "ssh -o ControlPath=/tmp/codex-qwen-%C -o ServerAliveInterval=30 -p 50101"
try:
    subprocess.run([*ssh, "mkdir -p /root/lable /root/video /root/pt /root/pt/backups; test -e /root/pt/backups/lable_before_20261001.tar.gz || tar -czf /root/pt/backups/lable_before_20261001.tar.gz -C /root lable"], check=True)
    label_dir = Path(manifest["items"][0]["label"]).parent
    video_dir = Path(manifest["items"][0]["video"]).parent
    subprocess.run(["rsync", "-a", "--partial", "--include=*.JSON", "--exclude=*", "-e", rsh, str(label_dir) + "/", "root@connect.westc.seetacloud.com:/root/lable/"], check=True)
    subprocess.run(["rsync", "-a", "--partial", "--files-from=" + str(folder / "video_files.txt"), "-e", rsh, str(video_dir) + "/", "root@connect.westc.seetacloud.com:/root/video/"], check=True)
    status = {"status": "complete", "videos": len(manifest["items"])}
except Exception as error:
    status = {"status": "failed", "error": str(error)}
(folder / "upload_status.json").write_text(json.dumps(status))
print(status, flush=True)

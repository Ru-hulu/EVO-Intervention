"""delta 下载校验通过后，完成此前请求的全部回滚 episode 可视化。"""

import json
from pathlib import Path
import subprocess
import time
import traceback

folder = Path(__file__).resolve().parent
project = folder.parents[1]
manifest = json.loads((folder / "manifest.json").read_text())
status_path = folder / "rollback_plot_status.json"
status_path.write_text(json.dumps({"status": "waiting_for_delta"}))
try:
    while True:
        path = folder / "pipeline_status.json"
        if path.exists():
            status = json.loads(path.read_text())
            if status["status"] == "failed":
                raise RuntimeError(status)
            if status["status"] == "complete":
                break
        time.sleep(60)
    output = project / "outputs/all_rollback_episodes_20261001"
    subprocess.run([
        str(Path.home() / ".local/bin/uv"), "run", "--no-project", "--with", "matplotlib",
        "--python", "/opt/homebrew/bin/python3", "python", str(project / "scripts/plot_episode_progress.py"),
        "--delta-dir", manifest["output_dir"], "--label-dir", str(Path(manifest["items"][0]["label"]).parent),
        "--output-dir", str(output), "--rollback-only",
    ], check=True, cwd=project)
    status = {"status": "complete", "image": str(output / "rollback_progress.png")}
except Exception:
    status = {"status": "failed", "error": traceback.format_exc()}
status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
print(status, flush=True)

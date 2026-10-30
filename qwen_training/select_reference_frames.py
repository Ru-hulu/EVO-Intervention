"""本地专家轨迹选帧窗口；只在用户点击保存后导出五张原始帧。

python -m qwen_training.select_reference_frames --video /path/file-000.mp4 \
    --seconds 23 --output-dir /path/expert_references/file-000_first23s
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from urllib.parse import urlparse
import uuid


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Selection:
    def __init__(self, video, seconds, output):
        self.video, self.output = video.resolve(), output.resolve()
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(self.video),
        ], capture_output=True, text=True, check=True)
        self.stream = json.loads(probe.stdout)["streams"][0]
        self.fps = Fraction(self.stream["avg_frame_rate"])
        self.seconds = min(seconds, float(self.stream["duration"]))
        self.count = min(math.ceil(self.seconds * self.fps), int(self.stream["nb_frames"]))
        if self.count < 5:
            raise ValueError("选择范围至少需要五帧")
        self.output.mkdir(parents=True, exist_ok=True)
        signature = f"{self.video}:{self.video.stat().st_size}:{self.video.stat().st_mtime_ns}:{self.count}"
        self.source_key = hashlib.sha256(signature.encode()).hexdigest()[:16]
        self.cache = self.output / ".preview_cache" / self.source_key
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.token = uuid.uuid4().hex
        self.draft = self.output / ".selection_draft.json"
        self.saved = self.output / "selection.json"
        self.chosen, self.saved_chosen = [], []
        for path in (self.saved, self.draft):
            if path.exists():
                data = json.loads(path.read_text())
                if data.get("source_key") == self.source_key:
                    self.chosen = self.validate(data["frame_indices"], exact=False)
                    if path == self.saved:
                        self.saved_chosen = self.chosen.copy()

    def prepare_previews(self):
        if len(list(self.cache.glob("*.jpg"))) != self.count:
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(self.video), "-map", "0:v:0", "-frames:v", str(self.count),
                "-vf", "scale=640:-2", "-fps_mode", "passthrough", "-q:v", "3",
                "-start_number", "0", str(self.cache / "%06d.jpg"),
            ], check=True)

    def validate(self, frames, exact):
        if (not isinstance(frames, list) or len(frames) > 5 or (exact and len(frames) != 5)
                or any(type(f) is not int or not 0 <= f < self.count for f in frames)
                or len(set(frames)) != len(frames)):
            raise ValueError("请选择范围内的五个不同帧" if exact else "帧号无效")
        return sorted(frames)

    def state(self):
        return {
            "video": str(self.video), "fps": float(self.fps), "frame_count": self.count,
            "seconds": self.seconds, "width": self.stream["width"], "height": self.stream["height"],
            "selected": self.chosen, "saved": self.saved_chosen,
            "output_dir": str(self.output), "token": self.token,
        }

    def save_draft(self, frames):
        self.chosen = self.validate(frames, exact=False)
        write_json(self.draft, {"source_key": self.source_key, "frame_indices": self.chosen})

    def export(self, frames):
        frames = self.validate(frames, exact=True)
        temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=self.output))
        try:
            expression = "+".join(f"eq(n\\,{frame})" for frame in frames)
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(self.video), "-map", "0:v:0", "-vf", f"select={expression}",
                "-frames:v", "5", "-fps_mode", "passthrough", "-start_number", "1",
                str(temporary / "reference-%03d.png"),
            ], capture_output=True, text=True, check=True)
            if len(list(temporary.glob("reference-*.png"))) != 5:
                raise RuntimeError("未能从原视频导出五张图片")
            version = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            destination = self.output / version
            temporary.rename(destination)
            images = [str(destination / f"reference-{i:03d}.png") for i in range(1, 6)]
            result = {
                "source_key": self.source_key, "source_video": str(self.video),
                "fps": float(self.fps), "frame_index_base": 0,
                "range_seconds": [0, self.seconds], "range_end_exclusive": True,
                "frame_indices": frames, "reference_images": images,
                "frames": [
                    {"reference_id": f"reference-{i + 1:03d}", "frame_index": frame,
                     "time_seconds": float(Fraction(frame, 1) / self.fps), "image": images[i]}
                    for i, frame in enumerate(frames)
                ],
                "selected_by": "user", "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(destination / "selection.json", result)
            write_json(self.saved, result)
            self.save_draft(frames)
            self.saved_chosen = frames.copy()
            return {"selection_file": str(self.saved), "reference_images": images}
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def make_handler(selection):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, content, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store" if mime.startswith("application/json") else "private, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, data, status=200):
            self.send(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send(Path(__file__).with_name("reference_selector.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                with selection.lock:
                    self.json(selection.state())
            elif match := re.fullmatch(r"/frames/(\d+)\.jpg", path):
                frame = int(match[1])
                if 0 <= frame < selection.count:
                    self.send((selection.cache / f"{frame:06d}.jpg").read_bytes(), "image/jpeg")
                else:
                    self.json({"error": "帧号超出范围"}, 404)
            else:
                self.json({"error": "Not found"}, 404)

        def do_POST(self):
            if self.headers.get("X-Selection-Token") != selection.token:
                self.json({"error": "请刷新选帧窗口后重试"}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 4096:
                    raise ValueError("请求长度无效")
                data = json.loads(self.rfile.read(size))
                with selection.lock:
                    if self.path == "/api/draft":
                        selection.save_draft(data["frames"])
                        self.json({"ok": True})
                    elif self.path == "/api/save":
                        self.json(selection.export(data["frames"]))
                    else:
                        self.json({"error": "Not found"}, 404)
            except (ValueError, KeyError) as error:
                self.json({"error": str(error)}, 400)
            except Exception as error:
                self.json({"error": "保存失败：" + str(error)}, 500)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=23)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    selection = Selection(args.video, args.seconds, args.output_dir)
    selection.prepare_previews()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(selection))
    url = f"http://127.0.0.1:{server.server_port}/"
    write_json(selection.output / ".selector_runtime.json", {"url": url, "source_video": str(selection.video)})
    print(url, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

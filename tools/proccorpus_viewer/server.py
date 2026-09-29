#!/usr/bin/env python3
import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote
from urllib.parse import urlparse

EPISODE_RE = re.compile(r"episode_\d{6}")
CAMERAS = {
    "global": "observation.images.global_image",
    "right": "observation.images.right_image",
    "wrist": "observation.images.wrist_image",
}


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_segments(rows):
    segments = []
    for row in rows:
        label = row.get("sub_task") or "unlabeled"
        frame = int(row["frame_id"])
        if not segments or segments[-1]["label"] != label:
            segments.append({"start": frame, "end": frame + 1, "label": label})
        else:
            segments[-1]["end"] = frame + 1
    return segments


class App(ThreadingHTTPServer):
    def __init__(self, address, dataset: Path, media: Path, static_dir: Path):
        self.dataset = dataset.resolve()
        self.media = media.resolve()
        self.static_dir = static_dir.resolve()
        self.episodes = []
        for path in sorted(self.dataset.glob("episode_*.sub_task.jsonl")):
            if (self.dataset / f"{path.name.removesuffix('.sub_task.jsonl')}.annotation.json").is_file():
                continue
            rows = load_rows(path)
            if not rows:
                continue
            self.episodes.append(
                {
                    "id": path.name.removesuffix(".sub_task.jsonl"),
                    "frames": len(rows),
                    "segments": len(make_segments(rows)),
                }
            )
        for path in sorted(self.dataset.glob("episode_*.annotation.json")):
            annotation = json.loads(path.read_text(encoding="utf-8"))
            self.episodes.append(
                {
                    "id": path.name.removesuffix(".annotation.json"),
                    "frames": annotation["frames"],
                    "segments": len(annotation["segments"]),
                }
            )
        self.episodes.sort(key=lambda episode: episode["id"])
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: App

    def send_json(self, value, status=HTTPStatus.OK):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler requires this method name.
        path = unquote(urlparse(self.path).path)
        if path == "/api/episodes":
            self.send_json({"episodes": self.server.episodes})
            return
        if path.startswith("/api/episodes/"):
            episode_id = path.rsplit("/", 1)[-1]
            if not EPISODE_RE.fullmatch(episode_id):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            annotation_path = self.server.dataset / f"{episode_id}.annotation.json"
            if annotation_path.is_file():
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                video_path = self.server.media / f"{episode_id}.mp4"
                self.send_json(
                    {
                        "id": episode_id,
                        "task": annotation["task"],
                        "frames": annotation["frames"],
                        "fps": annotation["fps"],
                        "videos": {"global": f"/media/{episode_id}/global.mp4"} if video_path.is_file() else {},
                        "segments": annotation["segments"],
                        "overall_notes": annotation.get("overall_notes", ""),
                    }
                )
                return
            file_path = self.server.dataset / f"{episode_id}.sub_task.jsonl"
            if not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            rows = load_rows(file_path)
            self.send_json(
                {
                    "id": episode_id,
                    "task": rows[0].get("task", ""),
                    "frames": len(rows),
                    "fps": 30,
                    "videos": {
                        camera: f"/media/{episode_id}/{camera}.mp4"
                        for camera, directory in CAMERAS.items()
                        if (self.server.media / directory / f"{episode_id}.mp4").is_file()
                    },
                    "segments": make_segments(rows),
                }
            )
            return
        if path.startswith("/media/"):
            parts = path.strip("/").split("/")
            if len(parts) != 3 or not EPISODE_RE.fullmatch(parts[1]):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            camera = parts[2].removesuffix(".mp4")
            if parts[2] != f"{camera}.mp4" or camera not in CAMERAS:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            video_path = self.server.media / f"{parts[1]}.mp4"
            if camera != "global" or not video_path.is_file():
                video_path = self.server.media / CAMERAS[camera] / f"{parts[1]}.mp4"
            if not video_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_video(video_path)
            return
        if path in ("/", "/index.html"):
            body = (self.server.static_dir / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def send_video(self, path: Path):
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            elif match.group(2):
                length = int(match.group(2))
                start = max(0, size - length)
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            try:
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Browsers routinely cancel an old range request after seeking.
                pass

    def log_message(self, fmt, *args):
        print(f"{self.client_address[0]} {fmt % args}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/zhaobo/data/zhaobo/proc_examples/sub_task/Table30_open_the_drawer"),
    )
    parser.add_argument(
        "--media",
        type=Path,
        default=Path("/home/zhaobo/data/zhaobo/proc_media/extracted/Table30_open_the_drawer/chunk-000"),
    )
    args = parser.parse_args()
    if not args.dataset.is_dir():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if not args.media.is_dir():
        raise SystemExit(f"Media not found: {args.media}")
    app = App((args.bind, args.port), args.dataset, args.media, Path(__file__).parent)
    print(f"Serving {len(app.episodes)} episodes at http://{args.bind}:{args.port}", flush=True)
    app.serve_forever()


if __name__ == "__main__":
    main()

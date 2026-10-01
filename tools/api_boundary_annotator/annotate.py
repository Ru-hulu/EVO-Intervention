#!/usr/bin/env python3
"""Annotate subtask boundaries in a robot manipulation MP4 with a VLM behind an OpenAI-compatible API.

Follows the two annotation stages of ProcVLM (arXiv:2605.08774, appendix A.4):

1. Planning: from the video (and the task instruction, if known) the model decomposes the episode into an
   ordered list of atomic subtasks, each written in one of a fixed set of action patterns.
2. Temporal localization: given that plan and the same video, the model marks the frame where each
   subtask starts and the frame where it completes.

The video is sent as uniformly sampled frames, each labelled with its index in the original video, so
the returned frame numbers index the MP4 directly. Any server that implements ``/chat/completions`` with
image inputs works (OpenAI, vLLM, SGLang, DashScope, ...).

The endpoint, model and sampling settings are read from ``.env`` next to this script (see ``.env_example``);
environment variables and command-line flags override it.

python tools/api_boundary_annotator/annotate.py episode.mp4 --task "open the drawer" --output episode.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import cv2

PATTERNS = """\
- Grasp [object]: pick up an object while the gripper holds nothing, before placing or lifting it.
- Place [object] onto/into [location]: put down a held object at a location.
- Push [object] [direction]: move an object by pushing it, without grasping it.
- Tilt the gripper: change the gripper's orientation.
- Hang [object] on/above [location]: suspend a held object from a location.
- Press [object]: press a button or surface.
- Open [object]: open a drawer, door, lid or similar.
- Close [object]: close a drawer, door, lid or similar.
- Rotate [object]: turn an object, knob or handle."""

PLAN_PROMPT = """\
You are shown frames of a video in which robot arm(s) perform a manipulation task.
{task_line}
Analyse what the robot actually does in the video and decompose the episode into the ordered sequence of
atomic subtasks it performs. Every subtask must follow exactly one of these patterns:
{patterns}

Rules:
- Name concrete objects and locations as they appear in the video; do not list the robot arm as an object.
- Grasp and Place are separate subtasks. Use Push only when the object is moved without being grasped.
- List subtasks in the order they happen. If an attempt fails and is retried, list the subtask once.
- If two arms act, name the arm in the subtask, e.g. "Grasp the cup with the left arm".
- Output only a numbered list, one subtask per line, starting from 1. No other text."""

LOCALIZE_PROMPT = """\
You are shown frames of a video in which robot arm(s) perform a manipulation task. Each frame is preceded
by its frame index in the original video, which has {num_frames} frames (indices 0 to {last_frame}).
{task_line}
The episode was decomposed into these subtasks, in order:
{plan}

For each subtask, mark the frame where it starts and the frame where it is completed.
- Use the frame indices shown with the images.
- Look for changes in the end-effector pose and in object motion to find candidate start and complete
  frames. The start may be slightly early and the completion slightly late, so the whole action is covered.
- Subtasks happen in the listed order: a subtask must not start before the previous one starts.
- If a subtask is started but not finished, set complete_frame to null.
- If a subtask does not appear at all, set both frames to null and notes to "not present".
- If a subtask is attempted several times, mark the final successful attempt.

Return only this JSON object, with one entry per subtask in the same order and with the same names:
{{
  "task": "<the task>",
  "subtasks": [
    {{"id": 1, "name": "<subtask name>", "start_frame": <int or null>, "complete_frame": <int or null>,
      "notes": "<optional, at most 60 words>"}}
  ],
  "overall_notes": "<optional, at most 30 words; say so if the overall task was not completed>"
}}"""


def decode_video(video: Path) -> tuple[list, float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    # The container's frame count can be wrong, so the frames are counted while decoding.
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    if not decoded:
        raise ValueError(f"No frames decoded from {video}")

    return decoded, fps


def sample_frames(
    decoded: list, video: Path, max_side: int, step: int, min_frames: int, max_frames: int
) -> list[tuple[int, bytes]]:
    """Match ProcVLM's task-clip sampling: prefer a time step, with a minimum and a cap."""
    total = len(decoded)
    step = max(1, min(total // min_frames, step))
    if len(range(0, total, step)) > max_frames:
        step = (total + max_frames - 1) // max_frames
    frames = []
    for index in range(0, total, step):
        image = decoded[index]
        scale = max_side / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError(f"Cannot encode frame {index} of {video}")
        frames.append((index, jpeg.tobytes()))
    return frames


class ChatClient:
    """Minimal client for an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str | None, timeout: float, max_tokens: int):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], attempts: int = 3) -> str:
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens}
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(attempts):
            request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:500]
                # Rate limits and server errors are transient; anything else is a bad request.
                if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                    raise RuntimeError(f"{self.url} returned {error.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == attempts - 1:
                    raise
            time.sleep(2**attempt)
        raise AssertionError("unreachable")


def frame_content(frames: list[tuple[int, bytes]], fps: float) -> list[dict]:
    """Each image preceded by its original frame index, so the model answers in video frame indices."""
    content = []
    for index, jpeg in frames:
        stamp = f" (t={index / fps:.2f}s)" if fps else ""
        content.append({"type": "text", "text": f"Frame {index}{stamp}:"})
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def parse_plan(text: str) -> list[str]:
    plan = [match.group(1).strip() for match in re.finditer(r"^\s*\d+\s*[.)]\s*(.+?)\s*$", text, re.MULTILINE)]
    if not plan:
        raise ValueError(f"No numbered subtasks in the planning answer:\n{text}")
    return plan


def parse_json(text: str) -> dict:
    """The JSON object in an answer, which models often wrap in a code fence or a sentence."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The answer contains no JSON object")
    return json.loads(text[start : end + 1])


def check_localization(answer: dict, plan: list[str], num_frames: int) -> list[dict]:
    """The answer's subtasks, or a ValueError naming what violates the requested format."""
    subtasks = answer.get("subtasks")
    if not isinstance(subtasks, list) or len(subtasks) != len(plan):
        raise ValueError(f"Expected {len(plan)} subtasks, got {len(subtasks) if isinstance(subtasks, list) else 0}")
    checked, previous_start = [], -1
    for number, (subtask, name) in enumerate(zip(subtasks, plan, strict=True), 1):
        frames = {}
        for key in ("start_frame", "complete_frame"):
            value = subtask.get(key)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"Subtask {number} {key} must be an integer or null, got {value!r}")
            if value is not None and not 0 <= value < num_frames:
                raise ValueError(f"Subtask {number} {key} {value} is outside frames 0 to {num_frames - 1}")
            frames[key] = value
        start, complete = frames["start_frame"], frames["complete_frame"]
        if start is None and complete is not None:
            raise ValueError(f"Subtask {number} has a complete_frame but no start_frame")
        if start is not None and complete is not None and complete < start:
            raise ValueError(f"Subtask {number} completes at frame {complete}, before it starts at {start}")
        if start is not None:
            if start < previous_start:
                raise ValueError(f"Subtask {number} starts at frame {start}, before subtask {number - 1}")
            previous_start = start
        checked.append(
            {
                "id": number,
                "name": name,
                "start_frame": start,
                "complete_frame": complete,
                "notes": subtask.get("notes") or "",
            }
        )
    return checked


def make_segments(subtasks: list[dict], total: int, overall_notes: str) -> list[dict]:
    """Expand completion boundaries to dense task stages, as in ProcVLM's flattening step."""
    segments = []
    last_end = 0
    last_task = None
    completed = sorted(
        (sub for sub in subtasks if sub["start_frame"] is not None and sub["complete_frame"] is not None),
        key=lambda sub: sub["complete_frame"],
    )
    for sub in completed:
        end = min(total - 1, sub["complete_frame"])
        if end > last_end:
            segments.append({"start": last_end, "end": end, "label": sub["name"], "notes": sub["notes"]})
            last_end = end
        last_task = sub["name"]

    failed_count = sum(sub["start_frame"] is None or sub["complete_frame"] is None for sub in subtasks)
    done = "not completed" not in overall_notes.lower() and failed_count < len(subtasks) / 2
    if last_end < total:
        tail_label = "done" if done else (last_task or "not done")
        if segments and segments[-1]["label"] == tail_label:
            segments[-1]["end"] = total
        else:
            segments.append({"start": last_end, "end": total, "label": tail_label, "notes": ""})
    return segments


def annotate(
    video: str | Path,
    client: ChatClient,
    task: str | None = None,
    num_frames: int = 128,
    max_side: int = 448,
    retries: int = 2,
) -> dict:
    """Ordered subtasks of ``video`` with the frames where each starts and completes."""
    video = Path(video)
    decoded, fps = decode_video(video)
    total = len(decoded)
    plan_frames = sample_frames(decoded, video, max_side, max(1, int(fps)), 16, num_frames)
    images = frame_content(plan_frames, fps)
    task_line = f"The task instruction is: {task}" if task else "The task instruction is not given."

    prompt = PLAN_PROMPT.format(task_line=task_line, patterns=PATTERNS)
    plan = parse_plan(client.complete([{"role": "user", "content": [*images, {"type": "text", "text": prompt}]}]))

    prompt = LOCALIZE_PROMPT.format(
        num_frames=total,
        last_frame=total - 1,
        task_line=task_line,
        plan="\n".join(f"{number}. {name}" for number, name in enumerate(plan, 1)),
    )
    localize_frames = sample_frames(
        decoded, video, max_side, max(1, int(fps) // 2), max(16, min(64, 8 * len(plan))), num_frames
    )
    images = frame_content(localize_frames, fps)
    messages = [{"role": "user", "content": [*images, {"type": "text", "text": prompt}]}]
    for attempt in range(retries + 1):
        answer_text = client.complete(messages)
        try:
            answer = parse_json(answer_text)
            subtasks = check_localization(answer, plan, total)
            break
        except ValueError as error:  # json.JSONDecodeError is a ValueError too.
            if attempt == retries:
                raise ValueError(f"Localization answer still invalid after {retries} retries: {error}") from None
            # Show the model its answer and what is wrong with it, then ask again.
            messages += [
                {"role": "assistant", "content": answer_text},
                {"role": "user", "content": f"That answer is invalid: {error}. Return the corrected JSON only."},
            ]

    overall_notes = answer.get("overall_notes") or ""
    segments = make_segments(subtasks, total, overall_notes)

    return {
        "id": video.stem,
        "frames": total,
        "fps": fps,
        "video": {
            "path": str(video),
            "num_frames": total,
            "fps": fps,
            "sampled_frames": [index for index, _ in localize_frames],
            "planning_sampled_frames": [index for index, _ in plan_frames],
        },
        "model": client.model,
        "task": task or answer.get("task", ""),
        "plan": plan,
        "segments": segments,
        "subtasks": subtasks,
        "overall_notes": overall_notes,
    }


def read_env_file(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines of a dotenv file; blank lines, ``#`` comments and surrounding quotes are dropped."""
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def main():
    # The env file is located first, since its values become the defaults of the other flags.
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    env_args, _ = env_parser.parse_known_args()
    env_file = read_env_file(env_args.env_file) if env_args.env_file.is_file() else {}

    def setting(key: str, default: str | None = None) -> str | None:
        # Environment variables override the env file; an empty value counts as unset.
        return os.environ.get(key) or env_file.get(key) or default

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[env_parser]
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--task", help="Task instruction; if omitted, the model infers the task from the video.")
    parser.add_argument("--base-url", default=setting("ANNOTATOR_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=setting("ANNOTATOR_MODEL"))
    parser.add_argument("--api-key", default=setting("ANNOTATOR_API_KEY"))
    parser.add_argument("--num-frames", type=int, default=setting("ANNOTATOR_NUM_FRAMES", "128"))
    parser.add_argument("--max-side", type=int, default=setting("ANNOTATOR_MAX_SIDE", "448"))
    parser.add_argument("--max-tokens", type=int, default=setting("ANNOTATOR_MAX_TOKENS", "4096"))
    parser.add_argument("--timeout", type=float, default=setting("ANNOTATOR_TIMEOUT", "300"))
    parser.add_argument("--retries", type=int, default=setting("ANNOTATOR_RETRIES", "2"))
    parser.add_argument("--output", type=Path, help="Write the JSON here instead of printing it.")
    args = parser.parse_args()
    if args.num_frames < 1:
        parser.error("--num-frames must be positive")
    if not args.model:
        parser.error(f"no model: set ANNOTATOR_MODEL in {env_args.env_file} (see .env_example) or pass --model")

    client = ChatClient(args.base_url, args.model, args.api_key, args.timeout, args.max_tokens)
    result = annotate(args.video, client, args.task, args.num_frames, args.max_side, args.retries)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

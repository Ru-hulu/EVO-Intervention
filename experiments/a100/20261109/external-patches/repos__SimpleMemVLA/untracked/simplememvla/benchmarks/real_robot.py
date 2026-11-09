"""Real-robot spec supplied after collecting the user's LeRobot dataset."""
import json
import os
from pathlib import Path

from simplememvla.benchmarks import BenchmarkSpec

path = Path(os.environ["SIMPLEMEMVLA_ROBOT_CONFIG"])
values = json.loads(path.read_text())
required = ("num_actions_chunk", "action_dim", "state_dim", "robot_tag", "native_video_fps",
            "camera_labels", "repo_id", "root", "image_keys", "history_image_keys")
missing = [key for key in required if not values.get(key)]
if missing:
    raise ValueError(f"Fill {path} before training: {', '.join(missing)}")
for key in ("num_actions_chunk", "action_dim", "state_dim"):
    if type(values[key]) is not int or values[key] <= 0:
        raise ValueError(f"{key} must be a positive integer")
for key in ("native_video_fps", "history_video_sec", "history_video_fps"):
    if values[key] <= 0:
        raise ValueError(f"{key} must be positive")
if values["name"] != "real_robot":
    raise ValueError("name must be real_robot")
if not Path(values["root"]).is_absolute():
    raise ValueError("root must be an absolute dataset path")
if not set(values["history_image_keys"]).issubset(values["image_keys"]):
    raise ValueError("history_image_keys must be a subset of image_keys")
for key in values["image_keys"]:
    if key.split(".")[-1] not in values["camera_labels"]:
        raise ValueError(f"camera_labels must describe the last segment of {key}")
for key in ("image_keys", "history_image_keys"):
    values[key] = tuple(values[key])
SPEC = BenchmarkSpec(**values)

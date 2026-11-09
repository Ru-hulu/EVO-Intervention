# ruff: noqa: RUF001
"""Normalize current-action intervals without changing progress values or source annotations."""

import re
from collections import Counter

from qwen_training.datasets.proprimem.policies import POLICIES
from qwen_training.datasets.proprimem.prepare import annotation_events
from qwen_training.datasets.rmbench.labels import step_of
from qwen_training.datasets.rmbench.tasks import TASKS


def _pack(spans, start, stop, provenance, unit):
    cursor = start
    names, counts, intervals = [], Counter(), []
    for left, right, action in spans:
        if left != cursor or not left < right <= stop:
            raise ValueError("Current-action intervals must be contiguous, nonempty and in range")
        cursor = right
        stage = None
        instance = None
        if action is not None:
            if action not in names:
                names.append(action)
            stage = names.index(action)
            counts[action] += 1
            instance = counts[action]
        intervals.append(
            dict(
                start_frame=left,
                end_frame=right,
                action=action,
                stage_id=stage,
                instance=instance,
                state="active" if action is not None else "completed",
            )
        )
    if cursor != stop:
        raise ValueError("Current-action intervals do not cover the source range")
    return dict(
        version="current-action-v1",
        frame_range_convention="[start_frame, end_frame)",
        provenance=provenance,
        label_unit=unit,
        stage_vocabulary=names,
        intervals=intervals,
    )


def normalize_current_actions(annotation, task_id):
    """Completed-state boundaries imply action phases, not observed motion onset.

    IDs index this annotation's vocabulary, never a global class space. Repeated
    source segments retain segment-instance semantics, not physical repetition counts.
    """
    if annotation.get("frame_index_base") != 0 or annotation.get("frame_range_convention") != "[start_frame, end_frame)":
        raise ValueError("Require zero-based half-open source-frame intervals")
    total, stages = annotation["total_frames"], annotation["stages"]
    cursor = 0
    for stage in stages:
        if stage["start_frame"] != cursor or not cursor < stage["end_frame"] <= total:
            raise ValueError("Source stage boundaries are not contiguous")
        cursor = stage["end_frame"]
    if cursor != total or len(stages) != annotation["num_stages"]:
        raise ValueError("Source stages do not cover the video")
    if "progress" in annotation:
        raise ValueError("Scripted anomaly stages require a separate current-action policy")
    if annotation.get("annotation_semantics") == "current_action":
        spans = [(s["start_frame"], s["end_frame"], s["state_description"].strip()) for s in stages]
        if any(not action for _, _, action in spans):
            raise ValueError("Current-action labels must be nonempty")
        result = _pack(spans, 0, total, "annotated_current_action", "annotated_action")
        result.update(review_status=annotation.get("label_status"), human_review=annotation.get("human_review"))
        return result
    policy = POLICIES[task_id]
    schema, original, _ = policy
    spans = []
    if schema in {"rmbench", "libero", "cube_place", "cube_swing"}:
        completed = False
        for index, stage in enumerate(stages):
            text = stage["state_description"].strip()
            if schema == "libero" and "全部子任务已完成" in text:
                action = None
                completed = True
            else:
                if completed:
                    raise ValueError("Unexpected action after terminal completed state")
                if schema == "rmbench":
                    text = re.sub(r"（第\d+次）", "", text)
                    action = TASKS[original].steps[step_of(text, index, TASKS[original])]
                else:
                    action = re.sub(r"（第\d+个子任务）", "", text)
                    action = re.sub(
                        r"\s+for the (?:\w+|\d+(?:st|nd|rd|th)) time\b", "", action, flags=re.IGNORECASE
                    ).strip()
                if not action:
                    raise ValueError("Empty current action")
            spans.append((stage["start_frame"], stage["end_frame"], action))
        provenance, unit = "annotated_current_action", "annotated_action"
    else:
        events = annotation_events(annotation, policy)
        cursor = 0
        for action, completion in events:
            # At a completed-state boundary the previous action is already done.
            # [previous completion, this completion) is an inferred action phase.
            if not cursor < completion < total:
                raise ValueError("Completion boundaries cannot define a nonempty action phase")
            spans.append((cursor, completion, action))
            cursor = completion
        if not events:
            raise ValueError("No completion events")
        spans.append((cursor, total, None))
        provenance = "inferred_from_completion_boundaries"
        unit = "source_action_segment" if schema == "source_segments" else "annotated_action"
    return _pack(spans, 0, total, provenance, unit)


def normalize_recovery_actions(episode):
    """Keep every retry; convert the source's inclusive 30-FPS boundaries once."""
    subtasks = episode["subtasks"]
    spans = [(s["30FPSstart_frame"], s["30FPSend_frame"] + 1, s["name"].strip()) for s in subtasks]
    if not spans or any(not name for _, _, name in spans):
        raise ValueError("Recovery requires nonempty named subtasks")
    return _pack(spans, spans[0][0], spans[-1][1], "annotated_current_action", "annotated_action")

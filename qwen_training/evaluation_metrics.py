"""Metrics for generated stage text and progress; no GPU or model dependency."""

import math
import re
import unicodedata
from functools import lru_cache

from qwen_training.prompts import STAGE_NAMES


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).replace("。", ".")


def parse_response(text):
    """Parse canonical stage names; retain invalid/missing fields instead of guessing."""
    text = normalize(text)
    names = {normalize(value): key for key, value in STAGE_NAMES.items()}
    fields = {}
    for key, marker in [("nowid", "正在执行的阶段"), ("finishid", "已完成阶段"), ("restid", "未完成阶段")]:
        matches = re.findall(re.escape(marker) + r":([^;。\.\n]+)", text)
        raw = matches[0] if len(matches) == 1 else None
        if key == "nowid":
            fields[key] = names.get(raw)
        elif raw == "无":
            fields[key] = []
        elif raw:
            values = [names.get(piece) for piece in re.split("[、,]", raw)]
            fields[key] = sorted(values) if all(v is not None for v in values) and len(set(values)) == len(values) else None
        else:
            fields[key] = None
    intent = re.findall(r"任务意图(?:为|:)(.*?)[,，]当前任务状态", text)
    fields["intent"] = intent[0] if len(intent) == 1 else None
    fields["parse_valid"] = all(fields[key] is not None for key in ("nowid", "finishid", "restid", "intent"))
    fields["state_consistent"] = False
    if fields["parse_valid"]:
        ids = [fields["nowid"], *fields["finishid"], *fields["restid"]]
        fields["state_consistent"] = sorted(ids) == [1, 2, 3, 4]
    return fields


@lru_cache(maxsize=4096)
def edit_distance(left, right):
    """Levenshtein distance, with equal prefixes/suffixes removed first."""
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    left, right = left[prefix:], right[prefix:]
    while left and right and left[-1] == right[-1]:
        left, right = left[:-1], right[:-1]
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def quantile(values, weights, fraction):
    ordered = sorted(zip(values, weights))
    threshold = sum(weights) * fraction
    total = 0
    for value, weight in ordered:
        total += weight
        if total >= threshold:
            return value
    return ordered[-1][0]


def measure(rows, mixed=False):
    if not rows:
        return None
    weights = [({"short": 0.3, "context": 0.7}[r["mode"]] if mixed else 1.0) for r in rows]
    total = sum(weights)
    mean = lambda values: sum(w * value for w, value in zip(weights, values)) / total
    errors = [r["predicted_progress"] - r["target"]["progress"] for r in rows]
    if not all(math.isfinite(e) for e in errors):
        raise ValueError("Nonfinite progress prediction")
    absolute = [abs(e) for e in errors]
    parsed = [parse_response(r["response"]) for r in rows]
    confusion = [[0.0] * 5 for _ in range(4)]  # columns 1..4, invalid
    for row, prediction, weight in zip(rows, parsed, weights):
        confusion[row["target"]["nowid"] - 1][(prediction["nowid"] or 5) - 1] += weight
    f1_scores = []
    for stage in range(4):
        tp = confusion[stage][stage]
        fn = sum(confusion[stage]) - tp
        fp = sum(line[stage] for line in confusion) - tp
        f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    result = {
        "generated_responses": len(rows), "weighted_sample_count": total,
        "progress_mae_pp": mean(absolute) * 100,
        "progress_rmse_pp": math.sqrt(mean([e * e for e in errors])) * 100,
        "progress_bias_pp": mean(errors) * 100,
        "progress_median_ae_pp": quantile(absolute, weights, 0.5) * 100,
        "progress_p90_ae_pp": quantile(absolute, weights, 0.9) * 100,
        "progress_p95_ae_pp": quantile(absolute, weights, 0.95) * 100,
        "progress_within_5pp": mean([e <= 0.05 for e in absolute]),
        "progress_within_10pp": mean([e <= 0.1 for e in absolute]),
        "current_stage_accuracy": mean([p["nowid"] == r["target"]["nowid"] for r, p in zip(rows, parsed)]),
        "current_stage_macro_f1": sum(f1_scores) / 4,
        "stage_confusion": confusion,
        "parse_valid_rate": mean([p["parse_valid"] for p in parsed]),
        "state_consistent_rate": mean([p["state_consistent"] for p in parsed]),
        "full_state_accuracy": mean([all(p[k] == (sorted(r["target"][k]) if k != "nowid" else r["target"][k])
                                         for k in ("nowid", "finishid", "restid")) for r, p in zip(rows, parsed)]),
        "intent_exact_accuracy": mean([p["intent"] == normalize(r["target"]["intent"]) for r, p in zip(rows, parsed)]),
        "text_exact_accuracy": mean([normalize(r["response"]) == normalize(r["target_response"]) for r in rows]),
        "previous_stage_baseline_accuracy": mean([r["input_state"]["nowid"] == r["target"]["nowid"] for r in rows]),
    }
    for key in ("finishid", "restid"):
        tp = fp = fn = 0.0
        for row, prediction, weight in zip(rows, parsed, weights):
            expected = set(row["target"][key])
            actual = set(prediction[key] or [])
            tp += weight * len(expected & actual)
            fp += weight * len(actual - expected)
            fn += weight * len(expected - actual)
        result[key + "_exact_accuracy"] = mean([p[key] == sorted(r["target"][key]) for r, p in zip(rows, parsed)])
        result[key + "_micro_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    references = [normalize(r["target_response"]) for r in rows]
    distances = [r["text_edit_distance"] if "text_edit_distance" in r else edit_distance(normalize(r["response"]), ref)
                 for r, ref in zip(rows, references)]
    result["character_error_rate"] = sum(w * d for w, d in zip(weights, distances)) / sum(w * len(s) for w, s in zip(weights, references))
    return result


def summarize(rows):
    """Both prompt modes are evaluated for every sample; mixed scores use 30/70 weights."""
    for row in rows:
        row["text_edit_distance"] = edit_distance(normalize(row["response"]), normalize(row["target_response"]))
    result = {"short": measure([r for r in rows if r["mode"] == "short"]),
              "context": measure([r for r in rows if r["mode"] == "context"]),
              "mixed_30_70": measure(rows, mixed=True)}
    for field in ("dataset", "video_id", "sample_type"):
        result["by_" + field] = {}
        # video IDs are dataset-qualified to avoid mixing distinct file-010 videos.
        key = lambda r: r["dataset"] + "/" + r[field] if field == "video_id" else r[field]
        for value in sorted({key(r) for r in rows}):
            selected = [r for r in rows if key(r) == value]
            result["by_" + field][value] = {mode: measure([r for r in selected if r["mode"] == mode])
                                          for mode in ("short", "context")}
    result["by_transition"] = {}
    for previous, current in sorted({(r["input_state"]["nowid"], r["target"]["nowid"]) for r in rows}):
        selected = [r for r in rows if (r["input_state"]["nowid"], r["target"]["nowid"]) == (previous, current)]
        result["by_transition"][f"{previous}->{current}"] = {
            mode: measure([r for r in selected if r["mode"] == mode]) for mode in ("short", "context")}
    return result

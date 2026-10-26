"""生成结果的离线指标；不调用模型、不修复预测、不用CE loss推算准确率。"""

from collections import Counter, defaultdict
import json
import re
import unicodedata


def mean(values):
    return sum(values) / len(values) if values else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def score_m1(text, sample):
    frames, target = sample["inputs"]["frames"], sample["target"]
    score = {"json_valid": False, "format_valid": False, "predicted_valid": None,
             "validity_correct": False, "exact_match": False, "predicted_interval": None,
             "positive_iou": 0.0 if target["valid"] else None,
             "start_error_frames": None, "end_error_frames": None,
             "start_error_seconds": None, "end_error_seconds": None, "error": None}
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        score["error"] = "invalid_json"
        return score
    score["json_valid"] = True
    if not isinstance(value, dict) or type(value.get("valid")) is not int or value["valid"] not in (0, 1):
        score["error"] = "invalid_validity_field"
        return score
    score["predicted_valid"] = value["valid"]
    score["validity_correct"] = value["valid"] == target["valid"]
    keys = {"valid", "start_index", "end_index"} if value["valid"] else {"valid"}
    if set(value) != keys:
        score["error"] = "invalid_schema"
        return score
    if value["valid"]:
        start, end = value["start_index"], value["end_index"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(frames):
            score["error"] = "invalid_queue_boundaries"
            return score
        # 末边界是最后一张被选帧的原帧号+1，不能映射到下一张稀疏观测。
        a, b = frames[start]["frame_index"], frames[end - 1]["frame_index"] + 1
        score["predicted_interval"] = [a, b]
        if target["valid"]:
            left, right = target["start_index"], target["end_index"]
            ta, tb = frames[left]["frame_index"], frames[right - 1]["frame_index"] + 1
            overlap = max(0, min(b, tb) - max(a, ta))
            score["positive_iou"] = overlap / (b - a + tb - ta - overlap)
            score["start_error_frames"], score["end_error_frames"] = abs(a - ta), abs(b - tb)
            # 时间戳差用于秒误差；两个右边界各+1/fps，差值相消。
            score["start_error_seconds"] = abs(frames[start]["time_seconds"] - frames[left]["time_seconds"])
            score["end_error_seconds"] = abs(frames[end - 1]["time_seconds"] - frames[right - 1]["time_seconds"])
    score["format_valid"] = True
    score["exact_match"] = value == target
    return score


def summarize_m1(rows):
    scores = [row["scores"] for row in rows]
    n = len(rows)
    positives = [row for row in rows if row["target"]["valid"] == 1]
    tp = sum(row["target"]["valid"] == 1 and row["scores"]["predicted_valid"] == 1 for row in rows)
    fp = sum(row["target"]["valid"] == 0 and row["scores"]["predicted_valid"] == 1 for row in rows)
    confusion = {str(truth): {str(pred): 0 for pred in (0, 1, "unparseable")} for truth in (0, 1)}
    for row in rows:
        pred = row["scores"]["predicted_valid"]
        confusion[str(row["target"]["valid"])][str(pred) if pred is not None else "unparseable"] += 1
    precision, recall = ratio(tp, tp + fp), ratio(tp, len(positives))
    errors = [score for score in scores if score["start_error_frames"] is not None]
    return {
        "samples": n, "positive_samples": len(positives), "negative_samples": n - len(positives),
        "json_valid_rate": mean([s["json_valid"] for s in scores]),
        "format_valid_rate": mean([s["format_valid"] for s in scores]),
        "validity_accuracy": mean([s["validity_correct"] for s in scores]),
        "validity_precision": precision, "validity_recall": recall,
        "validity_f1": ratio(2 * tp, 2 * tp + fp + len(positives) - tp),
        "validity_confusion": confusion, "exact_match_rate": mean([s["exact_match"] for s in scores]),
        "positive_iou_mean": mean([row["scores"]["positive_iou"] for row in positives]),
        "positive_iou_50_recall": mean([row["scores"]["positive_iou"] >= 0.5 for row in positives]),
        "boundary_evaluated_samples": len(errors),
        **{key + "_mae": mean([s[key] for s in errors]) for key in
           ("start_error_frames", "end_error_frames", "start_error_seconds", "end_error_seconds")},
        "truncated_rate": mean([row.get("truncated", False) for row in rows]),
    }


def normalize_text(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).strip()


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def state_clauses(text):
    # 保留子句内部的对象、顺序和数量；仅按分号、句号及换行分割。
    return [clause for part in re.split(r"[;；。\n]+", text)
            if (clause := normalize_text(part).strip(".;"))]


def chinese_integer(text):
    if text.isdecimal():
        return int(text)
    digits = dict(zip("零〇一二三四五六七八九两", (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2)))
    total, current = 0, 0
    for char in text:
        if char in "十百千":
            total += (current or 1) * {"十": 10, "百": 100, "千": 1000}[char]
            current = 0
        else:
            current = current * 10 + digits[char]
    return total + current


def count_mentions(text):
    # 次数只识别显式“数字+次/遍”；按键编号、层号不冒充次数。
    return [chinese_integer(value) for value in
            re.findall(r"([0-9]+|[零〇一二三四五六七八九十百千两]+)(?:次|遍)", normalize_text(text))]


def score_m2(text, target):
    prediction, truth = normalize_text(text), normalize_text(target)
    predicted_clauses, target_clauses = state_clauses(text), state_clauses(target)
    ps, ts = set(predicted_clauses), set(target_clauses)
    counts, target_counts = count_mentions(text), count_mentions(target)
    return {
        "raw_exact": text == target, "normalized_exact": prediction == truth,
        "edit_distance": edit_distance(prediction, truth), "reference_characters": len(truth),
        "clause_sequence_exact": predicted_clauses == target_clauses, "clause_set_exact": ps == ts,
        "clause_tp": len(ps & ts), "clause_fp": len(ps - ts), "clause_fn": len(ts - ps),
        "clause_reordering_only": (Counter(predicted_clauses) == Counter(target_clauses)
                                  and predicted_clauses != target_clauses),
        "target_counts": target_counts, "predicted_counts": counts,
        "count_sequence_exact": counts == target_counts if target_counts else None,
        "count_absolute_errors": ([abs(a - b) for a, b in zip(counts, target_counts)]
                                  if target_counts and len(counts) == len(target_counts) else None),
    }


def summarize_m2(rows):
    scores = [row["scores"] for row in rows]
    counts = [s for s in scores if s["count_sequence_exact"] is not None]
    comparable = [s for s in counts if s["count_absolute_errors"] is not None]
    errors = [error for s in comparable for error in s["count_absolute_errors"]]
    tp, fp, fn = (sum(s["clause_" + key] for s in scores) for key in ("tp", "fp", "fn"))
    return {
        "prefixes": len(rows),
        "raw_exact_rate": mean([s["raw_exact"] for s in scores]),
        "normalized_exact_rate": mean([s["normalized_exact"] for s in scores]),
        "character_error_rate": ratio(sum(s["edit_distance"] for s in scores),
                                      sum(s["reference_characters"] for s in scores)),
        "clause_sequence_exact_rate": mean([s["clause_sequence_exact"] for s in scores]),
        "clause_set_exact_rate": mean([s["clause_set_exact"] for s in scores]),
        "clause_micro_precision": ratio(tp, tp + fp), "clause_micro_recall": ratio(tp, tp + fn),
        "clause_micro_f1": ratio(2 * tp, 2 * tp + fp + fn),
        "clause_reordering_only_samples": sum(s["clause_reordering_only"] for s in scores),
        "count_applicable_prefixes": len(counts),
        "count_sequence_exact_rate": mean([s["count_sequence_exact"] for s in counts]),
        "count_comparable_prefixes": len(comparable), "count_comparable_mentions": len(errors),
        "count_mae_when_same_number_of_mentions": mean(errors),
        "truncated_rate": mean([row.get("truncated", False) for row in rows]),
    }


def grouped_metrics(rows, summarizer, group_keys):
    result = {"overall": summarizer(rows)}
    for key in group_keys:
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        result["by_" + key] = {key: summarizer(group) for key, group in sorted(groups.items())}
    return result

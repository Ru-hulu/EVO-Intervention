"""生成评测的计分口径：失败计入分母、稀疏边界映射、次数与词面顺序。"""

import json

import pytest
import torch

from mem_lite.evaluation.metrics import (
    count_mentions, score_m1, score_m2, summarize_m1, summarize_m2,
)
from mem_lite.evaluation.runtime import generation_record


def m1_sample(valid=1):
    return {
        "inputs": {"frames": [{"frame_index": index, "time_seconds": index / 30}
                              for index in (0, 3, 9, 21)]},
        "target": {"valid": 1, "start_index": 1, "end_index": 3} if valid else {"valid": 0},
    }


@pytest.mark.parametrize("prediction,error", [
    ('```json\n{"valid":0}\n```', "invalid_json"),
    ('{"valid":true}', "invalid_validity_field"),
    ('{"valid":1,"start_index":0,"end_index":5}', "invalid_queue_boundaries"),
    ('{"valid":1,"start_index":2,"end_index":2}', "invalid_queue_boundaries"),
    ('{"valid":1,"start_index":false,"end_index":2}', "invalid_queue_boundaries"),
    ('{"valid":0,"start_index":0}', "invalid_schema"),
])
def test_malformed_outputs_are_not_repaired(prediction, error):
    result = score_m1(prediction, m1_sample())
    assert result["error"] == error
    assert not result["format_valid"] and not result["exact_match"]
    assert result["positive_iou"] == 0


def test_sparse_right_boundary_is_last_selected_frame_plus_one():
    sample = m1_sample()
    result = score_m1('{"valid":1,"start_index":0,"end_index":3}', sample)
    assert result["predicted_interval"] == [0, 10]  # 不是下一张稀疏帧21。
    assert result["positive_iou"] == pytest.approx(7 / 10)
    assert result["start_error_frames"] == 3 and result["end_error_frames"] == 0
    assert result["start_error_seconds"] == pytest.approx(0.1)
    exact = score_m1(json.dumps(sample["target"]), sample)
    assert exact["exact_match"] and exact["positive_iou"] == 1


def test_missed_positive_iou_is_zero_and_boundary_denominator_is_explicit():
    predictions = ['{"valid":1,"start_index":1,"end_index":3}', '{"valid":0}', 'bad', '{"valid":0}']
    samples = [m1_sample(), m1_sample(), m1_sample(), m1_sample(0)]
    rows = [{"target": sample["target"], "scores": score_m1(prediction, sample)}
            for prediction, sample in zip(predictions, samples)]
    metrics = summarize_m1(rows)
    assert metrics["validity_accuracy"] == 0.5
    assert metrics["positive_iou_mean"] == pytest.approx(1 / 3)
    assert metrics["positive_iou_50_recall"] == pytest.approx(1 / 3)
    assert metrics["boundary_evaluated_samples"] == 1
    assert metrics["end_error_frames_mae"] == 0
    assert metrics["validity_confusion"]["1"]["unparseable"] == 1
    assert summarize_m1(rows[-1:])["positive_iou_mean"] is None


def test_text_order_counts_and_same_meaning_are_distinct_metrics():
    target = "已拿起瓶子；已撒调料3次。"
    reordered = score_m2("已撒调料3次；已拿起瓶子。", target)
    assert reordered["clause_set_exact"] and reordered["clause_reordering_only"]
    assert not reordered["clause_sequence_exact"] and not reordered["normalized_exact"]
    equivalent = score_m2("已拿起瓶子；已撒调料三次。", target)
    assert equivalent["count_sequence_exact"] and not equivalent["normalized_exact"]
    assert count_mentions("按键3已按两次；按钮2已按十二次；重复20遍；一百零二次") == [2, 12, 20, 102]
    assert score_m2(" 已点击３次。\n", "已点击3次。")["normalized_exact"]


def test_missing_counts_count_as_wrong_and_mae_exposes_conditional_denominator():
    rows = [{"scores": score_m2(text, target)} for text, target in (
        ("已点击4次。", "已点击3次。"),
        ("已点击。", "已点击3次。"),
        ("已拿起。", "已拿起。"),
    )]
    metrics = summarize_m2(rows)
    assert metrics["count_applicable_prefixes"] == 2
    assert metrics["count_sequence_exact_rate"] == 0
    assert metrics["count_comparable_prefixes"] == metrics["count_comparable_mentions"] == 1
    assert metrics["count_mae_when_same_number_of_mentions"] == 1
    assert summarize_m2(rows[-1:])["count_sequence_exact_rate"] is None
    # CER按总参考字符数加权，不是逐样本CER的简单平均。
    assert metrics["character_error_rate"] == pytest.approx(
        sum(r["scores"]["edit_distance"] for r in rows) / sum(r["scores"]["reference_characters"] for r in rows))
    assert score_m2("", "已点击3次。")["edit_distance"] == 6


def test_generation_record_drops_padding_after_eos_and_marks_unfinished():
    class Tokenizer:
        eos_token_id = 1

        def decode(self, tokens, **kwargs):
            return str(tokens)

    record = generation_record(torch.tensor([8, 1, 0, 0]), Tokenizer(), 4)
    assert record["generated_token_ids"] == [8, 1] and not record["truncated"]
    assert generation_record(torch.tensor([8, 9, 10, 11]), Tokenizer(), 4)["truncated"]
    other_eos = generation_record(torch.tensor([8, 2, 0, 0]), Tokenizer(), 4, [1, 2])
    assert other_eos["generated_token_ids"] == [8, 2] and not other_eos["truncated"]

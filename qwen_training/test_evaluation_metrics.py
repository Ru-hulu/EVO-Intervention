"""Small, hand-checkable cases for evaluation scoring and result completeness."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from qwen_training.evaluation_metrics import edit_distance, measure, normalize, parse_response
from qwen_training.evaluation_report import load_complete_run, load_stopped_run
from qwen_training.prompts import TASK_INTENT, describe_target


def example(mode="short"):
    target = dict(nowid=2, finishid=[1], restid=[3, 4], intent=TASK_INTENT, progress=0.6)
    return dict(sample_id="test/1", mode=mode, target=target,
                input_state=dict(nowid=1, finishid=[], restid=[2, 3, 4]),
                predicted_progress=0.66, response=describe_target(target), target_response=describe_target(target))


class MetricsTests(unittest.TestCase):
    def test_canonical_and_punctuation(self):
        row = example()
        parsed = parse_response(row["response"])
        self.assertEqual(parsed["nowid"], 2)
        self.assertEqual(parsed["finishid"], [1])
        self.assertEqual(parsed["restid"], [3, 4])
        self.assertEqual(parsed["intent"], TASK_INTENT)
        self.assertTrue(parsed["parse_valid"] and parsed["state_consistent"])
        self.assertEqual(parsed, parse_response("  " + row["response"].replace("；", ";\n")))

    def test_malformed_is_not_silently_correct(self):
        row = example()
        row["response"] = "阶段可能是二"
        result = measure([row])
        self.assertEqual(result["current_stage_accuracy"], 0)
        self.assertEqual(result["full_state_accuracy"], 0)
        self.assertEqual(result["parse_valid_rate"], 0)
        self.assertEqual(result["stage_confusion"][1][4], 1)

    def test_hand_calculated_errors(self):
        row = example()
        result = measure([row])
        self.assertAlmostEqual(result["progress_mae_pp"], 6)
        self.assertAlmostEqual(result["progress_rmse_pp"], 6)
        self.assertAlmostEqual(result["progress_bias_pp"], 6)
        self.assertEqual(result["progress_within_5pp"], 0)
        self.assertEqual(result["progress_within_10pp"], 1)
        self.assertEqual(result["full_state_accuracy"], 1)
        self.assertEqual(result["character_error_rate"], 0)

    def test_weighting(self):
        short, context = example(), example("context")
        short["predicted_progress"] = 0.7
        context["predicted_progress"] = 0.6
        result = measure([short, context], mixed=True)
        self.assertAlmostEqual(result["progress_mae_pp"], 3)
        self.assertAlmostEqual(result["progress_rmse_pp"], (0.3 * 0.1**2)**0.5 * 100)
        self.assertAlmostEqual(result["weighted_sample_count"], 1)

    def test_stage_confusion_and_set_order(self):
        row = example()
        predicted = copy.deepcopy(row["target"])
        predicted.update(nowid=3, restid=[4, 2])
        row["response"] = describe_target(predicted)
        result = measure([row])
        self.assertEqual(result["stage_confusion"][1][2], 1)
        self.assertEqual(result["finishid_exact_accuracy"], 1)
        self.assertEqual(result["restid_exact_accuracy"], 0)
        self.assertAlmostEqual(result["restid_micro_f1"], 0.5)
        predicted.update(nowid=2, restid=[4, 3])
        row["response"] = describe_target(predicted)
        self.assertEqual(measure([row])["full_state_accuracy"], 1)
        self.assertEqual(measure([row])["text_exact_accuracy"], 0)

    def test_edit_distance(self):
        for left, right, expected in [("kitten", "sitting", 3), ("abc", "abc", 0), ("", "ab", 2),
                                      ("当前阶段二", "当前阶段三", 1), ("abcXYZdef", "abcdef", 3)]:
            self.assertEqual(edit_distance(left, right), expected)

    def test_complete_shards_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = dict(world_size=1, sample_ids=["test/1"], modes=["short", "context"], samples=1)
            (root/"manifest.json").write_text(json.dumps(manifest))
            row = example()
            row.update(dataset="test", video_id="file-1", task_index=0, episode=1, frame_indices=[1, 2])
            raw = (json.dumps(row) + "\n").encode()
            (root/"predictions-rank00.jsonl").write_bytes(raw)
            completion = dict(status="complete", rank=0, responses=2, samples=1,
                              predictions_sha256=hashlib.sha256(raw).hexdigest())
            (root/"complete-rank00.json").write_text(json.dumps(completion))
            with self.assertRaises(AssertionError):
                load_complete_run(root)
            second = dict(row, mode="context")
            raw += (json.dumps(second) + "\n").encode()
            completion["predictions_sha256"] = hashlib.sha256(raw).hexdigest()
            (root/"predictions-rank00.jsonl").write_bytes(raw)
            (root/"complete-rank00.json").write_text(json.dumps(completion))
            self.assertEqual(len(load_complete_run(root)[1]), 2)

    def test_stopped_run_keeps_only_complete_pairs_and_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root = parent / 'full'
            root.mkdir()
            (parent/'stop_for_partial_report.json').write_text(json.dumps(dict(
                status='stopped_for_partial_report', remaining_target_pids=[], finished_unix=1)))
            manifest = dict(world_size=1, sample_ids=['test/1', 'test/2'], modes=['short', 'context'], samples=2)
            original = json.dumps(manifest)
            (root/'manifest.json').write_text(original)
            first = example()
            first.update(dataset='test', video_id='file-1', task_index=0, episode=1, frame_indices=[1, 2])
            second = dict(first, mode='context')
            unpaired = dict(first, sample_id='test/2')
            raw = ''.join(json.dumps(row)+'\n' for row in [first, second, unpaired]).encode() + b'{"sample_id":'
            (root/'predictions-rank00.jsonl').write_bytes(raw)
            reported, rows, artifacts = load_stopped_run(root)
            self.assertEqual((reported['samples'], reported['planned_samples']), (1, 2))
            self.assertEqual(reported['unpaired_records_excluded'], 1)
            self.assertEqual(len(rows), 2)
            self.assertEqual(artifacts[0]['discarded_trailing_bytes'], len(b'{"sample_id":'))
            self.assertEqual((root/'manifest.json').read_text(), original)


if __name__ == "__main__":
    unittest.main()

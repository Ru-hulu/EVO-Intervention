import pytest
import evaluate_static as e

def test_absolute_error_and_anchor_drift():
    rows = [
        dict(episode_id="a", kind="original", source_sample_index=0, anchor_sample_index=0, target_pt=.2, predicted_pt=.25),
        dict(episode_id="a", kind="stationary", source_sample_index=0, anchor_sample_index=0, target_pt=.2, predicted_pt=.28),
        dict(episode_id="a", kind="stationary", source_sample_index=0, anchor_sample_index=0, target_pt=.2, predicted_pt=.27),
        dict(episode_id="a", kind="original", source_sample_index=1, anchor_sample_index=1, target_pt=.6, predicted_pt=.65),
    ]
    m=e.summarize(rows)
    assert m["mae_pp"] == pytest.approx(6.25)
    assert m["original_mae_pp"] == pytest.approx(5)
    assert m["stationary_mae_pp"] == pytest.approx(7.5)
    assert m["stationary_prediction_drift_mae_pp"] == pytest.approx(2.5)

def test_no_stationary_rows():
    rows=[dict(episode_id="a",kind="original",source_sample_index=0,anchor_sample_index=0,target_pt=0.,predicted_pt=.1)]
    m=e.summarize(rows)
    assert m["mae_pp"]==pytest.approx(10)
    assert m["stationary_mae_pp"] is None and m["stationary_prediction_drift_mae_pp"] is None

def test_plan_matches_training_stationary_sampler_and_all_views():
    total=0
    for episode in e.PLAN["episodes"]:
        source=e.aug.read_json(e.Path(episode["source_dir"])/"summary.json")
        rows=e.aug.rows_at(episode["source_dir"])
        fps=e.aug.source_fps(source)
        expected=[event for event in e.aug.choose_events(rows,episode["episode_id"],fps,source["source"]["frames"]) if event["kind"]=="stationary"]
        assert episode["events"]==expected
        expanded=list(e.aug.expand(rows,expected,fps))
        for row in expanded:
            anchor=rows[row["anchor_sample_index"]]
            assert row["pt"]==anchor["pt"] and row["images"]==anchor["images"]
        assert len(expanded)==episode["augmented_samples"]
        total+=len(expanded)
    assert total==e.PLAN["augmented_samples"]==11336

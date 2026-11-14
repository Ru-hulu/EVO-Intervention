"""Circle consistency and static mean absolute deviation, without imputation."""
import math
from .protocol import timeline, prediction_valid
from experiments.prm_five_groups_20261114.eval_common import correlations


def cycle_triplets(row, entry):
    assert entry['condition'] == 'rewind_only'
    original = timeline(row, entry)['original_positions']
    groups = []
    for event_index, event in enumerate(entry['events']):
        anchor, length = event['after_source_step'], event['duration_steps']
        first = event['first_inserted_step']
        points = []
        # Both turning endpoints have only two visits. Interior source frames
        # have exactly three event-local roles, even when two loops overlap.
        for source in range(anchor-length+1, anchor):
            reverse = first + anchor-source-1
            forward = first + length + source-(anchor-length)-1
            indices = [original[source], reverse, forward]
            assert all(entry['source_step_indices'][i] == source for i in indices)
            points.append(dict(source_step=source, source_frame=row['frame_indices'][source],
                               prediction_indices=indices))
        groups.append(dict(event_index=event_index, excluded_turning_source_steps=[anchor-length, anchor], points=points))
    return groups


def score(predictions, row, entry):
    assert len(predictions) == len(entry['target_pt'])
    assert all(v is None or prediction_valid(v) for v in predictions)
    pearson, spearman, reason = correlations(predictions, entry['target_pt'])
    result = dict(pearson=pearson, spearman=spearman, correlation_undefined_reason=reason,
        invalid_frames=sum(v is None for v in predictions), observations=len(predictions),
        circle_consistency=None, static_mae=None, events=[])
    if entry['condition'] == 'rewind_only':
        all_k = []; invalid_points = 0
        for event in cycle_triplets(row, entry):
            values = []
            for point in event['points']:
                p = [predictions[i] for i in point['prediction_indices']]
                k = (abs(p[2]-p[0])+abs(p[1]-p[0])+abs(p[2]-p[1]))/3 if all(prediction_valid(x) for x in p) else None
                point.update(predictions=p, k=k)
                if k is None:
                    invalid_points += 1
                else:
                    values.append(k); all_k.append(k)
            event.update(expected_points=len(event['points']), valid_points=len(values),
                mean=math.fsum(values)/len(values) if len(values) == len(event['points']) else None)
            result['events'].append(event)
        result.update(circle_consistency=math.fsum(all_k)/len(all_k) if all_k and not invalid_points else None,
            circle_reason='invalid_prediction_in_triplet' if invalid_points else None,
            expected_triplets=len(all_k)+invalid_points, valid_triplets=len(all_k),
            static_reason='not_applicable')
    else:
        assert entry['condition'] == 'static_only'
        maes = []
        for j, event in enumerate(entry['events']):
            indices = list(range(event['first_inserted_step'], event['last_inserted_step']+1))
            values = [predictions[i] for i in indices]
            valid = all(prediction_valid(v) for v in values)
            mean = math.fsum(values)/len(values) if valid else None
            mae = math.fsum(abs(v-mean) for v in values)/len(values) if valid else None
            if mae is not None:
                maes.append(mae)
            result['events'].append(dict(event_index=j, prediction_indices=indices, segment_mean=mean,
                static_mae=mae, expected_frames=len(values), valid_frames=sum(prediction_valid(v) for v in values)))
        result.update(static_mae=math.fsum(maes)/2 if len(maes) == 2 else None,
            static_reason=None if len(maes) == 2 else 'invalid_prediction_in_static_segment',
            circle_reason='not_applicable')
    return result

"""Append validated batches; never recompute or replace an existing prefix."""
from pathlib import Path
import time
from experiments.soc_disrupted_eval_20261115.common import read, save, digest
from experiments.soc_disrupted_eval_20261115.engine import empty_record, json_raw
from experiments.soc_disrupted_eval_20261115.protocol import validate_record, model_episode, prediction_valid
from experiments.soc_disrupted_eval_20261115.metrics import score

VECTORS=('predictions','raw_responses','input_audits','source_frame_indices',
         'playback_timestamps','playback_positions','target_pt')


def run_episode(backend,row,entry,identity,destination,execution,check=lambda:None,progress=lambda n:None):
    destination=Path(destination);partial=destination.with_suffix('.partial.json')
    if destination.exists():
        record=read(destination);validate_record(record,row,entry,identity,complete=True)
        assert record['status']=='completed' and record['initialization']==backend.audit
        return record
    record=read(partial) if partial.exists() else empty_record(row,entry,identity,backend.audit)
    start=validate_record(record,row,entry,identity)
    assert record['initialization']==backend.audit
    before={k:digest(record[k]) for k in VECTORS}
    episode=model_episode(row,entry);backend.begin(episode)
    for i in range(start):check();backend.rebuild_step(i)
    segments=record.setdefault('execution_segments',[])
    if start and not segments:
        segments.append(dict(start=0,end=start,mode='original_native_singleton',preserved=True))
    segment=None
    saved_count=start;last_save=time.monotonic()
    try:
        for offset in range(start,len(entry['target_pt']),execution['batch_size']):
            check()
            steps=list(range(offset,min(offset+execution['batch_size'],len(entry['target_pt']))))
            outputs=backend.predict_many(steps)
            assert len(outputs)==len(steps)
            for i,output in zip(steps,outputs):
                value,raw=output['prediction'],output['raw_response']
                if value is not None and not prediction_valid(value):
                    raw=dict(raw_response=raw,invalid_prediction=repr(value));value=None
                record['predictions'].append(value);record['raw_responses'].append(json_raw(raw))
                record['input_audits'].append(output['input_audit'])
                for k in ('source_frame_indices','playback_timestamps','target_pt'):record[k].append(entry[k][i])
                record['playback_positions'].append(episode['positions'][i])
            if segment is None:
                segment=dict(start=offset,end=offset,mode='validated_runtime',execution=execution,batch_sizes=[])
                segments.append(segment)
            segment.update(end=steps[-1]+1,saved_at=time.time());segment['batch_sizes'].append(len(steps))
            # Coalesce frequent JSON writes; finally flushes on cooperative pause.
            if steps[-1]+1-saved_count>=8 or time.monotonic()-last_save>=2:
                save(partial,record);progress(steps[-1]+1)
                saved_count=steps[-1]+1;last_save=time.monotonic()
    finally:
        # Cooperative pauses flush every finished prediction, including a short
        # pending write group. A hard crash can lose only uncommitted work.
        if len(record['predictions'])>saved_count:
            save(partial,record);progress(len(record['predictions']))
    assert all(digest(record[k][:start])==h for k,h in before.items()),'Existing prefix changed'
    validate_record(record,row,entry,identity,complete=True)
    record.update(status='completed',metrics=score(record['predictions'],row,entry))
    save(destination,record)
    return record


def validate_execution(record,approved):
    segments=record.get('execution_segments',[])
    if not segments:return
    cursor=0
    for segment in segments:
        assert segment['start']==cursor and segment['end']>cursor
        if segment['mode']=='validated_runtime':
            assert segment['execution']==approved
            assert sum(segment['batch_sizes'])==segment['end']-segment['start']
            assert all(0<n<=approved['batch_size'] for n in segment['batch_sizes'])
        else:
            assert cursor==0 and segment['mode']=='original_native_singleton' and segment['preserved']
        cursor=segment['end']
    assert cursor==len(record['predictions'])

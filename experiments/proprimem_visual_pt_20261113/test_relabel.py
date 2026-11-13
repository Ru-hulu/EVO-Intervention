import json
from pathlib import Path

import numpy as np
import pytest

from experiments.proprimem_visual_pt_20261113 import relabel as r


def test_strip_only_time_labels_preserves_stage_annotations():
    source = {"stages": [{"start_frame": 0, "end_frame": 3}], "task_prompt": "task",
              "annotation_status": "reviewed", "progress": {"policy": r.OLD_POLICY, "dense_progress": [0,.5,1]},
              "progress_review": {"status": "approved"}, "progress_annotation_status": "approved"}
    clean = r.strip_time_fields(source)
    assert clean == {k:source[k] for k in ("stages","task_prompt","annotation_status")}
    assert "progress" in source
    visual = {"progress": {"policy": r.POLICY, "dense_progress": [0,.2,1]}}
    assert r.strip_time_fields(visual) == visual


def test_cleanup_keeps_recover_and_splits_and_refreshes_hashes(tmp_path):
    root, setup = tmp_path/'data',tmp_path/'setup'
    setup.mkdir()
    sources = []
    for suite in r.SUITES:
        task = root/suite/suite/'task';task.mkdir(parents=True)
        annotation = task/'labels.json'
        r.save(annotation, {"total_frames":3,"stages":[{"name":"unchanged"}],
                            "progress":{"policy":r.OLD_POLICY,"dense_progress":[0,.5,1]}})
        r.save_lines(task/'metadata.jsonl',[{"sample_id":"s1","annotation_path":"labels.json",
                      "annotation_sha256":r.sha(annotation),"has_progress":True,"progress_policy":r.OLD_POLICY}])
        r.save(task/'metadata.json',{"trajectory_count":1,"progress_policy":r.OLD_POLICY,"progress_annotated_trajectory_count":1})
        r.save(root/suite/'manifest.json',{"tasks":[{"task_id":"T1","train":["s1"],"val":[],
                  "trajectory_index":f"{suite}/task/metadata.jsonl"}],"progress_update":{"policy":r.OLD_POLICY}})
        sources.append({"annotation":str(annotation),"metadata_index":str(task/'metadata.jsonl')})
    recover=root/'Recover'/'keep.json';r.save(recover,{"progress":{"policy":r.OLD_POLICY,"dense_progress":[0,.5,1]}})
    before=recover.read_bytes()
    report=r.remove_time_labels(root,setup,sources)
    assert report['removed_trajectory_labels']==dict.fromkeys(r.SUITES,1)
    assert recover.read_bytes()==before
    for source in sources:
        annotation=Path(source['annotation']);assert r.read(annotation)['stages']==[{"name":"unchanged"}]
        row=r.json_lines(source['metadata_index'])[0]
        assert 'has_progress' not in row and 'progress_policy' not in row
        assert row['annotation_sha256']==r.sha(annotation)
    for suite in r.SUITES:
        m=r.read(root/suite/'manifest.json');assert 'progress_update' not in m
        assert m['tasks'][0]['train']==['s1'] and m['tasks'][0]['val']==[]
        assert m['tasks'][0]['trajectory_index_sha256']==r.sha(root/suite/m['tasks'][0]['trajectory_index'])


def test_csv_verification_rejects_time_substitution_and_missing_frames(tmp_path):
    path=tmp_path/'p.csv';meta=tmp_path/'p.json'
    row={'csv':str(path),'provenance':str(meta),'frames':4,'source_id':'x','video_sha256':'v'}
    def write(values):
        np.savetxt(path,values,delimiter=',',header='frame_index,delta,pt',comments='')
        r.save(meta,{'protocol_sha256':'p','video_sha256':'v','source_id':'x','frame_count':4,'csv_sha256':r.sha(path)})
    write([[0,0,0],[1,1,.1],[2,2,.3],[3,7,1]])
    r.verify_csv(row,'p')
    write([[0,0,0],[1,1,1/3],[2,2,2/3],[3,7,1]])
    with pytest.raises(AssertionError):r.verify_csv(row,'p')
    write([[0,0,0],[1,1,.1],[3,7,1]])
    with pytest.raises(AssertionError):r.verify_csv(row,'p')


def test_queue_claims_are_disjoint_and_longest_first(tmp_path):
    db=r.db_connect(tmp_path)
    db.execute('CREATE TABLE jobs(source_id TEXT PRIMARY KEY,payload TEXT,frames INTEGER,status TEXT,worker_id INTEGER,started_at REAL)')
    for name,frames in [('short',2),('long',10)]:
        db.execute('INSERT INTO jobs VALUES(?,?,?,\'pending\',NULL,NULL)',(name,json.dumps({'id':name}),frames))
    db.commit();second=r.db_connect(tmp_path)
    assert r.claim(db,0)=={'id':'long'}
    assert r.claim(second,1)=={'id':'short'}
    assert r.claim(db,0) is None


def test_paths_cannot_escape_authorized_suite(tmp_path):
    with pytest.raises(ValueError):r.inside(tmp_path/'State'/'..'/'Recover'/'label',tmp_path/'State')


def test_recover_additions_require_exact_audit_and_original_files_stay_protected(tmp_path):
    root, setup = tmp_path/'data', tmp_path/'setup'
    original = root/'Recover'/'original.json'
    r.save(original, {'label': [0, .2, 1]})
    before = r.recover_snapshot(root)
    r.save(setup/'recover-before.json', before)
    assert r.verify_recover(root, setup)['file_set_unchanged']
    addition = root/'Recover'/'new-task'/'metadata.json'
    r.save(addition, {'source': 'separate import'})
    with pytest.raises(AssertionError): r.verify_recover(root, setup)
    after = r.recover_snapshot(root)
    receipt = setup/'additions.json'
    r.save(receipt, {'before_snapshot_sha256': r.sha(setup/'recover-before.json'),
                    'added_files': {k:v for k,v in after.items() if k not in before}})
    check = r.verify_recover(root, setup, receipt)
    assert check['existing_files_unchanged'] and not check['file_set_unchanged']
    assert check['added_file_count'] == 1
    r.save(addition, {'source': 'changed after audit'})
    with pytest.raises(AssertionError): r.verify_recover(root, setup, receipt)
    r.save(addition, {'source': 'separate import'})
    r.save(original, {'label': [0, .8, 1]})
    with pytest.raises(AssertionError): r.verify_recover(root, setup, receipt)
    original.unlink()
    with pytest.raises(AssertionError): r.verify_recover(root, setup, receipt)

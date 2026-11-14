"""Additional execution and untouched-prefix audit, then frozen scientific metrics."""
from .common import *
from .engine import VECTORS, validate_execution


def preserved_outputs():
    if (OPT/'previous-results-cleared.json').exists():
        return dict(previous_outputs_discarded_by_user_request=True,
                    audit=read(OPT/'previous-results-cleared.json'))
    inventory=read(OPT/'preserved-predictions.json')['files']
    count=0
    for name,expected in inventory.items():
        path=Path(name);backup=OPT/'original-output-backup'/path.relative_to(OUT)
        assert sha(backup)==expected
        if not name.endswith('.partial.json'):
            assert sha(path)==expected,'Pre-existing complete prediction changed'
        else:
            original,current=read(backup),read(path);n=len(original['predictions'])
            for key in VECTORS:assert current[key][:n]==original[key],(name,key)
            for key in ('identity','initialization','initialization_sha256'):
                assert current[key]==original[key],(name,key)
        count+=1
    return dict(files=count,completed_byte_identical=True,partial_prefixes_exact=True)


def report():
    assert read(OPT/'runtime-manifest.json')['files']==runtime_hashes()
    execution=read(OPT/'execution-configs.json');source=read(ROOT/'inputs.json')
    receipts={k:verify_choice(k) for k in jobs(source)}
    predictions=0
    for key,job in jobs(source).items():
        assert execution[key]['tuning_receipt_sha256']==sha(OPT/'tuning'/f'{key}.json')
        approved=dict(execution[key])
        memo_path=OPT/'memoization'/f'{key}.json'
        if execution[key].get('exact_input_memoization') and memo_path.exists():
            memo=read(memo_path)
            assert memo['status']=='passed' and memo['repeated_outputs_exact'] and memo['inputs_sha256']==sha(ROOT/'inputs.json')
            assert memo['code_sha256']==sha(Path(__file__).parent/'memo.py')
            approved['memoization_receipt_sha256']=sha(memo_path)
        for condition in CONDITIONS:
            for row,entry in task_inputs(source,job,condition):
                path=result_path(job['model'],condition,row['key'])
                if path.exists():
                    record=read(path)
                    if (OPT/'previous-results-cleared.json').exists():
                        assert record.get('execution_segments'),'Fresh result lacks runtime provenance'
                    validate_execution(record,approved);predictions+=1
    preservation=preserved_outputs()
    from experiments.soc_disrupted_eval_20261115.report import report as original_report
    result=original_report()
    result['execution_optimization']=dict(configurations=execution,preserved_outputs=preservation,
        checked_completed_curves=predictions,runtime_manifest_sha256=sha(OPT/'runtime-manifest.json'),
        note='Initialization singleton flag describes frozen reference; execution_segments record actual validated runtime batching.')
    save(OUT/'report.json',result)
    return result


if __name__=='__main__':
    r=report();print({k:r[k] for k in ('status','present_curves','observations')})

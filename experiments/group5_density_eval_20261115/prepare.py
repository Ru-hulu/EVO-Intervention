import os
import time
from .common import *


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    verify_code()
    # Re-audit all 195078 PNGs, all 900 model input maps and all 180 reused curves.
    from experiments.group5_local_density_20261115.verify import verify
    proof=verify();assert proof['status']=='prepared_waiting_for_user_instruction'
    data=read(DATA/'inputs.json');old=read(SOURCE)
    assert data['registry']==old['registry']
    base=ROOT/'code'
    # Model, memory, prompt and processor implementations are exactly those of
    # the completed uniform-FPS experiment, not a new inference implementation.
    for p,h in old['source_code_sha256'].items():assert sha(base/p)==h,p
    execution=dict(schema='cumulative-density-evaluation-v1',inputs_sha256=sha(DATA/'inputs.json'),
        preparation_verification_sha256=sha(DATA/'verification.json'),
        user_authorization='2026-11-15: start frozen cumulative-density evaluation; two GPUs per model/suite; tune each before formal inference',
        task_count=9,gpu_pairs=[[0,1],[2,3]],replica_candidates_per_gpu=[1,2,3,4],
        benchmark_seconds_per_candidate=40,checkpoint_policy='exact previously frozen matching-suite SOC35 final 1-epoch checkpoint',
        baseline_policy='reuse immutable 180 verified 6-FPS curves',
        new_curve_policy='720 full streams; fresh memory at trajectory start only; no splicing or reset at density boundaries',
        scoring=['common_6fps','full_rate'],metrics=['pearson','spearman','VS','ER','C'],
        source_checksums_sha256=sha(ROOT/'source-checksums.json'),
        optimization='identical native forward plus verified CPU prefetch; independently tuned batch and independent-trajectory concurrency',
        equivalence_tolerance=.002,prefetch_and_forest_resume_tolerance=.000001,
        safety=dict(per_process_allocator_fraction=.19,min_free_gpu_gib=16,max_temperature_c=80,
                    max_resident_fraction=.65,max_nonreclaimable_fraction=.85,no_new_oom=True),
        recurring_automation=False,prepared=time.time())
    assert not (ROOT/'execution.json').exists()
    save(ROOT/'execution.json',execution)
    save(ROOT/'preflight.json',dict(status='passed',inputs_sha256=execution['inputs_sha256'],cache_verification=proof,
        all_native_source_files_unchanged=True,checked=time.time()))
    print('Preflight passed: 9 two-GPU jobs, 180 reused curves, 720 new curves',flush=True)


if __name__=='__main__':prepare()

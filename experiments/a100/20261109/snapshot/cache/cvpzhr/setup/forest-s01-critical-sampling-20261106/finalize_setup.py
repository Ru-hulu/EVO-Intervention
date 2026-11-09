import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch

w=Path('/cache/cvpzhr/setup/forest-s01-critical-sampling-20261106')
a=Path('/cache/cvpzhr/experiments/forest-s01-decline/02-critical-sampling-20261106')
data=Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261106-v3-decline/State/01-cover_uncover_blocks')
c=json.loads((w/'config.json').read_text());rows=json.loads((w/'split.json').read_text())['records']
groups={};counts=collections.Counter()
for r in rows:
    if r['split']!='validation':continue
    ann=json.loads((data/r['annotation_path']).read_text())['progress'];mapping=json.loads((Path(r['cache_dir'])/'source_frames.json').read_text())
    values={}
    for m in mapping:
        assert not m['pause_copy']
        f=m['source_frame'];g=0 if r['category'].endswith('_success') else 4
        for interval in ann.get('penalty_intervals',[]):
            fail=interval['native_failure_frame'];error=interval['error_range'];recovery=interval['recovery_range']
            if fail is not None and f>=fail:g=3;break
            if error[0]<=f<error[1]:g=1;break
            if recovery and recovery[0]<=f<=recovery[1]:g=2;break
        values[str(m['frame_index'])]=g;counts[g]+=1
    groups[r['cache_dir']]=values
assert sum(counts.values())==8978 and len(groups)==28
(w/'validation-event-groups.json').write_text(json.dumps(groups)+'\n')
c['validation_event_groups']=str(w/'validation-event-groups.json')
(w/'config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
sys.path.insert(0,str(w/'code'))
from mem_lite.periodic_validation import validate
class Dummy(torch.nn.Module):
    def forward(self,inputs):
        return dict(pt=inputs['pred'],bin_logits=torch.zeros(4,101))
class Loader(list):dataset=range(4)
loader=Loader([dict(inputs={'pred':torch.tensor([.4,.6,.7,.9])},target_pt=torch.tensor([.8,.5,.7,0.]),
    suite_indices=torch.tensor([2,2,2,2]),event_group_indices=torch.tensor([0,1,2,3]))])
result=validate(Dummy(),loader,torch.device('cpu'))
assert result['num_samples']==4 and abs(result['loss']-float(torch.log(torch.tensor(101.))))<1e-5
assert result['by_event']['post_failure']['predicted_progress_above_20pct_rate']==1
assert result['by_event']['normal_success']['underestimate_over_10pp_rate']==1
assert result['by_event']['recovery']['mae_pp']==0
(w/'validation-metric-check.json').write_text(json.dumps(dict(passed=True,validation_group_counts=dict(counts),synthetic_result=result),indent=2)+'\n')
hashes={str(p.relative_to(w/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((w/'code').rglob('*.py'))}
(w/'code-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
for name in ['config.json','code-sha256.json','validation-metric-check.json','validation-event-groups.json']:
    shutil.copyfile(w/name,a/name)
print('PASS: weighted sampler audit and validation loss/event metrics; ready for four GPUs',dict(counts))

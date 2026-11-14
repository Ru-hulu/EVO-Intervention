"""Exact-input memoization only for stateless, deterministic singleton models.

FoREST is excluded: matching pictures alone do not imply matching memory.
The frozen input manifest and PNG hashes make path identity content identity.
"""
from collections import OrderedDict
import copy
from .common import *
from experiments.soc_disrupted_eval_20261115.protocol import image_slots, model_episode


class Memoized:
    def __init__(self,backend,enabled):
        self.backend=backend;self.enabled=enabled;self.audit=backend.audit
        if enabled:assert backend.name in ('procvlm-2b','robometer-4b') and backend.batch_size==1
        self.cache=OrderedDict();self.hits=0;self.misses=0

    def begin(self,episode):self.backend.begin(episode)
    def rebuild_step(self,i):self.backend.rebuild_step(i)

    def predict_many(self,steps):
        if not self.enabled:return self.backend.predict_many(steps)
        assert len(steps)==1
        i=steps[0];e=self.backend.episode;slots=image_slots(e,i,self.backend.name)
        key=(self.audit['checkpoint_sha256'],e['task'],tuple(e['images'][j][v] for j,v in slots))
        if key in self.cache:
            self.hits+=1;result=copy.deepcopy(self.cache.pop(key));self.cache[key]=copy.deepcopy(result)
            result['input_audit'].update(playback_step=i,playback_position=e['positions'][i],input_playback_slots=slots)
            return [result]
        self.misses+=1;result=self.backend.predict_many(steps)[0]
        if len(self.cache)>=100000:self.cache.popitem(last=False)
        self.cache[key]=copy.deepcopy(result)
        return [result]


def gpu_gate(wrapper,source,job,check):
    """Check real reference inputs, then exact repeated calls across episode resets."""
    from .tune import cases,compare
    references=[];outputs=[];repeated=[]
    key=f"{job['model']}-{job['suite']}"
    selected=[]
    for j,(row,entry,steps) in enumerate(cases(source,job)):
        proof=read(OPT/'tuning'/key/f'reference-{j}.json')
        assert proof['key']==row['key'] and proof['steps']==steps
        indices=sorted({round(k*(len(steps)-1)/7) for k in range(8)})
        selected.append((model_episode(row,entry),[steps[k] for k in indices],[proof['outputs'][k] for k in indices]))
    wrapper.cache.clear();wrapper.hits=wrapper.misses=0
    for ep,steps,reference in selected:
        wrapper.begin(ep)
        for i in steps:check();outputs.extend(wrapper.predict_many([i]))
        references.extend(reference)
    native=compare(references,outputs)
    assert native['max_abs_error']<=.002 and native['invalid_positions_equal'] and native['input_equivalence_verified'],native
    misses=wrapper.misses
    for ep,steps,_ in selected:
        wrapper.begin(ep)
        for i in steps:check();repeated.extend(wrapper.predict_many([i]))
    # JSON canonicalization treats native tuples and saved JSON lists identically.
    assert digest(outputs)==digest(repeated) and wrapper.misses==misses
    receipt=dict(status='passed',job_id=key,initialization_sha256=digest(wrapper.audit),
        compared_observations=len(outputs),native_comparison=native,repeated_outputs_exact=True,
        gpu_forwards=misses,cache_hits=wrapper.hits,code_sha256=sha(__file__),inputs_sha256=sha(ROOT/'inputs.json'))
    save(OPT/'memoization'/f'{key}.json',receipt)
    wrapper.cache.clear();wrapper.hits=wrapper.misses=0
    return receipt

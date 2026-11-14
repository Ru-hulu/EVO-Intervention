"""Conservative DDP GPU ownership proof across Linux PID namespaces.

NVML PIDs are opaque here: never guess a local PID from their numeric value.
After supplemental workers have exited, require exactly one compute process on
each GPU of the two-rank training group. Both ranks, their mapped device minors,
and a newly completed CUDA all-reduce/all-gather training log must agree across
a stable observation interval. That proves group ownership, not a per-rank PID
translation. Any ambiguity leaves the slot closed. No CUDA calls or signals.
"""
import json
from pathlib import Path
import re


def process_identity(pid, proc=Path('/proc')):
    fields=(proc/str(pid)/'stat').read_text().split(') ',1)[1].split()
    if fields[0]=='Z':raise ValueError('Zombie process')
    return {'pid':int(pid),'parent':int(fields[1]),'start_ticks':int(fields[19])}


def training_identity(group, base, minors, proc=Path('/proc')):
    """Read only the recorded torchrun and its two direct, configured ranks."""
    parent=process_identity(group['pid'],proc)
    parent_dir=proc/str(parent['pid'])
    children=[int(p) for p in (parent_dir/'task'/str(parent['pid'])/'children').read_text().split()]
    ranks=[];mapped=set();expected={minors[g] for g in group['gpus']}
    config=base/group['suite']/'training.config.json'
    for pid in children:
        root=proc/str(pid);identity=process_identity(pid,proc)
        command=(root/'cmdline').read_bytes().split(b'\0')
        if b'mem_lite.train_stage3' not in command:continue
        if b'--config' not in command or command[command.index(b'--config')+1].decode()!=str(config):
            raise ValueError('Unexpected training configuration')
        # Parse only device/rank keys; never record arbitrary environment data.
        allowed={b'CUDA_VISIBLE_DEVICES',b'LOCAL_RANK',b'RANK',b'WORLD_SIZE'}
        env={k.decode():v.decode() for item in (root/'environ').read_bytes().split(b'\0')
             if b'=' in item for k,v in [item.split(b'=',1)] if k in allowed}
        if env.get('CUDA_VISIBLE_DEVICES')!=','.join(map(str,group['gpus'])) or env.get('WORLD_SIZE')!='2':
            raise ValueError('Unexpected CUDA rank assignment')
        rank=int(env['LOCAL_RANK'])
        if str(rank)!=env['RANK'] or rank not in (0,1):raise ValueError('Unexpected DDP rank')
        devices={int(m.group(1)) for line in (root/'maps').read_text().splitlines()
                 if (m:=re.search(r'/dev/nvidia(\d+)$',line))}
        if not devices or not devices<=expected:raise ValueError('GPU device mappings do not match the recorded group')
        if identity['parent']!=parent['pid']:raise ValueError('Rank parent changed')
        mapped.update(devices);ranks.append(dict(identity,rank=rank,device_minors=sorted(devices)))
    if len(ranks)!=2 or {r['rank'] for r in ranks}!={0,1} or mapped!=expected:
        raise ValueError('Both training ranks and device mappings are required')
    return {'suite':group['suite'],'gpus':group['gpus'],'parent':parent,
            'ranks':sorted(ranks,key=lambda r:r['rank'])}


def latest_progress(suite, now, output_root=Path('/cache/cvpzhr/outputs')):
    path=output_root/f'forest-{suite.lower()}-qwen35-clip-depth-e1-20261114'/'train.jsonl'
    with path.open('rb') as stream:
        stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-65536));tail=stream.read()
    for line in reversed(tail.split(b'\n')[:-1]):
        try:row=json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
        if row.get('kind')!='train':continue
        if not 0<=now-row['timestamp']<=600:raise ValueError('Training log is stale')
        memory=row.get('peak_allocated_gib_per_gpu',[])
        if row.get('epoch')!=1 or len(memory)!=2 or not all(v>0 for v in memory):
            raise ValueError('Missing two-rank CUDA collective evidence')
        return {'step':row['step'],'timestamp':row['timestamp']}
    raise ValueError('No complete recent training log record')


class GroupOwnership:
    def __init__(self):
        self.candidates={};self.proofs={};self.status={}

    def observe(self, identity, progress, processes, now):
        key=identity['suite'];gpus=identity['gpus']
        proof=self.proofs.get(key)
        if proof and proof['identity']!=identity:
            self.proofs.pop(key);proof=None
        if proof:
            # Supplemental workers may now be present on other group GPUs.
            # A training PID disappearance invalidates the entire group proof.
            if all(proof['host_pids'][g] in processes[g] for g in gpus):
                self.status[key]={'status':'verified_group_ownership','proof':proof}
                return
            self.proofs.pop(key)
        if any(len(processes[g])!=1 for g in gpus):
            self.candidates.pop(key,None)
            self.status[key]={'status':'waiting_for_singleton_training_contexts'};return
        host_pids={g:processes[g][0] for g in gpus}
        # Two ranks must remain two distinct contexts, not one unknown process.
        if len(set(host_pids.values()))!=2:
            self.candidates.pop(key,None)
            self.status[key]={'status':'ambiguous_compute_contexts'};return
        prior=self.candidates.get(key)
        if (not prior or prior['identity']!=identity or prior['host_pids']!=host_pids
                or now-prior['last_seen']>30 or now<prior['last_seen']):
            prior={'identity':identity,'host_pids':host_pids,'started':now,
                   'initial_progress':progress,'last_seen':now}
            self.candidates[key]=prior
        prior['last_seen']=now
        if (now-prior['started']>=60 and progress['step']>prior['initial_progress']['step']
                and progress['timestamp']>prior['started']):
            proof=dict(prior,verified_at=now,verified_progress=progress,
                       method='singleton_NVML_contexts_spanning_fresh_two_rank_CUDA_collective')
            self.proofs[key]=proof;self.candidates.pop(key,None)
            self.status[key]={'status':'verified_group_ownership','proof':proof}
        else:self.status[key]={'status':'waiting_for_fresh_collective_and_60_seconds','since':prior['started']}

    def invalidate(self,suite,reason):
        self.proofs.pop(suite,None);self.candidates.pop(suite,None)
        self.status[suite]={'status':'unverified','reason':reason}

    def permits(self,group,gpu,processes):
        proof=self.proofs.get(group['suite'])
        return bool(proof and proof['identity']['parent']['pid']==group['pid']
                    and proof['identity']['gpus']==group['gpus']
                    and processes[gpu]==[proof['host_pids'][gpu]])

    def refresh(self,training,processes,now,base,minors):
        groups=[g for g in training.get('groups',[]) if g['phase']=='training']
        for suite in set(self.proofs)|set(self.candidates):
            if suite not in {g['suite'] for g in groups}:self.invalidate(suite,'training_group_changed')
        for group in groups:
            try:
                if not 0<=now-training.get('updated',0)<=90:raise ValueError('Controller is stale')
                identity=training_identity(group,base,minors)
                progress=latest_progress(group['suite'],now)
                self.observe(identity,progress,processes,now)
            except (OSError,ValueError,KeyError,IndexError) as exc:
                self.invalidate(group['suite'],f'{type(exc).__name__}: {exc}')

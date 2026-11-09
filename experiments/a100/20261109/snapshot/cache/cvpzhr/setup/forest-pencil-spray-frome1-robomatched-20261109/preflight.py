from pathlib import Path
import json, sys, hashlib, math, csv
import torch
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parent
policy=json.loads((ROOT/'launch-policy.json').read_text())
sys.path.insert(0,str(ROOT/'pencil/code'))
from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.train_stage3 import normalize_gradients
from transformers import get_cosine_schedule_with_warmup
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
initial=Path(policy['initial_checkpoint']);assert sha(initial)==policy['initial_checkpoint_sha256']
x=torch.load(initial,map_location='cpu',weights_only=True)
assert (x['epoch'],x['next_batch'],x['step'])==(1,0,19934)
assert all(torch.isfinite(v).all() for comp in x['model'].values() for v in comp.values())
report={'status':'passed','initial_checkpoint':str(initial),'initial_sha256':sha(initial),'tasks':{}}
for task,t in policy['tasks'].items():
 S=Path(t['setup_dir']);D=Path(t['cache_dir']);R=Path(t['robometer_cache']);cfg=read(S/'training.config.json');plan=read(D/'plan.json')
 rcfg=read(Path(cfg['run_protocol']['robometer_reference']))
 for k in ['model_path','model','history_capacity','history_source']:assert cfg[k]==x['config'][k],k
 assert cfg['epochs']==rcfg['training']['num_train_epochs']==3
 assert cfg['learning_rate']==rcfg['training']['learning_rate']==1e-4
 assert cfg['warmup_ratio']==rcfg['training']['warmup_ratio']==.03
 assert cfg['batch_size']*cfg['gradient_accumulation_steps']==rcfg['training']['per_device_train_batch_size']*rcfg['training']['gradient_accumulation_steps']
 assert cfg['expected_total_steps']==rcfg['expected_steps']
 result={}
 for split,ne,ns in [('train',t['counts'][0],t['counts'][1]),('validation',t['counts'][2],t['counts'][3])]:
  records=[r for r in plan['records'] if r['split']==split];paths=read(D/f'{split}.sample_files.json')
  assert len(records)==ne and paths==[r['output_dir'] for r in records]
  forest={};prompts={}
  for r in records:
   p=Path(r['output_dir']);summary=read(p/'summary.json');receipt=read(D/'receipts'/f"{r['episode_id']}.json")
   assert summary['augmentation']['enabled'] is False and summary['augmentation']['inserted_observations']==0
   for fn,key in [('samples.jsonl','samples_sha256'),('summary.json','summary_sha256')]:assert sha(p/fn)==receipt[key]
   assert sha(Path(r['progress_csv']))==r['progress_sha256']
   with Path(r['progress_csv']).open() as f:gt={int(row['frame_index']):float(row['pt']) for row in csv.DictReader(f)}
   prompts[r['episode_id']]=r['task_prompt']
   for line in (p/'samples.jsonl').read_text().splitlines():
    row=json.loads(line);key=(r['episode_id'],row['frame_index']);assert key not in forest
    assert row['pt']==gt[row['frame_index']]
    forest[key]=row['pt']
  robo={}
  for line in (R/f'{split}.jsonl').read_text().splitlines():
   row=json.loads(line);key=(row['episode_id'],row['source_frame_index']);assert key not in robo
   assert row['kind']=='original' and row['task']==prompts[row['episode_id']]
   robo[key]=row['pt']
  assert len(forest)==ns and forest.keys()==robo.keys()
  assert all(abs(forest[k]-robo[k])<1e-12 for k in forest)
  ds=M3Dataset(paths);assert len(ds)==ns and ds.history_source==cfg['history_source']
  if split=='train':assert ds.signature==cfg['data_signature']
  result[split]={'episodes':ne,'samples':ns,'labels_and_frame_ids_identical_to_robometer':True,'data_signature':ds.signature}
  del ds
 result.update(global_batch_size=cfg['batch_size']*cfg['gradient_accumulation_steps']*2,total_steps=cfg['expected_total_steps'],warmup_steps=math.ceil(cfg['expected_total_steps']*.03))
 report['tasks'][task]=result

# Accumulation must equal one large batch, including uneven last batches and DDP rank averaging.
torch.manual_seed(42)
for counts in [(9,9),(4,4),(8,8)]:
 X=torch.randn(sum(counts),5,dtype=torch.float64);Y=torch.randn(sum(counts),1,dtype=torch.float64)
 base=torch.nn.Linear(5,1,bias=True,dtype=torch.float64);state=base.state_dict()
 ((base(X)-Y)**2).mean().backward();expected=[p.grad.clone() for p in base.parameters()]
 grads=[];start=0
 for n in counts:
  model=torch.nn.Linear(5,1,bias=True,dtype=torch.float64);model.load_state_dict(state)
  for offset in range(0,n,3):
   xx=X[start+offset:start+min(offset+3,n)];yy=Y[start+offset:start+min(offset+3,n)]
   ((model(xx)-yy)**2).sum().backward()
  grads.append([p.grad.clone() for p in model.parameters()]);start+=n
 for i,p in enumerate(model.parameters()):p.grad=(grads[0][i]+grads[1][i])/2
 normalize_gradients(list(model.parameters()),sum(counts),2)
 assert all(torch.allclose(p.grad,v,rtol=1e-12,atol=1e-12) for p,v in zip(model.parameters(),expected))

# Exercise actual scheduler state save/restore and the full warmup/cosine schedule.
for total in [1392,1800]:
 warm=math.ceil(total*.03);p=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.AdamW([p],lr=1e-4)
 sched=get_cosine_schedule_with_warmup(opt,warm,total);rates=[opt.param_groups[0]['lr']]
 for step in range(total):
  p.grad=torch.zeros_like(p);opt.step();sched.step();rates.append(opt.param_groups[0]['lr'])
  if step==73:
   opt_state=opt.state_dict();sched_state=sched.state_dict()
 assert rates[0]==0 and abs(rates[warm]-1e-4)<1e-15 and abs(rates[-1])<1e-15
 assert all(a<=b for a,b in zip(rates[:warm],rates[1:warm+1]))
 assert all(a>=b for a,b in zip(rates[warm:-1],rates[warm+1:]))
 opt2=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=1e-4);s2=get_cosine_schedule_with_warmup(opt2,warm,total)
 opt2.load_state_dict(opt_state);s2.load_state_dict(sched_state);opt2.step();s2.step()
 assert abs(opt2.param_groups[0]['lr']-rates[75])<1e-15
report['gradient_accumulation_equivalence']='passed'
report['scheduler_and_resume_checks']='passed'
(ROOT/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))

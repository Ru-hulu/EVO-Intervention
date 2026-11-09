"""Fixed step14470; train 102 matched-state points, with complete causal histories."""
import json,time,hashlib,gc
from pathlib import Path
import torch
from mem_lite.datasets.m3_dataset import M3Dataset,Stage3Collator
from mem_lite.runtime import load_checkpoint,to_device
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder,load_vision_encoder
from select_episodes import ROOT,DATA
CHECKPOINT=Path('/cache/cvpzhr/evaluations/forest-s01-failure-continue-ce-episode55-step14470-20261108/checkpoint.pt')
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 torch.set_num_threads(4)
 chosen=json.loads((ROOT/'selection.json').read_text())['records'];assert len(chosen)==9
 datasets={s['episode']:M3Dataset([str(Path(s['row']['cache_dir'])/'samples.jsonl')]) for s in chosen}
 clean={}
 # Only the augmented episode needs rebuilding; original cached inputs stay immutable.
 for s in chosen:
  if not s['training_pause_copies']:continue
  ep=s['episode'];ds=datasets[ep];origin=ds.history_source
  visual,proc,digest=load_vision_encoder('/cache/cvpzhr/models/Qwen3.5-4B','cuda');assert digest==origin['vision_sha256']
  stream=StreamMemory(QwenFrameEncoder(visual),proc,origin['memory_config'],origin['max_pixels'],origin['min_pixels'])
  mapping=json.loads((Path(s['row']['cache_dir'])/'source_frames.json').read_text());wanted=set(s['matched_frames']);hist={}
  with torch.inference_mode():
   for i,m in enumerate(mapping):
    if m['pause_copy']:continue
    f=m['source_frame']
    if f>max(wanted):break
    item=ds[i];h=stream.step(item['images'][0],f,str(ep),item['task_prompt'])
    if f in wanted:hist[f]=h.cpu()
  clean[ep]=hist
  del visual,proc,stream;gc.collect();torch.cuda.empty_cache()
 model,processor,cfg=load_checkpoint(str(CHECKPOINT));assert not model.training
 collate=Stage3Collator(processor,cfg['model']['history_dim'],cfg['max_sequence_length'],cfg['history_capacity'])
 records=[];started=time.time()
 assert not (ROOT/'predictions.jsonl').exists()
 with torch.inference_mode(),(ROOT/'predictions.jsonl').open('w') as out:
  for s in chosen:
   ep=s['episode'];ds=datasets[ep];mapping=json.loads((Path(s['row']['cache_dir'])/'source_frames.json').read_text());wanted=set(s['matched_frames'])
   for i,m in enumerate(mapping):
    if m['source_frame'] not in wanted:continue
    original=ds[i];assert original['target_pt']==0
    conditions=[]
    if not m['pause_copy']:
     item=dict(original,frame_index=m['source_frame'])
     if ep in clean:item['history_tokens']=clean[ep][m['source_frame']]
     conditions.append(('normal_causal',item))
    if ep in clean:conditions.append(('exact_training_cache',original))
    for condition,item in conditions:
     inputs=to_device(collate([item])['inputs'],'cuda');res=model(inputs);probs=res['bin_probs'][0].float().cpu()
     r=dict(episode=ep,category=s['row']['category'],condition=condition,source_frame=m['source_frame'],cache_frame=m['frame_index'],pause_copy=m['pause_copy'],time_seconds=m['source_frame']/s['row']['fps'],target_pt=0.,predicted_pt=float(res['pt'][0]),p_zero=float(probs[0]),p_below10=float(probs[:11].sum()),probabilities=probs.tolist())
     records.append(r);out.write(json.dumps(r)+'\n');out.flush()
   print(json.dumps(dict(episode=ep,done_records=len(records),seconds=time.time()-started)),flush=True)
 summaries=[]
 for s in chosen:
  ep=s['episode']
  for condition in ['normal_causal','exact_training_cache']:
   rr=[r for r in records if r['episode']==ep and r['condition']==condition]
   if not rr:continue
   pred=torch.tensor([r['predicted_pt'] for r in rr],dtype=torch.float64)
   summaries.append(dict(episode=ep,category=s['row']['category'],sample_id=s['row']['sample_id'],condition=condition,n=len(rr),mean_prediction_pct=float(pred.mean()*100),max_prediction_pct=float(pred.max()*100),median_prediction_pct=float(pred.quantile(.5)*100),above20_count=int((pred>.2).sum()),above80_count=int((pred>.8).sum()),below1_count=int((pred<.01).sum()),mean_bin0_probability=float(sum(r['p_zero'] for r in rr)/len(rr)),time_range=[rr[0]['time_seconds'],rr[-1]['time_seconds']],same_layout_as_ep55=s['expected']==[1,2,0]))
 dump(ROOT/'summary.json',dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=hashlib.file_digest(CHECKPOINT.open('rb'),'sha256').hexdigest(),step=14470,training_modified=False,collection_started=False,batch_size=1,scope='train native-failed and only blue covered; full causal visual prefix, no added pauses',episodes=summaries,seconds=time.time()-started))
 dump(ROOT/'status.json',dict(phase='completed',episodes=9,records=len(records)))
if __name__=='__main__':main()

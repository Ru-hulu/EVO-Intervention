"""Cache original-frame Qwen visual deltas; never modify published labels."""
import argparse,hashlib,json,time,sys
from pathlib import Path
import av,numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent/'reference'))
from vision import load_vision_encoder
ROOT=Path('/cache/robot-baselines'); OUT=ROOT/'annotations/boundary-integral-20261105'
DATASETS=['ProPriMem-VLA-Failures-20261104','RoboMME-FrameSamp-20261105','RoboMME-SimpleMem-20261105']
def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False));tmp.replace(path)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',action='append');parser.add_argument('--sample',action='append');parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
 torch.set_num_threads(4)
 model,processor,vision_hash=load_vision_encoder('/cache/cvpzhr/models/Qwen3.5-4B','cuda')
 @torch.inference_mode()
 def encode(images):
  inputs=processor(images=images,return_tensors='pt',min_pixels=4096,max_pixels=262144)
  grid=inputs['image_grid_thw'].to('cuda');pixels=inputs['pixel_values'].to('cuda',dtype=next(model.parameters()).dtype)
  tokens=model(pixels,grid_thw=grid,return_dict=True).pooler_output.float()
  counts=(grid.prod(-1)//model.config.spatial_merge_size**2).tolist()
  return torch.stack([t.mean(0) for t in tokens.split(counts)]).cpu()
 for ds in args.dataset or DATASETS:
  root=ROOT/'viewer-datasets'/ds
  for row in map(json.loads,(root/'metadata.jsonl').read_text().splitlines()):
   if args.sample and row['sample_id'] not in args.sample:continue
   dest=args.out/ds/'deltas'/(row['sample_id']+'.json')
   if dest.exists():continue
   video=root/row['video_path'];started=time.time();previous=None;deltas=[];batch=[]
   def consume():
    nonlocal previous
    for feat in encode(batch):
     deltas.append(0. if previous is None else torch.linalg.vector_norm(feat-previous).item());previous=feat
    batch.clear()
   with av.open(str(video)) as container:
    container.streams.video[0].codec_context.thread_count=2
    for frame in container.decode(video=0):
     batch.append(frame.to_image())
     if len(batch)==8:consume()
    if batch:consume()
   assert len(deltas)==row['total_frames'] and np.isfinite(deltas).all()
   write(dest,dict(sample_id=row['sample_id'],deltas=deltas,video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),vision_sha256=vision_hash,feature='mean post-merger spatial tokens FP32; adjacent-frame raw L2',frames=len(deltas),seconds=time.time()-started))
   print(ds,row['sample_id'],len(deltas),round(time.time()-started,2),flush=True)
 write(args.out/'delta-status.json',dict(status='complete',time=time.time()))
if __name__=='__main__':main()

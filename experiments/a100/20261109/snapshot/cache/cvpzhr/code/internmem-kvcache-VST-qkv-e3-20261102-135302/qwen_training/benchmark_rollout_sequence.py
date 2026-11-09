"""Paired ordinary/verified greedy closed-loop controls across two rollbacks."""
import argparse
import json
from pathlib import Path
import time
import torch

from qwen_training.closed_loop_rollout import feedback_state, save_json, sha256
from qwen_training.evaluate_checkpoint import load_model
from qwen_training.prompts import build_messages
from qwen_training.rollout_inference import generate_verified
from qwen_training.validation_rollout_suite import EpisodeFrames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    root=args.directory
    suite=json.loads((root/'suite.json').read_text())
    spec=next(e for e in suite['episodes'] if e['video_id']=='file-010' and e['episode']==3)
    directory=root/spec['relative_dir']
    rows=[json.loads(l) for l in (directory/'rollout_index.jsonl').read_text().splitlines()]
    truth=json.loads((directory/'ground_truth.json').read_text())
    transitions=[i for i in range(1,len(truth)) if truth[i-1]['progress']-truth[i]['progress']>.03]
    starts=[i-4 for i in transitions[:2]]
    if len(starts)!=2:raise ValueError('Expected two rollback controls')
    torch.set_num_threads(2);torch.manual_seed(0)
    cache=EpisodeFrames(rows)
    ck=torch.load(suite['checkpoint']['checkpoint'],map_location='cpu',weights_only=True)
    model=load_model(ck,torch.device('cuda:0'));del ck
    ordinary_states=[{k:rows[i]['target'][k] for k in ('nowid','finishid','restid')} for i in starts]
    verified_states=[dict(s) for s in ordinary_states]
    drafts=[None,None];records=[]
    began=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for step in range(12):
            metas=[rows[i+step] for i in starts]
            images=torch.stack([cache.window(r['frame_indices']) for r in metas]).cuda()
            masks=torch.tensor([r['frame_mask'] for r in metas],dtype=torch.bool)
            visual=model.encode_history(images,masks)
            ordinary_messages=build_messages(ordinary_states,state_drop_probability=0.0)
            verified_messages=build_messages(verified_states,state_drop_probability=0.0)
            model.qwen.model.rope_deltas=None
            reference=model.generate(ordinary_messages,**visual,max_new_tokens=192)
            current=generate_verified(model,verified_messages,**visual,drafts=drafts,minimum_margin=1.0)
            for j,meta in enumerate(metas):
                record=dict(step=step,frame_index=meta['frame_index'],time_seconds=meta['time_seconds'],
                    ordinary_input_state=ordinary_states[j],verified_input_state=verified_states[j],
                    ordinary_response=reference['responses'][j],verified_response=current['responses'][j],
                    ordinary_progress=float(reference['progress'][j]),verified_progress=float(current['progress'][j]),
                    draft_accepted=current['draft_accepted'][j],minimum_margin=current['draft_minimum_margin'][j])
                records.append(record)
                if record['ordinary_response']!=record['verified_response']:
                    save_json(root/'benchmark/sequence.json',dict(status='failed',records=records))
                    raise ValueError('Closed-loop trajectories diverged')
                if abs(record['ordinary_progress']-record['verified_progress'])>1e-4:
                    raise ValueError('Progress regression path differs')
            ordinary_states=[feedback_state(r) for r in reference['responses']]
            verified_states=[feedback_state(r) for r in current['responses']]
            drafts=current['responses']
            save_json(root/'benchmark/sequence.json',dict(status='running',records=records))
            print(json.dumps(dict(step=step,samples=len(records),accepted=sum(r['draft_accepted'] for r in records))),flush=True)
    save_json(root/'benchmark/sequence.json',dict(status='passed',samples=len(records),
        checkpoint_sha256=sha256(suite['checkpoint']['checkpoint']),elapsed_seconds=time.monotonic()-began,
        note='Two 12-frame trajectories around rollback boundaries; each decoder feeds back its own output.',records=records))


if __name__=='__main__':main()

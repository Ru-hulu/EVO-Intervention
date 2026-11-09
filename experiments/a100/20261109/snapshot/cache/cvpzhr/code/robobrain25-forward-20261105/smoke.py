"""Short isolated GPU adaptation check; never initializes a full training run."""
import gc
import json
from pathlib import Path
import time
import torch
from transformers import set_seed
from grm import load_model,load_processor,Collator,training_inputs,generate,parse_score

ROOT=Path('/cache/cvpzhr/setup/robobrain25-forward-20261105')


def main():
    torch.set_num_threads(2);set_seed(42)
    config=json.loads((ROOT/'config.draft.json').read_text())
    assert json.loads((ROOT/'cpu-checks.json').read_text())['status']=='passed'
    rows=json.loads((ROOT/'smoke-rows.json').read_text())
    processor=load_processor(config['model_path'],config['min_pixels'],config['max_pixels'])
    collator=Collator(processor,config['max_length'])
    model=load_model(config['model_path'],lora=config['lora']).to('cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=1e-4)
    losses=[];gradients=[];times=[]
    for row in rows:
        model.train();optimizer.zero_grad(set_to_none=True)
        batch={k:v.to('cuda') for k,v in collator.encode([row]).items()}
        start=time.monotonic()
        loss=model(**training_inputs(batch),use_cache=False).loss
        assert torch.isfinite(loss);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(trainable,1.)
        assert torch.isfinite(grad) and grad>0
        assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
        optimizer.step();torch.cuda.synchronize()
        losses.append(float(loss));gradients.append(float(grad));times.append(time.monotonic()-start)
        print('smoke step',len(losses),'loss',losses[-1],flush=True)
    model.eval()
    probe={k:v.to('cuda') for k,v in collator.encode([rows[1]]).items()}
    with torch.no_grad():expected=float(model(**training_inputs(probe),use_cache=False).loss)
    result=generate(model,processor,probe,config['max_new_tokens'])[0]
    value=parse_score(result)
    checkpoint=ROOT/'smoke-adapter';model.save_pretrained(checkpoint,save_embedding_layers=False)
    peak=torch.cuda.max_memory_allocated()/2**30
    params=sum(p.numel() for p in trainable)
    del trainable,optimizer,model,batch,loss;gc.collect();torch.cuda.empty_cache()
    restored=load_model(config['model_path'],adapter=checkpoint).to('cuda').eval()
    with torch.no_grad():actual=float(restored(**training_inputs(probe),use_cache=False).loss)
    assert abs(actual-expected)<1e-5,(actual,expected)
    restored_result=generate(restored,processor,probe,config['max_new_tokens'])[0]
    assert result==restored_result,(result,restored_result)
    report=dict(status='passed',steps=len(losses),losses=losses,gradient_norms=gradients,
        step_seconds=times,trainable_parameters=params,peak_allocated_gib=peak,
        roundtrip_loss_difference=abs(actual-expected),generation=result,predicted_pt=value,
        adapter_save_load_verified=True,vision_and_base_frozen=True,formal_training_started=False)
    (ROOT/'gpu-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

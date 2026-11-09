"""Generate native GRM scores with fixed start and user-authorized final-frame reference."""
import argparse
import json
from pathlib import Path
import time
import torch
from grm import Dataset, Collator, load_model, load_processor, generate, parse_score
from metrics import summarize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--split', choices=['validation','augmented_validation'], default='augmented_validation')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--limit', type=int)
    args=p.parse_args();config=json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    processor=load_processor(config['model_path'],config['min_pixels'],config['max_pixels'])
    model=load_model(config['model_path'],adapter=args.adapter).to('cuda').eval()
    dataset=Dataset(config['data_dir'],args.split);collator=Collator(processor,config['max_length'])
    count=min(len(dataset),args.limit) if args.limit is not None else len(dataset)
    records=[];invalid=[];start=time.monotonic()
    with (args.output_dir/'predictions.jsonl').open('w') as f:
        for first in range(0,count,args.batch_size):
            rows=[dataset[i] for i in range(first,min(count,first+args.batch_size))]
            batch={k:v.to('cuda') for k,v in collator.encode(rows,supervised=False).items()}
            answers=generate(model,processor,batch,config['max_new_tokens'])
            for row,text in zip(rows,answers,strict=True):
                record={k:row[k] for k in ['episode_id','sample_index','kind','source_sample_index','anchor_sample_index']}
                record.update(target_pt=row['pt'],raw_response=text)
                try:record['predicted_pt']=parse_score(text);records.append(record)
                except ValueError as e:record.update(predicted_pt=None,error=str(e));invalid.append(record)
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
            f.flush()
    report=dict(status='completed' if not invalid else 'invalid_generations',samples=count,invalid_generations=len(invalid),
        invalid_ids=[(r['episode_id'],r['sample_index']) for r in invalid],split=args.split,
        adapter=str(args.adapter) if args.adapter else None,input_contract=config['input_contract'],
        uses_future_goal_reference=True,elapsed_seconds=time.monotonic()-start)
    # Partial or malformed predictions do not receive a misleading full-coverage score.
    if not invalid:report.update(summarize(records))
    (args.output_dir/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()

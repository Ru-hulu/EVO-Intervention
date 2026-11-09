import json
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator

def main():
    root=Path('/cache/cvpzhr')
    config=json.loads((root/'setup/train-4gpu.json').read_text())
    dataset=M3Dataset(json.loads((root/'setup/preflight-samples.json').read_text()))
    processor=AutoProcessor.from_pretrained(config['model_path'],max_pixels=config['max_pixels'],local_files_only=True)
    collator=Stage3Collator(processor,dataset.history_dim,config['max_sequence_length'],dataset.history_capacity)
    loader=DataLoader(Subset(dataset,[0,1,len(dataset)-2,len(dataset)-1]),batch_size=2,num_workers=config['num_workers'],
                      collate_fn=collator,multiprocessing_context='spawn',persistent_workers=True)
    batches=[{'shape':list(b['inputs']['input_ids'].shape),'target_pt':b['target_pt'].tolist()} for b in loader]
    report={'status':'PASS','workers':config['num_workers'],'batches':batches}
    (root/'setup/loader-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__':main()

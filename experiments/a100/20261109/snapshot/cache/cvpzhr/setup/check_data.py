import json
from pathlib import Path
import math
import time
from collections import Counter
import torch

root=Path('/cache/cvpzhr')
data=root/'datasets/forest-5fps'
started=time.time()
counts=Counter()
train=[]
validation=[]
common=None
for path in sorted(data.glob('*/*/*/**/samples.jsonl')):
    rel=path.relative_to(data)
    if rel.parts[0] not in ('train','validation'): continue
    directory=path.parent
    summary=json.loads((directory/'summary.json').read_text())
    signature=(summary['history_dim'],summary['history_capacity'],summary['history_source'])
    if common is None: common=signature
    assert common==signature,(directory,'inconsistent memory metadata')
    lines=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(lines)==summary['num_samples']
    previous=None
    for item in lines:
        frame=item['frame_index']
        assert previous is None or frame>previous
        assert math.isfinite(item['pt']) and 0<=item['pt']<=1
        assert set(item['images'])=={'cam_high','cam_left_wrist','cam_right_wrist'}
        for image in item['images'].values():
            assert not Path(image).is_absolute() and (directory/image).stat().st_size>0
            counts['images']+=1
        assert (item['history'] is None)==(previous is None)
        if item['history'] is not None:
            assert not Path(item['history']).is_absolute() and (directory/item['history']).stat().st_size>0
            cache=torch.load(directory/item['history'],map_location='cpu',weights_only=True,mmap=True)
            assert cache['through_frame']==previous,(directory,frame,'history must end at previous sample')
            tokens=cache['tokens']
            assert tokens.ndim==2 and tokens.shape[1]==summary['history_dim']
            assert 0<len(tokens)<=summary['history_capacity'] and not tokens.requires_grad
            del tokens,cache
            counts['histories']+=1
        previous=frame
    label='/'.join(rel.parts[:2])
    counts[label+'/episodes']+=1
    counts[label+'/samples']+=len(lines)
    (train if rel.parts[0]=='train' else validation).append(str(directory))
expected=json.loads((data/'dataset-counts.json').read_text())
for key,value in expected.items():
    dataset,split=key.split('/')
    for count in ('episodes','samples'):
        assert counts[f'{split}/{dataset}/{count}']==value[count],key
config=json.loads((root/'setup/train-4gpu.json').read_text())
assert set(config['sample_files'])==set(train)
(root/'setup/validation.sample_files.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
report={'status':'PASS','counts':dict(counts),'training_samples':sum(v for k,v in counts.items() if k.startswith('train/') and k.endswith('/samples')),
        'validation_samples':sum(v for k,v in counts.items() if k.startswith('validation/') and k.endswith('/samples')),
        'seconds':time.time()-started,'note':'All sample labels, view paths, history shapes/budgets and through_frame causality checked. History tensors memory-mapped to read metadata without scanning all tensor values; transfer archives independently SHA256-verified.'}
(root/'setup/data-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))

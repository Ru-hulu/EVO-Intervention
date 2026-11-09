"""Keep all abnormal observations and half of each normal episode per epoch."""
import math
from pathlib import Path
import torch
from torch.utils.data import Sampler
class HalfNormalSampler(Sampler):
    def __init__(self,dataset,normal_episode_dirs,rank,world_size,seed):
        normal={str(Path(p).resolve()) for p in normal_episode_dirs}
        self.normal={i:[] for i,(p,_) in enumerate(dataset.episodes) if str(p.resolve()) in normal}
        assert len(self.normal)==len(normal), 'Normal episode list does not match dataset'
        self.other=[]
        for i,(episode,_,_) in enumerate(dataset.samples):
            (self.normal[episode] if episode in self.normal else self.other).append(i)
        self.size=len(self.other)+sum(len(v)//2 for v in self.normal.values())
        self.rank,self.world_size,self.seed,self.epoch=rank,world_size,seed,0
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return math.ceil(self.size/self.world_size)
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch)
        indices=list(self.other)
        for values in self.normal.values():
            indices.extend(values[i] for i in torch.randperm(len(values),generator=g).tolist()[:len(values)//2])
        indices=[indices[i] for i in torch.randperm(len(indices),generator=g).tolist()]
        indices.extend(indices[:len(self)*self.world_size-len(indices)])
        return iter(indices[self.rank::self.world_size])

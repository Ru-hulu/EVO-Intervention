"""Use the verified SOC cache loaders unchanged; supervise only native progress."""
from functools import lru_cache
import sys
from .common import CACHE_SETUP

@lru_cache(maxsize=1)
def robometer_api():
    sys.path[:0]=[str(CACHE_SETUP/'runtime'),'/cache/cvpzhr/setup/three-model-e1-cleanval-20261108/code/robometer-upstream']
    from robometer_training import model
    return model

class Dataset:
    def __init__(self,config):
        if config['name']=='robometer-4b':
            robometer_api()
            from robometer_training.data import ProgressDataset
        else:
            from experiments.soc35_model_caches_20261114.proc_data import ProgressDataset
        self.rows=ProgressDataset(config['train_data']).rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]

class CachedCollator:
    def __init__(self,processor,kind,max_length):
        if kind=='robometer-4b':
            robometer_api()
            from robometer_training.data import ProgressCollator
        else:
            from experiments.soc35_model_caches_20261114.proc_data import ProgressCollator
        self.collator=ProgressCollator(processor,max_length)
    def __call__(self,rows):
        batch=self.collator(rows)
        batch.pop('progress_gt',None)
        return batch

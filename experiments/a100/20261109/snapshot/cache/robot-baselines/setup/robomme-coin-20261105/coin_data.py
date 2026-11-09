"""Piper dual-arm adapter. Training/inference share normalization and history assembly."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
ROOT=Path('/cache/robot-baselines')
DATA=ROOT/'data/robomme-state-coin-20261105'
RUN=ROOT/'outputs/robomme-state-coin-framesamp-20261105'
BASE=ROOT/'models/RoboMME/perceptual-framesamp-modul/79999'
REPO=ROOT/'repos/robomme_policy_learning'
SETUP=ROOT/'setup/robomme-coin-20261105'
IMAGE_KEYS=('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
DELTA=np.array([True]*6+[False]+[True]*6+[False])
def normalize(a,stats):
 return ((a-np.asarray(stats['q01']))/(np.asarray(stats['q99'])-np.asarray(stats['q01'])+1e-6)*2-1).astype(np.float32)
def absolute_actions(a,state):
 st=json.loads((DATA/'norm_stats.json').read_text())['norm_stats']['actions']
 a=(np.asarray(a)[...,:14]+1)*.5*(np.asarray(st['q99'])-np.asarray(st['q01'])+1e-6)+np.asarray(st['q01'])
 a[...,DELTA]+=np.asarray(state)[DELTA]
 return a.astype(np.float32)
class CoinDataset:
 def __init__(self,split,validation=False,features=True):
  from mme_vla_suite.training.config import PaligemmaTokenizer
  self.tokenizer=PaligemmaTokenizer(128)
  self.stats=json.loads((DATA/'norm_stats.json').read_text())['norm_stats']
  self.meta=json.loads((DATA/'dataset.json').read_text())
  self.features=features
  self.indices=[]
  ids=json.loads((DATA/'split.json').read_text())[split]
  for eid in ids:
   n=next(e['anchors'] for e in self.meta['episodes'] if e['episode']==eid)
   js=np.unique(np.linspace(0,n-1,min(32,n),dtype=int)) if validation else np.arange(n)
   self.indices.extend((eid,int(j)) for j in js)
 @lru_cache(maxsize=100)
 def episode(self,eid):
  p=DATA/f'episode_{eid:03d}'
  t=dict(np.load(p/'trajectory.npz'))
  t['images']=np.load(p/'images.npy',mmap_mode='r')
  if self.features:t['features']=np.load(p/'features.npy',mmap_mode='r')
  return t
 def __len__(self):return len(self.indices)
 def __getitem__(self,i):
  eid,j=self.indices[i]; ep=self.episode(eid); t=int(ep['anchors'][j]); state=ep['state'][t]
  act=ep['action'][np.minimum(t+np.arange(20),len(ep['action'])-1)].copy()
  act[:,DELTA]-=state[DELTA]
  actions=np.pad(normalize(act,self.stats['actions']),((0,0),(0,18)))
  ns=normalize(state,self.stats['state'])
  tokens,token_mask=self.tokenizer.tokenize(self.meta['prompt'],ns)
  hist=np.arange(j+1) if j<32 else np.linspace(0,j,32,dtype=int)
  image_emb=np.zeros((32,16,2048),np.float32); pos_emb=np.zeros((32,16,768),np.float32); mask=np.zeros(512,bool)
  if self.features:
   image_emb[:len(hist)]=ep['features'][hist]
   pos=np.load(DATA/'positions.npy',mmap_mode='r'); pos_emb[:len(hist)]=pos[ep['anchors'][hist]]
  mask[:len(hist)*16]=True
  return {'image':{k:ep['images'][j,v] for v,k in enumerate(IMAGE_KEYS)},'image_mask':{k:np.True_ for k in IMAGE_KEYS},'state':np.pad(ns,(0,18)),'actions':actions,'tokenized_prompt':tokens.astype(np.int32),'tokenized_prompt_mask':token_mask,'static_image_emb':image_emb.reshape(512,2048),'static_pos_emb':pos_emb.reshape(512,768),'static_state_emb':np.zeros((512,8),np.float32),'static_mask':mask}
def batch(dataset,indices):
 import jax
 from mme_vla_suite.models.integration.history_observation import HistAugObservation
 d=jax.tree.map(lambda *x:jax.numpy.asarray(np.stack(x)),*[dataset[int(i)] for i in indices])
 return HistAugObservation.from_dict(d),d['actions']

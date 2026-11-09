"""Wrong-target motion freezes progress; only damaged achievements lose credit."""
import math

NEAR=.12
RELEASE=.14
LIFT=.04
SHIFT=.06

def masks(geometry):
 frames=geometry['frames'];order=geometry['open_order'];active={};freeze=[];losses=[];invalid=set();closed_z={};repaired={};opened={};phase_before=None
 for f,s in enumerate(frames):
  phase=s['phase'];expected=phase if phase<3 else (order[phase-3] if phase<6 else None)
  if phase!=phase_before:active={}
  phase_before=phase
  d=s['tool_to_grasp_distance'];old=frames[max(0,f-3)]['tool_to_grasp_distance']
  for arm in range(2):
   target=min(range(3),key=lambda i:d[arm][i])
   wrong=expected is not None and target!=expected and d[arm][target]+.02<d[arm][expected]
   approaching=old[arm][target]-d[arm][target]>.002
   retreating=d[arm][target]-old[arm][target]>.002
   if arm in active:
    lid=active[arm]
    if target!=lid or not wrong or retreating or d[arm][lid]>RELEASE:active.pop(arm)
   if wrong and approaching and d[arm][target]<NEAR:active[arm]=target
  freeze.append(bool(active))
  correct_open=set(order[:max(0,phase-3)])
  for lid in range(min(phase,3)):
   cover=s['covers'][lid];block=s['blocks'][lid]
   closed_z.setdefault(lid,cover[2])
   covered=abs(cover[0]-block[0])<.03 and abs(cover[1]-block[1])<.03 and cover[2]<.742
   exposed=cover[2]-closed_z[lid]>=LIFT or math.hypot(cover[0]-block[0],cover[1]-block[1])>=SHIFT
   wrong_open=lid!=expected and lid not in correct_open
   opened[lid]=opened.get(lid,0)+1 if exposed and wrong_open else 0
   repaired[lid]=repaired.get(lid,0)+1 if covered else 0
   if opened[lid]>=3:invalid.add(f'phase:{lid+1}')
   if repaired[lid]>=3:invalid.discard(f'phase:{lid+1}')
  losses.append(set(invalid))
 return freeze,losses

def curve(deltas,weights,geometry):
 frames=geometry['frames'];n=len(frames);assert len(deltas)==n
 freeze,losses=masks(geometry)
 events=[(0,frames[0]['phase'])]
 for f,s in enumerate(frames[1:],1):
  if s['phase']!=frames[f-1]['phase']:events.append((f,s['phase']))
 base=[0.]*n;start,phase=events[0];value=sum(weights[f'phase:{i}'] for i in range(1,phase+1));base[0]=value
 for end,new_phase in events[1:]:
  assert new_phase>=phase
  gain=sum(weights[f'phase:{i}'] for i in range(phase+1,new_phase+1))
  mass=math.fsum(deltas[f] for f in range(start+1,end+1) if not freeze[f])
  if gain and mass<=0:raise ValueError('No verified correct-motion visual mass in completed stage')
  acc=0.
  for f in range(start+1,end+1):
   if not freeze[f]:acc+=deltas[f]
   base[f]=value+(gain*acc/mass if mass else 0.)
  value+=gain;start=end;phase=new_phase
 base[start+1:]=[value]*(n-start-1)
 dense=[max(0.,min(1.,p-sum(weights[g] for g in invalid))) for p,invalid in zip(base,losses)]
 output=[]
 for f in range(n):
  lost=losses[f]-(losses[f-1] if f else set());restored=(losses[f-1] if f else set())-losses[f]
  if lost or restored or (f and freeze[f]!=freeze[f-1]) or (f and frames[f]['phase']!=frames[f-1]['phase']):
   kind='regression' if lost else 'repair' if restored else 'freeze' if freeze[f] else 'resume' if f and freeze[f-1] else 'complete'
   output.append(dict(frame=f,kind=kind,progress=dense[f],rollback_ids=sorted(lost),restored_ids=sorted(restored)))
 return dense,freeze,losses,output

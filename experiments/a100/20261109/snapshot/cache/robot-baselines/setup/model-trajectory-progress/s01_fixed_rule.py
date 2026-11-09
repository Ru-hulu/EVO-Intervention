"""Apply the Memory Recover fixed cover-task rule to verified model evidence."""
import math

def build(delta,geometry,states,controls,weights):
 n=len(states);frames=geometry['frames'];order=geometry['open_order'];assert len(delta)==len(frames)==len(controls)==n
 failure=next((f for f,s in enumerate(states) if s['fail_flag']),None)
 native_success=next((f for f,s in enumerate(states) if s['success']),None)
 stop=failure if failure is not None else n
 motion=[False]*n;wrong=[False]*n;error_active=False;previous_phase=-1
 for f in range(1,n):
  s=frames[f];phase=s['phase'];target=phase if phase<3 else order[phase-3] if phase<6 else None
  motion[f]=max(abs(a-b) for a,b in zip(controls[f],controls[f-1]))>1e-4 or any(math.dist(s['tool_poses'][a][:3],frames[f-1]['tool_poses'][a][:3])>1e-4 for a in range(2))
  if phase!=previous_phase:error_active=False
  previous_phase=phase
  if target is None:continue
  old=frames[max(0,f-3)]['tool_to_grasp_distance'];d=s['tool_to_grasp_distance']
  correct=any(old[a][target]-d[a][target]>.002 and (min(range(3),key=lambda i:d[a][i])==target or d[a][target]<old[a][target] and all(d[a][i]>=old[a][i] for i in range(3) if i!=target)) for a in range(2))
  if correct:error_active=False
  for a in range(2):
   lid=min(range(3),key=lambda i:d[a][i])
   if lid!=target and d[a][lid]<.12 and d[a][lid]+.02<d[a][target] and old[a][lid]-d[a][lid]>.002 and not correct:error_active=True
  wrong[f]=error_active
 completed=[]
 for step in range(1,7):
  physical=next((f for f in range(stop) if states[f]['current_state_pointer']>=step),None)
  if physical is None:break
  end=physical+1
  if step<6:
   limit=next((f for f in range(physical+1,stop) if wrong[f]),stop)
   target=step if step<3 else order[step-3]
   for f in range(physical+1,limit):
    d=frames[f]['tool_to_grasp_distance'];old=frames[max(physical,f-3)]['tool_to_grasp_distance']
    if any(old[a][target]-d[a][target]>.002 and min(range(3),key=lambda i:d[a][i])==target for a in range(2)):
     limit=f;break
   active=[f for f in range(physical+1,limit) if motion[f] and not wrong[f]]
   if active:end=active[-1]+1
  else:
   assert native_success is not None;end=native_success+1
  completed.append(dict(step=str(step),physical_completion_frame=physical,end_frame=end))
 curve=[0.]*n;cursor=0;base=0.;stages=[]
 for item in completed:
  end=item['end_frame'];assert end>cursor
  eligible=[f for f in range(cursor,end) if not wrong[f] and motion[f]]
  # Keep the exact completion frame as an endpoint even if control targets stop changing.
  if item['step']=='6' and end-1 not in eligible:eligible.append(end-1)
  mass=math.fsum(delta[f] for f in eligible);assert mass>0,(item,'no correct-motion visual evidence')
  allowed=set(eligible);acc=0.;w=weights[item['step']]
  for f in range(cursor,end):
   if f in allowed:acc+=delta[f]
   curve[f]=base+w*acc/mass
  base+=w;curve[end-1]=base
  stages.append(dict(start_frame=cursor,end_frame=end,nowid=item['step'],physical_completion_frame=item['physical_completion_frame'],scored=True,weight=w,visual_mass=mass,excluded_frames=[f for f in range(cursor,end) if f not in allowed]))
  cursor=end
 curve[cursor:]=[base]*(n-cursor)
 if failure is not None:curve[failure:]=[0.]*(n-failure)
 assert all(b>=a-1e-10 for a,b in zip(curve[:stop],curve[1:stop]))
 if native_success is not None:
  assert failure is None and len(completed)==6
  curve[native_success:]=[1.]*(n-native_success)
  assert all(v<1 for v in curve[:native_success]),'Success reached one early'
 return dict(dense_progress=curve,normal_stages=stages,wrong_frames=[f for f in range(stop) if wrong[f]],stationary_frames=[f for f in range(stop) if not motion[f]],failure_frame=failure,first_native_success=native_success)

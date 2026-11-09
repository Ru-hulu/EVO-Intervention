from pathlib import Path
import json,subprocess,datetime
r=Path('/cache/robot-baselines');out=r/'eval-results'
report={'updated_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'root':str(r),'purpose':'Installation and closed-loop evaluation smoke tests; one episode per checkpoint, not benchmark performance measurement.','environments':{},'tests':{},'known_metadata_conflicts':[{'environments':['simplemem-libero','simplemem-rmbench','simplemem-robodojo'],'detail':'FLA 0.2.2 declares torch>=2.5. Official SimpleMemVLA deployment uses torch 2.4.1 plus its provided compatibility shim. Native kernel import and rollout tests recorded separately.'},{'environments':['imprint'],'detail':'Official lock specifies lerobot 0.3.3 at git 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5; that exact commit declares version 0.1.0. Source commit retained unchanged.'},{'environments':['robodojo'],'detail':'IsaacLab declares starlette 0.49.1 while IsaacSim requires 0.45.3; retained official RoboDojo installer pin 0.45.3. XPolicyLab websocket client requires >=14, while IsaacSim streaming metadata pins 12; websocket 16.1.1 retained for the benchmark client. Real runtime validation recorded separately.'}],'compatibility_changes':['RoboDojo src/eval_client/main.py: ArgumentParser(allow_abbrev=False) prevents IsaacLab preliminary parse from confusing --device with --device_id. Original preserved in setup/robodojo-main.original.py.','Applied upstream RMBench SAPIEN URDF UTF-8 and .srdf suffix fixes and MPLib planner compatibility edits inside isolated simulator environments.','Generated machine-specific robot asset configs with official helpers; RoboDojo Robots runtime copy is under runtime-assets, preserving downloaded original assets.'],'graphics':{'driver':'580.126.20','user_libraries':'system-libs/nvidia','host_driver_changed':False,'scope':'source setup/graphics.sh for per-process NVIDIA EGL/Vulkan library paths'},'preflight':{}}
packages=['torch','torchvision','transformers','numpy','triton','sapien','mujoco','robosuite','flash-attn','flash-linear-attention','causal-conv1d','nvidia-curobo','pytorch3d','jax','jaxlib','diffusers','isaacsim','isaaclab','lerobot']
for name in ['simplemem-libero','simplemem-rmbench','simplemem-robodojo','rmbench','robodojo','imprint']:
 p=r/'envs'/name/'bin/python'
 code='import importlib.metadata as m,json,platform\nnames='+repr(packages)+'\nd={}\nfor n in names:\n try:d[n]=m.version(n)\n except m.PackageNotFoundError:pass\nprint(json.dumps(dict(python=platform.python_version(),packages=d)))'
 report['environments'][name]={'python_path':str(p),**json.loads(subprocess.check_output([str(p),'-c',code],text=True))}
 log=r/'logs'/f'pip-check-{name}.log'
 check=subprocess.run(['/home/ma-user/.local/bin/uv','pip','check','--python',str(p)],capture_output=True,text=True)
 log.write_text(check.stdout+check.stderr)
 report['environments'][name]['metadata_check_exit_code']=check.returncode
 report['environments'][name]['metadata_check_log']=str(log)
 frozen=subprocess.check_output(['/home/ma-user/.local/bin/uv','pip','freeze','--python',str(p)],text=True,stderr=subprocess.DEVNULL)
 (r/'manifests'/f'installed-{name}.txt').write_text(frozen)
for name,relative,marker in [('LIBERO','logs/libero-preflight.log','LIBERO_RENDER_STEP_OK'),('SAPIEN','logs/sapien-preflight.log','SAPIEN_RENDER_STEP_OK'),('IsaacSim','logs/isaac-preflight.log','ISAAC_RENDER_STEP_OK'),('RMBench_official_task','logs/rmbench-official-task.log','OFFICIAL_RMBENCH_TASK_ACTION_OK')]:
 p=r/relative;report['preflight'][name]={'passed':marker in p.read_text(errors='replace'),'log':str(p)}
for name in ['simplemem-libero','simplemem-rmbench']:
 p=out/name/'videos/results.json'
 report['tests'][name]={'state':'complete' if p.exists() else 'pending','results':json.loads(p.read_text()) if p.exists() else None,'path':str(out/name)}
for name in ['imprint-full_attention','imprint-recurrent']:
 p=out/name/'summary.json'
 report['tests'][name]={'state':'complete' if p.exists() else 'pending','results':json.loads(p.read_text()) if p.exists() else None,'path':str(out/name)}
robodojo_results=list((r/'repos/RoboDojo/eval_result').rglob('dependency-smoke-20261104/_result.json'))
assert len(robodojo_results)==1, robodojo_results
p=robodojo_results[0]
report['tests']['simplemem-robodojo']={'path':str(out/'simplemem-robodojo'),'state':'complete','results':json.loads(p.read_text()),'official_result_path':str(p),'video_directory':str(p.parent),'policy_gpu':2,'simulator_gpu':3}
report['all_checkpoint_rollouts_completed']=all(x['state']=='complete' for x in report['tests'].values())
report['launcher_notes']=['The first IMPRINT full_attention shell wrapper was updated after starting its evaluator; after evaluator completion, Bash re-read the changed trailing bytes and emitted an EOF quoting error. The evaluator itself completed all 250 steps and saved summary.json, trials.jsonl and all videos. The current saved wrapper passes bash -n.','RoboDojo policy.log contains WebSocket handshake exceptions caused by the TCP readiness probe, before the simulator connected. Actual simulator client completed successfully. Future wrapper uses a WebSocket readiness probe.']
report['concurrent_gpu_workload_observed']='FOREST prepare_pt.py label-worker processes were using all four GPUs during validation; timing is not an exclusive-GPU benchmark.'
assert report['all_checkpoint_rollouts_completed']

(out/'validation-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps({k:v['state'] for k,v in report['tests'].items()}))

"""Simulator interpreter shim, scoped to the new collection only."""
import os,sys,runpy,json,time
from pathlib import Path
sys.path.insert(0,os.getcwd())
def main():
 if '-c' in sys.argv:
  i=sys.argv.index('-c');exec(compile(sys.argv[i+1],'<preflight>','exec'));sys.exit(0)
 sys.path.insert(0,os.getcwd())
 sys.argv=[a for a in sys.argv if a!='-u']
 script=Path(sys.argv[1]);sys.argv=sys.argv[1:]
 if script.name=='eval_policy_four_view.py':
  setup = Path(__file__).resolve().parent
  gate = setup / 'seed-handoff-request'
  if gate.exists():
   ready = setup / 'seed-handoff-ready.json'
   ready.write_text(json.dumps({'collector': os.getppid(), 'sim': os.getpid()}))
   while gate.exists():
    time.sleep(1)
  import capture
  capture.install_base()
  # Use the stock evaluator but replace only auxiliary video output.
  source=script.read_text().replace('task_env.cameras.get_third_view_rgb()', 'None')
  source=source.replace('            recorder.start()','            recorder.start()\n            import capture\n            capture.start(task_env, output / "evidence" / str(now_id), now_seed, 20)')
  source=source.replace('                        task_env.close_env(', '                        capture.close()\n                        task_env.close_env(')
  # The passive collector owns three videos; suppress the redundant five-view writer.
  source=source.replace('    recorder.write(observation, third_person)','    pass')
  source=source.replace('            recorder = FourViewRecorder(output, now_id, writer_factory)','            class NullRecorder:\n                def start(self): pass\n                def close(self): pass\n            recorder = NullRecorder()')
  exec(compile(source,str(script),'exec'),{'__name__':'__main__','__file__':str(script),'__package__':None})
 else:runpy.run_path(str(script),run_name='__main__')
if __name__=='__main__':main()

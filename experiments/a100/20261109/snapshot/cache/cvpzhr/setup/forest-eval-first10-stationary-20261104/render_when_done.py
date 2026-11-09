from pathlib import Path
import datetime,json,os,subprocess,sys,time
root=Path(__file__).resolve().parent
def write(value):
    p=root/"render-status.json";t=p.with_suffix(".tmp")
    t.write_text(json.dumps(value,indent=2)+"\n");t.replace(p)
write(dict(phase="waiting_for_evaluation",pid=os.getpid()))
try:
    while True:
        state=json.loads((root/"status.json").read_text())
        if state["phase"]=="completed":break
        if state["phase"]=="failed":raise RuntimeError(state.get("error"))
        os.kill(state["pid"],0)
        time.sleep(20)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",MPLBACKEND="Agg")
    subprocess.run([sys.executable,str(root/"render_results.py")],env=env,check=True)
    write(dict(phase="completed",completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
except BaseException as exc:
    write(dict(phase="failed",error=repr(exc)))
    raise

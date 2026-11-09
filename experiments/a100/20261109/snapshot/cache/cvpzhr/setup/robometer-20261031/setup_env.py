from pathlib import Path
import json,os,shutil,subprocess,tarfile,urllib.request,traceback
ROOT=Path('/cache/cvpzhr'); RUN=ROOT/'setup/robometer-20261031'; ENV=ROOT/'envs/robometer'; CODE=ROOT/'code/robometer-proprimem-20261031'; REV='352d160389daa964788de1ec933d1925f3a6de4f'
RUN.mkdir(parents=True,exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PIP_CACHE_DIR=str(RUN/'pip-cache'),OMP_NUM_THREADS='2')
def status(phase,**kw):
 (RUN/'environment-status.json').write_text(json.dumps(dict(phase=phase,pid=os.getpid(),**kw),indent=2))
try:
 status('copying_isolated_runtime')
 if not (ENV/'bin/python').exists(): subprocess.run([str(ROOT/'envs/python-3.12/bin/python3.12'),'-m','venv',str(ENV)],check=True)
 if not (ENV/'runtime-copy-complete').exists():
  shutil.copytree(ROOT/'envs/forest/lib/python3.12/site-packages',ENV/'lib/python3.12/site-packages',dirs_exist_ok=True,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
  (ENV/'runtime-copy-complete').write_text('Copied independent runtime from forest; no source changes.\n')
 status('installing_dependencies')
 subprocess.run([str(ENV/'bin/python'),'-m','pip','install','--index-url','https://pypi.tuna.tsinghua.edu.cn/simple','-r',str(CODE/'robometer_training/requirements.txt')],check=True)
 status('fetching_pinned_source')
 source=CODE/'third_party'/('robometer-'+REV)
 if not source.exists():
  archive=RUN/'upstream.tar.gz';urllib.request.urlretrieve('https://codeload.github.com/robometer/robometer/tar.gz/'+REV,archive)
  with tarfile.open(archive) as t:
   members=[m for m in t.getmembers() if ('/'+ 'robometer'+'/') in m.name and m.name.startswith('robometer-'+REV+'/')]
   t.extractall(CODE/'third_party',members=members,filter='data')
 env={**os.environ,'PYTHONPATH':str(source)}
 subprocess.run([str(ENV/'bin/python'),'-c','import torch,transformers,peft;from robometer.models.rbm import RBM; print(torch.__version__,transformers.__version__,peft.__version__)'],env=env,check=True)
 freeze=subprocess.check_output([str(ENV/'bin/python'),'-m','pip','freeze'],text=True)
 (RUN/'environment.lock.txt').write_text(freeze)
 checked=subprocess.run([str(ENV/'bin/python'),'-m','pip','check'],capture_output=True,text=True)
 (RUN/'pip-check.txt').write_text(checked.stdout+checked.stderr)
 status('ready',environment=str(ENV),source=str(source),pip_check_returncode=checked.returncode)
except Exception as e:
 status('failed',error=str(e));traceback.print_exc();raise

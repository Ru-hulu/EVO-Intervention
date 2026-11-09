import pathlib,time,json,os,signal,subprocess
s=pathlib.Path('/cache/robot-baselines/setup/imprint-extra-20261105');gate=s/'pause-full-for-allocation'
while not (s/'full-paused.json').exists():time.sleep(2)
x=json.loads((s/'full-paused.json').read_text());pid=x['pid']
cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'imprint-extra-20261105/worker.py' in cmd
children=[]
for line in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]:
 fields=line.split(None,2)
 if len(fields)==3 and fields[1]==str(pid) and 'serve_robomme_tcp.py' in fields[2]:children.append(int(fields[0]))
os.kill(pid,signal.SIGTERM)
for child in children:os.kill(child,signal.SIGTERM)
time.sleep(2);gate.unlink();(s/'full-paused.json').rename(s/'full-reallocated.json')
log=open('/cache/robot-baselines/logs/imprint-extra-full_attention.log','a')
p=subprocess.Popen(['bash',str(s/'launch.sh'),'full_attention','1'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(s/'allocation.json').write_text(json.dumps(dict(full_attention={'Huawei':{'S03':10,'S05':10},'4090C':{'S03':10},'4090D':{'S05':10}},recurrent={'Huawei':{'S03':20,'S05':20}},new_pid=p.pid,total=80),indent=2))

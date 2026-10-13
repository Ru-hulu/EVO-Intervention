"""One-shot completion collector; no recurring automation or training restart."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

from qwen_training.closed_loop_rollout import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-directory',required=True)
    parser.add_argument('--local-directory',type=Path,required=True)
    args=parser.parse_args()
    local=args.local_directory.resolve();local.mkdir(exist_ok=True,parents=True)
    connection=['-o','ControlPath=/tmp/evo-resume-a100-%C','-o','BatchMode=yes','-o','ConnectTimeout=15']
    host='root@36.212.230.67'
    script=f"from pathlib import Path; p=Path({args.remote_directory!r})/'result.json'; print(p.read_text() if p.exists() else '{{\"status\":\"running\"}}')"
    command=['ssh',*connection,host,'python3 -c '+shlex.quote(script)]
    errors=0
    try:
        while True:
            check=subprocess.run(command,capture_output=True,text=True,timeout=35)
            if check.returncode:
                errors+=1
                save_json(local/'delivery.json',dict(status='connection_retry',attempt=errors,error=check.stderr.strip(),updated_unix=time.time()))
                if errors>=5:raise RuntimeError('SSH unavailable after five retries')
                time.sleep(60);continue
            errors=0;status=json.loads(check.stdout)
            if status['status']=='failed':raise RuntimeError('Remote inference failed; partial artifacts preserved')
            if status['status']=='complete':break
            save_json(local/'delivery.json',dict(status='waiting_for_inference',updated_unix=time.time()))
            time.sleep(60)
        save_json(local/'delivery.json',dict(status='copying_artifacts',updated_unix=time.time()))
        if Path(args.remote_directory).name!=local.name:raise ValueError('Local/remote basenames must match')
        # Predictions are repetitive JSON. Stream gzip once instead of transferring
        # hundreds of uncompressed files over separate SFTP requests.
        remote=Path(args.remote_directory)
        archive_command=shlex.join(['tar','-C',str(remote.parent),'-czf','-',remote.name])
        with (local/'transfer.log').open('w') as transfer_log:
            source=subprocess.Popen(['ssh',*connection,host,archive_command],stdout=subprocess.PIPE,stderr=transfer_log)
            try:
                with tarfile.open(fileobj=source.stdout,mode='r|gz') as archive:
                    for member in archive:
                        parts=Path(member.name).parts
                        if not parts or parts[0]!=local.name or '..' in parts or not (member.isfile() or member.isdir()):
                            raise ValueError('Unexpected archive member')
                        archive.extract(member,path=local.parent,filter='data')
                source.stdout.close()
                if source.wait()!=0:raise RuntimeError('Artifact transfer failed')
            except BaseException:
                source.terminate();source.wait();raise
        save_json(local/'delivery.json',dict(status='rendering_reports',updated_unix=time.time()))
        with (local/'report-generation.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','qwen_training.validation_rollout_report',str(local)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        save_json(local/'delivery.json',dict(status='complete',report=str(local/'report.html'),finished_unix=time.time()))
        print(json.dumps(dict(status='complete',report=str(local/'report.html'))),flush=True)
    except Exception as error:
        save_json(local/'delivery.json',dict(status='failed',error=repr(error),finished_unix=time.time()))
        raise


if __name__=='__main__':main()

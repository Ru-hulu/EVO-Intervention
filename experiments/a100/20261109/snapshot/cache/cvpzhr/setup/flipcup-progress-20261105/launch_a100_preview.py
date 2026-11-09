"""One-off dependency chain: finish this model download, then compute only two samples."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path('/cache/cvpzhr/setup/flipcup-progress-20261105')
MODEL=Path('/cache/cvpzhr/models/Qwen3-VL-4B-Instruct')
DOWNLOAD_PID=821421
EXPECTED_SHARD_SHA256='046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6'


def main():
    started=time.monotonic()
    while not (MODEL/'download_manifest.json').exists():
        process=Path('/proc',str(DOWNLOAD_PID),'cmdline')
        if not process.exists() or b'download_model.py' not in process.read_bytes().split(b'\0'):
            raise RuntimeError('Model download ended without a verified manifest; see download.log')
        if time.monotonic()-started>3600:
            raise TimeoutError('Model download exceeded the one-off one-hour wait')
        time.sleep(5)
    manifest=json.loads((MODEL/'download_manifest.json').read_text())
    if manifest['downloaded_files']['model-00002-of-00002.safetensors']['sha256']!=EXPECTED_SHARD_SHA256:
        raise RuntimeError('Downloaded shard does not match the original Hugging Face LFS SHA256')
    command=[sys.executable,'-u',str(ROOT/'run_delta_job.py'),
             '--dataset-root','/cache/cvpzhr/datasets/flipcup-recovery-progress-20261105',
             '--model-dir',str(MODEL),'--gpu','3','--memory-fraction','0.08',
             '--episodes','episode_000000','episode_000001']
    print('Verified original model weights; launching exactly two episodes',flush=True)
    subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()

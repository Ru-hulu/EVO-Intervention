"""Fetch only files used by the original Qwen3-VL visual delta encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'pydeps'))
REPO = 'Qwen/Qwen3-VL-4B-Instruct'
REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from huggingface_hub import hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'model')
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    files = ['config.json', 'preprocessor_config.json', 'model.safetensors.index.json']
    def fetch(name):
        print(f'Downloading/verifying {name}', flush=True)
        return Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=args.model_dir))
    for name in files:
        fetch(name)
    index = json.loads((args.model_dir / files[-1]).read_text())
    shards = sorted({shard for key, shard in index['weight_map'].items()
                     if key.startswith('model.visual.')})
    if not shards:
        raise RuntimeError('No model.visual.* weights in the pinned checkpoint')
    for name in shards:
        fetch(name)
    manifest = {'model_repo': REPO, 'revision': REVISION,
                'downloaded_files': {name: {'bytes': (args.model_dir / name).stat().st_size,
                                            'sha256': digest(args.model_dir / name)}
                                     for name in files + shards}}
    (args.model_dir / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()

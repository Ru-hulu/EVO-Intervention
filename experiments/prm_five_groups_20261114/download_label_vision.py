"""Remote-only restore of the original Recover label encoder's vision shards."""
import json
from pathlib import Path
import time
from urllib.request import urlopen
from mem_lite.memory.download_clip import verify
from experiments.prm_five_groups_20261114.prepare_cache import BASE, save

REPO = 'Qwen/Qwen3-VL-4B-Instruct'
REV = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
OUT = Path('/cache/cvpzhr/models/Qwen3-VL-4B-Instruct')


def main():
    info = json.loads((BASE / 'qwen3vl-official-metadata.json').read_text())
    assert info['id'] == REPO and info['sha'] == REV
    rows = {x['rfilename']: x for x in info['siblings']}
    OUT.mkdir(exist_ok=True, parents=True)
    receipts = []

    def fetch(name):
        dest = OUT / name
        item = rows[name]
        if verify(dest, item):
            receipts.append({'name': name, 'identity': item, 'reused': True})
            return
        if dest.exists():
            raise ValueError(f'Existing final file fails verification: {dest}')
        urls = [f'https://modelscope.cn/api/v1/models/{REPO}/repo?Revision=master&FilePath={name}',
                f'https://hf-mirror.com/{REPO}/resolve/{REV}/{name}?download=true',
                f'https://hf-mirror.com/{REPO}/raw/{REV}/{name}']
        for attempt in range(6):
            for url in urls:
                try:
                    part = dest.with_suffix(dest.suffix + '.part')
                    with urlopen(url, timeout=45) as src, part.open('wb') as target:
                        for block in iter(lambda: src.read(8*1024*1024), b''):
                            target.write(block)
                    if not verify(part, item):
                        raise ValueError('Pinned Hugging Face hash mismatch')
                    part.replace(dest)
                    receipts.append({'name': name, 'identity': item, 'source': url})
                    print('Verified', name, item['size'], flush=True)
                    return
                except Exception as error:
                    print('retry', name, attempt, type(error).__name__, flush=True)
            time.sleep(5)
        raise RuntimeError(f'Cannot download {name}')

    for name in ['config.json', 'preprocessor_config.json', 'model.safetensors.index.json']:
        fetch(name)
    index = json.loads((OUT/'model.safetensors.index.json').read_text())['weight_map']
    for name in sorted({v for k,v in index.items() if k.startswith('model.visual.')}):
        fetch(name)
    save(OUT/'vision-download-receipt.json', {'repo':REPO,'revision':REV,'files':receipts,
         'scope':'vision shards only, not a complete language-model installation',
         'model_bytes_transit_local_computer':False})


if __name__ == '__main__':
    main()

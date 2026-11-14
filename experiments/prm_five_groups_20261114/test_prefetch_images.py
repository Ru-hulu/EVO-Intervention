import json
import os
from pathlib import Path
import pytest
from experiments.prm_five_groups_20261114.prefetch_images import publish,link_main
from experiments.prm_five_groups_20261114.prepare_cache import sha


def test_publish_never_overwrites_even_empty_consumer_directory(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();(stage/'image').write_bytes(b'verified')
    target=tmp_path/'target';target.mkdir()
    assert not publish(stage,target) and stage.exists() and not list(target.iterdir())
    (target/'existing').write_bytes(b'live consumer')
    assert not publish(stage,target) and (target/'existing').read_bytes()==b'live consumer'
    dest=tmp_path/'ready'
    assert publish(stage,dest) and (dest/'image').read_bytes()==b'verified' and not stage.exists()


def test_main_view_reuses_verified_exact_png_bytes(tmp_path):
    full=tmp_path/'fps5';full.mkdir()
    episode={'video_hashes':['v0','v1','v2'],'frame_indices':[0,6],'frames':7}
    hashes={}
    for index in [0,6]:
        name=f'{index:08d}-0.png';(full/name).write_bytes(b'bytes'+str(index).encode());hashes[name]=sha(full/name)
    (full/'verification.json').write_text(json.dumps({'input':{'video_hashes':episode['video_hashes'],'frame_indices':[0,6]},'decoded_native_frames':7,'png_sha256':hashes}))
    stage=tmp_path/'main';link_main(episode,full,stage)
    receipt=json.loads((stage/'verification.json').read_text())
    assert receipt['png_sha256']==hashes and receipt['input']['video_hashes']==['v0']
    assert os.stat(stage/'00000000-0.png').st_ino==os.stat(full/'00000000-0.png').st_ino
    (full/'00000000-0.png').write_bytes(b'corrupt')
    with pytest.raises(AssertionError):link_main(episode,full,tmp_path/'reject')

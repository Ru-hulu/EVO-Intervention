from copy import deepcopy
import json

import pytest

from experiments.prm_five_groups_20261114 import eval_common, stream_images
from experiments.prm_five_groups_20261114.prepare_cache import sha
from experiments.prm_five_groups_20261114.stream_manifests import fps_map, playback_map


@pytest.fixture
def source(tmp_path):
    import av
    import numpy as np
    videos = []
    for view in range(3):
        path = tmp_path/f'view{view}.mkv'
        with av.open(str(path), 'w') as out:
            stream = out.add_stream('ffv1', rate=30)
            stream.width = stream.height = 16
            stream.pix_fmt = 'bgr0'
            for i in range(121):
                pixels = np.zeros((16, 16, 3), dtype=np.uint8)
                pixels[:, :, 0] = (i*2) % 256
                pixels[:, :, 1] = view*70
                pixels[:, :, 2] = 255-i
                frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        videos.append(str(path))
    return {'key': 'Count/C01/test', 'suite': 'Count', 'task_id': 'C01',
            'frames': 121, 'source_rate': '30/1', 'frame_indices': list(range(0, 121, 6)),
            'timestamps': [i/30 for i in range(0, 121, 6)],
            'target_pt': [i/120 for i in range(0, 121, 6)],
            'progress_sha256': 'labels', 'input_issues': [],
            'task_prompt': 'test', 'videos': videos, 'video_hashes': [sha(p) for p in videos]}


def fps_entry(row, fps):
    return fps_map(row, {i: i/120 for i in range(121)}, fps)


def test_all_rates_retain_pinned_decoder_pixels_and_native_source_indices(source, tmp_path):
    cache = tmp_path/'cache'
    original_root = eval_common.IMAGE_ROOT
    full = stream_images.prepare(source, fps_entry(source, 30), cache_root=cache)
    native = json.loads((full/'verification.json').read_text())
    for fps in (20, 10, 5, 2):
        entry = fps_entry(source, fps)
        path = stream_images.prepare(source, entry, cache_root=cache)
        proof = json.loads((path/'verification.json').read_text())
        assert proof['input']['frame_indices'] == entry['source_frame_indices']
        assert all(h == native['png_sha256'][name] for name, h in proof['png_sha256'].items())
        assert proof['decoded_native_frames'] == 121
        assert stream_images.prepare(source, entry, cache_root=cache) == path
    assert eval_common.IMAGE_ROOT == original_root
    assert not list((cache/'.stream-staging').glob('*/failure.json'))


@pytest.mark.parametrize('variant', ['no_memory', 'static_only', 'rewind_only'])
def test_perturbations_reuse_unique_baseline_images_without_resetting_map(source, tmp_path, variant):
    entry = playback_map(source, variant, None if variant == 'no_memory' else 42)
    snapshot = deepcopy(entry)
    root = stream_images.prepare(source, entry, main_only=True, cache_root=tmp_path/'cache')
    proof = json.loads((root/'verification.json').read_text())
    assert root.name == 'fps5-main'
    assert proof['input']['frame_indices'] == source['frame_indices']
    assert len(proof['png_sha256']) == len(source['frame_indices'])
    assert entry == snapshot


def test_main_only_keeps_short_wrist_eligible_but_full_views_remain_blocked(source, tmp_path):
    source['input_issues'] = [{'view': 1, 'reason': 'view_frame_count_mismatch'}]
    entry = fps_entry(source, 20)
    path = stream_images.prepare(source, entry, main_only=True, cache_root=tmp_path/'cache')
    proof = json.loads((path/'verification.json').read_text())
    assert all(name.endswith('-0.png') for name in proof['png_sha256'])
    with pytest.raises(ValueError):
        stream_images.prepare(source, entry, cache_root=tmp_path/'cache')


def test_corrupt_existing_cache_is_rejected_without_overwriting(source, tmp_path):
    cache = tmp_path/'cache'
    entry = fps_entry(source, 20)
    path = stream_images.prepare(source, entry, main_only=True, cache_root=cache)
    frame = path/'00000000-0.png'
    frame.write_bytes(b'corrupted consumer cache')
    with pytest.raises(AssertionError):
        stream_images.prepare(source, entry, main_only=True, cache_root=cache)
    assert frame.read_bytes() == b'corrupted consumer cache'


def test_lost_publish_race_never_replaces_empty_consumer_directory(source, tmp_path, monkeypatch):
    real_publish = stream_images.publish
    def race(staged, destination):
        destination.mkdir(parents=True)
        return real_publish(staged, destination)
    monkeypatch.setattr(stream_images, 'publish', race)
    cache = tmp_path/'cache'
    original_root = eval_common.IMAGE_ROOT
    with pytest.raises(FileNotFoundError):
        stream_images.prepare(source, fps_entry(source, 20), cache_root=cache)
    assert eval_common.IMAGE_ROOT == original_root
    destinations = list((cache/'mapped-native').iterdir())
    assert len(destinations) == 1 and list(destinations[0].iterdir()) == []
    assert len(list((cache/'.stream-staging').glob('*/failure.json'))) == 1

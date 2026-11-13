"""Reuse only immutable clean image/PT observations, never trained model outputs."""
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import *


def literal(path, variable):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(variable)


def main():
    old = read(PREVIOUS / 'run.json')
    brain = read('/cache/cvpzhr/datasets/robobrain25-forward-cap50-static-jitter-20261105/episodes.json')
    prompts, caches, targets = {}, {}, {}
    for f in PREVIOUS.glob('robometer-*.jsonl'):
        for line in f.open():
            r = json.loads(line)
            prompts[r['episode_id']] = r['task']
            caches[r['episode_id']] = r['pixel_cache']
            targets[r['episode_id'], r['sample_index']] = (r['pt'], r['source_frame_index'])
    missing = Counter()
    files = set()
    episodes = []
    for e in old['episodes']:
        eid, d = e['episode_id'], Path(e['source_dir'])
        rows = [json.loads(l) for l in (d / 'samples.jsonl').read_text().splitlines() if l.strip()]
        assert len(rows) == e['samples'] == brain[eid]['native_samples']
        assert prompts[eid] == brain[eid]['task_prompt']
        frames = []
        for i, r in enumerate(rows):
            assert targets[eid, i] == (r['pt'], r['frame_index'])
            image = [str(d / r['images'][v]) if r['images'].get(v) else None for v in ['cam_high','cam_left_wrist','cam_right_wrist']]
            assert image[0]
            for view, p in zip(['main','left','right'], image):
                if p: files.add(p)
                else: missing[e['suite'] + '/' + view] += 1
            assert brain[eid]['native_images'][i] == [r['images'].get(v) for v in ['cam_high','cam_left_wrist','cam_right_wrist']]
            frames.append(dict(frame_index=r['frame_index'], pt=r['pt'], image=image))
        episodes.append(dict(episode_id=eid, suite=e['suite'], task_id=e['task'], task=prompts[eid],
                             source_dir=str(d), source_sha256=sha(d / 'samples.jsonl'), frames=frames,
                             pixel_cache=caches[eid]))
    with ThreadPoolExecutor(32) as pool:
        assert all(pool.map(lambda p: Path(p).is_file() and Path(p).stat().st_size > 0, files))
    loads = [0] * 4
    for e in sorted(episodes, key=lambda e: (-len(e['frames']), e['episode_id'])):
        shard = min(range(4), key=lambda j: loads[j]); e['shard'] = shard; loads[shard] += len(e['frames'])
    assert len(episodes) == 417 and sum(loads) == 66891
    suite_counts = {s:dict(episodes=sum(e['suite']==s for e in episodes), samples=sum(len(e['frames']) for e in episodes if e['suite']==s)) for s in ['State','Order','Count']}
    assert suite_counts == {'State':{'episodes':166,'samples':27036}, 'Order':{'episodes':111,'samples':23053}, 'Count':{'episodes':140,'samples':16802}}
    protocol = dict(models=MODELS, fps=5, augmentation='none', validation_manifest=str(PREVIOUS / 'run.json'),
        zero_shot='official released checkpoint only, no local fine-tuned adapters or heads',
        metrics=['Pearson(predicted_pt,target_pt)', 'Spearman(predicted_pt,target_pt)'],
        aggregation='per episode, then arithmetic mean; State/Order/Count and all episodes',
        invalid_policy='retain raw response and null; omit affected episodes with explicit counts; never impute or silently claim full valid coverage',
        constants='undefined, report counts', smoothing=False, monotonic_projection=False,
        generation=dict(temperature=0, seed=42, precision='bfloat16'),
        missing_views='existing protocol: black RGB 64x64 placeholder; missing sensor counts retained',
        inputs=dict(robometer='uniform observed prefix8 np.rint incl first/current; rightpad current; seven main+current three, 8 progress tokens',
                    grm='reference first/last main + BEFORE first three + AFTER current three; forward; goal image is actual final cached observation',
                    proc='recent8 main, leftpad first; official procedural_c text; no value head',
                    vlac='previous/current main; no reference demo; native critic_to_value_simple(simple=True); initial value0',
                    r2='new current main+previous generated think text+task; reset each episode; no annotation input'),
        expected=suite_counts)
    for name, data in [('manifest.json',dict(episodes=episodes, counts=suite_counts, missing_views=dict(missing), shard_samples=loads)),('protocol.json',protocol)]:
        path=ROOT / name
        if path.exists(): assert read(path)==data, 'Existing immutable protocol differs: '+name
        else: write(path,data)
    repo=Path('/cache/robot-baselines/repos')
    (ROOT / 'grm_prompt.txt').write_text(literal(repo/'Robo-Dopamine/examples/inference.py','SYSTEM_PROMPT'))
    (ROOT / 'r2_prompt.txt').write_text(literal(repo/'R2VLM/code/script/Alfred/infer_vllm_single.py','PROMPT'))
    (ROOT / 'proc_prompt.txt').write_text((repo/'ProcVLM/evqa/templates/procedural_c.txt').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    write(ROOT/'preflight.json', dict(status='passed', episodes=len(episodes), samples=sum(loads), counts=suite_counts,
                                     missing_views=dict(missing), checked_images=len(files), manifest_sha256=sha(ROOT/'manifest.json'),
                                     protocol_sha256=sha(ROOT/'protocol.json'), checked_at=now()))
    print(json.dumps(read(ROOT/'preflight.json')))


if __name__=='__main__': main()

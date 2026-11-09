from pathlib import Path
import ast, copy, hashlib, json, math, shutil

OLD = Path('/cache/cvpzhr/setup/forest-robometer-spray-official-e3-20261109')
ROOT = Path('/cache/cvpzhr/setup/forest-robometer-drawer-official-e3-20261109')
FD = Path('/cache/cvpzhr/datasets/forest-drawer-5fps-no-augmentation-20261109')
RD = Path('/cache/cvpzhr/datasets/robometer-drawer-5fps-h7-current3-clean-20261109')
FOUT = Path('/cache/cvpzhr/outputs/forest-drawer-frombase-matched-e3-20261109')
ROUT = Path('/cache/cvpzhr/outputs/robometer-drawer-fromofficial-clean-2gpu-e3-20261109')
FEVAL = Path('/cache/cvpzhr/evaluations/forest-drawer-frombase-matched-e3-cleanval-20261109')
TASK = 'drawer-inspection-close'
NTRAIN, NVAL, STEPS = 3138, 719, 175
read = lambda p: json.loads(Path(p).read_text())
def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

assert read(OLD/'status.json')['phase'] == 'completed'
assert not any(p.exists() for p in [ROOT, FOUT, ROUT, FEVAL])
plan = read(FD/'plan.json')
ready = read(FD/'cache-ready.json')
assert ready['counts'] == {'train': {'episodes': 21, 'samples': NTRAIN}, 'validation': {'episodes': 6, 'samples': NVAL}}
ROOT.mkdir()
replacements = {
    str(OLD): str(ROOT),
    '/cache/cvpzhr/outputs/forest-spray-frombase-matched-e3-20261109': str(FOUT),
    '/cache/cvpzhr/outputs/robometer-spray-fromofficial-clean-2gpu-e3-20261109': str(ROUT),
    '/cache/cvpzhr/evaluations/forest-spray-frombase-matched-e3-cleanval-20261109': str(FEVAL),
    '/cache/cvpzhr/datasets/forest-spray-5fps-no-augmentation-20261109': str(FD),
    '1919': str(NVAL), '1,919': '719', '7416': str(NTRAIN),
    '1236': str(STEPS*3), '412*i': '175*i',
    "['warmup_steps']==38": "['warmup_steps']==16",
    "('train',3138,25)": "('train',3138,21)",
    "('validation',719,7)": "('validation',719,6)",
    "'episodes':7": "'episodes':6",
    "['episodes']==7": "['episodes']==6",
    "len(records)==7": "len(records)==6",
    "len(episodes)==7": "len(episodes)==6",
    "len(by)==7": "len(by)==6",
    "len(bundle['episodes'])==7": "len(bundle['episodes'])==6",
    "for e in episodes)/7,": "for e in episodes)/6,",
    '7 条': '6 条',
    'forest-spray-validation-curves.zip': 'forest-drawer-validation-curves.zip',
    'spray-inspection-task': 'drawer-inspection-task', '喷瓶检查': '抽屉检查',
}

def adapt(p, dest):
    text = p.read_text()
    for a, b in replacements.items(): text = text.replace(a, b)
    if dest.name == 'plot_validation.py':
        text = text.replace("fig,axes=plt.subplots(4,2,figsize=(15.5,14.8))", "fig=plt.figure(figsize=(15.5,13.5))\ngrid=fig.add_gridspec(4,2,height_ratios=[1,1,1,.8])\naxes=np.array([fig.add_subplot(grid[i//2,i%2]) for i in range(6)])")
        text = text.replace("summary=axes.flat[-1];summary.axis('off')", "summary=fig.add_subplot(grid[3,:]);summary.axis('off')")
        text = text.replace('summary.text(.02,.76,', 'summary.text(.02,.68,')
        text = text.replace('summary.text(.02,.34,', 'summary.text(.55,.98,')
    if dest.name == 'plot_validation_curves.py':
        text = text.replace("fig,axes=plt.subplots(4,2,figsize=(14,13),dpi=130)", "fig=plt.figure(figsize=(14,13),dpi=130)\n    grid=fig.add_gridspec(4,2,height_ratios=[1,1,1,.8])\n    axes=np.array([fig.add_subplot(grid[i//2,i%2]) for i in range(6)])")
        text = text.replace("last=axes.flat[-1];last.axis('off')", "last=fig.add_subplot(grid[3,:]);last.axis('off')")
        text = text.replace('last.text(.04,.77,', 'last.text(.04,.68,')
        text = text.replace('last.text(.04,.59,', 'last.text(.55,.96,')
    ast.parse(text, filename=str(dest))
    dest.write_text(text)

files = {
    'forest': ['evaluate_final.py', 'plot_validation.py', 'verify_checkpoints.py'],
    'robometer': ['train_official.py', 'verify_robometer.py', 'postprocess_validation.py', 'plot_validation_curves.py'],
}
for name, scripts in files.items():
    dest = ROOT/name
    dest.mkdir()
    shutil.copytree(OLD/name/'code', dest/'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    hashes = {}
    for p in sorted((dest/'code').rglob('*')):
        if p.is_file():
            rel = p.relative_to(dest/'code')
            assert sha(p) == sha(OLD/name/'code'/rel), rel
            hashes[str(rel)] = sha(p)
    for n in scripts: adapt(OLD/name/n, dest/n)
    save(dest/'source-provenance.json', {
        'source_setup': str(OLD/name), 'model_training_code_byte_identical': True,
        'initialization_and_hyperparameters_unchanged': True,
        'changes': 'drawer dataset, output locations, derived sample/step counts, plot task names',
        'code_files_sha256': hashes,
    })
for n in ['pipeline.py', 'preflight.py']: adapt(OLD/n, ROOT/n)

signature = hashlib.sha256()
for folder in read(FD/'train.sample_files.json'):
    folder = Path(folder)
    summary = read(folder/'summary.json')
    assert summary['history_source'] == plan['history_source']
    signature.update((folder/'samples.jsonl').read_bytes())
    signature.update(json.dumps(summary, sort_keys=True).encode())
assert signature.hexdigest() == ready['data_signatures']['train']
f = read(OLD/'forest/training.config.json')
f.update(output_dir=str(FOUT), setup_dir=str(ROOT/'forest'),
         sample_files=read(FD/'train.sample_files.json'),
         data_signature=signature.hexdigest(),
         expected_steps_per_epoch=STEPS, expected_total_steps=STEPS*3)
assert f['history_source'] == plan['history_source']
f['run_protocol'].update(task='drawer', train_episodes=21, train_samples=NTRAIN, validation_episodes=6, validation_samples=NVAL,
                        robometer_reference=str(ROOT/'robometer/training.config.json'))
save(ROOT/'forest/training.config.json', f)
r = read(OLD/'robometer/training.config.json')
r.update(train_data=str(RD/'train.jsonl'), validation_data=str(RD/'validation.jsonl'),
         output_dir=str(ROUT), setup_dir=str(ROOT/'robometer'), task=TASK,
         cache_setup_dir='/cache/cvpzhr/setup/robometer-drawer-clean-cache-20261109',
         expected_train_samples=NTRAIN, expected_validation_samples=NVAL, expected_validation_episodes=6,
         steps_per_epoch=STEPS, expected_steps=STEPS*3, eval_steps=STEPS*3, save_steps=STEPS)
save(ROOT/'robometer/training.config.json', r)

policy = read(OLD/'launch-policy.json')
for name, out, ev in [('forest', FOUT, FEVAL), ('robometer', ROUT, ROUT/'evaluation')]:
    policy['tasks'][name].update(setup_dir=str(ROOT/name), output_dir=str(out), evaluation_dir=str(ev))
policy.update(train_episodes=21, train_samples=NTRAIN, validation_episodes=6, validation_samples=NVAL,
              configuration_choice='User requested previous spray official/base experiment unchanged except drawer dataset',
              previous_experiment=str(OLD), steps_per_epoch=STEPS, total_steps=STEPS*3,
              task=TASK, preserve_original_pt_rollbacks=True)
save(ROOT/'launch-policy.json', policy)

def diff(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for k in sorted(set(a)|set(b)):
            result += diff(a.get(k), b.get(k), (prefix+'.' if prefix else '')+k)
        return result
    return [] if a == b else [{'field': prefix, 'previous': a, 'current': b}]

allowed = {
    'forest': {'output_dir','setup_dir','sample_files','data_signature','expected_steps_per_epoch','expected_total_steps',
               'run_protocol.task','run_protocol.train_episodes','run_protocol.train_samples','run_protocol.validation_samples','run_protocol.validation_episodes','run_protocol.robometer_reference'},
    'robometer': {'train_data','validation_data','output_dir','setup_dir','task','cache_setup_dir','expected_train_samples',
                  'expected_validation_samples','expected_validation_episodes','steps_per_epoch','expected_steps','eval_steps','save_steps'},
}
audit = {}
for name, cfg in [('forest', f), ('robometer', r)]:
    changes = diff(read(OLD/name/'training.config.json'), cfg)
    assert {x['field'] for x in changes} <= allowed[name]
    audit[name] = changes
save(ROOT/'configuration-comparison.json', {'status':'passed', 'only_data_paths_counts_and_derived_steps_changed': True, 'differences':audit})
shutil.copy2(__file__, ROOT/'prepare_remote.py')
print(json.dumps({'status':'prepared','setup':str(ROOT),'steps_per_epoch':STEPS,'total_steps':STEPS*3}))

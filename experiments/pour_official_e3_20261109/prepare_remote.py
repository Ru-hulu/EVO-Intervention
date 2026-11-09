from pathlib import Path
import ast, hashlib, json, re, shutil

OLD = Path('/cache/cvpzhr/setup/forest-robometer-drawer-official-e3-20261109')
ROOT = Path('/cache/cvpzhr/setup/forest-robometer-pour-official-e3-20261109')
FD = Path('/cache/cvpzhr/datasets/forest-pour-5fps-no-augmentation-20261109')
RD = Path('/cache/cvpzhr/datasets/robometer-pour-5fps-h7-current3-clean-20261109')
FOUT = Path('/cache/cvpzhr/outputs/forest-pour-frombase-matched-e3-20261109')
ROUT = Path('/cache/cvpzhr/outputs/robometer-pour-fromofficial-clean-2gpu-e3-20261109')
FEVAL = Path('/cache/cvpzhr/evaluations/forest-pour-frombase-matched-e3-cleanval-20261109')
TASK = 'pour-water-into-three-cups-with-repeated-pouring'
NTRAIN, NVAL, STEPS = 6164, 1437, 343
read = lambda p: json.loads(Path(p).read_text())
def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

assert read(OLD/'status.json')['phase'] == 'completed'
assert not any(p.exists() for p in [ROOT, FOUT, ROUT, FEVAL])
plan = read(FD/'plan.json')
ready = read(FD/'cache-ready.json')
assert ready['counts'] == {'train': {'episodes': 22, 'samples': NTRAIN}, 'validation': {'episodes': 6, 'samples': NVAL}}
assert read(RD/'cache-ready.json')['status'] == 'ready'
ROOT.mkdir()

def adapt(p, dest):
    text = p.read_text().replace('drawer-inspection-close', TASK).replace('drawer', 'pour').replace('抽屉检查', '重复倒水')
    values = {3138:NTRAIN, 719:NVAL, 175:STEPS, 350:STEPS*2, 525:STEPS*3}
    text = re.sub(r'\b(3138|719|175|350|525)\b', lambda m:str(values[int(m[0])]), text)
    text = text.replace("('train',6164,21)", "('train',6164,22)")
    text = text.replace("['warmup_steps']==16", "['warmup_steps']==31")
    text = text.replace('原始进度回退保留', '重复操作期间真值保持不变')
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
        'changes': 'pour dataset, output locations, derived sample/step counts, plot task names',
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
         sample_files=read(FD/'train.sample_files.json'), data_signature=signature.hexdigest(),
         expected_steps_per_epoch=STEPS, expected_total_steps=STEPS*3)
assert f['history_source'] == plan['history_source']
f['run_protocol'].update(task='pour', train_episodes=22, train_samples=NTRAIN, validation_episodes=6, validation_samples=NVAL,
                        robometer_reference=str(ROOT/'robometer/training.config.json'))
save(ROOT/'forest/training.config.json', f)
r = read(OLD/'robometer/training.config.json')
r.update(train_data=str(RD/'train.jsonl'), validation_data=str(RD/'validation.jsonl'),
         output_dir=str(ROUT), setup_dir=str(ROOT/'robometer'), task=TASK,
         cache_setup_dir='/cache/cvpzhr/setup/robometer-pour-clean-cache-20261109',
         expected_train_samples=NTRAIN, expected_validation_samples=NVAL, expected_validation_episodes=6,
         steps_per_epoch=STEPS, expected_steps=STEPS*3, eval_steps=STEPS*3, save_steps=STEPS)
save(ROOT/'robometer/training.config.json', r)

policy = read(OLD/'launch-policy.json')
for name, out, ev in [('forest', FOUT, FEVAL), ('robometer', ROUT, ROUT/'evaluation')]:
    policy['tasks'][name].update(setup_dir=str(ROOT/name), output_dir=str(out), evaluation_dir=str(ev))
policy.update(train_episodes=22, train_samples=NTRAIN, validation_episodes=6, validation_samples=NVAL,
              configuration_choice='User requested previous comparison unchanged except pour dataset; fresh Qwen3.5-4B and official Robometer-4B; two GPUs each; three epochs',
              previous_experiment=str(OLD), steps_per_epoch=STEPS, total_steps=STEPS*3,
              task=TASK, original_pt_preserved=True,
              progress_rule='First pour and associated put-down per cup earn progress; repeats and idle hold original PT')
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
cache_scripts = ROOT/'cache_preparation'
cache_scripts.mkdir()
for p in Path('/cache/cvpzhr/setup/pour-clean-cache-preparation-20261109').glob('*.py'):
    if p.name != 'prepare_remote.py': shutil.copy2(p, cache_scripts/p.name)
print(json.dumps({'status':'prepared','setup':str(ROOT),'steps_per_epoch':STEPS,'total_steps':STEPS*3}))

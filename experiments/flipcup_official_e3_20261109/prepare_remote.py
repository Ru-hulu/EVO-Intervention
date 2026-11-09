from pathlib import Path
import ast, copy, hashlib, json, math, shutil

OLD = Path('/cache/cvpzhr/setup/forest-robometer-pencil-official-e3-20261109')
ROOT = Path('/cache/cvpzhr/setup/forest-robometer-flipcup-official-e3-20261109')
FD = Path('/cache/cvpzhr/datasets/forest-flipcup-5fps-no-augmentation-20261109')
RD = Path('/cache/cvpzhr/datasets/robometer-flipcup-5fps-h7-current3-clean-20261109')
FOUT = Path('/cache/cvpzhr/outputs/forest-flipcup-frombase-matched-e3-20261109')
ROUT = Path('/cache/cvpzhr/outputs/robometer-flipcup-fromofficial-clean-2gpu-e3-20261109')
FEVAL = Path('/cache/cvpzhr/evaluations/forest-flipcup-frombase-matched-e3-cleanval-20261109')
TASK = 'flip-two-cups-left-to-right-with-recovery'
NTRAIN, NVAL, STEPS = 7769, 1744, 432
read = lambda p: json.loads(Path(p).read_text())
def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

assert read(OLD/'status.json')['phase'] == 'completed'
assert not any(p.exists() for p in [ROOT, FOUT, ROUT, FEVAL])
plan = read(FD/'plan.json')
ready = read(FD/'cache-ready.json')
assert ready['counts'] == {'train': {'episodes': 28, 'samples': NTRAIN}, 'validation': {'episodes': 7, 'samples': NVAL}}
ROOT.mkdir()
replacements = {
    str(OLD): str(ROOT),
    '/cache/cvpzhr/outputs/forest-pencil-frombase-matched-e3-20261109': str(FOUT),
    '/cache/cvpzhr/outputs/robometer-pencil-fromofficial-clean-2gpu-e3-20261109': str(ROUT),
    '/cache/cvpzhr/evaluations/forest-pencil-frombase-matched-e3-cleanval-20261109': str(FEVAL),
    '/cache/cvpzhr/datasets/forest-pencil-5fps-no-augmentation-20261109': str(FD),
    '2177': str(NVAL), '2,177': '1,744', '10789': str(NTRAIN),
    '1800': str(STEPS*3), '600*i': '432*i',
    "['warmup_steps']==54": "['warmup_steps']==39",
    'forest-pencil-validation-curves.zip': 'forest-flipcup-validation-curves.zip',
    'pencil-case-task': 'flipcup-recovery-task', '笔袋检查': '翻杯恢复',
}
def adapt(p, dest):
    text = p.read_text()
    for a, b in replacements.items(): text = text.replace(a, b)
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
        'changes': 'flipcup dataset, output locations, derived sample/step counts, plot task names',
        'code_files_sha256': hashes,
    })
for n in ['pipeline.py', 'preflight.py']: adapt(OLD/n, ROOT/n)

f = read(OLD/'forest/training.config.json')
f.update(output_dir=str(FOUT), setup_dir=str(ROOT/'forest'),
         sample_files=read(FD/'train.sample_files.json'),
         data_signature=ready['data_signatures']['train'],
         expected_steps_per_epoch=STEPS, expected_total_steps=STEPS*3)
assert f['history_source'] == plan['history_source']
f['run_protocol'].update(task='flipcup', train_samples=NTRAIN, validation_samples=NVAL,
                        robometer_reference=str(ROOT/'robometer/training.config.json'))
save(ROOT/'forest/training.config.json', f)
r = read(OLD/'robometer/training.config.json')
r.update(train_data=str(RD/'train.jsonl'), validation_data=str(RD/'validation.jsonl'),
         output_dir=str(ROUT), setup_dir=str(ROOT/'robometer'), task=TASK,
         cache_setup_dir='/cache/cvpzhr/setup/robometer-flipcup-clean-e3-20261109',
         expected_train_samples=NTRAIN, expected_validation_samples=NVAL,
         steps_per_epoch=STEPS, expected_steps=STEPS*3, eval_steps=STEPS*3, save_steps=STEPS)
save(ROOT/'robometer/training.config.json', r)

policy = read(OLD/'launch-policy.json')
for name, out, ev in [('forest', FOUT, FEVAL), ('robometer', ROUT, ROUT/'evaluation')]:
    policy['tasks'][name].update(setup_dir=str(ROOT/name), output_dir=str(out), evaluation_dir=str(ev))
policy.update(train_samples=NTRAIN, validation_samples=NVAL,
              configuration_choice='User requested previous pencil official/base experiment unchanged except flipcup dataset',
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
               'run_protocol.task','run_protocol.train_samples','run_protocol.validation_samples','run_protocol.robometer_reference'},
    'robometer': {'train_data','validation_data','output_dir','setup_dir','task','cache_setup_dir','expected_train_samples',
                  'expected_validation_samples','steps_per_epoch','expected_steps','eval_steps','save_steps'},
}
audit = {}
for name, cfg in [('forest', f), ('robometer', r)]:
    changes = diff(read(OLD/name/'training.config.json'), cfg)
    assert {x['field'] for x in changes} <= allowed[name]
    audit[name] = changes
save(ROOT/'configuration-comparison.json', {'status':'passed', 'only_data_paths_counts_and_derived_steps_changed': True, 'differences':audit})
shutil.copy2(__file__, ROOT/'prepare_remote.py')
print(json.dumps({'status':'prepared','setup':str(ROOT),'steps_per_epoch':STEPS,'total_steps':STEPS*3}))

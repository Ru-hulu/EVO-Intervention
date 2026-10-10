"""Add intent to active A100 training indexes without changing other fields."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

from qwen_training.prompts import TASK_INTENT

JOB = Path('/data/evo-inter/verification/prompt_intent_20261009')
ROOTS = [
    (Path('/root/dataset/RW-RL-Dataset/piperx_insert_copper_screw'), 118990),
    (Path('/root/dataset/piperx-bolt-intentional-drop-reinsert-22ep-raw'), 80463),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_intent(record):
    return {**record, 'target': {k: v for k, v in record['target'].items() if k != 'intent'}}


def main():
    pending_files, reports = [], []
    for root, expected_count in ROOTS:
        samples = root / 'samples'
        backup = JOB / 'before' / root.name / 'samples'
        backup.mkdir(parents=True, exist_ok=True)
        for p in sorted(samples.iterdir()):
            assert p.is_file() and p.name in {'samples.jsonl', 'statistics.json', 'examples.json', 'validation.json'}
            target = backup / (p.name + '.gz')
            assert not target.exists(), target
            with p.open('rb') as src, gzip.open(target, 'wb', compresslevel=6) as out:
                shutil.copyfileobj(src, out)
        path = samples / 'samples.jsonl'
        pending = samples / 'samples.jsonl.intent.tmp'
        original_sha = sha(path)
        canonical_before, canonical_after = hashlib.sha256(), hashlib.sha256()
        n = 0
        with path.open() as src, pending.open('w') as out:
            for line in src:
                original = json.loads(line)
                assert original['target'].get('intent', TASK_INTENT) == TASK_INTENT
                updated = {**original, 'target': {**original['target'], 'intent': TASK_INTENT}}
                before, after = without_intent(original), without_intent(updated)
                assert before == after
                canonical_before.update(json.dumps(before, ensure_ascii=False, sort_keys=True).encode())
                canonical_after.update(json.dumps(after, ensure_ascii=False, sort_keys=True).encode())
                assert 'intent' not in updated['input_state']
                out.write(json.dumps(updated, ensure_ascii=False, separators=(',', ':')) + '\n')
                n += 1
        assert n == expected_count and canonical_before.digest() == canonical_after.digest()
        adapted_sha = sha(pending)
        pending_files.append((pending, path))
        for name in ['statistics.json', 'examples.json', 'validation.json']:
            path = samples / name
            obj = json.loads(path.read_text())
            if name == 'statistics.json':
                obj['settings']['task_intent'] = TASK_INTENT
            elif name == 'examples.json':
                for record in obj.values():
                    assert isinstance(record.get('target'), dict)
                    record['target']['intent'] = TASK_INTENT
            else:
                obj['samples_sha256'] = adapted_sha
                obj['intent_check'] = {'passed': True, 'samples_checked': n, 'intent': TASK_INTENT}
                obj['non_intent_fields_preserved'] = True
            pending = path.with_name(path.name + '.intent.tmp')
            pending.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
            pending_files.append((pending, path))
        reports.append({'dataset': root.name, 'samples': n, 'intent': TASK_INTENT,
                        'before_sha256': original_sha, 'after_sha256': adapted_sha,
                        'non_intent_fields_sha256': canonical_after.hexdigest(), 'backup': str(backup)})
        print('Prepared', root.name, n, 'labels; all other fields unchanged', flush=True)
    for pending, target in pending_files:
        pending.replace(target)
    for root, _ in ROOTS:
        p = root / 'samples'
        validation = json.loads((p / 'validation.json').read_text())
        assert validation['samples_sha256'] == sha(p / 'samples.jsonl')
    (JOB / 'intent_migration_report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
    print('PUBLISHED', sum(x['samples'] for x in reports), 'intent labels', flush=True)


if __name__ == '__main__':
    main()

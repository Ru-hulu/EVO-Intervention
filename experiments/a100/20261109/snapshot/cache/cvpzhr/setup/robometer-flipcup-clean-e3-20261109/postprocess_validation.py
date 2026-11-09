"""After final training/evaluation succeeds, verify labels and render every curve."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

SETUP = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_episodes(config):
    paths = read_json(Path(config['cache_setup_dir']) / 'source/validation.sample_files.json')
    episodes = []
    for folder in paths:
        source = read_json(Path(folder) / 'summary.json')['source']
        assert source['task'] == config['task'] and source['split'] == 'validation'
        text = Path(source['progress_csv']).read_bytes().decode('utf-8')
        assert hashlib.sha256(text.encode()).hexdigest() == source['progress_sha256']
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == source['frames']
        assert [int(r['frame_index']) for r in rows] == list(range(len(rows)))
        assert source['fps'] == 30
        episodes.append({**source, 'ground_truth_csv_text': text})
    assert len(episodes) == config['expected_validation_episodes']
    cached = [json.loads(line) for line in Path(config['validation_data']).read_text().splitlines()]
    assert len(cached) == config['expected_validation_samples']
    assert {r['episode_id'] for r in cached} == {s['episode_id'] for s in episodes}
    return episodes, cached


def render(config):
    output = Path(config['output_dir'])
    report = read_json(output / 'report.json')
    assert report['status'] == 'completed' and report['steps'] == config['expected_steps']
    assert report['task'] == config['task'] and report['additional_epochs'] == 3
    checkpoint = Path(report['final_checkpoint'])
    assert checkpoint == output / f"checkpoint-{config['expected_steps']}"
    last = read_json(output / 'last.json')
    assert last['complete'] and last['checkpoint'] == str(checkpoint)
    metrics = read_json(output / 'evaluation/metrics.json')
    assert metrics['eval_samples'] == config['expected_validation_samples']
    assert metrics['eval_episodes'] == config['expected_validation_episodes']
    predictions = [json.loads(line) for line in (output / 'evaluation/predictions.jsonl').read_text().splitlines()]
    episodes, cached = source_episodes(config)
    assert len(predictions) == len(cached)
    for prediction, row in zip(predictions, cached, strict=True):
        assert prediction['id'] == row['id'] and prediction['task_id'] == config['task']
        assert prediction['source_frame_index'] == row['source_frame_index']
        assert abs(prediction['target_pt'] - row['pt']) < 1e-6
    folder = output / 'evaluation/curves'
    folder.mkdir(exist_ok=True)
    bundle = {'checkpoint': str(checkpoint), 'metrics': metrics, 'episodes': episodes,
              'episode_metrics': read_json(output / 'evaluation/episode_metrics.json'),
              'predictions': predictions}
    write_json(folder / 'validation_bundle.json', bundle)
    env = {**os.environ, 'ROBOMETER_CURVES_DIR': str(folder), 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    subprocess.run([sys.executable, str(SETUP / 'plot_validation_curves.py')], env=env, check=True)
    images = list(folder.glob('*.prediction_curve.png'))
    assert len(images) == config['expected_validation_episodes']
    for image in [*images, folder / 'validation_overview.png']:
        from PIL import Image
        with Image.open(image) as im:
            im.verify()
        assert image.stat().st_size > 10000
    assert (folder / 'validation_overview.svg').stat().st_size > 10000
    fields = ['episode_id', 'source_frame_index', 'time_seconds', 'target_pt', 'predicted_pt', 'error_pp']
    fps = {s['episode_id']: s['fps'] for s in episodes}
    with (folder / 'validation_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in predictions:
            writer.writerow({'episode_id': r['episode_id'], 'source_frame_index': r['source_frame_index'],
                             'time_seconds': r['source_frame_index'] / fps[r['episode_id']],
                             'target_pt': r['target_pt'], 'predicted_pt': r['predicted_pt'],
                             'error_pp': 100 * (r['predicted_pt'] - r['target_pt'])})
    files = sorted(f for f in folder.iterdir() if f.suffix in ('.png', '.svg', '.json', '.csv'))
    archive = folder / 'validation_curves.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for file in files:
            zipped.write(file, file.name)
    return {'curves_dir': str(folder), 'overview': str(folder / 'validation_overview.png'),
            'archive': str(archive), 'episodes': len(images), 'samples': len(predictions),
            'checkpoint': str(checkpoint), 'metrics': metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    config = read_json(SETUP / 'training.config.json')
    if args.preflight:
        episodes, rows = source_episodes(config)
        import plot_validation_curves
        print(json.dumps({'status': 'ready', 'episodes': len(episodes), 'samples': len(rows),
                          'font': plot_validation_curves.FONT}), flush=True)
        return
    with (SETUP / 'postprocess.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'pid': os.getpid(), 'started_at': time.time(), 'phase': 'waiting_for_final_evaluation'}
        path = SETUP / 'postprocess-status.json'
        write_json(path, state)
        try:
            while True:
                training = read_json(SETUP / 'training-status.json')
                if training['phase'] == 'failed':
                    raise RuntimeError('Training failed: ' + training.get('error', 'inspect train.log'))
                if training['phase'] == 'completed':
                    break
                os.kill(training['pid'], 0)
                time.sleep(30)
            state.update(phase='rendering', updated_at=time.time())
            write_json(path, state)
            result = render(config)
            state.update(phase='completed', completed_at=time.time(), **result)
            write_json(path, state)
        except BaseException as error:
            state.update(phase='failed', failed_at=time.time(), error=repr(error))
            write_json(path, state)
            raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Read-only dataset boundary viewer. Run with the project's Python (PyAV available)."""
import argparse
import gc
import hashlib
import json
import mimetypes
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ARCHIVE_CONFIG = Path(__file__).with_name('catalog_archive.json')


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(relative)
    return path


def byte_range(header, size):
    if header is None:
        return 0, size - 1, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not match or not any(match.groups()):
        raise ValueError('Invalid byte range')
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError('Invalid suffix range')
        start, end = max(0, size - length), size - 1
    else:
        start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise ValueError('Unsatisfiable range')
    return start, end, True


class Catalog:
    def __init__(self, root):
        self.root, self.datasets = Path(root), {}
        self.refresh()

    def refresh(self):
        datasets = {}
        for index in sorted(self.root.glob('*/metadata.jsonl')):
            rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
            if not rows or not all('sample_id' in r and 'video_path' in r for r in rows):
                continue
            if len({r['sample_id'] for r in rows}) != len(rows):
                raise ValueError(f'Duplicate sample IDs in {index}')
            datasets[index.parent.name] = {r['sample_id']: r for r in rows}
        self.datasets = datasets

    def row(self, dataset, sample):
        return self.datasets[dataset][sample]

    def views(self, dataset, sample):
        row = self.row(dataset, sample)
        root = self.root / dataset
        views = {k: v['video_path'] for k, v in row.get('video_views', {}).items()}
        if not views:
            views = {'main_view': row['video_path']}
            # Older MemLong indexes predate the wrist-view columns.
            main = Path(row['video_path'])
            for view, folder in [('left_view', 'cam_left_wrist'), ('right_view', 'cam_right_wrist')]:
                candidate = main.parent / folder / main.name
                if (root / candidate).is_file():
                    views[view] = candidate.as_posix()
        return {view: inside(root, path) for view, path in views.items()}

    def summary(self):
        result = []
        names_path = Path(__file__).with_name('catalog_names.json')
        names = json.loads(names_path.read_text()) if names_path.exists() else {}
        for name, rows in self.datasets.items():
            tasks = {}
            for row in rows.values():
                key = row.get('task_id', row.get('task_slug', 'default'))
                task = tasks.setdefault(key, dict(id=key, name=row.get('task_name', key), samples=[]))
                task['samples'].append(dict(id=row['sample_id'], episode=row.get('episode_index'),
                                            annotated=bool(row.get('annotation_path')),
                                            suite=row.get('suite'), model=row.get('model_name'),
                                            failure_type=row.get('failure_type'), reason=row.get('failure_reason'), success=row.get('task_success')))
            for task in tasks.values():
                task['samples'].sort(key=lambda r: (r['episode'] is None, r['episode'] or 0, r['id']))
            result.append(dict(id=name, name=names.get(name, name), count=len(rows), annotated=sum(bool(r.get('annotation_path')) for r in rows.values()),
                               tasks=sorted(tasks.values(), key=lambda t: t['id'])))
        return result

    def end_hold(self, dataset, row, annotation, progress):
        if not dataset.startswith('Draft-') or row.get('task_success') is not False or not progress:
            return 0
        issue = progress.get('first_issue')
        if not issue:
            return 0
        stages = (annotation or {}).get('stages') or []
        start = max(issue['frame'], stages[-1]['start_frame'] if stages else 0)
        return max(0, round(4 * row['fps']) - (row['total_frames'] - start))

    def sample(self, dataset, sample):
        row = self.row(dataset, sample)
        annotation = None
        if row.get('annotation_path'):
            annotation = json.loads(inside(self.root / dataset, row['annotation_path']).read_text())
        root = self.root / dataset
        relative = Path('progress') / (sample + '.json')
        progress = json.loads(inside(root, relative).read_text()) if (root / relative).is_file() else None
        hold = self.end_hold(dataset, row, annotation, progress)
        frames = row['total_frames']
        if hold:
            if annotation:
                annotation['total_frames'] = frames + hold
                annotation['stages'][-1]['end_frame'] = frames + hold
            dense = progress.get('dense_progress')
            if dense:
                dense.extend([dense[-1]] * hold)
        return dict(review_hold_frames=hold, source_frames=frames, progress=progress, id=sample, dataset=dataset, task=row.get('task_name'),
                    subtask_boundary_url=('/api/boundaries?' + urlencode(dict(dataset=dataset, sample=sample)))
                        if row.get('subtask_boundary_path') else None,
                    prompt=row.get('task_prompt') or row.get('source_instruction') or '',
                    fps=(annotation or row)['fps'], total_frames=frames + hold,
                    annotation=annotation, views=list(self.views(dataset, sample)))


class Media:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pool, self.lock, self.jobs = ThreadPoolExecutor(max_workers=2), threading.Lock(), {}

    def prepare(self, source, hold=0):
        stat = source.stat()
        key = hashlib.sha256(f'{source}:{stat.st_size}:{stat.st_mtime_ns}:h264-v1:{hold}'.encode()).hexdigest()
        with self.lock:
            if key not in self.jobs:
                self.jobs[key] = self.pool.submit(self.convert, source, self.cache / f'{key}.mp4', hold)
            job = self.jobs[key]
        if not job.done():
            return dict(status='preparing'), None
        return dict(status='ready'), job.result()

    @staticmethod
    def convert(source, target, hold=0):
        import av
        try:
            with av.open(str(source)) as c:
                stream = c.streams.video[0]
                stream.codec_context.thread_count = 1
                if stream.codec_context.name == 'h264' and not hold:
                    return source
                if target.exists():
                    return target
                rate = stream.average_rate
                temp = target.with_suffix('.partial.mp4')
                with av.open(str(temp), 'w', format='mp4', options={'movflags': '+faststart'}) as out:
                    encoder = out.add_stream('libx264', rate=rate)
                    encoder.width, encoder.height = stream.width, stream.height
                    encoder.pix_fmt = 'yuv420p'
                    encoder.codec_context.thread_count = 1
                    encoder.options = {'preset': 'veryfast', 'crf': '20'}
                    for i, frame in enumerate(c.decode(video=0)):
                        frame = frame.reformat(format='yuv420p')
                        frame.pts, frame.time_base = i, 1 / rate
                        for packet in encoder.encode(frame):
                            out.mux(packet)
                    for j in range(hold):
                        frame.pts = i + j + 1
                        for packet in encoder.encode(frame):
                            out.mux(packet)
                    for packet in encoder.encode():
                        out.mux(packet)
                temp.replace(target)
                return target
        finally:
            gc.collect()


def handler(catalog, media, remote_viewer=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and str(args[1] if len(args) > 1 else '') not in ('200', '206'):
                super().log_message(fmt, *args)

        def json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                params = {k: v[0] for k, v in parse_qs(url.query).items()}
                if url.path in ('/', '/evaluation'):
                    page = 'evaluation.html' if url.path == '/evaluation' else 'index.html'
                    return self.file(Path(__file__).with_name(page))
                if url.path == '/api/catalog':
                    catalog.refresh()
                    entries = catalog.summary()
                    if remote_viewer:
                        try:
                            with urlopen(remote_viewer + '/api/catalog', timeout=3) as response:
                                entries.extend(row for row in json.load(response) if row['id'] not in catalog.datasets)
                        except (URLError, TimeoutError) as exc:
                            self.log_error('Remote viewer unavailable: %s', exc)
                    archived = set(json.loads(ARCHIVE_CONFIG.read_text()).get('archived', [])) if ARCHIVE_CONFIG.exists() else set()
                    return self.json([row for row in entries if row['id'] not in archived])
                if (remote_viewer and params.get('dataset') not in catalog.datasets
                        and url.path in ('/api/sample', '/api/boundaries', '/api/prepare', '/media')):
                    return self.proxy(remote_viewer + self.path)
                if url.path == '/api/sample':
                    return self.json(catalog.sample(params['dataset'], params['sample']))
                if url.path == '/api/boundaries':
                    row = catalog.row(params['dataset'], params['sample'])
                    source = inside(catalog.root / params['dataset'], row['subtask_boundary_path'])
                    return self.file(source)
                if url.path in ('/api/prepare', '/media'):
                    source = catalog.views(params['dataset'], params['sample'])[params['view']]
                    hold = catalog.sample(params['dataset'], params['sample'])['review_hold_frames']
                    state, path = media.prepare(source, hold)
                    if url.path == '/api/prepare':
                        if path:
                            state['url'] = '/media?' + urlencode(params)
                        return self.json(state)
                    if path is None:
                        return self.json(dict(error='Video is still preparing'), 409)
                    return self.file(path)
                self.json(dict(error='Not found'), 404)
            except (KeyError, FileNotFoundError):
                self.json(dict(error='Dataset, sample, view or file not found'), 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.json(dict(error=str(exc)), 500)

        def proxy(self, url):
            headers = {'Range': self.headers['Range']} if self.headers.get('Range') else {}
            try:
                response = urlopen(Request(url, headers=headers, method=self.command), timeout=30)
            except HTTPError as exc:
                response = exc
            with response:
                self.send_response(response.status)
                for key in ('Content-Type', 'Content-Length', 'Content-Range', 'Accept-Ranges', 'Cache-Control'):
                    if response.headers.get(key):
                        self.send_header(key, response.headers[key])
                self.end_headers()
                if self.command != 'HEAD':
                    while block := response.read(256 * 1024):
                        self.wfile.write(block)

        def file(self, path):
            size = path.stat().st_size
            try:
                start, end, partial = byte_range(self.headers.get('Range'), size)
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            if partial:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command == 'HEAD':
                return
            with path.open('rb') as f:
                f.seek(start)
                left = end - start + 1
                while left:
                    block = f.read(min(left, 256 * 1024))
                    if not block:
                        break
                    self.wfile.write(block)
                    left -= len(block)
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    parser.add_argument('--cache', type=Path, default=Path(__file__).with_name('cache'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--remote-viewer', help='Additional viewer streamed over an SSH-forwarded local URL')
    args = parser.parse_args()
    catalog = Catalog(args.datasets_root)
    print(f'Loaded {len(catalog.datasets)} datasets. Open http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), handler(catalog, Media(args.cache), args.remote_viewer)).serve_forever()

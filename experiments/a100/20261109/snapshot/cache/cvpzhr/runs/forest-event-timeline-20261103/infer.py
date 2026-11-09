"""Fresh online Forest inference with observational event tracing only."""
import json
import math
import os
from pathlib import Path
import time

import torch

from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.memory.manager import MemoryManager
from mem_lite.runtime import StreamingProgressPredictor, load_checkpoint

ROOT = Path(__file__).resolve().parent
OUT = Path('/cache/cvpzhr/result/第一次实验/评估结果/事件动态合并-20261103')
UNIT = Path('/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030/cache/validation/Count/04-pump_dish_soap/episode_001500')
CHECKPOINT = Path('/cache/cvpzhr/outputs/forest-proprimem-e3-20261031-0009/last.pt')


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


class EventList(list):
    def __init__(self, owner):
        self.owner = owner
        super().__init__()

    def append(self, event):
        owner = self.owner
        owner.created += 1
        event.trace_id = f'E{owner.created}'
        start = owner.frames_in_long
        end = round(2 * event.time - start + 1)
        assert start < end <= owner.frames_seen - 1
        owner.nodes[event.trace_id] = dict(id=event.trace_id, start=start, end=end,
            tokens=len(event.tokens), children=[], leaves=1, created_at=owner.frames_seen - 1)
        super().append(event)
        owner.operations.append(dict(kind='create', node=event.trace_id,
            tokens_after=len(event.tokens), active_after=owner.active_ids()))

    def __setitem__(self, key, values):
        owner = self.owner
        parents = self[key]
        assert isinstance(key, slice) and len(parents) == 2 and len(values) == 1
        left, right = [owner.nodes[e.trace_id] for e in parents]
        assert left['end'] == right['start']
        owner.merged += 1
        event = values[0]
        event.trace_id = f'M{owner.merged}'
        before = owner.active_ids()
        owner.nodes[event.trace_id] = dict(id=event.trace_id, start=left['start'], end=right['end'],
            tokens=len(event.tokens), children=[e.trace_id for e in parents],
            leaves=left['leaves'] + right['leaves'], created_at=owner.frames_seen - 1)
        super().__setitem__(key, values)
        owner.operations.append(dict(kind='merge', node=event.trace_id,
            parents=[e.trace_id for e in parents], tokens_before=sum(len(e.tokens) for e in parents),
            tokens_after=len(event.tokens), active_before=before, active_after=owner.active_ids()))


class TracedMemory(MemoryManager):
    def __init__(self, hidden_size, **config):
        self.reference = None
        super().__init__(hidden_size, **config)
        self.reference = MemoryManager(hidden_size, **config)

    def reset(self):
        super().reset()
        self.memory_long = EventList(self)
        self.created = self.merged = 0
        self.nodes, self.operations = {}, []
        if self.reference is not None:
            self.reference.reset()

    def active_ids(self):
        return [event.trace_id for event in self.memory_long]

    @torch.no_grad()
    def update(self, frame):
        self.operations = []
        super().update(frame)
        self.reference.update(frame)
        assert torch.equal(self.get_memory_tokens(), self.reference.get_memory_tokens()), 'Tracing changed memory tokens'
        assert len(self.get_memory_tokens()) <= self.token_budget
        assert sum(len(e.tokens) for e in self.memory_long) <= self.long_memory_tokens_quota
        nodes = [self.nodes[i] for i in self.active_ids()]
        assert not nodes or nodes[0]['start'] == 0
        assert all(a['end'] == b['start'] for a, b in zip(nodes, nodes[1:]))
        assert not nodes or nodes[-1]['end'] == self.frames_in_long


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'trace.json').exists(), 'Already inferred; do not repeat'
    meta = json.loads((UNIT / 'summary.json').read_text())
    samples = [json.loads(s) for s in (UNIT / 'samples.jsonl').read_text().splitlines()]
    started = time.perf_counter()
    model, processor, config = load_checkpoint(CHECKPOINT, 'cuda')
    online = StreamingProgressPredictor(model, processor, config['history_source'], config['max_sequence_length'])
    memory = TracedMemory(online.stream.encoder.hidden_size, **config['history_source']['memory_config'])
    online.stream.memory = memory
    data = M3Dataset([UNIT], history_dim=online.stream.encoder.hidden_size, load_history=False)
    load_seconds = time.perf_counter() - started
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(len(data)):
        sample = data[index]
        used = memory.active_ids()
        torch.cuda.synchronize()
        before = time.perf_counter()
        prediction = online(sample['images'], sample['frame_index'], sample['episode_id'], sample['task_prompt'])
        torch.cuda.synchronize()
        pt = float(prediction['pt'].reshape(-1)[0])
        assert math.isfinite(pt)
        record = dict(index=index, frame=sample['frame_index'], time=sample['frame_index'] / meta['source']['fps'],
            predicted_pt=pt, target_pt=sample['target_pt'], history_tokens_used=prediction['history_length'],
            history_events_used=used, events_after=memory.active_ids(), pending_start=memory.frames_in_long,
            long_tokens_after=sum(len(e.tokens) for e in memory.memory_long),
            total_tokens_after=len(memory.get_memory_tokens()), operations=list(memory.operations),
            seconds_with_trace_validation=time.perf_counter() - before)
        rows.append(record)
        if index % 10 == 0 or index == len(data) - 1:
            write(ROOT / 'inference-status.json', dict(samples=index + 1, total=len(data), time=time.time()))
            print(json.dumps({'samples':index + 1, 'events':memory.created, 'merges':memory.merged}), flush=True)
    assert len(rows) == len(samples) == 88
    assert len(rows[-1]['events_after']) == memory.created - memory.merged
    errors = [r['predicted_pt'] - r['target_pt'] for r in rows]
    trace = dict(task='重复挤洗洁精', episode='episode_001500', checkpoint=str(CHECKPOINT),
        source=meta['source'], memory_config=config['history_source']['memory_config'], sample_fps=5,
        rows=rows, nodes=memory.nodes, new_events=memory.created, merges=memory.merged,
        inference_run=True, original_manager_exact_match_all_frames=True, tail_flushed=False,
        prediction_uses_history_before_current_write=True, timeline_shows_state_after_current_write=True,
        model_load_seconds=load_seconds, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        mae_pp=100*math.fsum(abs(e) for e in errors)/len(errors),
        rmse_pp=100*math.sqrt(math.fsum(e*e for e in errors)/len(errors)),
        note='Real batch-1 online inference. Timing includes duplicate memory-manager checks and is not a production latency benchmark.')
    write(OUT / 'trace.json', trace)
    print(json.dumps({k:v for k,v in trace.items() if k not in ['rows','nodes']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

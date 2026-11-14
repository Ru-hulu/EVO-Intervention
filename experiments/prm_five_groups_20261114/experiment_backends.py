"""Seven native model adapters; instruction/images/history are the only inputs."""
from fractions import Fraction
import math
import sys

from .eval_common import uniform_indices
from .forest_adapter import load_verified_forest
from .forest_eval import images_at
from .official_baseline import KINDS, OLD, engine, model_audit, native_module, next_history, vlac_next
from .prepare_cache import sha
from .robometer_eval import MODEL, REVISION, load_image, model_file_audit
from .stream_runtime import install_proc_memory_policy, mapped_episode


def robometer_paths(row, entry, root, index):
    indices = entry['source_frame_indices']
    paths = ([root/f'{indices[j]:08d}-0.png' for j in uniform_indices(index)[:7]]
             if entry['memory_enabled'] else [])
    paths += [root/f'{indices[index]:08d}-{v}.png' if path else None
              for v, path in enumerate(row['videos'])]
    assert len(paths) == (10 if entry['memory_enabled'] else 3)
    return paths


def playback_positions(row, entry):
    rate = Fraction(row['source_rate'])
    positions = [round(t*float(rate)) for t in entry['playback_timestamps']]
    assert all(a < b for a, b in zip(positions, positions[1:])), 'Non-increasing presentation positions'
    return positions


class OfficialBackend:
    def __init__(self, model, memory_fraction):
        ref, hashes = model_audit(model)
        self.native = native_module(); self.kind = KINDS[model]
        cfg = dict(self.native.MODELS[model], batch=1)
        assert cfg['kind'] == self.kind and cfg['path'] == ref['path']
        if self.kind == 'proc':
            install_proc_memory_policy(self.native)
        self.predictor = engine(self.native, cfg, memory_fraction)
        self.audit = dict(model=model, regime='official_zero_shot', official_revision=ref['revision'],
            checkpoint=ref['path'], official_file_sha256=hashes, adapter_loaded=False,
            native_model_loaded=True, native_config=cfg,
            native_worker_sha256=sha(OLD/'worker.py'), native_common_sha256=sha(OLD/'common.py'),
            prompt_sha256={p: sha(OLD/p) for p in ('grm_prompt.txt', 'proc_prompt.txt', 'r2_prompt.txt')},
            uses_future_endpoint_reference=self.kind == 'grm')

    def begin(self, row, entry, root, prefix, should_pause):
        self.native.image.cache_clear()
        self.episode = mapped_episode(row, entry, root, self.kind)
        self.memory = entry['memory_enabled']
        return True

    def predict(self, index, predictions, raw):
        if self.kind == 'vlac' and index == 0:
            return 0., None, None
        history = (next_history(raw[-1]) if self.kind == 'r2' and self.memory and raw
                   else self.native.INITIAL_HISTORY)
        values, answers = self.predictor.infer([(self.episode, index, history)])
        assert len(values) == len(answers) == 1 and isinstance(answers[0], str)
        value = values[0]
        value = float(value) if value is not None and math.isfinite(value) else None
        if self.kind == 'vlac':
            value = vlac_next(predictions[-1], value)
        return value, answers[0], None


class RobometerBackend:
    def __init__(self):
        import torch
        sys.path[:0] = [str(OLD/'runtime'), '/cache/cvpzhr/setup/three-model-e1-cleanval-20261108/code/robometer-upstream']
        from robometer_training.model import load_model, load_processor, progress_batch, current_logits, progress_values
        hashes = model_file_audit()
        self.processor = load_processor(MODEL, min_pixels=4096, max_pixels=65536)
        self.model, _ = load_model(MODEL, self.processor)
        assert not any('lora_' in n for n, _ in self.model.named_parameters())
        self.model.to('cuda').eval().requires_grad_(False)
        self.batch, self.logits, self.values = progress_batch, current_logits, progress_values
        self.repeat_checked = False
        self.audit = dict(model='robometer-4b', regime='official_zero_shot', checkpoint=MODEL,
            official_revision=REVISION, official_file_sha256=hashes, adapter_loaded=False,
            native_model_loaded=True, strict_checkpoint_load=True,
            runtime_sha256=sha(OLD/'runtime/robometer_training/model.py'),
            min_pixels=4096, max_pixels=65536, uses_future_endpoint_reference=False)

    def begin(self, row, entry, root, prefix, should_pause):
        self.row, self.entry, self.root = row, entry, root
        load_image.cache_clear()
        return True

    def predict(self, index, predictions, raw):
        import torch
        paths = robometer_paths(self.row, self.entry, self.root, index)
        images = [load_image(str(p) if p else None) for p in paths]
        inputs = self.batch(self.processor, [(self.row['task_prompt'], images)])
        def infer():
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits = self.logits(self.model, {k: v.to('cuda') for k, v in inputs.items()})
                return float(self.values(logits).cpu().tolist()[0])
        value = infer()
        if not self.repeat_checked:
            repeated = infer()
            assert abs(value-repeated) < .002, 'Native singleton repeat validation failed'
            self.repeat_checked = True
        assert math.isfinite(value) and 0 <= value <= 1
        return value, None, None


class ForestBackend:
    def __init__(self, suite, memory_enabled):
        self.runtime, self.audit = load_verified_forest(suite, memory_enabled=memory_enabled)
        self.audit.update(model='forest', native_model_loaded=True)

    def begin(self, row, entry, root, prefix, should_pause):
        self.row, self.entry, self.root = row, entry, root
        self.positions = playback_positions(row, entry)
        self.runtime.reset()
        for i in range(len(prefix['predictions'])):
            if should_pause():
                return False
            self.runtime.advance_memory(images_at(root, row, entry['source_frame_indices'][i]),
                self.positions[i], row['key'], row['task_prompt'])
        return True

    def predict(self, index, predictions, raw):
        result = self.runtime.predict(images_at(self.root, self.row, self.entry['source_frame_indices'][index]),
            self.positions[index], self.row['key'], self.row['task_prompt'])
        return result['prediction'], result['raw_nonfinite'], result['history_length']


def load_backend(model, suite, memory_enabled, memory_fraction):
    import torch
    torch.set_num_threads(2); torch.manual_seed(42)
    if model not in KINDS or model == 'vlac-2b':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    if model == 'forest':
        backend = ForestBackend(suite, memory_enabled)
    elif model == 'robometer-4b':
        backend = RobometerBackend()
    else:
        backend = OfficialBackend(model, memory_fraction)
    backend.audit.update(backend_code_sha256=sha(__file__), torch=torch.__version__)
    return backend

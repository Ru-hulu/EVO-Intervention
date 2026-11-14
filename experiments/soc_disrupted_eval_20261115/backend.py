"""Matching final SOC35 weights with each model's original visual memory."""
import math
from .common import sha
from .protocol import image_slots


class Backend:
    def __init__(self, info):
        # Reuse only the strict suite/checkpoint loader, never its no-memory
        # encoder or predictor. No running worker or shared snapshot is changed.
        from experiments.group2_nomemory_20261115.backend import Backend as Loader, image
        from experiments.prm_five_groups_20261114.forest_adapter import ForestRuntime
        self.loader = Loader(info)
        self.name = info['model']; self.cfg = info['config']
        self.model = self.loader.model; self.processor = self.loader.processor; self.image = image
        self.runtime = ForestRuntime(self.model, self.processor, self.cfg, memory_enabled=True) if self.name == 'forest' else None
        self.audit = {k:v for k,v in self.loader.audit.items() if k not in ('history_length', 'input_boundary')}
        self.audit.update(memory_enabled=True, previous_text_used=False, future_endpoint_reference=False,
            backend_sha256=sha(__file__),
            memory_policy={'forest':'native CLIP/max_depth, unchanged token capacity',
                'procvlm-2b':'recent 8 presented main-view observations, left-pad first',
                'robometer-4b':'uniform 8 presented prefix steps, right-pad current; 7 main plus current 3 views'}[self.name],
            position_policy='monotonic playback position in native-frame units; never rewind memory position',
            input_boundary=['task', 'presented_images', 'playback_positions'], singleton=True)
        self.episode = None

    def begin(self, episode):
        assert set(episode) == {'key', 'task', 'images', 'positions'}
        assert len(episode['images']) == len(episode['positions'])
        assert all(a < b for a,b in zip(episode['positions'],episode['positions'][1:]))
        self.episode = episode
        self.image.cache_clear()
        if self.runtime is not None:
            self.runtime.reset()

    def rebuild_step(self, step):
        """Rebuild only native vision memory, without language calls or predictions."""
        if self.runtime is not None:
            e = self.episode
            self.runtime.advance_memory([self.image(p) for p in e['images'][step]],
                                        e['positions'][step], e['key'], e['task'])

    def predict(self, step):
        import torch
        e = self.episode; slots = image_slots(e, step, self.name)
        paths = [e['images'][i][v] for i,v in slots]
        audit = dict(playback_step=step, playback_position=e['positions'][step],
                     input_playback_slots=slots, input_images=len(slots))
        if self.runtime is not None:
            result = self.runtime.predict([self.image(p) for p in paths], e['positions'][step], e['key'], e['task'])
            audit['history_tokens'] = result['history_length']
            return dict(prediction=result['prediction'], raw_response=result['raw_nonfinite'], input_audit=audit)
        if self.name == 'robometer-4b':
            api = self.loader.api
            batch = api.progress_batch(self.processor, [(e['task'], [self.image(p) for p in paths])])
        else:
            from experiments.soc35_model_caches_20261114.proc_data import online_prompt
            self.processor.tokenizer.padding_side = 'left'
            batch = online_prompt(self.processor, e['task'], paths)
        assert 'labels' not in batch and 'past_key_values' not in batch
        assert batch['image_grid_thw'].shape[0] == len(slots)
        assert batch['input_ids'].shape[1] <= self.cfg['max_length'], 'Never truncate historical images'
        audit['input_tokens'] = int(batch['attention_mask'].sum())
        batch = {k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if self.name == 'robometer-4b':
                logits = api.current_logits(self.model, batch)
                value = float(api.progress_values(logits).item())
                raw = dict(progress_logits=logits.float().cpu().tolist()[0])
            else:
                from experiments.group2_nomemory_20261115.protocol import parse_progress
                width = batch['input_ids'].shape[1]
                output = self.model.generate(**batch, max_new_tokens=32, do_sample=False, use_cache=True,
                                             temperature=None, top_p=None, top_k=None)
                raw = self.processor.tokenizer.decode(output[0, width:], skip_special_tokens=True)
                value = parse_progress(raw)
        if value is not None and not math.isfinite(value):
            raw = dict(raw_response=raw, nonfinite_prediction=repr(value)); value = None
        return dict(prediction=value, raw_response=raw, input_audit=audit)

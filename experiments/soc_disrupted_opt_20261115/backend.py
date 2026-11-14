"""Bounded CPU image cache, independent-window batching, ordered FoREST memory.

The frozen native backend is retained as the reference and strict weight loader.
No checkpoint, processor settings, image slots, task text or decoder is changed.
"""
from collections import OrderedDict
import math
from experiments.soc_disrupted_eval_20261115.backend import Backend as Native
from experiments.soc_disrupted_eval_20261115.protocol import image_slots


class Backend(Native):
    def __init__(self, info, batch_size=1, cache=True):
        super().__init__(info)
        self.batch_size = batch_size
        self.cache_enabled = cache
        self.pixels = OrderedDict(); self.pixel_bytes = 0
        self.templates = OrderedDict()
        self.cache_limit = 512 * 1024**2

    def begin(self, episode):
        super().begin(episode)
        self.pixels.clear(); self.pixel_bytes = 0; self.templates.clear()

    def native_inputs(self, step):
        e = self.episode
        paths = [e['images'][i][v] for i,v in image_slots(e, step, self.name)]
        if self.name == 'robometer-4b':
            return dict(self.loader.api.progress_batch(self.processor,
                        [(e['task'], [self.image(p) for p in paths])]))
        from experiments.soc35_model_caches_20261114.proc_data import online_prompt
        self.processor.tokenizer.padding_side = 'left'
        return online_prompt(self.processor, e['task'], paths)

    def one_image(self, path):
        if path in self.pixels:
            value, size = self.pixels.pop(path); self.pixels[path] = (value, size)
            return value
        kwargs = dict(return_tensors='pt')
        if self.name == 'procvlm-2b': kwargs.update(min_pixels=4096, max_pixels=65536)
        value = dict(self.processor.image_processor(images=[self.image(path)], **kwargs))
        assert set(value) == {'pixel_values', 'image_grid_thw'}, set(value)
        size = sum(v.numel()*v.element_size() for v in value.values())
        while self.pixels and self.pixel_bytes + size > self.cache_limit:
            _, (_, old_size) = self.pixels.popitem(last=False); self.pixel_bytes -= old_size
        if size <= self.cache_limit:
            self.pixels[path] = (value, size); self.pixel_bytes += size
        return value

    def encoded(self, step, verify=False):
        import torch
        if not self.cache_enabled: return self.native_inputs(step)
        e = self.episode
        values = [self.one_image(e['images'][i][v]) for i,v in image_slots(e, step, self.name)]
        grids = torch.cat([v['image_grid_thw'] for v in values])
        pixels = torch.cat([v['pixel_values'] for v in values])
        key = (e['task'], tuple(map(tuple, grids.tolist())))
        reference = self.native_inputs(step) if verify or key not in self.templates else None
        if reference is not None:
            assert torch.equal(reference['pixel_values'], pixels), 'Cached pixels differ from native'
            assert torch.equal(reference['image_grid_thw'], grids), 'Cached image layout differs'
            template = {k:v.clone() for k,v in reference.items() if k not in ('pixel_values','image_grid_thw')}
            assert set(template) <= {'input_ids','attention_mask','mm_token_type_ids'}, set(template)
            if key in self.templates:
                assert all(torch.equal(v, self.templates[key][k]) for k,v in template.items())
            else:
                if len(self.templates) >= 128: self.templates.popitem(last=False)
                self.templates[key] = template
        result = dict(self.templates[key], pixel_values=pixels, image_grid_thw=grids)
        assert result['input_ids'].shape[1] <= self.cfg['max_length']
        return result

    def packed(self, steps, verify=False):
        import torch
        from torch.nn.functional import pad
        encoded = [self.encoded(i, verify) for i in steps]
        width = max(v['input_ids'].shape[1] for v in encoded)
        keys = set(encoded[0]) - {'pixel_values','image_grid_thw'}
        assert all(set(v)-{'pixel_values','image_grid_thw'} == keys for v in encoded)
        result = {}
        for k in keys:
            value = self.processor.tokenizer.pad_token_id if k == 'input_ids' else 0
            rows = []
            for item in encoded:
                difference = width-item[k].shape[1]
                padding = (difference,0) if self.name == 'procvlm-2b' else (0,difference)
                rows.append(pad(item[k], padding, value=value))
            result[k] = torch.cat(rows)
        for k in ('pixel_values','image_grid_thw'): result[k] = torch.cat([v[k] for v in encoded])
        return result, [int(v['attention_mask'].sum()) for v in encoded]

    def predict_many(self, steps):
        import torch
        e = self.episode
        audits = [dict(playback_step=i, playback_position=e['positions'][i],
                       input_playback_slots=image_slots(e,i,self.name),
                       input_images=len(image_slots(e,i,self.name))) for i in steps]
        if self.name == 'forest':
            # With one sample, execute the reference path literally.
            if len(steps) == 1: return [super().predict(steps[0])]
            from mem_lite.runtime import to_device
            from mem_lite.datasets.views import fill_views
            samples = []
            with torch.no_grad():
                for i, audit in zip(steps, audits):
                    images = fill_views([self.image(p) for p in e['images'][i]])
                    self.runtime._position(e['positions'][i], e['key'], e['task'])
                    history = self.runtime.runner.stream.step(images[0],e['positions'][i],e['key'],e['task'])
                    history = None if history is None else history.detach().clone()
                    audit['history_tokens'] = 0 if history is None else len(history)
                    samples.append(dict(images=images,task_prompt=e['task'],history_tokens=history))
                batch = self.runtime.runner.collator(samples)
                result = self.model(to_device(batch, self.runtime.device))
                values = result['pt'].float().cpu().tolist()
            return [dict(prediction=v if math.isfinite(v) else None,
                         raw_response=None if math.isfinite(v) else repr(v),input_audit=a)
                    for v,a in zip(values,audits)]
        batch, lengths = self.packed(steps)
        for audit, length in zip(audits,lengths): audit['input_tokens'] = length
        batch = {k:v.to('cuda') for k,v in batch.items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if self.name == 'robometer-4b':
                logits = self.loader.api.current_logits(self.model, batch)
                values = self.loader.api.progress_values(logits).float().cpu().tolist()
                raws = [dict(progress_logits=row) for row in logits.float().cpu().tolist()]
            else:
                from experiments.group2_nomemory_20261115.protocol import parse_progress
                width = batch['input_ids'].shape[1]
                output = self.model.generate(**batch, max_new_tokens=32, do_sample=False, use_cache=True,
                    temperature=None, top_p=None, top_k=None)
                raws = [self.processor.tokenizer.decode(r[width:],skip_special_tokens=True) for r in output]
                values = [parse_progress(r) for r in raws]
        return [dict(prediction=v,raw_response=r,input_audit=a) for v,r,a in zip(values,raws,audits)]

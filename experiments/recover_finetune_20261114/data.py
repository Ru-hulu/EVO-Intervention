"""Frozen native inputs; only explicitly defined progress outputs receive loss."""
from decimal import Decimal, ROUND_HALF_UP
import math
from .common import rows


def grm_answer(row):
    before, after = row['before_pt'], row['after_pt']
    if before != 0 or after != row['pt'] or not math.isfinite(after) or not 0 <= after <= 1:
        raise ValueError('This cache uses initial-state BEFORE; reject different score semantics')
    # Native GRM instruction requests integer percent. Raw continuous PT is retained.
    percent = int((Decimal(str(after)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    return f'<score>{"+" if percent else ""}{percent}%</score>'


class Dataset:
    def __init__(self, path): self.rows = rows(path)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class CachedCollator:
    def __init__(self, tokenizer, kind, max_length):
        self.tokenizer, self.kind, self.max_length = tokenizer, kind, max_length

    def __call__(self, records):
        import torch
        from experiments.recover_six_caches_20261114.loader import load_input
        if self.kind == 'procvlm-2b':
            from experiments.recover_six_caches_20261114.proc_progress_only import ProgressOnlyCollator
            batch = ProgressOnlyCollator(self.tokenizer, self.max_length)(records)
            # This experiment uses ProcVLM's text progress interface, no auxiliary head/CoT loss.
            batch.pop('progress_gt')
            return batch
        batches, labels = [], []
        for row in records:
            batch, _ = load_input(row)
            assert set(batch) <= {'input_ids','attention_mask','mm_token_type_ids','pixel_values','image_grid_thw'}
            assert batch['attention_mask'].eq(1).all()
            if self.kind != 'robometer-4b':
                answer = grm_answer(row)
                suffix = self.tokenizer.encode(answer + self.tokenizer.eos_token, add_special_tokens=False)
                old = batch['input_ids'].shape[1]
                batch['input_ids'] = torch.cat([batch['input_ids'], torch.tensor([suffix])], dim=1)
                batch['attention_mask'] = torch.ones_like(batch['input_ids'])
                if 'mm_token_type_ids' in batch:
                    batch['mm_token_type_ids'] = torch.cat([batch['mm_token_type_ids'], torch.zeros((1,len(suffix)),dtype=torch.long)], dim=1)
                labels.append([-100]*old + suffix)
            else: labels.append(row['pt'])
            assert batch['input_ids'].shape[1] <= self.max_length, 'Never truncate native inputs'
            batches.append(batch)
        width = max(b['input_ids'].shape[1] for b in batches)
        result = {}
        for key in batches[0]:
            assert all(key in b for b in batches)
            if key in ('pixel_values','image_grid_thw'):
                result[key] = torch.cat([b[key] for b in batches]); continue
            fill = self.tokenizer.pad_token_id if key == 'input_ids' else 0
            result[key] = torch.full((len(records),width),fill,dtype=batches[0][key].dtype)
            for i,b in enumerate(batches): result[key][i,:b[key].shape[1]] = b[key][0]
        if self.kind == 'robometer-4b': result['labels'] = torch.tensor(labels,dtype=torch.float32)
        else:
            result['labels'] = torch.full((len(records),width),-100,dtype=torch.long)
            for i,label in enumerate(labels): result['labels'][i,:len(label)] = torch.tensor(label)
        return result

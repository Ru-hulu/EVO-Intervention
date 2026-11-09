"""Chronological, length-bucketed video windows; never read Forest history files."""

import random

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .m3_dataset import M3Dataset
from .m3_inputs import M3Collator


def native_positions(types, grids, merge):
    """Qwen image/text mRoPE before VST insertion, calculated in the CPU worker."""
    pieces, cursor, image = [], 0, 0
    for modality, count in zip(*torch.unique_consecutive(types, return_counts=True)):
        count = int(count)
        if int(modality) == 0:
            pieces.append(torch.arange(count)[None].expand(3, -1) + cursor)
            cursor += count
        elif int(modality) == 1:
            t, h, w = grids[image].tolist()
            h, w = h // merge, w // merge
            if count != t * h * w:
                raise ValueError("Image grid and placeholder counts disagree")
            pieces.append(torch.stack((torch.zeros(count, dtype=torch.long),
                                      torch.arange(h).repeat_interleave(w).repeat(t),
                                      torch.arange(w).repeat(t * h))) + cursor)
            cursor += max(h, w)
            image += 1
        else:
            raise ValueError("VST input currently supports still-image triplets only")
    if image != 3:
        raise ValueError("Expected exactly three image views")
    return torch.cat(pieces, dim=1)


class VSTCollator:
    """Insert a shared learnable VST after each r main-view tokens, including remainder.

    All wrists and the progress query follow the VSTs in causal order. Original
    image tokens retain their 3-D layout, shifted by preceding inserted tokens.
    Labels and source frame ordinals never enter the model's input sequence.
    """

    def __init__(self, processor, compression_ratio=8, max_sequence_length=8192, padding_multiple=64):
        if compression_ratio < 1:
            raise ValueError("compression_ratio must be positive")
        self.base = M3Collator(processor, history_dim=1, history_capacity=0,
                               max_sequence_length=max_sequence_length)
        self.ratio, self.maximum, self.multiple = compression_ratio, max_sequence_length, padding_multiple
        self.merge = processor.image_processor.merge_size
        self.pad = processor.tokenizer.pad_token_id

    def __call__(self, samples, stream_batch=None, window_size=1):
        stream_batch = len(samples) if stream_batch is None else stream_batch
        if len(samples) != stream_batch * window_size:
            raise ValueError("Frames must be flattened in [stream, time] order")
        batch = self.base(samples)
        ids_list, positions_list, masks, slots = [], [], [], []
        for row, sample in enumerate(samples):
            valid = batch["attention_mask"][row].bool()
            ids, types = batch["input_ids"][row, valid], batch["mm_token_type_ids"][row, valid]
            positions = native_positions(types, batch["image_grid_thw"][row * 3:(row + 1) * 3], self.merge)
            main = torch.where(types == 1)[0]
            # First contiguous image run is the third-person camera.
            end = 1
            while end < len(main) and main[end] == main[end - 1] + 1:
                end += 1
            main = main[:end]
            boundaries = set(main[self.ratio - 1::self.ratio].tolist()) | {int(main[-1])}
            expanded, pos, vst, selected = [], [], [], []
            inserted = 0
            for j, token in enumerate(ids.tolist()):
                expanded.append(token)
                pos.append(positions[:, j] + inserted)
                vst.append(False)
                if j in boundaries:
                    selected.append(len(expanded))
                    expanded.append(self.pad)
                    inserted += 1
                    pos.append(positions[:, j] + inserted)
                    vst.append(True)
            ids_list.append(torch.tensor(expanded))
            positions_list.append(torch.stack(pos, dim=1))
            masks.append(torch.tensor(vst))
            slots.append(torch.tensor(selected))
        width = max(map(len, ids_list))
        width = (width + self.multiple - 1) // self.multiple * self.multiple
        if width > self.maximum:
            raise ValueError("VST inputs exceed max_sequence_length; truncation is forbidden")
        slot_width = max(map(len, slots))
        batch["input_ids"] = torch.stack([F.pad(x, (0, width - len(x)), value=self.pad) for x in ids_list])
        batch["attention_mask"] = torch.stack([F.pad(torch.ones(len(x), dtype=torch.bool), (0, width - len(x)))
                                               for x in ids_list])
        batch["position_ids"] = torch.stack([F.pad(x, (0, width - x.shape[1])) for x in positions_list], dim=1)
        batch["vst_mask"] = torch.stack([F.pad(x, (0, width - len(x))) for x in masks])
        batch["vst_indices"] = torch.stack([F.pad(x, (0, slot_width - len(x))) for x in slots])
        batch["vst_valid"] = torch.stack([F.pad(torch.ones(len(x), dtype=torch.bool), (0, slot_width - len(x)))
                                          for x in slots])
        batch["frame_valid"] = torch.tensor([s.get("frame_valid", True) for s in samples], dtype=torch.bool)
        batch["vst_flat_indices"] = batch["vst_mask"].flatten().nonzero().flatten()
        batch["stream_batch"], batch["window_size"] = stream_batch, window_size
        for key in ("mm_token_type_ids", "history_tokens", "history_valid_mask"):
            batch.pop(key, None)
        return batch


class EpisodeWindows(Dataset):
    """DDP schedules cover every real observation once per epoch, without duplication.

    All ranks have identical group/window counts. Missing episodes/last frames
    are explicitly masked. Groups of similarly long episodes reduce padding.
    A cache lives across windows of one group and resets when the group changes.
    """

    def __init__(self, sample_files, batch_size, window_size, rank=0, world_size=1, seed=42, epoch=0):
        self.frames = M3Dataset(sample_files, load_history=False)
        self.signature = self.frames.signature
        self.batch_size, self.window_size = batch_size, window_size
        episodes = [[] for _ in self.frames.episodes]
        for index, (episode, _, _) in enumerate(self.frames.samples):
            episodes[episode].append(index)
        self.episodes = episodes
        order = sorted(range(len(episodes)), key=lambda e: len(episodes[e]))
        rng = random.Random(seed + epoch)
        group_size = batch_size * world_size
        groups = [order[i:i + group_size] for i in range(0, len(order), group_size)]
        rng.shuffle(groups)
        self.windows = []
        for group_index, group in enumerate(groups):
            rng.shuffle(group)
            maximum = max(len(episodes[e]) for e in group)
            group = group + [None] * (group_size - len(group))
            local = group[rank * batch_size:(rank + 1) * batch_size]
            for start in range(0, maximum, window_size):
                indices = []
                for episode in local:
                    sequence = [] if episode is None else episodes[episode]
                    indices.extend(sequence[t] if t < len(sequence) else None
                                   for t in range(start, start + window_size))
                self.windows.append({"indices": indices, "group": group_index, "reset": start == 0,
                                     "group_end": start + window_size >= maximum})

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        spec = self.windows[index]
        fallback = next((i for i in spec["indices"] if i is not None), 0)
        samples = []
        for i in spec["indices"]:
            sample = self.frames[fallback if i is None else i]
            sample["frame_valid"] = i is not None
            samples.append(sample)
        return {**spec, "samples": samples, "stream_batch": self.batch_size, "window_size": self.window_size}


class WindowCollator:
    def __init__(self, *args, **kwargs):
        self.frames = VSTCollator(*args, **kwargs)

    def __call__(self, item):
        samples = item["samples"]
        return {"inputs": self.frames(samples, item["stream_batch"], item["window_size"]),
                "target_pt": torch.tensor([s["target_pt"] for s in samples]),
                "reset": item["reset"], "group_end": item["group_end"], "group": item["group"]}

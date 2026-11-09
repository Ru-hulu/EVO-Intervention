"""Check actual frozen visual features, causal replay, and existing training loader."""
import json
from pathlib import Path
import time

import torch
from PIL import Image
from transformers import AutoProcessor

import prepare_repeats as a
from mem_lite.memory.manager import MemoryManager
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder, load_vision_encoder
from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator


def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.08)
    plan=a.get_plan()
    visual,processor,digest=load_vision_encoder(a.MODEL,"cuda")
    if digest!=plan["history_source"]["vision_sha256"]:
        raise ValueError("Wrong vision weights")
    candidates=plan["episodes"]
    chosen=[]
    for predicate in (
        lambda e:e["source_fps_denominator"]!=1,
        lambda e:e["episode_id"].startswith("insert-screw-recover/"),
        lambda e: any(v is None for v in a.rows_at(e["source_dir"])[0]["images"].values()),
    ):
        matching=[e for e in candidates if predicate(e)]
        candidate=min(matching,key=lambda e:e["augmented_samples"])
        if candidate not in chosen: chosen.append(candidate)
    # Compare our raw-feature reuse against unmodified StreamMemory.step on actual images.
    e=chosen[0]
    source=Path(e["source_dir"]); summary=a.read_json(source/"summary.json")
    rows=a.rows_at(source)
    settings=summary["history_source"]
    reference=StreamMemory(QwenFrameEncoder(visual),processor,settings["memory_config"],
                           settings["max_pixels"],settings["min_pixels"])
    optimized=MemoryManager(summary["history_dim"],**settings["memory_config"])
    # All repeats and ensuing observations are checked, including long-memory compression.
    previous_source=None
    compared=0
    for row in a.expand(rows,e["holds"],a.source_fps(summary)):
        if row["source_sample_index"]!=previous_source:
            with Image.open(source/row["images"]["cam_high"]) as im:
                image=im.convert("RGB")
            native,height,width=a.encode_raw(visual,processor,image,settings)
            previous_source=row["source_sample_index"]
        expected=reference.step(image,row["frame_index"],"smoke","same task")
        actual=optimized.get_memory_tokens()
        if expected is None:
            if actual is not None: raise ValueError("First-frame history is not empty")
        elif not torch.equal(expected,actual):
            raise ValueError("Cached raw features differ from unmodified streaming replay")
        optimized.update(a.position_features(native,height,width,row["frame_index"]))
        compared+=1
    # Check native, unaugmented replay against the existing saved source history.
    baseline=StreamMemory(QwenFrameEncoder(visual),processor,settings["memory_config"],
                          settings["max_pixels"],settings["min_pixels"])
    original_comparisons=0
    for row in rows[:6]:
        with Image.open(source/row["images"]["cam_high"]) as im:
            history=baseline.step(im.convert("RGB"),row["frame_index"],"baseline","same task")
        if row["history"] is not None:
            saved=torch.load(source/row["history"],weights_only=True,map_location="cuda")["tokens"]
            if not torch.equal(saved,history):
                raise ValueError("Regenerated baseline differs from original cache")
            original_comparisons+=1
    receipts=[a.build_episode(plan,e,visual,processor) for e in chosen]
    data=M3Dataset([e["output_dir"] for e in chosen]+[chosen[0]["source_dir"]])
    training_processor=AutoProcessor.from_pretrained(a.MODEL,local_files_only=True,
                                                   max_pixels=65536,min_pixels=4096)
    collator=Stage3Collator(training_processor,data.history_dim,8192,data.history_capacity)
    samples=[]
    offset=0
    for e in chosen:
        samples.extend([data[offset],data[offset+1],data[offset+e["augmented_samples"]-1]])
        offset+=e["augmented_samples"]
    batch=collator(samples)
    if batch["inputs"]["history_tokens"].shape[0]!=len(samples):
        raise ValueError("Training collator failed")
    report=dict(status="passed",completed_at=a.now(),plan_sha256=plan["plan_sha256"],
                bitwise_equal_streaming_observations=compared,
                bitwise_equal_original_history_checks=original_comparisons,
                episodes=receipts,mixed_original_augmented_dataset_loaded=True,
                existing_training_collator_passed=True,
                batch_size=len(samples),sequence_length=batch["inputs"]["input_ids"].shape[1],
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                parameter_updates=0)
    a.write_json(a.SETUP/"gpu-smoke.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()


"""评测命令参数、输入来源记录和生成输出保存。"""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from mem_lite.datasets.dataset_manifest import load_sample_files


def parser_for(description, max_new_tokens):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="新建评测输出目录")
    parser.add_argument("--model-path", type=Path, help="基座目录，默认沿用检查点配置")
    parser.add_argument("--split", choices=("validation", "train"), default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--limit", type=int, help="只处理前N个窗口/队列，结果会标记为部分评测")
    return parser


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def evaluation_run(args, stage):
    if args.batch_size < 1 or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("批次、生成长度和limit必须为正数")
    files = load_sample_files(args.data_manifest, stage, args.split)
    source = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    origins = {str((args.data_manifest.resolve().parent / video["samples"][stage]).parent.resolve()):
               {"video_id": video["video_id"], "task": video["task"]}
               for video in source["videos"] if video["split"] == args.split}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "loading", "stage": stage, "split": args.split,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "data_manifest": str(args.data_manifest.resolve()), "data_manifest_sha256": sha256(args.data_manifest),
        "sample_files": [{"path": path, "sha256": sha256(path)} for path in files],
        "device": args.device, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "limit": args.limit, "generation": "greedy autoregressive, no teacher forcing",
    }
    write_json(output / "manifest.json", metadata)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        checkpoint.pop("optimizer", None)
        model_path = args.model_path or checkpoint["config"]["model_path"]
        metadata.update(status="running", model_path=str(Path(model_path).resolve()),
                        checkpoint_format=checkpoint.get("format"))
        write_json(output / "manifest.json", metadata)
        yield checkpoint, files, origins, metadata, output
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", metadata)


def device_settings(device):
    device = torch.device(device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("评测入口支持CUDA或CPU")
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("需要可用的BF16 CUDA设备，CPU离线检查请显式指定--device cpu")
        return device, torch.bfloat16
    return device, torch.float32


def autocast_for(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def generation_record(ids, tokenizer, budget, eos_token_ids=None):
    tokens = ids.tolist()
    stops = eos_token_ids if eos_token_ids is not None else tokenizer.eos_token_id
    stops = set(stops if isinstance(stops, (tuple, list)) else [stops])
    end = next((index for index, token in enumerate(tokens) if token in stops), None)
    ended = end is not None
    if ended:
        tokens = tokens[:end + 1]
    return {"prediction": tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "generated_token_ids": tokens, "generated_tokens": len(tokens),
            "truncated": not ended and len(tokens) >= budget}


def save_row(stream, errors, row, correct):
    text = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
    stream.write(text)
    stream.flush()
    if not correct:
        errors.write(text)
        errors.flush()

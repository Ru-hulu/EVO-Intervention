"""M1实际生成切分JSON的离线评测，默认只读取清单中的验证视频。"""

import json
from time import perf_counter

from peft import set_peft_model_state_dict
import torch
from transformers import AutoProcessor

from mem_lite.datasets.m1_dataset import M1Collator, M1Dataset
from mem_lite.evaluation.metrics import grouped_metrics, score_m1, summarize_m1
from mem_lite.evaluation.runtime import (
    autocast_for, device_settings, evaluation_run, generation_record, parser_for, save_row, write_json,
)
from mem_lite.models.m1_segmenter import M1Segmenter


def evaluate(args):
    device, dtype = device_settings(args.device)
    with evaluation_run(args, "m1") as (checkpoint, files, origins, metadata, output):
        config = checkpoint["config"]
        if checkpoint.get("format") == "mem-lite-m1-lora-v1":
            architecture, weights = config, checkpoint["lora"]
        elif checkpoint.get("format") == "mem-lite-stage2-v1":
            architecture, weights = config["m1"], checkpoint["model"]["m1_lora"]
        else:
            raise ValueError("M1评测需要第一阶段或第二阶段检查点")
        model = M1Segmenter(metadata["model_path"], dtype=dtype, gradient_checkpointing=False,
                            **{k: architecture[k] for k in ("lora_rank", "lora_alpha", "lora_dropout")})
        set_peft_model_state_dict(model.qwen, weights)
        model.to(device).requires_grad_(False).eval()
        processor = AutoProcessor.from_pretrained(metadata["model_path"], max_pixels=config["max_pixels"],
                                                  local_files_only=True)
        collator = M1Collator(processor, config["max_sequence_length"])
        dataset = M1Dataset(files)
        count = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
        metadata.update(available_samples=len(dataset), evaluated_samples=count, partial=count < len(dataset))
        rows, started = [], perf_counter()
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions, \
             (output / "errors.jsonl").open("w", encoding="utf-8") as errors:
            for first in range(0, count, args.batch_size):
                last = min(first + args.batch_size, count)
                # build_prompt仅使用prompt/images，训练答案response不参与生成。
                batch = collator.build_prompt([dataset[index] for index in range(first, last)])
                if batch["input_ids"].shape[1] + args.max_new_tokens > config["max_sequence_length"]:
                    raise ValueError("观测提示与生成预算超过max_sequence_length，不自动裁剪观测")
                batch = {key: tensor.to(device) for key, tensor in batch.items()}
                with torch.inference_mode(), autocast_for(device):
                    generated = model.generate(batch, args.max_new_tokens)
                for index, ids in zip(range(first, last), generated.cpu(), strict=True):
                    directory, sample = dataset.samples[index]
                    row = {**origins[str(directory.resolve())], "sample_index": index,
                           "sample_id": sample["sample_id"], "case": sample["metadata"]["case"],
                           "window": sample["metadata"]["window"], "target": sample["target"],
                           "target_interval": sample["metadata"]["target_interval"],
                           "queue_frame_indices": [frame["frame_index"] for frame in sample["inputs"]["frames"]],
                           **generation_record(ids, processor.tokenizer, args.max_new_tokens,
                                               model.qwen.generation_config.eos_token_id)}
                    row["scores"] = score_m1(row["prediction"], sample)
                    rows.append(row)
                    save_row(predictions, errors, row, row["scores"]["exact_match"])
        metrics = grouped_metrics(rows, summarize_m1, ("task", "video_id", "case"))
        metrics["elapsed_seconds"] = perf_counter() - started
        metrics["notes"] = [
            "validity只判断valid字段；format_valid同时要求字段完整、整数且边界合法，不修复输出。",
            "positive_iou覆盖所有真值有效样本；漏检、无法解析或边界非法计0。",
            "边界MAE仅统计真值有效且预测有效、边界合法的样本，分母为boundary_evaluated_samples。",
            "队列右边界映射为最后选中原帧号+1；不映射到下一张稀疏观测。",
        ]
        write_json(output / "metrics.json", metrics)
        return metrics


def main():
    args = parser_for(__doc__, max_new_tokens=96).parse_args()
    result = evaluate(args)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""M2累计事件前缀→历史tokens→状态文本的实际生成评测。"""

import json
from time import perf_counter

import torch
from transformers import AutoProcessor

from mem_lite.datasets.m2_dataset import M2Collator, M2Dataset
from mem_lite.evaluation.metrics import grouped_metrics, score_m2, summarize_m2
from mem_lite.evaluation.runtime import (
    autocast_for, device_settings, evaluation_run, generation_record, parser_for, save_row, write_json,
)
from mem_lite.models.stage2_model import Stage2Model
from mem_lite.train_stage2 import load_trainable_state


def evaluate(args):
    if args.prefix_batch_size < 1:
        raise ValueError("prefix_batch_size必须大于0")
    device, dtype = device_settings(args.device)
    with evaluation_run(args, "m2") as (checkpoint, files, origins, metadata, output):
        if checkpoint.get("format") != "mem-lite-stage2-v1":
            raise ValueError("M2评测需要包含事件编码器、M2及历史投影的第二阶段检查点")
        config = checkpoint["config"]
        model = Stage2Model(
            metadata["model_path"], {**config["m1"], "gradient_checkpointing": False},
            config["event_encoder"], config["m2_compressor"], config["vision_batch_size"],
            config["max_text_sequence_length"], dtype=dtype,
        )
        load_trainable_state(model, checkpoint["model"])
        model.to(device).requires_grad_(False).eval()
        processor = AutoProcessor.from_pretrained(metadata["model_path"], max_pixels=config["max_pixels"],
                                                  local_files_only=True)
        collator, dataset = M2Collator(processor), M2Dataset(files)
        count = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
        metadata.update(available_queues=len(dataset), evaluated_queues=count, partial=count < len(dataset),
                        prefix_batch_size=args.prefix_batch_size,
                        history_source="prepared annotated/perturbed event intervals; M1 predicted boundaries not used")
        rows, started = [], perf_counter()
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions, \
             (output / "errors.jsonl").open("w", encoding="utf-8") as errors:
            for first in range(0, count, args.batch_size):
                last = min(first + args.batch_size, count)
                batch = collator([dataset[index] for index in range(first, last)])
                # 真值留在评测记录侧；历史编码及生成路径不接收状态标签。
                batch.pop("state_descriptions")
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                references = []
                for index in range(first, last):
                    directory, sample = dataset.samples[index]
                    for target in sample["targets"]:
                        references.append({**origins[str(directory.resolve())], "queue_index": index,
                                           "sample_id": sample["sample_id"], "prefix_length": target["prefix_length"],
                                           "target": target["state_description"],
                                           "event_windows": sample["metadata"]["windows"][:target["prefix_length"]]})
                with torch.inference_mode(), autocast_for(device):
                    histories = model.encode_history(batch)
                    if len(histories) != len(references):
                        raise ValueError("历史前缀数量与评测记录不一致")
                    for prefix in range(0, len(histories), args.prefix_batch_size):
                        end = min(prefix + args.prefix_batch_size, len(histories))
                        generated = model.decoder.generate(histories[prefix:end], args.max_new_tokens)
                        for reference, ids in zip(references[prefix:end], generated.cpu(), strict=True):
                            row = {**reference, **generation_record(ids, model.decoder.tokenizer, args.max_new_tokens)}
                            row["scores"] = score_m2(row["prediction"], row["target"])
                            rows.append(row)
                            save_row(predictions, errors, row, row["scores"]["normalized_exact"])
        metadata["evaluated_prefixes"] = len(rows)
        metrics = grouped_metrics(rows, summarize_m2, ("task", "video_id", "prefix_length"))
        metrics["elapsed_seconds"] = perf_counter() - started
        metrics["notes"] = [
            "这些是实际生成文本的词面指标，不是语义判分；同义改写可能被计为不一致。",
            "normalized_exact仅做Unicode NFKC及空白归一化；CER为总编辑距离/总真值字符数，可能大于1。",
            "子句按分号、句号及换行拆分；子句顺序不是机器人真实动作时序的独立判定。",
            "次数指标仅适用于真值含显式数字+次/遍的前缀，按出现顺序比较；不判断次数属于哪个动作。",
            "次数MAE仅统计预测/真值次数项数相同的前缀，缺项和多项在次数序列准确率中计错。",
            "本评测使用预构造扰动事件队列，尚未评估在线M1错误向M2传播的影响。",
        ]
        write_json(output / "metrics.json", metrics)
        return metrics


def main():
    parser = parser_for(__doc__, max_new_tokens=256)
    parser.add_argument("--prefix-batch-size", type=int, default=4, help="每次生成的历史前缀数量")
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

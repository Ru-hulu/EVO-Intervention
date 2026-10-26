"""批量构建M1/M2样本，共享视频级训练/验证划分；只处理数据，不加载模型。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from mem_lite.datasets.dataset_manifest import (
    MANIFEST_FORMAT, create_manifest, save_manifest, summarize_manifest,
)
from mem_lite.datasets.prepare_m1 import DEFAULT_CONFIG as M1_CONFIG, prepare_m1
from mem_lite.datasets.prepare_m2 import DEFAULT_CONFIG as M2_CONFIG, prepare_m2


def build_dataset(manifest_path, dataset_root=None):
    """按既定清单构建/续建；已完成目录复用，仅重建未完成的.partial目录。"""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format_version"] != MANIFEST_FORMAT:
        raise ValueError("需要mem-lite视频级数据清单")
    if manifest["status"] == "complete":
        return manifest
    source_root = Path(dataset_root or manifest["dataset_root"]).resolve()
    manifest.update(dataset_root=str(source_root), status="building")
    manifest.pop("error", None)
    save_manifest(manifest_path, manifest)
    current = None
    try:
        for current in manifest["videos"]:
            annotation = source_root / current["annotation"]
            if hashlib.sha256(annotation.read_bytes()).hexdigest() != current["annotation_sha256"]:
                raise ValueError(f"标注已改变，请重新创建批次：{current['video_id']}")
            summaries = {}
            for stage in ("m1", "m2"):
                output = (manifest_path.parent / current["samples"][stage]).parent
                config = {**manifest["configs"][stage], "seed": current["seeds"][stage]}
                if output.exists():
                    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                    if summary["config"] != config or not (output / "samples.jsonl").is_file():
                        raise ValueError(f"已有样本与清单不一致：{output}")
                else:
                    temporary = output.with_name(output.name + ".partial")
                    if temporary.exists():
                        shutil.rmtree(temporary)
                    if stage == "m1":
                        summary = prepare_m1(annotation, temporary, config, source_root / current["video"],
                                             main_view=manifest["main_view"])
                    else:
                        summary = prepare_m2(annotation, temporary, config, source_root / current["video"])
                    count = summary["sample_count" if stage == "m1" else "queue_count"]
                    if count == 0:
                        raise ValueError(f"没有生成有效训练样本：{current['video_id']} / {stage}")
                    temporary.rename(output)
                summaries[stage] = summary
            m1, m2 = summaries["m1"], summaries["m2"]
            current.update(status="complete", counts={
                "m1_samples": m1["sample_count"], "m1_skipped_empty": m1["skipped_empty"],
                "m1_valid": sum(counts["valid"] for counts in m1["by_case"].values()),
                "m1_invalid": sum(counts["invalid"] for counts in m1["by_case"].values()),
                "m2_queues": m2["queue_count"], "m2_prefixes": m2["prefix_count"],
            })
            current.pop("error", None)
            manifest["summary"] = summarize_manifest(manifest)
            save_manifest(manifest_path, manifest)
            print(json.dumps({"video_id": current["video_id"], "split": current["split"],
                              **current["counts"]}, ensure_ascii=False), flush=True)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if current is not None:
            current.update(status="failed", error=manifest["error"])
        raise
    finally:
        manifest["summary"] = summarize_manifest(manifest)
        save_manifest(manifest_path, manifest)
    return manifest


def prepare_dataset(dataset_root, output, m1_config, m2_config, validation_fraction=0.2,
                    split_seed=42, main_view="cam_high", plan_only=False):
    manifest = create_manifest(dataset_root, m1_config, m2_config, validation_fraction, split_seed, main_view)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    save_manifest(manifest_path, manifest)
    return manifest if plan_only else build_dataset(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="MemLong根目录；续建时可用于迁移源视频路径")
    parser.add_argument("--output", type=Path, help="新建输出目录")
    parser.add_argument("--manifest", type=Path, help="构建计划清单或续建中断批次，保持原划分和种子")
    parser.add_argument("--plan-only", action="store_true", help="只保存视频划分，不解码或生成样本")
    parser.add_argument("--m1-config", type=Path)
    parser.add_argument("--m2-config", type=Path)
    parser.add_argument("--validation-fraction", type=float, help="默认0.2，按任务取整；0表示全训练集")
    parser.add_argument("--split-seed", type=int, help="默认42，与扰动配置种子独立")
    parser.add_argument("--main-view", help="默认cam_high")
    args = parser.parse_args()
    if args.manifest:
        if any(value is not None for value in (args.output, args.m1_config, args.m2_config,
                                               args.validation_fraction, args.split_seed, args.main_view)) or args.plan_only:
            parser.error("复用清单时不能重新划分或修改采样配置；请创建新批次")
        manifest = build_dataset(args.manifest, args.dataset_root)
        path = args.manifest.resolve()
    else:
        if args.dataset_root is None or args.output is None:
            parser.error("新批次需要--dataset-root和--output")
        m1_config = json.loads((args.m1_config or M1_CONFIG).read_text(encoding="utf-8"))
        m2_config = json.loads((args.m2_config or M2_CONFIG).read_text(encoding="utf-8"))
        manifest = prepare_dataset(args.dataset_root, args.output, m1_config, m2_config,
                                   0.2 if args.validation_fraction is None else args.validation_fraction,
                                   42 if args.split_seed is None else args.split_seed,
                                   args.main_view or "cam_high", args.plan_only)
        path = args.output.resolve() / "manifest.json"
    print(json.dumps({"manifest": str(path), "status": manifest["status"],
                      "summary": manifest["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

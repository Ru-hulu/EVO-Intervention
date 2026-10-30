"""读取一个13图窗口，生成六行 progress/done（空白帧输出 NA）；不需要输入标签。

python -m qwen_training.infer_reference --checkpoint /path/last.pt --input-json /path/window.json
"""

import argparse
import json
from pathlib import Path

import torch
from peft import set_peft_model_state_dict

from qwen_training.reference_images import load_window_images
from qwen_training.reference_io import FORMAT_VERSION, make_reference_message, parse_frame_predictions
from qwen_training.reference_model import ReferenceProgressModel


def load_reference_checkpoint(path, device, model_path=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("progress_format") != FORMAT_VERSION:
        raise ValueError(f"需要 {FORMAT_VERSION} 检查点；旧 stage/MEM 检查点不能用于此入口")
    model = ReferenceProgressModel(
        model_path or config["model_path"],
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        freeze_backbone=True, max_pixels=config["max_pixels"],
        lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    set_peft_model_state_dict(model.qwen, checkpoint["lora"])
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", help="可覆盖检查点记录的基座路径，需使用同一基座权重")
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, help="可选：保存原始回复、六帧解析及有效性")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    sample = json.loads(args.input_json.read_text(encoding="utf-8"))
    images = load_window_images(sample, args.input_json.resolve().parent)
    model = load_reference_checkpoint(args.checkpoint, device, args.model_path)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        result = model.generate(
            messages=[make_reference_message(sample["task"], sample["frames"])], images=[images],
            max_new_tokens=args.max_new_tokens,
        )
    response = result["responses"][0]
    print(response)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({
            "response": response,
            "frames": parse_frame_predictions(response),
            "valid": result["valid"][0].tolist(),
            "parse_valid": result["parse_valid"][0].tolist(),
            "input_frames": sample["frames"],
            "current_position": next(i for i, frame in enumerate(sample["frames"], 1) if frame.get("is_current")),
        }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

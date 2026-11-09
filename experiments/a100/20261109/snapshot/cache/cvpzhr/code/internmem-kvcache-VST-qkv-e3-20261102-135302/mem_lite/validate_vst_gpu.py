"""Actual 4B/BF16 model checks before expensive distributed training."""

import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.datasets.vst_dataset import VSTCollator
from mem_lite.models.vst_predictor import VSTProgressPredictor
from mem_lite.train_vst import device_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    config["model"]["lora_dropout"] = 0.0
    config["model"]["gradient_checkpointing"] = True
    torch.manual_seed(42)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    data = M3Dataset(json.loads(Path(config["sample_files"]).read_text()), load_history=False)
    processor = AutoProcessor.from_pretrained(config["model_path"], max_pixels=config["max_pixels"], local_files_only=True)
    collator = VSTCollator(processor, config["compression_ratio"])
    model = VSTProgressPredictor(config["model_path"], **config["model"]).cuda().eval()
    for i in model.full_layers:
        assert set(model.qwen.model.language_model.layers[i].self_attn.vst_projections) == {"q_proj", "k_proj", "v_proj"}
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vst_projection_parameters = sum(p.numel() for n, p in model.named_parameters() if ".vst_projections." in n)
    frames = [data[i] for i in range(3)]
    def batch(items, window=1):
        return device_batch(collator(items, 1, window), "cuda")
    with torch.no_grad():
        parallel = model(batch(frames, 3))
        sequential, cache = [], None
        for frame in frames:
            out = model(batch([frame]), cache=cache)
            cache = out["cache"]
            sequential.append(out["pt"])
        sequential = torch.cat(sequential)
        error = (sequential - parallel["pt"]).abs().max().item()
        # BF16 kernels vary with GEMM/attention batch shape even for the native
        # backbone. Establish algorithmic equivalence separately in FP32 below.
        if error > 0.02:
            raise AssertionError(f"Unexpectedly large BF16 sequential/window discrepancy {error}")
        altered = copy.deepcopy(frames)
        altered[2]["images"][0] = frames[0]["images"][1].resize(frames[2]["images"][0].size)
        future = model(batch(altered, 3))
        torch.testing.assert_close(parallel["pt"][:2], future["pt"][:2], atol=0, rtol=0)
        for i, (k, v) in parallel["cache"].layers.items():
            assert torch.isfinite(k).all() and torch.isfinite(v).all()
        counts = parallel["cache"].lengths.tolist()
    del parallel, future, cache, out
    model.train()
    first = model(batch([frames[0]]))
    k, v = first["cache"].layers[model.full_layers[0]]
    k.retain_grad(); v.retain_grad()
    second = model(batch([frames[1]]), torch.tensor([frames[1]["target_pt"]], device="cuda"), first["cache"])
    second["loss"].backward()
    key_grad, value_grad = k.grad.abs().sum().item(), v.grad.abs().sum().item()
    assert key_grad > 0 and value_grad > 0
    assert model.vst_embedding.grad.isfinite().all() and model.vst_embedding.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.qwen.model.visual.parameters())
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    detached = second["cache"].detach()
    model.zero_grad(set_to_none=True)
    del first, second, k, v
    third = model(batch([frames[2]]), torch.tensor([frames[2]["target_pt"]], device="cuda"), detached)
    third["loss"].backward()
    del third, detached
    model.zero_grad(set_to_none=True)
    model.float().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for layer in model.qwen.model.language_model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    with torch.no_grad():
        reference = model(batch(frames, 3))["pt"]
        cache, sequential = None, []
        for frame in frames:
            out = model(batch([frame]), cache=cache)
            cache = out["cache"]
            sequential.append(out["pt"])
        sequential = torch.cat(sequential)
        fp32_error = (reference - sequential).abs().max().item()
        torch.testing.assert_close(reference, sequential, atol=2e-5, rtol=1e-5)
    report = {"status": "passed", "bf16_window_vs_sequential_max_pt_error": error,
              "vst_projections": ["q", "k", "v"], "shared_output_projection": True,
              "trainable_parameters": trainable_parameters, "vst_projection_parameters": vst_projection_parameters,
              "fp32_window_vs_sequential_max_pt_error": fp32_error,
              "future_frame_isolation": True, "vst_tokens_after_3_frames": counts,
              "full_attention_layers": model.full_layers, "previous_frame_key_grad_l1": key_grad,
              "previous_frame_value_grad_l1": value_grad, "detached_next_backward": True,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    Path(args.output).write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

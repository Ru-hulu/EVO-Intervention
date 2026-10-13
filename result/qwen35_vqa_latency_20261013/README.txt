Qwen3.5-2B / Qwen3.5-4B: single A100 video-question-answering latency

Remote directory:
  /data/evo-inter/verification/qwen35_vqa_latency_20261013

Input:
  18 distinct, temporally ordered 640x480 PNG frames from the same 19.2-second
  store_laptop_and_headphones demonstration. Source video is retained as video.mp4.
  This uses 18 image items, not the native video-input path and its temporal compression.
  The English prompt contains 98 whitespace-delimited words.
  Both processors produce identical tensors: 5,562 tokens, including 5,400 image tokens.

Runtime:
  Existing isolated /data/evo-inter/.venv; PyTorch 2.10.0+cu128, Transformers 5.3.0.
  BF16, batch size 1, greedy decoding, thinking disabled, at most 128 new tokens.
  SDPA attention and the existing Gated DeltaNet kernel configuration.
  Image preprocessing runs on CPU. No torch.compile or CPU model offload.
  Model sizes run sequentially on the SAME GPU, each in a fresh process.
  Two warmups, then five measured requests for each model.

Run after making an A100 available (replace 0 with its nvidia-smi index):
  bash /data/evo-inter/verification/qwen35_vqa_latency_20261013/run_both.sh 0

The script refuses a GPU with compute processes, over 10% utilization, or less
than 30,000 MiB free VRAM. It never pauses or terminates existing processes.
GPU UUID selection ensures the requested physical card is used.

Outputs:
  results_2B.json / results_4B.json: per-run wall-clock and CUDA-event timings,
  generated answers, output-token counts, memory peaks, software and GPU versions.
  run_2B.log / run_4B.log: logs and progress.

Timing boundaries:
  End-to-end starts before opening the 18 PNGs and ends after answer text decoding.
  Includes image decode, CPU preprocessing, prompt template/tokenization/packing,
  transfer to GPU, vision encoder, visual merger, language prefill, first token,
  remaining generation, and output-to-text decoding.
  Model/processor loading, imports, video-to-PNG export and warmups are reported
  separately and are excluded from steady-state request latency.
  Warm requests benefit from the operating-system file cache.
  TTFT is time to the first emitted token, not the entire completed answer.
  CUDA module spans are measured with events in the first forward pass; nested
  vision/merger time is separated to avoid double-counting. No per-module CUDA
  synchronization is inserted. CPU work and other prefill work are reported as
  the remaining TTFT interval. The inclusive first-forward metric is diagnostic
  and must not be added again to the component timings.
  Generated answers may have different lengths, so report output-token counts
  and decoding tokens/second alongside end-to-end latency.

Completion status:
  Completed on GPU 2, 2026-10-13. Both models have 2 warmups and 5 measured runs.
  Input equivalence, output counts, additive timings and GPU-process snapshots passed.
  See summary.json, report.txt and results_2B.json / results_4B.json.
  The smallest-VRAM two-GPU training job and its lane were stopped with user authorization.
  Existing checkpoints were preserved. The other training/evaluation jobs remained active.

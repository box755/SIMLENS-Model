# SIMLENS model card

- **Authors:** Ren-Siang Wei, Sheng-You Fu, Yen-Liang Chen.
- **Task:** per-second relative replay-curve prediction from video and audio.
- **Base model:** Qwen/Qwen2.5-Omni-7B, Thinker component.
- **Adaptation:** rank-16 LoRA on decoder attention projections, alpha 32;
  linear scalar head and shared suffix-query embedding; frozen perceptual encoders.
- **Input:** frames sampled at 2 fps and resized to 112 pixels, 16 kHz mono audio;
  fixed text placeholder `video`. Four-token audio pooling is the default.
- **Output:** one unrestricted relative score per temporal grid second.
- **Training data:** available Mr.HiSum training videos and released replay labels.
- **Checkpoints:** seeds 42, 7 and 123, selected independently by held-out validation
  Spearman at steps 5,000 / 4,500 / 4,500. The files have not been retrained for release.
- **Files:** LoRA safetensors, adapter config, `head.pt` (head and learned query),
  validation metadata and the release input configuration. `head.pt` is loaded using
  PyTorch's weights-only mode; do not substitute arbitrary untrusted checkpoints.
- **Intended use:** research and human-guided review of relative replay patterns.
- **Not validated:** calibrated audience estimates, automatic deletion decisions,
  individual viewer profiling, or cross-backbone performance.

See README for the reported test results and docs/reproducibility.md for the
positional-setting discrepancy, data availability, metric definitions and remaining
historical provenance gaps. No new benchmark claims are made by this release.

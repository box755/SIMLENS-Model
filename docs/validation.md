# Release validation — 2026-09-13

Checks performed for v0.1.0:

- Built and installed the editable package in a separate virtual environment using
  the existing research environment's installed dependencies. Also built a regular
  wheel, installed it to a separate target directory, and imported the packaged
  default config in isolated Python mode. The wheel defaults to suffix readout.
- All four installed CLI entry points (`predict`, `train`, `evaluate`, `score`)
  successfully displayed their help without dataset downloads or GPU allocation.
- **18 tests passed**: package names and CLI targets, the FFmpeg environment override,
  config inheritance, checkpoint failure handling, listwise
  loss/gradients, constant targets, audio-pooling position selection, processor
  arguments, metric/target mappings, mock demo API, safe checkpoint extraction and
  hash-mismatch rejection.
- Imported the actual external Mr.HiSum metric code and ran a synthetic identical
  prediction/target smoke check; both AP outputs were 1.0. This is a wiring test,
  not a new benchmark result.
- Compared all three staged adapters, heads, adapter configs and validation metadata
  against the retained best checkpoints using SHA-256. All were identical. Loaded
  heads on CPU with `weights_only=True`; verified 1-by-3584 readout weights and
  3584-dimensional query embeddings. Safetensors files opened successfully.
- Compared the published training recipe against the retained three run configs.
  Model/data/loss/optimizer/training/evaluation settings match; the release adds
  portable paths, an inspected snapshot pin and explicit positional guards.
- AST comparisons confirm unchanged core forward, perceptual feature extraction,
  audio pooling, processor assembly, decoding, losses, curve metrics and target
  mapping functions (ignoring docstrings and packaging imports).
- Built and round-trip verified all three checkpoint ZIPs, including every member
  hash. Archives use portable forward-slash member paths, not Windows-only names.
- Scanned the intended source release for common token/private-key patterns and
  local absolute paths. Checkpoints, raw data, temporary tests, build artifacts and
  credential-handling helpers are excluded from Git.

After namespace cleanup, the source package and installed wheel use `simlens`,
`SIMLENSModel`, `SIMLENSTrainer`, and `SIMLENS_FFMPEG` consistently. All package
code and configuration were compared with the preceding revision after normalizing
names and excluding docstrings; no computational changes were found. The three
checkpoint sets and existing release ZIPs retain their recorded SHA-256 hashes.

The inspected environment was Python 3.11, PyTorch 2.11.0+cu128, Transformers 5.8.1,
PEFT 0.19.1, Windows 11 and an RTX 5090. Tests reported pre-existing pynvml and
Starlette/httpx deprecation warnings, not test failures.

## What was not verified

This was **not** a clean internet dependency installation on another machine, a
Linux portability test, a historical environment attestation, or a full GPU model
inference/benchmark rerun. The GPU already had another workload, which was not
stopped or altered. No model training was started and no published experiment
numbers were changed. See reproducibility.md for the remaining research-provenance
and coverage limitations; source availability alone does not resolve them.

# Additional checks — 2026-11-15

Made after v0.1.0 on the same machine (RTX 5090 with 32 GB, Windows 11). The GPU
checks used PyTorch 2.11.0+cu128 under Python 3.13; scoring and tests used Python 3.11.

- **GPU inference.** Loaded the seed-42 checkpoint through `simlens.inference.Predictor`
  and predicted Mr.HiSum test videos of 217 and 299 seconds. PyTorch allocated at most
  17.5 and 17.8 GiB, and `nvidia-smi` showed about 20 GB in use. Loading took 16–27
  seconds with the backbone already on disk. One video took 3.0–5.6 seconds including
  decoding.
- **Demo.** Started `demo/server.py` with the seed-42 checkpoint, uploaded a 217-second
  video and received 217 scores.
- **Agreement with the paper's predictions.** Ran the batch loop documented in
  `datasets/ytfresh/README.md` on 12 YT-Fresh videos with the seed-42 checkpoint. All 12
  predictions have the same length as the saved predictions of the paper's run, and the
  per-video Spearman correlation between new and saved scores is at least 0.998 (largest
  absolute difference 0.13). Differences of this size are expected from non-deterministic
  GPU kernels.
- **Released ID lists.** The lists in `datasets/splits/` have the counts given in the
  paper and hold exactly the IDs of the saved predictions: 1,874 for the Mr.HiSum test
  split (seed 42) and 406 for YTMR500 (three seeds). The data loader keeps every ID when
  it reads `mrhisum_paper_split.json`.
- **YT-Fresh files.** Scoring the saved predictions of the three seeds against
  `datasets/ytfresh/most_replayed.jsonl` with `simlens-score` gives the per-seed results
  in the paper (mAP@50 72.57, 72.76 and 72.73; Spearman 0.431, 0.439 and 0.436) on 1,230
  videos, with none missing.
- **Tests.** The 18 tests passed.

Not done: a rerun of the full Mr.HiSum test split, any training, and any test on Linux
or on another GPU.

# Third-party components

- **Qwen2.5-Omni-7B**: base model from Qwen, downloaded separately. Its published
  license is Apache-2.0: https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/LICENSE.
  This repository does not redistribute the full base-model weights or claim
  authorship of the backbone. Adapter users must also review the base-model terms.
- **MR.HiSum**: external dataset and metric implementation at
  https://github.com/MRHiSum/MR.HiSum. The evaluation wrapper imports its official
  routines from a user-provided checkout. That repository, its data and its notices
  are not relicensed by SIMLENS.
- **YTMR500 / most-replayed-data**: external dataset at
  https://github.com/ombretta/most-replayed-data. Dataset files and videos are not
  bundled here; follow upstream distribution terms.
- PyTorch, Transformers, PEFT, NumPy, SciPy, scikit-learn, Decord, FFmpeg and the
  other dependencies retain their own licenses. Installing the package does not
  replace those terms with the SIMLENS code license.

The original demo frontend is included; no third-party baseline repositories,
PANNs weights, raw YouTube videos or copied benchmark datasets are bundled.

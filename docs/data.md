# Data and official evaluation dependencies

No third-party videos or dataset archives are redistributed by this repository.
Obtain datasets under their source terms and retain the original video IDs.

## Mr.HiSum

Get `mr_hisum.h5`, `metadata.csv`, and `mr_hisum_split.json` from
[MR.HiSum](https://github.com/MRHiSum/MR.HiSum). Place them in `data/mrhisum/` and
the authorized/downloadable videos in `data/mrhisum/videos/video_ID.mp4`.
The official IDs already have the `video_` prefix; do not add it a second time.

The evaluation wrapper uses upstream metric routines, not a copied approximation.
The research setup recorded upstream revision `aeb667f`. To prepare it:

```bash
git clone https://github.com/MRHiSum/MR.HiSum.git third_party/MR.HiSum
git -C third_party/MR.HiSum checkout aeb667f
```

Alternatively set `MRHISUM_ROOT` to an existing checkout. Install upstream metric
dependencies as required by that checkout. The repository is needed for official
metrics, not for single-video inference. SIMLENS reads the released per-second
`gtscore` directly; no new interpolation is applied. Dataset loading preserves
official split order and filters to IDs with both a local video and labels.

The reported training/validation/test coverage was 26,132 / 1,867 / 1,874 videos.
Availability today can differ. A fresh local split is not automatically the exact
historical subset. Never compare new metrics without also checking evaluated IDs.
The IDs used in the paper are listed in `datasets/splits/` (see
[`datasets/README.md`](../datasets/README.md)).

## YTMR500

See [most-replayed-data](https://github.com/ombretta/most-replayed-data) for the
dataset. The unified scorer accepts the `YTMR_500.h5` file through `--gt-file`.
For a prediction of length T, second t takes heatmap bin `floor(100*t/T)`, capped
at 99; this is the historical step/bin lookup, not linear interpolation.

The retained evaluation contains 406 available videos (listed in
`datasets/splits/ytmr500_ids.txt`), with stored prediction
lengths 240–1,343 seconds. The inference entry used a 1,380-second decode ceiling.
It used the full retained prediction curves rather than 300-second prefixes or
independent decoder windows. The default public CLI cap is conservative (300 s);
set `--max-sec 1380` explicitly when reproducing long-video inputs.

## YT-Fresh

YT-Fresh is the authors' curated 1,230-video recent-video benchmark, evaluated
with the three checkpoints (seeds 42, 7 and 123) separately. Its 850-video
post-release subset (`upload_date` from 2025-04-01) is a subset of those 1,230
videos, not another test set. Target category quotas referenced Mr.HiSum, but
quota filling caused the final distribution to deviate.

The video IDs, metadata and recorded Most Replayed curves are released in
[`datasets/ytfresh/`](../datasets/ytfresh/README.md) under CC BY 4.0. The videos
are not redistributed. The scorer reads `datasets/ytfresh/most_replayed.jsonl`,
one line per video with `youtube_id` and a
`heatmap` list of `{start_time, value}` records. Seconds use the last recorded bin
start not later than that second (`searchsorted(..., side="right") - 1`). Flat
ground-truth curves are excluded by the existing scoring rule.

Mr.HiSum, YTMR500 and YT-Fresh use different label sources/mappings. Their metrics
must not be interpreted as isolated tests of publication date or video duration.

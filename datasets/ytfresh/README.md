# YT-Fresh

YT-Fresh is a test set of 1,230 recent YouTube videos with their Most Replayed curves. The SIMLENS paper uses it to check whether a model trained on Mr.HiSum still works on videos uploaded years later. It is a test set only: none of it is used for training or tuning.

No videos are stored here. Download them from YouTube with the IDs in `metadata.csv`.

## Files

### `metadata.csv`

One row per video, sorted by ID.

| Column | Content |
|---|---|
| `youtube_id` | the video's YouTube ID |
| `title` | the title shown on YouTube when the video was collected |
| `duration_s` | length in seconds |
| `upload_date` | upload date, `YYYY-MM-DD` |
| `view_count` | views when the video was collected |
| `youtube_category` | the category YouTube assigns to the video |
| `category_group` | which of our 15 search groups found the video (see [Composition](#composition)) |

### `most_replayed.jsonl`

One line per video, in the same order as `metadata.csv`. Each line is a JSON object like this one, shown with the first two of its 100 bins:

```json
{"youtube_id": "-0CWN4fQo9g", "heatmap": [{"start_time": 0.0, "end_time": 1.72, "value": 0.101841}, {"start_time": 1.72, "end_time": 3.44, "value": 0.008442}]}
```

`heatmap` is the video's public Most Replayed curve: 100 bins of equal length that cover the whole video. `value` is between 0 and 1, and 1 marks the most replayed moment. The curves were recorded in July and August 2026. YouTube updates this statistic as a video is watched, so a curve read today can differ.

## Predicting and scoring

**1. Videos.** Download the videos from YouTube and save each one as `<youtube_id>.mp4` in one folder, for example `data/ytfresh/videos/`.

**2. Predictions.** `simlens-score` reads a JSONL file: one JSON object per line, with `video_id` (the YouTube ID) and `s_hat` (one score per second). `simlens-predict` writes a different format, one indented JSON file per video, so its output cannot be passed to the scorer. Write the JSONL file with a loop like this, run from the repository root:

```python
import json
from pathlib import Path
from simlens.inference import Predictor

predictor = Predictor("checkpoints/seed42")
with open("ytfresh_predictions.jsonl", "w", encoding="utf-8") as out:
    for video in sorted(Path("data/ytfresh/videos").glob("*.mp4")):
        scores = predictor.predict(video)
        out.write(json.dumps({"video_id": video.stem, "s_hat": [round(float(s), 4) for s in scores]}) + "\n")
```

**3. Scores.**

```bash
simlens-score --pred ytfresh_predictions.jsonl --gt ytfresh --gt-file datasets/ytfresh/most_replayed.jsonl --out ytfresh_scores.json
```

The scorer turns each curve into one target per second: second *t* takes the value of the last bin that starts at or before *t*. It uses the metric code of the Mr.HiSum repository, so set that up first as described in [docs/data.md](../../docs/data.md).

Check `n_videos` and `n_missing_gt` in the output. Videos that are no longer on YouTube lower `n_videos` below 1,230. A prediction whose ID is not in `most_replayed.jsonl` is counted in `n_missing_gt`; it is not reported as an error.

## How the set was collected

- **Search.** Candidates were found through YouTube searches in English, Japanese, Korean, Spanish, Portuguese and Indonesian, over 15 category groups. The target number of videos per group follows Mr.HiSum's distribution over YouTube-8M verticals.
- **Filters.** A video was kept if it is 60–300 seconds long, has at least 50,000 views, was uploaded on or after July 1, 2023, and shows a public Most Replayed curve.
- **Popularity balance.** Within each group we aimed for equal numbers of videos with 50,000–499,999 views and with 500,000 or more.
- **Shortfalls.** Each group's quota was filled first. Groups with more eligible results then covered the quotas that other groups could not fill.
- **Availability.** Of 1,250 candidates, 1,230 could be downloaded. Eighteen had become private, had been deleted or belonged to terminated accounts, and two could not be resolved. Only the 1,230 are listed here.

## Composition

| Group | Videos | | Group | Videos | | Group | Videos |
|---|---:|---|---|---:|---|---|---:|
| `ent` | 326 | | `food` | 107 | | `net` | 18 |
| `games` | 248 | | `beauty` | 48 | | `science` | 16 |
| `pets` | 125 | | `hobby` | 28 | | `home` | 15 |
| `sports` | 120 | | `other` | 25 | | `biz` | 7 |
| `autos` | 118 | | `tech` | 22 | | `shop` | 7 |

These are the counts of the 1,230 released videos. The paper's supplementary table gives the targets and the counts of all 1,250 candidates.

| Uploaded | Videos |
|---|---:|
| July 2023 – December 2024 | 180 |
| January – March 2025 | 200 |
| April 2025 – August 2026 | 850 |

The 850 videos in the last row were uploaded after the public release of the backbone, Qwen2.5-Omni-7B, so they cannot be part of its pretraining data. Select them with `upload_date >= 2025-04-01`.

The median video is 196 seconds long (quartiles 149 and 246). The median view count is about 586,000 and the mean is 7.7 million, because a few videos are far more popular than the rest; 575 videos have fewer than 500,000 views and 655 have more.

## Known biases

- The category groups, YouTube's own categories and the YouTube-8M verticals of Mr.HiSum are related but different taxonomies, so the match to Mr.HiSum's distribution is approximate.
- Covering shortfalls from other groups raised the share of entertainment, pets and sports videos.
- Search favors popular uploads. Balancing the two view-count ranges reduces this but does not remove it.
- YouTube shows a Most Replayed curve only after a video has been watched enough, which filters for popularity again.

## Results in the paper

Zero-shot results of the three released SIMLENS checkpoints on all 1,230 videos. Scoring our saved predictions against `most_replayed.jsonl` with the command above gives these numbers. A new run differs slightly, because GPU kernels are not deterministic.

| Seed | mAP@50 | mAP@15 | Spearman |
|---|---:|---:|---:|
| 42 | 72.57 | 42.52 | 0.431 |
| 7 | 72.76 | 42.81 | 0.439 |
| 123 | 72.73 | 42.59 | 0.436 |
| Mean ± std | 72.69 ± 0.10 | 42.64 ± 0.15 | 0.435 ± 0.004 |

## License

The files in this folder are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/). If you use them, please cite the SIMLENS paper (see the [repository README](../../README.md#citation)).

The videos are not part of this release and remain the property of their owners. Titles, view counts and Most Replayed curves come from YouTube, and everyone who uses these files must comply with the [YouTube Terms of Service](https://www.youtube.com/static?template=terms). If you own a listed video and want it removed, write to ylchen@mgt.ncu.edu.tw.

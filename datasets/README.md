# Released data files

No videos are stored here. These files list the videos the paper used, so that results can be compared on the same videos.

## splits/

| File | Videos | Content |
|---|---:|---|
| `mrhisum_train_ids.txt` | 26,132 | Mr.HiSum training videos |
| `mrhisum_val_ids.txt` | 1,867 | Mr.HiSum validation videos |
| `mrhisum_test_ids.txt` | 1,874 | Mr.HiSum test videos |
| `mrhisum_paper_split.json` | | the three lists above as one split file, in the format the data loader reads |
| `ytmr500_ids.txt` | 406 | YTMR500 videos |

The text files have one ID per line.

- **Mr.HiSum IDs** are the keys of the official `mr_hisum.h5` and `mr_hisum_split.json`, for example `video_22692`, in the order of the official split. `metadata.csv` from [MR.HiSum](https://github.com/MRHiSum/MR.HiSum) maps them to YouTube IDs.
- **YTMR500 IDs** are YouTube IDs, as in [most-replayed-data](https://github.com/ombretta/most-replayed-data).

The lists are the official splits without the videos we could not download: 29,873 of the 31,892 Mr.HiSum videos and 406 of the 500 YTMR500 videos were available to us.

Our full-test results on Mr.HiSum, for SIMLENS and for the baselines we evaluated, are computed on the 1,874 test videos, and the zero-shot YTMR500 results on the 406. Results quoted from other papers and the smaller diagnostic experiments in the supplementary material use other sets of videos.

### Running on the paper's Mr.HiSum subset

Video availability changes over time, so the subset you can download may differ from ours.

The data loader does not read the text files. It reads the split file named by `paths.split` in the config and keeps the videos it finds in `data/mrhisum/videos`. To restrict a run to our subset, point `paths.split` at `datasets/splits/mrhisum_paper_split.json`.

For training, set it on the command line:

```bash
simlens-train --config simlens/configs/main.yaml --set paths.split=datasets/splits/mrhisum_paper_split.json --seed 42 --run-name my_run
```

`simlens-evaluate` has no `--set` option. Save a config that inherits from the main one, for example as `simlens/configs/paper_subset.yaml`:

```yaml
_base_: main.yaml
run_name: paper_subset
paths:
  split: datasets/splits/mrhisum_paper_split.json
```

```bash
simlens-evaluate --config simlens/configs/paper_subset.yaml --ckpt checkpoints/seed42 --split test
```

A listed video that is missing from your disk is skipped without an error. After the evaluation, compare the IDs in `outputs/paper_subset/eval_test/predictions.jsonl` with `mrhisum_test_ids.txt`; a count of 1,874 alone does not show that the videos are the same.

## ytfresh/

YT-Fresh, our test set of 1,230 recent YouTube videos: `metadata.csv` (ID, title, duration, upload date, views, categories) and `most_replayed.jsonl` (the 100-bin Most Replayed curve of each video). [ytfresh/README.md](ytfresh/README.md) describes the files, how the set was collected, how to predict and score it, and its license (CC BY 4.0).

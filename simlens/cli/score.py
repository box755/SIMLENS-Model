"""Compute the paper's metrics from saved per-video predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from ..training import metrics as slm
from ..training.official_map import mrhisum_ap, aggregate_map

# Values of k for which vP@k is reported
VP_KS = (1, 2, 3, 5)


def strict_vp_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    """
    Return vP@k: the fraction of the `k` lowest predicted seconds that are among the `k` lowest ground-truth seconds.
    """
    if pred.size < k:
        return float("nan")
    pb = set(np.argsort(pred, kind="stable")[:k].tolist())
    gb = set(np.argsort(gt, kind="stable")[:k].tolist())
    return len(pb & gb) / k


def ytmr_gt_to_seconds(heat: np.ndarray, T: int) -> np.ndarray:
    """Expand the 100 uniform bins of a `ytmr500` curve to `T` seconds."""
    idx = np.minimum((np.arange(T) / max(T, 1) * 100).astype(int), 99)
    return heat[idx]


def ytfresh_gt_to_seconds(starts: np.ndarray, vals: np.ndarray, T: int) -> np.ndarray:
    """Expand a `ytfresh` heatmap to `T` seconds: each second takes the value of the bin that contains it."""
    t = np.arange(T, dtype=np.float32)
    idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(vals) - 1)
    return vals[idx]


def load_gt(gt_name: str, gt_path: Path) -> dict[str, np.ndarray]:
    """
    Load the ground-truth curves of a dataset.

    Args:
        gt_name (`str`):
            `"mrhisum"`, `"ytmr500"` or `"ytfresh"`.
        gt_path (`Path`):
            Label file: an HDF5 file for `mrhisum` and `ytmr500`, a JSON Lines file for `ytfresh`.

    Returns:
        `dict`: Per-second curves for `mrhisum`, and binned curves for the other two, which are expanded to seconds
        when scored.
    """
    out = {}
    if gt_name == "mrhisum":
        with h5py.File(gt_path, "r") as f:
            for k in f:
                out[k] = np.asarray(f[k]["gtscore"], dtype=np.float64)
    elif gt_name == "ytmr500":
        with h5py.File(gt_path, "r") as f:
            for k in f:
                # 100 bins per video, expanded to seconds when scored
                out[k] = np.asarray(f[k]["heat-markers"], dtype=np.float64)
    elif gt_name == "ytfresh":
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                hm = r.get("heatmap")
                if not hm:
                    continue
                hm = sorted(hm, key=lambda h: h["start_time"])
                starts = np.asarray([h["start_time"] for h in hm], dtype=np.float32)
                vals = np.asarray([h["value"] for h in hm], dtype=np.float32)
                # Bin start times and values, expanded to seconds when scored
                out[r["youtube_id"]] = (starts, vals)
    else:
        raise ValueError(gt_name)
    return out


def score(pred_path: Path, gt_name: str, gt_path: Path) -> tuple[dict, list[dict]]:
    """
    Score a predictions file against the ground truth of a dataset.

    Args:
        pred_path (`Path`):
            JSON Lines file with one record per video, holding its id and its predicted curve.
        gt_name (`str`):
            `"mrhisum"`, `"ytmr500"` or `"ytfresh"`.
        gt_path (`Path`):
            Label file of the dataset.

    Returns:
        `tuple`: The metrics averaged over videos and the list of per-video rows.
    """
    gt_bank = load_gt(gt_name, gt_path)
    per_video, aps = [], []
    n_missing = 0
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            vid = rec.get("video_id") or rec.get("video_name") or rec.get("youtube_id")
            pred = np.asarray(rec.get("pred") or rec.get("s_hat"), dtype=np.float64).ravel()
            if vid not in gt_bank or pred.size == 0:
                n_missing += 1
                continue
            gt = gt_bank[vid]
            if gt_name == "ytmr500":
                gt = ytmr_gt_to_seconds(gt, len(pred))
            elif gt_name == "ytfresh":
                gt = ytfresh_gt_to_seconds(gt[0], gt[1], len(pred))
                # A constant ground-truth curve has no ranking to score
                if float(np.std(gt)) < 1e-8:
                    n_missing += 1
                    continue
            n = min(pred.size, gt.size)
            p, g = pred[:n], gt[:n]
            m = slm.curve_metrics(p, g)
            row = {"video_id": vid, "n_sec": n,
                   "spearman": m.get("spearman"), "top15_f1": m.get("top15_f1"),
                   "bottom15_f1": m.get("bottom15_f1")}
            for k in VP_KS:
                row[f"vp{k}"] = strict_vp_at_k(p, g, k)
            ap50, ap15 = mrhisum_ap(p, g)
            row["ap50"], row["ap15"] = ap50, ap15
            aps.append((ap50, ap15))
            per_video.append(row)

    def agg(key):
        xs = [r[key] for r in per_video if r[key] is not None and not math.isnan(r[key])]
        return float(np.mean(xs)) if xs else float("nan")

    sp = [r["spearman"] for r in per_video if r["spearman"] is not None and not math.isnan(r["spearman"])]
    mp = aggregate_map(aps)
    result = {
        "n_videos": len(per_video), "n_missing_gt": n_missing,
        "map50": mp["map50"], "map15": mp["map15"],
        "rho_vid": agg("spearman"),
        "pf1_15": agg("top15_f1"), "vf1_15": agg("bottom15_f1"),
        **{f"vp{k}": agg(f"vp{k}") for k in VP_KS},
        "rir": 100.0 * sum(1 for x in sp if x < 0) / len(sp) if sp else float("nan"),
    }
    return result, per_video


def main():
    ap = argparse.ArgumentParser(description="Unified paper metrics on saved per-video JSONL predictions")
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gt", required=True, choices=["mrhisum", "ytmr500", "ytfresh"])
    ap.add_argument("--gt-file", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("Output already exists; choose a new path")
    result, per_video = score(args.pred, args.gt, args.gt_file)
    if not per_video:
        ap.error("No videos were scored; check IDs, predictions and ground truth")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

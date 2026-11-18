"""Wrappers around the official Mr.HiSum implementation of mAP and keyshot F1."""

from __future__ import annotations

import sys
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def _mrhisum_root() -> Path:
    """Locate the official Mr.HiSum code: `MRHISUM_ROOT`, then `third_party/MR.HiSum`, then `../MR.HiSum`."""
    candidates = [Path("third_party/MR.HiSum"), Path("../MR.HiSum")]
    if os.environ.get("MRHISUM_ROOT"):
        candidates.insert(0, Path(os.environ["MRHISUM_ROOT"]))
    for c in candidates:
        if (c / "model" / "utils" / "evaluate_map.py").exists():
            return c.resolve()
    raise FileNotFoundError("Official Mr.HiSum code not found. Set MRHISUM_ROOT; see docs/data.md")


import torch


@lru_cache(maxsize=1)
def _official_functions():
    """Import the official Mr.HiSum evaluation functions on first use."""
    root = str(_mrhisum_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from model.utils.evaluate_map import generate_mrhisum_seg_scores, top50_summary, top15_summary
    from model.utils.generate_summary import generate_summary
    from model.utils.evaluation_metrics import evaluate_summary
    return generate_mrhisum_seg_scores, top50_summary, top15_summary, generate_summary, evaluate_summary


def mrhisum_ap(pred_per_sec, gt_per_sec) -> tuple[float, float]:
    """
    Compute the Mr.HiSum average precision of one video.

    Per-second scores are averaged over segments of five seconds, and the predicted segment scores are compared with
    the ground-truth top 50% and top 15% segments.

    Returns:
        `tuple`: The average precision at 50% and at 15%, or NaN when it is undefined.
    """
    generate_mrhisum_seg_scores, top50_summary, top15_summary, _, _ = _official_functions()
    pred = torch.as_tensor(np.asarray(pred_per_sec), dtype=torch.float32).flatten()
    gt = torch.as_tensor(np.asarray(gt_per_sec), dtype=torch.float32).flatten()
    n = min(pred.numel(), gt.numel())
    pred, gt = pred[:n], gt[:n]
    gt_seg = generate_mrhisum_seg_scores(gt, uniform_clip=5)
    gt50, gt15 = top50_summary(gt_seg), top15_summary(gt_seg)
    m_seg = generate_mrhisum_seg_scores(pred, uniform_clip=5)
    # Normalize the predicted segment scores with a softmax
    m_seg = torch.exp(m_seg) / (torch.exp(m_seg).sum() + 1e-7)
    m = m_seg.detach().cpu().numpy()
    try:
        return float(average_precision_score(gt50, m)), float(average_precision_score(gt15, m))
    except ValueError:
        return float("nan"), float("nan")


def aggregate_map(aps: list[tuple[float, float]]) -> dict:
    """Return the mean average precision at 50% and at 15% over videos, in percent."""
    a = np.array([x for x in aps if not (np.isnan(x[0]) or np.isnan(x[1]))], dtype=np.float64)
    if a.size == 0:
        return {"map50": float("nan"), "map15": float("nan"), "n": 0}
    return {"map50": float(a[:, 0].mean() * 100), "map15": float(a[:, 1].mean() * 100), "n": int(a.shape[0])}


def mrhisum_f1(pred_per_sec, n_frames: int, change_points, gt_summary) -> float:
    """
    Compute the Mr.HiSum keyshot F1 of one video.

    Shots are selected from the predicted scores with a knapsack over the given change points and compared with the
    ground-truth summary.

    Args:
        pred_per_sec (`np.ndarray` of shape `(num_seconds,)`):
            Predicted scores.
        n_frames (`int`):
            Length of the ground-truth curve. The prediction is padded or cut to this length.
        change_points (`np.ndarray` of shape `(num_shots, 2)`):
            Shot boundaries of the video.
        gt_summary (`np.ndarray`):
            Ground-truth summary of the video.

    Returns:
        `float`: The F1 score in percent, or NaN when it cannot be computed.
    """
    _, _, _, generate_summary, evaluate_summary = _official_functions()
    try:
        pred = np.asarray(pred_per_sec, dtype=np.float32).ravel()
        N = int(n_frames)
        # Pad or cut the prediction to the length of the ground truth
        if pred.shape[0] < N:
            pad = np.full(N - pred.shape[0], pred[-1] if pred.size else 0.0, dtype=np.float32)
            pred = np.concatenate([pred, pad])
        elif pred.shape[0] > N:
            pred = pred[:N]
        cps = np.asarray(change_points)
        nfps = [int(c[1] - c[0]) for c in cps]
        positions = np.arange(N)
        machine = generate_summary(pred, cps, [N], nfps, positions)
        gt = np.asarray(gt_summary)
        if gt.ndim == 1:
            gt = gt[None, :]
        # Pad or cut the selected summary to the length of the ground-truth summary
        L = gt.shape[1]
        if machine.shape[0] < L:
            machine = np.concatenate([machine, np.zeros(L - machine.shape[0], dtype=machine.dtype)])
        elif machine.shape[0] > L:
            machine = machine[:L]
        f, _kt, _ps = evaluate_summary(machine, gt, eval_method="avg")
        return float(f)
    except Exception:
        return float("nan")


def aggregate_f1(f1s: list[float]) -> dict:
    """Return the mean F1 over videos."""
    a = np.array([x for x in f1s if not np.isnan(x)], dtype=np.float64)
    return {"f1": float(a.mean()) if a.size else float("nan"), "n": int(a.size)}

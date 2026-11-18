"""Per-video metrics for replay curves and their aggregation over videos."""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def _safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    """Return the correlation computed by `fn`, or NaN when it is undefined."""
    if a.size < 2 or np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
        return float("nan")
    try:
        return float(fn(a, b)[0])
    except Exception:
        return float("nan")


def top_k_f1(pred: np.ndarray, target: np.ndarray, k_ratio: float = 0.15) -> float:
    """Return the F1 overlap between the predicted and the ground-truth top `k_ratio` seconds."""
    if pred.size == 0:
        return float("nan")
    k = max(1, int(round(pred.size * k_ratio)))
    pt = set(np.argsort(-pred)[:k].tolist())
    tt = set(np.argsort(-target)[:k].tolist())
    if not pt or not tt:
        return float("nan")
    tp = len(pt & tt)
    prec, rec = tp / len(pt), tp / len(tt)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def bottom_k_f1(pred: np.ndarray, target: np.ndarray, k_ratio: float = 0.15) -> float:
    """Return the F1 overlap between the predicted and the ground-truth bottom `k_ratio` seconds."""
    if pred.size == 0:
        return float("nan")
    k = max(1, int(round(pred.size * k_ratio)))
    pb = set(np.argsort(pred)[:k].tolist())
    tb = set(np.argsort(target)[:k].tolist())
    if not pb or not tb:
        return float("nan")
    tp = len(pb & tb)
    prec, rec = tp / len(pb), tp / len(tb)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# Values of k for which `valley_hit@k` is reported
VALLEY_KS = (1, 2, 3, 5, 10)


def valley_hit_at_k(pred: np.ndarray, target: np.ndarray, k: int = 2) -> float:
    """Return the fraction of the `k` lowest predicted seconds that lie in the ground-truth bottom 15%."""
    if pred.size == 0:
        return float("nan")
    kk = max(1, k)
    pred_valleys = set(np.argsort(pred)[:kk].tolist())
    gt_bottom = set(np.argsort(target)[: max(1, int(round(pred.size * 0.15)))].tolist())
    return float(len(pred_valleys & gt_bottom) / len(pred_valleys))


def curve_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """
    Compute all metrics for one video.

    Args:
        pred (`np.ndarray` of shape `(num_seconds,)`):
            Predicted curve.
        target (`np.ndarray` of shape `(num_seconds,)`):
            Ground-truth curve. The two curves are cut to their common length.

    Returns:
        `dict`: Spearman, Pearson and Kendall correlations, the top-15% and bottom-15% F1, `valley_hit@k` and the
        number of seconds.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    n = min(p.size, t.size)
    p, t = p[:n], t[:n]
    if n == 0:
        return {}
    out = {
        "spearman": _safe_corr(spearmanr, p, t),
        "pearson": _safe_corr(pearsonr, p, t),
        "kendall": _safe_corr(kendalltau, p, t),
        "top15_f1": top_k_f1(p, t, 0.15),
        "bottom15_f1": bottom_k_f1(p, t, 0.15),
        "n_sec": float(n),
    }
    for k in VALLEY_KS:
        out[f"valley_hit@{k}"] = valley_hit_at_k(p, t, k)
    return out


def _agg(xs: list[float], fn) -> float:
    """Apply `fn` to the values that are not NaN."""
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(fn(xs)) if xs else float("nan")


def aggregate_curve_metrics(per_video: list[dict[str, float]]) -> dict[str, float]:
    """
    Aggregate per-video metrics over a set of videos.

    Returns:
        `dict`: The mean and median of every metric, the fraction of videos with a negative Spearman correlation
        (`spearman_pct_neg`) and `per_video_spearman`, the mean Spearman correlation used for checkpoint selection.
    """
    if not per_video:
        return {}
    keys = [k for k in per_video[0] if k != "n_sec"]
    out: dict[str, float] = {"n_videos": float(len(per_video))}
    for k in keys:
        vals = [d.get(k, float("nan")) for d in per_video]
        out[f"{k}_mean"] = _agg(vals, np.mean)
        out[f"{k}_median"] = _agg(vals, np.median)
    sp = [d.get("spearman", float("nan")) for d in per_video]
    sp = [x for x in sp if not math.isnan(x)]
    out["spearman_pct_neg"] = float(sum(1 for x in sp if x < 0) / len(sp)) if sp else float("nan")
    out["per_video_spearman"] = out.get("spearman_mean", float("nan"))
    return out


def baseline_curves(target: np.ndarray) -> dict[str, np.ndarray]:
    """Return two trivial predictions for a target: its mean and a linear ramp over time."""
    t = np.asarray(target, dtype=np.float64).ravel()
    n = t.size
    return {
        "predict_mean": np.full(n, float(t.mean()) if n else 0.0),
        "position_only": np.linspace(0, 1, n) if n else np.zeros(0),
    }

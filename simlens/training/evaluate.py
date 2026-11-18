"""Evaluation on a dataset split: curve metrics, trivial baselines and the official Mr.HiSum metrics."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from simlens.utils import ensure_dir, write_json, write_jsonl
from .metrics import curve_metrics, aggregate_curve_metrics, baseline_curves
from .official_map import mrhisum_ap, aggregate_map, mrhisum_f1, aggregate_f1
from ..models.simlens_model import sdpa_ctx, to_device
from ..models import assembly as A
from ..data.dataset import first_item
from ..data.labels import H5Labels


@torch.no_grad()
def evaluate_model(cfg, model, processor, dataset, run_dir):
    """
    Evaluate `model` on `dataset` and write the results to `run_dir`.

    Writes `eval_metrics.json`, `evaluation_skips.jsonl` and, when `eval.dump_predictions` is set, `predictions.jsonl`
    with the predicted and ground-truth curves of every video.

    Args:
        cfg (`types.SimpleNamespace`):
            Run configuration.
        model (`SIMLENSModel`):
            Model to evaluate.
        processor (`Qwen2_5OmniProcessor`):
            Processor of the backbone.
        dataset (`torch.utils.data.Dataset`):
            Videos to evaluate, limited to the first `eval.test_limit` items when that option is set.
        run_dir (`str` or `os.PathLike`):
            Output directory.

    Returns:
        `dict`: Aggregated metrics of the model and of the two baselines, the number of evaluated videos, and the
        official mAP and F1 when they are enabled.
    """
    run_dir = ensure_dir(run_dir)
    lim = getattr(cfg.eval, "test_limit", None)
    if lim and len(dataset) > lim:
        dataset = Subset(dataset, list(range(lim)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=int(getattr(cfg.data, "num_workers", 0)), collate_fn=first_item)
    model.eval()
    want_map = bool(getattr(cfg.eval, "official_map", False))
    want_f1 = bool(getattr(cfg.eval, "official_f1", want_map))
    h5 = H5Labels(cfg.paths.h5) if want_f1 else None
    per_video, base_mean, base_pos, preds, aps, f1s = [], [], [], [], [], []
    skips = []
    # `data.audio_silence` replaces the waveform with zeros while keeping the audio tokens
    silence = bool(getattr(cfg.data, "audio_silence", False))
    if silence:
        print("[eval] R1 MODE: audio waveform zeroed, audio tokens kept", flush=True)
    for item in tqdm(loader, desc="eval", dynamic_ncols=True):
        try:
            audio = item["audio"]
            if silence and audio is not None and getattr(audio, "size", 0) > 0:
                audio = np.zeros_like(audio)
            inputs, _ = A.build_inputs(processor, item["frames"], audio,
                                       fps_in=cfg.data.fps_in, seconds_per_chunk=cfg.data.seconds_per_chunk,
                                       max_px=cfg.data.max_pixels, use_audio=cfg.data.use_audio,
                                       task_prompt=getattr(cfg.data, "task_prompt", "video"))
            with sdpa_ctx():
                s_hat = model(to_device(inputs, "cuda"))
            T = min(s_hat.shape[0], len(item["gtscore"]))
            sh = s_hat[:T].float().cpu().numpy(); gt = np.asarray(item["gtscore"][:T], dtype=np.float64)
        except torch.cuda.OutOfMemoryError:
            skips.append({"video_id": item.get("video_id"), "reason": "CUDA OOM"})
            print(f"[eval] OOM skip {item.get('video_id', '?')}")
            torch.cuda.empty_cache(); continue
        except Exception as e:
            skips.append({"video_id": item.get("video_id"), "reason": type(e).__name__})
            print(f"[eval] skip {item.get('video_id','?')}: {type(e).__name__}: {str(e)[:80]}")
            continue
        per_video.append(curve_metrics(sh, gt))
        if want_map:
            aps.append(mrhisum_ap(sh, gt))
        if want_f1:
            vid = item["video_id"]
            f1s.append(mrhisum_f1(sh, len(item["gtscore"]), h5.change_points(vid), h5.gt_summary(vid)))
        # Trivial baselines scored against the same target
        b = baseline_curves(gt)
        base_mean.append(curve_metrics(b["predict_mean"], gt))
        base_pos.append(curve_metrics(b["position_only"], gt))
        if getattr(cfg.eval, "dump_predictions", True):
            preds.append({"video_id": item["video_id"], "s_hat": [round(float(x), 4) for x in sh.tolist()],
                          "gt": [round(float(x), 4) for x in gt.tolist()]})

    result = {
        "model": aggregate_curve_metrics(per_video),
        "baseline_predict_mean": aggregate_curve_metrics(base_mean),
        "baseline_position_only": aggregate_curve_metrics(base_pos),
        "n_videos": len(per_video),
        "official_map": aggregate_map(aps) if want_map else "skipped",
        "official_f1": aggregate_f1(f1s) if want_f1 else "skipped",
    }
    write_json(run_dir / "eval_metrics.json", result)
    write_jsonl(run_dir / "evaluation_skips.jsonl", skips)
    if preds:
        write_jsonl(run_dir / "predictions.jsonl", preds)
    model.train()
    return result

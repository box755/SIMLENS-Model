"""Train a SIMLENS configuration, then evaluate the best checkpoint."""

from __future__ import annotations

import argparse, os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from pathlib import Path

from simlens.utils import load_config, dict_to_ns, set_seed, ensure_dir, save_resolved_config, write_json
from ..models.simlens_model import build_simlens_model
from ..data.dataset import StreamingVideoDataset
from ..training.trainer import SIMLENSTrainer
from ..training.evaluate import evaluate_model


def _ds(cfg, split, limit=None):
    """Build the dataset of a split from the configured paths."""
    return StreamingVideoDataset(
        split, videos_dir=cfg.paths.videos, h5_path=cfg.paths.h5, meta_path=cfg.paths.metadata,
        split_path=cfg.paths.split, fps_in=cfg.data.fps_in, px=cfg.data.px,
        max_sec=cfg.data.max_sec, use_audio=cfg.data.use_audio, limit=limit)


def _coerce(v: str):
    """Convert a `--set` value from its string form to a bool, `None`, number, list or string."""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [] if not inner else [_coerce(x.strip()) for x in inner.split(",")]
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _apply_set(cfg_d: dict, overrides: list[str]):
    """Apply `--set` overrides given as dotted keys, such as `data.use_audio=false`."""
    for ov in overrides:
        key, _, val = ov.partition("=")
        node = cfg_d
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=None, help="override cfg.seed (for 3-seed CIs)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="dotted-key override, repeatable: --set data.px=84 --set readout=suffix")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    cfg_d = load_config(args.config)
    if args.run_name:
        cfg_d["run_name"] = args.run_name
    if args.seed is not None:
        cfg_d["seed"] = args.seed
    _apply_set(cfg_d, args.overrides)
    cfg = dict_to_ns(cfg_d)
    set_seed(cfg.seed)
    run_dir = ensure_dir(Path(cfg.output_dir) / cfg.run_name)
    save_resolved_config(cfg_d, run_dir / "config.resolved.yaml")

    model, proc = build_simlens_model(cfg)
    tlim = getattr(cfg.data, "train_limit", None)
    trainer = SIMLENSTrainer(cfg, model, proc, _ds(cfg, "train", tlim), _ds(cfg, "val"), run_dir)
    summary = trainer.train()

    # Evaluate the best checkpoint on the configured evaluation split
    if not args.no_eval:
        best = run_dir / "checkpoints" / "lora_best"
        if best.exists():
            model.load_adapter(best)
        summary["test"] = evaluate_model(cfg, model, proc, _ds(cfg, cfg.eval.split), run_dir)
        write_json(run_dir / "summary.json", summary)
    print("[train] done ->", run_dir)


if __name__ == "__main__":
    main()

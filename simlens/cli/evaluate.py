"""Evaluate a SIMLENS checkpoint on a dataset split."""

from __future__ import annotations

import argparse, os, json
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from pathlib import Path

from simlens.utils import load_config, dict_to_ns, set_seed, ensure_dir
from ..models.simlens_model import build_simlens_model
from ..data.dataset import StreamingVideoDataset
from ..training.evaluate import evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="lora_best dir; omit for an untrained baseline")
    ap.add_argument("--split", default=None)
    args = ap.parse_args()

    cfg_d = load_config(args.config)
    cfg = dict_to_ns(cfg_d)
    set_seed(cfg.seed)
    split = args.split or cfg.eval.split
    run_dir = ensure_dir(Path(cfg.output_dir) / cfg.run_name / f"eval_{split}")

    model, proc = build_simlens_model(cfg)
    if args.ckpt:
        model.load_adapter(args.ckpt); print(f"[eval] loaded {args.ckpt}")
    ds = StreamingVideoDataset(split, videos_dir=cfg.paths.videos, h5_path=cfg.paths.h5,
                               meta_path=cfg.paths.metadata, split_path=cfg.paths.split,
                               fps_in=cfg.data.fps_in, px=cfg.data.px, max_sec=cfg.data.max_sec,
                               use_audio=cfg.data.use_audio)
    res = evaluate_model(cfg, model, proc, ds, run_dir)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

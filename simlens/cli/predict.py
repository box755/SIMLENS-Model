"""Export a single video's unnormalized replay scores to JSON and CSV."""
import argparse
import csv
import json
from pathlib import Path

from ..inference import DEFAULT_CONFIG


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/seed42")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", required=True, help="JSON output path; a sibling CSV is also written")
    ap.add_argument("--max-sec", type=float, default=None, help="Input prefix cap; default 300 s from config")
    ap.add_argument("--memory-fraction", type=float, default=0.82)
    args = ap.parse_args()
    out = Path(args.out)
    if out.suffix.lower() != ".json":
        ap.error("--out must end in .json")
    if out.exists() or out.with_suffix(".csv").exists():
        ap.error("Output already exists; choose a new --out path")
    if not Path(args.video).is_file():
        ap.error("Input video does not exist")

    from ..inference import Predictor
    predictor = Predictor(args.checkpoint, args.config, args.memory_fraction)
    scores = predictor.predict(args.video, args.max_sec)
    result = {
        "video_id": Path(args.video).stem,
        "s_hat": scores.tolist(),
        "n_seconds": len(scores),
        "max_input_seconds": args.max_sec if args.max_sec is not None else predictor.cfg.data.max_sec,
        "score_scale": "relative replay intensity; not a probability",
        "position_settings": predictor.position_settings(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["second", "score"])
        writer.writerows(enumerate(scores.tolist()))
    print(f"Wrote {len(scores)} scores to {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

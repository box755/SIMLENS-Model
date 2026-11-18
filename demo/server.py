"""
SIMLENS demo server: upload a video and view its predicted replay curve.

Usage:
    python demo/server.py           # load the model
    python demo/server.py --mock    # serve synthetic curves without a GPU

Endpoints:
    GET  /              the demo page
    POST /api/predict   video file -> {"scores": [...], "duration_s": ..., ...}
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

HERE = Path(__file__).resolve().parent
# Make the `simlens` package importable when the server is run from a source checkout
sys.path.insert(0, str(HERE.parent))

CONFIG = HERE.parent / "simlens" / "configs" / "main.yaml"
CKPT = HERE.parent / "checkpoints" / "seed42"
# Only the first `MAX_SEC` seconds of an upload are analyzed
MAX_SEC = 300.0

app = FastAPI(title="SIMLENS demo")
STATE: dict = {"mock": True, "model": None, "proc": None, "cfg": None}


def load_model():
    """Load the predictor and switch the server out of mock mode."""
    from simlens.inference import Predictor
    predictor = Predictor(CKPT, CONFIG)
    STATE.update(predictor=predictor, mock=False)
    print("[demo] model ready", flush=True)


def probe_duration(path: Path) -> int:
    """Return the duration of a video in whole seconds."""
    import decord
    vr = decord.VideoReader(str(path), num_threads=2)
    fps = vr.get_avg_fps() or 30.0
    return max(int(len(vr) / fps), 1)


def mock_curve(T: int, seed: int = 0) -> np.ndarray:
    """Return a synthetic curve of `T` seconds for trying the page without the model."""
    rng = np.random.default_rng(seed + T)
    t = np.linspace(0, 1, T)
    s = (0.55 * np.exp(-((t - 0.04) ** 2) / 0.004)
         + 0.9 * np.sin(2 * np.pi * (rng.uniform(2, 4)) * t + rng.uniform(0, 6))
         + 0.5 * np.sin(2 * np.pi * (rng.uniform(5, 9)) * t + rng.uniform(0, 6)))
    s += rng.normal(0, 0.15, T)
    k = max(3, T // 40) | 1
    s = np.convolve(s, np.ones(k) / k, mode="same")
    return s.astype(np.float64)


def real_curve(path: Path) -> np.ndarray:
    """Predict the replay curve of a video file with the loaded model."""
    return STATE["predictor"].predict(path, MAX_SEC)


@app.get("/")
def index():
    """Serve the demo page."""
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Score an uploaded video and return its curve as JSON."""
    t0 = time.time()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        if STATE["mock"]:
            T = probe_duration(tmp_path)
            scores = mock_curve(T)
        else:
            scores = real_curve(tmp_path)
            T = len(scores)
        return {"scores": [round(float(x), 4) for x in scores],
                "duration_s": int(T),
                "infer_ms": int((time.time() - t0) * 1000),
                "mock": STATE["mock"], "max_input_seconds": MAX_SEC}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    global CKPT, CONFIG, MAX_SEC
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="serve a plausible fake curve (no GPU)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--max-sec", type=float, default=MAX_SEC)
    args = ap.parse_args()
    CKPT, CONFIG, MAX_SEC = args.checkpoint, args.config, args.max_sec
    if not np.isfinite(MAX_SEC) or MAX_SEC <= 0:
        ap.error("--max-sec must be positive and finite")
    if not args.mock:
        load_model()
    else:
        print("[demo] MOCK mode — curves are synthetic", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

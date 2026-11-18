# Local video-to-curve demo

From the repository root, install `pip install -e ".[demo]"` and download a checkpoint.

```bash
python demo/server.py --checkpoint checkpoints/seed42
```

Open http://127.0.0.1:7860. The default input cap is **300 seconds**; longer uploads are
scored only through that prefix. Use `--max-sec 1380` for the inspected long-video
setting if GPU memory permits. No decoder sliding windows are used.

`python demo/server.py --mock` previews the UI without loading the model. These curves
are synthetic and marked **MOCK**. The chart rescales raw model scores to 0–100 for
display; this is not a replay probability or an absolute engagement score.

This is a local research demo, not a hardened public upload service. Keep the default
loopback host. Add authentication, upload limits and deployment isolation before any
public hosting. Do not start a second model when the GPU is already occupied.

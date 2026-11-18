"""Single-video inference with a trained checkpoint."""

from pathlib import Path

import numpy as np

from .utils import load_config, dict_to_ns, set_seed

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "main.yaml"


def validate_checkpoint(path):
    """Check that `path` holds the adapter and head files of a checkpoint and return it as a `Path`."""
    path = Path(path)
    for filename in ("adapter_config.json", "adapter_model.safetensors", "head.pt"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Missing {path / filename}. Download a complete checkpoint from Releases.")
    return path


class Predictor:
    """
    Predicts the replay curve of a video file with a trained checkpoint.

    Args:
        checkpoint (`str` or `os.PathLike`):
            Directory with `adapter_config.json`, `adapter_model.safetensors` and `head.pt`.
        config (`str` or `os.PathLike`, *optional*, defaults to `DEFAULT_CONFIG`):
            Configuration file the checkpoint was trained with.
        memory_fraction (`float`, *optional*, defaults to 0.82):
            Fraction of the GPU memory the process may use.

    Example:

    ```python
    >>> from simlens.inference import Predictor

    >>> predictor = Predictor("checkpoints/seed42")
    >>> scores = predictor.predict("video.mp4")
    ```
    """

    def __init__(self, checkpoint, config=DEFAULT_CONFIG, memory_fraction=0.82):
        checkpoint = validate_checkpoint(checkpoint)
        if not 0 < memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0, 1]")
        from .models.simlens_model import build_simlens_model
        self.cfg = dict_to_ns(load_config(config))
        self.cfg.model.cuda_mem_fraction = memory_fraction
        set_seed(self.cfg.seed)
        self.model, self.processor = build_simlens_model(self.cfg)
        self.model.load_adapter(checkpoint)
        self.model.eval()

    def predict(self, video, max_seconds=None):
        """
        Predict one replay score per second of `video`.

        Args:
            video (`str` or `os.PathLike`):
                Video file.
            max_seconds (`float`, *optional*):
                Only the first `max_seconds` seconds are analyzed. Defaults to `data.max_sec` of the configuration.

        Returns:
            `np.ndarray` of shape `(num_seconds,)`: Relative replay scores. They are not probabilities.
        """
        import torch
        from .models.assembly import build_inputs
        from .models.simlens_model import sdpa_ctx, to_device
        from .data.stream_decode import decode_video_fps, decode_audio_16k

        video = Path(video)
        if not video.is_file():
            raise FileNotFoundError(video)
        cap = float(max_seconds if max_seconds is not None else self.cfg.data.max_sec)
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError("max_seconds must be a positive finite number")
        cfg = self.cfg
        frames = decode_video_fps(video, px=cfg.data.px, fps=cfg.data.fps_in, max_sec=cap)
        if len(frames) < 2:
            raise ValueError("Video must yield at least two frames at the configured sampling rate")
        audio = decode_audio_16k(video, max_sec=cap) if cfg.data.use_audio else None
        inputs, _ = build_inputs(
            self.processor, frames, audio, fps_in=cfg.data.fps_in,
            seconds_per_chunk=cfg.data.seconds_per_chunk, max_px=cfg.data.max_pixels,
            use_audio=cfg.data.use_audio, task_prompt=getattr(cfg.data, "task_prompt", "video"))
        with torch.no_grad(), sdpa_ctx():
            scores = self.model(to_device(inputs)).float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError("Non-finite model output; inspect hardware and dependency compatibility")
        return scores

    def position_settings(self):
        """Return the positional settings of the processor and the backbone used by this predictor."""
        cfg = self.model.thinker.config
        return {
            "processor_chunk_seconds": self.cfg.data.seconds_per_chunk,
            "positional_chunk_seconds": getattr(cfg, "seconds_per_chunk", None),
            "position_ids_per_second": getattr(cfg, "position_id_per_seconds", None),
            "backbone_revision": getattr(self.cfg.model, "revision", None),
        }

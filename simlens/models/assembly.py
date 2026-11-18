"""Builds the model inputs from decoded frames and audio with the Qwen2.5-Omni processor."""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np

# Visual tokens per second at 112 x 112 pixels
VISUAL_TOK_PER_SEC = 16
# Bounds on the number of pixels per frame
DEFAULT_MAX_PX = 112 * 112
DEFAULT_MIN_PX = 56 * 56


@contextmanager
def _full_length_audio(processor, audio):
    """
    Temporarily widen the audio feature extractor's window so that audio longer than the window is kept whole.

    The feature extractor pads or truncates audio to a fixed window, which is 300 s in the released preprocessor
    configuration. Inside this context the window is set to the audio length, rounded up to a whole second, and the
    original values are restored on exit. Audio that fits in the window is processed with the original settings.
    """
    extractor = getattr(processor, "feature_extractor", None)
    stock = (getattr(extractor, "n_samples", None), getattr(extractor, "nb_max_frames", None))
    if audio is None or None in stock:
        yield
        return
    rate = int(extractor.sampling_rate)
    needed = int(math.ceil(len(audio) / rate)) * rate
    if needed <= stock[0]:
        yield
        return
    extractor.n_samples, extractor.nb_max_frames = needed, needed // int(extractor.hop_length)
    try:
        yield
    finally:
        extractor.n_samples, extractor.nb_max_frames = stock


def build_inputs(
    processor,
    frames: np.ndarray,
    audio: np.ndarray | None,
    *,
    fps_in: float = 2.0,
    seconds_per_chunk: float = 1.0,
    max_px: int = DEFAULT_MAX_PX,
    min_px: int = DEFAULT_MIN_PX,
    use_audio: bool = True,
    task_prompt: str = "video",
):
    """
    Build the processor inputs for one video.

    Args:
        processor (`Qwen2_5OmniProcessor`):
            Processor of the backbone.
        frames (`np.ndarray` of shape `(num_frames, height, width, 3)`):
            Decoded frames as `uint8`, sampled at `fps_in` frames per second.
        audio (`np.ndarray`, *optional*):
            Mono waveform at 16 kHz as `float32`, or `None` for a video-only input.
        fps_in (`float`, *optional*, defaults to 2.0):
            Sampling rate of `frames`.
        seconds_per_chunk (`float`, *optional*, defaults to 1.0):
            Length in seconds of the groups in which the processor interleaves video and audio tokens.
        max_px (`int`, *optional*, defaults to `DEFAULT_MAX_PX`):
            Maximum number of pixels per frame.
        min_px (`int`, *optional*, defaults to `DEFAULT_MIN_PX`):
            Minimum number of pixels per frame.
        use_audio (`bool`, *optional*, defaults to `True`):
            Whether to include the audio track.
        task_prompt (`str`, *optional*, defaults to `"video"`):
            Text placed after the video in the chat template.

    Returns:
        `tuple`: The processor outputs as CPU tensors and the number of seconds covered by the video tokens.
    """
    use_audio = bool(use_audio and audio is not None and audio.size > 0)
    # The audio track is passed together with the video, so the conversation has a single video entry
    content = [{"type": "video", "video": frames}, {"type": "text", "text": task_prompt}]
    conv = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    kwargs = dict(
        text=text, videos=[frames], return_tensors="pt", padding=True,
        fps=fps_in, seconds_per_chunk=seconds_per_chunk, use_audio_in_video=use_audio,
        min_pixels=min_px, max_pixels=max_px,
    )
    if use_audio:
        kwargs["audio"] = [audio]
    with _full_length_audio(processor, audio if use_audio else None):
        inputs = processor(**kwargs)
    inputs["use_audio_in_video"] = use_audio
    # The temporal size of the video grid is the number of seconds
    vgt = inputs.get("video_grid_thw")
    n_seconds = int(vgt[0][0]) if vgt is not None else 0
    return inputs, n_seconds

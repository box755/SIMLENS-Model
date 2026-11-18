"""Dataset that decodes one whole video per item."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from .stream_decode import decode_video_fps, decode_audio_16k
from .labels import H5Labels, load_metadata, load_split, available_ids, video_path


class StreamingVideoDataset(Dataset):
    """
    Dataset of Mr.HiSum videos. Each item is one decoded video with its replay curve.

    Args:
        split (`str`):
            `"train"`, `"val"` or `"test"`.
        videos_dir (`str` or `os.PathLike`):
            Directory with the `<video_id>.mp4` files.
        h5_path (`str` or `os.PathLike`):
            Path to `mr_hisum.h5`.
        meta_path (`str` or `os.PathLike`):
            Path to `metadata.csv`.
        split_path (`str` or `os.PathLike`):
            Path to the split file.
        fps_in (`float`, *optional*, defaults to 2.0):
            Frame sampling rate.
        px (`int`, *optional*, defaults to 112):
            Side length in pixels of the decoded frames.
        max_sec (`float`, *optional*, defaults to 300):
            Only the first `max_sec` seconds of a video are decoded.
        use_audio (`bool`, *optional*, defaults to `True`):
            Whether to decode the audio track.
        limit (`int`, *optional*):
            Keep only the first `limit` videos.
    """

    def __init__(self, split, *, videos_dir, h5_path, meta_path, split_path,
                 fps_in=2.0, px=112, max_sec=300, use_audio=True, limit=None):
        self.videos_dir = videos_dir
        self.fps_in = fps_in
        self.px = px
        self.max_sec = max_sec
        self.use_audio = use_audio
        self.h5 = H5Labels(h5_path)
        self.meta = load_metadata(meta_path)
        ids = load_split(split_path)[split]
        self.ids = available_ids(ids, videos_dir, self.h5)
        if limit:
            self.ids = self.ids[:limit]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """Return a dict with the `video_id`, `frames`, `audio` and `gtscore` of one video."""
        vid = self.ids[i]
        p = video_path(self.videos_dir, vid)
        frames = decode_video_fps(p, px=self.px, fps=self.fps_in, max_sec=self.max_sec)
        audio = decode_audio_16k(p, max_sec=self.max_sec) if self.use_audio else None
        gt = self.h5.gtscore(vid)
        return {"video_id": vid, "frames": frames, "audio": audio, "gtscore": gt}


def first_item(batch):
    """Collate function for a batch size of one: return the single item."""
    return batch[0]

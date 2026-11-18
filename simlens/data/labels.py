"""Access to the Mr.HiSum labels, metadata and split files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np


def load_split(split_path: str | Path) -> dict[str, list[str]]:
    """Load the split file and return the video ids under `"train"`, `"val"` and `"test"`."""
    with open(split_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k.replace("_keys", ""): v for k, v in raw.items()}


def load_metadata(meta_path: str | Path) -> dict[str, dict]:
    """Load `metadata.csv` and return the YouTube id, duration, views and labels of each video id."""
    out: dict[str, dict] = {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            out[vid] = {
                "youtube_id": row.get("youtube_id", ""),
                "duration": int(float(row["duration"])) if row.get("duration") else None,
                "views": int(float(row["views"])) if row.get("views") else None,
                "labels": row.get("labels", ""),
            }
    return out


class H5Labels:
    """Reader for `mr_hisum.h5`. The file is opened on first access."""

    def __init__(self, h5_path: str | Path):
        self.path = str(h5_path)
        self._h5: h5py.File | None = None

    @property
    def h5(self) -> h5py.File:
        """The open HDF5 file."""
        # Opened on first access, so that each DataLoader worker gets its own handle
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def gtscore(self, video_id: str) -> np.ndarray:
        """Return the per-second replay curve of a video as `float32`, shape `(num_seconds,)`."""
        return np.asarray(self.h5[f"{video_id}/gtscore"], dtype=np.float32)

    def change_points(self, video_id: str) -> np.ndarray:
        """Return the shot boundaries of a video."""
        return np.asarray(self.h5[f"{video_id}/change_points"])

    def gt_summary(self, video_id: str) -> np.ndarray:
        """Return the ground-truth summary of a video."""
        return np.asarray(self.h5[f"{video_id}/gt_summary"])

    def has(self, video_id: str) -> bool:
        """Return whether the file has an entry for `video_id`."""
        return video_id in self.h5

    def close(self) -> None:
        """Close the file if it is open."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def video_path(videos_dir: str | Path, video_id: str) -> Path:
    """Return the path of the video file of `video_id`."""
    return Path(videos_dir) / f"{video_id}.mp4"


def available_ids(ids: list[str], videos_dir: str | Path, h5: H5Labels) -> list[str]:
    """Keep the ids that have both a video file in `videos_dir` and an entry in the label file."""
    vd = Path(videos_dir)
    return [v for v in ids if (vd / f"{v}.mp4").exists() and h5.has(v)]

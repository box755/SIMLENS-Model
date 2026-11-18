"""Selects the positions from which the per-second hidden states are read."""

from __future__ import annotations

import torch
from .assembly import VISUAL_TOK_PER_SEC


def group_last(video_positions: torch.Tensor, tok_per_sec: int = VISUAL_TOK_PER_SEC) -> torch.Tensor:
    """
    Return the position of the last visual token of each second.

    Args:
        video_positions (`torch.LongTensor` of shape `(num_video_tokens,)`):
            Sequence positions of the visual tokens, in order.
        tok_per_sec (`int`, *optional*, defaults to `VISUAL_TOK_PER_SEC`):
            Number of visual tokens per second.

    Returns:
        `torch.LongTensor` of shape `(num_seconds,)`: One position per second.
    """
    n_full = video_positions.numel() // tok_per_sec
    if n_full == 0:
        return video_positions[-1:].clone() if video_positions.numel() else video_positions
    g = video_positions[: n_full * tok_per_sec].view(n_full, tok_per_sec)
    return g[:, -1].contiguous()


def second_readout_index(input_ids: torch.Tensor, video_token_id: int,
                         *, tok_per_sec: int = VISUAL_TOK_PER_SEC) -> torch.Tensor:
    """Return the position of the last visual token of each second, located from `input_ids`."""
    pos = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    return group_last(pos, tok_per_sec)

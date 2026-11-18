"""Training objectives for per-second replay scores: listwise, pointwise and hybrid."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def z_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Standardize a curve to zero mean and unit variance over time."""
    mu = x.mean()
    sd = x.std(unbiased=False).clamp_min(eps)
    return (x - mu) / sd


def tv_penalty(s: torch.Tensor, order: int = 2) -> torch.Tensor:
    """
    Return the total variation of `s`: the sum of squared (`order=2`) or absolute (`order=1`) successive differences.
    """
    if s.numel() < 2:
        return s.new_zeros(())
    d = s[1:] - s[:-1]
    return d.pow(2).sum() if order == 2 else d.abs().sum()


def delta_loss(s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Return the mean squared error between the first differences of the prediction and of the target."""
    if s.numel() < 2:
        return s.new_zeros(())
    return F.mse_loss(s[1:] - s[:-1], g[1:] - g[:-1])


def _listmle(pred: torch.Tensor, target: torch.Tensor, descending: bool = True) -> torch.Tensor:
    """
    Return the ListMLE loss: the negative Plackett-Luce log-likelihood of the target ordering under `pred`.

    The loss is divided by the number of seconds so that videos of different lengths are comparable.
    """
    order = torch.argsort(target, descending=descending)
    ps = pred[order].float()
    # cum[i] = logsumexp(ps[i:])
    cum = torch.logcumsumexp(ps.flip(0), dim=0).flip(0)
    return (cum - ps).sum() / ps.numel()


def bi_listmle(
    s: torch.Tensor, g: torch.Tensor, *, jitter: float = 1e-6, gen: torch.Generator | None = None,
    bidirectional: bool = True,
) -> tuple[torch.Tensor, bool]:
    """
    Compute the bidirectional ListMLE loss.

    One pass ranks the seconds from the highest target to the lowest, a mirrored pass ranks them from the lowest to the
    highest, and the two are averaged.

    Args:
        s (`torch.Tensor` of shape `(num_seconds,)`):
            Predicted scores.
        g (`torch.Tensor` of shape `(num_seconds,)`):
            Target scores.
        jitter (`float`, *optional*, defaults to 1e-6):
            Standard deviation of the Gaussian noise added to the target to break ties.
        gen (`torch.Generator`, *optional*):
            Random generator for the noise.
        bidirectional (`bool`, *optional*, defaults to `True`):
            Whether to include the mirrored pass. If `False`, only the descending pass is used.

    Returns:
        `tuple`: The loss and a flag. The flag is `False`, and the loss zero, when the target is constant or has fewer
        than two seconds.
    """
    if s.numel() < 2 or torch.allclose(g, g[0].expand_as(g)):
        return s.new_zeros(()), False
    # Break ties in the target with a small amount of noise
    if jitter > 0:
        noise = torch.randn(g.shape, generator=gen, device=g.device, dtype=g.dtype) * jitter
        g = g + noise
    # Descending pass: highest targets first
    down = _listmle(s, g, descending=True)
    if not bidirectional:
        return down, True
    # Mirrored pass on the negated scores: lowest targets first
    up = _listmle(-s, -g, descending=True)
    return 0.5 * (down + up), True


# Values accepted by `loss.arm` in the configuration
ARMS = ("pointwise", "listwise", "hybrid")


def compute_loss(
    s: torch.Tensor,
    gtscore: torch.Tensor,
    *,
    arm: str = "pointwise",
    tv_weight: float = 0.05,
    tv_order: int = 2,
    delta_weight: float = 0.3,
    hybrid_listmle_weight: float = 0.5,
    jitter: float = 1e-6,
    gen: torch.Generator | None = None,
    bidirectional: bool = True,
) -> dict[str, torch.Tensor | bool]:
    """
    Compute the training loss for one video.

    The target is standardized per video before the loss is applied.

    Args:
        s (`torch.Tensor` of shape `(num_seconds,)`):
            Predicted scores.
        gtscore (`torch.Tensor` of shape `(num_seconds,)`):
            Ground-truth replay curve.
        arm (`str`, *optional*, defaults to `"pointwise"`):
            Objective to use: `"listwise"` (bidirectional ListMLE plus the first-difference loss), `"pointwise"`
            (Smooth L1 plus total variation) or `"hybrid"` (pointwise plus bidirectional ListMLE).
        tv_weight (`float`, *optional*, defaults to 0.05):
            Weight of the total-variation term.
        tv_order (`int`, *optional*, defaults to 2):
            Order of the total-variation term, see `tv_penalty`.
        delta_weight (`float`, *optional*, defaults to 0.3):
            Weight of the first-difference loss in the listwise objective.
        hybrid_listmle_weight (`float`, *optional*, defaults to 0.5):
            Weight of the bidirectional ListMLE term in the hybrid objective.
        jitter (`float`, *optional*, defaults to 1e-6):
            Standard deviation of the tie-breaking noise, see `bi_listmle`.
        gen (`torch.Generator`, *optional*):
            Random generator for the tie-breaking noise.
        bidirectional (`bool`, *optional*, defaults to `True`):
            Whether the ListMLE term includes the mirrored pass.

    Returns:
        `dict`: The total loss under `"loss"`, its components, and `"listmle_valid"`, which is `False` when the ListMLE
        term could not be computed.
    """
    if arm not in ARMS:
        raise ValueError(f"loss arm must be one of {ARMS}, got {arm!r}")
    g_z = z_norm(gtscore.float())
    s = s.float()
    out: dict[str, torch.Tensor | bool] = {}

    if arm == "pointwise":
        sl = F.smooth_l1_loss(s, g_z)
        tv = tv_penalty(s, order=tv_order)
        out["loss_smoothl1"] = sl
        out["loss_tv"] = tv
        out["loss"] = sl + tv_weight * tv
        out["listmle_valid"] = True

    elif arm == "listwise":
        bl, valid = bi_listmle(s, g_z, jitter=jitter, gen=gen, bidirectional=bidirectional)
        dl = delta_loss(s, g_z)
        out["loss_bilistmle"] = bl
        out["loss_delta"] = dl
        out["loss"] = bl + delta_weight * dl
        out["listmle_valid"] = valid

    else:
        sl = F.smooth_l1_loss(s, g_z)
        tv = tv_penalty(s, order=tv_order)
        bl, valid = bi_listmle(s, g_z, jitter=jitter, gen=gen, bidirectional=bidirectional)
        out["loss_smoothl1"] = sl
        out["loss_tv"] = tv
        out["loss_bilistmle"] = bl
        out["loss"] = sl + tv_weight * tv + hybrid_listmle_weight * bl
        out["listmle_valid"] = valid

    return out

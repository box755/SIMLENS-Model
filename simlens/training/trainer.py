"""Training loop for SIMLENS: one video per step, with periodic validation, checkpointing and early stopping."""

from __future__ import annotations

import math, time
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from simlens.utils import ensure_dir, write_json, append_jsonl
from .metrics import curve_metrics, aggregate_curve_metrics
from ..models.simlens_model import sdpa_ctx, to_device
from ..models import assembly as A, losses as L
from ..data.dataset import first_item

TRAIN_STATE = "train_state.pt"


def _build_scheduler(opt, cfg, total_steps):
    """Build a learning-rate schedule with linear warmup followed by cosine decay to zero."""
    horizon = getattr(cfg.train, "cosine_horizon_steps", None) or total_steps
    horizon = min(int(horizon), int(total_steps))
    warmup = max(1, int(cfg.optim.warmup_ratio * horizon))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        if step >= horizon:
            return 0.0
        prog = (step - warmup) / max(1, horizon - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class SIMLENSTrainer:
    """
    Trainer that optimizes a `SIMLENSModel` one video at a time.

    Gradients are accumulated over `train.grad_accum` videos per optimizer step. The model is validated every
    `train.eval_every_steps` steps, the checkpoint with the best mean per-video Spearman correlation is saved as
    `lora_best`, and training stops early after `train.early_stop_patience` evaluations without improvement. Training
    resumes automatically from `checkpoints/lora_last` when that directory exists.

    Args:
        cfg (`types.SimpleNamespace`):
            Run configuration.
        model (`SIMLENSModel`):
            Model to train.
        processor (`Qwen2_5OmniProcessor`):
            Processor of the backbone.
        train_ds (`torch.utils.data.Dataset`):
            Training set.
        val_ds (`torch.utils.data.Dataset`):
            Validation set.
        run_dir (`str` or `os.PathLike`):
            Directory that receives the checkpoints, `metrics.jsonl` and `summary.json`.
    """

    def __init__(self, cfg, model, processor, train_ds, val_ds, run_dir):
        self.cfg = cfg
        self.model = model
        self.proc = processor
        self.device = "cuda"
        self.run_dir = ensure_dir(run_dir)
        self.ckpt_dir = ensure_dir(self.run_dir / "checkpoints")
        self.metrics_path = self.run_dir / "metrics.jsonl"

        nw = int(getattr(cfg.data, "num_workers", 0))
        self.train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=nw,
                                       collate_fn=first_item, drop_last=True)
        # Validate on the first `train.val_limit` videos only
        vlim = getattr(cfg.train, "val_limit", None)
        if vlim and len(val_ds) > vlim:
            val_ds = Subset(val_ds, list(range(vlim)))
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw,
                                     collate_fn=first_item)

        params = [p for p in model.parameters() if p.requires_grad]
        ratio = getattr(cfg.optim, "loraplus_ratio", None)
        # Optional separate learning rate for the LoRA B matrices
        if ratio:
            from peft.optimizers import create_loraplus_optimizer
            self.opt = create_loraplus_optimizer(
                model.peft, torch.optim.AdamW, lr=cfg.optim.lr, loraplus_lr_ratio=float(ratio),
                betas=tuple(cfg.optim.betas), weight_decay=cfg.optim.weight_decay)
            # The head and query embedding are outside the PEFT model and form their own parameter group
            extra = [p for n, p in model.named_parameters()
                     if p.requires_grad and not n.startswith("peft.")]
            if extra:
                self.opt.add_param_group({"params": extra, "lr": cfg.optim.lr,
                                          "weight_decay": cfg.optim.weight_decay})
            self.log(f"LoRA+ optimizer: ratio={ratio}, {len(extra)} non-LoRA tensors "
                     f"(head/query_embed) added at base lr")
        else:
            self.opt = torch.optim.AdamW(params, lr=cfg.optim.lr, betas=tuple(cfg.optim.betas),
                                         weight_decay=cfg.optim.weight_decay)
        self.params = params
        # Optimizer steps per epoch
        spe = math.ceil(len(self.train_loader) / cfg.train.grad_accum)
        self.total_steps = int(getattr(cfg.train, "max_steps", None) or spe * cfg.train.epochs)
        self.sched = _build_scheduler(self.opt, cfg, self.total_steps)

        self.best = float("-inf")
        self.bad_evals = 0
        self.start_step = 0
        self._resume()

    def log(self, msg):
        """Print a message with the `[trainer]` prefix."""
        print(f"[trainer] {msg}", flush=True)

    def metric(self, split, step, **kw):
        """Append one record to `metrics.jsonl`."""
        rec = {"split": split, "step": step, **{k: (round(float(v), 5) if isinstance(v, (int, float)) else v)
                                                 for k, v in kw.items()}}
        if getattr(self.cfg.logging, "metrics_jsonl", True):
            append_jsonl(self.metrics_path, rec)

    def _save(self, tag, *, train_state=False, meta=None):
        """Save the adapter under `checkpoints/<tag>`, optionally with the optimizer and scheduler state."""
        d = ensure_dir(self.ckpt_dir / tag)
        self.model.save_adapter(d)
        if meta:
            write_json(d / "meta.json", meta)
        if train_state:
            torch.save({"optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
                        "step": self.last_step, "best": self.best}, d / TRAIN_STATE)

    def _resume(self):
        """Restore the adapter and, when available, the optimizer and scheduler state from `checkpoints/lora_last`."""
        last = self.ckpt_dir / "lora_last"
        if last.exists() and any(last.iterdir()):
            try:
                self.model.load_adapter(last)
                ts = last / TRAIN_STATE
                if ts.exists():
                    st = torch.load(ts, map_location="cpu")
                    self.opt.load_state_dict(st["optimizer"]); self.sched.load_state_dict(st["scheduler"])
                    self.start_step = int(st.get("step", 0)); self.best = float(st.get("best", float("-inf")))
                    self.log(f"AUTO-RESUME from step {self.start_step}, best={self.best:.4f}")
                else:
                    self.log("warm-start: loaded adapter+head only")
            except Exception as e:
                self.log(f"resume failed ({e}); fresh start")

    def _forward(self, item):
        """Run the model on one video and return the predictions and targets cut to a common length."""
        inputs, _ = A.build_inputs(self.proc, item["frames"], item["audio"],
                                   fps_in=self.cfg.data.fps_in, seconds_per_chunk=self.cfg.data.seconds_per_chunk,
                                   max_px=self.cfg.data.max_pixels, use_audio=self.cfg.data.use_audio,
                                   task_prompt=getattr(self.cfg.data, "task_prompt", "video"))
        inputs = to_device(inputs, self.device)
        s_hat = self.model(inputs)
        T = min(s_hat.shape[0], len(item["gtscore"]))
        gt = torch.tensor(item["gtscore"][:T], device=self.device)
        return s_hat[:T], gt

    @torch.no_grad()
    def _validate(self, step):
        """Evaluate on the validation set and return the aggregated curve metrics."""
        self.model.eval()
        per_video = []
        for item in tqdm(self.val_loader, desc=f"val@{step}", leave=False, dynamic_ncols=True):
            try:
                with sdpa_ctx():
                    s_hat, gt = self._forward(item)
                per_video.append(curve_metrics(s_hat.float().cpu().numpy(), gt.float().cpu().numpy()))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
        m = aggregate_curve_metrics(per_video)
        self.metric("val", step, **m)
        self.model.train()
        return m

    def train(self):
        """Run the training loop and return the run summary."""
        cfg = self.cfg
        self.log(f"trainable={sum(p.numel() for p in self.params):,}  total_steps={self.total_steps}  arm={cfg.loss.arm}")
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        step = self.last_step = self.start_step
        micro = 0; consec_oom = 0; lw = []
        pbar = tqdm(total=self.total_steps, initial=step, desc="train", unit="step", dynamic_ncols=True)
        ga = cfg.train.grad_accum
        for epoch in range(cfg.train.epochs):
            for item in self.train_loader:
                if step >= self.total_steps:
                    break
                try:
                    with sdpa_ctx():
                        s_hat, gt = self._forward(item)
                        out = L.compute_loss(s_hat, gt, arm=cfg.loss.arm, tv_weight=cfg.loss.tv_weight,
                                             delta_weight=cfg.loss.delta_weight,
                                             hybrid_listmle_weight=cfg.loss.hybrid_listmle_weight,
                                             bidirectional=bool(getattr(cfg.loss, "bidirectional", True)))
                        loss = out["loss"]
                        (loss / ga).backward()
                    lw.append(float(out["loss"].detach())); micro += 1; consec_oom = 0
                except torch.cuda.OutOfMemoryError as e:
                    # Skip videos that do not fit in memory
                    self.opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    consec_oom += 1
                    self.log(f"OOM skip {item['video_id']} (n={len(item['gtscore'])}) [{consec_oom}]")
                    if consec_oom >= int(getattr(cfg.train, "max_consec_oom", 10)):
                        self.log("too many consecutive OOMs — aborting"); return self._finalize(step)
                    continue

                # Optimizer step after `grad_accum` videos
                if micro % ga == 0:
                    torch.nn.utils.clip_grad_norm_(self.params, cfg.train.max_grad_norm)
                    self.opt.step(); self.sched.step(); self.opt.zero_grad(set_to_none=True)
                    step += 1; self.last_step = step; pbar.update(1)

                    if step % cfg.train.log_every_steps == 0:
                        avg = sum(lw) / max(1, len(lw)); lw.clear()
                        postfix = {"loss": f"{avg:.3f}", "lr": f"{self.sched.get_last_lr()[0]:.2e}"}
                        pbar.set_postfix(**postfix)
                        self.metric("train", step, loss=avg, lr=self.sched.get_last_lr()[0], epoch=epoch)

                    # Validate, keep the best checkpoint and stop early when validation no longer improves
                    if step % cfg.train.eval_every_steps == 0:
                        m = self._validate(step)
                        cur = m.get("per_video_spearman", float("-inf"))
                        if cur > self.best + cfg.train.early_stop_min_delta:
                            self.best = cur; self.bad_evals = 0
                            self._save("lora_best", meta={"step": step, **m})
                            self.log(f"new best per_video_spearman={cur:.4f} @ step {step}")
                        else:
                            self.bad_evals += 1
                            if self.bad_evals >= cfg.train.early_stop_patience:
                                self.log(f"early stop @ step {step} (best={self.best:.4f})")
                                return self._finalize(step)

                    if step % cfg.train.save_every_steps == 0:
                        self._save("lora_last", train_state=True, meta={"step": step})
        return self._finalize(step)

    def _finalize(self, step):
        """Save the last checkpoint, run a final validation and write `summary.json`."""
        self.last_step = step
        self._save("lora_last", train_state=True, meta={"step": step})
        final = self._validate(step)
        summary = {"steps": step, "best_per_video_spearman": self.best, "final_val": final}
        write_json(self.run_dir / "summary.json", summary)
        self.log(f"done. best per_video_spearman={self.best:.4f}")
        return summary

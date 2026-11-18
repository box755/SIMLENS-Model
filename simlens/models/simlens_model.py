"""SIMLENS model: the Qwen2.5-Omni Thinker with LoRA adapters, suffix queries and a linear replay head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend

from .readout import second_readout_index, group_last
from .assembly import VISUAL_TOK_PER_SEC

_SDP = [SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]
# Attention backends used by the frozen vision and audio encoders
_ENC_SDP = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def to_device(inputs: dict, device="cuda") -> dict:
    """Move the processor outputs to `device` and cast floating-point tensors to bfloat16."""
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, torch.bfloat16) if torch.is_floating_point(v) else v.to(device)
        else:
            out[k] = v
    return out


def sdpa_ctx():
    """
    Return a context manager that selects the cuDNN kernel for scaled dot-product attention.

    The training and evaluation loops run the forward and backward passes inside this context, so that the attention
    recomputed by gradient checkpointing uses the same kernel as the forward pass.
    """
    return sdpa_kernel([SDPBackend.CUDNN_ATTENTION])


class SIMLENSModel(nn.Module):
    """
    Qwen2.5-Omni Thinker with LoRA adapters that predicts one replay score per second of video.

    The frozen vision and audio encoders embed the inputs, the LoRA-adapted decoder processes the interleaved sequence,
    and a linear head maps one hidden state per second to a score. With `readout="suffix"`, the hidden states are read
    from query tokens appended after the content, one per second. With `readout="causal"`, they are read from the last
    visual token of each second.

    Args:
        thinker_peft (`peft.PeftModel`):
            The Thinker wrapped by PEFT, used to save and load the adapter.
        thinker (`Qwen2_5OmniThinkerForConditionalGeneration`):
            The underlying Thinker with the LoRA layers injected.
        hidden_size (`int`):
            Hidden size of the decoder.
        video_token_id (`int`):
            Id of the video placeholder token.
        audio_token_id (`int`):
            Id of the audio placeholder token.
        head_dropout (`float`, *optional*, defaults to 0.0):
            Dropout applied to the hidden states before the linear head.
        vision_chunk_t (`int`, *optional*, defaults to 0):
            If positive, videos longer than this many seconds are encoded in independent chunks of this length.
            Otherwise the whole video is encoded in one pass.
        audio_pool_factor (`int`, *optional*, defaults to 1):
            Number of consecutive audio embeddings averaged into one token. 1 keeps every embedding.
        readout (`str`, *optional*, defaults to `"causal"`):
            Where the per-second hidden states are read, `"suffix"` or `"causal"`.
    """

    def __init__(self, thinker_peft, thinker, hidden_size: int, video_token_id: int,
                 audio_token_id: int, *, head_dropout: float = 0.0, vision_chunk_t: int = 0,
                 audio_pool_factor: int = 1, readout: str = "causal"):
        super().__init__()
        self.peft = thinker_peft
        self.thinker = thinker
        self.head = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(head_dropout)
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.vision_chunk_t = vision_chunk_t
        self.audio_pool_factor = int(audio_pool_factor)
        self.readout_mode = readout
        # Embedding shared by all suffix query tokens
        self.query_embed = nn.Parameter(torch.randn(hidden_size) * 0.02)

    @torch.no_grad()
    def _video_features(self, pixel_values_videos, video_grid_thw):
        """Encode the video with the frozen vision encoder, without gradients."""
        with sdpa_kernel(_ENC_SDP):
            if self.vision_chunk_t and int(video_grid_thw[0][0]) > self.vision_chunk_t:
                t, h, w = (int(x) for x in video_grid_thw[0])
                per_t = h * w
                outs = []
                for s in range(0, t, self.vision_chunk_t):
                    ct = min(self.vision_chunk_t, t - s)
                    pv = pixel_values_videos[s * per_t:(s + ct) * per_t]
                    gt = torch.tensor([[ct, h, w]], device=video_grid_thw.device, dtype=video_grid_thw.dtype)
                    outs.append(self.thinker.get_video_features(pv, gt, return_dict=True).pooler_output)
                return torch.cat(outs, 0).detach()
            return self.thinker.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True).pooler_output.detach()

    # Mel frames per audio encoder call: 60 s at 100 frames per second
    AUDIO_CHUNK_MEL = 6000

    @torch.no_grad()
    def _audio_features(self, input_features, feature_attention_mask):
        """Encode the mel spectrogram with the frozen audio encoder in 60-second chunks, without gradients."""
        with sdpa_kernel(_ENC_SDP):
            if feature_attention_mask is None:
                return self.thinker.get_audio_features(
                    input_features, return_dict=True).last_hidden_state.detach()
            L = int(feature_attention_mask[0].sum())
            C = self.AUDIO_CHUNK_MEL
            # Chunk boundaries; a final chunk shorter than 2 s is merged into the previous one
            bounds = list(range(0, max(L, 1), C)) + [max(L, 1)]
            if len(bounds) >= 3 and bounds[-1] - bounds[-2] < 200:
                bounds.pop(-2)
            outs = []
            for s, e in zip(bounds[:-1], bounds[1:]):
                outs.append(self.thinker.get_audio_features(
                    input_features[..., s:e],
                    feature_attention_mask=feature_attention_mask[..., s:e],
                    return_dict=True).last_hidden_state)
            return torch.cat(outs, dim=0).detach()

    def _pool_audio(self, input_ids, embeds, position_ids):
        """
        Average consecutive audio embeddings in groups of `audio_pool_factor`.

        Visual and text tokens are kept as they are. Within each run of audio tokens, every group of up to
        `audio_pool_factor` embeddings is replaced by its mean, and the pooled token keeps the position ids of the
        middle token of the group.

        Args:
            input_ids (`torch.LongTensor` of shape `(1, sequence_length)`):
                Token ids of the unpooled sequence.
            embeds (`torch.Tensor` of shape `(1, sequence_length, hidden_size)`):
                Input embeddings of the unpooled sequence.
            position_ids (`torch.LongTensor` of shape `(3, 1, sequence_length)`):
                Temporal, height and width position ids of the unpooled sequence.

        Returns:
            `tuple`: The pooled embeddings of shape `(1, pooled_length, hidden_size)`, the position ids of shape `(3,
            1, pooled_length)` and the indices of the visual tokens in the pooled sequence.
        """
        f = self.audio_pool_factor
        ids = input_ids[0]
        L = int(ids.shape[0])
        is_aud = (ids == self.audio_token_id).tolist()
        is_vid = (ids == self.video_token_id)
        pos_src = []
        pass_out, pass_src = [], []
        pool_out, pool_src, pool_seg = [], [], []
        # Plan the pooled sequence: which source tokens feed each output token
        i = 0
        while i < L:
            if not is_aud[i]:
                k = len(pos_src); pos_src.append(i); pass_out.append(k); pass_src.append(i); i += 1
            else:
                j = i
                while j < L and is_aud[j]:
                    j += 1
                for g in range(i, j, f):
                    grp = list(range(g, min(g + f, j)))
                    k = len(pos_src); pos_src.append(grp[len(grp) // 2])
                    pool_src.extend(grp); pool_seg.extend([k] * len(grp))
                i = j
        out_len = len(pos_src)
        dev = embeds.device
        pos_src_t = torch.tensor(pos_src, device=dev)
        new_pos = position_ids[:, 0, :].index_select(1, pos_src_t).unsqueeze(1)
        new_emb = embeds.new_zeros((out_len, embeds.shape[-1]))
        if pass_out:
            new_emb[torch.tensor(pass_out, device=dev)] = embeds[0].index_select(0, torch.tensor(pass_src, device=dev))
        # Average the embeddings of each audio group
        if pool_src:
            seg = torch.tensor(pool_seg, device=dev)
            src = torch.tensor(pool_src, device=dev)
            new_emb.index_add_(0, seg, embeds[0].index_select(0, src))
            cnt = torch.zeros(out_len, device=dev, dtype=embeds.dtype).index_add_(
                0, seg, torch.ones(len(pool_seg), device=dev, dtype=embeds.dtype))
            new_emb = (new_emb / cnt.clamp_min(1.0).unsqueeze(1)).to(embeds.dtype)
        # Indices of the visual tokens in the pooled sequence
        vid_new = [k for k, s in zip(pass_out, pass_src) if bool(is_vid[s])]
        return new_emb.unsqueeze(0), new_pos, torch.tensor(vid_new, device=dev)

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        Predict the replay curve of one video.

        Args:
            inputs (`dict`):
                Output of `build_inputs`, moved to the model device with `to_device`.

        Returns:
            `torch.Tensor` of shape `(num_seconds,)`: One unnormalized replay score per second.
        """
        th = self.thinker
        input_ids = inputs["input_ids"]
        # A single unpadded sequence needs no attention mask
        attn = None
        embeds = th.get_input_embeddings()(input_ids)

        # Replace the audio placeholders with the audio encoder outputs
        if inputs.get("input_features") is not None:
            af = self._audio_features(inputs["input_features"], inputs.get("feature_attention_mask"))
            af = af.to(embeds.device, embeds.dtype)
            _, _, am = th.get_placeholder_mask(input_ids, inputs_embeds=embeds)
            embeds = embeds.masked_scatter(am, af)

        # Replace the video placeholders with the vision encoder outputs
        vf = self._video_features(inputs["pixel_values_videos"], inputs["video_grid_thw"])
        vf = vf.to(embeds.device, embeds.dtype)
        _, vm, _ = th.get_placeholder_mask(input_ids, inputs_embeds=embeds, video_features=vf)
        embeds = embeds.masked_scatter(vm, vf)

        # Temporal, height and width position ids of every token
        afl = inputs["feature_attention_mask"].sum(-1) if inputs.get("feature_attention_mask") is not None else None
        position_ids, _ = th.get_rope_index(
            input_ids, None, inputs.get("video_grid_thw"), attn,
            bool(inputs.get("use_audio_in_video", False)), afl, inputs.get("video_second_per_grid"),
        )

        # Index of the last visual token of each second, after audio pooling when it is enabled
        vgt = inputs.get("video_grid_thw")
        n_sec = int(vgt[0][0]) if vgt is not None else 0
        n_vid = int((input_ids[0] == self.video_token_id).sum())
        tps = max(1, n_vid // n_sec) if n_sec else VISUAL_TOK_PER_SEC
        if self.audio_pool_factor > 1 and inputs.get("input_features") is not None:
            embeds, position_ids, vid_idx = self._pool_audio(input_ids, embeds, position_ids)
            idx = group_last(vid_idx, tok_per_sec=tps)
        else:
            idx = second_readout_index(input_ids, self.video_token_id, tok_per_sec=tps)

        # Suffix readout: append one query token per second, each with the position ids of that second's last visual
        # token
        T = idx.shape[0]
        if self.readout_mode == "suffix" and T > 0:
            qpos = position_ids[:, 0, idx]
            qemb = self.query_embed.to(embeds.dtype).unsqueeze(0).expand(T, -1)
            embeds = torch.cat([embeds[0], qemb], dim=0).unsqueeze(0)
            position_ids = torch.cat([position_ids, qpos.unsqueeze(1)], dim=2)
            read = slice(-T, None)
        else:
            read = idx

        # Run the decoder and map the selected hidden states to scores
        out = th.model(attention_mask=attn, position_ids=position_ids,
                       inputs_embeds=embeds, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state[0]
        read_hidden = self.drop(hidden[read])
        s_hat = self.head(read_hidden).squeeze(-1)
        return s_hat

    def save_adapter(self, out_dir):
        """Save the LoRA adapter, the linear head and the query embedding to `out_dir`."""
        from pathlib import Path
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        self.peft.save_pretrained(str(out))
        blob = {"head": self.head.state_dict(), "query_embed": self.query_embed.detach().cpu()}
        torch.save(blob, out / "head.pt")

    def load_adapter(self, ckpt_dir):
        """Load the LoRA adapter, the linear head and the query embedding from `ckpt_dir`."""
        from pathlib import Path
        d = Path(ckpt_dir)
        for required in ("adapter_config.json", "adapter_model.safetensors", "head.pt"):
            if not (d / required).is_file():
                raise FileNotFoundError(f"Incomplete SIMLENS checkpoint: missing {d / required}")
        self.peft.load_adapter(str(d), adapter_name="default")
        hp = d / "head.pt"
        if hp.exists():
            blob = torch.load(hp, map_location="cpu", weights_only=True)
            if isinstance(blob, dict) and "head" in blob:
                self.head.load_state_dict(blob["head"])
                if "query_embed" in blob:
                    self.query_embed.data = blob["query_embed"].to(self.query_embed.device, self.query_embed.dtype)
                elif self.readout_mode == "suffix":
                    raise ValueError("Suffix checkpoint is missing its learned query embedding")
            else:
                if self.readout_mode == "suffix":
                    raise ValueError("Legacy head-only checkpoint cannot initialize suffix-query readout")
                self.head.load_state_dict(blob)


def build_simlens_model(cfg):
    """
    Load the Qwen2.5-Omni Thinker, add the LoRA adapters and wrap it in a `SIMLENSModel`.

    Args:
        cfg (`types.SimpleNamespace`):
            Configuration loaded from one of the YAML files in `simlens/configs`.

    Returns:
        `tuple`: The model and the `Qwen2_5OmniProcessor` that prepares its inputs.
    """
    import transformers
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    from peft import LoraConfig, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("SIMLENS inference requires a CUDA GPU with cuDNN SDPA support")
    # Limit the GPU memory of the process and enable the cuDNN attention kernel
    mem_frac = float(getattr(cfg.model, "cuda_mem_fraction", 0.90))
    torch.cuda.set_per_process_memory_fraction(mem_frac, 0)
    torch.backends.cuda.enable_cudnn_sdp(True)

    name = cfg.model.base_model
    revision = getattr(cfg.model, "revision", None)
    processor = Qwen2_5OmniProcessor.from_pretrained(name, revision=revision)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        name, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})

    # Check that the backbone has the positional settings the configuration expects
    for field, expected_name in (("seconds_per_chunk", "expected_positional_chunk_seconds"),
                                 ("position_id_per_seconds", "expected_position_ids_per_second")):
        expected = getattr(cfg.model, expected_name, None)
        actual = getattr(thinker.config, field, None)
        if expected is not None and actual != expected:
            raise ValueError(f"Backbone config {field}={actual}, expected {expected}; see docs/reproducibility.md")

    # The processor interleaves video and audio in groups of `data.seconds_per_chunk` seconds, while the position ids
    # are grouped by the backbone's own `seconds_per_chunk`. `model.sync_position_chunk` sets the second to the first.
    if bool(getattr(cfg.model, "sync_position_chunk", False)):
        prev = int(getattr(thinker.config, "seconds_per_chunk", -1))
        thinker.config.seconds_per_chunk = int(cfg.data.seconds_per_chunk)
        print(f"[model] sync_position_chunk: config.seconds_per_chunk {prev} -> "
              f"{thinker.config.seconds_per_chunk} (matches processor)", flush=True)
    else:
        print(f"[model] sync_position_chunk=off; config.seconds_per_chunk="
              f"{getattr(thinker.config, 'seconds_per_chunk', None)}, "
              f"processor chunk={cfg.data.seconds_per_chunk}", flush=True)

    # Freeze the backbone
    for p in thinker.parameters():
        p.requires_grad_(False)

    # Add LoRA layers to the decoder; the audio encoder is excluded by default
    lora = LoraConfig(
        r=cfg.lora.rank, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        exclude_modules=getattr(cfg.lora, "exclude_modules", r".*audio_tower.*"),
        bias=cfg.lora.bias, task_type="CAUSAL_LM",
        use_rslora=bool(getattr(cfg.lora, "use_rslora", False)))
    peft_model = get_peft_model(thinker, lora)
    # Readout-only training: keep the LoRA layers at their zero initialization and train only the query embedding and
    # the linear head
    if bool(getattr(cfg.lora, "freeze", False)):
        n_frozen = 0
        for n, p in peft_model.named_parameters():
            if "lora_" in n and p.requires_grad:
                p.requires_grad_(False)
                n_frozen += p.numel()
        print(f"[probe] LoRA frozen ({n_frozen:,} params); backbone is untouched, "
              f"only the head/query embedding train", flush=True)
    # Enable gradient checkpointing to reduce memory use during training
    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inner = peft_model.base_model.model

    H = inner.config.text_config.hidden_size
    model = SIMLENSModel(
        peft_model, inner, H, inner.config.video_token_id, inner.config.audio_token_id,
        head_dropout=float(getattr(cfg.model, "head_dropout", 0.0)),
        vision_chunk_t=int(getattr(cfg.model, "vision_chunk_t", 0)),
        audio_pool_factor=int(getattr(cfg.model, "audio_pool_factor", 1)),
        readout=getattr(cfg, "readout", "causal"),
    )
    model.head = model.head.to("cuda", torch.bfloat16)
    model.query_embed = nn.Parameter(model.query_embed.detach().to("cuda", torch.bfloat16))
    return model, processor

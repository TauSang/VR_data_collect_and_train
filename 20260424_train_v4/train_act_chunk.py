"""
ACT with Action Chunking — True Action Chunking with Transformers

Key differences from the original train_act.py:
1. Predicts a CHUNK of K future actions instead of a single action
2. Uses TransformerDecoder with learned action queries
3. Supports temporal ensembling during inference  
4. Two-stage training: VR pretrain → MuJoCo finetune
5. Optional CVAE bottleneck for multi-modal action distribution

Architecture:
  Encoder: obs_seq → TransformerEncoder → context features
  Decoder: K learned action queries + context → TransformerDecoder → K actions
"""

import argparse
import copy
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


# ── v4: EMA weights helper ─────────────────────────────────────────────
class ModelEMA:
    """Exponential moving average of model weights.
    Updates EMA params in-place after every optimizer step."""
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(mv.detach(), alpha=1.0 - d)
            else:
                v.copy_(mv)

from common import (
    build_action_names,
    build_feature_names,
    denormalize_act,
    ensure_dir,
    fit_normalizer,
    load_config,
    load_segments,
    save_json,
    set_seed,
    split_segments_by_episode,
    summarize_segments,
    normalize_obs,
    normalize_act,
    Normalizer,
    SegmentData,
)

from typing import List, Dict, Tuple


# ─── Action Chunking Dataset ───────────────────────────────────────

def make_chunk_dataset(
    segments: List[SegmentData],
    seq_len: int,
    chunk_size: int,
    normalizer: Normalizer,
    obs_clip_z: float,
    act_clip_z: float,
    stride: int = 1,
):
    """
    Build dataset for action chunking:
      obs_seq: (seq_len, obs_dim) — observation history
      act_chunk: (chunk_size, act_dim) — K future actions starting at last obs timestep
      
    Each sample: obs[t-seq_len+1:t+1] → act[t:t+chunk_size]
    If segment too short, pad act_chunk with zeros and create a mask.
    stride > 1 reduces sample correlation (anti-overfitting).
    """
    xs, ys, masks, success, weights, meta, src_ids = [], [], [], [], [], [], []

    for seg_id, seg in enumerate(segments, start=1):
        T = seg.obs.shape[0]
        if T < seq_len:
            continue

        obs_z = normalize_obs(seg.obs, normalizer, obs_clip_z)
        act_z = normalize_act(seg.act, normalizer, act_clip_z)
        act_dim = act_z.shape[1]

        for t in range(seq_len - 1, T, stride):
            obs_start = t - seq_len + 1
            obs_chunk = obs_z[obs_start:t + 1]  # (seq_len, obs_dim)

            # Future actions: act[t], act[t+1], ..., act[t+chunk_size-1]
            act_end = min(t + chunk_size, T)
            actual_chunk = act_z[t:act_end]  # (actual_len, act_dim)
            actual_len = actual_chunk.shape[0]

            # Pad if needed
            if actual_len < chunk_size:
                pad = np.zeros((chunk_size - actual_len, act_dim), dtype=np.float32)
                act_chunk = np.concatenate([actual_chunk, pad], axis=0)
                mask = np.concatenate([
                    np.ones(actual_len, dtype=np.float32),
                    np.zeros(chunk_size - actual_len, dtype=np.float32),
                ])
            else:
                act_chunk = actual_chunk
                mask = np.ones(chunk_size, dtype=np.float32)

            xs.append(obs_chunk)
            ys.append(act_chunk)
            masks.append(mask)
            success.append(float(seg.success))
            weights.append(float(seg.weights[t]))
            src_ids.append(int(getattr(seg, "source_id", 0)))
            meta.append({
                "segment_id": seg_id,
                "episode_id": seg.episode_id,
                "target_id": seg.target_id,
                "outcome": seg.outcome,
                "step_index": t,
            })

    if not xs:
        raise RuntimeError("No samples after chunk dataset construction.")

    return (
        np.stack(xs, axis=0).astype(np.float32),      # (N, seq_len, obs_dim)
        np.stack(ys, axis=0).astype(np.float32),      # (N, chunk_size, act_dim)
        np.stack(masks, axis=0).astype(np.float32),   # (N, chunk_size)
        np.asarray(success, dtype=np.float32),          # (N,)
        np.asarray(weights, dtype=np.float32),          # (N,)
        np.asarray(src_ids, dtype=np.int64),             # (N,)
        meta,
    )


# ─── Model ──────────────────────────────────────────────────────────

class ACTChunkMLP(nn.Module):
    """
    MLP-based Action Chunking: proven MLP backbone (like BC) that predicts
    K future actions. Combines BC's generalization with ACT's smoother 
    action execution via temporal ensembling.
    
    Uses only the LAST observation in the sequence, so seq_len is irrelevant.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        chunk_size: int,
        hidden_dims: list = None,
        dropout: float = 0.1,
        **kwargs,  # accept and ignore transformer kwargs for config compatibility
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk_size = chunk_size

        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev, act_dim * chunk_size)
        self.success_head = nn.Sequential(
            nn.Linear(prev, prev // 4),
            nn.GELU(),
            nn.Linear(prev // 4, 1),
        )

    def forward(self, obs_seq, act_chunk=None, domain_id=None):
        # Use only the last observation from the sequence
        if obs_seq.dim() == 3:
            obs = obs_seq[:, -1, :]  # (B, obs_dim)
        else:
            obs = obs_seq  # (B, obs_dim)
        h = self.backbone(obs)
        actions = self.action_head(h).view(-1, self.chunk_size, self.act_dim)
        success_logit = self.success_head(h).squeeze(-1)
        return {
            "actions": actions,
            "success_logit": success_logit,
            "kl_loss": torch.tensor(0.0, device=obs_seq.device),
        }


class _FiLMBlock(nn.Module):
    """Single MLP layer with FiLM modulation by a domain embedding.

    Forward: x → Linear → LayerNorm → (1+γ) * x + β → GELU → Dropout
    The γ, β are produced from a shared domain embedding. When domain_id is
    missing or points at the "default" domain, γ=0, β=0 via zero-init, so the
    block reduces to the vanilla MLP — training remains backward-compatible.
    """

    def __init__(self, in_dim, out_dim, cond_dim, dropout):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.film = nn.Linear(cond_dim, 2 * out_dim)
        # Zero-init FiLM so the adapter starts as identity; warm start preserves
        # pretrain weights exactly (critical for finetune).
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond):
        h = self.norm(self.linear(x))
        gb = self.film(cond)
        g, b = gb.chunk(2, dim=-1)
        h = (1.0 + g) * h + b
        return self.drop(self.act(h))


class ACTChunkMLPFiLM(nn.Module):
    """Domain-adaptive ACT-Chunk MLP (ACT-DAFiLM).

    Architecture innovation over plain ACT:
      - A shared MLP backbone learns the joint policy.
      - A small domain embedding table (num_domains × cond_dim) feeds FiLM
        blocks that modulate each hidden layer with (γ, β) scale/shift.
      - Training: each sample carries its source_id (0=MJ, 1=VR, ...).
      - Inference: default domain_id=0 (canonical/MuJoCo), so the eval env
        never sees VR-specific activations — VR acts as a pure auxiliary
        teacher.
      - FiLM is zero-initialized → at step 0 the model equals a plain MLP,
        giving seamless finetune from ACTChunkMLP checkpoints (shared
        state_dict keys for backbone linear/norm + action_head + success_head).

    Why this matters for VR→MuJoCo transfer:
      - Plain mixing: VR's residual domain bias contaminates the shared
        backbone and drags MuJoCo eval down (our 80% vs 76% mixed experiment).
      - DAFiLM: VR-specific bias is absorbed into γ_VR, β_VR, while the
        shared backbone still benefits from the extra VR gradient signal.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        chunk_size: int,
        hidden_dims: list = None,
        dropout: float = 0.1,
        num_domains: int = 2,
        cond_dim: int = 16,
        default_domain_id: int = 0,
        **kwargs,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk_size = chunk_size
        self.num_domains = num_domains
        self.default_domain_id = default_domain_id

        self.domain_emb = nn.Embedding(num_domains, cond_dim)
        nn.init.normal_(self.domain_emb.weight, std=0.02)

        self.blocks = nn.ModuleList()
        prev = obs_dim
        for h in hidden_dims:
            self.blocks.append(_FiLMBlock(prev, h, cond_dim, dropout))
            prev = h
        self.action_head = nn.Linear(prev, act_dim * chunk_size)
        self.success_head = nn.Sequential(
            nn.Linear(prev, prev // 4),
            nn.GELU(),
            nn.Linear(prev // 4, 1),
        )

    def forward(self, obs_seq, act_chunk=None, domain_id=None):
        if obs_seq.dim() == 3:
            obs = obs_seq[:, -1, :]
        else:
            obs = obs_seq
        B = obs.shape[0]
        if domain_id is None:
            dom = torch.full((B,), int(self.default_domain_id),
                             dtype=torch.long, device=obs.device)
        elif isinstance(domain_id, int):
            dom = torch.full((B,), int(domain_id),
                             dtype=torch.long, device=obs.device)
        else:
            dom = domain_id.to(obs.device).long()
        cond = self.domain_emb(dom)

        h = obs
        for blk in self.blocks:
            h = blk(h, cond)
        actions = self.action_head(h).view(-1, self.chunk_size, self.act_dim)
        success_logit = self.success_head(h).squeeze(-1)
        return {
            "actions": actions,
            "success_logit": success_logit,
            "kl_loss": torch.tensor(0.0, device=obs_seq.device),
        }

    @staticmethod
    def load_from_mlp_state(state_dict):
        """Map ACTChunkMLP state_dict keys to ACTChunkMLPFiLM keys.

        ACTChunkMLP.backbone is nn.Sequential of [Linear, LayerNorm, GELU, Dropout] × N.
        Indices per block: 0=Linear, 1=LayerNorm. So keys look like
          backbone.0.weight/bias  (Linear)
          backbone.1.weight/bias  (LayerNorm)
          backbone.4.weight/bias  (next Linear)
          ...
        FiLM block keys: blocks.<i>.linear.{weight,bias}, blocks.<i>.norm.{weight,bias}.
        action_head.* and success_head.* are named identically.
        """
        remapped = {}
        seq_step = 4  # layers per MLP block
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                parts = k.split(".")
                idx = int(parts[1])
                block_idx, in_block = divmod(idx, seq_step)
                tail = ".".join(parts[2:])
                if in_block == 0:  # Linear
                    remapped[f"blocks.{block_idx}.linear.{tail}"] = v
                elif in_block == 1:  # LayerNorm
                    remapped[f"blocks.{block_idx}.norm.{tail}"] = v
                else:
                    pass  # GELU / Dropout have no params
            else:
                remapped[k] = v
        return remapped


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1], :]


class ACTChunk(nn.Module):
    """
    Action Chunking with Transformers.
    
    Encoder: obs_seq (B, seq_len, obs_dim) → context (B, seq_len, d_model)
    Decoder: K learned action queries → (B, K, act_dim) action chunk
    
    Optional CVAE: during training, encode ground-truth action chunk into
    latent z, use it to condition the decoder. During inference, sample z~N(0,1).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        chunk_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_cvae: bool = False,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.use_cvae = use_cvae
        self.latent_dim = latent_dim

        # Observation encoder
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # CVAE components (optional)
        if use_cvae:
            # Encoder for action chunk → latent
            self.action_proj = nn.Linear(act_dim, d_model)
            cvae_enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.cvae_encoder = nn.TransformerEncoder(cvae_enc_layer, num_layers=2)
            self.mu_proj = nn.Linear(d_model, latent_dim)
            self.logvar_proj = nn.Linear(d_model, latent_dim)
            self.latent_proj = nn.Linear(latent_dim, d_model)

        # Decoder
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Action output head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, act_dim),
        )

        # Success prediction head (auxiliary)
        self.success_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def encode_obs(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Encode observation sequence → context features."""
        x = self.obs_proj(obs_seq)       # (B, T, d_model)
        x = self.pos_enc(x)
        return self.encoder(x)           # (B, T, d_model)

    def encode_actions(self, act_chunk: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CVAE encoder: encode gt actions + context → latent (mu, logvar)."""
        B = act_chunk.shape[0]
        act_tokens = self.action_proj(act_chunk)  # (B, K, d_model)
        # Combine context and action tokens
        combined = torch.cat([context, act_tokens], dim=1)  # (B, T+K, d_model)
        z = self.cvae_encoder(combined)
        # Pool over sequence
        z_pooled = z.mean(dim=1)  # (B, d_model)
        mu = self.mu_proj(z_pooled)
        logvar = self.logvar_proj(z_pooled)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, context: torch.Tensor, latent: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode action chunk from context (and optional latent)."""
        B = context.shape[0]
        queries = self.action_queries.expand(B, -1, -1)  # (B, K, d_model)

        if latent is not None:
            # Add latent to queries
            latent_feat = self.latent_proj(latent).unsqueeze(1)  # (B, 1, d_model)
            queries = queries + latent_feat

        # Cross-attend to context
        decoded = self.decoder(queries, context)  # (B, K, d_model)

        actions = self.action_head(decoded)  # (B, K, act_dim)

        # Success from last context token
        success_logit = self.success_head(context[:, -1, :]).squeeze(-1)  # (B,)

        return actions, success_logit

    def forward(
        self,
        obs_seq: torch.Tensor,
        act_chunk: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Training (act_chunk provided): encode obs → (optionally CVAE) → decode → actions
        Inference (act_chunk=None): encode obs → decode with z=0 → actions
        """
        context = self.encode_obs(obs_seq)  # (B, T, d_model)

        kl_loss = torch.tensor(0.0, device=obs_seq.device)

        if self.use_cvae and act_chunk is not None and self.training:
            mu, logvar = self.encode_actions(act_chunk, context)
            z = self.reparameterize(mu, logvar)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        elif self.use_cvae:
            z = torch.zeros(obs_seq.shape[0], self.latent_dim, device=obs_seq.device)
        else:
            z = None

        actions, success_logit = self.decode(context, z)

        return {
            "actions": actions,          # (B, K, act_dim)
            "success_logit": success_logit,  # (B,)
            "kl_loss": kl_loss,
        }


# ─── Training ──────────────────────────────────────────────────────

def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-6)


@torch.no_grad()
def evaluate(model, loader, device, success_loss_weight, kl_weight):
    model.eval()
    total_losses, action_mses, chunk_maes = [], [], []
    first_step_mses = []

    for obs_seq, act_chunk, mask, success, weights, domain_id in loader:
        obs_seq = obs_seq.to(device)
        act_chunk = act_chunk.to(device)
        mask = mask.to(device)
        success = success.to(device)
        weights = weights.to(device)
        domain_id = domain_id.to(device)

        out = model(obs_seq, act_chunk, domain_id=domain_id)
        pred_actions = out["actions"]

        # Masked MSE over chunk
        diff2 = (pred_actions - act_chunk) ** 2  # (B, K, act_dim)
        per_step_mse = diff2.mean(dim=-1)  # (B, K)
        masked_mse = (per_step_mse * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # (B,)

        # First step accuracy (most important for closed-loop)
        first_mse = diff2[:, 0, :].mean(dim=-1)  # (B,)

        bce = F.binary_cross_entropy_with_logits(
            out["success_logit"], success, reduction="none"
        )

        loss = _weighted_mean(masked_mse, weights) + \
               success_loss_weight * _weighted_mean(bce, weights) + \
               kl_weight * out["kl_loss"]

        total_losses.append(loss.item())
        action_mses.append(masked_mse.mean().item())
        first_step_mses.append(first_mse.mean().item())

    return {
        "total_loss": float(np.mean(total_losses)),
        "action_mse": float(np.mean(action_mses)),
        "first_step_mse": float(np.mean(first_step_mses)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--pretrained", default=None,
                        help="Path to pretrained checkpoint for stage-2 finetuning")
    parser.add_argument("--inherit-normalizer", action="store_true",
                        help="Use the normalizer saved in --pretrained instead of refitting on new data. "
                             "Essential for cross-domain finetune (VR -> MuJoCo or vice versa).")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze MLP backbone during finetune; only heads are updated.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    g1_joints = list(cfg["data"]["g1_joint_names"])

    set_seed(int(cfg["train"]["seed"]))
    segments, data_summary = load_segments(cfg)

    train_segments, val_segments, split_summary = split_segments_by_episode(
        segments, train_split=float(cfg["data"]["train_split"]),
        seed=int(cfg["train"]["seed"]),
    )

    # Normalizer: either inherit from pretrain (cross-domain finetune) or refit.
    normalizer = None
    pretrained_ckpt = None
    if args.pretrained:
        pretrained_path = Path(args.pretrained)
        if pretrained_path.exists():
            pretrained_ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            if args.inherit_normalizer and "normalizer" in pretrained_ckpt:
                normalizer = Normalizer.from_dict(pretrained_ckpt["normalizer"])
                print(f"[ACT-Chunk] Inherited normalizer from {pretrained_path}")
    if normalizer is None:
        normalizer = fit_normalizer(train_segments)
        print("[ACT-Chunk] Fitted fresh normalizer from training data")

    seq_len = int(cfg["data"]["seq_len"])
    act_cfg = cfg.get("act_chunk", cfg.get("act", {}))
    chunk_size = int(act_cfg.get("chunk_size", 10))

    stride = int(cfg["data"].get("stride", 1))
    print(f"[ACT-Chunk] Building dataset: seq_len={seq_len}, chunk_size={chunk_size}, stride={stride}")

    x_train, y_train, m_train, s_train, w_train, d_train, _ = make_chunk_dataset(
        train_segments, seq_len=seq_len, chunk_size=chunk_size,
        normalizer=normalizer,
        obs_clip_z=float(cfg["data"]["obs_clip_z"]),
        act_clip_z=float(cfg["data"]["act_clip_z"]),
        stride=stride,
    )
    x_val, y_val, m_val, s_val, w_val, d_val, _ = make_chunk_dataset(
        val_segments, seq_len=seq_len, chunk_size=chunk_size,
        normalizer=normalizer,
        obs_clip_z=float(cfg["data"]["obs_clip_z"]),
        act_clip_z=float(cfg["data"]["act_clip_z"]),
        stride=1,  # val always stride=1 for fair comparison
    )

    print(f"[ACT-Chunk] Train: {x_train.shape[0]} samples, Val: {x_val.shape[0]} samples")
    uniq_tr = np.unique(d_train, return_counts=True)
    print(f"[ACT-Chunk] Train domain histogram: {dict(zip(uniq_tr[0].tolist(), uniq_tr[1].tolist()))}")

    train_ds = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train),
        torch.from_numpy(m_train), torch.from_numpy(s_train),
        torch.from_numpy(w_train), torch.from_numpy(d_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val), torch.from_numpy(y_val),
        torch.from_numpy(m_val), torch.from_numpy(s_val),
        torch.from_numpy(w_val), torch.from_numpy(d_val),
    )

    batch_size = int(cfg["train"]["batch_size"])

    # Sampler: optional WeightedRandomSampler by source_id to balance MJ:VR ratio.
    sampler_name = str(cfg["train"].get("sampler", "random")).lower()
    train_sampler = None
    train_shuffle = True
    if sampler_name == "weighted_source":
        source_ratio_cfg = cfg["train"].get("source_ratio", {})
        # keys may be str or int; normalize to int
        source_ratio = {int(k): float(v) for k, v in source_ratio_cfg.items()}
        # Count frames per source in training split
        unique_src, counts = np.unique(d_train, return_counts=True)
        count_per_src = dict(zip(unique_src.tolist(), counts.tolist()))
        default_ratio = float(cfg["train"].get("source_ratio_default", 1.0))
        sample_w = np.zeros(d_train.shape[0], dtype=np.float64)
        for s, cnt in count_per_src.items():
            r = source_ratio.get(int(s), default_ratio)
            # per-sample weight = ratio / count_in_source -> each source gets `ratio` total mass
            sample_w[d_train == s] = r / max(int(cnt), 1)
        num_samples = int(cfg["train"].get("samples_per_epoch", d_train.shape[0]))
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=num_samples,
            replacement=True,
        )
        train_shuffle = False
        print(f"[ACT-Chunk] Using WeightedRandomSampler: source_ratio={source_ratio}, "
              f"count_per_src={count_per_src}, samples_per_epoch={num_samples}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle,
                              sampler=train_sampler,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    obs_dim = int(x_train.shape[-1])
    act_dim = int(y_train.shape[-1])

    backbone = str(act_cfg.get("backbone", "transformer"))
    
    if backbone == "mlp":
        model = ACTChunkMLP(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=chunk_size,
            hidden_dims=list(act_cfg.get("hidden_dims", [256, 256, 128])),
            dropout=float(act_cfg.get("dropout", 0.1)),
        )
        model_class_name = "ACTChunkMLP"
    elif backbone == "mlp_film":
        model = ACTChunkMLPFiLM(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=chunk_size,
            hidden_dims=list(act_cfg.get("hidden_dims", [256, 256, 128])),
            dropout=float(act_cfg.get("dropout", 0.1)),
            num_domains=int(act_cfg.get("num_domains", 2)),
            cond_dim=int(act_cfg.get("cond_dim", 16)),
            default_domain_id=int(act_cfg.get("default_domain_id", 0)),
        )
        model_class_name = "ACTChunkMLPFiLM"
    else:
        model = ACTChunk(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=chunk_size,
            d_model=int(act_cfg.get("d_model", 256)),
            nhead=int(act_cfg.get("nhead", 8)),
            num_encoder_layers=int(act_cfg.get("num_encoder_layers", 4)),
            num_decoder_layers=int(act_cfg.get("num_decoder_layers", 2)),
            dim_feedforward=int(act_cfg.get("dim_feedforward", 512)),
            dropout=float(act_cfg.get("dropout", 0.1)),
            use_cvae=bool(act_cfg.get("use_cvae", False)),
            latent_dim=int(act_cfg.get("latent_dim", 32)),
        )
        model_class_name = "ACTChunk"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ACT-Chunk] device={device}, params={num_params:,}, "
          f"backbone={backbone}, chunk_size={chunk_size}")

    # Load pretrained weights (stage-2 finetuning). The checkpoint was already read
    # above (pretrained_ckpt) so we reuse it here.
    if pretrained_ckpt is not None:
        pretrained_state = pretrained_ckpt["model"]
        # If loading an ACTChunkMLP checkpoint into ACTChunkMLPFiLM, remap keys.
        if isinstance(model, ACTChunkMLPFiLM) and any(k.startswith("backbone.")
                                                      for k in pretrained_state):
            pretrained_state = ACTChunkMLPFiLM.load_from_mlp_state(pretrained_state)
            print("[ACT-Chunk] Remapped ACTChunkMLP checkpoint keys → FiLM blocks")
        model_state = model.state_dict()
        loaded_keys, skipped_keys = [], []
        for key, param in pretrained_state.items():
            if key in model_state and model_state[key].shape == param.shape:
                model_state[key] = param
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        model.load_state_dict(model_state)
        model.to(device)
        print(f"[ACT-Chunk] Loaded {len(loaded_keys)}/{len(pretrained_state)} pretrained params")
        if skipped_keys:
            print(f"  Skipped: {skipped_keys[:5]}{'...' if len(skipped_keys)>5 else ''}")
        if args.freeze_backbone:
            # For FiLM model, freeze block linear+norm (equivalent to MLP backbone)
            frozen = 0
            if hasattr(model, "backbone"):
                for p in model.backbone.parameters():
                    p.requires_grad = False
                    frozen += p.numel()
            elif hasattr(model, "blocks"):
                for blk in model.blocks:
                    for p in blk.linear.parameters():
                        p.requires_grad = False
                        frozen += p.numel()
                    for p in blk.norm.parameters():
                        p.requires_grad = False
                        frozen += p.numel()
            print(f"[ACT-Chunk] Froze backbone ({frozen:,} params); heads/FiLM remain trainable")
    elif args.pretrained:
        print(f"[WARN] Pretrained not found: {args.pretrained}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    success_loss_weight = float(cfg["train"].get("success_loss_weight", 0.2))
    kl_weight = float(act_cfg.get("kl_weight", 0.01))
    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 1.0))
    obs_noise_std = float(cfg["train"].get("obs_noise_std", 0.0))

    # ── 20260425 improvements ──
    loss_type = str(cfg["train"].get("loss_type", "mse")).lower()
    huber_delta = float(cfg["train"].get("huber_delta", 0.05))
    target_jitter_std = float(cfg["train"].get("target_jitter_std", 0.0))
    target_jitter_prob = float(cfg["train"].get("target_jitter_prob", 0.0))
    # target_rel_* features occupy indices 22..30 (9 dims) in the 31-dim obs.
    target_jitter_idx = list(range(22, 31))
    # Jitter is applied in raw-obs space (metres); convert via normalizer obs_std.
    _obs_std_t = torch.tensor(normalizer.obs_std, dtype=torch.float32)
    topk_save = int(cfg["train"].get("save_topk", 1))
    print(f"[ACT-Chunk] loss_type={loss_type} huber_delta={huber_delta} "
          f"target_jitter_std={target_jitter_std} jitter_prob={target_jitter_prob} "
          f"save_topk={topk_save}")

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("early_stop_patience", 25))

    scheduler_name = cfg["train"].get("lr_scheduler", "none")
    warmup_epochs = int(cfg["train"].get("lr_warmup_epochs", 0))
    scheduler = None
    if scheduler_name == "cosine":
        cosine = CosineAnnealingLR(opt, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6)
        if warmup_epochs > 0:
            warmup = LambdaLR(opt, lr_lambda=lambda e: (e + 1) / max(warmup_epochs, 1))
            scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
            print(f"[ACT-Chunk] LR schedule: warmup {warmup_epochs} epochs -> cosine (eta_min=1e-6)")
        else:
            scheduler = cosine
            print("[ACT-Chunk] LR schedule: cosine (eta_min=1e-6)")

    out_root = root / args.out
    run_dir = out_root / "act_chunk" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    best_val_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    # ── v4: EMA + smoothness config ──
    ema_decay = float(cfg["train"].get("ema_decay", 0.0))
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None
    action_smooth_weight = float(cfg["train"].get("action_smooth_weight", 0.0))
    eval_with_ema = bool(cfg["train"].get("eval_with_ema", ema is not None))
    print(f"[ACT-Chunk] ema_decay={ema_decay} eval_with_ema={eval_with_ema} "
          f"action_smooth_weight={action_smooth_weight}")

    print(f"[ACT-Chunk] Training for {epochs} epochs, patience={patience}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses, train_mses, train_first_mses, train_kls = [], [], [], []

        for obs_seq, act_chunk, mask, success, weights, domain_id in train_loader:
            obs_seq = obs_seq.to(device)
            act_chunk = act_chunk.to(device)
            mask = mask.to(device)
            success = success.to(device)
            weights = weights.to(device)
            domain_id = domain_id.to(device)

            if obs_noise_std > 0:
                obs_seq = obs_seq + torch.randn_like(obs_seq) * obs_noise_std

            # ── 20260425: target-rel jitter in normalized-obs space ──
            if target_jitter_std > 0 and target_jitter_prob > 0:
                # sample per-example gate
                B = obs_seq.shape[0]
                gate = (torch.rand(B, device=obs_seq.device) < target_jitter_prob).float()
                # per-dim normalized std for target_rel indices
                std_t = _obs_std_t.to(obs_seq.device)
                norm_sigma = target_jitter_std / std_t[target_jitter_idx].clamp(min=1e-6)
                noise = torch.randn(B, len(target_jitter_idx),
                                    device=obs_seq.device) * norm_sigma
                if obs_seq.dim() == 3:
                    # broadcast across seq_len dim
                    noise = noise.unsqueeze(1).expand(-1, obs_seq.shape[1], -1)
                    gate3 = gate.view(B, 1, 1)
                    obs_seq[..., target_jitter_idx] = (
                        obs_seq[..., target_jitter_idx] + noise * gate3
                    )
                else:
                    gate2 = gate.view(B, 1)
                    obs_seq[:, target_jitter_idx] = (
                        obs_seq[:, target_jitter_idx] + noise * gate2
                    )

            out = model(obs_seq, act_chunk, domain_id=domain_id)
            pred_actions = out["actions"]

            # ── 20260425: configurable loss (MSE or Huber) ──
            if loss_type == "huber":
                diff_abs = (pred_actions - act_chunk).abs()
                # element-wise Huber, keep chunk/act-dim structure for masking
                hub = torch.where(
                    diff_abs <= huber_delta,
                    0.5 * diff_abs ** 2,
                    huber_delta * (diff_abs - 0.5 * huber_delta),
                )
                per_step_mse = hub.mean(dim=-1)  # (B, K)
                diff2 = (pred_actions - act_chunk) ** 2  # keep for reporting
            else:
                diff2 = (pred_actions - act_chunk) ** 2
                per_step_mse = diff2.mean(dim=-1)  # (B, K)

            masked_mse = (per_step_mse * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # (B,)

            first_mse = diff2[:, 0, :].mean(dim=-1)

            bce = F.binary_cross_entropy_with_logits(
                out["success_logit"], success, reduction="none"
            )

            loss = _weighted_mean(masked_mse, weights) + \
                   success_loss_weight * _weighted_mean(bce, weights) + \
                   kl_weight * out["kl_loss"]

            # ── v4: action smoothness regularizer (penalize twitch between chunk steps) ──
            if action_smooth_weight > 0 and pred_actions.shape[1] >= 2:
                diff_a = pred_actions[:, 1:, :] - pred_actions[:, :-1, :]
                # mask pairs where both steps are valid
                pair_mask = mask[:, 1:] * mask[:, :-1]
                smooth_per = (diff_a ** 2).mean(dim=-1)  # (B, K-1)
                smooth_mean = (smooth_per * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp(min=1)
                loss = loss + action_smooth_weight * _weighted_mean(smooth_mean, weights)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()

            # ── v4: EMA update after every step ──
            if ema is not None:
                ema.update(model)

            train_losses.append(loss.item())
            train_mses.append(masked_mse.mean().item())
            train_first_mses.append(first_mse.mean().item())
            train_kls.append(out["kl_loss"].item())

        if scheduler is not None:
            scheduler.step()

        # ── v4: use EMA weights for validation if configured ──
        eval_model = ema.ema if (ema is not None and eval_with_ema) else model
        val_metrics = evaluate(eval_model, val_loader, device, success_loss_weight, kl_weight)
        lr = opt.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": float(np.mean(train_losses)),
            "train_mse": float(np.mean(train_mses)),
            "train_first_mse": float(np.mean(train_first_mses)),
            "train_kl": float(np.mean(train_kls)),
            "val_loss": val_metrics["total_loss"],
            "val_mse": val_metrics["action_mse"],
            "val_first_mse": val_metrics["first_step_mse"],
        }
        history.append(row)

        print(
            f"[ACT-Chunk] epoch={epoch:03d}"
            f" lr={lr:.2e}"
            f" train_mse={row['train_mse']:.6f}"
            f" val_mse={row['val_mse']:.6f}"
            f" val_1st={row['val_first_mse']:.6f}"
        )

        checkpoint = {
            "model": (ema.ema.state_dict() if (ema is not None and eval_with_ema) else model.state_dict()),
            "model_raw": model.state_dict(),
            "epoch": epoch,
            "normalizer": normalizer.to_dict(),
            "config": cfg,
            "chunk_size": chunk_size,
            "model_class": model_class_name,
        }
        torch.save(checkpoint, ckpt_dir / "last.pt")

        if row["val_mse"] < best_val_mse:
            best_val_mse = row["val_mse"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(checkpoint, ckpt_dir / "best.pt")
            # ── 20260425: top-K best checkpoints for inference-time ensemble ──
            if topk_save > 1:
                # maintain list of (val_mse, epoch) for saved topk
                topk_file = ckpt_dir / "topk_index.json"
                import json as _json
                if topk_file.exists():
                    topk_list = _json.loads(topk_file.read_text())
                else:
                    topk_list = []
                topk_list.append({"epoch": epoch, "val_mse": row["val_mse"]})
                topk_list.sort(key=lambda r: r["val_mse"])
                topk_list = topk_list[:topk_save]
                # save current checkpoint as topk_<rank>.pt
                # rewrite all rank files from current disk content of best.pt
                # strategy: just copy current best into a new slot per new best
                # (we only enter this branch when val improved, so save as top1;
                # rotate existing top1..topk-1 to top2..topk).
                import shutil
                for i in range(min(topk_save, len(topk_list)) - 1, 0, -1):
                    src = ckpt_dir / f"top{i}.pt"
                    dst = ckpt_dir / f"top{i+1}.pt"
                    if src.exists():
                        shutil.copy2(src, dst)
                torch.save(checkpoint, ckpt_dir / "top1.pt")
                topk_file.write_text(_json.dumps(topk_list, indent=2))
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"[ACT-Chunk] early stop at epoch={epoch}")
            break

    # ── Save metrics ──
    save_json(run_dir / "metrics.json", {
        "model": "ACTChunk",
        "chunk_size": chunk_size,
        "seq_len": seq_len,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "feature_names": build_feature_names(g1_joints),
        "action_names": build_action_names(g1_joints),
        "best_epoch": best_epoch,
        "best_val_action_mse": best_val_mse,
        "num_train_samples": int(x_train.shape[0]),
        "num_val_samples": int(x_val.shape[0]),
        "num_trainable_params": num_params,
        "data_summary": data_summary,
        "train_summary": summarize_segments(train_segments),
        "val_summary": summarize_segments(val_segments),
        "split_summary": split_summary,
        "normalizer": normalizer.to_dict(),
        "history": history,
    })

    # ── Plots ──
    epochs_x = [h["epoch"] for h in history]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_x, [h["train_mse"] for h in history], label="train")
    plt.plot(epochs_x, [h["val_mse"] for h in history], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Chunk Action MSE")
    plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_x, [h["train_first_mse"] for h in history], label="train")
    plt.plot(epochs_x, [h["val_first_mse"] for h in history], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("First-Step MSE")
    plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_x, [h["lr"] for h in history])
    plt.xlabel("epoch"); plt.ylabel("LR"); plt.title("Learning Rate")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close()

    print(f"\n[ACT-Chunk] Done! best_epoch={best_epoch}, best_val_mse={best_val_mse:.6f}")
    print(f"  Outputs: {run_dir}")


if __name__ == "__main__":
    main()

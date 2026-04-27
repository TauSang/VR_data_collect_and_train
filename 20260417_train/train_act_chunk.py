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
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

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
    xs, ys, masks, success, weights, meta = [], [], [], [], [], []

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

    def forward(self, obs_seq, act_chunk=None):
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

    for obs_seq, act_chunk, mask, success, weights in loader:
        obs_seq = obs_seq.to(device)
        act_chunk = act_chunk.to(device)
        mask = mask.to(device)
        success = success.to(device)
        weights = weights.to(device)

        out = model(obs_seq, act_chunk)
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
    parser.add_argument("--inherit-normalizer", action="store_true", default=False,
                        help="Inherit normalizer from pretrained checkpoint (critical for cross-domain finetune)")
    parser.add_argument("--freeze-backbone", action="store_true", default=False,
                        help="Freeze backbone layers, only train action_head (and optionally last backbone block)")
    parser.add_argument("--unfreeze-last-n", type=int, default=0,
                        help="Number of backbone blocks to unfreeze (from the end). 0=freeze all backbone.")
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

    # Normalizer: inherit from pretrained checkpoint or fit from data
    pretrained_ckpt = None
    if args.pretrained:
        pretrained_path = Path(args.pretrained)
        if pretrained_path.exists():
            pretrained_ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    if args.inherit_normalizer and pretrained_ckpt and "normalizer" in pretrained_ckpt:
        norm_dict = pretrained_ckpt["normalizer"]
        normalizer = Normalizer.from_dict(norm_dict)
        print(f"[ACT-Chunk] Inherited normalizer from pretrained checkpoint")
    else:
        normalizer = fit_normalizer(train_segments)
        if args.inherit_normalizer and not pretrained_ckpt:
            print(f"[WARN] --inherit-normalizer set but no pretrained checkpoint; using fresh normalizer")

    seq_len = int(cfg["data"]["seq_len"])
    act_cfg = cfg.get("act_chunk", cfg.get("act", {}))
    chunk_size = int(act_cfg.get("chunk_size", 10))

    stride = int(cfg["data"].get("stride", 1))
    print(f"[ACT-Chunk] Building dataset: seq_len={seq_len}, chunk_size={chunk_size}, stride={stride}")

    x_train, y_train, m_train, s_train, w_train, _ = make_chunk_dataset(
        train_segments, seq_len=seq_len, chunk_size=chunk_size,
        normalizer=normalizer,
        obs_clip_z=float(cfg["data"]["obs_clip_z"]),
        act_clip_z=float(cfg["data"]["act_clip_z"]),
        stride=stride,
    )
    x_val, y_val, m_val, s_val, w_val, _ = make_chunk_dataset(
        val_segments, seq_len=seq_len, chunk_size=chunk_size,
        normalizer=normalizer,
        obs_clip_z=float(cfg["data"]["obs_clip_z"]),
        act_clip_z=float(cfg["data"]["act_clip_z"]),
        stride=1,  # val always stride=1 for fair comparison
    )

    print(f"[ACT-Chunk] Train: {x_train.shape[0]} samples, Val: {x_val.shape[0]} samples")

    train_ds = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train),
        torch.from_numpy(m_train), torch.from_numpy(s_train),
        torch.from_numpy(w_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val), torch.from_numpy(y_val),
        torch.from_numpy(m_val), torch.from_numpy(s_val),
        torch.from_numpy(w_val),
    )

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
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

    # Load pretrained weights (stage-2 finetuning)
    if args.pretrained and pretrained_ckpt:
        pretrained_path = Path(args.pretrained)
        print(f"[ACT-Chunk] Loading pretrained weights: {pretrained_path}")
        pretrained_state = pretrained_ckpt["model"]

        # Handle dimension mismatch for two-stage training
        model_state = model.state_dict()
        loaded_keys = []
        skipped_keys = []
        for key, param in pretrained_state.items():
            if key in model_state and model_state[key].shape == param.shape:
                model_state[key] = param
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        model.load_state_dict(model_state)
        print(f"[ACT-Chunk] Loaded {len(loaded_keys)}/{len(pretrained_state)} params "
              f"(skipped {len(skipped_keys)} due to shape mismatch)")
        if skipped_keys:
            print(f"  Skipped: {skipped_keys[:5]}{'...' if len(skipped_keys)>5 else ''}")
    elif args.pretrained:
        print(f"[WARN] Pretrained not found: {args.pretrained}")

    # ── Layer freezing for finetune ──
    frozen_count = 0
    if args.freeze_backbone and hasattr(model, 'backbone'):
        # Freeze all backbone layers
        for param in model.backbone.parameters():
            param.requires_grad = False
            frozen_count += param.numel()

        # Optionally unfreeze last N blocks (each block = Linear+LayerNorm+GELU+Dropout = 4 modules)
        if args.unfreeze_last_n > 0:
            block_size = 4  # Linear, LayerNorm, GELU, Dropout
            total_modules = len(list(model.backbone.children()))
            unfreeze_from = max(0, total_modules - args.unfreeze_last_n * block_size)
            for i, module in enumerate(model.backbone.children()):
                if i >= unfreeze_from:
                    for param in module.parameters():
                        param.requires_grad = True
                        frozen_count -= param.numel()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[ACT-Chunk] Layer freezing: {frozen_count:,} frozen, "
              f"{trainable:,}/{total:,} trainable")

    # ── Optimizer: discriminative lr if configured ──
    base_lr = float(cfg["train"]["lr"])
    backbone_lr_mult = float(cfg["train"].get("backbone_lr_mult", 1.0))

    if backbone_lr_mult != 1.0 and hasattr(model, 'backbone') and not args.freeze_backbone:
        # Discriminative learning rates: lower lr for backbone
        backbone_params = list(model.backbone.parameters())
        head_params = [p for n, p in model.named_parameters()
                       if not n.startswith('backbone') and p.requires_grad]
        param_groups = [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
            {"params": head_params, "lr": base_lr},
        ]
        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
        print(f"[ACT-Chunk] Discriminative LR: backbone={base_lr * backbone_lr_mult:.2e}, head={base_lr:.2e}")
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params,
            lr=base_lr,
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
    # Store base lr per param group for warmup
    for pg in opt.param_groups:
        pg["_base_lr"] = pg["lr"]
    success_loss_weight = float(cfg["train"].get("success_loss_weight", 0.2))
    kl_weight = float(act_cfg.get("kl_weight", 0.01))
    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 1.0))
    obs_noise_std = float(cfg["train"].get("obs_noise_std", 0.0))
    loss_fn_name = str(cfg["train"].get("loss_fn", "mse"))  # mse | huber | l1

    # Huber loss is more robust to VR demonstration noise
    if loss_fn_name == "huber":
        huber_delta = float(cfg["train"].get("huber_delta", 1.0))
        action_loss_fn = nn.SmoothL1Loss(reduction="none", beta=huber_delta)
        print(f"[ACT-Chunk] Using Huber loss (delta={huber_delta})")
    elif loss_fn_name == "l1":
        action_loss_fn = nn.L1Loss(reduction="none")
        print(f"[ACT-Chunk] Using L1 loss")
    else:
        action_loss_fn = None  # will use inline MSE
        print(f"[ACT-Chunk] Using MSE loss")

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("early_stop_patience", 25))

    # LR scheduler with optional warmup
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    scheduler_name = cfg["train"].get("lr_scheduler", "none")
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-7)

    out_root = root / args.out
    run_dir = out_root / "act_chunk" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    best_val_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    print(f"[ACT-Chunk] Training for {epochs} epochs, patience={patience}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses, train_mses, train_first_mses, train_kls = [], [], [], []

        for obs_seq, act_chunk, mask, success, weights in train_loader:
            obs_seq = obs_seq.to(device)
            act_chunk = act_chunk.to(device)
            mask = mask.to(device)
            success = success.to(device)
            weights = weights.to(device)

            if obs_noise_std > 0:
                obs_seq = obs_seq + torch.randn_like(obs_seq) * obs_noise_std

            out = model(obs_seq, act_chunk)
            pred_actions = out["actions"]

            # Masked action loss over chunk (MSE / Huber / L1)
            if action_loss_fn is not None:
                per_dim_loss = action_loss_fn(pred_actions, act_chunk)  # (B, K, act_dim)
                per_step_loss = per_dim_loss.mean(dim=-1)  # (B, K)
            else:
                diff2 = (pred_actions - act_chunk) ** 2
                per_step_loss = diff2.mean(dim=-1)  # (B, K)

            masked_loss = (per_step_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # (B,)

            # First step MSE (always MSE for metrics, regardless of loss_fn)
            first_mse = ((pred_actions[:, 0, :] - act_chunk[:, 0, :]) ** 2).mean(dim=-1)

            bce = F.binary_cross_entropy_with_logits(
                out["success_logit"], success, reduction="none"
            )

            loss = _weighted_mean(masked_loss, weights) + \
                   success_loss_weight * _weighted_mean(bce, weights) + \
                   kl_weight * out["kl_loss"]

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()

            train_losses.append(loss.item())
            train_mses.append(masked_loss.mean().item())
            train_first_mses.append(first_mse.mean().item())
            train_kls.append(out["kl_loss"].item())

        # LR warmup + scheduler
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in opt.param_groups:
                pg["lr"] = pg.get("_base_lr", base_lr) * warmup_factor
        elif scheduler is not None:
            scheduler.step()

        val_metrics = evaluate(model, val_loader, device, success_loss_weight, kl_weight)
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
            "model": model.state_dict(),
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

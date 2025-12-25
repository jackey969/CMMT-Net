# networks/DualFrequencyUNet.py
# -*- coding: utf-8 -*-
"""
DualFrequencyUNet (Upgraded):
Modify 1) BFGA_Fusion -> Cross-Attention based LF<->HF interaction
Modify 2) Iterative Feedback: multiple rounds of LF->HF suppression + HF->LF edge reweighting (+ cross-attn)
Modify 3) Learnable Decomposition: replace Haar avgpool split with learnable low-pass operator

Notes:
- Keep interface: DualFrequencyUNet(in_chns, class_num, ...)
- Encoder/Decoder are imported from networks.unet
- Designed to be batch-size friendly: uses GroupNorm instead of BatchNorm inside fusion blocks
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.unet import Encoder, Decoder


# ============================================================
# 0) Small helpers
# ============================================================
def _make_gn(ch: int, num_groups: int = 8) -> nn.GroupNorm:
    g = min(num_groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


# ============================================================
# 1) Learnable Decomposition (Learnable Low-Pass + Residual HF)
# ============================================================
class LearnableDecomposition(nn.Module):
    """
    Replace cheap Haar-like split with learnable low-pass operator.

    Idea:
      - learn a low-pass feature: DWConv(stride=2) -> PWConv -> upsample back
      - LF = upsample(low)
      - HF = x - LF  (residual)
    This keeps LF/HF same shape as x, like your original haar_decomposition.

    Why DWConv stride-2?
      - stride-2 encourages smoothing + downsample (low-pass-ish)
      - depthwise keeps channel-wise filtering stable
    """
    def __init__(self, in_ch: int, pool_ks: int = 2):
        super().__init__()
        # depthwise stride conv (acts like learnable low-pass + downsample)
        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=3, stride=pool_ks, padding=1,
            groups=in_ch, bias=False
        )
        # pointwise mixing
        self.pw = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.norm = _make_gn(in_ch)
        self.act = nn.ReLU(inplace=True)

        # optional learnable scaling for HF residual
        self.hf_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # low-resolution low-pass
        x_low = self.dw(x)
        x_low = self.pw(x_low)
        x_low = self.act(self.norm(x_low))

        # upsample back to original
        x_lf = F.interpolate(x_low, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # residual high-frequency
        x_hf = (x - x_lf) * self.hf_scale
        return x_lf, x_hf


# ============================================================
# 2) Cross-Attention on 2D feature maps (LF<->HF)
# ============================================================
class CrossAttention2D(nn.Module):
    """
    Multi-head cross-attention for 2D feature maps.

    Inputs:
      q_feat: (B, C, H, W)  -> Query
      kv_feat:(B, C, H, W)  -> Key/Value
    Output:
      out:   (B, C, H, W)

    Implementation:
      - 1x1 conv projections for q,k,v
      - flatten spatial to tokens (HW)
      - attention per head
    """
    def __init__(self, channels: int, num_heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, f"channels={channels} must be divisible by num_heads={num_heads}"
        self.c = channels
        self.h = num_heads
        self.d = channels // num_heads
        self.scale = self.d ** -0.5

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_q = _make_gn(channels)
        self.norm_kv = _make_gn(channels)

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = q_feat.shape
        assert kv_feat.shape == (B, C, H, W)

        q_feat = self.norm_q(q_feat)
        kv_feat = self.norm_kv(kv_feat)

        q = self.q_proj(q_feat)  # (B,C,H,W)
        k = self.k_proj(kv_feat)
        v = self.v_proj(kv_feat)

        # (B, heads, HW, d)
        q = q.view(B, self.h, self.d, H * W).transpose(2, 3)  # (B,h,HW,d)
        k = k.view(B, self.h, self.d, H * W)                  # (B,h,d,HW)
        v = v.view(B, self.h, self.d, H * W).transpose(2, 3)  # (B,h,HW,d)

        # attn: (B,h,HW,HW)
        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B,h,HW,d)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# ============================================================
# 3) HF -> LF Edge Reweighting (use HF gradient cues)
# ============================================================
class HFEdgeReweight(nn.Module):
    """
    Use HF feature "gradient / edge" cues to reweight LF features at boundaries.

    Implementation:
      - fixed Sobel depthwise filters on HF feature map -> grad magnitude
      - compress to 1 channel "edge" map
      - produce gating map for LF (C-channel) via 1x1 conv + sigmoid
    """
    def __init__(self, channels: int):
        super().__init__()
        # Sobel kernels (fixed)
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32)

        self.register_buffer("sobel_x", kx.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", ky.view(1, 1, 3, 3))

        # compress HF->edge and lift edge->LF-gate
        self.edge_reduce = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            _make_gn(1),
            nn.ReLU(inplace=True),
        )
        self.to_gate = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, lf: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        # edge magnitude on reduced HF (more stable than per-channel sobel)
        hf_r = self.edge_reduce(hf)  # (B,1,H,W)

        gx = F.conv2d(hf_r, self.sobel_x, padding=1)
        gy = F.conv2d(hf_r, self.sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)  # (B,1,H,W)

        gate = self.to_gate(edge)  # (B,C,H,W)
        # emphasize LF near edges: (1 + gate) style is usually safer than pure mask
        lf_enh = lf * (1.0 + gate)
        return lf_enh


# ============================================================
# 4) Cross-Attn + Iterative Feedback Fusion (replaces BFGA_Fusion)
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class BFGA_Fusion_CrossIter(nn.Module):
    """
    Implements:
      - LF -> HF: structure suppression (mask) + cross-attn injection
      - HF -> LF: edge reweighting + cross-attn injection
      - Iterative feedback: repeat K times before final fusion

    Compared to original BFGA:
      - spatial gate becomes C-channel (depthwise) for finer suppression
      - use cross-attention for dynamic LF/HF interaction
      - use HF gradient cues to reweight LF boundary response
    """
    def __init__(
        self,
        channels: int,
        iters: int = 2,
        num_heads: int = 4,
        se_reduction: int = 16,
        gate_ks: int = 7
    ):
        super().__init__()
        assert iters >= 1
        self.iters = iters

        pad = gate_ks // 2
        # C-channel spatial gate (depthwise conv) : LF -> HF suppression
        self.lf2hf_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=gate_ks, padding=pad, groups=channels, bias=True),
            nn.Sigmoid()
        )

        # cross-attn modules
        self.ca_lf_to_hf = CrossAttention2D(channels, num_heads=num_heads)
        self.ca_hf_to_lf = CrossAttention2D(channels, num_heads=num_heads)

        # HF edge reweight LF
        self.hf_edge = HFEdgeReweight(channels)

        # small residual mixing after each iteration
        self.mix_lf = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), _make_gn(channels), nn.ReLU(inplace=True))
        self.mix_hf = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), _make_gn(channels), nn.ReLU(inplace=True))

        # final fusion
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.se = SEBlock(channels, reduction=se_reduction)

    def forward(self, feat_lf: torch.Tensor, feat_hf: torch.Tensor) -> torch.Tensor:
        if feat_lf.shape != feat_hf.shape:
            raise ValueError(
                f"Fusion expects same shape for feat_lf and feat_hf, got {tuple(feat_lf.shape)} vs {tuple(feat_hf.shape)}"
            )

        lf = feat_lf
        hf = feat_hf

        for _ in range(self.iters):
            # ---- LF -> HF (macro structure suppress noise) ----
            mask = self.lf2hf_gate(lf)         # (B,C,H,W)
            hf = hf * mask                     # suppress noisy HF regions guided by LF

            # cross-attn: let LF query HF (or reverse) to refine HF with global context
            hf = hf + self.ca_lf_to_hf(q_feat=lf, kv_feat=hf)
            hf = hf + self.mix_hf(hf)

            # ---- HF -> LF (use edge/gradient cues enhance boundary response) ----
            lf = self.hf_edge(lf, hf)          # boundary emphasis
            lf = lf + self.ca_hf_to_lf(q_feat=hf, kv_feat=lf)  # HF guides LF via attention
            lf = lf + self.mix_lf(lf)

        fused = torch.cat([lf, hf], dim=1)     # (B,2C,H,W)
        fused = self.reduce(fused)             # (B,C,H,W)
        fused = self.se(fused)
        return fused


# ============================================================
# 5) DualFrequencyUNet (with Learnable Decomposition)
# ============================================================
class DualFrequencyUNet(nn.Module):
    """
    Dual-Branch Frequency-Aware U-Net:
      - x -> (x_lf, x_hf) by LearnableDecomposition
      - two encoders (LF/HF)
      - fuse bottleneck via BFGA_Fusion_CrossIter
      - decoder uses LF skips (0..3) + fused bottleneck (4)
    """
    def __init__(
        self,
        in_chns: int,
        class_num: int,
        feature_chns=None,
        dropout=None,
        bilinear: bool = False,
        se_reduction: int = 16,
        # new knobs
        decomp_pool_ks: int = 2,
        fuse_iters: int = 2,
        attn_heads: int = 4,
        use_hf_adaptor: bool = True,
    ):
        super().__init__()

        if feature_chns is None:
            feature_chns = [16, 32, 64, 128, 256]
        if dropout is None:
            dropout = [0.05, 0.1, 0.2, 0.3, 0.5]

        params = {
            "in_chns": in_chns,
            "feature_chns": feature_chns,
            "dropout": dropout,
            "class_num": class_num,
            "bilinear": bilinear,
            "acti_func": "relu",
        }

        # Learnable frequency decomposition
        self.decomp = LearnableDecomposition(in_ch=in_chns, pool_ks=decomp_pool_ks)

        self.encoder_lf = Encoder(params)

        # HF adaptor (optional but recommended): constrain HF amplitude / stabilize
        self.use_hf_adaptor = bool(use_hf_adaptor)
        if self.use_hf_adaptor:
            self.hf_adaptor = nn.Sequential(
                nn.Conv2d(in_chns, in_chns, kernel_size=1, bias=False),
                _make_gn(in_chns),
                nn.ReLU(inplace=True),
            )
        else:
            self.hf_adaptor = nn.Identity()

        self.encoder_hf = Encoder(params)

        self.fusion = BFGA_Fusion_CrossIter(
            channels=feature_chns[-1],
            iters=fuse_iters,
            num_heads=attn_heads,
            se_reduction=se_reduction,
            gate_ks=7
        )

        self.decoder = Decoder(params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) learnable decomposition
        x_lf, x_hf = self.decomp(x)

        # 2) HF adaptor
        x_hf = self.hf_adaptor(x_hf)

        # 3) encode
        lf_feats = self.encoder_lf(x_lf)  # [lf0..lf4]
        hf_feats = self.encoder_hf(x_hf)  # [hf0..hf4]

        # 4) fuse bottleneck with cross-attn + iterative feedback
        fused_bottleneck = self.fusion(lf_feats[-1], hf_feats[-1])

        # 5) decode (LF skips + fused bottleneck)
        dec_feats = [lf_feats[0], lf_feats[1], lf_feats[2], lf_feats[3], fused_bottleneck]
        out = self.decoder(dec_feats)
        return out

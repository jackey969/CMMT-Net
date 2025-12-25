# -*- coding: utf-8 -*-
"""
Semi-Supervised Training: UNet <-> DualFrequencyUNet Cross Pseudo Supervision (MC Dropout)
+ RL-CutMix (applied on UNet only in this script)

MODIFIED (as requested):
(1) CPS -> Teacher-Student stopgrad + separate step
    - Update UNet with:   loss_unet = sup_unet + lambda_u * unsup_unet_from_dfu + mix_loss
      backward -> step(unet)
    - Update DFUNet with: loss_dfu  = sup_dfu  + lambda_u * unsup_dfu_from_unet
      backward -> step(dfu)
    NOTE: pseudo labels are always generated with torch.no_grad() => teacher stopgrad.

(2) Only stronger network acts as teacher (dynamic selection)
    - If sup_loss_unet < sup_loss_dfu: UNet is teacher, DFUNet is student
      => unsup_dfu_from_unet computed, unsup_unet_from_dfu = 0
    - Else: DFUNet is teacher, UNet is student
      => unsup_unet_from_dfu computed, unsup_dfu_from_unet = 0

Other notes:
- Kept your RL-CutMix agent update logic (optimizer_agent) unchanged.
- Fixed a stability issue: pseudo_u_unet is only defined when Bu>0; RL labels_all now uses
  pseudo from the current teacher (if unlabeled exists), else falls back to labeled only.
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.unet import UNet
from networks.DualFrequencyUNet import DualFrequencyUNet
from utils import losses, util
from utils.val_2D import test_single_volume


# ===================== 参数配置区 =====================
parser = argparse.ArgumentParser()

# 基础数据与实验路径
parser.add_argument("--root_path", type=str, default="./datasets/ACDC", help="dataset root")
parser.add_argument("--exp", type=str, default="ACDC/Semi_Cross_UNet_DFUnet_MCDropout", help="experiment name")
parser.add_argument("--model", type=str, default="unet_df", help="model name tag for saving")

# 训练超参数
parser.add_argument("--max_iterations", type=int, default=30000, help="max iters to train")
parser.add_argument("--batch_size", type=int, default=24, help="total batch size per gpu")
parser.add_argument("--labeled_bs", type=int, default=12, help="number of labeled samples in a batch")
parser.add_argument("--base_lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="network input size")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
parser.add_argument("--img_channels", type=int, default=1, help="1 if ACDC, 3 if GLAS")
parser.add_argument("--deterministic", type=int, default=1, help="deterministic training")
parser.add_argument("--load", default=False, action="store_true", help="restore checkpoint")

# 半监督参数
parser.add_argument("--labeled_num", type=int, default=7, help="number of labeled patients")
parser.add_argument("--mc_forward_times", type=int, default=8, help="MC Dropout forward passes")
parser.add_argument("--lambda_u", type=float, default=1.0, help="unsup weight")
parser.add_argument("--warmup_iters", type=int, default=5000, help="warmup iters for unsup weight")

# RL-CutMix 参数
parser.add_argument("--rlcm_use", type=int, default=1, help="enable RL-CutMix if 1")
parser.add_argument("--grid_N", type=int, default=8, help="grid size N (NxN)")
parser.add_argument("--scale_set", type=str, default="0.25,0.5,0.75", help="relative scales")
parser.add_argument("--agent_lr", type=float, default=1e-4, help="learning rate for CutMixAgent")
parser.add_argument("--ent_coef", type=float, default=0.01, help="entropy bonus coefficient")
parser.add_argument("--rlcm_prob", type=float, default=1.0, help="probability to apply RL-CutMix")
parser.add_argument("--agent_override_prob", type=float, default=0.3, help="prob to override agent by random bbox")

args = parser.parse_args()


# ===================== 辅助函数：半监督数据切片映射 =====================
def patients_to_slices(dataset, patiens_num):
    """根据病人数返回对应的切片数"""
    if "ACDC" in dataset:
        ref_dict = {
            "1": 32, "3": 68, "7": 136,
            "14": 256, "21": 396, "28": 512,
            "35": 664, "140": 1312
        }
    elif "MnM2" in dataset:
        ref_dict = {
            "1": 24, "3": 70, "5": 116, "6": 138, "7": 160,
            "9": 212, "10": 232, "25": 526, "50": 1122, "96": 2140
        }
    elif "Promise12" in dataset:
        ref_dict = {
            "1": 20, "3": 68, "5": 138, "7": 181,
            "10": 253, "15": 369, "20": 542,
            "25": 689, "30": 825, "35": 982
        }
    else:
        return patiens_num * 10

    if str(patiens_num) not in ref_dict:
        logging.warning(f"Labelnum={patiens_num} not found, fallback to default (ACDC-7=136).")
        return 136
    return ref_dict[str(patiens_num)]


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


# ===================== RL-CutMix 核心组件 =====================
def _grid_center_to_pixel(cx_idx: int, cy_idx: int, H: int, W: int, N: int):
    cell_h, cell_w = H / N, W / N
    cx = (cx_idx + 0.5) * cell_h
    cy = (cy_idx + 0.5) * cell_w
    return int(round(cx)), int(round(cy))


def _bbox_from_center_scale(cx_pix: int, cy_pix: int, side: int, H: int, W: int):
    half = side // 2
    top = max(0, cx_pix - half)
    left = max(0, cy_pix - half)
    bottom = min(H, top + side)
    right = min(W, left + side)
    top = max(0, bottom - side)
    left = max(0, right - side)
    return top, left, bottom, right


@torch.no_grad()
def apply_cutmix(image_a, image_b, label_a, label_b, bbox):
    """Apply CutMix based on provided bboxes."""
    assert image_a.shape == image_b.shape
    assert label_a.shape == label_b.shape
    B, C, H, W = image_a.shape

    mixed_images = image_a.clone()
    mixed_labels = label_a.clone()

    for b in range(B):
        t, l, btm, r = bbox[b]
        if (btm - t) <= 0 or (r - l) <= 0:
            continue
        mixed_images[b, :, t:btm, l:r] = image_b[b, :, t:btm, l:r]
        mixed_labels[b, t:btm, l:r] = label_b[b, t:btm, l:r]

    return mixed_images, mixed_labels


def ce_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    ce_map = F.cross_entropy(logits, labels.long(), reduction="none")
    return ce_map.view(ce_map.size(0), -1).mean(dim=1)


def soft_dice_per_sample(probs: torch.Tensor, labels: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    B, C, H, W = probs.shape
    onehot = F.one_hot(labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (2, 3)
    intersect = (probs * onehot).sum(dim=dims)
    denom = probs.sum(dim=dims) + onehot.sum(dim=dims) + eps
    dice = (2.0 * intersect + eps) / denom
    dice_mean = dice.mean(dim=1)
    return 1.0 - dice_mean


class CutMixAgent(nn.Module):
    def __init__(self, in_ch: int, N: int, num_scales: int):
        super().__init__()
        ch = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head_x = nn.Linear(ch, N)
        self.head_y = nn.Linear(ch, N)
        self.head_s = nn.Linear(ch, num_scales)

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        logits_x = self.head_x(feat)
        logits_y = self.head_y(feat)
        logits_s = self.head_s(feat)
        return logits_x, logits_y, logits_s

    @torch.no_grad()
    def decode_bbox(self, ax, ay, ascale, H, W, N, scale_values):
        B = ax.shape[0]
        min_side = min(H, W)
        bbox_list = []
        for i in range(B):
            cx_pix, cy_pix = _grid_center_to_pixel(int(ax[i]), int(ay[i]), H, W, N)
            side = max(4, int(round(scale_values[int(ascale[i])] * min_side)))
            bbox_list.append(_bbox_from_center_scale(cx_pix, cy_pix, side, H, W))
        return bbox_list

    def sample_actions(self, x, H, W, N, scale_values, device):
        logits_x, logits_y, logits_s = self.forward(x)
        dist_x = torch.distributions.Categorical(logits=logits_x)
        dist_y = torch.distributions.Categorical(logits=logits_y)
        dist_s = torch.distributions.Categorical(logits=logits_s)

        ax = dist_x.sample()
        ay = dist_y.sample()
        ascale = dist_s.sample()

        log_probs = dist_x.log_prob(ax) + dist_y.log_prob(ay) + dist_s.log_prob(ascale)
        entropy = dist_x.entropy() + dist_y.entropy() + dist_s.entropy()

        bbox_list = self.decode_bbox(ax, ay, ascale, H, W, N, scale_values)
        return bbox_list, log_probs, entropy, (ax, ay, ascale)


def random_bbox(B, H, W, scales, N):
    bbox_list = []
    min_side = min(H, W)
    for _ in range(B):
        cx_idx = random.randint(0, N - 1)
        cy_idx = random.randint(0, N - 1)
        scale_idx = random.randint(0, len(scales) - 1)
        cx_pix, cy_pix = _grid_center_to_pixel(cx_idx, cy_idx, H, W, N)
        side = max(4, int(round(scales[scale_idx] * min_side)))
        bbox_list.append(_bbox_from_center_scale(cx_pix, cy_pix, side, H, W))
    return bbox_list


# ===================== MC Dropout（BN保持eval，仅Dropout随机） =====================
def enable_dropout_only(model: nn.Module):
    """Enable dropout layers during eval (keeps BN in eval)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def mc_dropout(model: nn.Module, x: torch.Tensor, mc_times: int):
    """
    Returns:
        pseudo_label: (B,H,W)
        mean_probs  : (B,C,H,W)
    """
    was_training = model.training
    model.eval()
    enable_dropout_only(model)

    with torch.no_grad():
        probs_list = []
        for _ in range(mc_times):
            logits = model(x)
            probs = torch.softmax(logits, 1)
            probs_list.append(probs)

        mean_probs = torch.mean(torch.stack(probs_list, dim=0), dim=0)
        pseudo_label = torch.argmax(mean_probs, dim=1)

    model.train(was_training)
    return pseudo_label, mean_probs


# ===================== 训练主流程 =====================
def train(args, snapshot_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1) 数据
    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    logging.info(f"Total train slices: {total_slices}, Labeled slices: {labeled_slice}")

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
    )

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    # 2) 两个模型
    unet = UNet(in_chns=args.img_channels, class_num=args.num_classes).to(device)
    dfunet = DualFrequencyUNet(in_chns=args.img_channels, class_num=args.num_classes).to(device)

    optimizer_unet = optim.SGD(unet.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    optimizer_dfunet = optim.SGD(dfunet.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

    dice_loss_batch = losses.DiceLoss(args.num_classes)

    # RL-CutMix
    rl_enabled = bool(args.rlcm_use)
    scales = [float(s) for s in args.scale_set.split(",")]
    N = int(args.grid_N)
    ent_coef = float(args.ent_coef)
    rl_prob = float(args.rlcm_prob)
    override_prob = float(args.agent_override_prob)

    agent = CutMixAgent(in_ch=2 * args.img_channels, N=N, num_scales=len(scales)).to(device)
    optimizer_agent = optim.Adam(agent.parameters(), lr=args.agent_lr)
    logging.info(f"[RL-CutMix] Enabled | Grid={N}x{N} | Scales={scales} | ent_coef={ent_coef} | rl_p={rl_prob} | override_p={override_prob}")

    # 3) 可选：恢复 checkpoint（分别恢复）
    if args.load:
        ckpt_unet = os.path.join(snapshot_path, f"{args.model}_unet_best_model.pth")
        ckpt_dfu = os.path.join(snapshot_path, f"{args.model}_dfunet_best_model.pth")

        if os.path.exists(ckpt_unet):
            ckpt = torch.load(ckpt_unet, map_location=device)
            unet.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt, strict=False)
            logging.info(f"Loaded UNet checkpoint: {ckpt_unet}")
        else:
            logging.warning(f"UNet checkpoint not found: {ckpt_unet}")

        if os.path.exists(ckpt_dfu):
            ckpt = torch.load(ckpt_dfu, map_location=device)
            dfunet.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt, strict=False)
            logging.info(f"Loaded DualFrequencyUNet checkpoint: {ckpt_dfu}")
        else:
            logging.warning(f"DualFrequencyUNet checkpoint not found: {ckpt_dfu}")

    unet.train()
    dfunet.train()
    agent.train()

    max_epoch = args.max_iterations // len(trainloader) + 1
    iter_num = 0
    best_performance_unet = 0.0
    best_performance_dfu = 0.0

    logging.info(f"Start training: {args.max_iterations} iterations, {len(trainloader)} iter/epoch")
    iterator = tqdm(range(max_epoch), ncols=80)

    for epoch_num in iterator:
        for sampled in trainloader:
            # ====== 数据划分 ======
            image_batch = sampled["image"].to(device)    # (B,C,H,W)
            label_batch = sampled["label"].to(device)    # (B,H,W)
            labeled_bs = args.labeled_bs
            B, C, H, W = image_batch.shape

            img_labeled = image_batch[:labeled_bs]
            lbl_labeled = label_batch[:labeled_bs]

            img_unlabeled = image_batch[labeled_bs:]
            Bu = img_unlabeled.shape[0]

            # ====== unsup 权重 warmup ======
            if args.warmup_iters > 0:
                lambda_u = args.lambda_u * min(1.0, float(iter_num) / float(args.warmup_iters))
            else:
                lambda_u = args.lambda_u

            # =========================================================
            # 1) 监督：标注部分 (两个网络都用 GT 监督)
            # =========================================================
            logits_l_unet = unet(img_labeled)
            probs_l_unet = torch.softmax(logits_l_unet, dim=1)
            ce_sup_unet = F.cross_entropy(logits_l_unet, lbl_labeled.long(), reduction="mean")
            dice_sup_unet = dice_loss_batch(probs_l_unet, lbl_labeled.unsqueeze(1))
            sup_loss_unet = ce_sup_unet + dice_sup_unet

            logits_l_dfu = dfunet(img_labeled)
            probs_l_dfu = torch.softmax(logits_l_dfu, dim=1)
            ce_sup_dfu = F.cross_entropy(logits_l_dfu, lbl_labeled.long(), reduction="mean")
            dice_sup_dfu = dice_loss_batch(probs_l_dfu, lbl_labeled.unsqueeze(1))
            sup_loss_dfu = ce_sup_dfu + dice_sup_dfu

            # =========================================================
            # 2) 动态选择 teacher（只让更强网络当 teacher）
            #    sup_loss 更小 => 更强
            # =========================================================
            unet_is_teacher = bool(sup_loss_unet.item() < sup_loss_dfu.item())

            # =========================================================
            # 3) 无标注：仅 teacher -> student 的 CPS (stopgrad via no_grad)
            #    + 分开 step：分别更新 UNet / DFUNet
            # =========================================================
            unsup_unet_from_dfu = torch.tensor(0.0, device=device)
            unsup_dfu_from_unet = torch.tensor(0.0, device=device)

            # 仅用于 RL labels_all 的伪标签（取当前 teacher 的 pseudo）
            pseudo_u_teacher = None

            if Bu > 0:
                if unet_is_teacher:
                    # teacher UNet -> student DFUNet
                    pseudo_u_unet, _ = mc_dropout(unet, img_unlabeled, args.mc_forward_times)  # (Bu,H,W)
                    pseudo_u_teacher = pseudo_u_unet

                    logits_u_dfu = dfunet(img_unlabeled)
                    probs_u_dfu = torch.softmax(logits_u_dfu, dim=1)
                    ce_u_dfu = F.cross_entropy(logits_u_dfu, pseudo_u_unet.long(), reduction="mean")
                    dice_u_dfu = dice_loss_batch(probs_u_dfu, pseudo_u_unet.unsqueeze(1))
                    unsup_dfu_from_unet = ce_u_dfu + dice_u_dfu
                else:
                    # teacher DFUNet -> student UNet
                    pseudo_u_dfu, _ = mc_dropout(dfunet, img_unlabeled, args.mc_forward_times)  # (Bu,H,W)
                    pseudo_u_teacher = pseudo_u_dfu

                    logits_u_unet = unet(img_unlabeled)
                    probs_u_unet = torch.softmax(logits_u_unet, dim=1)
                    ce_u_unet = F.cross_entropy(logits_u_unet, pseudo_u_dfu.long(), reduction="mean")
                    dice_u_unet = dice_loss_batch(probs_u_unet, pseudo_u_dfu.unsqueeze(1))
                    unsup_unet_from_dfu = ce_u_unet + dice_u_unet

            # =========================================================
            # 4) RL-CutMix（仅 UNet，保持你原逻辑）
            #    这里的 labels_all：labeled 使用 GT；unlabeled 使用当前 teacher 的 pseudo（若存在）
            # =========================================================
            mix_loss = torch.tensor(0.0, device=device)
            loss_agent = torch.tensor(0.0, device=device)

            if rl_enabled and (random.random() < rl_prob):
                # 若没有 unlabeled 或 pseudo 尚不可用，则只用 labeled 做 RL-CutMix
                if Bu > 0 and pseudo_u_teacher is not None:
                    labels_all = torch.cat([lbl_labeled, pseudo_u_teacher], dim=0)  # (B,H,W)
                    imgs_all = torch.cat([img_labeled, img_unlabeled], dim=0)       # (B,C,H,W)
                else:
                    labels_all = lbl_labeled
                    imgs_all = img_labeled

                B2, _, _, _ = imgs_all.shape
                perm = torch.randperm(B2, device=device)
                img_A, lbl_A = imgs_all, labels_all
                img_B, lbl_B = imgs_all[perm], labels_all[perm]
                agent_in = torch.cat([img_A, img_B], dim=1)  # (B2,2C,H,W)

                bbox_agent, log_probs, entropy, actions = agent.sample_actions(
                    agent_in, H=H, W=W, N=N, scale_values=scales, device=device
                )

                override_mask = (torch.rand(B2, device=device) < override_prob).float()
                bbox_rand = random_bbox(B2, H, W, scales, N)
                final_bbox = []
                for i in range(B2):
                    final_bbox.append(bbox_rand[i] if override_mask[i] > 0 else bbox_agent[i])

                mixed_images, mixed_labels = apply_cutmix(img_A, img_B, lbl_A, lbl_B, final_bbox)

                logits_m = unet(mixed_images)
                probs_m = torch.softmax(logits_m, dim=1)
                ce_m = F.cross_entropy(logits_m, mixed_labels.long(), reduction="mean")
                dice_m = dice_loss_batch(probs_m, mixed_labels.unsqueeze(1))
                mix_loss = ce_m + dice_m

                with torch.no_grad():
                    ce_b = ce_per_sample(logits_m, mixed_labels)  # (B2,)
                    dice_b = soft_dice_per_sample(probs_m, mixed_labels, args.num_classes)  # (B2,)
                    reward = ce_b + dice_b
                    reward = (reward - reward.mean()) / (reward.std() + 1e-8)

                valid_mask = (1.0 - override_mask)  # (B2,)
                denom = valid_mask.sum().clamp_min(1e-8)
                loss_agent = -((reward * log_probs * valid_mask).sum() / denom) - ent_coef * (entropy.mean())

                # RL 日志统计
                ax, ay, ascale = actions
                avg_scale_idx = ascale.float().mean().item()
                raw_reward_mean = (ce_b + dice_b).mean().item()
                entropy_mean = entropy.mean().item()
                override_ratio = override_mask.mean().item() * 100

                bbox_area_ratios = []
                for (t, l, btm, r) in final_bbox:
                    area = (btm - t) * (r - l)
                    bbox_area_ratios.append(area / (H * W + 1e-8))
                avg_area_ratio = float(np.mean(bbox_area_ratios))
            else:
                # 若未使用 RL，本轮日志里给默认值
                avg_scale_idx = 0.0
                raw_reward_mean = 0.0
                entropy_mean = 0.0
                override_ratio = 0.0
                avg_area_ratio = 0.0

            # =========================================================
            # 5) 分开 backward/step (Teacher-Student stopgrad + separate step)
            #    - UNet: sup_unet + lambda_u*unsup_unet_from_dfu + mix_loss
            #    - DFUNet: sup_dfu + lambda_u*unsup_dfu_from_unet
            # =========================================================
            # ---- update UNet ----
            loss_unet = sup_loss_unet + lambda_u * unsup_unet_from_dfu + mix_loss
            optimizer_unet.zero_grad(set_to_none=True)
            loss_unet.backward()
            optimizer_unet.step()

            # ---- update DFUNet ----
            loss_dfu = sup_loss_dfu + lambda_u * unsup_dfu_from_unet
            optimizer_dfunet.zero_grad(set_to_none=True)
            loss_dfu.backward()
            optimizer_dfunet.step()

            # ---- update RL agent ----
            optimizer_agent.zero_grad(set_to_none=True)
            loss_agent.backward()
            optimizer_agent.step()

            # ====== 学习率衰减（Poly） ======
            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for pg in optimizer_unet.param_groups:
                pg["lr"] = lr_
            for pg in optimizer_dfunet.param_groups:
                pg["lr"] = lr_

            iter_num += 1

            # ====== 日志 ======
            total_loss = loss_unet.detach() + loss_dfu.detach()  # 仅用于监控（不是用于 backward 的总 loss）
            sup_loss = sup_loss_unet.detach() + sup_loss_dfu.detach()
            unsup_loss = unsup_unet_from_dfu.detach() + unsup_dfu_from_unet.detach()

            teacher_name = "UNet" if unet_is_teacher else "DFUNet"
            logging.info(
                f"[Iter {iter_num:05d}] "
                f"T={teacher_name} "
                f"loss(u={loss_unet.item():.4f}, d={loss_dfu.item():.4f}, sum={total_loss.item():.4f}) "
                f"sup={sup_loss.item():.4f}(u={sup_loss_unet.item():.4f},d={sup_loss_dfu.item():.4f}) "
                f"unsup={unsup_loss.item():.4f}(u<-d={unsup_unet_from_dfu.item():.4f},d<-u={unsup_dfu_from_unet.item():.4f}) "
                f"mix={mix_loss.item():.4f} "
                f"λu={lambda_u:.3f} lr={lr_:.2e} "
                f"RL:R={raw_reward_mean:.3f} H={entropy_mean:.3f} s={avg_scale_idx:.2f} A={avg_area_ratio * 100:.1f}% O={override_ratio:.1f}%"
            )

            # ===== 验证（分别评估两网并保存各自best）=====
            if iter_num % 200 == 0:
                # ---- eval UNet ----
                unet.eval()
                metric_list = 0.0
                for val_batch in valloader:
                    metric_i = test_single_volume(
                        val_batch["image"], val_batch["label"], unet, classes=args.num_classes
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                mean_dice = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_jacc = np.mean(metric_list, axis=0)[2]

                if mean_dice > best_performance_unet:
                    best_performance_unet = mean_dice
                    save_best = os.path.join(snapshot_path, f"{args.model}_unet_best_model.pth")
                    util.save_checkpoint(epoch_num, unet, optimizer_unet, torch.tensor(mean_dice), save_best)
                    logging.info(f"[UNet] BEST @ iter {iter_num}: Dice={mean_dice:.4f}, HD95={mean_hd95:.4f}, Jacc={mean_jacc:.4f}")

                for ci in range(args.num_classes - 1):
                    logging.info(
                        "[UNet] iter %d: val_%d_dice: %f  val_%d_hd95: %f  val_%d_jaccard: %f"
                        % (iter_num, ci + 1, metric_list[ci, 0],
                           ci + 1, metric_list[ci, 1],
                           ci + 1, metric_list[ci, 2])
                    )
                logging.info(
                    "[UNet] iter %d : mean_dice: %f  mean_hd95: %f  mean_jaccard: %f"
                    % (iter_num, mean_dice, mean_hd95, mean_jacc)
                )
                unet.train()

                # ---- eval DualFrequencyUNet ----
                dfunet.eval()
                metric_list = 0.0
                for val_batch in valloader:
                    metric_i = test_single_volume(
                        val_batch["image"], val_batch["label"], dfunet, classes=args.num_classes
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                mean_dice = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_jacc = np.mean(metric_list, axis=0)[2]

                if mean_dice > best_performance_dfu:
                    best_performance_dfu = mean_dice
                    save_best = os.path.join(snapshot_path, f"{args.model}_dfunet_best_model.pth")
                    util.save_checkpoint(epoch_num, dfunet, optimizer_dfunet, torch.tensor(mean_dice), save_best)
                    logging.info(f"[DFUNet] BEST @ iter {iter_num}: Dice={mean_dice:.4f}, HD95={mean_hd95:.4f}, Jacc={mean_jacc:.4f}")

                for ci in range(args.num_classes - 1):
                    logging.info(
                        "[DFUNet] iter %d: val_%d_dice: %f  val_%d_hd95: %f  val_%d_jaccard: %f"
                        % (iter_num, ci + 1, metric_list[ci, 0],
                           ci + 1, metric_list[ci, 1],
                           ci + 1, metric_list[ci, 2])
                    )
                logging.info(
                    "[DFUNet] iter %d : mean_dice: %f  mean_hd95: %f  mean_jaccard: %f"
                    % (iter_num, mean_dice, mean_hd95, mean_jacc)
                )
                dfunet.train()

            if iter_num >= args.max_iterations:
                iterator.close()
                break

        if iter_num >= args.max_iterations:
            break


# ===================== 启动入口 =====================
if __name__ == "__main__":
    # CUDNN 复现性
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 日志目录
    snapshot_path = os.path.join("./logs", args.exp, args.model)
    os.makedirs(snapshot_path, exist_ok=True)

    print(snapshot_path + "/log.log")
    logging.getLogger("").handlers = []
    logging.basicConfig(
        filename=snapshot_path + "/log.log",
        level=logging.DEBUG,
        filemode="w",
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # patch_size 自动调整（保留原逻辑）
    if "brats" in args.root_path.lower():
        args.patch_size = [128, 128]

    logging.info(str(args))
    train(args, snapshot_path)

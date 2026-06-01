"""
基于扩散模型 (DDPM/DDIM) 的 BLE 指纹地图补全
==============================================

功能：利用 Spring 全量数据（5行×26列×25AP）训练条件扩散模型，
      推理阶段补全 Winter 残缺数据（仅第3行×26列×23AP）为完整地图。

优化点（相比原版）：
  1. 真实 JSON 数据加载与归一化（替代模拟随机数据）
  2. 基于子采样的数据增强（利用每位置数百次 RSS 采样）
  3. 填充至 UNet 友好尺寸 (5,26)→(8,32)，避免下采样崩溃
  4. 多样化掩膜训练策略（单行/多行/随机散点/块状）
  5. 加权损失函数（强调未知区域）
  6. 学习率余弦退火 + 梯度裁剪
  7. DDIM 加速推理（1000步→50步）
  8. RePaint 技巧（已知区域强制保真）
  9. 多次采样 + 不确定性估计
  10. 完整评估指标（RMSE / MAE / CDF）
  11. 结果可视化（热图 + 误差图 + CDF 曲线 + 训练曲线）
"""

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
import numpy as np
import json
import ast
import os
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，直接保存图片
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------- 跨平台中文字体设置 ----------
# 优先级：Windows / Mac / Linux 常见中文字体依次尝试
_CN_CANDIDATES = [
    "Microsoft YaHei",   # Windows
    "SimHei",            # Windows
    "PingFang SC",       # macOS
    "Heiti SC",          # macOS
    "STHeiti",           # macOS
    "WenQuanYi Micro Hei",  # Linux
    "Noto Sans CJK SC",     # Linux
    "DejaVu Sans",       # 兜底（无中文但不报错）
]
_available = {f.name for f in fm.fontManager.ttflist}
_chosen = [f for f in _CN_CANDIDATES if f in _available]
plt.rcParams["font.sans-serif"] = _chosen if _chosen else ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print(f"[Font] 使用字体: {plt.rcParams['font.sans-serif'][0]}")


# =========================================================================
# 1. 全局配置
# =========================================================================
class Config:
    # --- 数据路径 ---
    SPRING_DATA_PATH = "spring_data.json"
    WINTER_DATA_PATH = "winter_data.json"

    # --- 网格尺寸 ---
    NUM_ROWS = 5
    NUM_COLS = 26
    PAD_ROWS = 8          # 填充后，对三层 UNet 下采样友好 (8→4→2→1)
    PAD_COLS = 32          # 填充后 (32→16→8→4)

    # --- RSS 归一化 ---
    RSS_MIN = -128.0
    RSS_MAX = -54.0

    # --- 训练超参 ---
    BATCH_SIZE = 16
    EPOCHS = 2000
    LR = 2e-4
    WEIGHT_DECAY = 1e-5
    NUM_TRAIN_TIMESTEPS = 1000
    NUM_AUGMENTATIONS = 128  # 更多子采样增强，缓解过拟合

    # --- 推理超参 ---
    NUM_INFERENCE_STEPS = 50   # DDIM 步数（原 DDPM 需 1000 步）
    NUM_SAMPLES = 10           # 多次采样取均值
    USE_REPAINT = True         # 启用 RePaint 已知区域保真

    # --- 输出 ---
    CHECKPOINT_DIR = "checkpoints"


# =========================================================================
# 2. 数据加载与预处理
# =========================================================================
class BLEFingerprintDataset:
    """BLE 指纹数据集：加载、归一化、增强、填充"""

    # Spring 的 25 个 AP 作为标准设备列表
    DEVICE_LIST = [
        "51", "52", "53", "54", "55", "56", "57", "58", "59", "60",
        "61", "62", "63", "64", "65", "66", "67", "69", "70", "71",
        "81", "82", "83", "84", "86",
    ]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.num_aps = len(self.DEVICE_LIST)          # 25
        self.dev2idx = {d: i for i, d in enumerate(self.DEVICE_LIST)}

    # ---------- 归一化 / 反归一化 ----------
    def normalize(self, rss):
        """RSS → [-1, 1]"""
        return 2.0 * (rss - self.cfg.RSS_MIN) / (self.cfg.RSS_MAX - self.cfg.RSS_MIN) - 1.0

    def denormalize(self, val):
        """[-1, 1] → RSS"""
        return (val + 1.0) / 2.0 * (self.cfg.RSS_MAX - self.cfg.RSS_MIN) + self.cfg.RSS_MIN

    # ---------- 加载 Spring ----------
    def load_spring(self):
        """
        返回:
            mean_map        [25, 5, 26]  均值指纹地图（归一化）
            augmented_maps  [N, 25, 5, 26]  子采样增强地图
        """
        with open(self.cfg.SPRING_DATA_PATH, "r") as f:
            raw = json.load(f)

        fill = self.normalize(self.cfg.RSS_MIN)                         # 缺失值填充
        mean_map = torch.full((self.num_aps, self.cfg.NUM_ROWS, self.cfg.NUM_COLS), fill)

        # 收集每个 (row, col, ap) 的原始 RSS 列表，用于子采样增强
        raw_samples: dict[tuple, list[float]] = {}

        for pos_key, devices in raw.items():
            row, col = ast.literal_eval(pos_key)       # "(1, 1)" → (1, 1)
            ri, ci = row - 1, col - 1                  # 转 0-index

            for dev_id, measurements in devices.items():
                if dev_id not in self.dev2idx:
                    continue
                ai = self.dev2idx[dev_id]
                rss_list = [m["rss"] for m in measurements]
                if len(rss_list) == 0:
                    continue
                mean_map[ai, ri, ci] = self.normalize(float(np.mean(rss_list)))
                raw_samples[(ri, ci, ai)] = rss_list

        # 子采样增强：每次随机取 50% 测量值求均值，产生带自然抖动的变体
        aug_maps = []
        for _ in range(self.cfg.NUM_AUGMENTATIONS):
            m = torch.full_like(mean_map, fill)
            for (r, c, a), rss_list in raw_samples.items():
                if len(rss_list) == 0:
                    continue
                n = max(1, len(rss_list) // 2)
                sampled = np.random.choice(rss_list, size=n, replace=False)
                m[a, r, c] = self.normalize(float(np.mean(sampled)))
            aug_maps.append(m)

        augmented_maps = torch.stack(aug_maps)          # [N, 25, 5, 26]
        return mean_map, augmented_maps

    # ---------- 加载 Winter ----------
    def load_winter(self):
        """
        返回:
            winter_map   [25, 5, 26]  仅第 3 行有值
            spatial_mask [1, 5, 26]   第 3 行 =1，其余 =0
            device_mask  [25, 1, 1]   Winter 中存在的设备 =1
        """
        with open(self.cfg.WINTER_DATA_PATH, "r") as f:
            raw = json.load(f)

        fill = self.normalize(self.cfg.RSS_MIN)
        winter_map = torch.full((self.num_aps, self.cfg.NUM_ROWS, self.cfg.NUM_COLS), fill)
        spatial_mask = torch.zeros(1, self.cfg.NUM_ROWS, self.cfg.NUM_COLS)
        device_mask = torch.zeros(self.num_aps, 1, 1)

        row_idx = 2  # 第 3 行 (0-index)
        for col_key, devices in raw.items():
            ci = int(col_key) - 1
            spatial_mask[0, row_idx, ci] = 1.0
            for dev_id, measurements in devices.items():
                if dev_id not in self.dev2idx:
                    continue
                ai = self.dev2idx[dev_id]
                rss_list = [m["rss"] for m in measurements]
                if len(rss_list) == 0:
                    continue
                winter_map[ai, row_idx, ci] = self.normalize(float(np.mean(rss_list)))
                device_mask[ai, 0, 0] = 1.0

        return winter_map, spatial_mask, device_mask

    # ---------- 填充 / 裁剪 ----------
    def pad(self, t):
        """[..., 5, 26] → [..., 8, 32]"""
        return F.pad(t, (0, self.cfg.PAD_COLS - self.cfg.NUM_COLS,
                         0, self.cfg.PAD_ROWS - self.cfg.NUM_ROWS))

    def crop(self, t):
        """[..., 8, 32] → [..., 5, 26]"""
        return t[..., : self.cfg.NUM_ROWS, : self.cfg.NUM_COLS]


# =========================================================================
# 3. 模型定义
# =========================================================================
def create_unet(num_aps: int = 25):
    """
    条件 Inpainting UNet

    输入通道 = 25 (x_t) + 1 (mask) + 25 (masked_map) = 51
    输出通道 = 25 (预测噪声)
    空间尺寸 = 8 × 32 → 三层下采样后 1 × 4，可正常工作
    """
    return UNet2DModel(
        sample_size=(8, 32),
        in_channels=num_aps * 2 + 1,       # 51
        out_channels=num_aps,               # 25
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=32,
    )


# =========================================================================
# 4. 多样化掩膜生成
# =========================================================================
def generate_diverse_masks(batch_size: int, cfg: Config) -> torch.Tensor:
    """
    训练时随机生成多种缺失模式，使模型泛化到任意缺失场景：
      - single_row (40%): 最接近 Winter 的场景
      - multi_row  (20%): 保留 2-3 行
      - random     (20%): 随机散点保留 30-70%
      - block      (20%): 保留一个矩形子区域
    """
    masks = torch.zeros(batch_size, 1, cfg.NUM_ROWS, cfg.NUM_COLS)

    for i in range(batch_size):
        mode = np.random.choice(
            ["single_row", "multi_row", "random", "block"],
            p=[0.4, 0.2, 0.2, 0.2],
        )
        if mode == "single_row":
            r = np.random.randint(0, cfg.NUM_ROWS)
            masks[i, 0, r, :] = 1.0

        elif mode == "multi_row":
            n = np.random.randint(2, min(4, cfg.NUM_ROWS + 1))
            rows = np.random.choice(cfg.NUM_ROWS, n, replace=False)
            masks[i, 0, rows, :] = 1.0

        elif mode == "random":
            ratio = np.random.uniform(0.3, 0.7)
            flat = (np.random.rand(cfg.NUM_ROWS * cfg.NUM_COLS) < ratio).astype(np.float32)
            masks[i, 0] = torch.from_numpy(flat.reshape(cfg.NUM_ROWS, cfg.NUM_COLS))

        else:  # block
            r1, r2 = sorted(np.random.choice(cfg.NUM_ROWS + 1, 2, replace=False))
            c1, c2 = sorted(np.random.choice(cfg.NUM_COLS + 1, 2, replace=False))
            r2 = max(r2, r1 + 1)
            c2 = max(c2, c1 + 1)
            masks[i, 0, r1:r2, c1:c2] = 1.0

    return masks


# =========================================================================
# 5. 训练
# =========================================================================
def train(cfg: Config):
    print("=" * 60)
    print("  BLE 指纹地图补全 — 训练阶段")
    print("=" * 60)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  设备: {device}")

    ds = BLEFingerprintDataset(cfg)
    model = create_unet(ds.num_aps).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  模型参数量: {param_count:.2f} M")

    noise_scheduler = DDPMScheduler(num_train_timesteps=cfg.NUM_TRAIN_TIMESTEPS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    # ---- 加载真实数据 ----
    print("  加载 Spring 数据 ...")
    mean_map, aug_maps = ds.load_spring()
    all_maps = torch.cat([mean_map.unsqueeze(0), aug_maps], dim=0)  # [N, 25, 5, 26]
    all_maps_pad = ds.pad(all_maps)                                  # [N, 25, 8, 32]
    print(f"  训练样本: {all_maps_pad.shape[0]} 张 (1 均值 + {cfg.NUM_AUGMENTATIONS} 增强)")

    # ---- 训练循环 ----
    model.train()
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    best_loss = float("inf")
    losses = []

    for epoch in range(1, cfg.EPOCHS + 1):
        idx = torch.randint(0, all_maps_pad.shape[0], (cfg.BATCH_SIZE,))
        clean = all_maps_pad[idx].to(device)
        bs = clean.shape[0]

        # 随机掩膜 → 填充
        masks = ds.pad(generate_diverse_masks(bs, cfg)).to(device)
        cond = clean * masks                                  # 条件残缺图

        # 前向扩散
        noise = torch.randn_like(clean)
        t = torch.randint(0, cfg.NUM_TRAIN_TIMESTEPS, (bs,), device=device).long()
        noisy = noise_scheduler.add_noise(clean, noise, t)

        # 拼接: [noisy, mask, cond]
        inp = torch.cat([noisy, masks, cond], dim=1)          # [B, 51, 8, 32]
        pred = model(inp, t).sample                            # [B, 25, 8, 32]

        # 加权 MSE：未知区域权重更大
        weight = (1.0 - masks) * 0.7 + 0.3                    # 未知=1.0, 已知=0.3
        loss = (weight * (pred - noise) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_sched.step()

        lv = loss.item()
        losses.append(lv)
        if lv < best_loss:
            best_loss = lv
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best.pt"))

        if epoch % 200 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d}/{cfg.EPOCHS} | Loss {lv:.6f} | Best {best_loss:.6f} | LR {lr_now:.2e}")

    torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "final.pt"))
    np.save(os.path.join(cfg.CHECKPOINT_DIR, "loss_curve.npy"), np.array(losses))
    print(f"\n  训练完成 — Best Loss: {best_loss:.6f}")

    return model, noise_scheduler, ds


# =========================================================================
# 6. 推理 (DDIM + RePaint)
# =========================================================================
@torch.no_grad()
def infer(model, train_scheduler, ds: BLEFingerprintDataset, cfg: Config):
    print("\n" + "=" * 60)
    print("  BLE 指纹地图补全 — 推理阶段 (DDIM + RePaint)")
    print("=" * 60)

    device = next(model.parameters()).device
    model.eval()

    # 加载 Winter
    w_map, w_mask, dev_mask = ds.load_winter()
    print(f"  Winter 已知设备: {int(dev_mask.sum())}/25, 已知行: 第 3 行")

    w_map_p = ds.pad(w_map).unsqueeze(0).to(device)       # [1, 25, 8, 32]
    w_mask_p = ds.pad(w_mask).unsqueeze(0).to(device)      # [1, 1,  8, 32]
    cond = w_map_p * w_mask_p

    # DDIM 调度器
    ddim = DDIMScheduler(
        num_train_timesteps=cfg.NUM_TRAIN_TIMESTEPS,
        beta_schedule="linear",
    )
    ddim.set_timesteps(cfg.NUM_INFERENCE_STEPS)
    alphas = ddim.alphas_cumprod.to(device)

    samples = []
    print(f"  采样 {cfg.NUM_SAMPLES} 次 × {cfg.NUM_INFERENCE_STEPS} 步 ...")

    for s in range(cfg.NUM_SAMPLES):
        x = torch.randn(1, ds.num_aps, cfg.PAD_ROWS, cfg.PAD_COLS, device=device)

        for i, t in enumerate(ddim.timesteps):
            t_in = t.unsqueeze(0).to(device)
            inp = torch.cat([x, w_mask_p, cond], dim=1)
            noise_pred = model(inp, t_in).sample
            x = ddim.step(noise_pred, t, x).prev_sample

            # RePaint: 已知区域在当前噪声水平上强制对齐真实值
            if cfg.USE_REPAINT:
                if i < len(ddim.timesteps) - 1:
                    a_t = alphas[t]
                    n_k = torch.randn_like(w_map_p)
                    x_known = a_t.sqrt() * w_map_p + (1 - a_t).sqrt() * n_k
                    x = w_mask_p * x_known + (1 - w_mask_p) * x
                else:
                    x = w_mask_p * w_map_p + (1 - w_mask_p) * x

        samples.append(ds.crop(x).cpu())

    samples = torch.cat(samples)                          # [S, 25, 5, 26]
    mean_out = samples.mean(0, keepdim=True)               # [1, 25, 5, 26]
    std_out = samples.std(0, keepdim=True)

    print(f"  补全形状: {mean_out.shape}")
    print(f"  平均不确定性 (std): {std_out.mean():.4f}")

    return mean_out, std_out, w_map, w_mask


# =========================================================================
# 7. 评估
# =========================================================================
def evaluate(gen, gt, mask, ds: BLEFingerprintDataset, winter_map=None):
    """
    gen        [1, 25, 5, 26]  生成地图
    gt         [25, 5, 26]      Spring 均值地图（真值参考）
    mask       [1, 5, 26]       Winter 掩膜
    winter_map [25, 5, 26]      Winter 原始数据（用于已知区域自洽检验）
    """
    print("\n" + "=" * 60)
    print("  评估指标")
    print("=" * 60)

    g = ds.denormalize(gen.squeeze(0))                    # [25, 5, 26]
    r = ds.denormalize(gt)                                 # [25, 5, 26]

    # 全图（vs Spring 真值）
    rmse_all = ((g - r) ** 2).mean().sqrt()
    print(f"  全图 vs Spring   RMSE : {rmse_all:.2f} dBm")

    # 未知区域（vs Spring 真值，核心评估指标）
    unk = (1 - mask.squeeze(0)).unsqueeze(0).expand_as(g).bool()
    err_unk = (g - r)[unk]
    rmse_unk = (err_unk ** 2).mean().sqrt()
    mae_unk = err_unk.abs().mean()
    print(f"  未知区域 vs Spring RMSE: {rmse_unk:.2f} dBm")
    print(f"  未知区域 vs Spring MAE : {mae_unk:.2f} dBm")

    # 已知区域自洽检验（vs Winter 自身数据，验证 RePaint 保真度）
    kn = mask.squeeze(0).unsqueeze(0).expand_as(g).bool()
    if winter_map is not None:
        w = ds.denormalize(winter_map)
        err_kn_self = (g - w)[kn]
        rmse_kn_self = (err_kn_self ** 2).mean().sqrt()
        print(f"  已知区域 vs Winter RMSE: {rmse_kn_self:.2f} dBm  (RePaint 保真, 应 ≈ 0)")
    # 已知区域季节差异参考
    err_kn_cross = (g - r)[kn]
    rmse_kn_cross = (err_kn_cross ** 2).mean().sqrt()
    print(f"  已知区域 vs Spring RMSE: {rmse_kn_cross:.2f} dBm  (含季节差异, 非模型误差)")

    # 季节差异基线（Winter row3 vs Spring row3，不受模型影响）
    if winter_map is not None:
        seasonal_err = (w - r)[kn]
        seasonal_rmse = (seasonal_err ** 2).mean().sqrt()
        print(f"  季节差异基线 (W vs S)  : {seasonal_rmse:.2f} dBm")

    # CDF
    ae = err_unk.abs().numpy()
    ae_sorted = np.sort(ae)
    cdf = np.arange(1, len(ae_sorted) + 1) / len(ae_sorted)
    print("\n  误差 CDF (未知区域 vs Spring):")
    for p in (50, 75, 90, 95):
        v = ae_sorted[min(int(len(ae_sorted) * p / 100), len(ae_sorted) - 1)]
        print(f"    {p:2d}%  ≤  {v:.2f} dBm")

    return dict(
        rmse_all=rmse_all.item(),
        rmse_unk=rmse_unk.item(),
        mae_unk=mae_unk.item(),
        rmse_kn_cross=rmse_kn_cross.item(),
        cdf_x=ae_sorted,
        cdf_y=cdf,
    )


# =========================================================================
# 8. 可视化
# =========================================================================
def _add_watermark(fig, text="程序员石磊"):
    """在整张图上铺满半透明倾斜水印"""
    fig.canvas.draw()  # 先渲染一次以获取正确尺寸
    # 在图的多个位置重复绘制水印文字
    for x in np.arange(0.1, 1.0, 0.3):
        for y in np.arange(0.1, 1.0, 0.25):
            fig.text(
                x, y, text,
                fontsize=36,
                color="gray",
                alpha=0.12,
                ha="center", va="center",
                rotation=30,
                fontweight="bold",
                transform=fig.transFigure,
                zorder=999,
            )


def visualize(gen, gt, winter_map, mask, ds, metrics, cfg):
    g_rss = ds.denormalize(gen.squeeze(0))
    r_rss = ds.denormalize(gt)
    w_rss = ds.denormalize(winter_map)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("基于扩散模型的 BLE 指纹地图补全", fontsize=18, fontweight="bold", y=0.98)

    ap = 5
    vmin, vmax = -120, -60

    # (0,0) Spring 真值
    im = axes[0, 0].imshow(r_rss[ap].numpy(), cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0, 0].set_title(f"Spring 真值全图 (AP {ds.DEVICE_LIST[ap]})", fontsize=13)
    axes[0, 0].set_ylabel("行")
    plt.colorbar(im, ax=axes[0, 0], label="RSS (dBm)")

    # (0,1) Winter 残缺
    im = axes[0, 1].imshow(w_rss[ap].numpy(), cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0, 1].set_title("Winter 残缺图 (仅第3行)", fontsize=13)
    plt.colorbar(im, ax=axes[0, 1], label="RSS (dBm)")

    # (0,2) 补全结果
    im = axes[0, 2].imshow(g_rss[ap].numpy(), cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0, 2].set_title("扩散模型补全结果", fontsize=13)
    plt.colorbar(im, ax=axes[0, 2], label="RSS (dBm)")

    # (1,0) 误差热图
    err = (g_rss - r_rss)[ap].abs().numpy()
    im = axes[1, 0].imshow(err, cmap="Reds", aspect="auto")
    axes[1, 0].set_title("绝对误差热图", fontsize=13)
    axes[1, 0].set_ylabel("行")
    plt.colorbar(im, ax=axes[1, 0], label="|误差| (dBm)")

    # (1,1) CDF
    axes[1, 1].plot(metrics["cdf_x"], metrics["cdf_y"] * 100, "b-", lw=2)
    axes[1, 1].set_xlabel("绝对误差 (dBm)")
    axes[1, 1].set_ylabel("累积概率 (%)")
    axes[1, 1].set_title("误差累积分布函数 (未知区域)", fontsize=13)
    axes[1, 1].axhline(90, color="r", ls="--", alpha=.5, label="90%")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=.3)

    # (1,2) 训练曲线
    lp = os.path.join(cfg.CHECKPOINT_DIR, "loss_curve.npy")
    if os.path.exists(lp):
        lc = np.load(lp)
        axes[1, 2].plot(lc, "g-", alpha=.25)
        w = min(50, max(1, len(lc) // 10))
        smooth = np.convolve(lc, np.ones(w) / w, mode="valid")
        axes[1, 2].plot(range(w - 1, len(lc)), smooth, "g-", lw=2)
        axes[1, 2].set_xlabel("训练轮次 (Epoch)")
        axes[1, 2].set_ylabel("损失 (Loss)")
        axes[1, 2].set_title("训练损失曲线", fontsize=13)
        axes[1, 2].set_yscale("log")
        axes[1, 2].grid(True, alpha=.3)
    else:
        axes[1, 2].text(.5, .5, "暂无训练数据", ha="center", va="center",
                         transform=axes[1, 2].transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ---- 半透明水印 ----
    _add_watermark(fig)

    out = "results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  可视化已保存 → {out}")


# =========================================================================
# 9. 主入口
# =========================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练，加载已有模型直接推理+出图")
    args = parser.parse_args()

    cfg = Config()
    ds = BLEFingerprintDataset(cfg)

    if args.skip_train:
        # ---- 仅推理出图模式：加载已有最佳模型 ----
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        model = create_unet(ds.num_aps).to(device)
        ckpt = os.path.join(cfg.CHECKPOINT_DIR, "best.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"  已加载模型: {ckpt}")
        scheduler = DDPMScheduler(num_train_timesteps=cfg.NUM_TRAIN_TIMESTEPS)
    else:
        # ---- 完整训练 ----
        model, scheduler, ds = train(cfg)

    # ② 推理补全
    gen_map, uncertainty, winter_map, winter_mask = infer(model, scheduler, ds, cfg)

    # ③ 加载真值用于评估（Spring 均值地图）
    gt_map, _ = ds.load_spring()

    # ④ 评估
    metrics = evaluate(gen_map, gt_map, winter_mask, ds, winter_map=winter_map)

    # ⑤ 可视化
    visualize(gen_map, gt_map, winter_map, winter_mask, ds, metrics, cfg)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LFM(nn.Module):
    def __init__(self, window_size, kernel_size, stride, condition_emb_dim, dropout_rate=0.05):
        super(LFM, self).__init__()
        self.W = window_size
        self.w = kernel_size
        self.s = stride
        # 使用 FFT 的实部与虚部拼接，维度翻倍（全频谱）
        self.d_k = 2 * self.w
        self.attention = nn.MultiheadAttention(embed_dim=self.d_k, num_heads=1, batch_first=True)
        # 将注意力后的局部频域特征映射到条件嵌入维度
        self.mlp = nn.Sequential(
            nn.Linear(self.d_k, condition_emb_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # x shape: (batch, window_size)
        sub_windows = x.unfold(dimension=1, size=self.w, step=self.s)
        # sub_windows shape: (batch, num_sub_windows, kernel_size)

        # Apply FFT，并拼接实部与虚部
        freq_repr = torch.fft.fft(sub_windows, dim=-1)
        freq_feat = torch.cat([freq_repr.real, freq_repr.imag], dim=-1) # (batch, num_sub_windows, 2*w)

        q = freq_feat[:, -1:, :] # (batch, 1, 2*w)
        k = v = freq_feat # (batch, num_sub_windows, 2*w)

        attn_output, _ = self.attention(q, k, v)
        # attn_output shape: (batch, 1, 2*(w/2 + 1))
        local_feat = attn_output.squeeze(1)
        return self.mlp(local_feat)

class GFM(nn.Module):
    def __init__(self, window_size, condition_emb_dim, dropout_rate=0.05):
        super(GFM, self).__init__()
        self.W = window_size
        # FFT 实部与虚部拼接，维度翻倍（全频谱）
        self.fft_dim = 2 * self.W
        self.mlp = nn.Sequential(
            nn.Linear(self.fft_dim, condition_emb_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # x shape: (batch, window_size)
        freq_repr = torch.fft.fft(x, dim=-1)
        freq_feat = torch.cat([freq_repr.real, freq_repr.imag], dim=-1) # (batch, 2*W)
        global_feature = self.mlp(freq_feat)
        return global_feature

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc_mu = nn.Sequential(
            nn.Linear(100, latent_dim),
        )
        # 直接输出方差（variance），通过 Softplus 保证为正
        self.fc_var = nn.Sequential(
            nn.Linear(100, latent_dim),
            nn.Softplus()
        )

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=-1)
        h = F.tanh(self.fc1(xc))
        h = F.tanh(self.fc2(h))
        mu = self.fc_mu(h)
        var = self.fc_var(h)
        return mu, var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, condition_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        # 输出同时预测重建的均值与方差（variance）
        self.fc3 = nn.Sequential(
            nn.Linear(100, output_dim * 2),
        )
        self.fc_mu_x = nn.Linear(100, output_dim)
        # 使用 Softplus 保证输出为正，即直接作为方差
        self.fc_var_x = nn.Sequential(
            nn.Linear(100, output_dim),
            nn.Softplus()
        )

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=-1)
        h = F.tanh(self.fc1(zc))
        h = F.tanh(self.fc2(h))
        recon_mu = self.fc_mu_x(h)
        recon_var = self.fc_var_x(h)
        return recon_mu, recon_var

class FCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, kernel_size=24, stride=8, condition_emb_dim=64, dropout_rate=0.05):
        super(FCVAE, self).__init__()
        self.lfm = LFM(input_dim, kernel_size, stride, condition_emb_dim, dropout_rate)
        self.gfm = GFM(input_dim, condition_emb_dim, dropout_rate)
        
        # LFM 与 GFM 均映射到相同的条件嵌入维度，拼接后得到总的 condition_dim
        lfm_output_dim = condition_emb_dim
        condition_dim = lfm_output_dim + condition_emb_dim  # = 2 * condition_emb_dim
        self.condition_dim = condition_dim

        self.encoder = Encoder(input_dim, latent_dim, condition_dim)
        self.decoder = Decoder(latent_dim, input_dim, condition_dim)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var.clamp_min(1e-12))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        local_feature = self.lfm(x)
        global_feature = self.gfm(x)
        c = torch.cat([local_feature, global_feature], dim=-1)
        # 维度健壮性检查
        if c.dim() != 2:
            raise RuntimeError(f"Condition feature 'c' must be 2D (batch, cond_dim), got shape={tuple(c.shape)}")
        if c.size(-1) != self.condition_dim:
            raise RuntimeError(
                f"Condition dim mismatch: got {c.size(-1)}, expected {self.condition_dim}. "
                f"local_feature={local_feature.size(-1)}, global_feature={global_feature.size(-1)}"
            )
        expected_encoder_in = self.encoder.fc1.in_features
        actual_encoder_in = x.size(-1) + c.size(-1)
        if actual_encoder_in != expected_encoder_in:
            raise RuntimeError(
                f"Encoder.fc1 in_features mismatch: expected {expected_encoder_in}, got {actual_encoder_in}. "
                f"x_dim={x.size(-1)}, c_dim={c.size(-1)}"
            )
        
        mu, var = self.encoder(x, c)
        z = self.reparameterize(mu, var)
        recon_mu, recon_var = self.decoder(z, c)
        return recon_mu, recon_var, mu, var, z

# ----------------------------
# Training & Evaluation (ELBO)
# ----------------------------
def gaussian_nll(recon_mu, recon_var, x, var_clip=(1e-8, 1e8)):
    """
    高斯 NLL：0.5 * [ log(2πσ²) + (x-μ)²/σ² ]
    recon_var 认为是方差，进行裁剪以保证数值稳定。
    """
    var = torch.clamp(recon_var, min=var_clip[0], max=var_clip[1])
    log_var = torch.log(var)
    nll_elem = 0.5 * (math.log(2 * math.pi) + log_var + (x - recon_mu) ** 2 / var)
    return nll_elem.mean()


def elbo_loss(recon_mu, recon_var, x, mu, var, z, beta=1.0):
    """
    计算 ELBO: NLL_Gaussian + beta * KL(mu, var)

    - 重构项使用高斯 NLL（由 decoder 输出 recon_mu 与 recon_var 给出）
    - KL 采用采样估计：0.5 * [ z^2 - log(var) - (z - mu)^2 / var ]
    - reduction 支持 "mean" 或 "sum"
    """
    # 高斯 NLL 作为重构损失（全局均值）
    recon_loss = gaussian_nll(recon_mu, recon_var, x)

    # KL 散度基于采样的估计（先按维度求和，再对 batch 聚合）
    var = var.clamp_min(1e-12)
    log_var = torch.log(var)
    kl_elem = 0.5 * (z.pow(2) - log_var - (z - mu).pow(2) / var)
    kl = kl_elem.mean()

    loss = recon_loss + beta * kl
    return loss, recon_loss, kl


def train_epoch(model, dataloader, optimizer, device, beta=1.0, grad_clip=None, log_interval=100):
    """单轮训练，返回 (avg_loss, avg_recon, avg_kl)。"""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # 兼容 (x, _) 或 x
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device).float()

        optimizer.zero_grad(set_to_none=True)

        recon_mu, recon_var, mu, var, z = model(x)
        loss, recon_loss, kl = elbo_loss(recon_mu, recon_var, x, mu, var, z, beta=beta)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl.item()
        num_batches += 1

        if log_interval and (batch_idx + 1) % log_interval == 0:
            print(f"Train [{batch_idx + 1}/{len(dataloader)}] loss={total_loss/num_batches:.4f} recon={total_recon/num_batches:.4f} kl={total_kl/num_batches:.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_recon = total_recon / max(num_batches, 1)
    avg_kl = total_kl / max(num_batches, 1)
    return avg_loss, avg_recon, avg_kl


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, beta=1.0):
    """单轮评估，返回 (avg_loss, avg_recon, avg_kl)。"""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0

    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device).float()

        recon_mu, recon_var, mu, var, z = model(x)
        loss, recon_loss, kl = elbo_loss(recon_mu, recon_var, x, mu, var, z, beta=beta)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_recon = total_recon / max(num_batches, 1)
    avg_kl = total_kl / max(num_batches, 1)
    return avg_loss, avg_recon, avg_kl


def fit(model, train_loader, val_loader, optimizer, device, epochs=10, beta=1.0, grad_clip=None, log_interval=100, scheduler=None):
    """
    多轮训练入口。返回训练过程中的历史指标：
    {
        "train": [(loss, recon, kl), ...],
        "val": [(loss, recon, kl), ...]
    }
    """
    history = {"train": [], "val": []}
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            beta=beta, grad_clip=grad_clip, log_interval=log_interval,
        )
        val_metrics = evaluate_epoch(model, val_loader, device, beta=beta)

        if scheduler is not None:
            # 若为 ReduceLROnPlateau 之类，可改为以 val loss 驱动
            try:
                scheduler.step(val_metrics[0])
            except Exception:
                scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train: loss={train_metrics[0]:.4f} recon={train_metrics[1]:.4f} kl={train_metrics[2]:.4f} | "
            f"val: loss={val_metrics[0]:.4f} recon={val_metrics[1]:.4f} kl={val_metrics[2]:.4f}"
        )

    return history


__all__ = [
    "LFM",
    "GFM",
    "Encoder",
    "Decoder",
    "FCVAE",
    "elbo_loss",
    "train_epoch",
    "evaluate_epoch",
    "fit",
]

# ----------------------------
# STL-like Synthetic Time Series Generation
# ----------------------------
import math

def _generate_random_seasonal(
    series_length,
    min_components=1,
    max_components=5,
    amplitude_range=(0.2, 1.5),
    period_range=(8, None),
    phase_range=(0.0, 2 * math.pi),
    device=None,
):
    """
    生成季节性分量：若干随机振幅、随机周期、随机相位的正弦相加。
    返回 shape: (series_length,)
    """
    device = device or torch.device("cpu")
    t = torch.arange(series_length, device=device, dtype=torch.float32)

    if period_range[1] is None:
        period_max = max(10, series_length // 2)
        period_min = max(2, min(period_range[0], period_max))
        period_range = (period_min, period_max)

    num_components = torch.randint(low=min_components, high=max_components + 1, size=(1,)).item()
    seasonal = torch.zeros(series_length, device=device)

    for _ in range(num_components):
        amplitude = torch.empty(1, device=device).uniform_(*amplitude_range).item()
        period = torch.randint(low=period_range[0], high=period_range[1] + 1, size=(1,), device=device).item()
        phase = torch.empty(1, device=device).uniform_(*phase_range).item()
        seasonal = seasonal + amplitude * torch.sin(2 * math.pi * t / period + phase)

    return seasonal


def _generate_trend(
    series_length,
    kind="linear",
    device=None,
    # 可调范围
    slope_range=(-0.01, 0.01),
    intercept_range=(-0.5, 0.5),
    quad_a_range=(-1e-4, 1e-4),
    quad_b_range=(-5e-3, 5e-3),
    exp_a_range=(0.5, 1.5),
    exp_b_range=(-3e-3, 3e-3),
    logistic_L_range=(0.5, 2.0),
    logistic_k_range=(1e-3, 1e-2),
    piecewise_knot_range=(0.3, 0.7),
):
    """
    生成常见趋势分量：linear/quadratic/exponential/logistic/piecewise/random_walk。
    返回 shape: (series_length,)
    """
    device = device or torch.device("cpu")
    t = torch.arange(series_length, device=device, dtype=torch.float32)
    t_norm = (t - t.min()) / max(1.0, (t.max() - t.min()).item())

    if kind == "linear":
        slope = torch.empty(1, device=device).uniform_(*slope_range).item()
        intercept = torch.empty(1, device=device).uniform_(*intercept_range).item()
        trend = slope * t + intercept
    elif kind == "quadratic":
        a = torch.empty(1, device=device).uniform_(*quad_a_range).item()
        b = torch.empty(1, device=device).uniform_(*quad_b_range).item()
        c = torch.empty(1, device=device).uniform_(*intercept_range).item()
        trend = a * (t ** 2) + b * t + c
    elif kind == "exponential":
        a = torch.empty(1, device=device).uniform_(*exp_a_range).item()
        b = torch.empty(1, device=device).uniform_(*exp_b_range).item()
        trend = a * torch.exp(b * t)
    elif kind == "logistic":
        L = torch.empty(1, device=device).uniform_(*logistic_L_range).item()
        k = torch.empty(1, device=device).uniform_(*logistic_k_range).item()
        t0 = torch.empty(1, device=device).uniform_(0.3, 0.7).item()  # 中心点在 30%-70% 处
        trend = L / (1.0 + torch.exp(-k * (t_norm - t0)))
    elif kind == "piecewise":
        knot = torch.empty(1, device=device).uniform_(*piecewise_knot_range).item()
        knot_idx = int(knot * (series_length - 1))
        slope1 = torch.empty(1, device=device).uniform_(*slope_range).item()
        slope2 = torch.empty(1, device=device).uniform_(*slope_range).item()
        intercept = torch.empty(1, device=device).uniform_(*intercept_range).item()
        trend = torch.where(
            t <= knot_idx,
            intercept + slope1 * t,
            intercept + slope1 * knot_idx + slope2 * (t - knot_idx),
        )
    elif kind == "random_walk":
        steps = torch.randn(series_length, device=device) * 0.02
        trend = torch.cumsum(steps, dim=0)
    else:
        raise ValueError(f"Unsupported trend kind: {kind}")

    return trend


def generate_stl_series(
    series_length,
    seasonal_kwargs=None,
    trend_kind=None,
    noise_std=0.05,
    level_range=(-0.3, 0.3),
    normalize=True,
    device=None,
):
    """
    生成单条 STL 思路的时序：y = level + trend + seasonal + noise。
    返回: dict(series, seasonal, trend, residual)
    """
    device = device or torch.device("cpu")
    seasonal_kwargs = seasonal_kwargs or {}

    seasonal = _generate_random_seasonal(series_length, device=device, **seasonal_kwargs)

    if trend_kind is None:
        trend_kind = [
            "linear",
            "quadratic",
            "exponential",
            "logistic",
            "piecewise",
            "random_walk",
        ][torch.randint(0, 6, size=(1,)).item()]

    trend = _generate_trend(series_length, kind=trend_kind, device=device)

    level = torch.empty(1, device=device).uniform_(*level_range).item()
    noise = torch.randn(series_length, device=device) * noise_std

    series = level + trend + seasonal + noise

    if normalize:
        # z-score 标准化：均值 0，方差 1
        def _z(x):
            mu = x.mean()
            sd = x.std(unbiased=False).clamp_min(1e-6)
            return (x - mu) / sd

        series = _z(series)
        seasonal = _z(seasonal)
        trend = _z(trend)
        noise = _z(noise)

    return {
        "series": series,        # (L,)
        "seasonal": seasonal,    # (L,)
        "trend": trend,          # (L,)
        "residual": noise,       # (L,)
    }


def generate_stl_dataset(
    num_series,
    series_length,
    seasonal_kwargs=None,
    noise_std=0.05,
    normalize=True,
    device=None,
    seed=None,
    return_components=False,
    zero_ratio=0.0,
):
    """
    生成批量 STL 风格的时序数据。
    参数:
      - zero_ratio: 恒定为0的数据比例 (0.0-1.0)
    返回:
      - X: shape (num_series, series_length)
      - 可选 components: dict of tensors
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = device or torch.device("cpu")
    X = torch.empty(num_series, series_length, device=device)

    comps = {
        "seasonal": torch.empty(num_series, series_length, device=device),
        "trend": torch.empty(num_series, series_length, device=device),
        "residual": torch.empty(num_series, series_length, device=device),
    } if return_components else None

    # 计算零值数据的数量
    num_zero_series = int(num_series * zero_ratio)
    num_normal_series = num_series - num_zero_series

    for i in range(num_series):
        if i < num_zero_series:
            # 生成恒定为0的数据
            X[i] = torch.zeros(series_length, device=device)
            if comps is not None:
                comps["seasonal"][i] = torch.zeros(series_length, device=device)
                comps["trend"][i] = torch.zeros(series_length, device=device)
                comps["residual"][i] = torch.zeros(series_length, device=device)
        else:
            # 生成正常的STL数据
            sample = generate_stl_series(
                series_length,
                seasonal_kwargs=seasonal_kwargs,
                trend_kind=None,
                noise_std=noise_std,
                normalize=normalize,
                device=device,
            )
            X[i] = sample["series"]
            if comps is not None:
                comps["seasonal"][i] = sample["seasonal"]
                comps["trend"][i] = sample["trend"]
                comps["residual"][i] = sample["residual"]

    return (X, comps) if return_components else X


def build_stl_dataloaders(
    num_train,
    num_val,
    series_length,
    batch_size=64,
    seasonal_kwargs=None,
    noise_std=0.05,
    normalize=True,
    device=None,
    seed=None,
    shuffle=True,
    zero_ratio=0.0,
):
    """
    生成合成数据并构建 DataLoader。
    参数:
      - zero_ratio: 恒定为0的数据比例 (0.0-1.0)
    """
    device = device or torch.device("cpu")
    if seed is not None:
        torch.manual_seed(seed)

    X_train = generate_stl_dataset(
        num_train, series_length,
        seasonal_kwargs=seasonal_kwargs,
        noise_std=noise_std,
        normalize=normalize,
        device=device,
        seed=None,
        return_components=False,
        zero_ratio=zero_ratio,
    )
    X_val = generate_stl_dataset(
        num_val, series_length,
        seasonal_kwargs=seasonal_kwargs,
        noise_std=noise_std,
        normalize=normalize,
        device=device,
        seed=None,
        return_components=False,
        zero_ratio=zero_ratio,
    )

    train_ds = torch.utils.data.TensorDataset(X_train)
    val_ds = torch.utils.data.TensorDataset(X_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# 导出符号
__all__ += [
    "_generate_random_seasonal",
    "_generate_trend",
    "generate_stl_series",
    "generate_stl_dataset",
    "build_stl_dataloaders",
]

# # from tools.fcvae_model import FCVAE, train_epoch, evaluate_epoch, fit
# import torch
# # from tools.fcvae_model import FCVAE, fit, build_stl_dataloaders

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 构造合成数据的 DataLoader
# series_length = 256
# train_loader, val_loader = build_stl_dataloaders(
#     num_train=100000,
#     num_val=1000,
#     series_length=series_length,
#     batch_size=128,
#     seasonal_kwargs=dict(min_components=2, max_components=10, amplitude_range=(0.1, 3), period_range=(8, 1440)),
#     noise_std=0.05,
#     normalize=True,
#     device=device,
#     seed=42,
#     zero_ratio=0.01,  # 10%的数据为恒定为0的数据
# )

# # 定义并训练 FCVAE
# model = FCVAE(input_dim=series_length, latent_dim=8, kernel_size=24, stride=8, condition_emb_dim=64, dropout_rate=0.05).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

# history = fit(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     optimizer=optimizer,
#     device=device,
#     epochs=100,
#     beta=1.0,               # 可做 KL 退火：前几轮小，后面逐步增大
#     grad_clip=1.0,
#     log_interval=50,
# )

# # 训练完成后保存模型权重
# save_path = "fcvae_model_final.pth"
# torch.save(model.state_dict(), save_path)
# print(f"Model saved to {save_path}")

# # 评测阶段：重新构建同结构模型并加载权重后评估
# loaded_model = FCVAE(
#     input_dim=series_length,
#     latent_dim=8,
#     kernel_size=24,
#     stride=8,
#     condition_emb_dim=64,
#     dropout_rate=0.05,
# ).to(device)
# loaded_model.load_state_dict(torch.load(save_path, map_location=device))

# val_loss, val_recon, val_kl = evaluate_epoch(
#     model=loaded_model,
#     dataloader=val_loader,
#     device=device,
#     beta=1.0,
# )
# print(f"Loaded model eval | val: loss={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f}")
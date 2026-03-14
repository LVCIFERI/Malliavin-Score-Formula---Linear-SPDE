
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, math, time, argparse, copy, random
from scipy.spatial import distance
from statistics import NormalDist
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. UTILS
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalise_grid(x):
    x = x.float()
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def build_coord_features(name, x_grid):
    x = normalise_grid(x_grid)
    x_centered = 2.0 * x - 1.0
    feats = [x_centered]

    if name == 'Quadratic':
        feats += [x_centered ** 2, x_centered ** 3]
    elif name == 'Melbourne':
        feats += [
            torch.sin(2 * math.pi * x), torch.cos(2 * math.pi * x),
            torch.sin(4 * math.pi * x), torch.cos(4 * math.pi * x),
        ]
    elif name == 'Gridwatch':
        feats += [
            torch.sin(2 * math.pi * x), torch.cos(2 * math.pi * x),
            torch.sin(4 * math.pi * x), torch.cos(4 * math.pi * x),
            torch.sin(8 * math.pi * x), torch.cos(8 * math.pi * x),
        ]
    else:
        feats += [x_centered ** 2]
    return torch.stack(feats, dim=0)


# ============================================================
# 1. DATASETS (exactly as HDM, Lim et al. NeurIPS 2023)
# ============================================================

class QuadraticDataset(Dataset):
    """f(x;a,b) = ax² + b, a ∈ {-1,1}, b ~ N(0,1), x ∈ [-10,10], /50 normalised."""
    def __init__(self, num_data=1000, num_points=100, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.x = torch.linspace(-10., 10., num_points).unsqueeze(0).repeat(num_data, 1)
        a = torch.randint(0, 2, (num_data, 1), generator=g).repeat(1, num_points) * 2 - 1
        b = torch.randn(num_data, 1, generator=g).repeat(1, num_points)
        self.y = (a * self.x ** 2 + b) / 50.0
        self.num_data = num_data
    def __len__(self): return self.num_data
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class MelbourneDataset(Dataset):
    """Melbourne Pedestrian counting — 24-hour time series, standardised.
    Source: timeseriesclassification.com/description.php?Dataset=MelbournePedestrian"""
    def __init__(self, phase='train', seed=42):
        super().__init__()
        from scipy.io.arff import loadarff
        raw = loadarff('./data/MelbournePedestrian_TEST.arff')
        samples = []
        for row in raw[0]:
            vals = [float(row[f'att{i+1}']) for i in range(24)]
            if not any(np.isnan(vals)):
                samples.append(vals)
        y_full = np.array(samples, dtype=np.float32)
        rng = np.random.default_rng(seed)
        n = len(y_full)
        idx = rng.permutation(n)
        split = int(n * 0.8)
        train_idx, test_idx = idx[:split], idx[split:]
        y = y_full[train_idx] if phase == 'train' else y_full[test_idx]
        self.mean = y_full[train_idx].mean()
        self.std = y_full[train_idx].std() + 1e-8
        y = (y - self.mean) / self.std
        self.x = torch.arange(24, dtype=torch.float32).unsqueeze(0).repeat(len(y), 1)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class GridwatchDataset(Dataset):
    """UK grid energy demand — 288 five-minute intervals per day, instance-normalised.
    Source: gridwatch.templar.co.uk/download.php"""
    def __init__(self, phase='train', seed=87):
        super().__init__()
        import pandas as pd
        data = pd.read_csv('./data/gridwatch_clean.csv', index_col=0)
        num_points, num_samples = 288, 1013
        demand = data[' demand'].values
        y_full = np.zeros((num_samples, num_points), dtype=np.float32)
        for i in range(num_samples):
            y_full[i] = demand[i*num_points:(i+1)*num_points]
        mu = y_full.mean(axis=1, keepdims=True)
        std = y_full.std(axis=1, keepdims=True) + 1e-8
        y_full = (y_full - mu) / std
        rng = np.random.default_rng(seed)
        n = len(y_full)
        idx = rng.permutation(n)
        split = int(n * 0.8)
        train_idx, test_idx = idx[:split], idx[split:]
        y = y_full[train_idx] if phase == 'train' else y_full[test_idx]
        self.x = torch.arange(num_points, dtype=torch.float32).unsqueeze(0).repeat(len(y), 1)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


# ============================================================
# 2. GP / PERIODIC KERNEL NOISE (Hilbert-valued Wiener process)
# ============================================================

class HilbertNoise:
    """Q^{1/2}-sampler and regularised (Q+λI)^(-1/2) whitening operator.

    Improvements over the original version:
      1) the kernel is built on the actual dataset grid, not an unrelated [-10,10] grid;
      2) Melbourne can use a periodic-SE hybrid kernel matching daily seasonality;
      3) the whitening floor is configurable and numerically safer."""
    def __init__(
        self,
        x_grid,
        kernel_type='se',
        hyp_len=0.2,
        hyp_gain=1.0,
        periodic_len=0.35,
        periodic_mix=0.7,
        whiten_floor=1.0,
        jitter=1e-6,
    ):
        x = normalise_grid(x_grid).flatten().unsqueeze(-1)
        D2 = torch.cdist(x / max(hyp_len, 1e-6), x / max(hyp_len, 1e-6)).pow(2)

        if kernel_type == 'periodic_se':
            raw_d = torch.cdist(x, x)
            K_per = torch.exp(
                -2.0 * torch.sin(math.pi * raw_d).pow(2) / max(periodic_len ** 2, 1e-6)
            )
            K_se = torch.exp(-0.5 * D2)
            K = hyp_gain * (periodic_mix * K_per + (1.0 - periodic_mix) * K_se)
        elif kernel_type == 'matern12':
            D = torch.sqrt(torch.clamp(D2, min=1e-12))
            K = hyp_gain * torch.exp(-D)
        else:
            K = hyp_gain * torch.exp(-0.5 * D2)

        K = K + jitter * torch.eye(K.shape[0], dtype=K.dtype)
        eig_val, eig_vec = np.linalg.eigh(K.cpu().numpy())
        eig_val = np.maximum(eig_val, 0.0)
        V = torch.from_numpy(eig_vec).float()
        lam = torch.from_numpy(eig_val).float()

        self.M = V @ torch.diag(torch.sqrt(lam))

        lam_floor = whiten_floor * lam.mean()
        lam_reg = lam + lam_floor
        lam_inv_sqrt = 1.0 / torch.sqrt(lam_reg)
        lam_inv_sqrt = lam_inv_sqrt / lam_inv_sqrt.max().clamp_min(1e-8)
        self.W = V @ torch.diag(lam_inv_sqrt) @ V.T

        eff_rank = (lam.sum() ** 2 / (lam.pow(2).sum() + 1e-12)).item()
        cond = (lam_reg.max() / lam_reg.min()).item()
        print(
            f"  GP kernel: type={kernel_type}, grid={len(x_grid)}, eff_rank={eff_rank:.1f}, "
            f"λ_floor={lam_floor:.4f}, cond={cond:.1f}"
        )

    def sample(self, size, device=None):
        z = torch.randn(list(size), device=device)
        M = self.M.to(device or z.device)
        return z @ M.T

    def whiten(self, x):
        return x @ self.W.to(x.device).T


# ============================================================
# 3. COSINE VP-SDE
# ============================================================

class CosineVPSDE:
    def __init__(self):
        self.s = 0.008
        self.T = 0.9946
        self.eps = 1e-5
        self.log_alpha_0 = math.log(math.cos(self.s / (1. + self.s) * math.pi / 2.))

    def marginal_log_mean_coeff(self, t):
        return torch.log(torch.clamp(
            torch.cos((t + self.s) / (1. + self.s) * math.pi / 2.), 1e-8, 1.
        )) - self.log_alpha_0

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(torch.clamp(
            1. - torch.exp(2 * self.marginal_log_mean_coeff(t)), min=1e-8
        ))

    def beta(self, t):
        return math.pi / (1 + self.s) * torch.tan(
            (t + self.s) / (1 + self.s) * math.pi / 2
        )

    def snr(self, t):
        log_a = self.marginal_log_mean_coeff(t)
        return torch.exp(2 * log_a) / torch.clamp(1 - torch.exp(2 * log_a), min=1e-8)


# ============================================================
# 4. 1D FOURIER NEURAL OPERATOR (improved)
# ============================================================

class SpectralConv1d(nn.Module):
    """True channel-mixing spectral convolution."""
    def __init__(self, channels, modes, temb_dim, dim):
        super().__init__()
        n_freq = dim // 2 + 1
        self.n_modes = min(modes, n_freq)
        scale = 1.0 / math.sqrt(channels)
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, self.n_modes, dtype=torch.cfloat)
        )
        self.temb_proj = nn.Linear(temb_dim, channels)
        nn.init.zeros_(self.temb_proj.bias)

    def forward(self, x, temb):
        B, C, D = x.shape
        x = x + self.temb_proj(F.silu(temb)).unsqueeze(-1)
        x_ft = torch.fft.rfft(x, norm='ortho')
        out_ft = torch.zeros(B, C, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.n_modes] = torch.einsum(
            'bcm,com->bom', x_ft[:, :, :self.n_modes], self.weight
        )
        return torch.fft.irfft(out_ft, n=D, norm='ortho')


class FourierLayer(nn.Module):
    """Residual Fourier block with channel-mixing spectral conv and local MLP."""
    def __init__(self, channels, modes, temb_dim, dim, mlp_expansion=3.0, dropout=0.0):
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.spec_conv = SpectralConv1d(channels, modes, temb_dim, dim)
        self.local_conv = nn.Conv1d(channels, channels, 1)
        hidden = int(channels * mlp_expansion)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, 1),
        )

    def forward(self, x, temb):
        h = F.gelu(self.norm1(x))
        x = x + self.spec_conv(h, temb) + self.local_conv(h)
        h = F.gelu(self.norm2(x))
        x = x + self.mlp(h)
        return x


class FNO1D(nn.Module):
    """ε-prediction network with coordinate features and self-conditioning."""
    def __init__(
        self,
        dim,
        coord_features,
        hidden=256,
        modes=50,
        n_layers=4,
        temb_dim=256,
        self_condition=True,
        dropout=0.0,
    ):
        super().__init__()
        self.temb_dim = temb_dim
        self.self_condition = self_condition
        self.register_buffer('coord_features', coord_features.float(), persistent=False)

        in_channels = 1 + coord_features.shape[0] + (1 if self_condition else 0)

        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.time_to_hidden = nn.Linear(temb_dim, hidden)

        self.lifting = nn.Conv1d(in_channels, hidden, 1)
        self.layers = nn.ModuleList([
            FourierLayer(hidden, modes, temb_dim, dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.GroupNorm(min(8, hidden), hidden)
        self.projection = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 1),
        )

    def get_temb(self, t, dim):
        half = dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        ang = (1000 * t).float().unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb

    def forward(self, x, t, self_cond=None):
        B, D = x.shape
        temb = self.time_mlp(self.get_temb(t, self.temb_dim))
        coord = self.coord_features.unsqueeze(0).expand(B, -1, -1)

        pieces = [x.unsqueeze(1)]
        if self.self_condition:
            if self_cond is None:
                self_cond = torch.zeros_like(x)
            pieces.append(self_cond.unsqueeze(1))
        pieces.append(coord)

        h = self.lifting(torch.cat(pieces, dim=1))
        h = h + self.time_to_hidden(temb).unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, temb)
        h = F.gelu(self.final_norm(h))
        return self.projection(h).squeeze(1)


# ============================================================
# 5. EMA + TRAINING
# ============================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model, step=None):
        decay = self.decay
        if step is not None:
            decay = min(self.decay, (1 + step) / (10 + step))
        for sp, mp in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(decay).add_(mp.data, alpha=1 - decay)

    def get(self):
        return self.shadow


def train(model, dataset, sde, W, cfg):
    """Training improvements:
      - coordinate-aware FNO with self-conditioning
      - true residual spectral blocks
      - min-SNR weighting
      - cosine LR decay with warmup
      - blended Malliavin loss after warmup"""
    device = cfg['device']
    use_mal = cfg.get('use_malliavin', True)

    model = model.to(device)
    model.train()
    ema = EMA(model, decay=cfg.get('ema_decay', 0.9995))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        betas=(0.9, 0.99),
        weight_decay=cfg.get('weight_decay', 1e-4),
        amsgrad=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=(device == 'cuda'),
    )

    total_steps = cfg['n_epochs'] * len(loader)
    warmup_steps = max(100, int(0.02 * total_steps))
    min_lr = cfg.get('min_lr', cfg['lr'] / 50)
    snr_clip = cfg.get('snr_clip', 5.0)

    def set_lr(step):
        if step < warmup_steps:
            lr = cfg['lr'] * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = min_lr + (cfg['lr'] - min_lr) * cosine
        for g in opt.param_groups:
            g['lr'] = lr
        return lr

    losses = []
    step = 0
    print(f"  Malliavin loss: {'on' if use_mal else 'off'}")
    print(f"  Self-conditioning: {cfg.get('self_condition', True)}")
    print(f"  SNR clip: {snr_clip:.2f}")

    for epoch in range(cfg['n_epochs']):
        for _, y in loader:
            lr_now = set_lr(step)
            y = y.to(device)
            B = y.shape[0]

            t = torch.rand(B, device=device) * (sde.T - sde.eps) + sde.eps
            e = W.sample(y.shape, device=device)
            alpha_t = sde.diffusion_coeff(t)[:, None]
            sigma_t = sde.marginal_std(t)[:, None]
            u_t = y * alpha_t + e * sigma_t

            self_cond = None
            if cfg.get('self_condition', True) and torch.rand(1).item() < 0.5:
                with torch.no_grad():
                    self_cond = model(u_t, t, self_cond=None).detach()

            pred = model(u_t, t, self_cond=self_cond)
            error = pred + e  # pred ≈ -e

            loss_std = error.pow(2).mean(dim=1)

            if use_mal:
                error_whitened = W.whiten(error)
                loss_mal = error_whitened.pow(2).mean(dim=1)
                progress = min(1.0, step / max(1, int(0.15 * total_steps)))
                mal_w = cfg.get('malliavin_final_weight', 0.7) * progress
                per_sample = (1.0 - mal_w) * loss_std + mal_w * loss_mal
            else:
                loss_mal = loss_std
                per_sample = loss_std
                mal_w = 0.0

            snr = sde.snr(t)
            weights = torch.clamp(snr, max=snr_clip) / torch.clamp(snr, min=1e-8)
            loss = (weights * per_sample).mean()

            if step == 0:
                print(
                    f"  Step 0 OK: L_std={loss_std.mean().item():.4f}, "
                    f"L_mal={loss_mal.mean().item():.4f}, lr={lr_now:.2e}"
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()
            ema.update(model, step=step)

            if step % 200 == 0:
                losses.append(loss_std.mean().item())
            if step % 2000 == 0 and step > 0:
                print(
                    f"  Step {step:6d} | lr={lr_now:.2e} | "
                    f"L_std={loss_std.mean().item():.4f} | "
                    f"L_mal={loss_mal.mean().item():.4f} | mal_w={mal_w:.2f}"
                )
            step += 1

    return losses, ema.get()


# ============================================================
# 6. SAMPLING (improved reverse SPDE)
# ============================================================

@torch.no_grad()
def sample_fn(
    model,
    sde,
    W,
    n,
    dim,
    steps=1000,
    device='cpu',
    thr=10.0,
    self_condition=True,
    heun=True,
    final_denoise=True,
    corrector_steps=0,
    corrector_snr=0.08,
):
    """Reverse-time solver with:
      - self-conditioning at inference
      - Heun correction on the drift
      - optional late-stage Langevin corrector
      - final x0 denoising step"""
    model.eval()
    ts = torch.linspace(sde.T, sde.eps, steps + 1, device=device)
    sigma_T = sde.marginal_std(ts[0]).item()
    u = W.sample((n, dim), device=device) * sigma_T
    eps_sc = torch.zeros_like(u) if self_condition else None

    def drift_fn(x, t_scalar, eps_pred):
        vec_t = torch.full((x.shape[0],), t_scalar.item(), device=x.device)
        beta_t = sde.beta(vec_t)[:, None]
        score = eps_pred / sde.marginal_std(vec_t)[:, None]
        return 0.5 * beta_t * x + beta_t * score

    for i in range(steps):
        t = ts[i]
        t_next = ts[i + 1]
        dt = (t - t_next).item()

        vec_t = torch.full((n,), t.item(), device=device)
        eps_hat = model(u, vec_t, self_cond=eps_sc if self_condition else None)
        drift = drift_fn(u, t, eps_hat)

        noise = W.sample(u.shape, device=device)
        beta_t = sde.beta(vec_t)[:, None]
        u_euler = u + dt * drift + torch.sqrt(torch.clamp(beta_t * dt, min=1e-12)) * noise

        if heun and i < steps - 1:
            vec_next = torch.full((n,), t_next.item(), device=device)
            eps_next = model(
                u_euler, vec_next,
                self_cond=eps_hat if self_condition else None
            )
            drift_next = drift_fn(u_euler, t_next, eps_next)
            u = u + 0.5 * dt * (drift + drift_next) + \
                torch.sqrt(torch.clamp(beta_t * dt, min=1e-12)) * noise
            if self_condition:
                eps_sc = eps_next
        else:
            u = u_euler
            if self_condition:
                eps_sc = eps_hat

        if corrector_steps > 0 and i > int(0.65 * steps):
            sigma_t = sde.marginal_std(vec_t)[:, None]
            for _ in range(corrector_steps):
                eps_corr = model(u, vec_t, self_cond=eps_sc if self_condition else None)
                score = eps_corr / sigma_t
                z = W.sample(u.shape, device=device)
                grad_norm = score.norm(dim=1).mean().clamp_min(1e-6)
                noise_norm = z.norm(dim=1).mean().clamp_min(1e-6)
                step_size = 2.0 * (corrector_snr * noise_norm / grad_norm) ** 2
                u = u + step_size * score + math.sqrt(2.0 * step_size) * z
                if self_condition:
                    eps_sc = eps_corr

        norms = u.norm(dim=1)
        mask = norms > thr
        if mask.any():
            u[mask] = u[mask] / norms[mask, None] * thr

    if final_denoise:
        vec_eps = torch.full((n,), sde.eps, device=device)
        eps_hat = model(u, vec_eps, self_cond=eps_sc if self_condition else None)
        alpha = sde.diffusion_coeff(vec_eps)[:, None]
        sigma = sde.marginal_std(vec_eps)[:, None]
        u = (u + sigma * eps_hat) / alpha

    return u


# ============================================================
# 7. MMD POWER TEST (exact HDM protocol)
# ============================================================

def _kern(X, Y, g=-1):
    XY = np.vstack((X, Y))
    D = distance.cdist(XY, XY, 'euclidean') / np.sqrt(X.shape[1])
    if g == -1:
        pos = D[D > 0]
        g = np.median(pos) if len(pos) else 1.0
    return np.exp(-0.5 / max(g ** 2, 1e-12) * D ** 2)

def _mmd2(K, M, N):
    return ((K[:M, :M].sum() - np.trace(K[:M, :M])) / (M * (M - 1))
            - 2 * K[:M, M:].sum() / (M * N)
            + (K[M:, M:].sum() - np.trace(K[M:, M:])) / (N * (N - 1)))

def _test(X, Y, np_=1000):
    M, N = X.shape[0], Y.shape[0]
    K = _kern(X, Y)
    mmd = _mmd2(K, M, N)
    nulls = []
    for _ in range(np_):
        p = np.random.permutation(M + N)
        nulls.append(_mmd2(K[p][:, p], M, N))
    return int(mmd > np.quantile(nulls, 0.95))

def calc_power(yg, yr, nt=30, np_=1000):
    G = yg.cpu().numpy() if torch.is_tensor(yg) else yg
    R = yr.cpu().numpy() if torch.is_tensor(yr) else yr
    n_t = max(1, min(G.shape[0], R.shape[0]) // 10)
    M, N = max(2, G.shape[0] // n_t), max(2, R.shape[0] // n_t)
    pows = []
    for trial in range(nt):
        rej = [_test(G[t*M:(t+1)*M], R[t*N:(t+1)*N], np_) for t in range(n_t)]
        pows.append(np.mean(rej) * 100)
        if (trial + 1) % 10 == 0:
            print(f"    Trial {trial + 1}/{nt}, mean: {np.mean(pows):.1f}%")
    mu = float(np.mean(pows))
    ci = NormalDist.from_samples(pows).stdev * NormalDist().inv_cdf(0.975) / max(nt - 1, 1) ** 0.5
    return mu, ci


# ============================================================
# 8. PLOTTING — publication quality (no titles, for LaTeX captions)
# ============================================================

def plot_results(yr, yg, xg, ps, losses, name, out):
    yr = yr.cpu().numpy() if torch.is_tensor(yr) else yr
    yg = yg.cpu().numpy() if torch.is_tensor(yg) else yg
    xg = xg.cpu().numpy() if torch.is_tensor(xg) else xg
    x0 = xg[0] if xg.ndim > 1 else xg
    n_plot = min(100, len(yr), len(yg))
    nm = name.lower()

    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif', 'mathtext.fontset': 'cm',
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.direction': 'in', 'ytick.direction': 'in',
    })
    c_real = '#2166AC'
    c_gen = '#D6604D'

    yl = min(yr[:n_plot].min(), yg[:n_plot].min())
    yh = max(yr[:n_plot].max(), yg[:n_plot].max())
    pad = (yh - yl) * 0.06
    yl -= pad
    yh += pad

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for i in range(n_plot):
        ax.plot(x0, yr[i], color=c_real, alpha=0.25, lw=0.6)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_xlim(x0.min(), x0.max())
    ax.set_ylim(yl, yh)
    ax.tick_params(top=True, right=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(f'{out}/{nm}_real.pdf', bbox_inches='tight')
    plt.savefig(f'{out}/{nm}_real.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for i in range(n_plot):
        ax.plot(x0, yg[i], color=c_gen, alpha=0.25, lw=0.6)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_xlim(x0.min(), x0.max())
    ax.set_ylim(yl, yh)
    ax.tick_params(top=True, right=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(f'{out}/{nm}_gen.pdf', bbox_inches='tight')
    plt.savefig(f'{out}/{nm}_gen.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for i in range(n_plot):
        ax1.plot(x0, yr[i], color=c_real, alpha=0.25, lw=0.6)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.set_xlim(x0.min(), x0.max())
    ax1.set_ylim(yl, yh)
    ax1.tick_params(top=True, right=True)
    for i in range(n_plot):
        ax2.plot(x0, yg[i], color=c_gen, alpha=0.25, lw=0.6)
    ax2.set_xlabel('$x$')
    ax2.set_xlim(x0.min(), x0.max())
    ax2.tick_params(top=True, right=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(f'{out}/{nm}_compare.pdf', bbox_inches='tight')
    plt.savefig(f'{out}/{nm}_compare.png', dpi=300, bbox_inches='tight')
    plt.close()

    if losses:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        steps = np.arange(len(losses)) * 200
        ax.semilogy(steps, losses, color='#2c3e50', lw=0.8)
        ax.set_xlabel('Training step')
        ax.set_ylabel(r'$\Vert s_\theta + \varepsilon \Vert^2$')
        ax.tick_params(top=True, right=True)
        ax.grid(True, alpha=0.15, lw=0.4)
        plt.tight_layout(pad=0.4)
        plt.savefig(f'{out}/{nm}_loss.pdf', bbox_inches='tight')
        plt.savefig(f'{out}/{nm}_loss.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  Saved: {out}/{nm}_real.pdf, {nm}_gen.pdf, {nm}_compare.pdf, {nm}_loss.pdf")


# ============================================================
# 9. DATASET CONFIGS (retuned for Quadratic / Melbourne)
# ============================================================

CONFIGS = {
    'Quadratic': {
        'dim': 100,
        'kernel_type': 'se',
        'hyp_len': 0.22,
        'hyp_gain': 1.0,
        'whiten_floor': 0.75,
        'hidden': 320,
        'modes': 36,
        'n_layers': 6,
        'temb_dim': 256,
        'dropout': 0.00,
        'n_epochs': 3000,
        'batch_size': 128,
        'lr': 6e-4,
        'min_lr': 1e-5,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'n_sample': 1000,
        'n_steps': 800,
        'threshold': 10.0,
        'rescale': 50.0,
        'ema_decay': 0.9995,
        'snr_clip': 5.0,
        'use_malliavin': True,
        'malliavin_final_weight': 0.65,
        'self_condition': True,
        'corrector_steps': 1,
        'corrector_snr': 0.08,
    },
    'Melbourne': {
        'dim': 24,
        'kernel_type': 'periodic_se',
        'hyp_len': 0.18,
        'hyp_gain': 1.0,
        'periodic_len': 0.40,
        'periodic_mix': 0.80,
        'whiten_floor': 0.50,
        'hidden': 256,
        'modes': 12,
        'n_layers': 6,
        'temb_dim': 256,
        'dropout': 0.02,
        'n_epochs': 3000,
        'batch_size': 64,
        'lr': 6e-4,
        'min_lr': 5e-6,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'n_sample': 200,
        'n_steps': 800,
        'threshold': 10.0,
        'rescale': 1.0,
        'ema_decay': 0.9990,
        'snr_clip': 5.0,
        'use_malliavin': True,
        'malliavin_final_weight': 0.75,
        'self_condition': True,
        'corrector_steps': 1,
        'corrector_snr': 0.10,
    },
    'Gridwatch': {
        'dim': 288,
        'kernel_type': 'se',
        'hyp_len': 0.10,
        'hyp_gain': 1.0,
        'whiten_floor': 1.00,
        'hidden': 256,
        'modes': 20,
        'n_layers': 4,
        'temb_dim': 256,
        'dropout': 0.00,
        'n_epochs': 5000,
        'batch_size': 200,
        'lr': 1e-3,
        'min_lr': 2e-5,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'n_sample': 102,
        'n_steps': 1000,
        'threshold': 17.0,
        'rescale': 1.0,
        'ema_decay': 0.9995,
        'snr_clip': 5.0,
        'use_malliavin': False,
        'malliavin_final_weight': 0.0,
        'self_condition': True,
        'corrector_steps': 0,
        'corrector_snr': 0.0,
    },
}


def get_datasets(name):
    if name == 'Quadratic':
        return QuadraticDataset(1000, 100, seed=42), QuadraticDataset(1000, 100, seed=43)
    elif name == 'Melbourne':
        return MelbourneDataset('train', seed=42), MelbourneDataset('test', seed=42)
    elif name == 'Gridwatch':
        return GridwatchDataset('train', seed=87), GridwatchDataset('test', seed=87)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ============================================================
# 10. MAIN
# ============================================================

def run_dataset(name, device, out, n_epochs_override=None):
    cfg = {**CONFIGS[name], 'device': device}
    if n_epochs_override is not None:
        cfg['n_epochs'] = n_epochs_override

    ds, ds_test = get_datasets(name)
    x_grid = ds.x[0]
    coord_features = build_coord_features(name, x_grid)

    print("=" * 65)
    print("Score-Based Diffusion in Function Space (improved)")
    print(f"Dataset: {name} (dim={cfg['dim']})")
    print("=" * 65)
    print(f"FNO: hidden={cfg['hidden']}, layers={cfg['n_layers']}, modes={cfg['modes']}")
    print(f"Coords: {coord_features.shape[0]} channels | self-cond={cfg['self_condition']}")
    print(f"Noise kernel: {cfg['kernel_type']}, len={cfg['hyp_len']}, gain={cfg['hyp_gain']}")
    print(f"Schedule: cosine VP-SDE | Sampling: {cfg['n_steps']} steps | Heun: True")
    print(f"Training: {cfg['n_epochs']} epochs × batch {cfg['batch_size']} | lr={cfg['lr']}")
    print(f"Device: {device}\n")

    W = HilbertNoise(
        x_grid=x_grid,
        kernel_type=cfg['kernel_type'],
        hyp_len=cfg['hyp_len'],
        hyp_gain=cfg['hyp_gain'],
        periodic_len=cfg.get('periodic_len', 0.35),
        periodic_mix=cfg.get('periodic_mix', 0.7),
        whiten_floor=cfg.get('whiten_floor', 1.0),
    )
    sde = CosineVPSDE()
    model = FNO1D(
        cfg['dim'],
        coord_features=coord_features,
        hidden=cfg['hidden'],
        modes=cfg['modes'],
        n_layers=cfg['n_layers'],
        temb_dim=cfg['temb_dim'],
        self_condition=cfg['self_condition'],
        dropout=cfg.get('dropout', 0.0),
    )
    print(f"FNO params: {sum(p.numel() for p in model.parameters()):,}\n")

    print("Training...")
    print("-" * 50)
    t0 = time.time()
    losses, ema_model = train(model, ds, sde, W, cfg)
    print(f"Done in {time.time() - t0:.0f}s\n")

    n_s = min(cfg['n_sample'], len(ds_test))
    print(f"Sampling {n_s} functions (using EMA model)...")
    yg = sample_fn(
        ema_model, sde, W, n_s, cfg['dim'],
        steps=cfg['n_steps'],
        device=device,
        thr=cfg['threshold'],
        self_condition=cfg['self_condition'],
        heun=True,
        final_denoise=True,
        corrector_steps=cfg.get('corrector_steps', 0),
        corrector_snr=cfg.get('corrector_snr', 0.08),
    )

    _, yt = next(iter(DataLoader(ds_test, batch_size=n_s)))
    yt = yt.to(device)

    sc = cfg['rescale']
    print(f"  Gen: [{(yg * sc).min():.1f}, {(yg * sc).max():.1f}]")
    print(f"  Real: [{(yt * sc).min():.1f}, {(yt * sc).max():.1f}]\n")

    print("MMD power test (30 trials)...")
    mu, ci = calc_power(yg, yt)
    ps = f"{mu:.1f} ± {ci:.1f}"
    print(f"  Power: {ps}%\n")

    print("Plotting...")
    plot_results(yt * sc, yg * sc, ds.x, ps, losses, name, out)
    print(f"Done with {name}!\n")
    return mu, ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='Quadratic',
        choices=['Quadratic', 'Melbourne', 'Gridwatch', 'All'],
    )
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = './output'
    os.makedirs(out, exist_ok=True)
    os.makedirs('./data', exist_ok=True)

    if args.dataset == 'All':
        results = {}
        for name in ['Quadratic', 'Melbourne', 'Gridwatch']:
            try:
                mu, ci = run_dataset(name, device, out, n_epochs_override=args.n_epochs)
                results[name] = (mu, ci)
            except Exception as ex:
                print(f"ERROR on {name}: {ex}\n")
                results[name] = None

        print("\n" + "=" * 65)
        print("SUMMARY")
        print("=" * 65)
        for name, res in results.items():
            if res is not None:
                print(f"  {name:12s}  Power: {res[0]:.1f} ± {res[1]:.1f}%")
            else:
                print(f"  {name:12s}  FAILED")
    else:
        run_dataset(args.dataset, device, out, n_epochs_override=args.n_epochs)


if __name__ == '__main__':
    main()

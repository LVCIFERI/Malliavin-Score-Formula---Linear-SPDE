"""
Score-Based Diffusion in Function Space via Malliavin Calculus
==============================================================
Q-regularised Malliavin score Qρ^{μ_t} (Theorem 4) implemented as
ε-prediction: s_θ ≈ -ε, score recovered via Qρ = -ε/σ(t).
Reverse SPDE sampling via Euler-Maruyama (Corollary 5.1).

Datasets: Quadratic, Melbourne, Gridwatch (HDM benchmark, Lim et al. NeurIPS 2023)
Architecture: 1D Fourier Neural Operator (FNO)
Noise: GP(0, K_SE) — Hilbert-valued Wiener process with SE kernel
Evaluation: MMD kernel two-sample test power (30 trials, 95% CI)

Data download:
  Melbourne: http://www.timeseriesclassification.com/description.php?Dataset=MelbournePedestrian
  Gridwatch: https://www.gridwatch.templar.co.uk/download.php
  Place files as: ./data/MelbournePedestrian_TEST.arff, ./data/gridwatch_clean.csv
  (Or copy from hdm-official-master/data/)

Usage: python spde.py --dataset Quadratic|Melbourne|Gridwatch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, math, time, argparse, copy
from scipy.spatial import distance
from statistics import NormalDist
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. DATASETS (exactly as HDM, Lim et al. NeurIPS 2023)
# ============================================================

class QuadraticDataset(Dataset):
    """f(x;a,b) = ax² + b, a ∈ {-1,1}, b ~ N(0,1), x ∈ [-10,10], /50 normalised."""
    def __init__(self, num_data=1000, num_points=100, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.x = torch.linspace(-10., 10., num_points).unsqueeze(0).repeat(num_data, 1)
        a = torch.randint(0, 2, (num_data, 1)).repeat(1, num_points) * 2 - 1
        b = torch.randn(num_data, 1).repeat(1, num_points)
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
        np.random.seed(seed)
        n = len(y_full)
        idx = np.random.permutation(n)
        split = int(n * 0.8)
        train_idx, test_idx = idx[:split], idx[split:]
        if phase == 'train':
            y = y_full[train_idx]
        else:
            y = y_full[test_idx]
        # Standardise (fit on train) — NO extra scaling
        self.mean = y_full[train_idx].mean()
        self.std = y_full[train_idx].std()
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
        # Instance normalise (zero mean, unit var per sample)
        mu = y_full.mean(axis=1, keepdims=True)
        std = y_full.std(axis=1, keepdims=True) + 1e-8
        y_full = (y_full - mu) / std
        np.random.seed(seed)
        n = len(y_full)
        idx = np.random.permutation(n)
        split = int(n * 0.8)
        train_idx, test_idx = idx[:split], idx[split:]
        y = y_full[train_idx] if phase == 'train' else y_full[test_idx]
        self.x = torch.arange(num_points, dtype=torch.float32).unsqueeze(0).repeat(len(y), 1)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


# ============================================================
# 2. GP KERNEL NOISE (Hilbert-valued Wiener process)
# ============================================================

class HilbertNoise:
    """GP(0, K_SE) — covariance operator Q of the Hilbert-valued Wiener process.
    
    Novel contribution (from Theorem 4): the Malliavin covariance γ_t = Q(1-ᾱ)
    defines a Cameron-Martin norm. We use Tikhonov-regularised Q^{-1/2} to
    spectrally reweight the ε-prediction error:
      W = (Q + λI)^{-1/2},  λ = tr(Q)/dim (mean eigenvalue)
    This upweights fine detail in modes where Q has significant energy, while
    bounding amplification of noise-floor modes (null space of Q)."""
    def __init__(self, grid, hyp_len=0.8, hyp_gain=1.0):
        x = torch.linspace(-10, 10, grid).unsqueeze(-1)
        D = torch.cdist(x / hyp_len, x / hyp_len).pow(2)
        K = hyp_gain * torch.exp(-D)
        eig_val, eig_vec = np.linalg.eigh(K.numpy() + 1e-6 * np.eye(grid))
        eig_val = np.maximum(eig_val, 0.0)
        V = torch.from_numpy(eig_vec).float()
        lam = torch.from_numpy(eig_val).float()
        self.M = V @ torch.diag(torch.sqrt(lam))       # Q^{1/2} for sampling

        # Tikhonov-regularised Malliavin whitening: (Q + λI)^{-1/2}
        # λ = mean eigenvalue — bounds condition number to dim/1 ≈ effective rank
        lam_tikhonov = lam.mean()
        lam_reg = lam + lam_tikhonov
        lam_inv_sqrt = 1.0 / torch.sqrt(lam_reg)
        # Normalise so ||W||_op ≈ 1 (don't change loss scale)
        lam_inv_sqrt = lam_inv_sqrt / lam_inv_sqrt.max()
        self.W = V @ torch.diag(lam_inv_sqrt) @ V.T

        eff_rank = lam.sum()**2 / (lam**2).sum()
        cond = (lam_reg.max() / lam_reg.min()).item()
        print(f"  GP kernel: grid={grid}, eff_rank={eff_rank:.0f}, "
              f"λ_tikh={lam_tikhonov:.4f}, cond(W)={cond:.0f}")

    def sample(self, size):
        return torch.randn(list(size)) @ self.M.T

    def whiten(self, x):
        """Apply (Q+λI)^{-1/2}: Tikhonov-regularised Cameron-Martin projection."""
        return x @ self.W.to(x.device).T


# ============================================================
# 3. COSINE VP-SDE
# ============================================================

class CosineVPSDE:
    def __init__(self):
        self.s = 0.008; self.T = 0.9946; self.eps = 1e-5
        self.log_alpha_0 = math.log(math.cos(self.s / (1. + self.s) * math.pi / 2.))
    def marginal_log_mean_coeff(self, t):
        return torch.log(torch.clamp(torch.cos(
            (t + self.s) / (1. + self.s) * math.pi / 2.), 1e-8, 1.)) - self.log_alpha_0
    def diffusion_coeff(self, t): return torch.exp(self.marginal_log_mean_coeff(t))
    def marginal_std(self, t):
        return torch.sqrt(torch.clamp(1. - torch.exp(2 * self.marginal_log_mean_coeff(t)), min=1e-8))
    def beta(self, t):
        return math.pi / 2 * 2 / (self.s + 1) * torch.tan(
            (t + self.s) / (1 + self.s) * math.pi / 2)


# ============================================================
# 4. 1D FOURIER NEURAL OPERATOR
# ============================================================

class SpectralConv1d(nn.Module):
    """Spectral convolution: FFT → weight multiply → IFFT.
    No complex parameters, no indexing into complex tensors.
    modes is clamped to n_freq = dim//2+1 (rfft output size)."""
    def __init__(self, channels, modes, temb_dim, dim):
        super().__init__()
        n_freq = dim // 2 + 1
        self.n_modes = min(modes, n_freq)  # can't have more modes than freq bins
        scale = 1.0 / math.sqrt(channels)
        self.wr = nn.Parameter(scale * torch.randn(channels, self.n_modes))
        self.wi = nn.Parameter(scale * torch.randn(channels, self.n_modes))
        self.temb_proj = nn.Linear(temb_dim, channels)
        nn.init.zeros_(self.temb_proj.bias)

    def forward(self, x, temb):
        B, C, D = x.shape
        x = x + self.temb_proj(F.gelu(temb)).unsqueeze(-1)
        x_ft = torch.fft.rfft(x, norm='ortho')       # (B, C, D//2+1)
        n_freq = x_ft.shape[-1]

        # Pad weights if needed (when n_modes < n_freq)
        pad_size = n_freq - self.n_modes
        if pad_size > 0:
            wr_full = F.pad(self.wr, (0, pad_size))
            wi_full = F.pad(self.wi, (0, pad_size))
        else:
            wr_full = self.wr
            wi_full = self.wi

        x_r, x_i = x_ft.real, x_ft.imag
        o_r = x_r * wr_full.unsqueeze(0) - x_i * wi_full.unsqueeze(0)
        o_i = x_r * wi_full.unsqueeze(0) + x_i * wr_full.unsqueeze(0)
        return torch.fft.irfft(torch.complex(o_r, o_i), n=D, norm='ortho')

class FourierLayer(nn.Module):
    """Fourier layer: preact → spectral conv + soft-gating → MLP + skip."""
    def __init__(self, channels, modes, temb_dim, dim, is_last=False, mlp_expansion=4.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.spec_conv = SpectralConv1d(channels, modes, temb_dim, dim)
        self.skip_w = nn.Conv1d(channels, channels, 1)
        self.skip_gate = nn.Conv1d(channels, channels, 1)
        mlp_hidden = int(channels * mlp_expansion)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, mlp_hidden, 1), nn.GELU(),
            nn.Conv1d(mlp_hidden, channels, 1))
        self.mlp_skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x, temb):
        h = F.gelu(self.norm1(x))
        x = self.spec_conv(h, temb) + self.skip_w(h) * torch.sigmoid(self.skip_gate(h))
        h2 = F.gelu(x)
        x = self.mlp(h2) + self.mlp_skip(h2)
        return x

class FNO1D(nn.Module):
    """s_θ(t, u_t) ≈ -ε.  Score: Qρ = s_θ/σ(t) [Thm 4 reparameterised]."""
    def __init__(self, dim, hidden=256, modes=50, n_layers=4, temb_dim=256):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.lifting = nn.Conv1d(1, hidden, 1)
        self.layers = nn.ModuleList([
            FourierLayer(hidden, modes, temb_dim, dim) for _ in range(n_layers)])
        self.projection = nn.Sequential(nn.Conv1d(hidden, hidden, 1), nn.GELU(), nn.Conv1d(hidden, 1, 1))
    def get_temb(self, t, dim):
        half = dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        a = (1000 * t).float().unsqueeze(-1) * f.unsqueeze(0)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
    def forward(self, x, t):
        temb = self.time_mlp(self.get_temb(t, self.temb_dim))
        h = self.lifting(x.unsqueeze(1)) + temb.unsqueeze(-1)
        for layer in self.layers: h = layer(h, temb)
        return self.projection(h).squeeze(1)


# ============================================================
# 5. EMA + TRAINING
# ============================================================

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters(): p.requires_grad_(False)
    def update(self, model):
        for sp, mp in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)
    def get(self): return self.shadow

def train(model, dataset, sde, W, cfg):
    """ε-prediction with optional Malliavin-weighted loss (Theorem 4).
    
    Standard DSM: L = E[||s_θ + ε||²]
    Malliavin DSM: L_M = E[||(Q+λI)^{-1/2}(s_θ + ε)||²]  (novel, Theorem 4)
    
    Malliavin loss is used when eff_rank(Q)/dim is high enough (>10%) so
    the whitening is well-conditioned. For low-rank Q (Gridwatch: 4%),
    whitening amplifies null-space noise → use standard L² only."""
    device = cfg['device']
    use_mal = cfg.get('use_malliavin', True)
    model = model.to(device); model.train()
    ema = EMA(model, decay=0.9999)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], amsgrad=True)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    total_steps = cfg['n_epochs'] * max(1, len(dataset) // cfg['batch_size'])
    decay_step = int(total_steps * 0.6)
    losses = []; step = 0; lr_decayed = False
    if use_mal:
        print(f"  Using Malliavin-weighted loss (Cameron-Martin norm, Theorem 4)")
    else:
        print(f"  Using standard L² loss (Q too low-rank for Malliavin weighting)")
    for epoch in range(cfg['n_epochs']):
        for _, y in loader:
            if step == decay_step and not lr_decayed:
                for g in opt.param_groups: g['lr'] = cfg['lr'] / 10
                print(f"  LR decay at step {step}: {cfg['lr']:.1e} → {cfg['lr']/10:.1e}")
                lr_decayed = True
            y = y.to(device); B = y.shape[0]
            t = torch.rand(B, device=device) * (sde.T - sde.eps) + sde.eps
            e = W.sample(y.shape).to(device)
            u_t = y * sde.diffusion_coeff(t)[:, None] + e * sde.marginal_std(t)[:, None]
            error = model(u_t, t) - (-e)

            loss_std = error.pow(2).sum(1).mean()

            if use_mal:
                error_whitened = W.whiten(error)
                loss_mal = error_whitened.pow(2).sum(1).mean()
                alpha = max(0.0, 1.0 - step / (total_steps * 0.3))
                loss = alpha * loss_std + (1 - alpha) * loss_mal
            else:
                loss = loss_std
                loss_mal = loss_std  # for logging

            if step == 0:
                if torch.isnan(loss):
                    print(f"  NaN at step 0!"); return losses, ema.get()
                print(f"  Step 0 OK, L_std={loss_std.item():.2f}" +
                      (f", L_mal={loss_mal.item():.2f}" if use_mal else ""))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step(); ema.update(model); step += 1
            if step % 200 == 0: losses.append(loss_std.item())
            if step % 2000 == 0:
                msg = f"  Step {step}, Loss={loss_std.item():.4f}"
                if use_mal: msg += f", L_mal={loss_mal.item():.4f}, α={alpha:.2f}"
                print(msg)
    return losses, ema.get()


# ============================================================
# 6. SAMPLING (Euler-Maruyama reverse SPDE, Corollary 5.1)
# ============================================================

@torch.no_grad()
def sample_fn(model, sde, W, n, dim, steps=1000, device='cpu', thr=10.0):
    """Reverse SPDE: dû = [β/2 û + β Qρ] dt + √β Q^{1/2} dW̃"""
    model.eval()
    ts = torch.linspace(sde.T, sde.eps, steps + 1).to(device)
    u = W.sample((n, dim)).to(device) * sde.marginal_std(torch.tensor(sde.T, device=device))
    for i in range(steps):
        s = ts[i]; vec_s = torch.full((n,), s.item(), device=device)
        score = model(u, vec_s) * torch.pow(sde.marginal_std(vec_s), -1)[:, None]
        beta_dt = sde.beta(vec_s) * (s - ts[i+1])
        eta = W.sample(u.shape).to(device)
        u = (1 + beta_dt/2)[:, None] * u + beta_dt[:, None] * score + torch.sqrt(beta_dt)[:, None] * eta
        norms = u.norm(dim=1); mask = norms > thr
        if mask.any(): u[mask] = u[mask] / norms[mask, None] * thr
    return u


# ============================================================
# 7. MMD POWER TEST (exact HDM protocol)
# ============================================================

def _kern(X, Y, g=-1):
    XY = np.vstack((X, Y))
    D = distance.cdist(XY, XY, 'euclidean') / np.sqrt(X.shape[1])
    if g == -1: g = np.median(D[D > 0])
    return np.exp(-0.5 / g**2 * D**2)

def _mmd2(K, M, N):
    return ((K[:M,:M].sum()-np.trace(K[:M,:M]))/(M*(M-1))
            - 2*K[:M,M:].sum()/(M*N)
            + (K[M:,M:].sum()-np.trace(K[M:,M:]))/(N*(N-1)))

def _test(X, Y, np_=1000):
    M, N = X.shape[0], Y.shape[0]
    K = _kern(X, Y); mmd = _mmd2(K, M, N)
    nulls = []
    for _ in range(np_):
        p = np.random.permutation(M+N)
        nulls.append(_mmd2(K[p][:,p], M, N))
    return int(mmd > np.quantile(nulls, 0.95))

def calc_power(yg, yr, nt=30, np_=1000):
    G = yg.cpu().numpy() if torch.is_tensor(yg) else yg
    R = yr.cpu().numpy() if torch.is_tensor(yr) else yr
    n_t = min(G.shape[0], R.shape[0]) // 10
    M, N = G.shape[0]//n_t, R.shape[0]//n_t
    pows = []
    for trial in range(nt):
        rej = [_test(G[t*M:(t+1)*M], R[t*N:(t+1)*N], np_) for t in range(n_t)]
        pows.append(np.mean(rej)*100)
        if (trial+1) % 10 == 0: print(f"    Trial {trial+1}/{nt}, mean: {np.mean(pows):.1f}%")
    mu = np.mean(pows)
    ci = NormalDist.from_samples(pows).stdev * NormalDist().inv_cdf(0.975) / (nt-1)**0.5
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

    # --- Style ---
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif', 'mathtext.fontset': 'cm',
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.direction': 'in', 'ytick.direction': 'in',
    })
    c_real = '#2166AC'   # steel blue
    c_gen  = '#D6604D'   # muted red-orange

    yl = min(yr[:n_plot].min(), yg[:n_plot].min())
    yh = max(yr[:n_plot].max(), yg[:n_plot].max())
    pad = (yh - yl) * 0.06
    yl -= pad; yh += pad

    # --- (a) Real data only ---
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for i in range(n_plot):
        ax.plot(x0, yr[i], color=c_real, alpha=0.25, lw=0.6)
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
    ax.set_xlim(x0.min(), x0.max()); ax.set_ylim(yl, yh)
    ax.tick_params(top=True, right=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(f'{out}/{nm}_real.pdf', bbox_inches='tight')
    plt.savefig(f'{out}/{nm}_real.png', dpi=300, bbox_inches='tight'); plt.close()

    # --- (b) Generated only ---
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for i in range(n_plot):
        ax.plot(x0, yg[i], color=c_gen, alpha=0.25, lw=0.6)
    ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
    ax.set_xlim(x0.min(), x0.max()); ax.set_ylim(yl, yh)
    ax.tick_params(top=True, right=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(f'{out}/{nm}_gen.pdf', bbox_inches='tight')
    plt.savefig(f'{out}/{nm}_gen.png', dpi=300, bbox_inches='tight'); plt.close()

    # --- (c) Side-by-side (for quick comparison) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for i in range(n_plot):
        ax1.plot(x0, yr[i], color=c_real, alpha=0.25, lw=0.6)
    ax1.set_xlabel('$x$'); ax1.set_ylabel('$f(x)$')
    ax1.set_xlim(x0.min(), x0.max()); ax1.set_ylim(yl, yh)
    ax1.tick_params(top=True, right=True)
    for i in range(n_plot):
        ax2.plot(x0, yg[i], color=c_gen, alpha=0.25, lw=0.6)
    ax2.set_xlabel('$x$')
    ax2.set_xlim(x0.min(), x0.max())
    ax2.tick_params(top=True, right=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(f'{out}/{nm}_compare.pdf', bbox_inches='tight')
    plt.savefig(f'{out}/{nm}_compare.png', dpi=300, bbox_inches='tight'); plt.close()

    # --- (d) Training loss ---
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
        plt.savefig(f'{out}/{nm}_loss.png', dpi=300, bbox_inches='tight'); plt.close()

    print(f"  Saved: {out}/{nm}_real.pdf, {nm}_gen.pdf, {nm}_compare.pdf, {nm}_loss.pdf")


# ============================================================
# 9. DATASET CONFIGS (matching HDM exactly)
# ============================================================

CONFIGS = {
    'Quadratic': {
        'dim': 100, 'hyp_len': 0.8, 'hyp_gain': 1.0,
        'hidden': 256, 'modes': 50, 'n_layers': 4, 'temb_dim': 256,
        'n_epochs': 4000, 'batch_size': 100, 'lr': 1e-3, 'grad_clip': 1.0,
        'n_sample': 1000, 'n_steps': 1000, 'threshold': 10.0, 'rescale': 50.0,
        'use_malliavin': True,   # eff_rank/dim ≈ 30/100 = 30% — well-conditioned
    },
    'Melbourne': {
        'dim': 24, 'hyp_len': 2.0, 'hyp_gain': 1.0,
        'hidden': 256, 'modes': 12, 'n_layers': 4, 'temb_dim': 256,
        'n_epochs': 2000, 'batch_size': 100, 'lr': 1e-3, 'grad_clip': 1.0,
        'n_sample': 200, 'n_steps': 1000, 'threshold': 10.0, 'rescale': 1.0,
        'use_malliavin': True,   # eff_rank/dim ≈ 9/24 = 38% — well-conditioned
    },
    'Gridwatch': {
        'dim': 288, 'hyp_len': 1.8, 'hyp_gain': 1.0,
        'hidden': 256, 'modes': 20, 'n_layers': 4, 'temb_dim': 256,
        'n_epochs': 5000, 'batch_size': 200, 'lr': 1e-3, 'grad_clip': 1.0,
        'n_sample': 102, 'n_steps': 1000, 'threshold': 17.0, 'rescale': 1.0,
        'use_malliavin': False,
        # modes=20 constrains FNO to low-freq output matching GP eff_rank≈9.
        # With modes=144, the model outputs errors in 135 null-space modes
        # that the reverse SDE can't correct → noise accumulates over 1000 steps.
    },
}


def get_datasets(name):
    if name == 'Quadratic':
        return QuadraticDataset(1000, 100, seed=42), QuadraticDataset(1000, 100, seed=43)
    elif name == 'Melbourne':
        return MelbourneDataset('train'), MelbourneDataset('test')
    elif name == 'Gridwatch':
        return GridwatchDataset('train'), GridwatchDataset('test')
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ============================================================
# 10. MAIN
# ============================================================

def run_dataset(name, device, out, n_epochs_override=None):
    """Run full pipeline for one dataset."""
    cfg = {**CONFIGS[name], 'device': device}
    if n_epochs_override is not None:
        cfg['n_epochs'] = n_epochs_override

    ds, ds_test = get_datasets(name)

    print("=" * 65)
    print(f"Score-Based Diffusion in Function Space (Malliavin, Thm 4)")
    print(f"Dataset: {name} (dim={cfg['dim']})")
    print("=" * 65)
    print(f"Score: Qρ = -ε/σ(t) [Thm 4, ε-prediction]")
    print(f"FNO: hidden={cfg['hidden']}, layers={cfg['n_layers']}, modes={cfg['modes']}")
    print(f"GP noise: SE kernel, len={cfg['hyp_len']}, gain={cfg['hyp_gain']}")
    print(f"Schedule: cosine VP-SDE, Sampling: {cfg['n_steps']} EM steps")
    steps_per_epoch = len(ds) // cfg['batch_size']
    total_steps = cfg['n_epochs'] * steps_per_epoch
    print(f"Training: {cfg['n_epochs']} epochs × {steps_per_epoch} batches = {total_steps} steps, lr={cfg['lr']}")
    print(f"Device: {device}\n")

    W = HilbertNoise(cfg['dim'], cfg['hyp_len'], cfg['hyp_gain'])
    sde = CosineVPSDE()
    model = FNO1D(cfg['dim'], cfg['hidden'], cfg['modes'], cfg['n_layers'], cfg['temb_dim'])
    print(f"FNO params: {sum(p.numel() for p in model.parameters()):,}\n")

    print("Training..."); print("-" * 50)
    t0 = time.time()
    losses, ema_model = train(model, ds, sde, W, cfg)
    print(f"Done in {time.time()-t0:.0f}s\n")

    n_s = min(cfg['n_sample'], len(ds_test))
    print(f"Sampling {n_s} functions (using EMA model)...")
    yg = sample_fn(ema_model, sde, W, n_s, cfg['dim'], cfg['n_steps'], device, cfg['threshold'])

    _, yt = next(iter(DataLoader(ds_test, batch_size=n_s)))
    yt = yt.to(device)

    sc = cfg['rescale']
    print(f"  Gen: [{(yg*sc).min():.1f}, {(yg*sc).max():.1f}]")
    print(f"  Real: [{(yt*sc).min():.1f}, {(yt*sc).max():.1f}]\n")

    print("MMD power test (30 trials)...")
    mu, ci = calc_power(yg, yt)
    ps = f"{mu:.1f} \u00b1 {ci:.1f}"
    print(f"  Power: {ps}%\n")

    print("Plotting...")
    plot_results(yt * sc, yg * sc, ds.x, ps, losses, name, out)
    print(f"Done with {name}!\n")
    return mu, ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Quadratic',
                        choices=['Quadratic', 'Melbourne', 'Gridwatch', 'All'])
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Override number of training epochs')
    args = parser.parse_args()

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
            if res:
                print(f"  {name:12s}  Power: {res[0]:.1f} \u00b1 {res[1]:.1f}%")
            else:
                print(f"  {name:12s}  FAILED")
    else:
        run_dataset(args.dataset, device, out, n_epochs_override=args.n_epochs)


if __name__ == '__main__':
    main()

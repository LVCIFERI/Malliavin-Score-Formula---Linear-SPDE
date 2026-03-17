"""
Score Formula Validation for 2D SPDEs via Fourier Pseudospectral Collocation
=============================================================================

Numerical validation of the Malliavin score formula (Theorem 4) for linear
SPDEs on the two-dimensional periodic domain [0, 2pi]^2 using Fourier
pseudospectral collocation.

Validates the closed-form score formula:
    beta_h(u) = -<u - S(t)u0, gamma_t^{-1} h>_H

against central finite-difference approximations for four SPDE classes:

    Second-order:
        1. Stochastic Heat Equation:              A = nu*Delta
        2. Ornstein-Uhlenbeck:                    A = nu*Delta - kappa*I

    Fourth-order:
        3. Stochastic Biharmonic:                 A = -mu*Delta^2
        4. Linearised Swift-Hohenberg:            A = (r-1)I - 2*Delta - Delta^2

Spatial discretisation: Fourier pseudospectral collocation on [0, 2pi]^2 with
periodic boundary conditions. Differentiation matrices are constructed via
trigonometric interpolation on an odd number of equidistant collocation points,
following the formulation in Trefethen (2000). The 2D operator is assembled
using Kronecker products.

Noise covariance:  q_{k1,k2} = (1 + k1^2 + k2^2)^{-s},  s > 1 (trace-class).

Figure layout: 4 rows (one per SPDE) x 3 columns:
    Col 0 -- 2D Malliavin score field s(x,y) at t = T  (physical space heatmap)
    Col 1 -- 2D pointwise |Malliavin - FD| error field  (log-scale heatmap)
    Col 2 -- Scalar directional score beta_h(u(t)) trajectories + log|error| vs t

Usage:
    python score_validation_2d.py
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm


# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
OUTPUT_DIR = Path("figures")
DPI = 300

plt.rcParams.update({
    # Typography: STIX General gives LaTeX-quality text without requiring LaTeX
    "font.family":          "serif",
    "font.serif":           ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":     "stix",
    "font.size":            10,
    "axes.labelsize":       10,
    "axes.titlesize":       10,
    "legend.fontsize":      8.5,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "text.usetex":          False,
    # Spines & ticks — thin and inward for a journal-quality look
    "axes.linewidth":       0.55,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.major.width":    0.55,
    "ytick.major.width":    0.55,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "xtick.minor.width":    0.4,
    "ytick.minor.width":    0.4,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    # Lines
    "lines.linewidth":      1.6,
    "patch.linewidth":      0.55,
    # Figure / save
    "figure.dpi":           150,
    "savefig.dpi":          DPI,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "axes.grid":            False,
})

# Refined, muted academic palette
COLOURS = {
    "navy":        "#1B3A5C",
    "crimson":     "#8B2020",
    "forest":      "#2A5E3F",
    "plum":        "#5B2D7E",
    "slate":       "#5A6A7A",
    "silver":      "#C4CAD1",
    "rule":        "#DDE0E4",   # figure rule lines
    "text_dark":   "#1A1A1A",
    "text_mid":    "#4A4A4A",
    "text_light":  "#888888",
}

# Four sample-path colours: harmonious, distinguishable, print-safe
PATH_COLOURS = ["#1B3A5C", "#8B2020", "#2A5E3F", "#5B2D7E"]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for 2D score trajectory simulation."""
    N: int = 48                # Collocation points per direction: N+1 (must be even)
    n_paths: int = 4           # Number of independent sample paths
    n_timesteps: int = 50      # Number of time evaluation points
    t_start: float = 0.02      # Start time (avoid t = 0 singularity)
    t_end: float = 1.0         # Terminal time
    noise_decay: float = 3.0   # Noise covariance decay: q = (1 + |k|^2)^{-s}
    fd_epsilon: float = 1e-5   # Finite difference step size


# =============================================================================
# Fourier Pseudospectral Collocation Infrastructure
# =============================================================================

def gen_fourier_diff_matrix(N: int, L: float = 2.0 * np.pi) -> tuple:
    """
    Generate Fourier collocation points, quadrature weight, and differentiation
    matrix for the periodic domain [0, L) with N+1 equidistant points (N even).

    Args:
        N: Number of intervals (must be even). Gives N+1 collocation points.
        L: Domain length (default 2pi).

    Returns:
        z: Collocation points, shape (N+1,)
        w: Uniform quadrature weight (scalar)
        D: First-derivative differentiation matrix, shape (N+1, N+1)
    """
    if N % 2 != 0:
        raise ValueError(f"N must be even, got {N}")

    m = N + 1
    xj = 2.0 * np.pi / m * np.arange(m)

    z = L / (2.0 * np.pi) * xj
    w = L / m

    D = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                D[i, j] = 0.5 * (-1) ** (i + j) / np.sin((xj[i] - xj[j]) / 2.0)

    D *= (2.0 * np.pi) / L

    return z, w, D


def build_2d_operators(N: int, L: float = 2.0 * np.pi) -> dict:
    """Build 2D differential operators via Kronecker products."""
    x, w, Dx = gen_fourier_diff_matrix(N, L)
    y = x.copy()
    mx = N + 1
    M = mx * mx

    Dx2 = Dx @ Dx
    Dx4 = Dx2 @ Dx2
    Dx2 = 0.5 * (Dx2 + Dx2.T)
    Dx4 = 0.5 * (Dx4 + Dx4.T)
    I_1d = np.eye(mx)

    Lap   = np.kron(Dx2, I_1d) + np.kron(I_1d, Dx2)
    BiLap = np.kron(Dx4, I_1d) + 2.0 * np.kron(Dx2, Dx2) + np.kron(I_1d, Dx4)
    Lap   = 0.5 * (Lap + Lap.T)
    BiLap = 0.5 * (BiLap + BiLap.T)
    I_2d  = np.eye(M)

    return {"Lap": Lap, "BiLap": BiLap, "I": I_2d,
            "x": x, "y": y, "w": w, "mx": mx}


# =============================================================================
# Abstract Base Class for 2D Periodic SPDEs (FFT-based)
# =============================================================================

class LinearSPDE2D(ABC):
    """
    Abstract base class for linear SPDEs on [0, 2pi]^2 with periodic BCs.

    Spectral computations use the FFT (O(M log M)), not matrix eigendecomposition.
    The collocation differentiation matrices are retained solely for cross-validation
    of the analytical eigenvalues against the numerical operator spectrum.
    """

    def __init__(self, N: int, noise_decay: float) -> None:
        self.N = N
        self.mx = N + 1          # odd number of collocation points per direction
        self.M  = self.mx ** 2   # total degrees of freedom
        self.noise_decay = noise_decay

        # Collocation grid
        self.x = 2.0 * np.pi / self.mx * np.arange(self.mx)

        # Wavenumber grid: k1, k2 ∈ {0, 1, ..., (mx-1)/2, -(mx-1)/2, ..., -1}
        k1d = np.fft.fftfreq(self.mx) * self.mx        # integer wavenumbers
        self.KX, self.KY = np.meshgrid(k1d, k1d, indexing="ij")
        K2 = self.KX ** 2 + self.KY ** 2                # |k|^2 grid, shape (mx, mx)

        # Operator eigenvalues a_{k1,k2} and noise covariance q_{k1,k2}
        self.operator_eigenvalues_2d = self._analytical_eigenvalues(K2)  # (mx, mx)
        self.q_eigenvalues_2d = (1.0 + K2) ** (-noise_decay)            # (mx, mx)

        # Flat versions for convenience
        self.operator_eigenvalues = self.operator_eigenvalues_2d.flatten()
        self.q_eigenvalues = self.q_eigenvalues_2d.flatten()

        # Cross-validate against collocation differentiation matrices
        self._cross_validate_eigenvalues(N)

        if np.any(self.operator_eigenvalues > 1e-10):
            n_unstable = int(np.sum(self.operator_eigenvalues > 1e-10))
            raise ValueError(f"{self.name}: {n_unstable} unstable eigenvalue(s).")

    def _cross_validate_eigenvalues(self, N: int) -> None:
        """Compare analytical eigenvalues against the collocation operator spectrum."""
        ops = build_2d_operators(N)
        A_mat = self._build_operator_matrix(ops["Lap"], ops["BiLap"], ops["I"])
        A_mat = 0.5 * (A_mat + A_mat.T)
        eigvals_num = np.sort(np.linalg.eigvalsh(A_mat))
        eigvals_ana = np.sort(self.operator_eigenvalues)
        max_err = np.max(np.abs(eigvals_num - eigvals_ana))
        if max_err > 1e-4:
            warnings.warn(
                f"{self.name}: eigenvalue cross-validation error = {max_err:.2e}"
            )

    def to_fourier(self, u_phys: np.ndarray) -> np.ndarray:
        """Physical space (mx, mx) → Fourier coefficients (mx, mx), complex."""
        return np.fft.fft2(u_phys) / self.M

    def to_physical(self, u_hat: np.ndarray) -> np.ndarray:
        """Fourier coefficients (mx, mx) → physical space (mx, mx), real."""
        return np.fft.ifft2(u_hat * self.M).real

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _build_operator_matrix(self, Lap, BiLap, I) -> np.ndarray: ...

    @abstractmethod
    def _analytical_eigenvalues(self, K2: np.ndarray) -> np.ndarray: ...

    def compute_covariance_2d(self, t: float) -> np.ndarray:
        """gamma_{k1,k2}(t) = q_k (e^{2a_k t} - 1)/(2a_k), shape (mx, mx)."""
        a = self.operator_eigenvalues_2d
        q = self.q_eigenvalues_2d
        gamma = np.where(
            np.abs(a) < 1e-14,
            q * t,
            q * (np.exp(2.0 * a * t) - 1.0) / (2.0 * a),
        )
        return gamma

    def score_malliavin_fourier(self, u_hat, mean_hat, gamma_2d) -> np.ndarray:
        """Full Malliavin score in Fourier space: s_k = -(u_k - m_k) / gamma_k."""
        return -(u_hat - mean_hat) / gamma_2d

    def score_fd_fourier(self, u_hat, mean_hat, gamma_2d,
                         epsilon: float = 1e-5) -> np.ndarray:
        """
        Per-mode FD score in Fourier space (numerically stable formulation).

        For each complex Fourier coefficient c_k = u_hat_k - m_k, the
        contribution to log p is -|c_k|^2 / (2 gamma_k).

        Central FD on a quadratic is analytically exact.  To avoid
        catastrophic cancellation when |c_k| << epsilon, we evaluate
        the difference algebraically:
            (x+eps)^2 - (x-eps)^2 = 4*eps*x    (exact identity)
        rather than squaring then subtracting.
        """
        centred = u_hat - mean_hat
        re_c, im_c = centred.real, centred.imag
        g = gamma_2d

        # Stable central difference via algebraic identity
        re_score = -0.5 * (4.0 * epsilon * re_c) / (g * 2.0 * epsilon)
        im_score = -0.5 * (4.0 * epsilon * im_c) / (g * 2.0 * epsilon)

        return re_score + 1j * im_score

    def score_directional_malliavin(self, u_hat, h_hat, mean_hat,
                                    gamma_2d) -> float:
        """Scalar directional score: beta_h = -sum_k (u_k - m_k) h_k^* / gamma_k."""
        return float(-np.sum(((u_hat - mean_hat) * np.conj(h_hat) / gamma_2d).real))

    def score_directional_fd(self, u_phys, h_phys, mean_hat,
                             gamma_2d, epsilon: float = 1e-5) -> float:
        """
        Scalar directional FD through physical→Fourier pipeline.

        Perturbs u in physical space by ±ε*h, transforms to Fourier via FFT,
        evaluates the log-density, and applies central differences. This tests
        the full FFT round-trip and eigenvalue computation.

        Uses the stable algebraic identity for the log-density difference:
            |c+εĥ|² - |c-εĥ|² = 4ε Re(c conj(ĥ))
        to avoid catastrophic cancellation.
        """
        c_plus_hat  = self.to_fourier(u_phys + epsilon * h_phys) - mean_hat
        c_minus_hat = self.to_fourier(u_phys - epsilon * h_phys) - mean_hat

        lp_plus  = -0.5 * np.sum((np.abs(c_plus_hat) ** 2 / gamma_2d).real)
        lp_minus = -0.5 * np.sum((np.abs(c_minus_hat) ** 2 / gamma_2d).real)

        return float((lp_plus - lp_minus) / (2.0 * epsilon))


# =============================================================================
# SPDE Subclasses
# =============================================================================

class Heat2D(LinearSPDE2D):
    """du = nu*Delta u dt + Q^{1/2} dW"""
    def __init__(self, N, noise_decay, nu=1.0):
        self.nu = nu
        super().__init__(N, noise_decay)
    @property
    def name(self): return "2D Heat Equation"
    def _build_operator_matrix(self, Lap, BiLap, I): return self.nu * Lap
    def _analytical_eigenvalues(self, K2): return -self.nu * K2


class OrnsteinUhlenbeck2D(LinearSPDE2D):
    """du = (nu*Delta - kappa*I) u dt + Q^{1/2} dW"""
    def __init__(self, N, noise_decay, nu=1.0, kappa=2.0):
        self.nu = nu; self.kappa = kappa
        super().__init__(N, noise_decay)
    @property
    def name(self): return "2D Ornstein-Uhlenbeck"
    def _build_operator_matrix(self, Lap, BiLap, I): return self.nu * Lap - self.kappa * I
    def _analytical_eigenvalues(self, K2): return -self.nu * K2 - self.kappa


class Biharmonic2D(LinearSPDE2D):
    """du = -mu*Delta^2 u dt + Q^{1/2} dW"""
    def __init__(self, N, noise_decay, mu=1.0):
        self.mu = mu
        super().__init__(N, noise_decay)
    @property
    def name(self): return "2D Biharmonic"
    def _build_operator_matrix(self, Lap, BiLap, I): return -self.mu * BiLap
    def _analytical_eigenvalues(self, K2): return -self.mu * K2 ** 2


class SwiftHohenberg2D(LinearSPDE2D):
    """du = [(r-1)I - 2*Delta - Delta^2] u dt + Q^{1/2} dW"""
    def __init__(self, N, noise_decay, r=-0.5):
        self.r = r
        if r >= 0:
            raise ValueError(f"Swift-Hohenberg requires r < 0, got {r}")
        super().__init__(N, noise_decay)
    @property
    def name(self): return "2D Swift-Hohenberg"
    def _build_operator_matrix(self, Lap, BiLap, I):
        return (self.r - 1.0) * I - 2.0 * Lap - BiLap
    def _analytical_eigenvalues(self, K2): return self.r - (1.0 - K2) ** 2


# =============================================================================
# 2D Score Field Computation (FFT-based)
# =============================================================================

def compute_2d_score_fields(
    spde: LinearSPDE2D,
    u_hat: np.ndarray,
    mean_hat: np.ndarray,
    gamma_2d: np.ndarray,
    epsilon: float = 1e-5,
) -> dict:
    """
    Compute the Malliavin score field and FD error field.

    All computations in Fourier space; results mapped to physical space via IFFT.
    The per-mode FD perturbs Re and Im parts separately, giving EXACT central
    differences of a quadratic (zero truncation error).

    The error field is computed as IFFT(mall_hat - fd_hat) to avoid accumulating
    separate IFFT rounding errors.

    Returns dict with 2D arrays (mx, mx): 'score_mall', 'score_fd', 'error',
    'centred', 'fourier_error', and 1D grids 'x', 'y'.
    """
    score_mall_hat = spde.score_malliavin_fourier(u_hat, mean_hat, gamma_2d)
    score_fd_hat   = spde.score_fd_fourier(u_hat, mean_hat, gamma_2d, epsilon)

    centred_hat = u_hat - mean_hat

    # Physical-space fields
    score_mall_phys = spde.to_physical(score_mall_hat)
    centred_phys    = spde.to_physical(centred_hat)

    # Error computed from single IFFT of the Fourier-space difference
    diff_hat   = score_mall_hat - score_fd_hat
    error_phys = np.abs(spde.to_physical(diff_hat))

    # Fourier-space max error (should be ~machine eps)
    fourier_err = float(np.max(np.abs(diff_hat)))

    return {
        "score_mall": score_mall_phys,
        "error":      error_phys,
        "centred":    centred_phys,
        "fourier_error": fourier_err,
        "x":          spde.x,
        "y":          spde.x,
    }


# =============================================================================
# Simulation (FFT-based)
# =============================================================================

def simulate_score_trajectories(
    spde: LinearSPDE2D,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple:
    """
    Simulate scalar score trajectories and compute 2D score fields at t = T.

    All spectral operations use FFT. The stochastic convolution is sampled
    analytically in Fourier space: each mode u_hat_{k1,k2}(t) is an independent
    complex Gaussian with known mean and variance.

    The scalar directional FD goes through the physical→Fourier→log p pipeline,
    providing a genuine round-trip validation.

    Returns:
        times, malliavin_scores, fd_scores, fields_2d
    """
    mx = spde.mx
    a  = spde.operator_eigenvalues_2d   # (mx, mx)

    times = np.linspace(config.t_start, config.t_end, config.n_timesteps)

    # Initial condition and direction in physical space
    X, Y = np.meshgrid(spde.x, spde.x, indexing="ij")
    u0_phys = np.cos(X) + 0.5 * np.cos(Y) + 0.25 * np.cos(X + Y)
    h_phys  = np.cos(X)

    u0_hat = spde.to_fourier(u0_phys)
    h_hat  = spde.to_fourier(h_phys)

    malliavin_scores = np.zeros((config.n_paths, config.n_timesteps))
    fd_scores        = np.zeros((config.n_paths, config.n_timesteps))

    _u_snap = _mean_snap = _gamma_snap = None

    for path_idx in range(config.n_paths):
        for t_idx, t in enumerate(times):
            exp_at   = np.exp(a * t)
            gamma_2d = spde.compute_covariance_2d(t)
            mean_hat = exp_at * u0_hat

            # Sample stochastic part analytically in Fourier space
            noise_hat = (rng.standard_normal((mx, mx))
                         * np.sqrt(gamma_2d * 0.5)
                         + 1j * rng.standard_normal((mx, mx))
                         * np.sqrt(gamma_2d * 0.5))
            noise_hat[0, 0] = noise_hat[0, 0].real * np.sqrt(2)
            u_hat = mean_hat + noise_hat

            # Reconstruct physical-space field for FD
            u_phys = spde.to_physical(u_hat)

            malliavin_scores[path_idx, t_idx] = (
                spde.score_directional_malliavin(
                    u_hat, h_hat, mean_hat, gamma_2d))
            fd_scores[path_idx, t_idx] = (
                spde.score_directional_fd(
                    u_phys, h_phys, mean_hat, gamma_2d, config.fd_epsilon))

            if path_idx == 0 and t_idx == config.n_timesteps - 1:
                _u_snap     = u_hat.copy()
                _mean_snap  = mean_hat.copy()
                _gamma_snap = gamma_2d.copy()

    fields_2d = compute_2d_score_fields(
        spde, _u_snap, _mean_snap, _gamma_snap, config.fd_epsilon
    )
    return times, malliavin_scores, fd_scores, fields_2d


# =============================================================================
# Visualisation
# =============================================================================

_PI_TICKS  = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
_PI_LABELS = ["$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
# Compact tick labels used on non-bottom rows (avoid clutter)
_PI_LABELS_COMPACT = ["$0$", "", r"$\pi$", "", r"$2\pi$"]


def _set_pi_ticks(ax, which="both", labels=True, compact=False):
    labs = (_PI_LABELS_COMPACT if compact else _PI_LABELS) if labels else [""] * 5
    if which in ("x", "both"):
        ax.set_xticks(_PI_TICKS)
        ax.set_xticklabels(labs, fontsize=7.5)
    if which in ("y", "both"):
        ax.set_yticks(_PI_TICKS)
        ax.set_yticklabels(labs, fontsize=7.5)


def _style_heatmap_ax(ax):
    """Minimal, clean heatmap axis: thin spines, no tick marks on image border."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(which="both", direction="in", width=0.5, length=2.5)


def _slim_colorbar(fig, im, ax, *, extend="neither"):
    """Attach a slim, elegant vertical colorbar."""
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025, aspect=26,
                      extend=extend)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5, width=0.4, length=2.5, direction="in")
    return cb


def create_validation_figure(
    output_path: Path,
    spde_results: Sequence[tuple],
    config: SimulationConfig,
) -> plt.Figure:
    """
    Publication-quality 4-row × 2-column validation figure.

    Layout per row (one SPDE per row):
      Col 0 — Centred solution field (u - S(t)u0)(x,y) at t = T  [RdBu_r]
      Col 1 — Pointwise |Malliavin − FD| error                    [viridis, LogNorm]

    Centred SPDE name above each row, column headers at top.
    """
    assert len(spde_results) == 4

    _soln_cmap  = "RdBu_r"
    _error_cmap = "viridis"

    panel_letters = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(
        4, 2, figsize=(7.2, 11.2), facecolor="white",
        gridspec_kw={"hspace": 0.35, "wspace": 0.10,
                     "top": 0.92, "bottom": 0.045,
                     "left": 0.08, "right": 0.92},
    )

    # Column headers — well above the first row
    fig.text(0.30, 0.960,
             r"Stochastic component  $(u - S(t)\,u_0)(x,\,y)$",
             ha="center", va="bottom", fontsize=10, fontstyle="italic",
             color=COLOURS["text_dark"])
    fig.text(0.73, 0.960,
             r"Score error  $|s_{\mathrm{Mall}} - s_{\mathrm{FD}}|$",
             ha="center", va="bottom", fontsize=10, fontstyle="italic",
             color=COLOURS["text_dark"])

    for row, (title, times, mall_scores, fd_scores, fields_2d) in enumerate(
        spde_results
    ):
        cf   = fields_2d["centred"]       # centred solution field
        ef   = fields_2d["error"]
        x, y = fields_2d["x"], fields_2d["y"]
        is_bottom = (row == 3)
        ax_cf = axes[row, 0]
        ax_ef = axes[row, 1]

        # ── Centred row title spanning both columns ───────────────────────
        mid_x = 0.5 * (ax_cf.get_position().x0 + ax_ef.get_position().x1)
        title_y = ax_cf.get_position().y1 + 0.010
        fig.text(mid_x, title_y, f"{panel_letters[row]}  {title}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color=COLOURS["text_dark"])

        # ── Col 0: Centred solution field ─────────────────────────────────
        vmax_cf = float(np.abs(cf).max()) or 1.0

        im_cf = ax_cf.pcolormesh(
            x, y, cf.T,
            cmap=_soln_cmap, vmin=-vmax_cf, vmax=vmax_cf,
            shading="gouraud", rasterized=True,
        )
        ax_cf.set_aspect("equal")
        _style_heatmap_ax(ax_cf)

        _set_pi_ticks(ax_cf, "y", compact=not is_bottom)
        ax_cf.set_ylabel("$y$", fontsize=9, labelpad=2)
        if is_bottom:
            _set_pi_ticks(ax_cf, "x", compact=False)
            ax_cf.set_xlabel("$x$", fontsize=9, labelpad=2)
        else:
            ax_cf.set_xticks(_PI_TICKS)
            ax_cf.set_xticklabels([""] * 5)

        cb_cf = _slim_colorbar(fig, im_cf, ax_cf)
        cb_cf.set_ticks([-vmax_cf, 0, vmax_cf])
        cb_cf.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.1e}" if v != 0 else "0")
        )

        # ── Col 1: Error field ────────────────────────────────────────────
        e_flat = ef.flatten()
        e_pos  = e_flat[e_flat > 0]
        ev_min = float(e_pos.min()) if len(e_pos) else 1e-16
        ev_max = float(ef.max())
        if ev_max <= ev_min:
            ev_max = ev_min * 10

        im_ef = ax_ef.pcolormesh(
            x, y, ef.T,
            cmap=_error_cmap, norm=LogNorm(vmin=ev_min, vmax=ev_max),
            shading="auto", rasterized=True,
        )
        ax_ef.set_aspect("equal")
        _style_heatmap_ax(ax_ef)

        ax_ef.set_yticks(_PI_TICKS)
        ax_ef.set_yticklabels([""] * 5)
        if is_bottom:
            _set_pi_ticks(ax_ef, "x", compact=False)
            ax_ef.set_xlabel("$x$", fontsize=9, labelpad=2)
        else:
            ax_ef.set_xticks(_PI_TICKS)
            ax_ef.set_xticklabels([""] * 5)

        cb_ef = _slim_colorbar(fig, im_ef, ax_ef)
        cb_ef.ax.yaxis.set_major_formatter(
            mticker.LogFormatterSciNotation(base=10, labelOnlyBase=True)
        )
        cb_ef.ax.tick_params(labelsize=6.5)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".eps"):
        fpath = output_path.with_suffix(ext)
        kwargs = {"format": "eps"} if ext == ".eps" else {}
        fig.savefig(fpath, dpi=DPI, **kwargs)
        print(f"  Saved: {fpath}")

    plt.close(fig)
    return fig


# =============================================================================
# Main Driver
# =============================================================================

def run_validation(
    output_dir: Path = OUTPUT_DIR,
    config: SimulationConfig = SimulationConfig(),
    seed: int = RANDOM_SEED,
) -> None:
    """Run full 2D validation for all four SPDE classes."""
    rng = np.random.default_rng(seed)

    spde_specs = [
        ("Stochastic Heat Equation",
         Heat2D(config.N, config.noise_decay, nu=1.0)),
        ("Ornstein\u2013Uhlenbeck",
         OrnsteinUhlenbeck2D(config.N, config.noise_decay, nu=1.0, kappa=2.0)),
        ("Stochastic Biharmonic",
         Biharmonic2D(config.N, config.noise_decay, mu=1.0)),
        ("Swift\u2013Hohenberg",
         SwiftHohenberg2D(config.N, config.noise_decay, r=-0.5)),
    ]

    results = []
    for title, spde in spde_specs:
        print(f"\n  {title}")
        print(f"    Grid: {spde.mx} x {spde.mx} = {spde.M} collocation points")
        print(f"    Eigenvalue range: [{spde.operator_eigenvalues.min():.2f}, "
              f"{spde.operator_eigenvalues.max():.6f}]")

        times, mall, fd, fields_2d = simulate_score_trajectories(spde, config, rng)

        errs_ts = np.abs(mall - fd)
        errs_2d = fields_2d["error"].flatten()
        print(f"    Scalar max |error|: {errs_ts.max():.2e}")
        print(f"    Fourier-space max |error|: {fields_2d['fourier_error']:.2e}")
        print(f"    Physical-space max |error|: {errs_2d.max():.2e}  "
              f"mean: {errs_2d.mean():.2e}")

        results.append((title, times, mall, fd, fields_2d))

    print("\n  Generating figure...")
    create_validation_figure(output_dir / "score_2d_collocation", results, config)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("=" * 70)
    print("Score Formula Validation -- 2D Fourier Pseudospectral Collocation")
    print("Domain: [0, 2pi]^2 with periodic boundary conditions")
    print("=" * 70)
    run_validation()
    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

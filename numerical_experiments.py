"""
Score Formula Validation for Infinite-Dimensional SPDEs
========================================================

Numerical validation of the Malliavin score formula for linear SPDEs
using spectral Galerkin discretisation in the sine basis.

This module validates the closed-form score formula:
    β_h(u) = -⟨u - S(t)u₀, γ_t⁻¹ h⟩

against finite-difference baselines for eight SPDE classes:

    Second-order:
        - Stochastic Heat Equation:     A = Δ
        - Ornstein-Uhlenbeck:           A = Δ - αI
        - Advection-Diffusion:          A = νΔ
        - Fractional Laplacian:         A = -(-Δ)^α

    Fourth-order:
        - Biharmonic:                   A = -Δ²
        - Cahn-Hilliard (linear):       A = -Δ² + βΔ
        - Swift-Hohenberg (linear):     A = r - (1 + Δ)²

    General:
        - Polynomial in Laplacian:      A = Σⱼ cⱼ Δʲ

Usage:
    python score_validation.py
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42
OUTPUT_DIR = Path("figures")
DPI = 300

# Updated font sizes to match document text (typically 10-11pt)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,           # Base font size matching document
    "axes.labelsize": 11,      # Axis labels same as text
    "axes.titlesize": 12,      # Titles slightly larger
    "legend.fontsize": 10,     # Legend readable
    "xtick.labelsize": 10,     # Tick labels
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
})

COLOURS = {
    "blue": "#2D5A7B",
    "orange": "#D4763A",
    "green": "#4A7C59",
    "purple": "#7B4A7C",
    "red": "#B54A4A",
    "teal": "#2A9D8F",
    "grey": "#666666",
    "light_grey": "#CCCCCC",
}


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for score trajectory simulation."""
    n_modes: int = 64
    n_paths: int = 4
    n_timesteps: int = 50
    t_start: float = 0.02
    t_end: float = 1.0
    noise_decay: float = 2.0
    fd_epsilon: float = 1e-5


# =============================================================================
# Abstract Base Class
# =============================================================================

class LinearSPDE(ABC):
    """
    Abstract base class for linear SPDEs with spectral Galerkin discretisation.

    Solves SPDEs of the form:
        du = Au dt + Q^{1/2} dW

    on the domain (0,1) with homogeneous Dirichlet boundary conditions.
    The eigenbasis is {√2 sin(kπx)}_{k≥1} with Laplacian eigenvalues λ_k = (kπ)².

    Subclasses must implement:
        - name: Human-readable identifier
        - _compute_operator_eigenvalues: Define a_k for the drift operator A
    """

    def __init__(self, n_modes: int, time: float, noise_decay: float) -> None:
        """
        Initialise the SPDE discretisation.

        Args:
            n_modes: Number of Fourier modes (truncation level N)
            time: Terminal time t for the solution u(t)
            noise_decay: Decay rate for noise covariance q_k = k^{-noise_decay}
        """
        self.n_modes = n_modes
        self.time = time
        self.noise_decay = noise_decay

        # Mode indices k = 1, 2, ..., N
        self.modes = np.arange(1, n_modes + 1)

        # Laplacian eigenvalues: -Δφ_k = (kπ)² φ_k
        self.laplacian_eigenvalues = (np.pi * self.modes) ** 2

        # Noise covariance eigenvalues: Q φ_k = q_k φ_k
        self.q_eigenvalues = self.modes ** (-noise_decay)
        self.q_sqrt_eigenvalues = np.sqrt(self.q_eigenvalues)

        # Compute operator-specific quantities
        self._compute_operator_eigenvalues()
        self._validate_stability()
        self._compute_semigroup_and_covariance()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the SPDE."""
        pass

    @abstractmethod
    def _compute_operator_eigenvalues(self) -> None:
        """
        Compute the eigenvalues a_k of the drift operator A.

        Must set self.operator_eigenvalues as a numpy array of shape (n_modes,).
        """
        pass

    def _validate_stability(self) -> None:
        """Verify that all eigenvalues are negative (dissipative system)."""
        if not np.all(self.operator_eigenvalues < 0):
            unstable_modes = np.where(self.operator_eigenvalues >= 0)[0] + 1
            raise ValueError(
                f"{self.name}: Non-dissipative eigenvalues detected at modes {unstable_modes}. "
                f"Values: {self.operator_eigenvalues[unstable_modes - 1]}. "
                f"The framework requires a_k < 0 for all k."
            )

    def _compute_semigroup_and_covariance(self) -> None:
        """
        Compute semigroup S(t) and Malliavin covariance γ_t eigenvalues.

        Semigroup:  S(t)φ_k = e^{a_k t} φ_k

        Malliavin covariance (Lyapunov equation solution):
            γ_t = ∫₀ᵗ S(s) Q S(s)* ds

        In eigenspace:
            γ_k = q_k ∫₀ᵗ e^{2a_k s} ds = q_k (e^{2a_k t} - 1) / (2a_k)
        """
        t = self.time
        a = self.operator_eigenvalues
        q = self.q_eigenvalues

        # Semigroup eigenvalues: S(t)_k = e^{a_k t}
        self.semigroup_eigenvalues = np.exp(t * a)

        # Malliavin covariance eigenvalues with numerically stable formula
        gamma = np.empty(self.n_modes)
        for k in range(self.n_modes):
            if np.abs(a[k]) < 1e-14:
                # Limit as a → 0: γ = qt
                gamma[k] = q[k] * t
            else:
                # Standard formula: γ = q(e^{2at} - 1) / (2a)
                gamma[k] = q[k] * (np.exp(2.0 * a[k] * t) - 1.0) / (2.0 * a[k])

        if not np.all(gamma > 0):
            raise ValueError(
                f"{self.name}: Non-positive covariance eigenvalues detected. "
                f"Min value: {gamma.min():.2e}"
            )

        self.gamma_eigenvalues = gamma
        self.gamma_inverse_eigenvalues = 1.0 / gamma

    def score_malliavin(
        self,
        u_coeffs: np.ndarray,
        h_coeffs: np.ndarray,
        mean_coeffs: np.ndarray,
    ) -> float:
        """
        Compute the directional score using the Malliavin formula.

        The score (logarithmic derivative) along direction h ∈ H_t is:

            β_h(u) = -⟨u - m, γ_t⁻¹ h⟩_H = -Σ_k (u_k - m_k) h_k / γ_k

        where m = S(t)u₀ is the mean.

        Args:
            u_coeffs: Fourier coefficients of the state u(t)
            h_coeffs: Fourier coefficients of the direction h
            mean_coeffs: Fourier coefficients of the mean S(t)u₀

        Returns:
            The directional score β_h(u)
        """
        centred = u_coeffs - mean_coeffs
        return float(-np.sum(centred * h_coeffs * self.gamma_inverse_eigenvalues))

    def score_finite_difference(
        self,
        u_coeffs: np.ndarray,
        h_coeffs: np.ndarray,
        mean_coeffs: np.ndarray,
        epsilon: float = 1e-5,
    ) -> float:
        """
        Compute the directional score using central finite differences.

        Approximates:
            β_h(u) = ∂_h log p(u) ≈ [log p(u + εĥ) - log p(u - εĥ)] / (2ε)

        where ĥ = h/‖h‖ is the unit direction and p is the Gaussian density.

        Args:
            u_coeffs: Fourier coefficients of the state
            h_coeffs: Fourier coefficients of the direction
            mean_coeffs: Fourier coefficients of the mean
            epsilon: Finite difference step size

        Returns:
            Finite difference approximation of β_h(u)
        """
        def log_density(u: np.ndarray) -> float:
            """Log-density of N(mean, γ) up to constant."""
            centred = u - mean_coeffs
            return float(-0.5 * np.sum(centred ** 2 * self.gamma_inverse_eigenvalues))

        h_norm = float(np.sqrt(np.sum(h_coeffs ** 2)))
        scaled_eps = epsilon * h_norm

        log_plus = log_density(u_coeffs + scaled_eps * h_coeffs)
        log_minus = log_density(u_coeffs - scaled_eps * h_coeffs)

        return (log_plus - log_minus) / (2.0 * scaled_eps)


# =============================================================================
# Second-Order SPDEs
# =============================================================================

class HeatEquation(LinearSPDE):
    """
    Stochastic Heat Equation.

        du = Δu dt + Q^{1/2} dW

    Eigenvalues: a_k = -(kπ)²

    The fundamental parabolic SPDE modelling diffusion processes.
    """

    @property
    def name(self) -> str:
        return "Heat Equation"

    def _compute_operator_eigenvalues(self) -> None:
        self.operator_eigenvalues = -self.laplacian_eigenvalues


class OrnsteinUhlenbeck(LinearSPDE):
    """
    Ornstein-Uhlenbeck Process in function space.

        du = (Δu - αu) dt + Q^{1/2} dW

    Eigenvalues: a_k = -(kπ)² - α

    Adds linear damping to the heat equation, ensuring faster convergence
    to the stationary distribution.

    Args:
        alpha: Damping coefficient (α > 0)
    """

    def __init__(
        self,
        n_modes: int,
        time: float,
        noise_decay: float,
        alpha: float = 2.0,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"Damping coefficient α must be positive, got {alpha}")
        self.alpha = alpha
        super().__init__(n_modes, time, noise_decay)

    @property
    def name(self) -> str:
        return f"Ornstein-Uhlenbeck (α={self.alpha})"

    def _compute_operator_eigenvalues(self) -> None:
        self.operator_eigenvalues = -self.laplacian_eigenvalues - self.alpha


class AdvectionDiffusion(LinearSPDE):
    """
    Advection-Diffusion Equation (diffusion-dominated).

        du = νΔu dt + Q^{1/2} dW

    Eigenvalues: a_k = -ν(kπ)²

    Models transport with diffusion. The parameter ν controls the
    diffusion strength (Péclet number).

    Args:
        nu: Diffusion coefficient (ν > 0)
    """

    def __init__(
        self,
        n_modes: int,
        time: float,
        noise_decay: float,
        nu: float = 0.1,
    ) -> None:
        if nu <= 0:
            raise ValueError(f"Diffusion coefficient ν must be positive, got {nu}")
        self.nu = nu
        super().__init__(n_modes, time, noise_decay)

    @property
    def name(self) -> str:
        return f"Advection-Diffusion (ν={self.nu})"

    def _compute_operator_eigenvalues(self) -> None:
        self.operator_eigenvalues = -self.nu * self.laplacian_eigenvalues


class FractionalLaplacian(LinearSPDE):
    """
    Fractional Laplacian SPDE.

        du = -(-Δ)^α u dt + Q^{1/2} dW

    Eigenvalues: a_k = -(kπ)^{2α}

    Models anomalous diffusion and Lévy flights. The fractional exponent α
    interpolates between:
        - α → 0: Bounded perturbation of identity
        - α = 0.5: Half-Laplacian (related to Cauchy process)
        - α = 1: Standard Laplacian (Brownian motion)
        - α > 1: Super-diffusion (smoother solutions)

    Args:
        alpha: Fractional exponent (0 < α ≤ 2)
    """

    def __init__(
        self,
        n_modes: int,
        time: float,
        noise_decay: float,
        alpha: float = 0.75,
    ) -> None:
        if not 0 < alpha <= 2:
            raise ValueError(f"Fractional exponent α must be in (0, 2], got {alpha}")
        self.alpha = alpha
        super().__init__(n_modes, time, noise_decay)

    @property
    def name(self) -> str:
        return f"Fractional (α={self.alpha})"

    def _compute_operator_eigenvalues(self) -> None:
        # a_k = -(kπ)^{2α} = -[(kπ)²]^α
        self.operator_eigenvalues = -(self.laplacian_eigenvalues ** self.alpha)


# =============================================================================
# Fourth-Order SPDEs
# =============================================================================

class Biharmonic(LinearSPDE):
    """
    Biharmonic (Fourth-Order) SPDE.

        du = -Δ²u dt + Q^{1/2} dW

    Eigenvalues: a_k = -(kπ)⁴

    Models thin plate vibrations, beam bending, and appears in the
    linearised Kuramoto-Sivashinsky equation. The fourth-order operator
    provides stronger smoothing than the Laplacian.
    """

    @property
    def name(self) -> str:
        return "Biharmonic"

    def _compute_operator_eigenvalues(self) -> None:
        self.operator_eigenvalues = -(self.laplacian_eigenvalues ** 2)


class CahnHilliard(LinearSPDE):
    """
    Linearised Cahn-Hilliard Equation.

        du = (-Δ² + βΔ)u dt + Q^{1/2} dW

    Eigenvalues: a_k = -(kπ)⁴ + β(kπ)² = (kπ)²[β - (kπ)²]

    Models phase separation and spinodal decomposition. The parameter β
    controls the competition between the destabilising second-order term
    and the stabilising fourth-order term.

    Stability requires: β < (π)² ≈ 9.87 (otherwise mode k=1 is unstable)

    Args:
        beta: Second-order coefficient (β < π² for stability)
    """

    def __init__(
        self,
        n_modes: int,
        time: float,
        noise_decay: float,
        beta: float = 1.0,
    ) -> None:
        self.beta = beta
        # Check stability before calling super().__init__
        min_laplacian = np.pi ** 2  # First mode
        if beta >= min_laplacian:
            raise ValueError(
                f"Cahn-Hilliard unstable: β={beta} ≥ π²≈{min_laplacian:.2f}. "
                f"Mode k=1 would have positive eigenvalue."
            )
        super().__init__(n_modes, time, noise_decay)

    @property
    def name(self) -> str:
        return f"Cahn-Hilliard (β={self.beta})"

    def _compute_operator_eigenvalues(self) -> None:
        # a_k = -(kπ)⁴ + β(kπ)² = (kπ)²[β - (kπ)²]
        lap = self.laplacian_eigenvalues
        self.operator_eigenvalues = -lap ** 2 + self.beta * lap


class SwiftHohenberg(LinearSPDE):
    """
    Linearised Swift-Hohenberg Equation.

        du = [r - (1 + Δ)²]u dt + Q^{1/2} dW

    Eigenvalues: a_k = r - (1 - (kπ)²)²

    A paradigmatic model for pattern formation near onset. The control
    parameter r determines proximity to the instability threshold.

    Stability requires: r < min_k (1 - (kπ)²)² ≈ 78.96 for integer k

    For r > 0, modes near (kπ)² ≈ 1 become weakly damped, leading to
    pattern selection. Since k must be a positive integer and π² ≈ 9.87,
    no mode is exactly critical.

    Args:
        r: Control parameter (r < 78.96 for stability)
    """

    def __init__(
        self,
        n_modes: int,
        time: float,
        noise_decay: float,
        r: float = 0.0,
    ) -> None:
        self.r = r
        # Check stability: need r < (1 - (kπ)²)² for all k ≥ 1
        # Minimum over k ≥ 1 is at k = 1: (1 - π²)² ≈ 78.96
        critical_value = (1 - np.pi ** 2) ** 2
        if r >= critical_value:
            raise ValueError(
                f"Swift-Hohenberg unstable: r={r} ≥ (1-π²)²≈{critical_value:.2f}. "
                f"Mode k=1 would have positive eigenvalue."
            )
        super().__init__(n_modes, time, noise_decay)

    @property
    def name(self) -> str:
        if self.r == 0:
            return "Swift-Hohenberg"
        return f"Swift-Hohenberg (r={self.r})"

    def _compute_operator_eigenvalues(self) -> None:
        # a_k = r - (1 - (kπ)²)²
        lap = self.laplacian_eigenvalues
        self.operator_eigenvalues = self.r - (1 - lap) ** 2


# =============================================================================
# General Polynomial SPDE
# =============================================================================

@dataclass
class PolynomialCoefficients:
    """
    Coefficients for polynomial-in-Laplacian operator.

    Represents A = Σⱼ cⱼ Δʲ where Δʲ means j applications of Δ.

    The coefficients dict maps power j → coefficient cⱼ.

    Examples:
        - Heat: {1: 1.0}                    → A = Δ
        - Biharmonic: {2: -1.0}             → A = -Δ²
        - Cahn-Hilliard: {2: -1.0, 1: β}    → A = -Δ² + βΔ
    """
    coefficients: dict = field(default_factory=lambda: {1: 1.0})

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("At least one coefficient must be specified")
        if not all(isinstance(k, int) and k >= 0 for k in self.coefficients.keys()):
            raise ValueError("Powers must be non-negative integers")


class PolynomialLaplacian(LinearSPDE):
    """
    General Polynomial-in-Laplacian SPDE.

        du = [Σⱼ cⱼ Δʲ] u dt + Q^{1/2} dW

    Eigenvalues: a_k = Σⱼ cⱼ (-1)ʲ (kπ)^{2j}

    This is the most general form supported by the spectral framework.
    Any operator that is a polynomial function of the Laplacian can be
    represented.

    Args:
        poly: PolynomialCoefficients specifying the operator

    Examples:
        >>> # Heat equation: A = Δ
        >>> poly = PolynomialCoefficients({1: 1.0})
        >>> spde = PolynomialLaplacian(64, 0.5, 2.0, poly)

        >>> # Biharmonic: A = -Δ²
        >>> poly = PolynomialCoefficients({2: -1.0})

        >>> # Mixed: A = -Δ² + 0.5Δ - 0.1I
        >>> poly = PolynomialCoefficients({2: -1.0, 1: 0.5, 0: -0.1})
    """

    def __init__(
        self,
        n_modes: int,
        time: float,
        noise_decay: float,
        poly: PolynomialCoefficients = PolynomialCoefficients(),
    ) -> None:
        self.poly = poly
        super().__init__(n_modes, time, noise_decay)

    @property
    def name(self) -> str:
        terms = []
        for power in sorted(self.poly.coefficients.keys(), reverse=True):
            coeff = self.poly.coefficients[power]
            if power == 0:
                terms.append(f"{coeff:+.2g}I")
            elif power == 1:
                terms.append(f"{coeff:+.2g}Δ")
            else:
                terms.append(f"{coeff:+.2g}Δ^{power}")
        return f"Polynomial ({' '.join(terms)})"

    def _compute_operator_eigenvalues(self) -> None:
        """
        Compute a_k = Σⱼ cⱼ (-λ_k)ʲ where λ_k = (kπ)².

        Note: Δφ_k = -λ_k φ_k, so Δʲφ_k = (-λ_k)ʲ φ_k = (-1)ʲ λ_k^j φ_k
        """
        lap = self.laplacian_eigenvalues
        eigenvalues = np.zeros(self.n_modes)

        for power, coeff in self.poly.coefficients.items():
            # Δʲ has eigenvalue (-λ)ʲ = (-1)ʲ λʲ
            eigenvalues += coeff * ((-1) ** power) * (lap ** power)

        self.operator_eigenvalues = eigenvalues


# =============================================================================
# Simulation
# =============================================================================

def simulate_score_trajectories(
    spde_factory: type[LinearSPDE],
    spde_kwargs: dict,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate score trajectories along sample paths of an SPDE.

    For each sample path, we:
    1. Generate Brownian increments dW
    2. Compute the solution u(t) = S(t)u₀ + ∫₀ᵗ S(t-s) Q^{1/2} dW(s)
    3. Evaluate both Malliavin and finite-difference scores

    Args:
        spde_factory: Class to instantiate for each time point
        spde_kwargs: Keyword arguments for the SPDE constructor
        config: Simulation configuration
        rng: Random number generator

    Returns:
        Tuple of (times, malliavin_scores, fd_scores) with shapes
        (n_timesteps,), (n_paths, n_timesteps), (n_paths, n_timesteps)
    """
    times = np.linspace(config.t_start, config.t_end, config.n_timesteps)
    dt = times[1] - times[0]

    # Initial condition: u₀ = e₁ + 0.5e₂ + 0.25e₃
    u0 = np.zeros(config.n_modes)
    u0[:3] = [1.0, 0.5, 0.25]

    # Direction: h = e₁ (first eigenmode)
    h = np.zeros(config.n_modes)
    h[0] = 1.0

    malliavin_scores = np.zeros((config.n_paths, config.n_timesteps))
    fd_scores = np.zeros((config.n_paths, config.n_timesteps))

    for path_idx in range(config.n_paths):
        # Pre-generate Brownian increments for this path
        dW = rng.standard_normal((config.n_timesteps, config.n_modes)) * np.sqrt(dt)

        for t_idx, t in enumerate(times):
            spde = spde_factory(
                n_modes=config.n_modes,
                time=t,
                noise_decay=config.noise_decay,
                **spde_kwargs,
            )

            # Build stochastic integral: ∫₀ᵗ S(t-s) Q^{1/2} dW(s)
            # Discretised as: Σⱼ S(t - tⱼ) Q^{1/2} ΔWⱼ
            stochastic_integral = np.zeros(config.n_modes)
            for j in range(t_idx + 1):
                s = times[j]
                stochastic_integral += (
                    np.exp(spde.operator_eigenvalues * (t - s))
                    * spde.q_sqrt_eigenvalues
                    * dW[j]
                )

            # Solution: u(t) = S(t)u₀ + stochastic integral
            u_t = spde.semigroup_eigenvalues * u0 + stochastic_integral
            mean = spde.semigroup_eigenvalues * u0

            # Compute scores
            malliavin_scores[path_idx, t_idx] = spde.score_malliavin(u_t, h, mean)
            fd_scores[path_idx, t_idx] = spde.score_finite_difference(
                u_t, h, mean, epsilon=config.fd_epsilon
            )

    return times, malliavin_scores, fd_scores


# =============================================================================
# Visualisation - Updated for Professor's Requirements
# =============================================================================

def create_2x2_figure(
    output_path: Path,
    spde_specs: Sequence[tuple[type[LinearSPDE], dict, str]],
    config: SimulationConfig,
    rng: np.random.Generator,
) -> plt.Figure:
    """
    Create a 2x2 figure with 4 SPDE subplots.
    
    Each subplot has:
    - Main panel: score trajectories
    - Lower panel: error on log scale
    
    Args:
        output_path: Path for saving the figure
        spde_specs: List of 4 (class, kwargs, title) tuples
        config: Simulation configuration
        rng: Random number generator
    
    Returns:
        The matplotlib Figure object
    """
    assert len(spde_specs) == 4, "Need exactly 4 SPDEs for 2x2 grid"
    
    path_colours = [
        COLOURS["blue"],
        COLOURS["purple"],
        COLOURS["orange"],
        COLOURS["green"],
    ]
    
    # Create figure - full page width, appropriate height for 2x2
    fig = plt.figure(figsize=(7.5, 8))  # Width matches \textwidth, good height
    
    # Compute all data first to get global error range
    all_results = []
    all_errors = []
    
    for spde_class, spde_kwargs, title in spde_specs:
        times, mall_scores, fd_scores = simulate_score_trajectories(
            spde_class, spde_kwargs, config, rng
        )
        errors = np.abs(mall_scores - fd_scores)
        all_results.append((times, mall_scores, fd_scores, errors, title))
        all_errors.append(errors)
    
    # Compute dynamic y-limits for error plots
    all_errors_flat = np.concatenate([e.flatten() for e in all_errors])
    error_min = max(all_errors_flat.min(), 1e-16)
    error_max = all_errors_flat.max()
    y_min = error_min / 10
    y_max = error_max * 10
    
    # Create 2x2 grid with space for error subplots
    outer_grid = fig.add_gridspec(
        2, 2,
        hspace=0.35,
        wspace=0.30,
        top=0.95,
        bottom=0.08,
        left=0.10,
        right=0.98,
    )
    
    for idx, (times, mall_scores, fd_scores, errors, title) in enumerate(all_results):
        row = idx // 2
        col = idx % 2
        
        # Create inner grid for main plot + error plot
        inner_grid = outer_grid[row, col].subgridspec(
            2, 1,
            height_ratios=[3, 1],
            hspace=0.08,
        )
        
        ax_main = fig.add_subplot(inner_grid[0])
        ax_error = fig.add_subplot(inner_grid[1])
        
        # Plot trajectories
        for path_idx in range(config.n_paths):
            colour = path_colours[path_idx % len(path_colours)]
            
            # Malliavin: solid lines
            ax_main.plot(
                times, mall_scores[path_idx],
                "-", color=colour, linewidth=1.8, alpha=0.9
            )
            
            # Finite difference: hollow circles (sparse)
            ax_main.scatter(
                times[::5], fd_scores[path_idx, ::5],
                s=35, marker="o",
                facecolors="white", edgecolors=colour,
                linewidths=1.5, zorder=4,
            )
            
            # Error
            ax_error.semilogy(
                times, errors[path_idx] + 1e-16,
                "-", color=colour, linewidth=1.2, alpha=0.85
            )
        
        # Styling - main panel
        ax_main.axhline(0, color=COLOURS["light_grey"], linewidth=0.8)
        ax_main.set_title(title, fontweight="bold", fontsize=11, pad=6)
        ax_main.set_xlim([0, config.t_end + 0.02])
        ax_main.tick_params(labelbottom=False)
        ax_main.set_ylabel(r"$\beta_h(u(t))$", fontsize=11)
        
        # Styling - error panel
        ax_error.axhline(
            1e-12, color=COLOURS["grey"],
            linestyle="--", linewidth=1.0, alpha=0.7
        )
        ax_error.set_xlabel(r"$t$", fontsize=11)
        ax_error.set_xlim([0, config.t_end + 0.02])
        ax_error.set_ylim([y_min, y_max])
        ax_error.set_ylabel("|Error|", fontsize=10)
        
        # Legend on first panel only
        if idx == 0:
            legend_elements = [
                Line2D([0], [0], color=COLOURS["grey"], linewidth=1.8, label="Malliavin"),
                Line2D([0], [0], marker="o", color=COLOURS["grey"], markerfacecolor="white",
                       markersize=6, markeredgewidth=1.5, linestyle="None", label="Finite Diff."),
            ]
            ax_main.legend(
                handles=legend_elements, loc="upper right",
                frameon=True, framealpha=0.95, fontsize=10,
                edgecolor=COLOURS["light_grey"],
            )
    
    # Save in multiple formats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=DPI)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=DPI)
    fig.savefig(output_path.with_suffix(".eps"), format='eps', dpi=DPI)
    
    print(f"Saved: {output_path.with_suffix('.png')}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.eps')}")
    
    plt.close(fig)
    return fig


def create_validation_figures(
    output_dir: Path = OUTPUT_DIR,
    config: SimulationConfig = SimulationConfig(),
    seed: int = RANDOM_SEED,
) -> None:
    """
    Create two 2x2 figures: one for second-order SPDEs, one for fourth-order.
    
    Args:
        output_dir: Directory for output figures
        config: Simulation configuration
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    
    # Second-order SPDEs (2x2)
    second_order_specs = [
        (HeatEquation, {}, "Heat Equation"),
        (OrnsteinUhlenbeck, {"alpha": 2.0}, "Ornstein–Uhlenbeck"),
        (AdvectionDiffusion, {"nu": 0.1}, "Advection–Diffusion"),
        (FractionalLaplacian, {"alpha": 0.75}, "Fractional Laplacian"),
    ]
    
    print("\n" + "=" * 60)
    print("Generating Second-Order SPDEs Figure")
    print("=" * 60)
    create_2x2_figure(
        output_dir / "score_second_order",
        second_order_specs,
        config,
        rng,
    )
    
    # Fourth-order SPDEs (2x2)
    fourth_order_specs = [
        (Biharmonic, {}, "Biharmonic"),
        (CahnHilliard, {"beta": 1.0}, "Cahn–Hilliard"),
        (SwiftHohenberg, {"r": 0.0}, "Swift–Hohenberg"),
        (PolynomialLaplacian, {"poly": PolynomialCoefficients({2: -1.0, 1: 0.5})}, "Polynomial"),
    ]
    
    print("\n" + "=" * 60)
    print("Generating Fourth-Order SPDEs Figure")
    print("=" * 60)
    create_2x2_figure(
        output_dir / "score_fourth_order",
        fourth_order_specs,
        config,
        rng,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run the score formula validation for all SPDE classes."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 70)
    print("Score Formula Validation")
    print("Infinite-Dimensional SPDEs via Spectral Galerkin Discretisation")
    print("=" * 70)

    create_validation_figures()

    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

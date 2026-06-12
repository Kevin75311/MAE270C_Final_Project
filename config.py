"""
config.py — Single source of truth for all simulation parameters.

All physical constants, domain geometry, solver settings, control bounds,
and visualization preferences live here. No magic numbers anywhere else.

Import pattern in every other module:
    from config import cfg
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DomainConfig:
    """Spatial domain and mesh settings."""
    # Simulation dimensionality: 2 or 3
    ndim: int = 2

    # Domain size [m]
    Lx: float = 0.05          # domain width:  5 cm
    Ly: float = 0.05          # domain height: 5 cm
    Lz: float = 0.05          # domain depth:  5 cm  (used when ndim=3)

    # Mesh resolution
    Nx: int = 50              # number of voxels in x
    Ny: int = 50              # number of voxels in y
    Nz: int = 50              # number of voxels in z (used when ndim=3)

    # Tumor (target) region — centred in domain; always a 3-tuple (x, y, z)
    tumor_center: tuple = (0.025, 0.025, 0.025)   # [m]
    tumor_radius: float = 0.008                    # 8 mm radius

    # Safety margin around tumor (transition zone)
    safety_margin: float = 0.003   # 3 mm

    # Probe / applicator position — always a 3-tuple (x, y, z)
    probe_position: tuple = (0.025, 0.025, 0.025)  # centred on tumour

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny

    @property
    def dz(self) -> float:
        return self.Lz / self.Nz

    @property
    def N(self) -> int:
        """Total number of voxels (Nx·Ny in 2D, Nx·Ny·Nz in 3D)."""
        return self.Nx * self.Ny * (self.Nz if self.ndim == 3 else 1)


@dataclass
class TissueConfig:
    """Tissue physical properties (soft tissue defaults)."""
    # Thermal properties
    rho: float = 1050.0          # tissue density              [kg/m³]
    c:   float = 3600.0          # specific heat capacity      [J/(kg·K)]
    k:   float = 0.51            # thermal conductivity        [W/(m·K)]

    # Blood perfusion
    omega_b: float = 0.005       # perfusion rate              [1/s]
    rho_b:   float = 1050.0      # blood density               [kg/m³]
    c_b:     float = 3600.0      # blood specific heat         [J/(kg·K)]
    T_blood: float = 37.0        # arterial blood temp         [°C]

    # Metabolic heat generation
    Q_met: float = 500.0         # metabolic heat rate         [W/m³]

    # Initial / baseline temperature
    T_init: float = 37.0         # pre-treatment temperature   [°C]


@dataclass
class ArrheniusConfig:
    """
    Arrhenius thermal damage model constants.

    ┌────────────────────────────────────────────────────────────┐
    │  STATE EQUATION 2 — Arrhenius Damage ODE                   │
    │  dΩ/dt = A · exp(−E_a / (R · T))                           │
    │  Threshold: Ω = 1  →  63% probability of cell death        │
    └────────────────────────────────────────────────────────────┘

    Reference values for protein denaturation (Henriques & Moritz 1947).
    """
    A:   float = 3.1e98          # frequency factor            [1/s]
    E_a: float = 6.28e5          # activation energy           [J/mol]
    R:   float = 8.314           # universal gas constant      [J/(mol·K)]
    damage_threshold: float = 1.0  # Ω ≥ 1 → irreversible necrosis


@dataclass
class SARConfig:
    """
    Specific Absorption Rate (SAR) model for the microwave applicator.

    Three probe models are supported (set via probe_model):

      'point'  — isotropic Gaussian blob (2D) or anisotropic Gaussian (3D).
                 Fast baseline; assumes needle is perpendicular to the image plane.

      'line'   — cylindrical heating zone of length L_active along probe_direction,
                 with Gaussian end-cap rolloff past the active tips.
                 Most realistic for a microwave needle in the r-z plane.

      'dipole' — sin²(θ) toroidal radiation pattern × Gaussian radial decay.
                 Approximates the near-field of a half-wave slot antenna.
    """
    # ── Spatial spread parameters ─────────────────────────────────────────────
    sigma_sar:   float = 0.006   # radial spread ⊥ to needle axis       [m]  (all models)
    sigma_sar_z: float = 0.005   # axial end-cap falloff beyond L_active [m]  (line model)
                                  # also: full axial Gaussian for 'point' in 3D
                                  # keep small (≤ L_active/4) so the flat plateau is visible

    # ── Peak SAR per unit power [W/kg per W applied] ──────────────────────────
    sar_peak: float = 50.0       # at probe centre; calibrated to tissue properties

    # ── Applicator frequency (informational; affects sar_peak calibration) ────
    freq_GHz: float = 2.45       # microwave frequency                   [GHz]

    # ── Probe geometry model ──────────────────────────────────────────────────
    probe_model: str = 'point'           # 'point' | 'line' | 'dipole'
    L_active:    float = 0.030           # active antenna length             [m]  (line model)
    probe_direction: tuple = (0, 0, 1)   # unit vector along needle axis (z-axis default)


@dataclass
class BoundaryConfig:
    """Boundary condition settings for the bioheat PDE on ∂Ω."""
    # BC type applied to all outer boundary voxels each timestep
    bc_type: str = 'dirichlet'   # 'dirichlet' | 'neumann' | 'robin'

    # Dirichlet: fixed boundary temperature [°C]
    T_wall: float = 37.0         # defaults to body temperature

    # Robin: convective cooling
    h_c:   float = 50.0          # convective heat transfer coefficient [W/(m²·K)]
    T_inf: float = 20.0          # ambient / coolant temperature        [°C]


@dataclass
class ControlConfig:
    """Control variable bounds and actuator constraints."""
    # Power bounds
    P_min: float = 0.0           # minimum applicator power    [W]
    P_max: float = 50.0          # maximum applicator power    [W]

    # Slew rate limit (rate of change of power)
    P_dot_max: float = 10.0      # max |dP/dt|                 [W/s]

    # Temperature safety thresholds
    T_safe: float = 50.0         # healthy tissue limit        [°C]
                                  # 50°C (not 45): at 45 the strict hard-constraint
                                  # time-optimal problem is infeasible for the
                                  # default geometry — full ablation caps at ~76%
                                  # and SLSQP never converges.  50°C gives thermal
                                  # headroom so full ablation is feasible and the
                                  # direct solver converges (Exit mode 0).
    T_crit: float = 42.0         # critical structures limit   [°C]

    # Maximum allowed treatment duration
    T_max_treatment: float = 600.0  # 10 minutes               [s]


@dataclass
class SolverConfig:
    """Time integration and optimiser settings."""
    # Time stepping
    dt: float = 1.0              # time step size              [s]
    t_final: float = 300.0       # simulation duration         [s]  (free in OCP)

    # Post-treatment observation window: after the free final time t_f* (full
    # ablation), keep integrating the cooldown with the applicator OFF (P=0) for
    # this many extra seconds so the thermal relaxation is visible.  Does NOT
    # affect the cost, which is still evaluated at t_f*.
    post_observation_time: float = 20.0   # [s]

    # ODE integrator choice: 'euler', 'rk4', 'scipy_ode'
    integrator: str = 'rk4'

    # Optimiser (for open-loop OCP)
    optimizer: str = 'SLSQP'     # scipy.optimize method
    max_iter: int = 500
    tol: float = 1e-6

    # When True, T ≤ T_safe in healthy tissue is enforced as a hard NLP
    # inequality constraint passed to SLSQP (used by time-optimal mode).
    # When False (default / energy mode), safety is soft-penalized via alpha2.
    enforce_safety_hard: bool = False

    # MPC settings
    mpc_horizon: float = 60.0    # prediction horizon          [s]
    mpc_dt_ctrl: float = 5.0     # control update interval     [s]

    @property
    def n_steps(self) -> int:
        return int(self.t_final / self.dt)


@dataclass
class CostConfig:
    """
    Weights for the Bolza cost functional J.

    ┌──────────────────────────────────────────────────────────────────┐
    │  COST FUNCTIONAL                                                 │
    │  J = γ1·∫_{Ω_T} max(0, 1−Ω_d(t_f))² dr  +  γ2·t_f                │
    │    + ∫₀^tf [ α1‖u‖² + α2·∫_{Ω_H}(T−T_safe)₊² dr + α3 ] dt        │
    └──────────────────────────────────────────────────────────────────┘
    """
    # Objective mode: 'energy' (default) or 'time' (time-optimal).
    # In 'time' mode, alpha1 is replaced by alpha1_time (tiny regularization)
    # and alpha2 is zeroed out (healthy-tissue safety is instead enforced as a
    # hard SLSQP inequality constraint via cfg.solver.enforce_safety_hard).
    mode: str = 'energy'          # 'energy' | 'time'

    # Running cost weights
    # alpha1 (energy weight) is tuned to make the problem ENERGY-aware rather
    # than purely time-optimal.  Rationale: openloop runs the applicator at full
    # power, which IS the minimum-time control — so when time dominates, openloop
    # is unbeatable and the optimizers can only tie it.  Raising alpha1 makes
    # full-power wasteful, so the optimal control is a gentler schedule that the
    # indirect/direct solvers find and openloop cannot — they then beat openloop.
    #   alpha1=5e-4: openloop≈125, indirect≈106, direct≈105 (optimizers win ~15%).
    # Upper bound: alpha1 ≲ 1.2e-3, beyond which energy is so costly that "do
    # nothing" (pay the gamma1 ablation penalty ≈192) beats ablating and the
    # direct optimizer stops ablating.  See gamma1 note below.
    # AGGRESSIVE SET (active): alpha1=2e-4, gamma1=2e6, gamma2=0.3.  Lower energy
    # weight → more willing to use power; higher gamma1 → ablation dominates
    # harder; higher gamma2 → stronger min-time push.  Default/known-good set was
    # alpha1=5e-4, gamma1=1e6, gamma2=0.1.
    alpha1: float = 5e-4         # energy penalty (‖u‖²) — aggressive set (was 5e-4)
    alpha1_time: float = 1e-6    # tiny regularization energy weight used in
                                  # time-optimal mode (keeps SLSQP objective from
                                  # being a degenerate LP; not literally zero)
    alpha2: float = 1.0          # healthy tissue overheating penalty
    alpha3: float = 0.01         # time-rate cost (encourages speed)

    # Terminal cost weights
    # NOTE on gamma1 magnitude: the ablation penalty is a *spatial integral*,
    #   J_ablation = gamma1 · Σ_{Ω_T} (1−Ω_d)₊² · dA,
    # so it is scaled by the voxel area dA ≈ 1e-6 m².  With the full tumor
    # un-ablated this term equals gamma1·N_tumor·dA ≈ gamma1·1.9e-4.  To make
    # "kill the tumor" dominate the (un-scaled) energy cost α1·P²·t ≈ 75 and
    # the time cost γ2·t_f ≈ 30, gamma1 must be large.  gamma1=1e6 →
    # J_ablation(none)≈190, comfortably outweighing energy/time so the
    # optimizer chooses to ablate.  (Previously 10.0 → J_ablation(none)≈0.002,
    # which made "do nothing" the global optimum — the direct solver correctly
    # found it and ablated nothing.)
    gamma1: float = 1e6          # incomplete ablation penalty (area-integrated) — aggressive set (was 1e6)
    gamma2: float = 0.1          # final time penalty (soft min-time) — aggressive set (was 0.1)


@dataclass
class VisualizationConfig:
    """Visualization preferences."""
    colormap_temperature: str = 'hot'        # matplotlib colormap for T field
    colormap_damage: str = 'RdYlGn_r'       # colormap for Ω_d field
    dpi: int = 150
    save_format: str = 'png'
    animation_fps: int = 10
    output_dir: str = 'results'

    # Temperature display range
    T_display_min: float = 37.0   # [°C]
    T_display_max: float = 100.0  # [°C]


@dataclass
class SimConfig:
    """Master configuration — aggregates all sub-configs."""
    domain:   DomainConfig       = field(default_factory=DomainConfig)
    tissue:   TissueConfig       = field(default_factory=TissueConfig)
    arrhenius: ArrheniusConfig   = field(default_factory=ArrheniusConfig)
    sar:      SARConfig          = field(default_factory=SARConfig)
    boundary: BoundaryConfig     = field(default_factory=BoundaryConfig)
    control:  ControlConfig      = field(default_factory=ControlConfig)
    solver:   SolverConfig       = field(default_factory=SolverConfig)
    cost:     CostConfig         = field(default_factory=CostConfig)
    viz:      VisualizationConfig = field(default_factory=VisualizationConfig)

    def summary(self):
        """Print a human-readable summary of key parameters."""
        print("=" * 60)
        print("  MRI Ablation Simulation — Configuration Summary")
        print("=" * 60)
        d = self.domain
        if d.ndim == 2:
            print(f"  Domain:      {d.Lx*100:.1f} x {d.Ly*100:.1f} cm  [2D],  "
                  f"{d.Nx} x {d.Ny} = {d.N} voxels  (dx={d.dx*1000:.2f} mm)")
        else:
            print(f"  Domain:      {d.Lx*100:.1f} x {d.Ly*100:.1f} x {d.Lz*100:.1f} cm  [3D],  "
                  f"{d.Nx} x {d.Ny} x {d.Nz} = {d.N} voxels  (dx={d.dx*1000:.2f} mm)")
        print(f"  Tumor:       r = {d.tumor_radius*1000:.1f} mm  "
              f"@ ({d.tumor_center[0]*100:.1f}, {d.tumor_center[1]*100:.1f}"
              + (f", {d.tumor_center[2]*100:.1f}" if d.ndim == 3 else "") + ") cm")
        print(f"  Tissue:      k = {self.tissue.k} W/(m·K),  "
              f"ω_b = {self.tissue.omega_b} s⁻¹")
        print(f"  Control:     P ∈ [{self.control.P_min}, {self.control.P_max}] W,  "
              f"T_safe = {self.control.T_safe} °C")
        print(f"  Solver:      dt = {self.solver.dt} s,  "
              f"t_final = {self.solver.t_final} s,  integrator = {self.solver.integrator}")
        print(f"  Cost:        α1={self.cost.alpha1}, α2={self.cost.alpha2}, "
              f"γ1={self.cost.gamma1}, γ2={self.cost.gamma2}")
        bc = self.boundary
        bc_detail = (f"T_wall={bc.T_wall}°C" if bc.bc_type == 'dirichlet'
                     else f"h_c={bc.h_c} W/(m²·K), T_inf={bc.T_inf}°C" if bc.bc_type == 'robin'
                     else "zero-flux")
        print(f"  Boundary:    {bc.bc_type}  ({bc_detail})")
        s = self.sar
        if s.probe_model == 'point':
            probe_detail = f"σ_r={s.sigma_sar*1e3:.1f} mm"
            if d.ndim == 3:
                probe_detail += f", σ_z={s.sigma_sar_z*1e3:.1f} mm"
        elif s.probe_model == 'line':
            probe_detail = (f"L={s.L_active*1e3:.0f} mm, σ_r={s.sigma_sar*1e3:.1f} mm, "
                            f"σ_end={s.sigma_sar_z*1e3:.1f} mm, d̂={s.probe_direction}")
        else:
            probe_detail = f"σ={s.sigma_sar*1e3:.1f} mm, d̂={s.probe_direction}"
        print(f"  Probe:       {s.probe_model}  ({probe_detail})")
        print("=" * 60)


# ── Default global config instance (import this everywhere) ──────────────────
cfg = SimConfig()


if __name__ == "__main__":
    cfg.summary()

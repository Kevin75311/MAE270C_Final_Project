"""
control/ocp.py — Optimal Control Problem: cost functional in Bolza form.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  COST FUNCTIONAL  J[u, t_f]  (Bolza form)                          │
    │                                                                      │
    │  J = Φ(x(t_f), t_f)  +  ∫₀^t_f L(x(t), u(t)) dt                 │
    │                                                                      │
    │  Mayer term (terminal cost):                                         │
    │    Φ = γ1 · ∫_{Ω_T} max(0, 1 − Ω_d(t_f))² dr                     │
    │      + γ2 · t_f                                                     │
    │                                                                      │
    │  Lagrangian (running cost):                                          │
    │    L = α1 · ‖u(t)‖²                                                │
    │      + α2 · ∫_{Ω_H} max(0, T(t) − T_safe)² dr                    │
    │      + α3                                                            │
    └──────────────────────────────────────────────────────────────────────┘

All integrals over Ω are approximated as sums over voxels × voxel area.
"""

import numpy as np
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_region_masks, voxel_volume


class CostFunctional:
    """
    Evaluates the Bolza cost J and its components for a given trajectory.
    """

    def __init__(self, cfg: SimConfig = default_cfg):
        self.cfg  = cfg
        self.dA   = voxel_volume(cfg)  # voxel volume [m³] in 3D, area [m²] in 2D

        # Build region masks once
        tumor_2d, healthy_2d, _ = build_region_masks(cfg)
        self.tumor_mask   = tumor_2d.ravel()    # shape (N,)
        self.healthy_mask = healthy_2d.ravel()  # shape (N,)

        # Cost weights (shorthand references)
        self.alpha1 = cfg.cost.alpha1
        self.alpha2 = cfg.cost.alpha2
        self.alpha3 = cfg.cost.alpha3
        self.gamma1 = cfg.cost.gamma1
        self.gamma2 = cfg.cost.gamma2

        # Safety threshold
        self.T_safe = cfg.control.T_safe

    # ── Running cost L ────────────────────────────────────────────────────────

    def energy_penalty(self, u: float) -> float:
        """
        α1 · ‖u‖²   [W²]
        Penalizes excessive power input.
        """
        return self.alpha1 * u**2

    def healthy_tissue_penalty(self, T_flat: np.ndarray) -> float:
        """
        α2 · ∫_{Ω_H} max(0, T − T_safe)² dr   [°C² · m²]
        Penalizes temperatures above T_safe in healthy tissue.
        """
        T_healthy  = T_flat[self.healthy_mask]
        overshoot  = np.maximum(0.0, T_healthy - self.T_safe)
        integral   = np.sum(overshoot**2) * self.dA
        return self.alpha2 * integral

    def time_rate_cost(self) -> float:
        """
        α3  (constant rate — encourages finishing quickly).
        """
        return self.alpha3

    def running_cost(self, T_flat: np.ndarray, u: float) -> float:
        """
        L(x, u) = α1‖u‖²  +  α2·∫_{Ω_H}(T−T_safe)₊²dr  +  α3
        """
        return (self.energy_penalty(u)
                + self.healthy_tissue_penalty(T_flat)
                + self.time_rate_cost())

    # ── Terminal cost Φ ───────────────────────────────────────────────────────

    def incomplete_ablation_penalty(self, Omega_flat: np.ndarray) -> float:
        """
        γ1 · ∫_{Ω_T} max(0, 1 − Ω_d)² dr
        Penalizes tumor voxels where Ω_d < 1 (not fully ablated).
        """
        Omega_tumor = Omega_flat[self.tumor_mask]
        shortfall   = np.maximum(0.0, 1.0 - Omega_tumor)
        integral    = np.sum(shortfall**2) * self.dA
        return self.gamma1 * integral

    def time_penalty(self, t_f: float) -> float:
        """
        γ2 · t_f   (soft minimum-time term).
        """
        return self.gamma2 * t_f

    def terminal_cost(self, Omega_flat: np.ndarray, t_f: float) -> float:
        """
        Φ(x(t_f), t_f) = γ1·∫_{Ω_T}(1−Ω_d)₊²dr  +  γ2·t_f
        """
        return (self.incomplete_ablation_penalty(Omega_flat)
                + self.time_penalty(t_f))

    # ── Total cost over a trajectory ──────────────────────────────────────────

    def total_cost(self, T_history: np.ndarray,
                   Omega_history: np.ndarray,
                   u_history: np.ndarray,
                   t_f: float,
                   dt: float = None) -> dict:
        """
        Integrate J over a complete recorded trajectory.

        Parameters
        ----------
        T_history     : temperature history,  shape (n_steps+1, N)
        Omega_history : damage history,        shape (n_steps+1, N)
        u_history     : control history,       shape (n_steps,)
        t_f           : final time [s]

        Returns
        -------
        dict with keys: 'J_total', 'J_running', 'J_terminal',
                        'J_energy', 'J_healthy', 'J_time_rate', 'J_ablation', 'J_tf'
        """
        if dt is None:
            dt = self.cfg.solver.dt

        n_steps = len(u_history)

        J_energy     = 0.0
        J_healthy    = 0.0
        J_time_rate  = 0.0

        for k in range(n_steps):
            T_k = T_history[k]
            u_k = u_history[k]
            J_energy    += self.energy_penalty(u_k)      * dt
            J_healthy   += self.healthy_tissue_penalty(T_k) * dt
            J_time_rate += self.time_rate_cost()          * dt

        J_running  = J_energy + J_healthy + J_time_rate
        J_ablation = self.incomplete_ablation_penalty(Omega_history[-1])
        J_tf       = self.time_penalty(t_f)
        J_terminal = J_ablation + J_tf
        J_total    = J_running + J_terminal

        return {
            'J_total':     J_total,
            'J_running':   J_running,
            'J_terminal':  J_terminal,
            'J_energy':    J_energy,
            'J_healthy':   J_healthy,
            'J_time_rate': J_time_rate,
            'J_ablation':  J_ablation,
            'J_tf':        J_tf,
        }

    def ablation_completeness(self, Omega_flat: np.ndarray) -> float:
        """
        Fraction of tumor voxels with Ω_d ≥ 1 (fully ablated).
        Returns a value in [0, 1].
        """
        Omega_tumor = Omega_flat[self.tumor_mask]
        return float(np.mean(Omega_tumor >= self.cfg.arrhenius.damage_threshold))

    def max_healthy_temperature(self, T_flat: np.ndarray) -> float:
        """Maximum temperature in healthy tissue [°C]."""
        return float(T_flat[self.healthy_mask].max())

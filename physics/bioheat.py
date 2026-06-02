"""
physics/bioheat.py — Pennes bioheat equation: discretized state equation #1.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  STATE EQUATION 1 — Pennes Bioheat PDE (continuous)                 │
    │                                                                     │
    │  ρ(r)c(r) ∂T/∂t = ∇·(k(r)∇T)                                        │
    │                  − ω_b ρ_b c_b (T − T_b)                            │
    │                  + Q_met(r)                                         │
    │                  + Q_source(r, u, t)                                │
    └─────────────────────────────────────────────────────────────────────┘

    Discretized (finite difference) form:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  STATE EQUATION 1 — Discretized ODE                                 │
    │                                                                     │
    │  ẋ_T = M⁻¹ [ K_d · x_T                   (diffusion)                │
    │             − W_b · (x_T − T_b · 1)       (perfusion cooling)       │
    │             + Q_met                         (metabolic heat)        │
    │             + B_P · P(t) ]                 (ablation source)        │
    └─────────────────────────────────────────────────────────────────────┘

    where the sparse matrices K_d, W_b are assembled in discretization.py.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from scipy import sparse
from config import SimConfig, cfg as default_cfg
from physics.discretization import build_system_matrix, build_perfusion_matrix
from physics.sar_model import get_control_input_vector
from physics.boundary_conditions import apply_boundary_conditions


class BioHeatSolver:
    """
    Stateful solver for the discretized Pennes bioheat PDE.

    Usage
    -----
        solver = BioHeatSolver(cfg, sar_field)
        T = solver.initialize()
        for each timestep:
            T = solver.step(T, P)
    """

    def __init__(self, cfg: SimConfig = default_cfg, sar_field: np.ndarray = None):
        self.cfg = cfg
        N = cfg.domain.N

        # ── Build system matrices (once) ──────────────────────────────────────
        self.M_inv_A, self.M_inv = build_system_matrix(cfg)

        W_b        = build_perfusion_matrix(cfg)
        M_diag_inv = self.M_inv.diagonal()

        # M⁻¹ · W_b  (diagonal × diagonal = diagonal)
        self._M_inv_Wb_diag = M_diag_inv * W_b.diagonal()

        # Constant forcing: M⁻¹ · (W_b·T_b·1  +  Q_met·1)
        T_b   = cfg.tissue.T_blood
        Q_met = cfg.tissue.Q_met
        self._const_rhs = M_diag_inv * (
            self._M_inv_Wb_diag / M_diag_inv * T_b + Q_met
        )
        # Simpler: M⁻¹ · (ω_b ρ_b c_b · T_b + Q_met)
        wb_val = cfg.tissue.omega_b * cfg.tissue.rho_b * cfg.tissue.c_b
        self._const_rhs = M_diag_inv * (wb_val * T_b + Q_met)

        # ── SAR field (precomputed control input map) ─────────────────────────
        self.sar_field = sar_field   # shape (Ny, Nx) or None

        # Boundary condition settings — driven by cfg.boundary
        self.bc_type = cfg.boundary.bc_type

    def initialize(self) -> np.ndarray:
        """Return flat initial temperature vector x_T(0) = T_init everywhere."""
        return np.full(self.cfg.domain.N, self.cfg.tissue.T_init)

    def rhs(self, T_flat: np.ndarray, P: float) -> np.ndarray:
        """
        Evaluate the right-hand side  ẋ_T = f(T, P).

        ┌──────────────────────────────────────────────────────────┐
        │  f(T, P) = M⁻¹·K_d·T  −  M⁻¹·W_b·(T − T_b·1)             │
        │          + M⁻¹·Q_met  +  M⁻¹·B_P·P                       │
        │                                                          │
        │  Term 1: diffusion        (from M_inv_A)                 │
        │  Term 2: perfusion sink   (linear in T)                  │ 
        │  Term 3: constant forcing (metabolic + perfusion offset) │
        │  Term 4: ablation source  (control input)                │
        └──────────────────────────────────────────────────────────┘
        """
        # ── Terms 1+2: Diffusion + perfusion damping  M⁻¹(K_d − W_b)T ─────────
        # M_inv_A = M⁻¹·(K_d − W_b) already encodes the perfusion sink.
        # Do NOT add a separate perfusion term; it would double-count W_b.
        drift = self.M_inv_A @ T_flat

        # ── Term 3: Constant offset  M⁻¹·(W_b·T_b + Q_met) ──────────────────
        const = self._const_rhs

        # ── Term 4: Ablation heat source  M⁻¹·B_P·P ─────────────────────────
        if self.sar_field is not None and P > 0.0:
            Q_source_flat = get_control_input_vector(self.sar_field, P, self.cfg)
            M_diag_inv    = self.M_inv.diagonal()
            source        = M_diag_inv * Q_source_flat
        else:
            source = np.zeros(self.cfg.domain.N)

        return drift + const + source

    def step_euler(self, T_flat: np.ndarray, P: float, dt: float) -> np.ndarray:
        """Forward Euler timestep: T_{k+1} = T_k + dt · f(T_k, P)."""
        return T_flat + dt * self.rhs(T_flat, P)

    def step_rk4(self, T_flat: np.ndarray, P: float, dt: float) -> np.ndarray:
        """
        4th-order Runge-Kutta timestep.

        Assumes P is held constant over [t, t+dt]  (zero-order hold).
        """
        k1 = self.rhs(T_flat,             P)
        k2 = self.rhs(T_flat + 0.5*dt*k1, P)
        k3 = self.rhs(T_flat + 0.5*dt*k2, P)
        k4 = self.rhs(T_flat +     dt*k3, P)
        return T_flat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def step(self, T_flat: np.ndarray, P: float,
             dt: float = None) -> np.ndarray:
        """
        Advance T by one timestep using the configured integrator.
        Applies boundary conditions after the ODE update.
        """
        if dt is None:
            dt = self.cfg.solver.dt

        integrator = self.cfg.solver.integrator
        if integrator == 'euler':
            T_new = self.step_euler(T_flat, P, dt)
        elif integrator == 'rk4':
            T_new = self.step_rk4(T_flat, P, dt)
        else:
            raise ValueError(f"Unknown integrator: '{integrator}'")

        # Apply boundary conditions
        bc = self.cfg.boundary
        if bc.bc_type == 'robin':
            bc_kwargs = {'h_c': bc.h_c, 'T_inf': bc.T_inf}
        elif bc.bc_type == 'dirichlet':
            bc_kwargs = {'T_val': bc.T_wall}
        else:
            bc_kwargs = {}
        T_new = apply_boundary_conditions(
            T_new, bc_type=self.bc_type, cfg=self.cfg, **bc_kwargs
        )

        # Physical temperature floor (no sub-physiological temperatures)
        T_new = np.maximum(T_new, self.cfg.tissue.T_blood - 1.0)

        return T_new


if __name__ == "__main__":
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from physics.sar_model import compute_sar_field
    sar = compute_sar_field()
    solver = BioHeatSolver(cfg=default_cfg, sar_field=sar)
    T = solver.initialize()
    print(f"Initial T:  mean = {T.mean():.2f} °C,  max = {T.max():.2f} °C")
    T1 = solver.step(T, P=30.0)
    print(f"After 1 step (P=30W, dt={default_cfg.solver.dt}s):  "
          f"mean = {T1.mean():.3f} °C,  max = {T1.max():.3f} °C")

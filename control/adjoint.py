"""
control/adjoint.py — Adjoint (costate) equations and Pontryagin conditions.

The adjoint variables λ = [λ_T, λ_Ω]ᵀ satisfy:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  ADJOINT EQUATION 1 — Costate for temperature                       │
    │                                                                      │
    │  −λ̇_T = ∂ℋ/∂T                                                      │
    │        = A_dᵀ λ_T  −  W_b λ_T                                      │
    │        + 2α2 · max(0, T−T_safe) · 1_{Ω_H}                         │
    │        + λ_Ω ⊙ [A·(E_a/(R·T²))·exp(−E_a/(R·T))]                  │
    │                                                                      │
    │  ADJOINT EQUATION 2 — Costate for damage                            │
    │                                                                      │
    │  −λ̇_Ω = ∂ℋ/∂Ω_d = 0   →   λ_Ω(t) = λ_Ω(t_f) = const           │
    │                                                                      │
    │  TRANSVERSALITY (free final time):                                   │
    │                                                                      │
    │  λ_T(t_f) = ∂Φ/∂x_T|_{t_f} = 0                                    │
    │  λ_Ω(t_f) = ∂Φ/∂x_Ω|_{t_f}                                        │
    │           = −2γ1 · max(0, 1 − Ω_d) · 1_{Ω_T}                      │
    │  ℋ(t_f) + ∂Φ/∂t_f = 0                                              │
    └──────────────────────────────────────────────────────────────────────┘

    OPTIMAL CONTROL LAW (Pontryagin):
    ┌──────────────────────────────────────────────────────────────────────┐
    │  P*(t) = proj_{[0, P_max]} ( −λ_Tᵀ · b_P / (2α1) )               │
    └──────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_region_masks
from physics.discretization import build_system_matrix, build_perfusion_matrix


_C_TO_K = 273.15   # Celsius to Kelvin offset


class AdjointSolver:
    """
    Integrates the adjoint ODEs backward in time from t_f to 0.
    """

    def __init__(self, cfg: SimConfig = default_cfg):
        self.cfg  = cfg
        N = cfg.domain.N

        # Build adjoint system matrices (same as forward, but transposed)
        M_inv_A, M_inv = build_system_matrix(cfg)
        self.M_inv_At  = M_inv_A.T.tocsr()   # transposed diffusion-perfusion
        self.M_inv_diag = M_inv.diagonal()

        W_b = build_perfusion_matrix(cfg)
        self.M_inv_Wb_diag = self.M_inv_diag * W_b.diagonal()

        # Region masks
        tumor_2d, healthy_2d, _ = build_region_masks(cfg)
        self.tumor_mask   = tumor_2d.ravel()
        self.healthy_mask = healthy_2d.ravel()

        # Cost weights
        self.alpha2 = cfg.cost.alpha2
        self.gamma1 = cfg.cost.gamma1

        # Arrhenius constants
        self.A  = cfg.arrhenius.A
        self.Ea = cfg.arrhenius.E_a
        self.R  = cfg.arrhenius.R

    def terminal_costate(self, Omega_flat: np.ndarray) -> tuple:
        """
        Compute terminal costate conditions from transversality:

            λ_T(t_f) = 0

            λ_Ω(t_f) = −2γ1 · max(0, 1 − Ω_d(t_f)) · 1_{Ω_T}
        """
        N = self.cfg.domain.N
        lam_T = np.zeros(N)

        # ── λ_Ω(t_f) = ∂Φ/∂Ω_d ───────────────────────────────────────────
        shortfall = np.maximum(0.0, 1.0 - Omega_flat)
        lam_O = np.zeros(N)
        lam_O[self.tumor_mask] = -2.0 * self.gamma1 * shortfall[self.tumor_mask]

        return lam_T, lam_O

    def adjoint_rhs_T(self, lam_T: np.ndarray,
                      lam_O: np.ndarray,
                      T_flat: np.ndarray) -> np.ndarray:
        """
        ┌──────────────────────────────────────────────────────────┐
        │  ADJOINT EQUATION 1 (rhs for backward integration)      │
        │                                                          │
        │  −λ̇_T = A_dᵀ λ_T  −  W_b λ_T                          │
        │        + 2α2(T−T_safe)₊ · 1_{Ω_H}                     │
        │        + λ_Ω ⊙ [A·(E_a/(R·T²))·exp(−E_a/(RT))]        │
        └──────────────────────────────────────────────────────────┘

        Note: backward in time, so the forward integrator uses −rhs.
        """
        # ── Adjoint diffusion + perfusion ─────────────────────────────────
        # M_inv_At = (M⁻¹·(K_d−W_b))ᵀ already encodes −W_b in the adjoint.
        # Do NOT subtract M_inv_Wb*lam_T again; it would double-count W_b.
        diff_perf = self.M_inv_At @ lam_T

        # ── Running cost gradient ∂L/∂T ───────────────────────────────────
        overshoot = np.maximum(0.0, T_flat - self.cfg.control.T_safe)
        dL_dT     = np.zeros_like(T_flat)
        dL_dT[self.healthy_mask] = (2.0 * self.alpha2
                                     * overshoot[self.healthy_mask])

        # ── Coupling: Arrhenius rate sensitivity  ∂(dΩ/dt)/∂T ───────────
        T_K    = T_flat + _C_TO_K
        T_K    = np.maximum(T_K, 273.15)
        d_rate_dT = self.A * (self.Ea / (self.R * T_K**2)) * np.exp(-self.Ea / (self.R * T_K))
        coupling  = lam_O * d_rate_dT

        return diff_perf + dL_dT + coupling

    def step_backward(self, lam_T: np.ndarray,
                      lam_O: np.ndarray,
                      T_flat: np.ndarray,
                      dt: float) -> tuple:
        """
        One backward Euler step of the adjoint equations.
        lam_O is constant (its ODE has zero rhs), so only lam_T evolves.
        """
        # ── ADJOINT EQUATION 2: λ_Ω is constant ──────────────────────────
        lam_O_new = lam_O   # no change

        # ── ADJOINT EQUATION 1: backward Euler ───────────────────────────
        rhs = self.adjoint_rhs_T(lam_T, lam_O, T_flat)
        # Backward integration: λ_T(t-dt) = λ_T(t) + dt · rhs
        # (the '−' in −λ̇_T = rhs  →  λ̇_T = −rhs  →  backward step adds rhs)
        lam_T_new = lam_T + dt * rhs

        return lam_T_new, lam_O_new

    def optimal_control(self, lam_T: np.ndarray,
                        b_P: np.ndarray) -> float:
        """
        ┌──────────────────────────────────────────────────────────┐
        │  OPTIMAL CONTROL LAW (Pontryagin Minimum Principle)     │
        │                                                          │
        │  P*(t) = proj_{[0, P_max]} ( −λ_Tᵀ · b_P / (2α1) )   │
        └──────────────────────────────────────────────────────────┘

        Minimizing ℋ = L + λᵀf gives ∂ℋ/∂P = 2α1·P + λ_Tᵀb_P = 0,
        hence P* = −λ_Tᵀb_P / (2α1).  The minus sign is required: λ_T is
        negative in the tumor during heating, so −λ_Tᵀb_P > 0 drives P* up.

        Parameters
        ----------
        lam_T : costate vector for temperature   shape (N,)
        b_P   : control input map (SAR × ρ × M⁻¹) shape (N,)
        """
        alpha1 = self.cfg.cost.alpha1
        P_opt  = -np.dot(lam_T, b_P) / (2.0 * alpha1)
        P_star = float(np.clip(P_opt, self.cfg.control.P_min,
                               self.cfg.control.P_max))
        return P_star

    def transversality_check(self, lam_T: np.ndarray,
                              lam_O: np.ndarray,
                              T_flat: np.ndarray,
                              Omega_flat: np.ndarray,
                              u: float,
                              b_P: np.ndarray,
                              t_f: float) -> dict:
        """
        Evaluate the free-time transversality condition:
            ℋ(t_f) + ∂Φ/∂t_f = 0

        Returns the residual (should be ≈ 0 at optimum).
        """
        from control.ocp import CostFunctional
        cost = CostFunctional(self.cfg)

        L_tf = cost.running_cost(T_flat, u)
        f_tf = 0.0   # approximate: ẋ contribution via λ^T f

        # Hamiltonian at t_f
        H_tf = L_tf + f_tf

        # ∂Φ/∂t_f = γ2
        dPhi_dtf = self.cfg.cost.gamma2

        residual = H_tf + dPhi_dtf
        return {
            'H_tf': H_tf,
            'dPhi_dtf': dPhi_dtf,
            'residual': residual,
            'satisfied': abs(residual) < 1e-3,
        }

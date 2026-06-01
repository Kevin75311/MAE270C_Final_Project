"""
physics/arrhenius.py — Arrhenius thermal damage model: state equation #2.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  STATE EQUATION 2 — Arrhenius Damage ODE                            │
    │                                                                     │
    │  dΩ_d(r,t)/dt = A · exp(−E_a / (R · T(r,t)))                        │
    │                                                                     │
    │  with initial condition:  Ω_d(r, 0) = 0   ∀ r ∈ Ω                   │
    │                                                                     │
    │  Interpretation:                                                    │
    │    Ω_d < 1  →  reversible thermal stress                            │
    │    Ω_d = 1  →  63% cell death probability (irreversible necrosis)   │
    │    Ω_d > 1  →  complete tissue destruction                          │
    │                                                                     │
    │  Note: T must be in Kelvin (K) for the Arrhenius rate.              │
    └─────────────────────────────────────────────────────────────────────┘

Reference: Henriques & Moritz (1947), Moritz (1947).
           A = 3.1×10⁹⁸ s⁻¹,  E_a = 6.28×10⁵ J/mol  (protein denaturation).
"""

import numpy as np
from config import SimConfig, cfg as default_cfg


# Conversion: Celsius to Kelvin
_C_TO_K = 273.15


class ArrheniusDamage:
    """
    Vectorized Arrhenius damage integrator over the full spatial domain.

    The state x_Omega is a flat vector of shape (N,), one entry per voxel.

    Usage
    -----
        damage = ArrheniusDamage(cfg)
        Omega  = damage.initialize()
        for each timestep:
            Omega = damage.step(Omega, T_flat)
    """

    def __init__(self, cfg: SimConfig = default_cfg):
        self.cfg = cfg
        self.A   = cfg.arrhenius.A
        self.Ea  = cfg.arrhenius.E_a
        self.R   = cfg.arrhenius.R
        self.threshold = cfg.arrhenius.damage_threshold

        # Precompute the exponent numerator (constant)
        self._neg_Ea_over_R = -self.Ea / self.R

    def initialize(self) -> np.ndarray:
        """Return flat initial damage vector Ω_d(0) = 0 everywhere."""
        return np.zeros(self.cfg.domain.N)

    def rate(self, T_flat_celsius: np.ndarray) -> np.ndarray:
        """
        ┌──────────────────────────────────────────────────────────┐
        │  dΩ/dt = A · exp(−E_a / (R · T))                         │
        │                                                          │
        │  T in Kelvin.  Rate units: [s⁻¹]                         │
        └──────────────────────────────────────────────────────────┘

        Parameters
        ----------
        T_flat_celsius : temperature field [°C], shape (N,)

        Returns
        -------
        dOmega_dt : damage rate [s⁻¹],  shape (N,)
        """
        T_K = T_flat_celsius + _C_TO_K   # convert to Kelvin

        # Clip to avoid overflow in exp at very low temperatures
        T_K = np.maximum(T_K, 273.15)    # floor at 0 °C (physiological minimum)

        exponent    = self._neg_Ea_over_R / T_K
        dOmega_dt   = self.A * np.exp(exponent)

        return dOmega_dt

    def step_euler(self, Omega: np.ndarray,
                   T_flat: np.ndarray, dt: float) -> np.ndarray:
        """Forward Euler integration of the damage ODE."""
        return Omega + dt * self.rate(T_flat)

    def step_rk4(self, Omega: np.ndarray,
                 T_flat: np.ndarray, dt: float) -> np.ndarray:
        """
        RK4 integration of the damage ODE.

        T is treated as constant over [t, t+dt]  (operator splitting).
        Since the ODE is autonomous in Omega (rate depends only on T,
        not on Omega itself), RK4 reduces to simple quadrature of the rate.
        """
        r = self.rate(T_flat)
        # RK4 for constant-rate ODE: exact integral = r * dt
        # (kept as RK4 form for consistency with bioheat.py)
        k1 = r
        k2 = r    # rate unchanged within dt (T fixed)
        k3 = r
        k4 = r
        return Omega + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def step(self, Omega: np.ndarray,
             T_flat: np.ndarray,
             dt: float = None) -> np.ndarray:
        """
        Advance Ω_d by one timestep using the configured integrator.
        Clamps Omega ≥ 0 (physical constraint).
        """
        if dt is None:
            dt = self.cfg.solver.dt

        integrator = self.cfg.solver.integrator
        if integrator in ('euler',):
            Omega_new = self.step_euler(Omega, T_flat, dt)
        else:
            Omega_new = self.step_rk4(Omega, T_flat, dt)

        return np.maximum(Omega_new, 0.0)   # damage is non-decreasing

    # ── Analysis helpers ─────────────────────────────────────────────────────

    def is_ablated(self, Omega: np.ndarray) -> np.ndarray:
        """Boolean mask: True where Ω_d ≥ threshold (irreversible necrosis)."""
        return Omega >= self.threshold

    def ablation_fraction(self, Omega: np.ndarray,
                          region_mask: np.ndarray = None) -> float:
        """
        Fraction of voxels (or of a specified region) that are fully ablated.

        Parameters
        ----------
        region_mask : optional boolean mask (N,) — restrict to tumor region.
        """
        ablated = self.is_ablated(Omega)
        if region_mask is not None:
            return ablated[region_mask].mean()
        return ablated.mean()

    def thermal_dose_CEM43(self, T_flat_celsius: np.ndarray,
                           dt: float = None) -> np.ndarray:
        """
        Cumulative Equivalent Minutes at 43 °C (CEM43) — alternative
        clinical thermal dose metric.

            CEM43 += dt/60 · R^(43 − T)    [minutes]
            R = 0.5 for T ≥ 43 °C
            R = 0.25 for T < 43 °C

        Returns incremental CEM43 for a single timestep.
        """
        if dt is None:
            dt = self.cfg.solver.dt
        T = T_flat_celsius
        R = np.where(T >= 43.0, 0.5, 0.25)
        return (dt / 60.0) * R**(43.0 - T)


if __name__ == "__main__":
    damage = ArrheniusDamage()
    Omega  = damage.initialize()
    T_test = np.full(default_cfg.domain.N, 60.0)   # 60 °C uniform field

    rate = damage.rate(T_test)
    print(f"Damage rate at 60°C: {rate[0]:.4e} s⁻¹")

    # How long to reach Omega = 1?
    t_to_ablate = 1.0 / rate[0]
    print(f"Time to Omega=1 at 60°C: {t_to_ablate:.2f} s  ({t_to_ablate/60:.2f} min)")

    # Simulate 10 seconds
    for _ in range(10):
        Omega = damage.step(Omega, T_test)
    print(f"Omega after 10 s at 60°C: {Omega[0]:.4f}")
    print(f"Ablated fraction: {damage.ablation_fraction(Omega):.1%}")

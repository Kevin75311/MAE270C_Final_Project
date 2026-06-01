"""
utils/validators.py — Physical sanity checks on state and control.
"""

import numpy as np
from config import SimConfig, cfg as default_cfg


def check_temperature_physical(T_flat: np.ndarray, cfg: SimConfig = default_cfg):
    """Warn if temperatures are outside a physically plausible range."""
    T_min, T_max = T_flat.min(), T_flat.max()
    if T_min < 20.0:
        print(f"  WARNING: min temperature {T_min:.1f}°C < 20°C (sub-physiological)")
    if T_max > 300.0:
        print(f"  WARNING: max temperature {T_max:.1f}°C > 300°C (tissue vaporization)")
    return T_min, T_max


def check_damage_physical(Omega_flat: np.ndarray):
    """Warn if damage values are negative (should never occur)."""
    if Omega_flat.min() < 0:
        print(f"  WARNING: negative damage values detected — numerical instability?")
    return Omega_flat.min(), Omega_flat.max()


def check_stability_cfl(cfg: SimConfig = default_cfg) -> bool:
    """
    Check the CFL stability condition for explicit time integration:
        dt ≤ dx² / (2 · α_thermal)
    where α = k / (ρ·c) is thermal diffusivity.
    """
    alpha = cfg.tissue.k / (cfg.tissue.rho * cfg.tissue.c)
    dt_max_x = cfg.domain.dx**2 / (2.0 * alpha)
    dt_max_y = cfg.domain.dy**2 / (2.0 * alpha)
    dt_max   = min(dt_max_x, dt_max_y)

    ok = cfg.solver.dt <= dt_max
    if not ok:
        print(f"  WARNING: dt={cfg.solver.dt}s exceeds CFL limit {dt_max:.4f}s  "
              f"— use RK4 or reduce dt")
    else:
        print(f"  CFL OK: dt={cfg.solver.dt}s  ≤  dt_CFL={dt_max:.4f}s")
    return ok, dt_max

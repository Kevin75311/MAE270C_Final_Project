"""
physics/sar_model.py — Microwave SAR (Specific Absorption Rate) model.

Models the volumetric heat deposition from the ablation applicator:

    ┌────────────────────────────────────────────────────────────────┐
    │  Q_source(r, u, t) = SAR(r, q(t)) · P(t)                     │
    │                    = σ(r) · |E(r, q(t))|² · P(t)             │
    │                                                                │
    │  Gaussian approximation:                                       │
    │  SAR(r) = SAR_peak · exp(−|r − r_probe|² / (2σ²))            │
    └────────────────────────────────────────────────────────────────┘

Units:  SAR [W/kg],   Q_source [W/m³]
"""

import numpy as np
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_mesh


def compute_sar_field(probe_position: tuple = None,
                      cfg: SimConfig = default_cfg) -> np.ndarray:
    """
    Compute the normalized SAR spatial distribution b(r) [1/kg].

    This encodes the antenna near-field pattern (precomputed once at the
    start of the simulation).  For a moving probe, call this each time
    the probe position changes.

    Parameters
    ----------
    probe_position : (x_probe, y_probe) in [m].  Defaults to cfg value.

    Returns
    -------
    sar_field : 2D array (Ny, Nx)  — normalized SAR pattern [1/kg]
    """
    if probe_position is None:
        probe_position = cfg.domain.probe_position

    X, Y, _, _ = build_mesh(cfg)
    xp, yp = probe_position

    # ── Gaussian near-field approximation ────────────────────────────────────
    sigma  = cfg.sar.sigma_sar
    dist2  = (X - xp)**2 + (Y - yp)**2
    sar_field = cfg.sar.sar_peak * np.exp(-dist2 / (2.0 * sigma**2))

    return sar_field   # shape (Ny, Nx)


def compute_Q_source(sar_field: np.ndarray,
                     P: float,
                     cfg: SimConfig = default_cfg) -> np.ndarray:
    """
    ┌────────────────────────────────────────────────────────────────┐
    │  STATE EQUATION — Control Input (Heat Source)                 │
    │  Q_source(r, t) = ρ(r) · SAR(r) · P(t)                      │
    │               [W/m³]  = [kg/m³] · [W/kg] · [W/W]            │
    └────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    sar_field : precomputed SAR pattern (Ny, Nx)   [W/kg per unit power]
    P         : applied power                       [W]

    Returns
    -------
    Q_source : volumetric heat deposition (Ny, Nx) [W/m³]
    """
    # Multiply by tissue density to convert SAR [W/kg] → power density [W/m³]
    Q_source = cfg.tissue.rho * sar_field * P
    return Q_source


def get_control_input_vector(sar_field: np.ndarray,
                              P: float,
                              cfg: SimConfig = default_cfg) -> np.ndarray:
    """
    Return the flat control input vector b_P · P  [W/m³]  shape (N,).

    This is the term that enters the discretized state equation:
        ẋ_T += M⁻¹ · (b_P · P)
    """
    Q = compute_Q_source(sar_field, P, cfg)
    return Q.ravel()


if __name__ == "__main__":
    sar = compute_sar_field()
    Q   = compute_Q_source(sar, P=30.0)
    print(f"SAR field:   max = {sar.max():.2e} W/kg,  sum = {sar.sum():.2e}")
    print(f"Q_source:    max = {Q.max():.2e} W/m³  at P = 30 W")

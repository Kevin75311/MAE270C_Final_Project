"""
physics/sar_model.py — Microwave SAR (Specific Absorption Rate) model.

┌────────────────────────────────────────────────────────────────────┐
│  Q_source(r, t) = ρ(r) · SAR(r) · P(t)                           │
│                                                                    │
│  Three probe models (cfg.sar.probe_model):                        │
│                                                                    │
│  'point'  — Isotropic Gaussian (2D) or anisotropic Gaussian (3D)  │
│             SAR = peak · exp(−r_perp²/2σ_r²)                      │
│             · exp(−r_z²/2σ_z²)  [3D only]                        │
│                                                                    │
│  'line'   — Cylindrical active zone + Gaussian end-cap falloff    │
│             SAR = peak · exp(−r_perp²/2σ_r²) · envelope(z_along) │
│             envelope = 1  if |z| ≤ L/2                            │
│                       exp(−(|z|−L/2)²/2σ_z²)  otherwise          │
│                                                                    │
│  'dipole' — sin²(θ) toroidal pattern × Gaussian radial decay      │
│             SAR = peak · sin²(θ) · exp(−r²/2σ²) · e^(1/2)        │
│             (e^(1/2) normalises so peak = sar_peak at r=σ, θ=π/2) │
└────────────────────────────────────────────────────────────────────┘

Units:  SAR [W/kg per W applied],   Q_source [W/m³]
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_mesh


# ── Private helpers: one function per probe model ─────────────────────────────

def _resolve_direction(cfg: SimConfig):
    """
    Return (d_3d, d_2d) — normalised direction vectors.

    d_3d : (3,) unit vector  (always 3D)
    d_2d : (2,) unit vector in x-y plane (projection of d_3d; fallback = y-axis)
    """
    d3 = np.asarray(cfg.sar.probe_direction, dtype=float)
    d3 = d3 / np.linalg.norm(d3)

    d2 = d3[:2].copy()
    norm2 = np.linalg.norm(d2)
    d2 = d2 / norm2 if norm2 > 1e-10 else np.array([0.0, 1.0])

    return d3, d2


def _sar_point(X, Y, Z, xp, yp, zp, cfg: SimConfig) -> np.ndarray:
    """
    Isotropic Gaussian in 2D; anisotropic Gaussian in 3D (σ_z > σ_r).
    Reproduces the original behaviour exactly.
    """
    sigma_r = cfg.sar.sigma_sar
    peak    = cfg.sar.sar_peak

    if cfg.domain.ndim == 2:
        dist2 = (X - xp)**2 + (Y - yp)**2
        return peak * np.exp(-dist2 / (2.0 * sigma_r**2))
    else:
        sigma_z  = cfg.sar.sigma_sar_z
        dist2_xy = (X - xp)**2 + (Y - yp)**2
        dist2_z  = (Z - zp)**2
        return peak * np.exp(-dist2_xy / (2.0 * sigma_r**2)
                             -dist2_z  / (2.0 * sigma_z**2))


def _sar_line(X, Y, Z, xp, yp, zp, cfg: SimConfig) -> np.ndarray:
    """
    Line-source model: cylindrical heating zone of length L_active along
    probe_direction, with Gaussian end-cap rolloff past the active tips.

    Geometry
    --------
    For each voxel r:
      z_along = signed projection of (r − r_p) onto d̂  (axial coordinate)
      r_perp  = perpendicular distance from the needle axis

    SAR formula
    -----------
      SAR = peak · exp(−r_perp² / 2σ_r²) · envelope(z_along)

      envelope(z) = 1                              if |z| ≤ L/2
                  = exp(−(|z| − L/2)² / 2σ_z²)   if |z| > L/2
    """
    d3, d2  = _resolve_direction(cfg)
    sigma_r = cfg.sar.sigma_sar
    sigma_z = cfg.sar.sigma_sar_z
    half_L  = cfg.sar.L_active / 2.0
    peak    = cfg.sar.sar_peak

    if cfg.domain.ndim == 2:
        # If the needle is perpendicular to the 2-D slice (x-y component ≈ 0),
        # every in-plane voxel is inside the active zone (z_along = 0 ≤ L/2)
        # and all displacement is purely perpendicular → isotropic disk.
        if np.linalg.norm(d3[:2]) < 1e-10:
            dist2 = (X - xp)**2 + (Y - yp)**2
            return peak * np.exp(-dist2 / (2.0 * sigma_r**2))
        dx, dy   = X - xp, Y - yp
        z_along  = dx * d2[0] + dy * d2[1]
        # perpendicular displacement = displacement − projection onto d̂
        px = dx - z_along * d2[0]
        py = dy - z_along * d2[1]
        r_perp2  = px**2 + py**2
    else:
        dx, dy, dz = X - xp, Y - yp, Z - zp
        z_along  = dx * d3[0] + dy * d3[1] + dz * d3[2]
        px = dx - z_along * d3[0]
        py = dy - z_along * d3[1]
        pz = dz - z_along * d3[2]
        r_perp2  = px**2 + py**2 + pz**2

    radial  = np.exp(-r_perp2 / (2.0 * sigma_r**2))
    excess  = np.maximum(np.abs(z_along) - half_L, 0.0)
    axial   = np.exp(-excess**2 / (2.0 * sigma_z**2))

    return peak * radial * axial


def _sar_dipole(X, Y, Z, xp, yp, zp, cfg: SimConfig) -> np.ndarray:
    """
    Simplified half-wave dipole: sin²(θ) toroidal pattern × Gaussian decay.

    θ = angle between displacement vector (r − r_p) and the needle axis d̂.
    sin²(θ) = 1 − cos²(θ) = 1 − (d̂ · r̂)²

    The pattern is zero on the needle axis and maximal in the equatorial plane.
    A normalisation factor e^(1/2) ensures the peak SAR equals cfg.sar.sar_peak
    (achieved at r = σ, θ = π/2, i.e., one σ away from the probe, perpendicular).
    """
    d3, d2  = _resolve_direction(cfg)
    sigma   = cfg.sar.sigma_sar
    peak    = cfg.sar.sar_peak
    norm    = np.exp(0.5)   # so max(sin²·exp(-r²/2σ²)) = 1 at r=σ

    if cfg.domain.ndim == 2:
        # If the needle is perpendicular to the 2-D slice, every in-plane
        # direction is equatorial (θ = π/2, sin²θ = 1) → isotropic ring.
        if np.linalg.norm(d3[:2]) < 1e-10:
            r2 = (X - xp)**2 + (Y - yp)**2
            return peak * norm * np.exp(-r2 / (2.0 * sigma**2))
        dx, dy = X - xp, Y - yp
        r2     = dx**2 + dy**2
        dot    = dx * d2[0] + dy * d2[1]
    else:
        dx, dy, dz = X - xp, Y - yp, Z - zp
        r2  = dx**2 + dy**2 + dz**2
        dot = dx * d3[0] + dy * d3[1] + dz * d3[2]

    # Suppress divide-by-zero at r=0; the where result is overridden to 0 there anyway
    with np.errstate(invalid='ignore', divide='ignore'):
        cos2  = np.where(r2 > 1e-20, dot**2 / r2, 0.0)
    sin2  = 1.0 - cos2
    decay = np.exp(-r2 / (2.0 * sigma**2))

    return peak * norm * sin2 * decay


# ── Public API ────────────────────────────────────────────────────────────────

def compute_sar_field(probe_position: tuple = None,
                      cfg: SimConfig = default_cfg) -> np.ndarray:
    """
    Compute the normalised SAR spatial distribution [W/kg per W applied].

    Dispatches to the model selected by cfg.sar.probe_model.

    Parameters
    ----------
    probe_position : 3-tuple (x, y, z) [m].  Defaults to cfg.domain.probe_position.
                     Interpreted as the midpoint of the active zone for 'line'.

    Returns
    -------
    sar_field : shape (Ny, Nx) in 2D  or  (Nz, Ny, Nx) in 3D
    """
    if probe_position is None:
        probe_position = cfg.domain.probe_position

    X, Y, Z, _, _, _ = build_mesh(cfg)
    xp, yp, zp = probe_position

    model = cfg.sar.probe_model
    if model == 'point':
        return _sar_point(X, Y, Z, xp, yp, zp, cfg)
    elif model == 'line':
        return _sar_line(X, Y, Z, xp, yp, zp, cfg)
    elif model == 'dipole':
        return _sar_dipole(X, Y, Z, xp, yp, zp, cfg)
    else:
        raise ValueError(
            f"Unknown probe_model '{model}'. Choose: point, line, dipole")


def probe_needle_endpoints(cfg: SimConfig = default_cfg):
    """
    Return the (start, end) 3D coordinates of the active antenna zone.

    Useful for drawing the needle in visualisations.

    Returns
    -------
    start, end : each a (3,) array [m]
    """
    rp  = np.asarray(cfg.domain.probe_position, dtype=float)
    d3, _ = _resolve_direction(cfg)
    half_L = cfg.sar.L_active / 2.0
    return rp - half_L * d3, rp + half_L * d3


def compute_Q_source(sar_field: np.ndarray,
                     P: float,
                     cfg: SimConfig = default_cfg) -> np.ndarray:
    """
    ┌────────────────────────────────────────────────────────────────┐
    │  STATE EQUATION — Control Input (Heat Source)                 │
    │  Q_source(r, t) = ρ(r) · SAR(r) · P(t)    [W/m³]            │
    └────────────────────────────────────────────────────────────────┘
    """
    return cfg.tissue.rho * sar_field * P


def get_control_input_vector(sar_field: np.ndarray,
                              P: float,
                              cfg: SimConfig = default_cfg) -> np.ndarray:
    """Flat control input vector b_P · P  [W/m³]  shape (N,)."""
    return compute_Q_source(sar_field, P, cfg).ravel()


if __name__ == "__main__":
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    import copy
    base_cfg = default_cfg

    for model in ('point', 'line', 'dipole'):
        c = copy.deepcopy(base_cfg)
        c.sar.probe_model = model
        sar = compute_sar_field(cfg=c)
        Q   = compute_Q_source(sar, P=30.0, cfg=c)
        print(f"[{model:6s}]  SAR max = {sar.max():.3e} W/kg  |  "
              f"Q max = {Q.max():.3e} W/m³")
        if model == 'line':
            s, e = probe_needle_endpoints(c)
            print(f"         needle: {s*100} cm  →  {e*100} cm")

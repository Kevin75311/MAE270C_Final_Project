"""
physics/mesh.py — Domain geometry, voxel grid, and region masks.

Builds the spatial discretization and classifies each voxel as:
  - Ω_T  : tumor (target)
  - Ω_H  : healthy tissue (protected)
  - Ω_M  : safety margin / transition zone

Supports both 2D (cfg.domain.ndim == 2) and 3D (cfg.domain.ndim == 3).
All public functions dispatch on cfg.domain.ndim so callers need no changes
when switching dimensionality — just set cfg.domain.ndim.

Return convention for build_mesh:
    Always returns a 6-tuple  (X, Y, Z, x_vec, y_vec, z_vec)
    In 2D:  Z = None,  z_vec = None
    In 3D:  shapes are (Nz, Ny, Nx)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from config import SimConfig, cfg as default_cfg


def build_mesh(cfg: SimConfig = default_cfg):
    """
    Build the Cartesian voxel grid.

    Returns
    -------
    X, Y, Z   : coordinate arrays [m]
                  2D: X, Y shape (Ny, Nx),       Z = None
                  3D: X, Y, Z shape (Nz, Ny, Nx)
    x_vec, y_vec, z_vec : 1D coordinate vectors
                  2D: z_vec = None
    """
    x_vec = np.linspace(0, cfg.domain.Lx, cfg.domain.Nx)
    y_vec = np.linspace(0, cfg.domain.Ly, cfg.domain.Ny)

    if cfg.domain.ndim == 2:
        X, Y = np.meshgrid(x_vec, y_vec)          # (Ny, Nx)
        return X, Y, None, x_vec, y_vec, None

    # 3D: meshgrid with indexing='ij' → (Nx, Ny, Nz), then transpose to (Nz, Ny, Nx)
    z_vec = np.linspace(0, cfg.domain.Lz, cfg.domain.Nz)
    X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')  # (Nx, Ny, Nz)
    X = X.transpose(2, 1, 0)   # → (Nz, Ny, Nx)
    Y = Y.transpose(2, 1, 0)
    Z = Z.transpose(2, 1, 0)
    return X, Y, Z, x_vec, y_vec, z_vec


def build_region_masks(cfg: SimConfig = default_cfg):
    """
    Classify every voxel into one of three regions:

        Ω = Ω_T ∪ Ω_H ∪ Ω_M,   pairwise disjoint

    Returns
    -------
    tumor_mask   : bool array — True inside Ω_T
                   2D: (Ny, Nx),  3D: (Nz, Ny, Nx)
    healthy_mask : bool array — True inside Ω_H
    margin_mask  : bool array — True inside Ω_M (transition)
    """
    X, Y, Z, _, _, _ = build_mesh(cfg)

    cx, cy, cz = cfg.domain.tumor_center
    r      = cfg.domain.tumor_radius
    margin = cfg.domain.safety_margin

    if cfg.domain.ndim == 2:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    else:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)

    tumor_mask   = dist <= r
    margin_mask  = (dist > r) & (dist <= r + margin)
    healthy_mask = dist > r + margin

    return tumor_mask, healthy_mask, margin_mask


def flatten(field: np.ndarray) -> np.ndarray:
    """N-D field → 1D state vector (N,)  [row-major / C order]."""
    return field.ravel()


def unflatten(vec: np.ndarray, cfg: SimConfig = default_cfg) -> np.ndarray:
    """
    1D state vector (N,) → spatial field.

    2D: reshapes to (Ny, Nx)
    3D: reshapes to (Nz, Ny, Nx)
    """
    if cfg.domain.ndim == 2:
        return vec.reshape(cfg.domain.Ny, cfg.domain.Nx)
    return vec.reshape(cfg.domain.Nz, cfg.domain.Ny, cfg.domain.Nx)


def voxel_volume(cfg: SimConfig = default_cfg) -> float:
    """
    Volume of a single voxel [m³] in 3D, or effective area [m²] in 2D.

    Used for converting voxel sums to spatial integrals in the cost functional.
    """
    if cfg.domain.ndim == 2:
        return cfg.domain.dx * cfg.domain.dy
    return cfg.domain.dx * cfg.domain.dy * cfg.domain.dz


# Backward-compatible alias (was voxel_area)
def voxel_area(cfg: SimConfig = default_cfg) -> float:
    return voxel_volume(cfg)


if __name__ == "__main__":
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from config import cfg as default_cfg

    for ndim in (2, 3):
        default_cfg.domain.ndim = ndim
        X, Y, Z, xv, yv, zv = build_mesh(default_cfg)
        tm, hm, mm = build_region_masks(default_cfg)
        shape_str = f"{X.shape}" if ndim == 2 else f"{X.shape}"
        print(f"\n── ndim={ndim} ─────────────────────────────────")
        print(f"  Mesh shape:    {shape_str}")
        print(f"  N voxels:      {default_cfg.domain.N}")
        print(f"  Tumor voxels:  {tm.sum()}   ({100*tm.mean():.1f}%)")
        print(f"  Healthy:       {hm.sum()}   ({100*hm.mean():.1f}%)")
        print(f"  Margin:        {mm.sum()}   ({100*mm.mean():.1f}%)")
        print(f"  Voxel vol:     {voxel_volume(default_cfg):.3e} m{'³' if ndim==3 else '²'}")

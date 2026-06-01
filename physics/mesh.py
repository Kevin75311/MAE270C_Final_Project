"""
physics/mesh.py — Domain geometry, voxel grid, and region masks.

Builds the spatial discretization and classifies each voxel as:
  - Ω_T  : tumor (target)
  - Ω_H  : healthy tissue (protected)
  - Ω_M  : safety margin / transition zone
"""

import numpy as np
from config import SimConfig, cfg as default_cfg


def build_mesh(cfg: SimConfig = default_cfg):
    """
    Build the 2D Cartesian voxel grid.

    Returns
    -------
    X, Y     : 2D coordinate arrays [m],  shape (Ny, Nx)
    x_vec    : 1D x-coordinates           shape (Nx,)
    y_vec    : 1D y-coordinates           shape (Ny,)
    """
    x_vec = np.linspace(0, cfg.domain.Lx, cfg.domain.Nx)
    y_vec = np.linspace(0, cfg.domain.Ly, cfg.domain.Ny)
    X, Y = np.meshgrid(x_vec, y_vec)   # shape: (Ny, Nx)
    return X, Y, x_vec, y_vec


def build_region_masks(cfg: SimConfig = default_cfg):
    """
    Classify every voxel into one of three regions:

        Ω = Ω_T ∪ Ω_H ∪ Ω_M,   pairwise disjoint

    Returns
    -------
    tumor_mask   : bool array (Ny, Nx)  — True inside Ω_T
    healthy_mask : bool array (Ny, Nx)  — True inside Ω_H
    margin_mask  : bool array (Ny, Nx)  — True inside Ω_M (transition)
    """
    X, Y, _, _ = build_mesh(cfg)

    cx, cy = cfg.domain.tumor_center
    r      = cfg.domain.tumor_radius
    margin = cfg.domain.safety_margin

    # Euclidean distance from each voxel to tumor center
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    tumor_mask   = dist <= r
    margin_mask  = (dist > r) & (dist <= r + margin)
    healthy_mask = dist > r + margin

    return tumor_mask, healthy_mask, margin_mask


def flatten(field_2d: np.ndarray) -> np.ndarray:
    """2D field (Ny, Nx) → 1D state vector (N,)  [row-major]."""
    return field_2d.ravel()


def unflatten(vec: np.ndarray, cfg: SimConfig = default_cfg) -> np.ndarray:
    """1D state vector (N,) → 2D field (Ny, Nx)."""
    return vec.reshape(cfg.domain.Ny, cfg.domain.Nx)


def voxel_area(cfg: SimConfig = default_cfg) -> float:
    """Area of a single voxel [m²]."""
    return cfg.domain.dx * cfg.domain.dy


if __name__ == "__main__":
    X, Y, xv, yv = build_mesh()
    tm, hm, mm = build_region_masks()
    print(f"Grid:          {X.shape[1]} x {X.shape[0]} = {X.size} voxels")
    print(f"Tumor voxels:  {tm.sum()}   ({100*tm.mean():.1f}%)")
    print(f"Healthy voxels:{hm.sum()}   ({100*hm.mean():.1f}%)")
    print(f"Margin voxels: {mm.sum()}   ({100*mm.mean():.1f}%)")

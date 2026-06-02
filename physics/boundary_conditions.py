"""
physics/boundary_conditions.py — Boundary condition enforcement.

Supports three BC types on ∂Ω:
  - Dirichlet : T = T_val          (fixed temperature)
  - Neumann   : k ∂T/∂n = 0        (zero flux — already built into K_d via
                                     ghost-cell correction in discretization.py)
  - Robin     : k ∂T/∂n + h(T−T∞) = 0   (convective cooling)

Call apply_boundary_conditions() each timestep AFTER the ODE update.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from config import SimConfig, cfg as default_cfg
from physics.mesh import unflatten, flatten


def _boundary_indices(cfg: SimConfig):
    """
    Return flat indices of all boundary voxels.

    2D: outer ring of the (Ny, Nx) grid — 4 edges
    3D: outer shell of the (Nz, Ny, Nx) grid — 6 faces
    """
    Nx, Ny = cfg.domain.Nx, cfg.domain.Ny
    idx = []

    if cfg.domain.ndim == 2:
        for j in range(Ny):
            for i in range(Nx):
                if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                    idx.append(j * Nx + i)
    else:
        Nz = cfg.domain.Nz
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    if (i == 0 or i == Nx-1 or
                            j == 0 or j == Ny-1 or
                            k == 0 or k == Nz-1):
                        idx.append(k * Ny * Nx + j * Nx + i)

    return np.array(idx, dtype=int)


def apply_dirichlet(T_flat: np.ndarray,
                    T_val: float,
                    cfg: SimConfig = default_cfg) -> np.ndarray:
    """
    Dirichlet BC:  T(r, t) = T_val   for r ∈ ∂Ω_D

    Pins the outer boundary voxels to a fixed temperature (e.g. body
    temperature or large-vessel temperature).
    """
    T_out = T_flat.copy()
    bc_idx = _boundary_indices(cfg)
    T_out[bc_idx] = T_val
    return T_out


def apply_neumann(T_flat: np.ndarray,
                  cfg: SimConfig = default_cfg) -> np.ndarray:
    """
    Neumann BC:  k ∂T/∂n = 0   (insulated / symmetry plane)

    Already incorporated into the FD Laplacian via ghost-cell correction
    in discretization.py.  This function is a no-op placeholder for
    explicitness in the solver loop.
    """
    return T_flat


def apply_robin(T_flat: np.ndarray,
                h_c: float,
                T_inf: float,
                cfg: SimConfig = default_cfg) -> np.ndarray:
    """
    Robin (convective) BC:  k ∂T/∂n + h_c·(T − T_inf) = 0

    One-sided FD approximation at each boundary face:
        T_boundary = (k·T_interior + h_c·dξ·T_inf) / (k + h_c·dξ)

    2D: 4 edges (left/right/bottom/top)
    3D: 6 faces (±x, ±y, ±z)

    Parameters
    ----------
    h_c   : convective heat transfer coefficient  [W/(m²·K)]
    T_inf : ambient / fluid temperature           [°C]
    """
    T_nd  = unflatten(T_flat, cfg)    # (Ny,Nx) in 2D  or  (Nz,Ny,Nx) in 3D
    k     = cfg.tissue.k
    dx    = cfg.domain.dx
    dy    = cfg.domain.dy

    denom_x = k + h_c * dx
    denom_y = k + h_c * dy

    if cfg.domain.ndim == 2:
        # Left / right walls  (x dimension = axis 1)
        T_nd[:, 0]    = (k * T_nd[:, 1]    + h_c * dx * T_inf) / denom_x
        T_nd[:, -1]   = (k * T_nd[:, -2]   + h_c * dx * T_inf) / denom_x
        # Bottom / top walls  (y dimension = axis 0)
        T_nd[0,  :]   = (k * T_nd[1,  :]   + h_c * dy * T_inf) / denom_y
        T_nd[-1, :]   = (k * T_nd[-2, :]   + h_c * dy * T_inf) / denom_y
    else:
        dz     = cfg.domain.dz
        denom_z = k + h_c * dz
        # x faces  (axis 2)
        T_nd[:, :, 0]  = (k * T_nd[:, :, 1]  + h_c * dx * T_inf) / denom_x
        T_nd[:, :, -1] = (k * T_nd[:, :, -2] + h_c * dx * T_inf) / denom_x
        # y faces  (axis 1)
        T_nd[:, 0, :]  = (k * T_nd[:, 1, :]  + h_c * dy * T_inf) / denom_y
        T_nd[:, -1, :] = (k * T_nd[:, -2, :] + h_c * dy * T_inf) / denom_y
        # z faces  (axis 0)
        T_nd[0, :, :]  = (k * T_nd[1, :, :]  + h_c * dz * T_inf) / denom_z
        T_nd[-1, :, :] = (k * T_nd[-2, :, :] + h_c * dz * T_inf) / denom_z

    return flatten(T_nd)


def apply_boundary_conditions(T_flat: np.ndarray,
                               bc_type: str = 'dirichlet',
                               cfg: SimConfig = default_cfg,
                               **kwargs) -> np.ndarray:
    """
    Dispatcher: apply the selected boundary condition type.

    Parameters
    ----------
    bc_type : 'dirichlet' | 'neumann' | 'robin'
    kwargs  : passed to the specific BC function
                dirichlet → T_val=37.0
                robin     → h_c=..., T_inf=...
    """
    if bc_type == 'dirichlet':
        T_val = kwargs.get('T_val', cfg.tissue.T_blood)
        return apply_dirichlet(T_flat, T_val, cfg)
    elif bc_type == 'neumann':
        return apply_neumann(T_flat, cfg)
    elif bc_type == 'robin':
        return apply_robin(T_flat, kwargs['h_c'], kwargs['T_inf'], cfg)
    else:
        raise ValueError(f"Unknown BC type: '{bc_type}'. "
                         "Choose from 'dirichlet', 'neumann', 'robin'.")

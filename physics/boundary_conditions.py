"""
physics/boundary_conditions.py — Boundary condition enforcement.

Supports three BC types on ∂Ω:
  - Dirichlet : T = T_val          (fixed temperature)
  - Neumann   : k ∂T/∂n = 0        (zero flux — already built into K_d via
                                     ghost-cell correction in discretization.py)
  - Robin     : k ∂T/∂n + h(T−T∞) = 0   (convective cooling)

Call apply_boundary_conditions() each timestep AFTER the ODE update.
"""

import numpy as np
from config import SimConfig, cfg as default_cfg
from physics.mesh import unflatten, flatten


def _boundary_indices(cfg: SimConfig):
    """Return flat indices of all boundary voxels (outer ring of the grid)."""
    Nx, Ny = cfg.domain.Nx, cfg.domain.Ny
    idx = []
    for j in range(Ny):
        for i in range(Nx):
            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                idx.append(j * Nx + i)
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

    Implemented as a correction to the boundary voxel temperatures via
    a one-sided finite-difference approximation:

        T_boundary_new = (k·T_interior + h_c·dx·T_inf) / (k + h_c·dx)

    Parameters
    ----------
    h_c   : convective heat transfer coefficient  [W/(m²·K)]
    T_inf : ambient / fluid temperature           [°C]
    """
    T_2d  = unflatten(T_flat, cfg)
    k     = cfg.tissue.k
    dx    = cfg.domain.dx
    dy    = cfg.domain.dy
    Nx, Ny = cfg.domain.Nx, cfg.domain.Ny

    denom_x = k + h_c * dx
    denom_y = k + h_c * dy

    # Left / right walls (x = 0 and x = Lx)
    T_2d[:, 0]    = (k * T_2d[:, 1]    + h_c * dx * T_inf) / denom_x
    T_2d[:, -1]   = (k * T_2d[:, -2]   + h_c * dx * T_inf) / denom_x

    # Bottom / top walls (y = 0 and y = Ly)
    T_2d[0,  :]   = (k * T_2d[1,  :]   + h_c * dy * T_inf) / denom_y
    T_2d[-1, :]   = (k * T_2d[-2, :]   + h_c * dy * T_inf) / denom_y

    return flatten(T_2d)


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

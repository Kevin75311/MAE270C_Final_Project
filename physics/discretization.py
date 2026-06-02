"""
physics/discretization.py — Finite-difference spatial discretization.

Assembles the sparse matrices that appear in the discretized bioheat PDE:

    M · ẋ_T = K_d · x_T  −  W_b · (x_T − T_b·1)  +  Q_met  +  B_P · P(t)

where:
    M   = diag(ρ_i · c_i)              mass matrix           [N × N]
    K_d = sparse FD Laplacian × k      diffusion matrix      [N × N]
    W_b = diag(ω_b · ρ_b · c_b)       perfusion matrix      [N × N]
    B_P = SAR(r) vector                control input map     [N]
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from scipy import sparse
from config import SimConfig, cfg as default_cfg


def build_mass_matrix(cfg: SimConfig = default_cfg) -> sparse.csr_matrix:
    """
    M = diag(ρ · c)  [kg·J / (m³·K)]

    Uniform tissue assumed; heterogeneous extension: pass spatially varying
    rho_field and c_field arrays shaped (Ny, Nx).
    """
    N = cfg.domain.N
    diag_vals = np.full(N, cfg.tissue.rho * cfg.tissue.c)
    return sparse.diags(diag_vals, format='csr')


def _build_1d_laplacian(n: int, r: float) -> sparse.csr_matrix:
    """
    1D tridiagonal FD Laplacian for n nodes, diffusion coefficient r = k/dξ².
    Neumann (zero-flux) BCs applied via ghost-cell correction at both ends.
    """
    main = np.full(n, -2.0 * r)
    off  = np.full(n - 1, r)
    L    = sparse.diags([off, main, off], [-1, 0, 1], shape=(n, n)).tolil()
    L[0,  0]  += r   # ghost-cell: zero-flux at ξ = 0
    L[-1, -1] += r   # ghost-cell: zero-flux at ξ = L
    return L.tocsr()


def build_diffusion_matrix(cfg: SimConfig = default_cfg) -> sparse.csr_matrix:
    """
    K_d = k · L_FD   [W / (m² · K)]

    Dispatches on cfg.domain.ndim:

    2D — 5-point stencil via Kronecker sum of two 1D operators:
        K_d = Iy ⊗ Lx  +  Ly ⊗ Ix
        State ordering: row-major (Ny, Nx) → flat index j*Nx + i

    3D — 7-point stencil via Kronecker sum of three 1D operators:
        K_d = Iz ⊗ Iy ⊗ Lx  +  Iz ⊗ Ly ⊗ Ix  +  Lz ⊗ Iy ⊗ Ix
        State ordering: row-major (Nz, Ny, Nx) → flat index k*Ny*Nx + j*Nx + i
    """
    Nx, Ny = cfg.domain.Nx, cfg.domain.Ny
    k      = cfg.tissue.k

    Lx_1d = _build_1d_laplacian(Nx, k / cfg.domain.dx**2)
    Ly_1d = _build_1d_laplacian(Ny, k / cfg.domain.dy**2)

    Ix = sparse.eye(Nx, format='csr')
    Iy = sparse.eye(Ny, format='csr')

    if cfg.domain.ndim == 2:
        # ── 5-point stencil ───────────────────────────────────────────────────
        K_d = sparse.kron(Iy, Lx_1d) + sparse.kron(Ly_1d, Ix)
    else:
        # ── 7-point stencil ───────────────────────────────────────────────────
        Nz    = cfg.domain.Nz
        Lz_1d = _build_1d_laplacian(Nz, k / cfg.domain.dz**2)
        Iz    = sparse.eye(Nz, format='csr')
        Ixy   = sparse.eye(Nx * Ny, format='csr')

        K_d = (sparse.kron(Iz, sparse.kron(Iy, Lx_1d))   # x-diffusion
             + sparse.kron(Iz, sparse.kron(Ly_1d, Ix))    # y-diffusion
             + sparse.kron(Lz_1d, Ixy))                   # z-diffusion

    return K_d.tocsr()


def build_perfusion_matrix(cfg: SimConfig = default_cfg) -> sparse.csr_matrix:
    """
    W_b = diag(ω_b · ρ_b · c_b)   [W / (m³ · K)]

    Blood perfusion acts as a linear damping term in the bioheat equation:
        −W_b · (T − T_b · 1)
    """
    N = cfg.domain.N
    wb = cfg.tissue.omega_b * cfg.tissue.rho_b * cfg.tissue.c_b
    return sparse.diags(np.full(N, wb), format='csr')


def build_system_matrix(cfg: SimConfig = default_cfg):
    """
    Assemble the combined system matrix A for:

        M · ẋ_T = A · x_T  +  rhs_const  +  B_P · P(t)

    where  A = K_d − W_b  (diffusion minus perfusion damping).

    Returns
    -------
    M_inv_A : M⁻¹ · A   — the effective drift matrix (M is diagonal → cheap inversion)
    M_inv   : M⁻¹        — diagonal inverse mass matrix
    """
    M   = build_mass_matrix(cfg)
    K_d = build_diffusion_matrix(cfg)
    W_b = build_perfusion_matrix(cfg)

    A = K_d - W_b   # combined diffusion-perfusion matrix

    # M is diagonal → M⁻¹ is just reciprocal of diagonal
    M_diag_inv = 1.0 / M.diagonal()
    M_inv      = sparse.diags(M_diag_inv, format='csr')
    M_inv_A    = M_inv @ A

    return M_inv_A, M_inv


if __name__ == "__main__":
    M_inv_A, M_inv = build_system_matrix()
    print(f"System matrix shape:   {M_inv_A.shape}")
    print(f"Non-zeros in M_inv_A:  {M_inv_A.nnz}")
    print(f"Spectral estimate:     max diagonal = {M_inv_A.diagonal().max():.4f}")

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


def build_diffusion_matrix(cfg: SimConfig = default_cfg) -> sparse.csr_matrix:
    """
    K_d = k · L_FD   [W / (m² · K)]

    L_FD is the 2D finite-difference Laplacian on a uniform (dx, dy) grid
    with Neumann (zero-flux) boundary conditions on all edges.

    Uses the 5-point stencil:
        L_FD[i,j] = (T[i+1,j] + T[i-1,j] − 2T[i,j]) / dx²
                  + (T[i,j+1] + T[i,j-1] − 2T[i,j]) / dy²
    """
    Nx, Ny = cfg.domain.Nx, cfg.domain.Ny
    N  = cfg.domain.N
    k  = cfg.tissue.k
    dx, dy = cfg.domain.dx, cfg.domain.dy

    rx = k / dx**2   # x-direction diffusion coefficient
    ry = k / dy**2   # y-direction diffusion coefficient

    # Build 1D tridiagonal operators for x and y separately, then Kronecker-sum
    # ── x-direction (along columns within each row) ──────────────────────────
    main_x = np.full(Nx, -2.0 * rx)
    off_x  = np.full(Nx - 1, rx)
    Lx_1d  = sparse.diags([off_x, main_x, off_x], [-1, 0, 1], shape=(Nx, Nx))

    # Neumann BC: zero flux at x=0 and x=Lx
    Lx_1d = Lx_1d.tolil()
    Lx_1d[0,  0]  += rx   # ghost-cell correction
    Lx_1d[-1, -1] += rx
    Lx_1d = Lx_1d.tocsr()

    # ── y-direction (along rows) ──────────────────────────────────────────────
    main_y = np.full(Ny, -2.0 * ry)
    off_y  = np.full(Ny - 1, ry)
    Ly_1d  = sparse.diags([off_y, main_y, off_y], [-1, 0, 1], shape=(Ny, Ny))

    Ly_1d = Ly_1d.tolil()
    Ly_1d[0,  0]  += ry
    Ly_1d[-1, -1] += ry
    Ly_1d = Ly_1d.tocsr()

    # ── 2D Laplacian via Kronecker sum ────────────────────────────────────────
    Ix = sparse.eye(Nx, format='csr')
    Iy = sparse.eye(Ny, format='csr')
    K_d = sparse.kron(Iy, Lx_1d) + sparse.kron(Ly_1d, Ix)

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

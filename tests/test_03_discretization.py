"""
tests/test_03_discretization.py — Discretization matrix tests.

Checks shape, symmetry of K_d, diagonal values of M and W_b,
and that M_inv_A has negative diagonal (stable system).

Run standalone:
    python tests/test_03_discretization.py
Run with pytest:
    pytest tests/test_03_discretization.py -v
"""
import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tests._helpers import make_cfg, save_fig
from physics.mesh import build_mesh, unflatten
from physics.discretization import (build_mass_matrix, build_diffusion_matrix,
                                     build_perfusion_matrix, build_system_matrix)


def test_discretization_matrices(ndim=2):
    """
    Checks shape, symmetry of K_d, diagonal values of M and W_b,
    and that M_inv_A diagonal is negative (stable linear system).

    Visual: K_d sparsity pattern, M_inv_A diagonal field, K_d row sums.
    """
    cfg = make_cfg(ndim)
    N = cfg.domain.N

    M       = build_mass_matrix(cfg)
    K_d     = build_diffusion_matrix(cfg)
    W_b     = build_perfusion_matrix(cfg)
    M_inv_A, M_inv = build_system_matrix(cfg)

    # ── Shape assertions ──────────────────────────────────────────────────
    for mat, name in [(M, 'M'), (K_d, 'K_d'), (W_b, 'W_b'), (M_inv_A, 'M_inv_A')]:
        assert mat.shape == (N, N), f"{name}.shape={mat.shape}, expected ({N},{N})"

    # ── Mass matrix ───────────────────────────────────────────────────────
    np.testing.assert_allclose(M.diagonal(),
                                cfg.tissue.rho * cfg.tissue.c, rtol=1e-10)

    # ── Diffusion matrix ──────────────────────────────────────────────────
    diff_sym = (K_d - K_d.T).data
    if len(diff_sym) > 0:
        assert np.max(np.abs(diff_sym)) < 1e-8, \
            f"K_d not symmetric; max |K_d - K_d^T| = {np.max(np.abs(diff_sym)):.2e}"

    assert np.all(K_d.diagonal() <= 0), "K_d diagonal must be ≤ 0"

    # ── Perfusion matrix ──────────────────────────────────────────────────
    np.testing.assert_allclose(W_b.diagonal(),
                                cfg.tissue.omega_b * cfg.tissue.rho_b * cfg.tissue.c_b,
                                rtol=1e-10)

    # ── System stability ──────────────────────────────────────────────────
    assert np.all(M_inv_A.diagonal() < 0), \
        "M_inv_A diagonal must be < 0 (stable system)"

    print(f"  ndim={ndim}  N={N}  K_d nnz={K_d.nnz}")
    print(f"  K_d diagonal: [{K_d.diagonal().min():.3e}, {K_d.diagonal().max():.3e}]")
    print(f"  M_inv_A diagonal: [{M_inv_A.diagonal().min():.3e}, "
          f"{M_inv_A.diagonal().max():.3e}]")

    # ── Figure ────────────────────────────────────────────────────────────
    X, Y, Z, xv, yv, zv = build_mesh(cfg)

    if ndim == 2:
        X2, Y2 = X, Y
        diag_2d  = M_inv_A.diagonal().reshape(cfg.domain.Ny, cfg.domain.Nx)
        rowsum2d = np.array(K_d.sum(axis=1)).ravel().reshape(cfg.domain.Ny, cfg.domain.Nx)
    else:
        iz = cfg.domain.Nz // 2
        X2 = X[iz]; Y2 = Y[iz]
        diag_nd  = unflatten(M_inv_A.diagonal(), cfg)
        rowsum_nd = unflatten(np.array(K_d.sum(axis=1)).ravel(), cfg)
        diag_2d   = diag_nd[iz]
        rowsum2d  = rowsum_nd[iz]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].spy(K_d, markersize=max(0.3, 1.5 - ndim*0.5), color='navy')
    axes[0].set_title(f'K_d sparsity  ({K_d.nnz} nnz)', fontsize=11)
    axes[0].set_xlabel('column'); axes[0].set_ylabel('row')

    im2 = axes[1].pcolormesh(X2*100, Y2*100, diag_2d, cmap='RdBu_r', shading='auto')
    fig.colorbar(im2, ax=axes[1], label='(M⁻¹A)_ii  [s⁻¹]')
    z_lbl = f'  z={zv[iz]*100:.1f}cm' if ndim == 3 else ''
    axes[1].set_title(f'M⁻¹A diagonal{z_lbl}', fontsize=11)
    axes[1].set_xlabel('x [cm]'); axes[1].set_ylabel('y [cm]')
    axes[1].set_aspect('equal')

    im3 = axes[2].pcolormesh(X2*100, Y2*100, rowsum2d, cmap='bwr', shading='auto')
    fig.colorbar(im3, ax=axes[2], label='Row sum of K_d')
    axes[2].set_title(f'K_d row sums{z_lbl}', fontsize=11)
    axes[2].set_xlabel('x [cm]'); axes[2].set_ylabel('y [cm]')
    axes[2].set_aspect('equal')

    fig.suptitle(f'Test 03: Discretization Matrices  (ndim={ndim})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'03_discretization_{ndim}d.png', cfg)


# ── pytest entry points ───────────────────────────────────────────────────────

def test_discretization_matrices_2d():
    test_discretization_matrices(ndim=2)

def test_discretization_matrices_3d():
    test_discretization_matrices(ndim=3)


if __name__ == '__main__':
    from tests._helpers import run_suite
    run_suite([test_discretization_matrices])

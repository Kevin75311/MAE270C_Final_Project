"""
tests/test_04_boundary_conditions.py — Boundary condition tests.

Checks Dirichlet pins boundary voxels, Neumann is identity,
Robin interpolates between interior and ambient.

Run standalone:
    python tests/test_04_boundary_conditions.py
Run with pytest:
    pytest tests/test_04_boundary_conditions.py -v
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
from matplotlib.colors import Normalize

from tests._helpers import make_cfg, save_fig
from physics.mesh import build_mesh, unflatten
from physics.boundary_conditions import (apply_dirichlet, apply_neumann,
                                          apply_robin)


def test_boundary_conditions(ndim=2):
    """
    Checks:
      - Dirichlet: all boundary voxels pinned to T_val, interior unchanged.
      - Neumann: identity operation (no change).
      - Robin: boundary cooled toward T_inf without over-cooling.

    Visual: T field before/after each BC type (axial slice for 3D).
    """
    cfg = make_cfg(ndim)
    N  = cfg.domain.N
    Nx, Ny, Nz = cfg.domain.Nx, cfg.domain.Ny, cfg.domain.Nz
    T_warm = np.full(N, 60.0)
    T_val  = cfg.tissue.T_blood     # 37°C

    # ── Dirichlet ─────────────────────────────────────────────────────────
    T_dir  = apply_dirichlet(T_warm, T_val, cfg)
    T_dir_nd = unflatten(T_dir, cfg)

    if ndim == 2:
        assert np.all(T_dir_nd[0,  :] == T_val),  "Dirichlet top row"
        assert np.all(T_dir_nd[-1, :] == T_val),  "Dirichlet bottom row"
        assert np.all(T_dir_nd[:,  0] == T_val),  "Dirichlet left col"
        assert np.all(T_dir_nd[:, -1] == T_val),  "Dirichlet right col"
        interior_val = T_dir_nd[Ny//2, Nx//2]
    else:
        # All 6 faces pinned
        assert np.all(T_dir_nd[:, :,  0] == T_val), "Dirichlet x=0 face"
        assert np.all(T_dir_nd[:, :, -1] == T_val), "Dirichlet x=Nx face"
        assert np.all(T_dir_nd[:,  0, :] == T_val), "Dirichlet y=0 face"
        assert np.all(T_dir_nd[:, -1, :] == T_val), "Dirichlet y=Ny face"
        assert np.all(T_dir_nd[ 0, :, :] == T_val), "Dirichlet z=0 face"
        assert np.all(T_dir_nd[-1, :, :] == T_val), "Dirichlet z=Nz face"
        interior_val = T_dir_nd[Nz//2, Ny//2, Nx//2]

    assert interior_val == 60.0, "Interior voxel must remain unchanged"

    # ── Neumann ───────────────────────────────────────────────────────────
    T_neu = apply_neumann(T_warm, cfg)
    np.testing.assert_array_equal(T_neu, T_warm, "Neumann must be identity")

    # ── Robin ─────────────────────────────────────────────────────────────
    h_c, T_inf = 50.0, 20.0
    T_rob    = apply_robin(T_warm, h_c=h_c, T_inf=T_inf, cfg=cfg)
    T_rob_nd = unflatten(T_rob, cfg)

    if ndim == 2:
        bnd = np.concatenate([T_rob_nd[0, :], T_rob_nd[-1, :],
                               T_rob_nd[:, 0], T_rob_nd[:, -1]])
    else:
        bnd = np.concatenate([T_rob_nd[:, :, 0].ravel(), T_rob_nd[:, :, -1].ravel(),
                               T_rob_nd[:, 0, :].ravel(), T_rob_nd[:, -1, :].ravel(),
                               T_rob_nd[0, :, :].ravel(), T_rob_nd[-1, :, :].ravel()])

    assert np.all(bnd < 60.0),  "Robin should cool boundary below 60°C"
    assert np.all(bnd > T_inf), "Robin should not cool boundary below T_inf"

    print(f"  ndim={ndim}")
    print(f"  Dirichlet: boundary = {T_val}°C, interior = {interior_val}°C [OK]")
    print(f"  Neumann: identity [OK]")
    print(f"  Robin: boundary range = [{bnd.min():.1f}, {bnd.max():.1f}] °C [OK]")

    # ── Figure ────────────────────────────────────────────────────────────
    X, Y, Z, xv, yv, zv = build_mesh(cfg)
    norm = Normalize(vmin=15, vmax=65)

    if ndim == 2:
        X2, Y2 = X, Y
        def _slice2d(T_flat):
            return unflatten(T_flat, cfg)
    else:
        iz = Nz // 2
        X2, Y2 = X[iz], Y[iz]
        def _slice2d(T_flat):
            return unflatten(T_flat, cfg)[iz]

    labels = ['Input (60°C uniform)', f'Dirichlet\n(T_boundary={T_val}°C)',
              'Neumann (identity)', f'Robin\n(h_c={h_c}, T_inf={T_inf}°C)']
    fields = [T_warm, T_dir, T_neu, T_rob]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, T_f, label in zip(axes, fields, labels):
        im = ax.pcolormesh(X2*100, Y2*100, _slice2d(T_f),
                           cmap='hot', norm=norm, shading='auto')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('x [cm]'); ax.set_aspect('equal')
    axes[0].set_ylabel('y [cm]')
    z_note = f'  (axial z={zv[iz]*100:.1f}cm)' if ndim == 3 else ''
    fig.colorbar(im, ax=axes, label=f'Temperature [°C]{z_note}',
                 fraction=0.01, pad=0.04)

    fig.suptitle(f'Test 04: Boundary Conditions  (ndim={ndim})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'04_boundary_conditions_{ndim}d.png', cfg)


# ── pytest entry points ───────────────────────────────────────────────────────

def test_boundary_conditions_2d():
    test_boundary_conditions(ndim=2)

def test_boundary_conditions_3d():
    test_boundary_conditions(ndim=3)


if __name__ == '__main__':
    from tests._helpers import run_suite
    run_suite([test_boundary_conditions])

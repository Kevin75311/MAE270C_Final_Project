"""
tests/test_01_mesh.py — Mesh geometry and region mask tests.

Checks grid dimensions, region mask disjointness, and completeness
for both 2D and 3D configurations.

Run standalone:
    python tests/test_01_mesh.py
Run with pytest:
    pytest tests/test_01_mesh.py -v
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
import matplotlib.patches as mpatches

from tests._helpers import make_cfg, save_fig, OUT
from physics.mesh import build_mesh, build_region_masks


def test_mesh_geometry(ndim=2):
    """
    Checks grid dimensions, region mask disjointness, and completeness.

    2D visual: 3 panels — domain partition, tumor mask, distance field (x-y).
    3D visual: 2 rows × 3 cols — top row = axial (x-y), bottom row = meridional (x-z).
               Both rows show: partition map, tumor mask, distance field.
               Full grid resolution used in both cases.
    """
    cfg = make_cfg(ndim, small=False)   # always full resolution
    X, Y, Z, xv, yv, zv = build_mesh(cfg)
    tumor, healthy, margin = build_region_masks(cfg)

    Nx, Ny, Nz = cfg.domain.Nx, cfg.domain.Ny, cfg.domain.Nz

    # ── Shape assertions ──────────────────────────────────────────────────
    if ndim == 2:
        assert X.shape == (Ny, Nx), f"X.shape={X.shape}"
        assert Y.shape == (Ny, Nx)
        assert Z is None
        assert zv is None
    else:
        assert X.shape == (Nz, Ny, Nx), f"X.shape={X.shape}"
        assert Y.shape == (Nz, Ny, Nx)
        assert Z.shape == (Nz, Ny, Nx)
        assert zv.shape == (Nz,)

    assert xv.shape == (Nx,)
    assert yv.shape == (Ny,)

    # ── Mask assertions ───────────────────────────────────────────────────
    assert not np.any(tumor & healthy),  "tumor/healthy overlap"
    assert not np.any(tumor & margin),   "tumor/margin overlap"
    assert not np.any(margin & healthy), "margin/healthy overlap"
    assert (tumor | healthy | margin).all(), "some voxels unclassified"
    assert tumor.sum() > 0
    assert tumor.sum() < cfg.domain.N
    assert healthy.sum() > tumor.sum()

    # ── Spacing assertions ────────────────────────────────────────────────
    np.testing.assert_allclose(xv[1] - xv[0],
                               (xv[-1] - xv[0]) / (Nx - 1), rtol=1e-9)
    np.testing.assert_allclose(yv[1] - yv[0],
                               (yv[-1] - yv[0]) / (Ny - 1), rtol=1e-9)

    print(f"  ndim={ndim}  N={cfg.domain.N}  grid={Nx}×{Ny}" +
          (f"×{Nz}" if ndim == 3 else ""))
    print(f"  Tumor:   {tumor.sum()} voxels ({100*tumor.mean():.1f}%)")
    print(f"  Margin:  {margin.sum()} voxels ({100*margin.mean():.1f}%)")
    print(f"  Healthy: {healthy.sum()} voxels ({100*healthy.mean():.1f}%)")

    # ── Figure ────────────────────────────────────────────────────────────
    cx, cy, cz = cfg.domain.tumor_center
    cmap_r = matplotlib.colors.ListedColormap(['#2ecc71', '#f1c40f', '#e74c3c'])

    def _make_region_map(t, h, m):
        rm = np.zeros_like(t, dtype=float)
        rm[t] = 2.0; rm[m] = 1.0; rm[h] = 0.0
        return rm

    def _draw_tumor_circles(ax, cx_cm, cy_cm):
        for r_val, ls in [(cfg.domain.tumor_radius, '--'),
                          (cfg.domain.tumor_radius + cfg.domain.safety_margin, ':')]:
            ax.add_patch(plt.Circle((cx_cm, cy_cm), r_val*100,
                                     fill=False, edgecolor='black',
                                     linewidth=1.5, linestyle=ls))

    def _draw_panel(ax, fig, Xp, Yp, tumor_p, margin_p, healthy_p,
                    xlabel, ylabel, title, mode):
        """mode: 'partition', 'mask', 'distance'"""
        if mode == 'partition':
            rm = _make_region_map(tumor_p, healthy_p, margin_p)
            ax.pcolormesh(Xp*100, Yp*100, rm, cmap=cmap_r, vmin=0, vmax=2,
                          shading='auto')
            _draw_tumor_circles(ax, cx*100, cy*100)
        elif mode == 'mask':
            ax.pcolormesh(Xp*100, Yp*100, tumor_p.astype(float),
                          cmap='Reds', shading='auto', vmin=0, vmax=1)
        elif mode == 'distance':
            dist = np.sqrt((Xp - cx)**2 + (Yp - cy)**2) * 100
            im = ax.pcolormesh(Xp*100, Yp*100, dist, cmap='viridis', shading='auto')
            ax.contour(Xp*100, Yp*100, dist,
                       levels=[cfg.domain.tumor_radius*100,
                                (cfg.domain.tumor_radius + cfg.domain.safety_margin)*100],
                       colors=['white', 'yellow'], linewidths=1.5)
            fig.colorbar(im, ax=ax, label='dist [cm]', fraction=0.046, pad=0.04)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')

    if ndim == 2:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        _draw_panel(axes[0], fig, X, Y, tumor, margin, healthy,
                    'x [cm]', 'y [cm]', 'Domain partition  Ω_T ∪ Ω_M ∪ Ω_H', 'partition')
        patches = [mpatches.Patch(color='#2ecc71', label='Ω_H (healthy)'),
                   mpatches.Patch(color='#f1c40f', label='Ω_M (margin)'),
                   mpatches.Patch(color='#e74c3c', label='Ω_T (tumor)')]
        axes[0].legend(handles=patches, fontsize=7, loc='upper right')
        _draw_panel(axes[1], fig, X, Y, tumor, margin, healthy,
                    'x [cm]', 'y [cm]', f'Tumor mask  ({tumor.sum()} voxels)', 'mask')
        _draw_panel(axes[2], fig, X, Y, tumor, margin, healthy,
                    'x [cm]', 'y [cm]', 'Distance field from center', 'distance')

    else:
        # Axial slice: z = center
        iz = int(np.clip(round(cz / cfg.domain.dz), 0, Nz - 1))
        # Meridional slice: y = center (x-z plane)
        iy = int(np.clip(round(cy / cfg.domain.dy), 0, Ny - 1))

        # Axial 2D fields (Ny × Nx)
        Xax, Yax = X[iz], Y[iz]
        t_ax = tumor[iz]; m_ax = margin[iz]; h_ax = healthy[iz]

        # Meridional 2D fields (Nz × Nx) — Z on vertical axis, X on horizontal
        Xme = X[:, iy, :]   # shape (Nz, Nx), values along x
        Zme = Z[:, iy, :]   # shape (Nz, Nx), values along z
        t_me = tumor[:, iy, :]; m_me = margin[:, iy, :]; h_me = healthy[:, iy, :]

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))

        # Row 0: axial (x-y at z=cz)
        z_label = f'z = {zv[iz]*100:.1f} cm'
        _draw_panel(axes[0,0], fig, Xax, Yax, t_ax, m_ax, h_ax,
                    'x [cm]', 'y [cm]', f'Axial partition  ({z_label})', 'partition')
        patches = [mpatches.Patch(color='#2ecc71', label='Ω_H'),
                   mpatches.Patch(color='#f1c40f', label='Ω_M'),
                   mpatches.Patch(color='#e74c3c', label='Ω_T')]
        axes[0,0].legend(handles=patches, fontsize=7, loc='upper right')
        _draw_panel(axes[0,1], fig, Xax, Yax, t_ax, m_ax, h_ax,
                    'x [cm]', 'y [cm]', f'Axial tumor mask  ({t_ax.sum()} vox)', 'mask')
        _draw_panel(axes[0,2], fig, Xax, Yax, t_ax, m_ax, h_ax,
                    'x [cm]', 'y [cm]', 'Axial distance field', 'distance')

        # Row 1: meridional (x-z at y=cy)
        y_label = f'y = {yv[iy]*100:.1f} cm'
        _draw_panel(axes[1,0], fig, Xme, Zme, t_me, m_me, h_me,
                    'x [cm]', 'z [cm]', f'Meridional partition  ({y_label})', 'partition')
        axes[1,0].legend(handles=patches, fontsize=7, loc='upper right')
        _draw_panel(axes[1,1], fig, Xme, Zme, t_me, m_me, h_me,
                    'x [cm]', 'z [cm]', f'Meridional tumor mask  ({t_me.sum()} vox)', 'mask')
        # Distance in x-z plane from (cx, cz)
        dist_me = np.sqrt((Xme - cx)**2 + (Zme - cz)**2) * 100
        im_me = axes[1,2].pcolormesh(Xme*100, Zme*100, dist_me, cmap='viridis', shading='auto')
        axes[1,2].contour(Xme*100, Zme*100, dist_me,
                          levels=[cfg.domain.tumor_radius*100,
                                  (cfg.domain.tumor_radius + cfg.domain.safety_margin)*100],
                          colors=['white', 'yellow'], linewidths=1.5)
        fig.colorbar(im_me, ax=axes[1,2], label='dist [cm]', fraction=0.046, pad=0.04)
        axes[1,2].set_xlabel('x [cm]', fontsize=9); axes[1,2].set_ylabel('z [cm]', fontsize=9)
        axes[1,2].set_title('Meridional distance field', fontsize=10)
        axes[1,2].set_aspect('equal')

    fig.suptitle(f'Test 01: Mesh Geometry  (ndim={ndim})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'01_mesh_geometry_{ndim}d.png', cfg)


# ── pytest entry points ───────────────────────────────────────────────────────

def test_mesh_geometry_2d():
    test_mesh_geometry(ndim=2)

def test_mesh_geometry_3d():
    test_mesh_geometry(ndim=3)


if __name__ == '__main__':
    from tests._helpers import run_suite
    run_suite([test_mesh_geometry])

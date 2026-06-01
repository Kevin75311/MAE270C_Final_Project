"""
visualization/damage_plots.py — Thermal damage field and necrosis boundary plots.

Visualizes Ω_d(x, y, t):
  - Filled colormap (green=safe, red=ablated) matching clinical conventions
  - Necrosis boundary isoline (Ω_d = 1)
  - Tumor boundary overlay
  - PyVista 3D isosurface (optional)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_mesh, unflatten


def plot_damage_field(Omega_flat: np.ndarray,
                      t: float = 0.0,
                      cfg: SimConfig = default_cfg,
                      ax=None, fig=None,
                      save_path: str = None) -> plt.Figure:
    """
    2D Arrhenius damage field Ω_d(x, y, t).

    Color encoding (clinical convention):
      Ω_d = 0.0      : green  (no damage)
      Ω_d = 0.5      : yellow (sub-lethal stress)
      Ω_d = 1.0      : orange (threshold — onset of irreversible necrosis)
      Ω_d ≥ 1.0      : red    (complete ablation)
    """
    Omega_2d = unflatten(Omega_flat, cfg)
    X, Y, _, _ = build_mesh(cfg)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # ── Colormap clipped at Ω = 2 for display ────────────────────────────
    vmax_display = 2.0
    norm  = Normalize(vmin=0.0, vmax=vmax_display)
    cmap  = plt.get_cmap(cfg.viz.colormap_damage)

    pcm = ax.pcolormesh(X * 100, Y * 100, Omega_2d,
                        cmap=cmap, norm=norm, shading='auto')

    # ── Necrosis boundary: Ω_d = 1 ───────────────────────────────────────
    try:
        cs_nec = ax.contour(X * 100, Y * 100, Omega_2d,
                            levels=[1.0],
                            colors=['black'], linewidths=[2.5])
        ax.clabel(cs_nec, fmt='Ω=1  (necrosis)', fontsize=9,
                  manual=False)
    except Exception:
        pass  # contour may fail if Omega is all zero

    # ── Additional damage isolines ────────────────────────────────────────
    try:
        cs_sub = ax.contour(X * 100, Y * 100, Omega_2d,
                            levels=[0.5, 1.5],
                            colors=['orange', 'darkred'],
                            linewidths=[1.0, 1.5],
                            linestyles=['--', ':'])
    except Exception:
        pass

    # ── Tumor region boundary ─────────────────────────────────────────────
    cx, cy = cfg.domain.tumor_center
    ax.add_patch(Circle((cx*100, cy*100), cfg.domain.tumor_radius * 100,
                        fill=False, edgecolor='white',
                        linewidth=2.0, linestyle='--', label='Tumor boundary'))
    ax.add_patch(Circle((cx*100, cy*100),
                        (cfg.domain.tumor_radius + cfg.domain.safety_margin) * 100,
                        fill=False, edgecolor='lightgray',
                        linewidth=1.0, linestyle=':', label='Safety margin'))

    # ── Colorbar ──────────────────────────────────────────────────────────
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Thermal damage  Ω_d  [dimensionless]', fontsize=10)
    cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    cbar.set_ticklabels(['0\n(no damage)', '0.5', '1.0\n(threshold)', '1.5', '≥2\n(ablated)'])

    ax.set_xlabel('x [cm]', fontsize=11)
    ax.set_ylabel('y [cm]', fontsize=11)
    ax.set_title(f'Thermal damage  Ω_d(x,y)   t = {t:.1f} s', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=cfg.viz.dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_necrosis_boundary(Omega_flat: np.ndarray,
                            T_flat: np.ndarray,
                            t: float,
                            cfg: SimConfig = default_cfg,
                            ax=None, fig=None,
                            save_path: str = None) -> plt.Figure:
    """
    Overlay the Ω_d = 1 necrosis boundary on the temperature heatmap.
    Combines both state fields in one clinical view.
    """
    from visualization.field_plots import plot_temperature_field

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    plot_temperature_field(T_flat, t=t, cfg=cfg, ax=ax, fig=fig)

    Omega_2d = unflatten(Omega_flat, cfg)
    X, Y, _, _ = build_mesh(cfg)

    try:
        cs = ax.contour(X * 100, Y * 100, Omega_2d,
                        levels=[1.0], colors=['lime'], linewidths=[3.0])
        ax.clabel(cs, fmt='Ω_d=1\n(necrosis)', fontsize=9)
    except Exception:
        pass

    ax.set_title(f'T field + necrosis boundary   t = {t:.1f} s', fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=cfg.viz.dpi, bbox_inches='tight')
    return fig


def plot_damage_3d_pyvista(Omega_flat: np.ndarray,
                            cfg: SimConfig = default_cfg):
    """
    3D isosurface of the necrosis boundary (Ω_d = 1) using PyVista.
    Requires:  pip install pyvista
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed.  Run:  pip install pyvista")
        return

    Nx, Ny  = cfg.domain.Nx, cfg.domain.Ny
    Om_2d   = unflatten(Omega_flat, cfg)

    grid = pv.ImageData()
    grid.dimensions = (Nx, Ny, 1)
    grid.spacing    = (cfg.domain.dx * 100, cfg.domain.dy * 100, 1.0)
    grid.point_data['Omega_d'] = Om_2d.ravel(order='F')

    pl = pv.Plotter()
    pl.add_volume(grid, cmap='RdYlGn_r', opacity='linear',
                  clim=[0, 2],
                  scalar_bar_args={'title': 'Ω_d'})

    isosurf = grid.contour([1.0], scalars='Omega_d')
    if isosurf.n_points > 0:
        pl.add_mesh(isosurf, color='black', opacity=0.8,
                    label='Necrosis boundary (Ω=1)')

    pl.add_axes()
    pl.show_bounds(xlabel='x [cm]', ylabel='y [cm]')
    pl.show()

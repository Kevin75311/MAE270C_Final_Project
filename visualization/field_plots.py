"""
visualization/field_plots.py — 2D temperature field plots (COMSOL-style heatmaps).

Uses matplotlib pcolormesh for high-quality filled color maps with:
  - Diverging/sequential colormaps matching clinical display conventions
  - Tumor and healthy region boundary overlays
  - Colorbar with physical units
  - Optional PyVista 3D rendering (requires pyvista installation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_mesh, build_region_masks, unflatten


def plot_temperature_field(T_flat: np.ndarray,
                           t: float = 0.0,
                           cfg: SimConfig = default_cfg,
                           ax=None, fig=None,
                           save_path: str = None) -> plt.Figure:
    """
    2D temperature heatmap at a single timestep.

    Renders T(x, y, t) with:
      - 'hot' colormap over [T_init, T_display_max]
      - Tumor boundary (white dashed circle)
      - Safety margin boundary (gray dashed circle)
      - T_safe isoline (cyan contour)
      - T = 60 °C ablation isoline (red contour)
    """
    T_2d = unflatten(T_flat, cfg)
    X, Y, _, _ = build_mesh(cfg)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    vmin = cfg.viz.T_display_min
    vmax = cfg.viz.T_display_max
    norm = Normalize(vmin=vmin, vmax=vmax)

    # ── Main heatmap ─────────────────────────────────────────────────────
    pcm = ax.pcolormesh(X * 100, Y * 100, T_2d,
                        cmap=cfg.viz.colormap_temperature,
                        norm=norm, shading='auto')

    # ── Isotherms ────────────────────────────────────────────────────────
    levels_iso = [cfg.control.T_safe, 55.0, 60.0, 80.0]
    cs = ax.contour(X * 100, Y * 100, T_2d,
                    levels=levels_iso,
                    colors=['cyan', 'yellow', 'red', 'white'],
                    linewidths=[1.5, 1.0, 2.0, 1.0],
                    linestyles=['--', '--', '-', ':'])
    ax.clabel(cs, fmt='%.0f°C', fontsize=8)

    # ── Region boundaries ────────────────────────────────────────────────
    cx, cy   = cfg.domain.tumor_center
    r_tumor  = cfg.domain.tumor_radius
    r_margin = r_tumor + cfg.domain.safety_margin

    tumor_circle  = plt.Circle((cx*100, cy*100), r_tumor*100,
                                fill=False, edgecolor='white',
                                linewidth=2, linestyle='--', label='Tumor boundary')
    margin_circle = plt.Circle((cx*100, cy*100), r_margin*100,
                                fill=False, edgecolor='lightgray',
                                linewidth=1, linestyle=':', label='Safety margin')
    ax.add_patch(tumor_circle)
    ax.add_patch(margin_circle)

    # ── Colorbar ─────────────────────────────────────────────────────────
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature [°C]', fontsize=11)

    ax.set_xlabel('x [cm]', fontsize=11)
    ax.set_ylabel('y [cm]', fontsize=11)
    ax.set_title(f'Temperature field  T(x,y)   t = {t:.1f} s', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=cfg.viz.dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_slice_comparison(T_history: np.ndarray,
                          t_vec: np.ndarray,
                          snapshot_indices: list,
                          cfg: SimConfig = default_cfg,
                          save_path: str = None) -> plt.Figure:
    """
    Side-by-side temperature snapshots at selected timesteps.
    """
    n_snap = len(snapshot_indices)
    fig, axes = plt.subplots(1, n_snap, figsize=(4 * n_snap, 4.5))
    if n_snap == 1:
        axes = [axes]

    X, Y, _, _ = build_mesh(cfg)
    vmin = cfg.viz.T_display_min
    vmax = cfg.viz.T_display_max
    norm = Normalize(vmin=vmin, vmax=vmax)

    for ax, idx in zip(axes, snapshot_indices):
        T_2d = unflatten(T_history[idx], cfg)
        t    = t_vec[idx]

        pcm = ax.pcolormesh(X * 100, Y * 100, T_2d,
                            cmap=cfg.viz.colormap_temperature,
                            norm=norm, shading='auto')
        ax.contour(X * 100, Y * 100, T_2d,
                   levels=[cfg.control.T_safe, 60.0],
                   colors=['cyan', 'red'], linewidths=1.5)

        cx, cy = cfg.domain.tumor_center
        ax.add_patch(plt.Circle((cx*100, cy*100),
                                cfg.domain.tumor_radius * 100,
                                fill=False, edgecolor='white',
                                linewidth=1.5, linestyle='--'))
        ax.set_title(f't = {t:.0f} s', fontsize=11)
        ax.set_xlabel('x [cm]')
        ax.set_aspect('equal')
        if ax == axes[0]:
            ax.set_ylabel('y [cm]')

    fig.colorbar(pcm, ax=axes, label='Temperature [°C]',
                 fraction=0.015, pad=0.04)
    fig.suptitle('Temperature field snapshots', fontsize=13, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=cfg.viz.dpi, bbox_inches='tight')
    return fig


def plot_temperature_3d_pyvista(T_flat: np.ndarray,
                                 cfg: SimConfig = default_cfg):
    """
    3D volume rendering of the temperature field using PyVista.
    Requires:  pip install pyvista

    Produces a COMSOL-like interactive 3D view with:
      - Volume rendering of T field
      - Clipping plane
      - Isosurface at T = 60°C
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed.  Run:  pip install pyvista")
        return

    Nx, Ny = cfg.domain.Nx, cfg.domain.Ny
    T_2d   = unflatten(T_flat, cfg)

    # Create UniformGrid (PyVista structured grid)
    grid = pv.ImageData()
    grid.dimensions = (Nx, Ny, 1)
    grid.spacing    = (cfg.domain.dx * 100, cfg.domain.dy * 100, 1.0)  # cm
    grid.point_data['Temperature'] = T_2d.ravel(order='F')

    pl = pv.Plotter()
    pl.add_volume(grid, cmap='hot', opacity='linear',
                  clim=[cfg.viz.T_display_min, cfg.viz.T_display_max],
                  scalar_bar_args={'title': 'T [°C]'})

    # Isosurface at ablation temperature
    isosurf = grid.contour([60.0], scalars='Temperature')
    if isosurf.n_points > 0:
        pl.add_mesh(isosurf, color='red', opacity=0.5, label='60°C isosurface')

    pl.add_axes()
    pl.show_bounds(xlabel='x [cm]', ylabel='y [cm]')
    pl.show()

"""
visualization/field_plots.py — Temperature field plots, 2D and 3D-aware.

2D mode (cfg.domain.ndim == 2):
  Single x-y heatmap with isotherms and region boundaries.

3D mode (cfg.domain.ndim == 3):
  Two side-by-side cross-sections through the probe position:
    Left  — Axial view:      z = probe_z  (x-y plane, shows radial spread)
    Right — Meridional view: y = probe_y  (x-z plane, shows depth profile)
  The meridional view has z on the vertical axis and x on the horizontal,
  giving a 'z vs r' read with no azimuthal ambiguity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_mesh, build_region_masks, unflatten


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_slices(field_nd: np.ndarray, cfg: SimConfig):
    """
    Extract two 2D cross-sections from a 3D field (Nz, Ny, Nx).

    Slices pass through the probe position (xp, yp, zp):
      horiz : (Ny, Nx)  at z = nearest grid plane to zp  (axial view)
      vert  : (Nz, Nx)  at y = nearest grid plane to yp  (meridional view)
    """
    _, yp, zp = cfg.domain.probe_position
    z_idx = int(np.clip(round(zp / cfg.domain.dz), 0, cfg.domain.Nz - 1))
    y_idx = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny - 1))
    return field_nd[z_idx, :, :], field_nd[:, y_idx, :], z_idx, y_idx


def _add_tumor_xy(ax, cfg, color='white', lw=2.0):
    """Overlay tumor and margin circles on an x-y axes."""
    cx, cy = cfg.domain.tumor_center[:2]
    r_T = cfg.domain.tumor_radius
    r_M = r_T + cfg.domain.safety_margin
    ax.add_patch(plt.Circle((cx*100, cy*100), r_T*100,
                             fill=False, edgecolor=color,
                             linewidth=lw, linestyle='--', label='Tumor'))
    ax.add_patch(plt.Circle((cx*100, cy*100), r_M*100,
                             fill=False, edgecolor='lightgray',
                             linewidth=1.0, linestyle=':', label='Margin'))


def _add_tumor_xz(ax, cfg, color='white', lw=2.0):
    """
    Overlay tumor circle on an x-z (meridional) axes.
    The sphere cross-section at y = tumor_cy is a circle of the same radius.
    """
    cx, _, cz = cfg.domain.tumor_center
    r_T = cfg.domain.tumor_radius
    r_M = r_T + cfg.domain.safety_margin
    ax.add_patch(plt.Circle((cx*100, cz*100), r_T*100,
                             fill=False, edgecolor=color,
                             linewidth=lw, linestyle='--', label='Tumor'))
    ax.add_patch(plt.Circle((cx*100, cz*100), r_M*100,
                             fill=False, edgecolor='lightgray',
                             linewidth=1.0, linestyle=':'))


def _pcolor_with_contours(ax, fig, xv, yv, field_2d, norm, cmap,
                           iso_levels, iso_colors, iso_lw,
                           xlabel, ylabel, title, add_cbar=True):
    """Shared pcolormesh + contour rendering for one panel."""
    pcm = ax.pcolormesh(xv * 100, yv * 100, field_2d,
                        cmap=cmap, norm=norm, shading='auto')
    try:
        cs = ax.contour(xv * 100, yv * 100, field_2d,
                        levels=iso_levels, colors=iso_colors,
                        linewidths=iso_lw)
        ax.clabel(cs, fmt='%.0f°C', fontsize=7, inline=True)
    except Exception:
        pass
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
    cbar = None
    if add_cbar:
        cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('T [°C]', fontsize=10)
    return pcm, cbar


# ── Public plotting functions ─────────────────────────────────────────────────

def plot_temperature_field(T_flat: np.ndarray,
                           t: float = 0.0,
                           cfg: SimConfig = default_cfg,
                           ax=None, fig=None,
                           save_path: str = None) -> plt.Figure:
    """
    Temperature heatmap at a single timestep.

    2D: single x-y panel.
    3D: two panels — axial (x-y at z=probe_z) and meridional (x-z at y=probe_y).
    """
    vmin = cfg.viz.T_display_min
    vmax = cfg.viz.T_display_max
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cfg.viz.colormap_temperature
    iso_levels = [cfg.control.T_safe, 55.0, 60.0, 80.0]
    iso_colors = ['cyan', 'yellow', 'red', 'white']
    iso_lw     = [1.5, 1.0, 2.0, 1.0]

    X, Y, Z, xv, yv, zv = build_mesh(cfg)

    if cfg.domain.ndim == 2:
        # ── 2D: single panel ─────────────────────────────────────────────────
        T_2d = unflatten(T_flat, cfg)
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        _pcolor_with_contours(ax, fig, xv, yv, T_2d, norm, cmap,
                              iso_levels, iso_colors, iso_lw,
                              'x [cm]', 'y [cm]',
                              f'Temperature  T(x,y)   t = {t:.1f} s')
        _add_tumor_xy(ax, cfg)
        ax.legend(loc='upper right', fontsize=8)

    else:
        # ── 3D: axial + meridional panels ────────────────────────────────────
        T_3d = unflatten(T_flat, cfg)
        horiz, vert, z_idx, y_idx = _extract_slices(T_3d, cfg)
        xp, yp, zp = cfg.domain.probe_position
        z_shown = zv[z_idx] * 100
        y_shown = yv[y_idx] * 100

        fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(f'Temperature field   t = {t:.1f} s', fontsize=13, y=1.01)

        # Left — axial (x-y at z=probe_z)
        _pcolor_with_contours(ax_h, fig, xv, yv, horiz, norm, cmap,
                              iso_levels, iso_colors, iso_lw,
                              'x [cm]', 'y [cm]',
                              f'Axial  (z = {z_shown:.1f} cm)')
        _add_tumor_xy(ax_h, cfg)
        ax_h.legend(loc='upper right', fontsize=8)

        # Right — meridional (x-z at y=probe_y), z on vertical axis
        pcm_v, _ = _pcolor_with_contours(ax_v, fig, xv, zv, vert, norm, cmap,
                                          iso_levels, iso_colors, iso_lw,
                                          'x [cm]', 'z [cm]',
                                          f'Meridional  (y = {y_shown:.1f} cm)',
                                          add_cbar=True)
        _add_tumor_xz(ax_v, cfg)
        ax_v.axvline(xp * 100, color='lime', linewidth=1.0,
                     linestyle=':', alpha=0.7, label='Probe axis')
        ax_v.legend(loc='upper right', fontsize=8)

        fig.tight_layout()

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
    Temperature snapshots at selected timesteps.

    2D: single row of x-y panels.
    3D: two rows — axial (top) and meridional (bottom) — one column per snapshot.
    """
    X, Y, Z, xv, yv, zv = build_mesh(cfg)
    n_snap = len(snapshot_indices)
    vmin = cfg.viz.T_display_min
    vmax = cfg.viz.T_display_max
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cfg.viz.colormap_temperature
    iso_levels = [cfg.control.T_safe, 60.0]
    iso_colors = ['cyan', 'red']
    iso_lw     = [1.5, 1.5]

    if cfg.domain.ndim == 2:
        # ── 2D: one row ───────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, n_snap, figsize=(4 * n_snap, 4.5))
        if n_snap == 1:
            axes = [axes]
        last_pcm = None
        for ax, idx in zip(axes, snapshot_indices):
            T_2d = unflatten(T_history[idx], cfg)
            pcm, _ = _pcolor_with_contours(ax, fig, xv, yv, T_2d, norm, cmap,
                                            iso_levels, iso_colors, iso_lw,
                                            'x [cm]',
                                            'y [cm]' if ax is axes[0] else '',
                                            f't = {t_vec[idx]:.0f} s',
                                            add_cbar=False)
            _add_tumor_xy(ax, cfg, lw=1.5)
            last_pcm = pcm
        fig.colorbar(last_pcm, ax=axes, label='T [°C]', fraction=0.015, pad=0.04)
        fig.suptitle('Temperature snapshots', fontsize=13)

    else:
        # ── 3D: two rows (axial / meridional) ────────────────────────────────
        xp, yp, zp = cfg.domain.probe_position
        fig, axes = plt.subplots(2, n_snap,
                                  figsize=(4 * n_snap, 8),
                                  gridspec_kw={'hspace': 0.45, 'wspace': 0.15})
        if n_snap == 1:
            axes = axes.reshape(2, 1)

        last_pcm = None
        for col, idx in enumerate(snapshot_indices):
            T_3d = unflatten(T_history[idx], cfg)
            horiz, vert, z_idx, y_idx = _extract_slices(T_3d, cfg)
            t_label = f't = {t_vec[idx]:.0f} s'

            # Top row: axial
            ax_h = axes[0, col]
            pcm, _ = _pcolor_with_contours(ax_h, fig, xv, yv, horiz, norm, cmap,
                                            iso_levels, iso_colors, iso_lw,
                                            'x [cm]',
                                            'y [cm]' if col == 0 else '',
                                            t_label, add_cbar=False)
            _add_tumor_xy(ax_h, cfg, lw=1.2)
            if col == 0:
                ax_h.set_ylabel('Axial\ny [cm]', fontsize=9)
            last_pcm = pcm

            # Bottom row: meridional
            ax_v = axes[1, col]
            pcm, _ = _pcolor_with_contours(ax_v, fig, xv, zv, vert, norm, cmap,
                                            iso_levels, iso_colors, iso_lw,
                                            'x [cm]',
                                            'z [cm]' if col == 0 else '',
                                            '', add_cbar=False)
            _add_tumor_xz(ax_v, cfg, lw=1.2)
            ax_v.axvline(xp * 100, color='lime', lw=0.8, linestyle=':', alpha=0.7)
            if col == 0:
                ax_v.set_ylabel('Meridional\nz [cm]', fontsize=9)

        fig.colorbar(last_pcm, ax=axes.ravel().tolist(),
                     label='T [°C]', fraction=0.012, pad=0.04)
        fig.suptitle('Temperature snapshots — axial (top) and meridional (bottom)',
                     fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=cfg.viz.dpi, bbox_inches='tight')
    return fig


def plot_temperature_3d_pyvista(T_flat: np.ndarray,
                                 cfg: SimConfig = default_cfg):
    """
    3D volume rendering of the temperature field using PyVista.
    Requires:  pip install pyvista

    Works for both ndim=2 (extruded slab) and ndim=3 (true volume).
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed.  Run:  pip install pyvista")
        return

    T_nd = unflatten(T_flat, cfg)

    if cfg.domain.ndim == 2:
        Nx, Ny = cfg.domain.Nx, cfg.domain.Ny
        grid = pv.ImageData()
        grid.dimensions = (Nx, Ny, 1)
        grid.spacing    = (cfg.domain.dx * 100, cfg.domain.dy * 100, 1.0)
        grid.point_data['Temperature'] = T_nd.ravel(order='F')
    else:
        Nx, Ny, Nz = cfg.domain.Nx, cfg.domain.Ny, cfg.domain.Nz
        grid = pv.ImageData()
        grid.dimensions = (Nx, Ny, Nz)
        grid.spacing    = (cfg.domain.dx*100, cfg.domain.dy*100, cfg.domain.dz*100)
        # Reorder (Nz,Ny,Nx) → Fortran order (Nx,Ny,Nz) for PyVista
        grid.point_data['Temperature'] = T_nd.transpose(2, 1, 0).ravel(order='F')

    pl = pv.Plotter()
    pl.add_volume(grid, cmap='hot', opacity='linear',
                  clim=[cfg.viz.T_display_min, cfg.viz.T_display_max],
                  scalar_bar_args={'title': 'T [°C]'})
    isosurf = grid.contour([60.0], scalars='Temperature')
    if isosurf.n_points > 0:
        pl.add_mesh(isosurf, color='red', opacity=0.5, label='60°C isosurface')
    pl.add_axes()
    pl.show_bounds(xlabel='x [cm]', ylabel='y [cm]', zlabel='z [cm]')
    pl.show()

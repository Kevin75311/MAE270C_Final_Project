"""
visualization/damage_plots.py — Thermal damage field and necrosis boundary plots.

2D mode (cfg.domain.ndim == 2):
  Single x-y damage heatmap with Ω_d = 1 necrosis boundary.

3D mode (cfg.domain.ndim == 3):
  Two side-by-side panels through the probe position:
    Left  — Axial view:      z = probe_z  (x-y plane)
    Right — Meridional view: y = probe_y  (x-z plane, z on vertical axis)

Color encoding (clinical convention, both modes):
  Ω_d = 0   → green  (no damage)
  Ω_d = 0.5 → yellow (sub-lethal stress)
  Ω_d = 1   → onset of irreversible necrosis  (black contour)
  Ω_d ≥ 2   → complete ablation (red)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_mesh, unflatten


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_slices(field_nd: np.ndarray, cfg: SimConfig):
    """Extract axial and meridional 2D slices from a 3D field (Nz, Ny, Nx)."""
    _, yp, zp = cfg.domain.probe_position
    z_idx = int(np.clip(round(zp / cfg.domain.dz), 0, cfg.domain.Nz - 1))
    y_idx = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny - 1))
    return field_nd[z_idx, :, :], field_nd[:, y_idx, :], z_idx, y_idx


def _damage_pcolor(ax, fig, xv, yv, field_2d, norm, cmap, add_cbar=True):
    """Render damage colormap and isolines on one axes panel."""
    pcm = ax.pcolormesh(xv * 100, yv * 100, field_2d,
                        cmap=cmap, norm=norm, shading='auto')
    # Necrosis boundary (Ω_d = 1)
    try:
        cs1 = ax.contour(xv * 100, yv * 100, field_2d,
                         levels=[1.0], colors=['black'], linewidths=[2.5])
        ax.clabel(cs1, fmt='Ω=1', fontsize=8, inline=True)
    except Exception:
        pass
    # Sub-lethal and ablated isolines
    try:
        ax.contour(xv * 100, yv * 100, field_2d,
                   levels=[0.5, 1.5],
                   colors=['orange', 'darkred'],
                   linewidths=[1.0, 1.5],
                   linestyles=['--', ':'])
    except Exception:
        pass
    cbar = None
    if add_cbar:
        cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Ω_d', fontsize=10)
        cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
        cbar.set_ticklabels(['0', '0.5', '1.0\n(threshold)', '1.5', '≥2'])
    return pcm, cbar


def _add_tumor_xy(ax, cfg, color='white', lw=2.0):
    cx, cy = cfg.domain.tumor_center[:2]
    r_T = cfg.domain.tumor_radius
    r_M = r_T + cfg.domain.safety_margin
    ax.add_patch(Circle((cx*100, cy*100), r_T*100,
                        fill=False, edgecolor=color, linewidth=lw,
                        linestyle='--', label='Tumor'))
    ax.add_patch(Circle((cx*100, cy*100), r_M*100,
                        fill=False, edgecolor='lightgray', linewidth=1.0,
                        linestyle=':'))


def _add_tumor_xz(ax, cfg, color='white', lw=2.0):
    cx, _, cz = cfg.domain.tumor_center
    r_T = cfg.domain.tumor_radius
    ax.add_patch(Circle((cx*100, cz*100), r_T*100,
                        fill=False, edgecolor=color, linewidth=lw,
                        linestyle='--', label='Tumor'))


# ── Public plotting functions ─────────────────────────────────────────────────

def plot_damage_field(Omega_flat: np.ndarray,
                      t: float = 0.0,
                      cfg: SimConfig = default_cfg,
                      ax=None, fig=None,
                      save_path: str = None) -> plt.Figure:
    """
    Arrhenius damage field Ω_d at a single timestep.

    2D: single x-y panel.
    3D: axial + meridional panels side by side.
    """
    vmax_display = 2.0
    norm = Normalize(vmin=0.0, vmax=vmax_display)
    cmap = plt.get_cmap(cfg.viz.colormap_damage)
    X, Y, Z, xv, yv, zv = build_mesh(cfg)

    if cfg.domain.ndim == 2:
        Omega_2d = unflatten(Omega_flat, cfg)
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        _damage_pcolor(ax, fig, xv, yv, Omega_2d, norm, cmap)
        _add_tumor_xy(ax, cfg)
        ax.set_xlabel('x [cm]', fontsize=11)
        ax.set_ylabel('y [cm]', fontsize=11)
        ax.set_title(f'Thermal damage  Ω_d(x,y)   t = {t:.1f} s', fontsize=12)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)

    else:
        Omega_3d = unflatten(Omega_flat, cfg)
        horiz, vert, z_idx, y_idx = _extract_slices(Omega_3d, cfg)
        xp, yp, zp = cfg.domain.probe_position
        z_shown = zv[z_idx] * 100
        y_shown = yv[y_idx] * 100

        fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(f'Thermal damage  Ω_d   t = {t:.1f} s', fontsize=13, y=1.01)

        # Axial
        _damage_pcolor(ax_h, fig, xv, yv, horiz, norm, cmap)
        _add_tumor_xy(ax_h, cfg)
        ax_h.set_xlabel('x [cm]', fontsize=10)
        ax_h.set_ylabel('y [cm]', fontsize=10)
        ax_h.set_title(f'Axial  (z = {z_shown:.1f} cm)', fontsize=11)
        ax_h.set_aspect('equal')
        ax_h.legend(loc='upper right', fontsize=8)

        # Meridional
        _damage_pcolor(ax_v, fig, xv, zv, vert, norm, cmap)
        _add_tumor_xz(ax_v, cfg)
        ax_v.axvline(xp * 100, color='lime', lw=1.0, linestyle=':', alpha=0.7,
                     label='Probe axis')
        ax_v.set_xlabel('x [cm]', fontsize=10)
        ax_v.set_ylabel('z [cm]', fontsize=10)
        ax_v.set_title(f'Meridional  (y = {y_shown:.1f} cm)', fontsize=11)
        ax_v.set_aspect('equal')
        ax_v.legend(loc='upper right', fontsize=8)

        fig.tight_layout()

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

    2D: single panel.
    3D: axial + meridional panels (lime contour for necrosis boundary).
    """
    from visualization.field_plots import plot_temperature_field

    X, Y, Z, xv, yv, zv = build_mesh(cfg)

    if cfg.domain.ndim == 2:
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        plot_temperature_field(T_flat, t=t, cfg=cfg, ax=ax, fig=fig)
        Omega_2d = unflatten(Omega_flat, cfg)
        try:
            cs = ax.contour(xv * 100, yv * 100, Omega_2d,
                            levels=[1.0], colors=['lime'], linewidths=[3.0])
            ax.clabel(cs, fmt='Ω_d=1', fontsize=9)
        except Exception:
            pass
        ax.set_title(f'T field + necrosis boundary   t = {t:.1f} s', fontsize=12)

    else:
        # Reuse the 3D temperature figure and overlay necrosis contours
        fig = plot_temperature_field(T_flat, t=t, cfg=cfg)
        axes = fig.get_axes()
        ax_h, ax_v = axes[0], axes[1]

        Omega_3d = unflatten(Omega_flat, cfg)
        horiz_O, vert_O, _, _ = _extract_slices(Omega_3d, cfg)

        for ax_plt, xvec, yvec, field in [
                (ax_h, xv, yv, horiz_O),
                (ax_v, xv, zv, vert_O)]:
            try:
                cs = ax_plt.contour(xvec * 100, yvec * 100, field,
                                    levels=[1.0], colors=['lime'], linewidths=[2.5])
                ax_plt.clabel(cs, fmt='Ω=1', fontsize=8)
            except Exception:
                pass

        fig.suptitle(f'T field + necrosis boundary   t = {t:.1f} s',
                     fontsize=13, y=1.01)

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

    Om_nd = unflatten(Omega_flat, cfg)

    if cfg.domain.ndim == 2:
        Nx, Ny = cfg.domain.Nx, cfg.domain.Ny
        grid = pv.ImageData()
        grid.dimensions = (Nx, Ny, 1)
        grid.spacing    = (cfg.domain.dx*100, cfg.domain.dy*100, 1.0)
        grid.point_data['Omega_d'] = Om_nd.ravel(order='F')
    else:
        Nx, Ny, Nz = cfg.domain.Nx, cfg.domain.Ny, cfg.domain.Nz
        grid = pv.ImageData()
        grid.dimensions = (Nx, Ny, Nz)
        grid.spacing    = (cfg.domain.dx*100, cfg.domain.dy*100, cfg.domain.dz*100)
        grid.point_data['Omega_d'] = Om_nd.transpose(2, 1, 0).ravel(order='F')

    pl = pv.Plotter()
    pl.add_volume(grid, cmap='RdYlGn_r', opacity='linear',
                  clim=[0, 2], scalar_bar_args={'title': 'Ω_d'})
    isosurf = grid.contour([1.0], scalars='Omega_d')
    if isosurf.n_points > 0:
        pl.add_mesh(isosurf, color='black', opacity=0.8,
                    label='Necrosis boundary (Ω=1)')
    pl.add_axes()
    pl.show_bounds(xlabel='x [cm]', ylabel='y [cm]', zlabel='z [cm]')
    pl.show()

"""
tests/test_02_sar.py — SAR heat source model tests.

Covers all three probe models (point, line, dipole) in 2D and 3D.

Run standalone:
    python tests/test_02_sar.py
Run with pytest:
    pytest tests/test_02_sar.py -v
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
from physics.mesh import build_mesh
from physics.sar_model import compute_sar_field, probe_needle_endpoints


# ── Shared helpers ────────────────────────────────────────────────────────────

def _row_profile(sar, xv, cfg, axis='x'):
    """Extract 1-D profile through probe centre along x or y (2D) or axially (3D)."""
    xp, yp, zp = cfg.domain.probe_position
    if cfg.domain.ndim == 2:
        if axis == 'x':
            row = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny - 1))
            return xv, sar[row, :]
        else:
            col = int(np.clip(round(xp / cfg.domain.dx), 0, cfg.domain.Nx - 1))
            from physics.mesh import build_mesh as _bm
            _, _, _, _, yv, _ = _bm(cfg)
            return yv, sar[:, col]
    else:
        iz = int(np.clip(round(zp / cfg.domain.dz), 0, cfg.domain.Nz - 1))
        iy = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny - 1))
        return xv, sar[iz, iy, :]


def _needle_in_plane(cfg, xaxis, yaxis):
    """Return True when the active zone has visible extent in the (xaxis, yaxis) plane."""
    if cfg.sar.probe_model != 'line':
        return False
    d3 = np.asarray(cfg.sar.probe_direction, dtype=float)
    d3 = d3 / np.linalg.norm(d3)
    coord = {'x': 0, 'y': 1, 'z': 2}
    # The line is visible only if d3 has a component in either axis of the plot plane
    return abs(d3[coord[xaxis]]) > 1e-6 or abs(d3[coord[yaxis]]) > 1e-6


def _draw_needle(ax, cfg, xaxis='x', yaxis='y'):
    """Draw the active needle extent as a line on a 2D axes."""
    if not _needle_in_plane(cfg, xaxis, yaxis):
        return
    start, end = probe_needle_endpoints(cfg)
    coord = {'x': 0, 'y': 1, 'z': 2}
    xs = [start[coord[xaxis]] * 100, end[coord[xaxis]] * 100]
    ys = [start[coord[yaxis]] * 100, end[coord[yaxis]] * 100]
    ax.plot(xs, ys, 'w-', linewidth=3, alpha=0.8, label='Active zone')
    ax.plot(xs, ys, 'c--', linewidth=1.5, alpha=0.9)


# ── Test functions ────────────────────────────────────────────────────────────

def test_sar_field(ndim=2, probe_model='point'):
    """
    Checks the SAR field for one probe model and dimensionality.

    Assertions:
      - Correct shape and non-negative values
      - Peak within a grid diagonal of the probe centre
      - Peak value close to cfg.sar.sar_peak (±20%; dipole peaks at σ, not centre)
      - Model-specific: line has flat SAR inside active zone;
                        dipole has zero SAR on the needle axis
    """
    cfg = make_cfg(ndim, small=False)
    cfg.sar.probe_model = probe_model

    sar = compute_sar_field(cfg=cfg)
    X, Y, Z, xv, yv, zv = build_mesh(cfg)
    xp, yp, zp = cfg.domain.probe_position

    # ── Shape ─────────────────────────────────────────────────────────────
    if ndim == 2:
        assert sar.shape == (cfg.domain.Ny, cfg.domain.Nx)
    else:
        assert sar.shape == (cfg.domain.Nz, cfg.domain.Ny, cfg.domain.Nx)
    assert np.all(sar >= 0), "SAR must be non-negative"

    # ── Model-specific assertions ──────────────────────────────────────────
    sigma_r = cfg.sar.sigma_sar
    d3 = np.asarray(cfg.sar.probe_direction, dtype=float)
    d3 = d3 / np.linalg.norm(d3)
    d2t = d3[:2]; n2 = np.linalg.norm(d2t)
    d2  = d2t / n2 if n2 > 1e-10 else np.array([0.0, 1.0])
    # True when needle is perpendicular to the 2-D x-y plane (z-dominant direction)
    needle_perp_2d = (ndim == 2 and n2 < 1e-10)

    if probe_model == 'point':
        # Peak must be within 3 grid diagonals of probe centre
        grid_diag = np.sqrt(cfg.domain.dx**2 + cfg.domain.dy**2)
        peak_idx = np.unravel_index(sar.argmax(), sar.shape)
        if ndim == 2:
            px_err = np.sqrt((X[peak_idx] - xp)**2 + (Y[peak_idx] - yp)**2)
        else:
            px_err = np.sqrt((X[peak_idx] - xp)**2 + (Y[peak_idx] - yp)**2
                             + (Z[peak_idx] - zp)**2)
        assert px_err < grid_diag * 3, \
            f"Point SAR peak {px_err*100:.2f} cm from probe centre"
        np.testing.assert_allclose(sar.max(), cfg.sar.sar_peak, rtol=0.05)

    elif probe_model == 'line':
        # SAR at the probe centre must equal sar_peak (whole active zone is flat at peak)
        if ndim == 2:
            ic = int(np.clip(round(xp / cfg.domain.dx), 0, cfg.domain.Nx-1))
            ir = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny-1))
            centre_sar = sar[ir, ic]
        else:
            ic = int(np.clip(round(xp / cfg.domain.dx), 0, cfg.domain.Nx-1))
            ir = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny-1))
            iz = int(np.clip(round(zp / cfg.domain.dz), 0, cfg.domain.Nz-1))
            centre_sar = sar[iz, ir, ic]
        np.testing.assert_allclose(centre_sar, cfg.sar.sar_peak, rtol=0.05,
                                    err_msg="Line SAR at probe centre should be sar_peak")

    elif probe_model == 'dipole':
        # On the needle axis (displaced along d̂ by σ), SAR should be near zero.
        # In the equatorial plane (displaced perpendicularly by σ), SAR ≈ sar_peak.
        # Special case: 2-D with needle ⊥ to slice — the whole x-y plane is equatorial
        # (sin²θ = 1 everywhere), so there is no in-plane axial null to check.
        if needle_perp_2d:
            # Pattern is isotropic: verify isotropy and that SAR at r=σ ≈ sar_peak
            ix_eq = int(np.clip(round((xp + sigma_r) / cfg.domain.dx), 0, cfg.domain.Nx-1))
            iy_c  = int(np.clip(round(yp  / cfg.domain.dy), 0, cfg.domain.Ny-1))
            ix_c  = int(np.clip(round(xp  / cfg.domain.dx), 0, cfg.domain.Nx-1))
            iy_eq = int(np.clip(round((yp + sigma_r) / cfg.domain.dy), 0, cfg.domain.Ny-1))
            sar_x = sar[iy_c,  ix_eq]   # r=σ along x
            sar_y = sar[iy_eq, ix_c ]   # r=σ along y
            np.testing.assert_allclose(sar_x, sar_y, rtol=0.05,
                err_msg="Dipole 2D (needle⊥slice): SAR should be isotropic")
            assert sar_x > cfg.sar.sar_peak * 0.5, \
                f"Dipole 2D (needle⊥slice): SAR at r=σ should be ~sar_peak, got {sar_x:.3e}"
        elif ndim == 2:
            # Axial point: probe_center + σ * d2 (along needle direction in plane)
            ax_x = xp + sigma_r * d2[0]
            ax_y = yp + sigma_r * d2[1]
            ix_ax = int(np.clip(round(ax_x / cfg.domain.dx), 0, cfg.domain.Nx-1))
            iy_ax = int(np.clip(round(ax_y / cfg.domain.dy), 0, cfg.domain.Ny-1))
            sar_on_axis = sar[iy_ax, ix_ax]
            # Equatorial point: probe_center + σ * perp (perpendicular to d2)
            perp = np.array([-d2[1], d2[0]])
            eq_x = xp + sigma_r * perp[0]
            eq_y = yp + sigma_r * perp[1]
            ix_eq = int(np.clip(round(eq_x / cfg.domain.dx), 0, cfg.domain.Nx-1))
            iy_eq = int(np.clip(round(eq_y / cfg.domain.dy), 0, cfg.domain.Ny-1))
            sar_equatorial = sar[iy_eq, ix_eq]
            assert sar_on_axis < cfg.sar.sar_peak * 0.1, \
                f"Dipole SAR on needle axis should be ~0, got {sar_on_axis:.3e}"
            assert sar_equatorial > cfg.sar.sar_peak * 0.5, \
                f"Dipole SAR in equatorial plane should be high, got {sar_equatorial:.3e}"
        else:
            ax_x = xp + sigma_r * d3[0]
            ax_y = yp + sigma_r * d3[1]
            ax_z = zp + sigma_r * d3[2]
            ix_ax = int(np.clip(round(ax_x / cfg.domain.dx), 0, cfg.domain.Nx-1))
            iy_ax = int(np.clip(round(ax_y / cfg.domain.dy), 0, cfg.domain.Ny-1))
            iz_ax = int(np.clip(round(ax_z / cfg.domain.dz), 0, cfg.domain.Nz-1))
            sar_on_axis = sar[iz_ax, iy_ax, ix_ax]
            # Equatorial: displace in x-direction (perpendicular to z-axis)
            ix_eq = int(np.clip(round((xp + sigma_r) / cfg.domain.dx), 0, cfg.domain.Nx-1))
            sar_equatorial = sar[int(np.clip(round(zp / cfg.domain.dz), 0, cfg.domain.Nz-1)),
                                  int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny-1)),
                                  ix_eq]
            assert sar_on_axis < cfg.sar.sar_peak * 0.1, \
                f"Dipole SAR on needle axis should be ~0, got {sar_on_axis:.3e}"
            assert sar_equatorial > cfg.sar.sar_peak * 0.5, \
                f"Dipole SAR in equatorial plane should be high, got {sar_equatorial:.3e}"

    # ── Line-specific: flat axial profile inside active zone ──────────────
    if probe_model == 'line':
        half_L = cfg.sar.L_active / 2.0
        sigma_r = cfg.sar.sigma_sar
        if needle_perp_2d:
            # Needle goes into the page: the 2-D slice shows an isotropic disk
            # (cross-section of the cylinder).  No axial flat zone visible in x-y.
            # Just verify isotropy: SAR at (xp+σ, yp) ≈ SAR at (xp, yp+σ).
            ix_r = int(np.clip(round((xp + sigma_r) / cfg.domain.dx), 0, cfg.domain.Nx-1))
            iy_c = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny-1))
            ix_c = int(np.clip(round(xp / cfg.domain.dx), 0, cfg.domain.Nx-1))
            iy_r = int(np.clip(round((yp + sigma_r) / cfg.domain.dy), 0, cfg.domain.Ny-1))
            np.testing.assert_allclose(sar[iy_c, ix_r], sar[iy_r, ix_c], rtol=0.05,
                err_msg="Line 2D (needle⊥slice): cross-section should be isotropic")
        elif ndim == 2:
            # Needle lies in the 2-D plane: check flat plateau along needle axis
            ix = int(np.clip(round(xp / cfg.domain.dx), 0, cfg.domain.Nx-1))
            axial_profile = sar[:, ix]
            iy_c = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny-1))
            y_in = yp + half_L * 0.8
            iy_in = int(np.clip(round(y_in / cfg.domain.dy), 0, cfg.domain.Ny-1))
            ratio = axial_profile[iy_in] / axial_profile[iy_c]
            assert ratio > 0.9, \
                f"Line SAR should be flat inside active zone; ratio={ratio:.3f}"
        else:
            iz_c = int(np.clip(round(zp / cfg.domain.dz), 0, cfg.domain.Nz-1))
            iy_c = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny-1))
            ix_c = int(np.clip(round(xp / cfg.domain.dx), 0, cfg.domain.Nx-1))
            z_in = zp + half_L * 0.8
            iz_in = int(np.clip(round(z_in / cfg.domain.dz), 0, cfg.domain.Nz-1))
            ratio = sar[iz_in, iy_c, ix_c] / sar[iz_c, iy_c, ix_c]
            assert ratio > 0.9, \
                f"Line SAR should be flat inside active zone (3D); ratio={ratio:.3f}"

    print(f"  [{probe_model:6s} ndim={ndim}]  "
          f"SAR max={sar.max():.3e} W/kg  shape={sar.shape}")

    # ── Figure ────────────────────────────────────────────────────────────
    _plot_sar(sar, cfg, ndim, probe_model, xv, yv, zv, xp, yp, zp)


def _plot_sar(sar, cfg, ndim, probe_model, xv, yv, zv, xp, yp, zp):
    """Generate and save the SAR visualisation figure."""
    sigma_r = cfg.sar.sigma_sar

    if ndim == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        X, Y, _, _, _, _ = build_mesh(cfg)

        im = axes[0].pcolormesh(X*100, Y*100, sar/1e3, cmap='hot', shading='auto')
        fig.colorbar(im, ax=axes[0], label='SAR [kW/kg]')
        axes[0].plot(xp*100, yp*100, 'b+', markersize=14, markeredgewidth=3, label='Probe')
        _draw_needle(axes[0], cfg, xaxis='x', yaxis='y')
        for mult, ls, lbl in [(1, '--', '1σ_r'), (2, ':', '2σ_r')]:
            axes[0].add_patch(plt.Circle((xp*100, yp*100), mult*sigma_r*100,
                                          fill=False, edgecolor='cyan',
                                          linewidth=1.2, linestyle=ls, label=lbl))
        axes[0].legend(fontsize=8)
        axes[0].set_xlabel('x [cm]'); axes[0].set_ylabel('y [cm]')
        axes[0].set_title(f'SAR field — {probe_model} model', fontsize=11)
        axes[0].set_aspect('equal')

        # Radial (x) profile
        xp_arr, prof_x = _row_profile(sar, xv, cfg, axis='x')
        # Axial (y) profile
        yp_arr, prof_y = _row_profile(sar, xv, cfg, axis='y')
        axes[1].plot(xp_arr*100, prof_x/1e3, 'r-', lw=2, label='Radial (x, y=probe_y)')
        axes[1].plot(yp_arr*100, prof_y/1e3, 'b-', lw=2, label='Axial (y, x=probe_x)')
        if probe_model == 'line':
            half_L = cfg.sar.L_active / 2.0
            axes[1].axvspan((yp - half_L)*100, (yp + half_L)*100,
                            alpha=0.12, color='blue', label='Active zone')
        axes[1].axvline(xp*100, color='gray', linestyle=':', linewidth=1)
        axes[1].set_xlabel('Position [cm]'); axes[1].set_ylabel('SAR [kW/kg]')
        axes[1].set_title('Radial vs axial profiles', fontsize=11)
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    else:
        # 3D: axial slice + meridional slice + profiles
        iz = int(np.clip(round(zp / cfg.domain.dz), 0, cfg.domain.Nz-1))
        iy = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny-1))
        ix = int(np.clip(round(xp / cfg.domain.dx), 0, cfg.domain.Nx-1))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Axial slice (x-y at z=probe_z)
        im0 = axes[0].pcolormesh(xv*100, yv*100, sar[iz]/1e3,
                                  cmap='hot', shading='auto')
        fig.colorbar(im0, ax=axes[0], label='SAR [kW/kg]')
        axes[0].plot(xp*100, yp*100, 'b+', markersize=12, markeredgewidth=3)
        axes[0].set_xlabel('x [cm]'); axes[0].set_ylabel('y [cm]')
        axes[0].set_title(f'Axial  z={zv[iz]*100:.1f} cm', fontsize=11)
        axes[0].set_aspect('equal')

        # Meridional slice (x-z at y=probe_y)
        im1 = axes[1].pcolormesh(xv*100, zv*100, sar[:, iy, :]/1e3,
                                  cmap='hot', shading='auto')
        fig.colorbar(im1, ax=axes[1], label='SAR [kW/kg]')
        _draw_needle(axes[1], cfg, xaxis='x', yaxis='z')
        axes[1].plot(xp*100, zp*100, 'b+', markersize=12, markeredgewidth=3)
        axes[1].set_xlabel('x [cm]'); axes[1].set_ylabel('z [cm]')
        axes[1].set_title(f'Meridional  y={yv[iy]*100:.1f} cm', fontsize=11)
        axes[1].set_aspect('equal')

        # Radial and axial profiles
        prof_x = sar[iz, iy, :]            # radial (x at z=probe_z, y=probe_y)
        prof_z = sar[:, iy, ix]            # axial  (z at x=probe_x, y=probe_y)
        axes[2].plot(xv*100, prof_x/1e3, 'r-', lw=2, label='Radial (x)')
        axes[2].plot(zv*100, prof_z/1e3, 'b-', lw=2, label='Axial (z)')
        if probe_model == 'line':
            half_L = cfg.sar.L_active / 2.0
            axes[2].axvspan((zp - half_L)*100, (zp + half_L)*100,
                            alpha=0.12, color='blue', label='Active zone')
        axes[2].axvline(xp*100, color='gray', linestyle=':', linewidth=1)
        axes[2].set_xlabel('Position [cm]'); axes[2].set_ylabel('SAR [kW/kg]')
        axes[2].set_title('Radial (x) vs axial (z) profiles', fontsize=11)
        axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Test 02: SAR Model — {probe_model}  (ndim={ndim})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'02_sar_{probe_model}_{ndim}d.png', cfg)


# ── pytest entry points ───────────────────────────────────────────────────────

def test_sar_point_2d():    test_sar_field(ndim=2, probe_model='point')
def test_sar_point_3d():    test_sar_field(ndim=3, probe_model='point')
def test_sar_line_2d():     test_sar_field(ndim=2, probe_model='line')
def test_sar_line_3d():     test_sar_field(ndim=3, probe_model='line')
def test_sar_dipole_2d():   test_sar_field(ndim=2, probe_model='dipole')
def test_sar_dipole_3d():   test_sar_field(ndim=3, probe_model='dipole')


def _plot_comparison(ndim=2):
    """
    Overlay axial + radial 1-D profiles for all three models on one figure.
    This makes the structural differences obvious at a glance.
    """
    models  = ['point', 'line', 'dipole']
    colors  = {'point': '#e41a1c', 'line': '#377eb8', 'dipole': '#4daf4a'}
    styles  = {'point': '-',       'line': '--',      'dipole': ':'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for model in models:
        cfg = make_cfg(ndim, small=False)
        cfg.sar.probe_model = model
        sar = compute_sar_field(cfg=cfg)
        X, Y, Z, xv, yv, zv = build_mesh(cfg)
        xp, yp, zp = cfg.domain.probe_position

        c, ls = colors[model], styles[model]

        if ndim == 2:
            # Radial (x at y=probe_y)
            iy = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny-1))
            prof_r = sar[iy, :]
            axes[0].plot((xv - xp)*100, prof_r/1e3, color=c, ls=ls, lw=2, label=model)
            # Axial (y at x=probe_x)
            ix = int(np.clip(round(xp / cfg.domain.dx), 0, cfg.domain.Nx-1))
            prof_a = sar[:, ix]
            axes[1].plot((yv - yp)*100, prof_a/1e3, color=c, ls=ls, lw=2, label=model)
        else:
            iz = int(np.clip(round(zp / cfg.domain.dz), 0, cfg.domain.Nz-1))
            iy = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny-1))
            ix = int(np.clip(round(xp / cfg.domain.dx), 0, cfg.domain.Nx-1))
            prof_r = sar[iz, iy, :]
            axes[0].plot((xv - xp)*100, prof_r/1e3, color=c, ls=ls, lw=2, label=model)
            prof_a = sar[:, iy, ix]
            axes[1].plot((zv - zp)*100, prof_a/1e3, color=c, ls=ls, lw=2, label=model)

    # Annotate active zone on axial plot
    cfg0 = make_cfg(ndim, small=False)
    half_L = cfg0.sar.L_active / 2.0
    axes[1].axvspan(-half_L*100, half_L*100, alpha=0.10, color='blue', label='Line active zone')
    axes[1].axvline(0, color='gray', lw=0.8, ls=':')

    for ax, title, xlabel in [
        (axes[0], 'Radial profile  (⊥ to needle axis)', 'Radial offset [cm]'),
        (axes[1], 'Axial profile  (∥ to needle axis)',  'Axial offset from probe centre [cm]'),
    ]:
        ax.set_xlabel(xlabel); ax.set_ylabel('SAR [kW/kg]')
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.axvline(0, color='gray', lw=0.8, ls=':')

    fig.suptitle(f'Test 02: SAR model comparison — {ndim}D', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'02_sar_comparison_{ndim}d.png', cfg0)
    print(f"  [saved] comparison figure for ndim={ndim}")


def _plot_dipole_3d_shape():
    """
    Standalone 3D shape visualisation for the dipole SAR model.

    The dipole pattern is SAR = peak * e^(1/2) * sin²(θ) * exp(-r²/2σ²) where θ
    is the angle from the needle axis (z).  This creates:
      - Zero SAR on the needle axis (θ = 0, π)  → null on z-axis away from probe
      - Max SAR in the equatorial plane (θ = π/2)
      - Combined 3D shape: oblate spheroid with polar dimples (NOT a classic torus —
        the centre is not hollow because the code sets sin²=1 at r=0)

    Four panels:
      1. 3D scatter of voxels above 15% of peak (shows the 3D volume)
      2. Meridional x-z slice  → classic two-lobe pattern
      3. Axial x-y slice       → circular (all directions equatorial)
      4. Far-field polar plot  sin²(θ) — the directional factor in isolation
    """
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3D projection)

    cfg = make_cfg(ndim=3, small=True)   # 15^3 — fast enough for scatter
    cfg.sar.probe_model = 'dipole'
    sar = compute_sar_field(cfg=cfg)
    X, Y, Z, xv, yv, zv = build_mesh(cfg)
    xp, yp, zp = cfg.domain.probe_position

    iz = int(np.clip(round(zp / cfg.domain.dz), 0, cfg.domain.Nz - 1))
    iy = int(np.clip(round(yp / cfg.domain.dy), 0, cfg.domain.Ny - 1))

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle('Dipole SAR — 3D shape  (needle axis = z,  pattern = sin²θ · e^(−r²/2σ²))',
                 fontsize=12, fontweight='bold')

    # ── Panel 1: 3D scatter above threshold ──────────────────────────────
    ax3d = fig.add_subplot(1, 4, 1, projection='3d')
    thresh = sar.max() * 0.15
    mask   = sar.ravel() > thresh
    sc = ax3d.scatter(X.ravel()[mask] * 100,
                      Y.ravel()[mask] * 100,
                      Z.ravel()[mask] * 100,
                      c=sar.ravel()[mask], cmap='hot', s=12, alpha=0.35,
                      vmin=0, vmax=sar.max())
    # Draw needle axis through probe
    ax3d.plot([xp*100, xp*100], [yp*100, yp*100],
              [zv[0]*100, zv[-1]*100], 'b-', lw=2, label='Needle axis')
    ax3d.set_xlabel('x [cm]'); ax3d.set_ylabel('y [cm]'); ax3d.set_zlabel('z [cm]')
    ax3d.set_title(f'SAR > 15% max  ({mask.sum()} voxels)', fontsize=9)
    fig.colorbar(sc, ax=ax3d, label='SAR [W/kg]', shrink=0.55, pad=0.1)
    ax3d.legend(fontsize=7)

    # ── Panel 2: Meridional x-z slice (shows two-lobe null on axis) ──────
    ax2 = fig.add_subplot(1, 4, 2)
    im2 = ax2.pcolormesh(xv*100, zv*100, sar[:, iy, :]/1e3,
                         cmap='hot', shading='auto')
    fig.colorbar(im2, ax=ax2, label='SAR [kW/kg]')
    ax2.axhline(zp*100, color='cyan', lw=0.8, ls=':', alpha=0.7)
    ax2.axvline(xp*100, color='cyan', lw=0.8, ls=':', alpha=0.7)
    ax2.plot(xp*100, zp*100, 'b+', ms=12, mew=2.5)
    # Mark the null region on the z-axis (above/below probe)
    ax2.annotate('null\n(θ=0)', xy=(xp*100, (zp + 0.015)*100),
                 xytext=(xp*100 + 0.8, (zp + 0.015)*100),
                 fontsize=7, color='cyan',
                 arrowprops=dict(arrowstyle='->', color='cyan', lw=0.8))
    ax2.set_xlabel('x [cm]'); ax2.set_ylabel('z [cm]')
    ax2.set_title('Meridional x-z slice\n(two lobes, null on needle axis)', fontsize=9)
    ax2.set_aspect('equal')

    # ── Panel 3: Axial x-y slice at z=probe_z (equatorial — circular) ────
    ax3 = fig.add_subplot(1, 4, 3)
    im3 = ax3.pcolormesh(xv*100, yv*100, sar[iz]/1e3,
                         cmap='hot', shading='auto')
    fig.colorbar(im3, ax=ax3, label='SAR [kW/kg]')
    ax3.plot(xp*100, yp*100, 'b+', ms=12, mew=2.5)
    ax3.set_xlabel('x [cm]'); ax3.set_ylabel('y [cm]')
    ax3.set_title('Axial x-y slice  (z = probe)\nequatorial plane → circular', fontsize=9)
    ax3.set_aspect('equal')

    # ── Panel 4: Far-field polar pattern sin²(θ) ─────────────────────────
    ax4 = fig.add_subplot(1, 4, 4, projection='polar')
    theta = np.linspace(0, 2 * np.pi, 720)
    r_ff  = np.sin(theta) ** 2
    ax4.plot(theta, r_ff, 'r-', lw=2, label='sin²θ')
    ax4.fill(theta, r_ff, alpha=0.15, color='red')
    ax4.set_title('Far-field directional\nfactor  sin²θ', fontsize=9, pad=14)
    ax4.set_rticks([0.5, 1.0])
    ax4.annotate('θ = 0\n(null)', xy=(0, 1.05), xycoords='data',
                 fontsize=7, color='navy', ha='center')
    ax4.annotate('θ = 90°\n(max)', xy=(np.pi/2, 1.12), xycoords='data',
                 fontsize=7, color='crimson', ha='center')

    fig.tight_layout()
    save_fig(fig, '02_sar_dipole_3d_shape.png', cfg)
    print("  [saved] dipole 3D shape figure  →  02_sar_dipole_3d_shape.png")
    print("  Note: combined 3D pattern is NOT a classic hollow torus.")
    print("        Centre has high SAR (sin²=1 by convention at r=0).")
    print("        Null only exists ON the z-axis AWAY from probe (θ=0 or π, r>0).")


if __name__ == '__main__':
    from tests._helpers import run_suite, OUT
    import itertools

    passed = failed = 0
    for ndim, model in itertools.product([2, 3], ['point', 'line', 'dipole']):
        print(f"\n{'─'*60}")
        print(f"Running test_sar_field  (ndim={ndim}, probe_model={model}) ...")
        try:
            test_sar_field(ndim=ndim, probe_model=model)
            print(f"  PASSED [OK]")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAILED  {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Figures: {OUT}/")

    print(f"\n── Comparison figures ──────────────────────────────────────────")
    _plot_comparison(ndim=2)
    _plot_comparison(ndim=3)

    # print(f"\n── Dipole 3D shape ─────────────────────────────────────────────")
    # _plot_dipole_3d_shape()

"""
tests/test_physics_visual.py — Physics module tests with visual output.

Each test function:
  1. Exercises a physics module with assertions (correctness checks)
  2. Saves a figure to results/tests/ for visual inspection

Run all tests and generate figures:
    pytest tests/test_physics_visual.py -v

Run standalone (also opens interactive windows):
    python tests/test_physics_visual.py --show

Figure output: results/tests/*.png
"""

import os
import sys

# Force UTF-8 output so unicode math symbols render on any platform
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for pytest
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

# ── Make project root importable when running standalone ─────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import cfg

from physics.mesh import build_mesh, build_region_masks, unflatten
from physics.sar_model import compute_sar_field, get_control_input_vector
from physics.discretization import (build_mass_matrix, build_diffusion_matrix,
                                     build_perfusion_matrix, build_system_matrix)
from physics.boundary_conditions import (apply_dirichlet, apply_neumann,
                                          apply_robin, apply_boundary_conditions)
from physics.bioheat import BioHeatSolver
from physics.arrhenius import ArrheniusDamage

from visualization.field_plots import plot_temperature_field
from visualization.damage_plots import plot_damage_field

# ── Output directory ──────────────────────────────────────────────────────────
OUT = os.path.join(_ROOT, 'results', 'tests')
os.makedirs(OUT, exist_ok=True)

_SHOW = '--show' in sys.argv


def _save(fig, name: str):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=cfg.viz.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Test 1 — Mesh geometry (physics/mesh.py)
# ═════════════════════════════════════════════════════════════════════════════

def test_mesh_geometry():
    """
    Checks grid dimensions, region mask disjointness, and completeness.

    Visual: domain partition map, tumor mask, and distance field.
    """
    X, Y, _, xv, yv, _ = build_mesh(cfg)
    tumor, healthy, margin = build_region_masks(cfg)

    # ── Assertions ────────────────────────────────────────────────────────
    assert X.shape == (cfg.domain.Ny, cfg.domain.Nx)
    assert Y.shape == (cfg.domain.Ny, cfg.domain.Nx)
    assert xv.shape == (cfg.domain.Nx,)
    assert yv.shape == (cfg.domain.Ny,)

    assert not np.any(tumor & healthy),  "tumor/healthy overlap"
    assert not np.any(tumor & margin),   "tumor/margin overlap"
    assert not np.any(margin & healthy), "margin/healthy overlap"
    assert (tumor | healthy | margin).all(), "some voxels unclassified"

    assert tumor.sum() > 0
    assert tumor.sum() < cfg.domain.N
    assert healthy.sum() > tumor.sum()

    dx_actual = (xv[-1] - xv[0]) / (cfg.domain.Nx - 1)
    dy_actual = (yv[-1] - yv[0]) / (cfg.domain.Ny - 1)
    np.testing.assert_allclose(xv[1] - xv[0], dx_actual, rtol=1e-9)
    np.testing.assert_allclose(yv[1] - yv[0], dy_actual, rtol=1e-9)

    print(f"  Tumor: {tumor.sum()} voxels ({100*tumor.mean():.1f}%)")
    print(f"  Margin: {margin.sum()} voxels ({100*margin.mean():.1f}%)")
    print(f"  Healthy: {healthy.sum()} voxels ({100*healthy.mean():.1f}%)")

    # ── Figure ────────────────────────────────────────────────────────────
    region_map = np.zeros_like(X, dtype=float)
    region_map[tumor]   = 2.0
    region_map[margin]  = 1.0
    region_map[healthy] = 0.0

    cx, cy = cfg.domain.tumor_center[:2]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: combined region map
    cmap_r = matplotlib.colors.ListedColormap(['#2ecc71', '#f1c40f', '#e74c3c'])
    axes[0].pcolormesh(X*100, Y*100, region_map, cmap=cmap_r,
                       vmin=0, vmax=2, shading='auto')
    for r_val, ls in [(cfg.domain.tumor_radius, '--'),
                      (cfg.domain.tumor_radius + cfg.domain.safety_margin, ':')]:
        axes[0].add_patch(plt.Circle((cx*100, cy*100), r_val*100,
                                     fill=False, edgecolor='black',
                                     linewidth=2, linestyle=ls))
    patches = [mpatches.Patch(color='#2ecc71', label='Healthy Ω_H'),
               mpatches.Patch(color='#f1c40f', label='Margin Ω_M'),
               mpatches.Patch(color='#e74c3c', label='Tumor Ω_T')]
    axes[0].legend(handles=patches, fontsize=8)
    axes[0].set_title('Domain partition  Ω = Ω_T ∪ Ω_M ∪ Ω_H', fontsize=11)
    axes[0].set_xlabel('x [cm]'); axes[0].set_ylabel('y [cm]')
    axes[0].set_aspect('equal')

    # Panel 2: tumor mask
    axes[1].pcolormesh(X*100, Y*100, tumor.astype(float),
                       cmap='Reds', shading='auto', vmin=0, vmax=1)
    axes[1].set_title(f'Tumor mask  ({tumor.sum()} voxels)', fontsize=11)
    axes[1].set_xlabel('x [cm]'); axes[1].set_aspect('equal')

    # Panel 3: distance field
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2) * 100
    im3 = axes[2].pcolormesh(X*100, Y*100, dist, cmap='viridis', shading='auto')
    axes[2].contour(X*100, Y*100, dist,
                    levels=[cfg.domain.tumor_radius*100,
                            (cfg.domain.tumor_radius + cfg.domain.safety_margin)*100],
                    colors=['white', 'yellow'], linewidths=1.5)
    fig.colorbar(im3, ax=axes[2], label='Distance from center [cm]')
    axes[2].set_title('Distance field', fontsize=11)
    axes[2].set_xlabel('x [cm]'); axes[2].set_aspect('equal')

    fig.suptitle('Test 1: Mesh Geometry', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, '01_mesh_geometry.png')


# ═════════════════════════════════════════════════════════════════════════════
# Test 2 — SAR heat source model (physics/sar_model.py)
# ═════════════════════════════════════════════════════════════════════════════

def test_sar_field():
    """
    Checks that the Gaussian SAR field peaks at the probe and decays correctly.

    Visual: 2D SAR heatmap and radial profile vs expected Gaussian.
    """
    sar = compute_sar_field(cfg=cfg)
    X, Y, _, _, _, _ = build_mesh(cfg)
    xp, yp = cfg.domain.probe_position[:2]
    sigma  = cfg.sar.sigma_sar

    # ── Assertions ────────────────────────────────────────────────────────
    assert sar.shape == (cfg.domain.Ny, cfg.domain.Nx)
    assert np.all(sar >= 0)

    peak_idx = np.unravel_index(sar.argmax(), sar.shape)
    dist_error = np.sqrt((X[peak_idx] - xp)**2 + (Y[peak_idx] - yp)**2)
    dx_grid = (X[0, -1] - X[0, 0]) / (cfg.domain.Nx - 1)
    assert dist_error < dx_grid * 2, \
        f"SAR peak {dist_error*100:.2f} cm from probe"

    np.testing.assert_allclose(sar.max(), cfg.sar.sar_peak, rtol=0.01)

    # Value at 1σ should be ~0.607 of peak
    row = cfg.domain.Ny // 2
    col_1sigma = np.argmin(np.abs(X[row, :] - (xp + sigma)))
    ratio = sar[row, col_1sigma] / sar.max()
    assert 0.4 < ratio < 0.75, f"Gaussian 1σ ratio = {ratio:.3f}, expected ~0.607"

    print(f"  SAR peak: {sar.max():.3e} W/kg  at probe ({xp*100:.1f}, {yp*100:.1f}) cm")
    print(f"  Value at 1σ: ratio = {ratio:.3f}  (expected 0.607)")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].pcolormesh(X*100, Y*100, sar/1e3, cmap='hot', shading='auto')
    fig.colorbar(im, ax=axes[0], label='SAR [kW/kg]')
    axes[0].plot(xp*100, yp*100, 'b+', markersize=14, markeredgewidth=3,
                 label='Probe')
    for mult, ls, label in [(1, '--', '1σ'), (2, ':', '2σ'), (3, '-.', '3σ')]:
        axes[0].add_patch(plt.Circle((xp*100, yp*100), mult*sigma*100,
                                     fill=False, edgecolor='cyan',
                                     linewidth=1.5, linestyle=ls, label=label))
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel('x [cm]'); axes[0].set_ylabel('y [cm]')
    axes[0].set_title('SAR field  [W/kg per W applied]', fontsize=11)
    axes[0].set_aspect('equal')

    x_cm    = X[row, :] * 100
    x_m     = X[row, :]
    expected = cfg.sar.sar_peak * np.exp(-(x_m - xp)**2 / (2*sigma**2))
    axes[1].plot(x_cm, sar[row, :] / 1e3, 'r-', linewidth=2, label='Simulated')
    axes[1].plot(x_cm, expected / 1e3, 'k--', linewidth=1.5, alpha=0.7,
                 label='Expected Gaussian')
    axes[1].axvline(xp*100,          color='blue', linestyle='--', linewidth=1, label='Probe')
    axes[1].axvline((xp + sigma)*100, color='cyan', linestyle=':',  linewidth=1, label='1σ')
    axes[1].axvline((xp - sigma)*100, color='cyan', linestyle=':',  linewidth=1)
    axes[1].set_xlabel('x [cm]'); axes[1].set_ylabel('SAR [kW/kg]')
    axes[1].set_title('Radial SAR profile (horizontal slice)', fontsize=11)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle('Test 2: SAR Heat Source Model', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, '02_sar_field.png')


# ═════════════════════════════════════════════════════════════════════════════
# Test 3 — Discretization matrices (physics/discretization.py)
# ═════════════════════════════════════════════════════════════════════════════

def test_discretization_matrices():
    """
    Checks shape, symmetry of K_d, diagonal values of M and W_b,
    and that M_inv_A has negative diagonal (stable system).

    Visual: K_d sparsity pattern, M_inv_A diagonal as a 2D field, K_d row sums.
    """
    M       = build_mass_matrix(cfg)
    K_d     = build_diffusion_matrix(cfg)
    W_b     = build_perfusion_matrix(cfg)
    M_inv_A, M_inv = build_system_matrix(cfg)
    N = cfg.domain.N

    # ── Assertions ────────────────────────────────────────────────────────
    assert M.shape     == (N, N)
    assert K_d.shape   == (N, N)
    assert W_b.shape   == (N, N)
    assert M_inv_A.shape == (N, N)

    np.testing.assert_allclose(M.diagonal(),
                                cfg.tissue.rho * cfg.tissue.c, rtol=1e-10)

    diff_sym = (K_d - K_d.T).data
    if len(diff_sym) > 0:
        assert np.max(np.abs(diff_sym)) < 1e-10, "K_d not symmetric"

    assert np.all(K_d.diagonal() <= 0), "K_d diagonal must be ≤ 0"

    np.testing.assert_allclose(W_b.diagonal(),
                                cfg.tissue.omega_b * cfg.tissue.rho_b * cfg.tissue.c_b,
                                rtol=1e-10)

    assert np.all(M_inv_A.diagonal() < 0), "M_inv_A diagonal must be < 0 (stable)"

    print(f"  K_d: {K_d.shape},  nnz = {K_d.nnz}")
    print(f"  K_d diagonal: [{K_d.diagonal().min():.3e}, {K_d.diagonal().max():.3e}]")
    print(f"  M_inv_A diagonal: [{M_inv_A.diagonal().min():.3e}, "
          f"{M_inv_A.diagonal().max():.3e}]")

    # ── Figure ────────────────────────────────────────────────────────────
    X, Y, _, _, _, _ = build_mesh(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].spy(K_d, markersize=0.5, color='navy')
    axes[0].set_title(f'K_d sparsity pattern  ({K_d.nnz} non-zeros)', fontsize=11)
    axes[0].set_xlabel('column'); axes[0].set_ylabel('row')

    diag_2d = M_inv_A.diagonal().reshape(cfg.domain.Ny, cfg.domain.Nx)
    im2 = axes[1].pcolormesh(X*100, Y*100, diag_2d, cmap='RdBu_r', shading='auto')
    fig.colorbar(im2, ax=axes[1], label='(M⁻¹A)_ii  [s⁻¹]')
    axes[1].set_title('M⁻¹A diagonal\n(system eigenvalue estimate)', fontsize=11)
    axes[1].set_xlabel('x [cm]'); axes[1].set_ylabel('y [cm]')
    axes[1].set_aspect('equal')

    rowsum = np.array(K_d.sum(axis=1)).ravel().reshape(cfg.domain.Ny, cfg.domain.Nx)
    im3 = axes[2].pcolormesh(X*100, Y*100, rowsum, cmap='bwr', shading='auto')
    fig.colorbar(im3, ax=axes[2], label='Row sum of K_d  [W/(m²·K)]')
    axes[2].set_title('K_d row sums\n(≈0 interior, nonzero at BC corners)', fontsize=11)
    axes[2].set_xlabel('x [cm]'); axes[2].set_ylabel('y [cm]')
    axes[2].set_aspect('equal')

    fig.suptitle('Test 3: Discretization Matrices', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, '03_discretization_matrices.png')


# ═════════════════════════════════════════════════════════════════════════════
# Test 4 — Boundary conditions (physics/boundary_conditions.py)
# ═════════════════════════════════════════════════════════════════════════════

def test_boundary_conditions():
    """
    Checks Dirichlet pins boundary voxels, Neumann is identity,
    Robin interpolates between interior and ambient.

    Visual: T field before/after each BC type.
    """
    N  = cfg.domain.N
    Nx = cfg.domain.Nx
    Ny = cfg.domain.Ny
    T_warm = np.full(N, 60.0)

    # Dirichlet
    T_val = cfg.tissue.T_blood
    T_dir = apply_dirichlet(T_warm, T_val, cfg)
    T_dir2d = T_dir.reshape(Ny, Nx)
    assert np.all(T_dir2d[0,  :] == T_val), "Dirichlet top row"
    assert np.all(T_dir2d[-1, :] == T_val), "Dirichlet bottom row"
    assert np.all(T_dir2d[:,  0] == T_val), "Dirichlet left col"
    assert np.all(T_dir2d[:, -1] == T_val), "Dirichlet right col"
    assert T_dir2d[Ny//2, Nx//2] == 60.0,   "Interior unchanged"

    # Neumann
    T_neu = apply_neumann(T_warm, cfg)
    np.testing.assert_array_equal(T_neu, T_warm, "Neumann must be identity")

    # Robin
    h_c, T_inf = 50.0, 20.0
    T_rob = apply_robin(T_warm, h_c=h_c, T_inf=T_inf, cfg=cfg)
    T_rob2d = T_rob.reshape(Ny, Nx)
    assert np.all(T_rob2d[0, :] < 60.0), "Robin should cool boundary"
    assert np.all(T_rob2d[0, :] > T_inf), "Robin should not over-cool"

    print(f"  Dirichlet: boundary set to {T_val}°C [OK]")
    print(f"  Neumann: identity [OK]")
    print(f"  Robin: boundary range = "
          f"[{T_rob2d[0,:].min():.1f}, {T_rob2d[0,:].max():.1f}] °C [OK]")

    # ── Figure ────────────────────────────────────────────────────────────
    X, Y, _, _, _, _ = build_mesh(cfg)
    norm = Normalize(vmin=15, vmax=65)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    labels = ['Input (60°C uniform)', f'Dirichlet\n(T_boundary={T_val}°C)',
              'Neumann (identity)', f'Robin\n(h_c={h_c}, T_inf={T_inf}°C)']
    fields = [T_warm, T_dir, T_neu, T_rob]

    for ax, T, label in zip(axes, fields, labels):
        im = ax.pcolormesh(X*100, Y*100, T.reshape(Ny, Nx),
                           cmap='hot', norm=norm, shading='auto')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('x [cm]'); ax.set_aspect('equal')
    axes[0].set_ylabel('y [cm]')
    fig.colorbar(im, ax=axes, label='Temperature [°C]', fraction=0.01, pad=0.04)

    fig.suptitle('Test 4: Boundary Conditions', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, '04_boundary_conditions.png')


# ═════════════════════════════════════════════════════════════════════════════
# Test 5 — BioHeatSolver: no-power equilibrium (physics/bioheat.py)
# ═════════════════════════════════════════════════════════════════════════════

def test_bioheat_no_power():
    """
    With P=0 the domain should remain near T_blood (perfusion + metabolic
    heat balance).  No unphysical drift allowed.

    Visual: T field at t=0 vs t=100s and probe voxel time trace.
    """
    sar     = compute_sar_field(cfg=cfg)
    bioheat = BioHeatSolver(cfg=cfg, sar_field=sar)
    T       = bioheat.initialize()
    n_steps = 100
    dt      = cfg.solver.dt

    T_hist   = [T.copy()]
    T_center = [T[cfg.domain.N // 2]]

    for _ in range(n_steps):
        T = bioheat.step(T, P=0.0, dt=dt)
        T_hist.append(T.copy())
        T_center.append(T[cfg.domain.N // 2])

    # ── Assertions ────────────────────────────────────────────────────────
    assert T.min() >= cfg.tissue.T_blood - 2.0, "Too cold at P=0"
    assert T.max() <= cfg.tissue.T_blood + 10.0, "Too hot at P=0"
    assert np.all(T >= 30.0), "Temperature < 30°C is unphysical"
    assert np.all(T <= 50.0), "Temperature > 50°C without power is unphysical"

    print(f"  T range after {n_steps}s (P=0): "
          f"[{T.min():.2f}, {T.max():.2f}] °C")

    # ── Figure ────────────────────────────────────────────────────────────
    X, Y, _, _, _, _ = build_mesh(cfg)
    norm = Normalize(vmin=36, vmax=42)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (idx, label) in zip(axes[:2], [(0, 't = 0s'), (n_steps, f't = {n_steps}s')]):
        im = ax.pcolormesh(X*100, Y*100, T_hist[idx].reshape(cfg.domain.Ny, cfg.domain.Nx),
                           cmap='hot', norm=norm, shading='auto')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]')
        ax.set_aspect('equal')
    fig.colorbar(im, ax=axes[:2], label='Temperature [°C]', fraction=0.02)

    t_ax = np.arange(n_steps + 1) * dt
    axes[2].plot(t_ax, T_center, 'r-', linewidth=2, label='Probe voxel')
    axes[2].axhline(cfg.tissue.T_blood, color='blue', linestyle='--',
                    linewidth=1, label=f'T_blood = {cfg.tissue.T_blood}°C')
    axes[2].set_xlabel('Time [s]'); axes[2].set_ylabel('Temperature [°C]')
    axes[2].set_title('Probe voxel temperature (P = 0 W)', fontsize=11)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    fig.suptitle('Test 5: BioHeatSolver — No-Power Equilibrium',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, '05_bioheat_no_power.png')


# ═════════════════════════════════════════════════════════════════════════════
# Test 6 — BioHeatSolver: heating dynamics (physics/bioheat.py)
# ═════════════════════════════════════════════════════════════════════════════

def test_bioheat_heating():
    """
    Applies P_max for 120s.  Checks monotone heating at probe, Dirichlet BCs
    pin the boundary, and healthy tissue temperature rises sensibly.

    Visual: temperature snapshots at 4 time points + time history traces.
    """
    sar = compute_sar_field(cfg=cfg)
    tumor_mask, healthy_mask, _ = build_region_masks(cfg)
    tm = tumor_mask.ravel()
    hm = healthy_mask.ravel()

    n_steps = 120
    dt      = cfg.solver.dt
    P_max   = cfg.control.P_max
    t_vec   = np.arange(n_steps + 1) * dt

    bioheat = BioHeatSolver(cfg=cfg, sar_field=sar)
    T = bioheat.initialize()
    T_hist        = [T.copy()]
    T_max_tumor   = [T[tm].max()]
    T_max_healthy = [T[hm].max()]

    for _ in range(n_steps):
        T = bioheat.step(T, P=P_max, dt=dt)
        T_hist.append(T.copy())
        T_max_tumor.append(T[tm].max())
        T_max_healthy.append(T[hm].max())

    # ── Assertions ────────────────────────────────────────────────────────
    center = cfg.domain.N // 2
    assert T_hist[-1][tm].mean() > cfg.tissue.T_init + 10.0, \
        "Tumor should heat by at least 10°C"

    for k in range(60):
        assert T_hist[k+1][center] >= T_hist[k][center] - 0.1, \
            f"Temperature dipped at step {k}"

    Nx, Ny = cfg.domain.Nx, cfg.domain.Ny
    T_final_2d = T_hist[-1].reshape(Ny, Nx)
    bc_vals = np.concatenate([T_final_2d[0, :], T_final_2d[-1, :],
                               T_final_2d[:, 0], T_final_2d[:, -1]])
    assert bc_vals.max() <= cfg.tissue.T_blood + 2.0, \
        "Boundary should stay near T_blood (Dirichlet BC)"

    print(f"  Tumor center T after {n_steps}s: {T_hist[-1][center]:.1f} °C")
    print(f"  Max healthy T after {n_steps}s:  {T_hist[-1][hm].max():.1f} °C")
    print(f"  Boundary T max: {bc_vals.max():.2f} °C")

    # ── Figure ────────────────────────────────────────────────────────────
    snap_steps = [0, 30, 60, 120]
    fig = plt.figure(figsize=(16, 9))

    for col, s in enumerate(snap_steps):
        ax = fig.add_subplot(2, 4, col + 1)
        plot_temperature_field(T_hist[s], t=s*dt, cfg=cfg, ax=ax, fig=fig)
        ax.set_title(f't = {s}s', fontsize=11)

    ax_hist = fig.add_subplot(2, 1, 2)
    ax_hist.plot(t_vec, T_max_tumor,   'r-', linewidth=2, label='max T (tumor)')
    ax_hist.plot(t_vec, T_max_healthy, 'b-', linewidth=2, label='max T (healthy)')
    ax_hist.axhline(cfg.control.T_safe, color='orange', linestyle='--',
                    linewidth=1.5, label=f'T_safe = {cfg.control.T_safe}°C')
    ax_hist.axhline(60.0, color='red', linestyle=':', linewidth=1,
                    label='60°C ablation threshold', alpha=0.7)
    ax_hist.set_xlabel('Time [s]'); ax_hist.set_ylabel('Temperature [°C]')
    ax_hist.set_title(f'Temperature histories  (P = {P_max} W constant)', fontsize=11)
    ax_hist.legend(fontsize=9); ax_hist.grid(True, alpha=0.3)

    fig.suptitle('Test 6: BioHeatSolver — Heating Dynamics',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, '06_bioheat_heating.png')


# ═════════════════════════════════════════════════════════════════════════════
# Test 7 — Arrhenius rate vs temperature (physics/arrhenius.py)
# ═════════════════════════════════════════════════════════════════════════════

def test_arrhenius_rate():
    """
    Checks rate is near-zero at 37°C, large at 60°C, and monotone in T.
    Verifies the time-to-ablate at 60°C is clinically reasonable (seconds to minutes).

    Visual: rate and time-to-ablate curves from 37°C to 100°C.
    """
    damage = ArrheniusDamage(cfg=cfg)
    N      = cfg.domain.N

    rate_37 = damage.rate(np.full(N, 37.0))[0]
    rate_50 = damage.rate(np.full(N, 50.0))[0]
    rate_60 = damage.rate(np.full(N, 60.0))[0]
    rate_80 = damage.rate(np.full(N, 80.0))[0]

    # ── Assertions ────────────────────────────────────────────────────────
    assert rate_37 >= 0
    assert rate_60 > rate_50 > rate_37, "Rate must be monotone in T"
    assert rate_80 > rate_60

    if rate_37 > 0:
        assert 1.0 / rate_37 > 1e6, "Time to ablate at 37°C must be >> 1M s"

    t_ablate_60 = 1.0 / rate_60
    assert 0.1 < t_ablate_60 < 600.0, \
        f"Time to ablate at 60°C = {t_ablate_60:.1f}s (expected 0.1–600s)"

    print(f"  Rate at 37°C: {rate_37:.3e} s⁻¹")
    print(f"  Rate at 60°C: {rate_60:.3e} s⁻¹  "
          f"(time to Ω=1: {t_ablate_60:.1f} s = {t_ablate_60/60:.2f} min)")
    print(f"  Rate at 80°C: {rate_80:.3e} s⁻¹")

    # ── Figure ────────────────────────────────────────────────────────────
    T_range = np.linspace(37, 100, 300)
    rates   = np.array([damage.rate(np.full(N, t))[0] for t in T_range])
    t_ablate = np.where(rates > 0, 1.0 / rates, np.inf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].semilogy(T_range, rates, 'r-', linewidth=2)
    for T_mark, color, label in [(37, 'green', '37°C body temp'),
                                  (43, 'orange', '43°C threshold'),
                                  (60, 'red', '60°C ablation')]:
        axes[0].axvline(T_mark, color=color, linestyle=':', linewidth=1.5,
                        label=label)
    axes[0].set_xlabel('Temperature [°C]')
    axes[0].set_ylabel('dΩ/dt  [s⁻¹]')
    axes[0].set_title('Arrhenius damage rate\n'
                      r'$\frac{d\Omega}{dt} = A \exp\!\left(-\frac{E_a}{RT}\right)$',
                      fontsize=11)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, which='both')

    finite = t_ablate < 1e10
    axes[1].semilogy(T_range[finite], t_ablate[finite] / 60.0, 'b-', linewidth=2)
    axes[1].axhline(10, color='orange', linestyle='--', linewidth=1, label='10 min')
    axes[1].axhline(1,  color='red',    linestyle='--', linewidth=1, label='1 min')
    axes[1].axvline(60, color='gray',   linestyle=':',  linewidth=1, label='60°C')
    axes[1].set_xlabel('Temperature [°C]')
    axes[1].set_ylabel('Time to Ω_d = 1  [minutes]')
    axes[1].set_title('Time to irreversible necrosis\n(Ω_d ≥ 1 threshold)', fontsize=11)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, which='both')

    fig.suptitle('Test 7: Arrhenius Rate Temperature Dependence',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, '07_arrhenius_rate.png')


# ═════════════════════════════════════════════════════════════════════════════
# Test 8 — Arrhenius damage accumulation (physics/arrhenius.py)
# ═════════════════════════════════════════════════════════════════════════════

def test_arrhenius_accumulation():
    """
    Simulates damage at constant 60°C.  Checks Omega is monotone and
    crosses the necrosis threshold close to the analytical prediction.

    Visual: Omega field at crossing time, final field, and Omega vs time.
    """
    damage  = ArrheniusDamage(cfg=cfg)
    N       = cfg.domain.N
    T_const = np.full(N, 60.0)
    rate    = damage.rate(T_const)[0]
    t_analytical = 1.0 / rate

    dt      = cfg.solver.dt
    n_steps = min(int(t_analytical * 2.5) + 1, 600)

    Omega   = damage.initialize()
    Om_hist = [Omega.copy()]
    Om_center = [0.0]

    for _ in range(n_steps):
        Omega = damage.step(Omega, T_const, dt=dt)
        Om_hist.append(Omega.copy())
        Om_center.append(Omega[N // 2])

    t_vec = np.arange(n_steps + 1) * dt

    # ── Assertions ────────────────────────────────────────────────────────
    assert Om_hist[0].max() == 0.0, "Initial damage must be 0"

    for k in range(n_steps):
        assert Om_hist[k+1][N//2] >= Om_hist[k][N//2] - 1e-12, \
            f"Damage decreased at step {k}"

    crossed = next((k for k, Om in enumerate(Om_hist) if Om[N//2] >= 1.0), None)
    assert crossed is not None, "Ω never reached threshold 1.0"
    t_crossed = crossed * dt
    assert t_crossed < t_analytical * 2.0, \
        f"Crossed at {t_crossed:.1f}s vs analytical {t_analytical:.1f}s"

    print(f"  Rate at 60°C: {rate:.4e} s⁻¹")
    print(f"  Analytical t(Ω=1): {t_analytical:.2f}s")
    print(f"  Simulated  t(Ω=1): {t_crossed:.2f}s  "
          f"(ratio = {t_crossed/t_analytical:.3f})")

    # ── Figure ────────────────────────────────────────────────────────────
    X, Y, _, _, _, _ = build_mesh(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Omega field at crossing
    if crossed is not None:
        Om_2d = Om_hist[crossed].reshape(cfg.domain.Ny, cfg.domain.Nx)
        im = axes[0].pcolormesh(X*100, Y*100, Om_2d,
                                cmap='RdYlGn_r', vmin=0, vmax=2, shading='auto')
        fig.colorbar(im, ax=axes[0], label='Ω_d')
        axes[0].contour(X*100, Y*100, Om_2d, levels=[1.0],
                        colors='black', linewidths=2)
        axes[0].set_title(f'Ω_d at t={t_crossed:.0f}s\n(Ω=1 crossing)', fontsize=10)
    axes[0].set_xlabel('x [cm]'); axes[0].set_ylabel('y [cm]')
    axes[0].set_aspect('equal')

    Om_final = Om_hist[-1].reshape(cfg.domain.Ny, cfg.domain.Nx)
    im2 = axes[1].pcolormesh(X*100, Y*100, Om_final,
                              cmap='RdYlGn_r', vmin=0, vmax=2, shading='auto')
    fig.colorbar(im2, ax=axes[1], label='Ω_d')
    axes[1].contour(X*100, Y*100, Om_final, levels=[1.0],
                    colors='black', linewidths=2)
    axes[1].set_title(f'Final Ω_d  t={n_steps}s', fontsize=10)
    axes[1].set_xlabel('x [cm]'); axes[1].set_aspect('equal')

    axes[2].plot(t_vec, Om_center, 'r-', linewidth=2, label='Ω_d probe voxel')
    axes[2].axhline(1.0, color='black', linestyle='--', linewidth=2,
                    label='Threshold Ω=1')
    axes[2].axvline(t_analytical, color='blue', linestyle=':',
                    label=f'Analytical: {t_analytical:.1f}s')
    if crossed:
        axes[2].axvline(t_crossed, color='orange', linestyle=':',
                        label=f'Simulated: {t_crossed:.1f}s')
    axes[2].set_xlabel('Time [s]'); axes[2].set_ylabel('Ω_d')
    axes[2].set_title('Ω_d accumulation at 60°C', fontsize=11)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    fig.suptitle('Test 8: Arrhenius Damage Accumulation (T = 60°C constant)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, '08_arrhenius_accumulation.png')


# ═════════════════════════════════════════════════════════════════════════════
# Test 9 — Coupled BioHeat + Arrhenius (full state)
# ═════════════════════════════════════════════════════════════════════════════

def test_coupled_bioheat_arrhenius():
    """
    Runs both state equations coupled for 180s at P_max.
    Checks damage grows, temperatures remain physical, and
    ablation progress makes sense.

    Visual: T and Ω fields at t=60s and t=180s, plus time-history traces.
    """
    sar     = compute_sar_field(cfg=cfg)
    bioheat = BioHeatSolver(cfg=cfg, sar_field=sar)
    damage  = ArrheniusDamage(cfg=cfg)
    tumor_mask, healthy_mask, _ = build_region_masks(cfg)
    tm = tumor_mask.ravel()
    hm = healthy_mask.ravel()

    n_steps = 180
    dt      = cfg.solver.dt
    P_max   = cfg.control.P_max
    t_vec   = np.arange(n_steps + 1) * dt

    T     = bioheat.initialize()
    Omega = damage.initialize()
    T_hist  = [T.copy()]
    Om_hist = [Omega.copy()]
    T_max_tumor   = [T[tm].max()]
    T_max_healthy = [T[hm].max()]
    frac_ablated  = [0.0]

    for _ in range(n_steps):
        T     = bioheat.step(T, P=P_max, dt=dt)
        Omega = damage.step(Omega, T, dt=dt)
        T_hist.append(T.copy())
        Om_hist.append(Omega.copy())
        T_max_tumor.append(T[tm].max())
        T_max_healthy.append(T[hm].max())
        frac_ablated.append((Omega[tm] >= 1.0).mean())

    # ── Assertions ────────────────────────────────────────────────────────
    assert Om_hist[-1][tm].max() > 0.1, "Tumor damage should exceed 0.1 after 180s"
    assert T_hist[-1].max() < 200.0, "Temperature > 200°C is unphysical"
    assert T_hist[-1].min() > 30.0,  "Temperature < 30°C is unphysical"

    center = cfg.domain.N // 2
    for k in range(n_steps):
        assert Om_hist[k+1][center] >= Om_hist[k][center] - 1e-12, \
            f"Damage decreased at step {k}"

    print(f"  T max tumor at 180s:    {T_hist[-1][tm].max():.1f} °C")
    print(f"  T max healthy at 180s:  {T_hist[-1][hm].max():.1f} °C")
    print(f"  Ω max tumor at 180s:    {Om_hist[-1][tm].max():.4f}")
    print(f"  Fraction ablated:       {frac_ablated[-1]:.1%}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    snaps = [(60, '60s'), (180, '180s')]

    for col, (s, label) in enumerate(snaps):
        ax_t = fig.add_subplot(3, 2, col + 1)
        plot_temperature_field(T_hist[s], t=s, cfg=cfg, ax=ax_t, fig=fig)
        ax_t.set_title(f'Temperature  t = {label}', fontsize=11)

        ax_d = fig.add_subplot(3, 2, col + 3)
        plot_damage_field(Om_hist[s], t=s, cfg=cfg, ax=ax_d, fig=fig)
        ax_d.set_title(f'Damage Ω_d  t = {label}', fontsize=11)

    ax_t2 = fig.add_subplot(3, 2, 5)
    ax_t2.plot(t_vec, T_max_tumor,   'r-', linewidth=2, label='max T (tumor)')
    ax_t2.plot(t_vec, T_max_healthy, 'b-', linewidth=2, label='max T (healthy)')
    ax_t2.axhline(cfg.control.T_safe, color='orange', linestyle='--',
                  linewidth=1.5, label=f'T_safe={cfg.control.T_safe}°C')
    ax_t2.axhline(60.0, color='red', linestyle=':', linewidth=1, alpha=0.7,
                  label='60°C ablation')
    ax_t2.set_xlabel('Time [s]'); ax_t2.set_ylabel('Temperature [°C]')
    ax_t2.set_title('Temperature histories', fontsize=11)
    ax_t2.legend(fontsize=8); ax_t2.grid(True, alpha=0.3)

    ax_ab = fig.add_subplot(3, 2, 6)
    ax_ab.plot(t_vec, np.array(frac_ablated) * 100, 'darkred', linewidth=2)
    ax_ab.fill_between(t_vec, np.array(frac_ablated) * 100, alpha=0.2, color='darkred')
    ax_ab.axhline(95, color='green', linestyle='--', linewidth=1.5,
                  label='95% clinical target')
    ax_ab.set_xlabel('Time [s]'); ax_ab.set_ylabel('Tumor ablated [%]')
    ax_ab.set_title('Ablation progress', fontsize=11)
    ax_ab.set_ylim(-2, 103)
    ax_ab.legend(fontsize=9); ax_ab.grid(True, alpha=0.3)

    fig.suptitle('Test 9: Coupled BioHeat + Arrhenius  (P = P_max constant)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, '09_coupled_bioheat_arrhenius.png')


# ═════════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if _SHOW:
        matplotlib.use('TkAgg')

    tests = [
        test_mesh_geometry,
        test_sar_field,
        test_discretization_matrices,
        test_boundary_conditions,
        test_bioheat_no_power,
        test_bioheat_heating,
        test_arrhenius_rate,
        test_arrhenius_accumulation,
        test_coupled_bioheat_arrhenius,
    ]

    passed = failed = 0
    for fn in tests:
        print(f"\n{'─'*60}")
        print(f"Running {fn.__name__} ...")
        try:
            fn()
            print(f"  PASSED [OK]")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED [!!]  AssertionError: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR  [!!]  {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'═'*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Figures: {OUT}/")

    if _SHOW:
        plt.show()

"""
tests/test_05_bioheat.py — BioHeatSolver tests.

Test A: no-power equilibrium — domain stays near T_blood.
Test B: heating dynamics — monotone rise at P_max, Dirichlet BC pinning.

Run standalone:
    python tests/test_05_bioheat.py
Run with pytest:
    pytest tests/test_05_bioheat.py -v
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
from physics.mesh import build_mesh, build_region_masks, unflatten
from physics.sar_model import compute_sar_field
from physics.bioheat import BioHeatSolver
from visualization.field_plots import plot_temperature_field


def test_bioheat_no_power(ndim=2):
    """
    With P=0 the domain should remain near T_blood (perfusion + metabolic
    heat balance).  No unphysical drift allowed.

    Visual: T field at t=0 vs t=100s and probe voxel time trace.
    """
    cfg = make_cfg(ndim, small=True)
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
    assert T.min() >= cfg.tissue.T_blood - 2.0, \
        f"Too cold at P=0: T.min()={T.min():.2f}"
    assert T.max() <= cfg.tissue.T_blood + 10.0, \
        f"Too hot at P=0: T.max()={T.max():.2f}"
    assert np.all(T >= 30.0), "Temperature < 30°C is unphysical"
    assert np.all(T <= 50.0), "Temperature > 50°C without power is unphysical"

    print(f"  ndim={ndim}  T range after {n_steps}s (P=0): "
          f"[{T.min():.2f}, {T.max():.2f}] °C")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))

    ax0 = fig.add_subplot(1, 3, 1)
    plot_temperature_field(T_hist[0], t=0.0, cfg=cfg, ax=ax0, fig=fig)
    ax0.set_title('t = 0 s', fontsize=11)

    ax1 = fig.add_subplot(1, 3, 2)
    plot_temperature_field(T_hist[n_steps], t=n_steps*dt, cfg=cfg, ax=ax1, fig=fig)
    ax1.set_title(f't = {n_steps} s', fontsize=11)

    ax2 = fig.add_subplot(1, 3, 3)
    t_ax = np.arange(n_steps + 1) * dt
    ax2.plot(t_ax, T_center, 'r-', linewidth=2, label='Probe voxel')
    ax2.axhline(cfg.tissue.T_blood, color='blue', linestyle='--',
                linewidth=1, label=f'T_blood = {cfg.tissue.T_blood}°C')
    ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Temperature [°C]')
    ax2.set_title('Probe voxel  (P = 0 W)', fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Test 05a: BioHeatSolver — No-Power Equilibrium  (ndim={ndim})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'05a_bioheat_no_power_{ndim}d.png', cfg)


def test_bioheat_heating(ndim=2):
    """
    Applies P_max for 120s.  Checks monotone heating at probe, Dirichlet BCs
    pin the boundary, and healthy tissue temperature rises sensibly.

    Visual: temperature snapshots at 4 time points + time history traces.
    """
    cfg   = make_cfg(ndim, small=True)
    sar   = compute_sar_field(cfg=cfg)
    tumor_mask, healthy_mask, _ = build_region_masks(cfg)
    tm = tumor_mask.ravel()
    hm = healthy_mask.ravel()

    n_steps = 120
    dt    = cfg.solver.dt
    P_max = cfg.control.P_max
    t_vec = np.arange(n_steps + 1) * dt

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
        "Tumor should heat by ≥ 10°C"

    for k in range(60):
        assert T_hist[k+1][center] >= T_hist[k][center] - 0.1, \
            f"Temperature dipped at step {k}"

    T_final_nd = unflatten(T_hist[-1], cfg)
    if ndim == 2:
        bc_vals = np.concatenate([T_final_nd[0, :], T_final_nd[-1, :],
                                   T_final_nd[:, 0], T_final_nd[:, -1]])
    else:
        bc_vals = np.concatenate([T_final_nd[:, :, 0].ravel(),
                                   T_final_nd[:, :, -1].ravel(),
                                   T_final_nd[:, 0, :].ravel(),
                                   T_final_nd[:, -1, :].ravel(),
                                   T_final_nd[0, :, :].ravel(),
                                   T_final_nd[-1, :, :].ravel()])
    assert bc_vals.max() <= cfg.tissue.T_blood + 2.0, \
        f"Boundary should stay near T_blood (got {bc_vals.max():.1f}°C)"

    print(f"  ndim={ndim}")
    print(f"  Tumor center T at {n_steps}s: {T_hist[-1][center]:.1f} °C")
    print(f"  Max healthy T at {n_steps}s:  {T_hist[-1][hm].max():.1f} °C")
    print(f"  Boundary T max: {bc_vals.max():.2f} °C")

    # ── Figure ────────────────────────────────────────────────────────────
    snap_steps = [0, 30, 60, 120]
    fig = plt.figure(figsize=(16, 9))

    for col, s in enumerate(snap_steps):
        ax = fig.add_subplot(2, 4, col + 1)
        plot_temperature_field(T_hist[s], t=s*dt, cfg=cfg, ax=ax, fig=fig)
        ax.set_title(f't = {s}s', fontsize=10)

    ax_hist = fig.add_subplot(2, 1, 2)
    ax_hist.plot(t_vec, T_max_tumor,   'r-', linewidth=2, label='max T (tumor)')
    ax_hist.plot(t_vec, T_max_healthy, 'b-', linewidth=2, label='max T (healthy)')
    ax_hist.axhline(cfg.control.T_safe, color='orange', linestyle='--',
                    linewidth=1.5, label=f'T_safe = {cfg.control.T_safe}°C')
    ax_hist.axhline(60.0, color='red', linestyle=':', linewidth=1,
                    label='60°C ablation', alpha=0.7)
    ax_hist.set_xlabel('Time [s]'); ax_hist.set_ylabel('Temperature [°C]')
    ax_hist.set_title(f'Temperature histories  (P = {P_max} W constant)', fontsize=11)
    ax_hist.legend(fontsize=9); ax_hist.grid(True, alpha=0.3)

    fig.suptitle(f'Test 05b: BioHeatSolver — Heating Dynamics  (ndim={ndim})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'05b_bioheat_heating_{ndim}d.png', cfg)


# ── pytest entry points ───────────────────────────────────────────────────────

def test_bioheat_no_power_2d():
    test_bioheat_no_power(ndim=2)

def test_bioheat_no_power_3d():
    test_bioheat_no_power(ndim=3)

def test_bioheat_heating_2d():
    test_bioheat_heating(ndim=2)

def test_bioheat_heating_3d():
    test_bioheat_heating(ndim=3)


if __name__ == '__main__':
    from tests._helpers import run_suite
    run_suite([test_bioheat_no_power, test_bioheat_heating])

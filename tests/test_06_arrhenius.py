"""
tests/test_06_arrhenius.py — Arrhenius damage model tests.

Test A: rate vs temperature — monotone, physiologically reasonable.
Test B: damage accumulation at 60°C — crosses threshold near analytical prediction.

Run standalone:
    python tests/test_06_arrhenius.py
Run with pytest:
    pytest tests/test_06_arrhenius.py -v
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
from physics.arrhenius import ArrheniusDamage


def test_arrhenius_rate(ndim=2):
    """
    Checks rate is near-zero at 37°C, large at 60°C, and monotone in T.
    Verifies time-to-ablate at 60°C is clinically reasonable.

    Visual: rate and time-to-ablate curves from 37°C to 100°C.
    (Rate curve is dimensionality-independent; ndim only changes N.)
    """
    cfg    = make_cfg(ndim, small=True)
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

    print(f"  ndim={ndim}  Rate at 37°C: {rate_37:.3e} s⁻¹")
    print(f"  Rate at 60°C: {rate_60:.3e} s⁻¹  "
          f"(time to Ω=1: {t_ablate_60:.1f}s = {t_ablate_60/60:.2f} min)")
    print(f"  Rate at 80°C: {rate_80:.3e} s⁻¹")

    # ── Figure (rate curve is the same regardless of ndim) ────────────────
    T_range = np.linspace(37, 100, 300)
    rates   = np.array([damage.rate(np.full(N, t))[0] for t in T_range])
    t_ablate = np.where(rates > 0, 1.0 / rates, np.inf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].semilogy(T_range, rates, 'r-', linewidth=2)
    for T_mark, color, label in [(37, 'green', '37°C body temp'),
                                  (43, 'orange', '43°C threshold'),
                                  (60, 'red',    '60°C ablation')]:
        axes[0].axvline(T_mark, color=color, linestyle=':', linewidth=1.5, label=label)
    axes[0].set_xlabel('Temperature [°C]')
    axes[0].set_ylabel('dΩ/dt  [s⁻¹]')
    axes[0].set_title(r'Arrhenius rate  $A\exp(-E_a/RT)$', fontsize=11)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, which='both')

    finite = t_ablate < 1e10
    axes[1].semilogy(T_range[finite], t_ablate[finite] / 60.0, 'b-', linewidth=2)
    axes[1].axhline(10, color='orange', linestyle='--', linewidth=1, label='10 min')
    axes[1].axhline(1,  color='red',    linestyle='--', linewidth=1, label='1 min')
    axes[1].axvline(60, color='gray',   linestyle=':',  linewidth=1, label='60°C')
    axes[1].set_xlabel('Temperature [°C]')
    axes[1].set_ylabel('Time to Ω_d = 1  [min]')
    axes[1].set_title('Time to irreversible necrosis', fontsize=11)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, which='both')

    fig.suptitle(f'Test 06a: Arrhenius Rate vs Temperature  (ndim={ndim})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'06a_arrhenius_rate_{ndim}d.png', cfg)


def test_arrhenius_accumulation(ndim=2):
    """
    Simulates damage at constant 60°C.  Checks Omega is monotone and
    crosses the necrosis threshold close to the analytical prediction.

    Visual: Ω_d field at crossing time, final field, and Ω vs time.
    """
    cfg    = make_cfg(ndim, small=True)
    damage = ArrheniusDamage(cfg=cfg)
    N      = cfg.domain.N
    T_const = np.full(N, 60.0)
    rate    = damage.rate(T_const)[0]
    t_analytical = 1.0 / rate

    dt      = cfg.solver.dt
    n_steps = min(int(t_analytical * 2.5) + 1, 600)

    Omega    = damage.initialize()
    Om_hist  = [Omega.copy()]
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

    print(f"  ndim={ndim}  Rate at 60°C: {rate:.4e} s⁻¹")
    print(f"  Analytical t(Ω=1): {t_analytical:.2f}s")
    print(f"  Simulated  t(Ω=1): {t_crossed:.2f}s  "
          f"(ratio = {t_crossed/t_analytical:.3f})")

    # ── Figure ────────────────────────────────────────────────────────────
    X, Y, Z, xv, yv, zv = build_mesh(cfg)

    if ndim == 2:
        def _sl(Om_flat):
            return unflatten(Om_flat, cfg), xv, yv
    else:
        iz = cfg.domain.Nz // 2
        def _sl(Om_flat):
            return unflatten(Om_flat, cfg)[iz], xv, yv

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    Om_cross, xp, yp = _sl(Om_hist[crossed])
    im = axes[0].pcolormesh(xp*100, yp*100, Om_cross,
                             cmap='RdYlGn_r', vmin=0, vmax=2, shading='auto')
    fig.colorbar(im, ax=axes[0], label='Ω_d')
    axes[0].contour(xp*100, yp*100, Om_cross, levels=[1.0],
                    colors='black', linewidths=2)
    axes[0].set_title(f'Ω_d at crossing  t={t_crossed:.0f}s', fontsize=10)
    axes[0].set_xlabel('x [cm]'); axes[0].set_ylabel('y [cm]')
    axes[0].set_aspect('equal')

    Om_fin, xp2, yp2 = _sl(Om_hist[-1])
    im2 = axes[1].pcolormesh(xp2*100, yp2*100, Om_fin,
                               cmap='RdYlGn_r', vmin=0, vmax=2, shading='auto')
    fig.colorbar(im2, ax=axes[1], label='Ω_d')
    axes[1].contour(xp2*100, yp2*100, Om_fin, levels=[1.0],
                    colors='black', linewidths=2)
    axes[1].set_title(f'Final Ω_d  t={n_steps}s', fontsize=10)
    axes[1].set_xlabel('x [cm]'); axes[1].set_aspect('equal')

    axes[2].plot(t_vec, Om_center, 'r-', linewidth=2, label='Ω_d probe voxel')
    axes[2].axhline(1.0, color='black', linestyle='--', linewidth=2, label='Threshold Ω=1')
    axes[2].axvline(t_analytical, color='blue', linestyle=':',
                    label=f'Analytical: {t_analytical:.1f}s')
    axes[2].axvline(t_crossed, color='orange', linestyle=':',
                    label=f'Simulated: {t_crossed:.1f}s')
    axes[2].set_xlabel('Time [s]'); axes[2].set_ylabel('Ω_d')
    axes[2].set_title('Ω_d accumulation at 60°C', fontsize=11)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Test 06b: Arrhenius Accumulation T=60°C  (ndim={ndim})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'06b_arrhenius_accumulation_{ndim}d.png', cfg)


# ── pytest entry points ───────────────────────────────────────────────────────

def test_arrhenius_rate_2d():
    test_arrhenius_rate(ndim=2)

def test_arrhenius_rate_3d():
    test_arrhenius_rate(ndim=3)

def test_arrhenius_accumulation_2d():
    test_arrhenius_accumulation(ndim=2)

def test_arrhenius_accumulation_3d():
    test_arrhenius_accumulation(ndim=3)


if __name__ == '__main__':
    from tests._helpers import run_suite
    run_suite([test_arrhenius_rate, test_arrhenius_accumulation])

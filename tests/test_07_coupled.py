"""
tests/test_07_coupled.py — Coupled BioHeatSolver + ArrheniusDamage tests.

Runs both state equations for 180s at P_max and verifies:
  - damage grows in tumor
  - temperatures stay physical
  - ablation progress is monotone

Run standalone:
    python tests/test_07_coupled.py
Run with pytest:
    pytest tests/test_07_coupled.py -v
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
from physics.mesh import build_region_masks
from physics.sar_model import compute_sar_field
from physics.bioheat import BioHeatSolver
from physics.arrhenius import ArrheniusDamage
from visualization.field_plots import plot_temperature_field
from visualization.damage_plots import plot_damage_field


def test_coupled_bioheat_arrhenius(ndim=2):
    """
    Runs both state equations coupled for 180s at P_max.
    Checks damage grows, temperatures remain physical, and
    ablation progress makes sense.

    Visual: T and Ω fields at t=60s and t=180s, plus time-history traces.
    """
    cfg     = make_cfg(ndim, small=True)
    sar     = compute_sar_field(cfg=cfg)
    bioheat = BioHeatSolver(cfg=cfg, sar_field=sar)
    damage  = ArrheniusDamage(cfg=cfg)
    tumor_mask, healthy_mask, _ = build_region_masks(cfg)
    tm = tumor_mask.ravel()
    hm = healthy_mask.ravel()

    n_steps = 180
    dt    = cfg.solver.dt
    P_max = cfg.control.P_max
    t_vec = np.arange(n_steps + 1) * dt

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
    assert Om_hist[-1][tm].max() > 0.1, \
        f"Tumor damage should exceed 0.1 after {n_steps}s"
    assert T_hist[-1].max() < 200.0, "Temperature > 200°C is unphysical"
    assert T_hist[-1].min() > 30.0,  "Temperature < 30°C is unphysical"

    center = cfg.domain.N // 2
    for k in range(n_steps):
        assert Om_hist[k+1][center] >= Om_hist[k][center] - 1e-12, \
            f"Damage decreased at step {k}"

    print(f"  ndim={ndim}")
    print(f"  T max tumor  at {n_steps}s: {T_hist[-1][tm].max():.1f} °C")
    print(f"  T max healthy at {n_steps}s: {T_hist[-1][hm].max():.1f} °C")
    print(f"  Ω max tumor  at {n_steps}s: {Om_hist[-1][tm].max():.4f}")
    print(f"  Fraction ablated: {frac_ablated[-1]:.1%}")

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
    ax_ab.fill_between(t_vec, np.array(frac_ablated) * 100,
                       alpha=0.2, color='darkred')
    ax_ab.axhline(95, color='green', linestyle='--', linewidth=1.5,
                  label='95% clinical target')
    ax_ab.set_xlabel('Time [s]'); ax_ab.set_ylabel('Tumor ablated [%]')
    ax_ab.set_title('Ablation progress', fontsize=11)
    ax_ab.set_ylim(-2, 103)
    ax_ab.legend(fontsize=9); ax_ab.grid(True, alpha=0.3)

    fig.suptitle(f'Test 07: Coupled BioHeat + Arrhenius  P=P_max  (ndim={ndim})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'07_coupled_{ndim}d.png', cfg)


# ── pytest entry points ───────────────────────────────────────────────────────

def test_coupled_bioheat_arrhenius_2d():
    test_coupled_bioheat_arrhenius(ndim=2)

def test_coupled_bioheat_arrhenius_3d():
    test_coupled_bioheat_arrhenius(ndim=3)


if __name__ == '__main__':
    from tests._helpers import run_suite
    run_suite([test_coupled_bioheat_arrhenius])

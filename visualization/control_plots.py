"""
visualization/control_plots.py — Control trajectory, cost history, and costate plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_region_masks


def plot_power_history(u_history: np.ndarray,
                       t_vec: np.ndarray,
                       cfg: SimConfig = default_cfg,
                       save_path: str = None) -> plt.Figure:
    """P(t) control profile and power statistics."""
    t_ctrl = t_vec[:len(u_history)]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    # Power trajectory
    axes[0].step(t_ctrl, u_history, where='post', color='steelblue', linewidth=2)
    axes[0].axhline(cfg.control.P_max, color='red', linestyle='--',
                    linewidth=1, alpha=0.6, label=f'P_max = {cfg.control.P_max} W')
    axes[0].set_ylabel('Power P(t) [W]', fontsize=11)
    axes[0].set_ylim(-2, cfg.control.P_max * 1.15)
    axes[0].legend(fontsize=9)
    axes[0].set_title('Applied ablation power', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Cumulative energy
    dt = cfg.solver.dt
    energy_cumul = np.cumsum(u_history) * dt / 1000.0   # kJ
    axes[1].fill_between(t_ctrl, energy_cumul, alpha=0.35, color='steelblue')
    axes[1].plot(t_ctrl, energy_cumul, color='steelblue', linewidth=1.5)
    axes[1].set_ylabel('Cumulative energy [kJ]', fontsize=11)
    axes[1].set_xlabel('Time [s]', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=cfg.viz.dpi, bbox_inches='tight')
    return fig


def plot_temperature_histories(T_history: np.ndarray,
                                t_vec: np.ndarray,
                                cfg: SimConfig = default_cfg,
                                save_path: str = None) -> plt.Figure:
    """
    Time histories of key temperature metrics:
      - Max temperature in tumor
      - Max temperature in healthy tissue
      - Mean temperature in tumor
    """
    tumor_2d, healthy_2d, _ = build_region_masks(cfg)
    tm = tumor_2d.ravel()
    hm = healthy_2d.ravel()

    T_max_tumor   = np.array([T_history[k][tm].max()  for k in range(len(t_vec))])
    T_mean_tumor  = np.array([T_history[k][tm].mean() for k in range(len(t_vec))])
    T_max_healthy = np.array([T_history[k][hm].max()  for k in range(len(t_vec))])

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(t_vec, T_max_tumor,   label='max T (tumor)',   color='red',    linewidth=2)
    ax.plot(t_vec, T_mean_tumor,  label='mean T (tumor)',  color='orange', linewidth=1.5, linestyle='--')
    ax.plot(t_vec, T_max_healthy, label='max T (healthy)', color='steelblue', linewidth=2)
    ax.axhline(cfg.control.T_safe, color='blue', linestyle=':', linewidth=1.5,
               label=f'T_safe = {cfg.control.T_safe}°C')
    ax.axhline(60.0, color='darkred', linestyle=':', linewidth=1.0,
               label='60°C ablation threshold', alpha=0.6)

    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Temperature [°C]', fontsize=11)
    ax.set_title('Temperature time histories', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=cfg.viz.dpi, bbox_inches='tight')
    return fig


def plot_ablation_progress(Omega_history: np.ndarray,
                            t_vec: np.ndarray,
                            cfg: SimConfig = default_cfg,
                            save_path: str = None) -> plt.Figure:
    """Fraction of tumor ablated over time (Ω_d ≥ 1 criterion)."""
    tumor_2d, _, _ = build_region_masks(cfg)
    tm = tumor_2d.ravel()
    threshold = cfg.arrhenius.damage_threshold

    frac = np.array([(Omega_history[k][tm] >= threshold).mean()
                     for k in range(len(t_vec))])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_vec, frac * 100, color='darkred', linewidth=2)
    ax.axhline(95, color='green', linestyle='--', linewidth=1.5,
               label='95% clinical target')
    ax.axhline(100, color='black', linestyle=':', linewidth=1.0, alpha=0.5)
    ax.fill_between(t_vec, frac * 100, alpha=0.2, color='darkred')

    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Tumor ablated [%]', fontsize=11)
    ax.set_ylim(-2, 103)
    ax.set_title('Ablation completeness over time', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=cfg.viz.dpi, bbox_inches='tight')
    return fig


def plot_cost_breakdown(cost_dict: dict,
                        cfg: SimConfig = default_cfg,
                        save_path: str = None) -> plt.Figure:
    """Bar chart of cost functional components."""
    keys   = ['J_energy', 'J_healthy', 'J_time_rate', 'J_ablation', 'J_tf']
    labels = ['Energy\n(α1‖u‖²)', 'Healthy T\n(α2 penalty)',
              'Time rate\n(α3)', 'Ablation\nincomplete', 'Final time\n(γ2·tf)']
    values = [cost_dict.get(k, 0.0) for k in keys]
    colors = ['steelblue', 'orange', 'gray', 'red', 'purple']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.bar_label(bars, fmt='%.3e', fontsize=8, padding=3)
    ax.set_ylabel('Cost contribution', fontsize=11)
    ax.set_title(f"Cost breakdown   J_total = {cost_dict.get('J_total', 0):.4e}",
                 fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=cfg.viz.dpi, bbox_inches='tight')
    return fig

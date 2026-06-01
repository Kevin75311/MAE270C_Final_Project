"""
control/constraints.py — Path and terminal constraint evaluation.

Constraints in the OCP:

  Path (must hold ∀t ∈ [0, t_f]):
    g1(x, u) : T(r, t) ≤ T_safe      ∀ r ∈ Ω_H
    g2(u)    : 0 ≤ P(t) ≤ P_max
    g3(u)    : |dP/dt| ≤ Ṗ_max

  Terminal (must hold at t = t_f):
    h1(x_f)  : Ω_d(r, t_f) ≥ 1       ∀ r ∈ Ω_T   (tumor fully ablated)
    h2(t_f)  : t_f ∈ (0, T_max]

Returns signed constraint violations:  value > 0 means VIOLATED.
"""

import numpy as np
from config import SimConfig, cfg as default_cfg
from physics.mesh import build_region_masks


class ConstraintChecker:
    """Evaluates all OCP constraints and reports violations."""

    def __init__(self, cfg: SimConfig = default_cfg):
        self.cfg = cfg
        tumor_2d, healthy_2d, _ = build_region_masks(cfg)
        self.tumor_mask   = tumor_2d.ravel()
        self.healthy_mask = healthy_2d.ravel()

    # ── Path constraints ──────────────────────────────────────────────────────

    def temperature_safety(self, T_flat: np.ndarray) -> dict:
        """
        g1:  T(r, t) ≤ T_safe   ∀ r ∈ Ω_H

        Returns max violation [°C] and fraction of violated voxels.
        Violation > 0  →  constraint is active / violated.
        """
        T_healthy  = T_flat[self.healthy_mask]
        violation  = T_healthy - self.cfg.control.T_safe  # > 0 if violated
        max_viol   = float(violation.max()) if len(violation) > 0 else 0.0
        frac_viol  = float((violation > 0).mean())
        return {
            'max_violation_degC': max_viol,
            'fraction_violated':  frac_viol,
            'satisfied':          max_viol <= 0.0,
        }

    def control_bounds(self, P: float, P_prev: float = None,
                       dt: float = None) -> dict:
        """
        g2:  0 ≤ P(t) ≤ P_max
        g3:  |dP/dt| ≤ Ṗ_max   (only checked if P_prev is provided)
        """
        P_min   = self.cfg.control.P_min
        P_max   = self.cfg.control.P_max
        P_dot_max = self.cfg.control.P_dot_max

        box_ok  = (P_min <= P <= P_max)
        rate_ok = True

        if P_prev is not None and dt is not None:
            dP_dt  = abs(P - P_prev) / dt
            rate_ok = dP_dt <= P_dot_max

        return {
            'box_satisfied':  box_ok,
            'rate_satisfied': rate_ok,
            'satisfied':      box_ok and rate_ok,
        }

    def check_path(self, T_flat: np.ndarray, P: float,
                   P_prev: float = None, dt: float = None) -> dict:
        """Check all path constraints at a single timestep."""
        temp   = self.temperature_safety(T_flat)
        ctrl   = self.control_bounds(P, P_prev, dt)
        return {
            'temperature': temp,
            'control':     ctrl,
            'all_satisfied': temp['satisfied'] and ctrl['satisfied'],
        }

    # ── Terminal constraints ──────────────────────────────────────────────────

    def ablation_complete(self, Omega_flat: np.ndarray) -> dict:
        """
        h1:  Ω_d(r, t_f) ≥ 1   ∀ r ∈ Ω_T

        Returns fraction ablated and whether full ablation is achieved.
        """
        threshold     = self.cfg.arrhenius.damage_threshold
        Omega_tumor   = Omega_flat[self.tumor_mask]
        frac_ablated  = float((Omega_tumor >= threshold).mean())
        mean_omega    = float(Omega_tumor.mean())
        min_omega     = float(Omega_tumor.min())
        return {
            'fraction_ablated':   frac_ablated,
            'mean_omega_tumor':   mean_omega,
            'min_omega_tumor':    min_omega,
            'satisfied':          frac_ablated >= 1.0,   # ALL voxels ablated
            'satisfied_95pct':    frac_ablated >= 0.95,  # clinical 95% criterion
        }

    def time_constraint(self, t_f: float) -> dict:
        """h2:  0 < t_f ≤ T_max."""
        T_max = self.cfg.control.T_max_treatment
        ok    = 0 < t_f <= T_max
        return {'t_f': t_f, 'T_max': T_max, 'satisfied': ok}

    def check_terminal(self, Omega_flat: np.ndarray, t_f: float) -> dict:
        """Check all terminal constraints."""
        ablation = self.ablation_complete(Omega_flat)
        time_c   = self.time_constraint(t_f)
        return {
            'ablation': ablation,
            'time':     time_c,
            'all_satisfied': ablation['satisfied_95pct'] and time_c['satisfied'],
        }

    # ── Full trajectory summary ───────────────────────────────────────────────

    def check_trajectory(self, T_history: np.ndarray,
                         Omega_history: np.ndarray,
                         u_history: np.ndarray,
                         t_f: float) -> dict:
        """
        Evaluate constraints over the full recorded trajectory.

        Returns a summary dict with worst-case path violations and
        terminal constraint status.
        """
        n_steps = len(u_history)
        dt      = self.cfg.solver.dt

        max_temp_viol = -np.inf
        any_ctrl_viol = False

        for k in range(n_steps):
            temp_c = self.temperature_safety(T_history[k])
            ctrl_c = self.control_bounds(
                u_history[k],
                u_history[k-1] if k > 0 else None,
                dt
            )
            max_temp_viol = max(max_temp_viol, temp_c['max_violation_degC'])
            if not ctrl_c['satisfied']:
                any_ctrl_viol = True

        terminal = self.check_terminal(Omega_history[-1], t_f)

        return {
            'path': {
                'max_temperature_violation_degC': max_temp_viol,
                'any_control_violation': any_ctrl_viol,
                'satisfied': (max_temp_viol <= 0.0) and (not any_ctrl_viol),
            },
            'terminal': terminal,
            'overall_satisfied': (
                (max_temp_viol <= 0.0)
                and (not any_ctrl_viol)
                and terminal['all_satisfied']
            ),
        }

    def print_summary(self, result: dict):
        """Pretty-print a constraint check result dict."""
        print("\n── Constraint Summary ─────────────────────────────────────")
        path = result['path']
        term = result['terminal']
        print(f"  Path — temperature:  max violation = "
              f"{path['max_temperature_violation_degC']:.2f} °C  "
              f"({'OK' if not path['any_control_violation'] else 'VIOLATED'})")
        print(f"  Path — control:      "
              f"{'OK' if not path['any_control_violation'] else 'VIOLATED'}")
        ablation = term['ablation']
        print(f"  Terminal — ablation: {ablation['fraction_ablated']:.1%} ablated  "
              f"(min Ω = {ablation['min_omega_tumor']:.3f})  "
              f"{'OK' if ablation['satisfied_95pct'] else 'INCOMPLETE'}")
        print(f"  Terminal — time:     t_f = {term['time']['t_f']:.1f} s  "
              f"({term['time']['t_f']/60:.1f} min)  "
              f"{'OK' if term['time']['satisfied'] else 'EXCEEDED'}")
        print(f"  OVERALL: {'✓ FEASIBLE' if result['overall_satisfied'] else '✗ INFEASIBLE'}")
        print("────────────────────────────────────────────────────────────\n")

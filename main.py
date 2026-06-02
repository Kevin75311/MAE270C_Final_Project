"""
main.py — MRI Thermal Ablation Optimal Control Simulation.

Orchestration:
  1. Load and display configuration
  2. Validate mesh and CFL stability
  3. Select solver mode and run
  4. Check constraints
  5. Visualize and save results

Arguments (all freely combinable)
----------------------------------
  --mode    Solver / optimizer
              openloop  fixed bang-bang power schedule, no optimization (default)
              indirect  PMP gradient projection (forward-adjoint sweeps)
              direct    single-shooting NLP via scipy SLSQP
              mpc       receding-horizon Model Predictive Control

  --ndim    Spatial dimensionality
              2         2D x-y slice, isotropic Gaussian SAR (default)
              3         3D volume, anisotropic / line SAR along probe z-axis

  --bc      Boundary condition on ∂Ω
              dirichlet fixed wall temperature T_wall = 37 °C (default)
              neumann   zero heat flux — insulated boundary
              robin     convective cooling  h_c, T_inf from config.py

  --probe   SAR / antenna model
              point     isotropic Gaussian blob (2D) or anisotropic Gaussian (3D) (default)
              line      cylindrical heating zone of length L_active with Gaussian end-cap rolloff
              dipole    sin²(θ) toroidal near-field pattern (slot antenna approximation)

  --animate generate a GIF animation of the ablation (2D only, slow)
  --no-save skip saving the trajectory npz file to results/

Example combinations
--------------------
  python main.py                                        # openloop, 2D, dirichlet, point
  python main.py --mode indirect --probe line           # PMP + realistic needle, 2D
  python main.py --mode mpc --ndim 3 --probe line       # MPC, 3D, line source
  python main.py --mode direct --bc robin --probe dipole
  python main.py --mode openloop --ndim 3 --bc neumann --probe line --animate
"""

import argparse
import os
import numpy as np

from config import cfg

from physics.mesh import build_mesh, build_region_masks
from physics.sar_model import compute_sar_field

from control.solver import (
    solve_openloop,
    solve_indirect,
    solve_direct,
    solve_mpc,
)
from control.constraints import ConstraintChecker

from visualization.field_plots import plot_temperature_field, plot_slice_comparison
from visualization.damage_plots import plot_damage_field, plot_necrosis_boundary
from visualization.control_plots import (plot_power_history, plot_temperature_histories,
                                         plot_ablation_progress, plot_cost_breakdown)
from visualization.animation import animate_ablation

from utils.io_utils import save_trajectory
from utils.validators import check_stability_cfl, check_temperature_physical


def main(mode: str = 'openloop', bc: str = None, ndim: int = None,
         probe: str = None, save: bool = True, animate: bool = False):
    # ─────────────────────────────────────────────────────────────────────────
    # 1. Configuration summary
    # ─────────────────────────────────────────────────────────────────────────
    if ndim is not None:
        cfg.domain.ndim = ndim
    if bc is not None:
        cfg.boundary.bc_type = bc
    if probe is not None:
        cfg.sar.probe_model = probe
    cfg.summary()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Mesh and stability validation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Mesh & stability check ──────────────────────────────────────")
    X, Y, Z, xv, yv, zv = build_mesh(cfg)
    tumor_mask, healthy_mask, margin_mask = build_region_masks(cfg)
    print(f"  Voxels: {cfg.domain.Nx} × {cfg.domain.Ny} = {cfg.domain.N}")
    print(f"  Tumor: {tumor_mask.sum()} voxels  |  "
          f"Healthy: {healthy_mask.ravel().sum()} voxels")
    stable, dt_cfl = check_stability_cfl(cfg)

    sar = compute_sar_field(cfg=cfg)
    print(f"  SAR peak:  {sar.max():.2e} W/kg  @ probe position")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Solve
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n── Solver mode: {mode.upper()} ──────────────────────────────────────")
    os.makedirs(cfg.viz.output_dir, exist_ok=True)

    if mode == 'openloop':
        # Mode 0: fixed bang-bang schedule, no optimization
        result = solve_openloop(cfg, verbose=True)

    elif mode == 'indirect':
        # Mode 1: PMP gradient projection (forward-adjoint sweeps)
        result = solve_indirect(cfg, verbose=True)

    elif mode == 'direct':
        # Mode 2: direct single-shooting NLP (scipy SLSQP)
        result = solve_direct(cfg, verbose=True)

    elif mode == 'mpc':
        # Mode 3: receding-horizon MPC with optional MRI noise
        def noisy_mri(T_true, t):
            noise_std = 0.5   # MRI thermometry noise ~ 0.5°C
            return T_true + np.random.normal(0, noise_std, T_true.shape)

        result = solve_mpc(cfg, mri_feedback_fn=noisy_mri, verbose=True)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose: openloop, indirect, direct, mpc")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Constraint verification
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Constraint verification ──────────────────────────────────────")
    checker = ConstraintChecker(cfg)
    checker.print_summary(result['constraints'])

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Visualize
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Generating figures ───────────────────────────────────────────")
    T_hist  = result['T_history']
    Om_hist = result['Omega_history']
    u_hist  = result['u_history']
    t_vec   = result['t_vec']
    t_f     = t_vec[-1]
    out     = cfg.viz.output_dir

    plot_temperature_field(
        T_hist[-1], t=t_f, cfg=cfg,
        save_path=os.path.join(out, 'T_final.png'))

    plot_damage_field(
        Om_hist[-1], t=t_f, cfg=cfg,
        save_path=os.path.join(out, 'damage_final.png'))

    plot_necrosis_boundary(
        Om_hist[-1], T_hist[-1], t=t_f, cfg=cfg,
        save_path=os.path.join(out, 'necrosis_boundary.png'))

    n = len(t_vec)
    snaps = [n // 4, n // 2, 3 * n // 4, n - 1]
    plot_slice_comparison(T_hist, t_vec, snaps, cfg,
                          save_path=os.path.join(out, 'T_snapshots.png'))

    plot_power_history(u_hist, t_vec, cfg,
                       save_path=os.path.join(out, 'power_history.png'))

    plot_temperature_histories(T_hist, t_vec, cfg,
                               save_path=os.path.join(out, 'temp_histories.png'))

    plot_ablation_progress(Om_hist, t_vec, cfg,
                           save_path=os.path.join(out, 'ablation_progress.png'))

    plot_cost_breakdown(result['cost'], cfg,
                        save_path=os.path.join(out, 'cost_breakdown.png'))

    if animate:
        animate_ablation(T_hist, Om_hist, t_vec, cfg,
                         save_path=os.path.join(out, 'ablation.gif'),
                         stride=max(1, len(t_vec) // 60))

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Save trajectory
    # ─────────────────────────────────────────────────────────────────────────
    if save:
        save_trajectory(result, run_name=f"run_{mode}", cfg=cfg)

    print(f"\n── Done.  Results in: {out}/ ────────────────────────────────────")
    print(f"  J_total = {result['cost']['J_total']:.4e}")
    frac = result['constraints']['terminal']['ablation']['fraction_ablated']
    print(f"  Ablation completeness: {frac:.1%}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MRI Thermal Ablation Optimal Control Simulation")
    parser.add_argument(
        '--mode', type=str, default='openloop',
        choices=['openloop', 'indirect', 'direct', 'mpc'],
        help=(
            "Solver mode: "
            "openloop=fixed schedule (no opt), "
            "indirect=PMP gradient projection, "
            "direct=single-shooting NLP, "
            "mpc=receding-horizon MPC"
        ))
    parser.add_argument(
        '--ndim', type=int, default=None,
        choices=[2, 3],
        help="Simulation dimensionality: 2=2D (default), 3=3D with probe z-axis geometry")
    parser.add_argument(
        '--bc', type=str, default=None,
        choices=['dirichlet', 'neumann', 'robin'],
        help=(
            "Boundary condition type on ∂Ω: "
            "dirichlet=fixed T_wall (default 37°C), "
            "neumann=zero-flux (insulated), "
            "robin=convective cooling (h_c, T_inf from config.py)"
        ))
    parser.add_argument(
        '--probe', type=str, default=None,
        choices=['point', 'line', 'dipole'],
        help=(
            "Probe SAR model: "
            "point=isotropic Gaussian (default), "
            "line=cylindrical active zone with end-cap falloff, "
            "dipole=sin²(θ) toroidal near-field pattern"
        ))
    parser.add_argument('--animate', action='store_true',
                        help="Generate GIF animation (slow)")
    parser.add_argument('--no-save', action='store_true',
                        help="Do not save trajectory to disk")
    args = parser.parse_args()

    main(mode=args.mode, bc=args.bc, ndim=args.ndim, probe=args.probe,
         save=not args.no_save, animate=args.animate)

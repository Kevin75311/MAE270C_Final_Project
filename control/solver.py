"""
control/solver.py — Four OCP solving modes.

  0. solve_openloop()  — forward simulation with a fixed schedule, no optimization
  1. solve_indirect()  — PMP / gradient projection (forward-adjoint sweep iteration)
  2. solve_direct()    — direct single-shooting NLP via scipy.optimize
  3. solve_mpc()       — receding-horizon Model Predictive Control

  Internal utility:
  forward_simulate()   — shared forward integration loop used by all modes
"""

import numpy as np
from scipy.optimize import minimize
from config import SimConfig, cfg as default_cfg
from physics.bioheat import BioHeatSolver
from physics.arrhenius import ArrheniusDamage
from physics.sar_model import compute_sar_field, get_control_input_vector
from physics.mesh import build_region_masks
from control.ocp import CostFunctional
from control.adjoint import AdjointSolver
from control.constraints import ConstraintChecker


def _result_dict(T_history, Omega_history, u_history,
                 t_vec, cost_info, constraint_info, cfg, t_f=None):
    return {
        'T_history':     T_history,      # (n+1, N)
        'Omega_history': Omega_history,  # (n+1, N)
        'u_history':     u_history,      # (n,)
        't_vec':         t_vec,          # (n+1,)
        't_f':           t_f if t_f is not None else float(t_vec[-1]),  # treatment-end time
        'cost':          cost_info,
        'constraints':   constraint_info,
        'cfg':           cfg,
    }


# ── Free final time: optimal stopping ─────────────────────────────────────────

def optimal_stop_index(Om_history: np.ndarray,
                       cfg: SimConfig = default_cfg) -> int:
    """
    Free-final-time optimal stopping index.

    This is a free-t_f Bolza problem with a *positive* running cost
    (α₁P² + α₂·overshoot² + α₃ ≥ α₃ > 0) and a soft min-time term γ₂·t_f.
    The transversality condition ℋ(t_f) + γ₂ = 0 would require
    ℋ(t_f) = −γ₂ < 0, but once the tumor is fully ablated λ_T(t_f)=λ_Ω(t_f)=0
    so ℋ(t_f) = L(t_f) ≥ α₃ > 0 — the interior condition is infeasible.
    Hence the optimum lies on the boundary where the terminal ablation
    constraint Ω_d ≥ 1 first becomes active: continuing past that instant only
    adds positive running cost and γ₂·dt with zero terminal benefit.

    Returns the first step index k at which every tumor voxel satisfies
    Ω_d ≥ threshold.  If full ablation is never reached, returns the last
    index (n) so the trajectory is not truncated.
    """
    tumor_mask = build_region_masks(cfg)[0].ravel()
    threshold  = cfg.arrhenius.damage_threshold
    fully = (Om_history[:, tumor_mask] >= threshold).all(axis=1)  # (n+1,)
    hits  = np.flatnonzero(fully)
    return int(hits[0]) if hits.size > 0 else Om_history.shape[0] - 1


# ── Shared forward integration ────────────────────────────────────────────────

def forward_simulate(u_schedule,
                     cfg: SimConfig = default_cfg,
                     verbose: bool = False,
                     free_final_time: bool = True) -> dict:
    """
    Integrate the state equations forward under a prescribed control schedule.

    Parameters
    ----------
    u_schedule : callable  P = u_schedule(t)  or array of shape (n_steps,)
    free_final_time : if True (default), truncate the trajectory at the
                      optimal stopping time t_f* = first instant of full tumor
                      ablation (see optimal_stop_index).  The reported cost and
                      constraints — including γ₂·t_f — then use t_f*, not the
                      fixed horizon t_final.  Set False to report the full
                      fixed-horizon trajectory.

    Returns
    -------
    Standard result dict.
    """
    dt  = cfg.solver.dt
    n   = cfg.solver.n_steps
    N   = cfg.domain.N

    sar     = compute_sar_field(cfg=cfg)
    bioheat = BioHeatSolver(cfg=cfg, sar_field=sar)
    damage  = ArrheniusDamage(cfg=cfg)
    cost_fn = CostFunctional(cfg=cfg)

    T_hist  = np.zeros((n + 1, N))
    Om_hist = np.zeros((n + 1, N))
    u_hist  = np.zeros(n)
    t_vec   = np.linspace(0, cfg.solver.t_final, n + 1)

    T_hist[0]  = bioheat.initialize()
    Om_hist[0] = damage.initialize()

    tumor_mask, healthy_mask, _ = build_region_masks(cfg)

    for k in range(n):
        t_k = t_vec[k]
        P_k = u_schedule(t_k) if callable(u_schedule) else float(u_schedule[k])
        P_k = float(np.clip(P_k, cfg.control.P_min, cfg.control.P_max))

        T_hist[k + 1]  = bioheat.step(T_hist[k],  P_k, dt)
        Om_hist[k + 1] = damage.step(Om_hist[k], T_hist[k], dt)
        u_hist[k]      = P_k

        if verbose and k % 50 == 0:
            ablated = (Om_hist[k + 1] >= 1.0)[tumor_mask.ravel()].mean()
            T_max_h = T_hist[k + 1][healthy_mask.ravel()].max()
            print(f"  t={t_vec[k+1]:6.1f}s  P={P_k:5.1f}W  "
                  f"T_max_healthy={T_max_h:.1f}°C  tumor_ablated={ablated:.1%}")

    # ── Free final time: stop at the optimal instant; cost uses t_f* ──────────
    if free_final_time:
        k_stop = optimal_stop_index(Om_hist, cfg)
    else:
        k_stop = n

    T_cost  = T_hist[:k_stop + 1]
    Om_cost = Om_hist[:k_stop + 1]
    u_cost  = u_hist[:k_stop]
    t_f     = float(t_vec[k_stop])

    costs = cost_fn.total_cost(T_cost, Om_cost, u_cost, t_f, dt)
    chk   = ConstraintChecker(cfg)
    cons  = chk.check_trajectory(T_cost, Om_cost, u_cost, t_f)

    # ── Post-treatment observation window (applicator OFF, P=0) ───────────────
    # Continue the cooldown past t_f* for visualization only — the cost above is
    # unchanged.  Skipped when the tumor never fully ablated (k_stop == n).
    obs_steps = 0
    if free_final_time and k_stop < n:
        obs_steps = int(round(cfg.solver.post_observation_time / dt))

    if obs_steps > 0:
        T_obs  = np.zeros((obs_steps, N))
        Om_obs = np.zeros((obs_steps, N))
        T_c, Om_c = T_hist[k_stop].copy(), Om_hist[k_stop].copy()
        for j in range(obs_steps):
            T_n  = bioheat.step(T_c, 0.0, dt)   # applicator off
            Om_n = damage.step(Om_c, T_c, dt)   # damage keeps accumulating if still hot
            T_obs[j], Om_obs[j] = T_n, Om_n
            T_c, Om_c = T_n, Om_n
        T_ret  = np.vstack([T_cost, T_obs])
        Om_ret = np.vstack([Om_cost, Om_obs])
        u_ret  = np.concatenate([u_cost, np.zeros(obs_steps)])
        t_ret  = np.concatenate([t_vec[:k_stop + 1],
                                 t_f + dt * np.arange(1, obs_steps + 1)])
    else:
        T_ret, Om_ret, u_ret, t_ret = T_cost, Om_cost, u_cost, t_vec[:k_stop + 1]

    return _result_dict(T_ret, Om_ret, u_ret, t_ret, costs, cons, cfg, t_f=t_f)


# ── Mode 0: Open-loop (no optimization) ──────────────────────────────────────

def solve_openloop(cfg: SimConfig = default_cfg,
                   schedule=None,
                   verbose: bool = True) -> dict:
    """
    Mode 0 — forward simulation with a fixed, non-optimized control schedule.

    No optimization is performed.  Serves as a baseline for comparing the
    optimized modes.

    Parameters
    ----------
    schedule : callable  P = schedule(t) [W], optional.
               Defaults to a bang-bang heuristic:
                 full power for the first 60% of t_final,
                 half power for the next 20%, then off.
    """
    if schedule is None:
        t_f   = cfg.solver.t_final
        P_max = cfg.control.P_max

        def schedule(t: float) -> float:
            if t < t_f * 0.6:
                return P_max
            elif t < t_f * 0.8:
                return P_max * 0.5
            return 0.0

    return forward_simulate(schedule, cfg, verbose=verbose)


# ── Mode 1: Indirect method (PMP / gradient projection) ──────────────────────

def solve_indirect(cfg: SimConfig = default_cfg,
                   u_init: np.ndarray = None,
                   max_iter: int = 300,
                   tol: float = 1e-2,
                   relaxation: float = 0.15,
                   verbose: bool = True) -> dict:
    """
    Mode 1 — indirect method: PMP gradient projection via forward-adjoint sweeps.

    Algorithm
    ---------
    Each iteration k performs one forward-adjoint sweep:

      1. Forward pass  — integrate state x(t) under u^k(t)
      2. Terminal costate  — set λ(t_f) from transversality:
             λ_T(t_f) = 0
             λ_Ω(t_f) = −2γ₁ max(0, 1 − Ω_d(t_f)) · 1_{Ω_T}
      3. Backward pass  — integrate adjoint λ_T(t) backward in time;
             λ_Ω is constant (its adjoint ODE has zero rhs)
      4. Pontryagin projection  — compute candidate control:
             ũ(t) = clip( −λ_T(t)ᵀ b_P / (2α₁),  0,  P_max )
      5. Relaxed update  — u^{k+1} = (1 − β) u^k + β ũ
      6. Convergence check  — stop if  ‖u^{k+1} − u^k‖_∞ < tol

    Parameters
    ----------
    u_init    : initial control guess, shape (n_steps,).  Defaults to 30% P_max.
    max_iter  : maximum number of forward-adjoint sweeps.
    tol       : convergence tolerance on max control change [W].
    relaxation: β ∈ (0, 1] — mixing weight for the Pontryagin update.
                Smaller values stabilize convergence at the cost of more sweeps.
                The Pontryagin control here is bang-bang (saturates at 0 or
                P_max), so a large β makes the fixed-point iteration limit-cycle
                between "under-ablated → full power" and "ablated → zero power".
                β≈0.15 damps this and converges to full tumor ablation; β=0.5
                oscillates and stalls at ~42%.
    """
    dt = cfg.solver.dt
    n  = cfg.solver.n_steps
    N  = cfg.domain.N

    sar     = compute_sar_field(cfg=cfg)
    bioheat = BioHeatSolver(cfg=cfg, sar_field=sar)
    damage  = ArrheniusDamage(cfg=cfg)
    cost_fn = CostFunctional(cfg=cfg)
    adj     = AdjointSolver(cfg=cfg)

    # b_P: effective control gain vector  (M⁻¹ · ρ·SAR_unit),  shape (N,)
    # This is dẋ_T/dP at each voxel — how a 1-W increase changes temperature rate.
    b_P = adj.M_inv_diag * get_control_input_vector(sar, 1.0, cfg)

    if u_init is None:
        u_k = np.full(n, cfg.control.P_max * 0.3)
    else:
        u_k = np.clip(u_init.copy(), cfg.control.P_min, cfg.control.P_max)

    if verbose:
        print(f"Starting indirect (PMP) solver  "
              f"(n={n} steps,  max_iter={max_iter},  "
              f"tol={tol:.0e},  β={relaxation}) ...")

    delta = np.inf
    for it in range(max_iter):

        # ── 1. Forward pass ───────────────────────────────────────────────
        T_hist  = np.zeros((n + 1, N))
        Om_hist = np.zeros((n + 1, N))
        T_hist[0]  = bioheat.initialize()
        Om_hist[0] = damage.initialize()

        for k in range(n):
            T_hist[k + 1]  = bioheat.step(T_hist[k],  u_k[k], dt)
            Om_hist[k + 1] = damage.step(Om_hist[k], T_hist[k], dt)

        # ── 2. Terminal costate from transversality ───────────────────────
        lam_T, lam_O = adj.terminal_costate(Om_hist[-1])

        # ── 3. Backward pass ──────────────────────────────────────────────
        # lam_T_hist[k] = λ_T at time t_k (needed to form u*(t_k))
        lam_T_hist    = np.zeros((n + 1, N))
        lam_T_hist[n] = lam_T

        for k in range(n - 1, -1, -1):
            # Step t_{k+1} → t_k; adjoint RHS evaluated at T(t_{k+1})
            lam_T, lam_O = adj.step_backward(lam_T, lam_O, T_hist[k + 1], dt)
            lam_T_hist[k] = lam_T

        # ── 4. Pontryagin projection ──────────────────────────────────────
        # PMP minimizes ℋ = L + λᵀf  ⟹  ∂ℋ/∂P = 2α₁P + λ_Tᵀb_P = 0
        #   ⟹  P* = −λ_Tᵀb_P / (2α₁).  The leading minus sign is essential:
        # λ_T is negative in the tumor during heating, so −λ_Tᵀb_P > 0 turns
        # the applicator ON.  (A previous version omitted it and produced P*≡0.)
        alpha1 = cfg.cost.alpha1
        u_pmp = np.array([
            float(np.clip(-np.dot(lam_T_hist[k], b_P) / (2.0 * alpha1),
                          cfg.control.P_min, cfg.control.P_max))
            for k in range(n)
        ])

        # ── 5. Relaxed update ─────────────────────────────────────────────
        u_new = (1.0 - relaxation) * u_k + relaxation * u_pmp
        delta = float(np.max(np.abs(u_new - u_k)))

        # ── 6. Logging ────────────────────────────────────────────────────
        if verbose and (it % 10 == 0 or delta < tol):
            costs   = cost_fn.total_cost(T_hist, Om_hist, u_k,
                                          cfg.solver.t_final, dt)
            ablated = float(np.mean(Om_hist[-1][cost_fn.tumor_mask] >= 1.0))
            print(f"  iter {it:4d}:  Δu = {delta:.3e}  "
                  f"J = {costs['J_total']:.4e}  ablated = {ablated:.1%}")

        u_k = u_new

        if delta < tol:
            if verbose:
                print(f"  Converged at iteration {it}  (Δu = {delta:.2e})")
            break
    else:
        if verbose:
            print(f"  Warning: did not converge after {max_iter} iterations  "
                  f"(Δu = {delta:.2e} > tol={tol:.0e})")

    return forward_simulate(u_k, cfg, verbose=False)


# ── Mode 2: Direct method (single-shooting NLP) ───────────────────────────────

def solve_direct(cfg: SimConfig = default_cfg,
                 u_init: np.ndarray = None,
                 verbose: bool = True) -> dict:
    """
    Mode 2 — direct single-shooting NLP via scipy.optimize.minimize.

    Decision variable: u ∈ ℝ^n_steps  (piecewise-constant power schedule).

    At every optimizer call the full PDE is re-integrated from x₀, so runtime
    is O(n_eval · n · N).  Intended as a reference solver; for large grids
    prefer solve_indirect() (adjoint gradients) or direct collocation (CasADi).

    Gradients are approximated by SLSQP via finite differences.  Box
    constraints on u are enforced through scipy bounds.

    Energy mode (cfg.cost.mode == 'energy', default):
      Path constraints (T ≤ T_safe) are soft-penalized via α₂ in the cost.
      No SLSQP inequality constraints are passed.

    Time-optimal mode (cfg.cost.mode == 'time', cfg.solver.enforce_safety_hard == True):
      α₁ is reduced to a tiny regularization (alpha1_time) and α₂ is zeroed,
      so the cost is almost purely γ₂·t_f + γ₁·ablation + α₃·t.  T ≤ T_safe in
      healthy tissue is promoted to a hard SLSQP inequality constraint so that
      nothing else limits power.
      The objective and the safety-margin constraint share a per-call memoized
      forward rollout (keyed on u_vec.tobytes()) so the PDE is integrated only
      ONCE per SLSQP function evaluation, not twice.
    """
    dt = cfg.solver.dt
    n  = cfg.solver.n_steps
    N  = cfg.domain.N

    sar        = compute_sar_field(cfg=cfg)
    bioheat    = BioHeatSolver(cfg=cfg, sar_field=sar)
    damage     = ArrheniusDamage(cfg=cfg)
    cost_fn    = CostFunctional(cfg=cfg)
    checker    = ConstraintChecker(cfg=cfg)
    healthy_mask = checker.healthy_mask

    if u_init is None:
        # Time-optimal mode: the early-stop makes t_f a staircase function of u
        # (≈zero finite-difference gradient), so SLSQP cannot descend the min-time
        # objective and largely returns the warm start.  Minimum time wants maximum
        # power, so start at full power — this lands in the minimum-time region
        # instead of falsely "converging" at a slow interior guess.  Energy mode
        # keeps the neutral half-power start.
        u_init = (np.full(n, cfg.control.P_max) if cfg.cost.mode == 'time'
                  else np.full(n, cfg.control.P_max * 0.5))

    call_count  = [0]
    tumor_mask  = cost_fn.tumor_mask
    threshold   = cfg.arrhenius.damage_threshold

    # ── Memoized single forward integration ──────────────────────────────────
    # Cache stores only two scalars per key so memory stays bounded.
    # Created fresh each solve_direct() call (no cross-call leakage).
    _cache: dict = {}   # bytes → (J_value: float, safety_margin: float)

    def _evaluate(u_vec: np.ndarray):
        """
        Run the forward PDE once and return (J, safety_margin).

        The safety_margin is computed incrementally (one scalar updated each
        step) alongside J so no large arrays need to be stored in the cache.
        """
        key = u_vec.tobytes()
        if key in _cache:
            return _cache[key]

        call_count[0] += 1
        T     = bioheat.initialize()
        Omega = damage.initialize()
        J_run = 0.0
        # Track worst healthy-tissue temperature seen over all simulated steps.
        max_healthy_T = float('-inf')

        for k in range(n):
            P_k    = float(np.clip(u_vec[k], cfg.control.P_min, cfg.control.P_max))
            J_run += cost_fn.running_cost(T, P_k) * dt
            T      = bioheat.step(T, P_k, dt)
            Omega  = damage.step(Omega, T, dt)

            # Track worst healthy temperature using the POST-step state so the
            # final/stop-instant temperature is included in the safety margin.
            # (Sampling before the step would omit the hottest, last timestep —
            # the SLSQP constraint would then be leaky by one dt.)
            T_h_max = float(T[healthy_mask].max()) if healthy_mask.any() else float('-inf')
            if T_h_max > max_healthy_T:
                max_healthy_T = T_h_max

            # Free final time: once the tumor is fully ablated, stop the clock.
            # Running longer only adds positive running cost + γ₂·dt, so this
            # rewards controls that ablate sooner (the optimizer minimizes t_f).
            if (Omega[tumor_mask] >= threshold).all():
                J_val    = J_run + cost_fn.terminal_cost(Omega, (k + 1) * dt)
                s_margin = cfg.control.T_safe - max_healthy_T
                _cache[key] = (J_val, s_margin)
                return _cache[key]

        J_val    = J_run + cost_fn.terminal_cost(Omega, cfg.solver.t_final)
        s_margin = cfg.control.T_safe - max_healthy_T
        _cache[key] = (J_val, s_margin)
        return _cache[key]

    def objective(u_vec: np.ndarray) -> float:
        J_val, _ = _evaluate(u_vec)
        return J_val

    bounds = [(cfg.control.P_min, cfg.control.P_max)] * n

    # ── Hard safety constraint (time-optimal mode only) ───────────────────────
    if cfg.solver.enforce_safety_hard:
        # SLSQP 'ineq' constraint: fun(u) >= 0 means feasible.
        # Returns T_safe − max_{t, r ∈ Ω_H} T(r,t); negative → violated.
        nlp_constraints = [{'type': 'ineq',
                             'fun':  lambda u: _evaluate(u)[1]}]
    else:
        nlp_constraints = ()   # energy mode: no hard constraints (current behaviour)

    if verbose:
        mode_label = cfg.cost.mode
        hard_label = ' + hard T_safe constraint' if cfg.solver.enforce_safety_hard else ''
        print(f"Starting direct (single-shooting) solver  "
              f"(n={n} variables,  method={cfg.solver.optimizer},  "
              f"objective={mode_label}{hard_label}) ...")

    result = minimize(
        objective,
        u_init,
        method=cfg.solver.optimizer,
        bounds=bounds,
        constraints=nlp_constraints,
        options={'maxiter': cfg.solver.max_iter,
                 'ftol':    cfg.solver.tol,
                 'disp':    verbose},
    )

    if verbose:
        status = 'converged' if result.success else 'DID NOT converge'
        print(f"  Optimisation {status}  after {call_count[0]} evaluations.  "
              f"J* = {result.fun:.4e}")

    return forward_simulate(result.x, cfg, verbose=False)


# ── Mode 3: MPC (receding-horizon) ───────────────────────────────────────────

def solve_mpc(cfg: SimConfig = default_cfg,
              mri_feedback_fn=None,
              verbose: bool = True) -> dict:
    """
    Mode 3 — receding-horizon Model Predictive Control.

    At each MPC step:
      1. Receive MRI temperature observation (optionally noisy).
      2. Solve a short OCP over the prediction horizon T_p.
      3. Apply the first control action to the true (simulated) state.
      4. Advance the true state and repeat.

    Parameters
    ----------
    mri_feedback_fn : callable  T_obs = mri_feedback_fn(T_true, t), optional.
                      Simulates MRI acquisition noise / latency.
                      If None, the true simulated state is used directly.

    Note
    ----
    The inner horizon optimizer (_mpc_horizon_optimize) currently uses a
    bang-off-bang heuristic rather than a true OCP solver.  Replace it with
    solve_direct() or solve_indirect() applied to the prediction horizon for
    a full MPC implementation.
    """
    dt_ctrl = cfg.solver.mpc_dt_ctrl
    T_p     = cfg.solver.mpc_horizon
    t_total = cfg.solver.t_final
    dt      = cfg.solver.dt
    N       = cfg.domain.N

    n_ctrl_steps = int(t_total / dt_ctrl)
    n_inner      = int(dt_ctrl / dt)

    sar      = compute_sar_field(cfg=cfg)
    bioheat  = BioHeatSolver(cfg=cfg, sar_field=sar)
    damage   = ArrheniusDamage(cfg=cfg)
    cost_fn  = CostFunctional(cfg=cfg)
    checker  = ConstraintChecker(cfg=cfg)

    T_hist  = [bioheat.initialize()]
    Om_hist = [damage.initialize()]
    u_hist  = []
    t_vec   = [0.0]

    T_curr  = T_hist[0].copy()
    Om_curr = Om_hist[0].copy()

    for m in range(n_ctrl_steps):
        t_now = m * dt_ctrl

        T_obs  = mri_feedback_fn(T_curr, t_now) if mri_feedback_fn else T_curr
        n_pred = int(T_p / dt)
        P_best = _mpc_horizon_optimize(T_obs, Om_curr, n_pred, dt, sar,
                                        bioheat, damage, cost_fn, cfg)

        P_apply = P_best[0]
        for j in range(n_inner):
            T_curr  = bioheat.step(T_curr, P_apply, dt)
            Om_curr = damage.step(Om_curr, T_curr, dt)
            T_hist.append(T_curr.copy())
            Om_hist.append(Om_curr.copy())
            u_hist.append(P_apply)
            t_vec.append(t_now + (j + 1) * dt)

        ablated = checker.check_terminal(Om_curr, t_now)
        if verbose:
            frac = ablated['ablation']['fraction_ablated']
            print(f"  MPC t={t_now:6.1f}s  P={P_apply:5.1f}W  ablated={frac:.1%}")

        if ablated['all_satisfied']:
            if verbose:
                print(f"  Terminal constraint satisfied at t = {t_vec[-1]:.1f} s")
            break

    T_arr  = np.array(T_hist)
    Om_arr = np.array(Om_hist)
    u_arr  = np.array(u_hist)
    t_arr  = np.array(t_vec)
    t_f    = t_arr[-1]

    costs = cost_fn.total_cost(T_arr, Om_arr, u_arr, t_f, dt)
    cons  = checker.check_trajectory(T_arr, Om_arr, u_arr, t_f)

    return _result_dict(T_arr, Om_arr, u_arr, t_arr, costs, cons, cfg)


def _mpc_horizon_optimize(T0, Omega0, n_pred, dt, sar,
                           bioheat, damage, cost_fn, cfg):
    """
    Placeholder inner optimizer for MPC: bang-off-bang heuristic.

    Replace with solve_direct() or solve_indirect() on the horizon window
    for a true MPC implementation.
    """
    _, healthy_2d, _ = build_region_masks(cfg)
    healthy_mask = healthy_2d.ravel()

    T_max_h = T0[healthy_mask].max()
    if T_max_h < cfg.control.T_safe - 2.0:
        P_opt = cfg.control.P_max
    elif T_max_h < cfg.control.T_safe:
        P_opt = cfg.control.P_max * 0.5
    else:
        P_opt = 0.0

    return np.full(n_pred, P_opt)

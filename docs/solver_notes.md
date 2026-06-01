# 
---
forward → Not an optimization method

forward_simulate() just integrates the state equations forward with a prescribed bang-bang schedule. It produces no optimal u*. It's a simulation utility that the other two modes call internally for their final
trajectory.

---
openloop → 13.2 Direct Methods, but only the simplest variant: direct single-shooting

solve_open_loop() makes the full power trajectory u = [P_0, …, P_{n-1}] ∈ ℝ^n the decision variable and calls scipy.optimize.minimize(SLSQP). At every optimizer iteration it re-integrates the entire PDE from scratch.
That is exactly single-shooting.

What it is not:
- Not collocation (Hermite-Simpson, Radau, etc.) — there are no collocated state variables, only the control trajectory
- Not using an adjoint gradient — SLSQP approximates ∂J/∂u via finite differences internally; the AdjointSolver in adjoint.py is never called
- Path constraints (T ≤ T_safe in Ω_H) are only soft-penalized via α₂ in the cost, not enforced as hard NLP constraints

---
mpc → 13.3 MPC, but with a heuristic inner optimizer

solve_mpc() has the correct receding-horizon structure — observe MRI state, optimize over horizon T_p, apply first action, repeat. That matches 13.3.

However _mpc_horizon_optimize() is not actually solving an OCP over the horizon. It's a bang-off-bang rule:

if T_max_healthy < T_safe - 2:   P = P_max
elif T_max_healthy < T_safe:     P = P_max * 0.5
else:                            P = 0

A real MPC implementation would replace this with solve_open_loop() (or a faster direct collocation / QP) called on the current state over the T_p window.

---
AdjointSolver → 13.1, scaffolded but not wired in

control/adjoint.py is fully implemented — backward Euler integration of λ_T, terminal costate from transversality (λ_Ω(t_f) = −2γ₁(1−Ω_d)₊), and the Pontryagin projection P* = clip(λ_Tᵀ b_P / 2α₁, 0, P_max). But no
TPBVP shooter exists in solver.py that uses it. To implement 13.1 you'd need to add a function that:

1. Integrates state forward: x(t), t ∈ [0, t_f]
2. Sets terminal costate via terminal_costate(Omega[-1])
3. Integrates adjoint backward: λ(t), t ∈ [t_f, 0]
4. Reads off P*(t) from optimal_control(lam_T[t], b_P)
5. Iterates until the control trajectory converges (single shooting on the TPBVP)

---
Summary table

TPBVP shooter exists in solver.py that uses it. To implement 13.1 you'd need to add a function that:

1. Integrates state forward: x(t), t ∈ [0, t_f]
2. Sets terminal costate via terminal_costate(Omega[-1])
3. Integrates adjoint backward: λ(t), t ∈ [t_f, 0]
4. Reads off P*(t) from optimal_control(lam_T[t], b_P)
5. Iterates until the control trajectory converges (single shooting on the TPBVP)

---
Summary table

┌───────────────┬─────────┬───────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────┐
│  Solver mode  │ Section │        What's actually implemented        │                                 Gap                                  │
├───────────────┼─────────┼───────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ forward       │ —       │ Forward simulation only                   │ No optimization                                                      │
├───────────────┼─────────┼───────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ openloop      │ 13.2    │ Direct single-shooting via SLSQP          │ No adjoint gradients; no collocation; path constraints are soft only │
├───────────────┼─────────┼───────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ mpc           │ 13.3    │ Receding-horizon structure + MRI feedback │ Inner optimizer is a heuristic, not a real OCP solver                │
├───────────────┼─────────┼───────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ AdjointSolver │ 13.1    │ Full adjoint + Pontryagin law             │ No TPBVP shooter in solver.py to use it                              │
└───────────────┴─────────┴───────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────┘
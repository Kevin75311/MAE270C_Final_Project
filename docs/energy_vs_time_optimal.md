# Energy-Optimal vs. Time-Optimal Control

Notes on what changes — mathematically and in code — if we move the MRI ablation
OCP from its current **energy-optimal** (energy-dominated mixed cost) formulation
to a **time-optimal** one (drop or shrink the energy term). All notation follows
`control/adjoint.py` / `control/ocp.py`.

---

## 1. What "energy-optimal" means right now

The running cost is

```
L = α₁‖P‖²  +  α₂ ∫_{Ω_H} max(0, T − T_safe)² dr  +  α₃
```

The term that makes it **energy-optimal** is `α₁‖P‖²` — a *quadratic penalty on the
control itself*. There is already a time cost too (the `α₃` running constant and the
terminal `γ₂·t_f`), so the problem today is really a **mixed energy + time** cost,
energy-dominated.

The single most important consequence of the quadratic term is the **shape of the
optimal control law**.

---

## 2. The core difference: quadratic vs. linear Hamiltonian in P

The Hamiltonian is `ℋ = L + λᵀf`. Only `f_T` depends on `P`, through
`∂f_T/∂P = M⁻¹ B_P =: b_P`. So

```
ℋ = α₁P²  +  (terms without P)  +  (λ_Tᵀ b_P) P
```

**Energy-optimal (α₁ > 0):** `ℋ` is **quadratic and strictly convex** in `P`.
Stationarity gives an interior solution:

```
∂ℋ/∂P = 2α₁P + λ_Tᵀ b_P = 0   ⟹   P*(t) = clip( −λ_Tᵀ b_P / (2α₁), 0, P_max )
```

This is the law in the code. It is a **continuous, Lipschitz** function of the
costate — power ramps smoothly and saturates at the bounds only occasionally.

**Time-optimal (α₁ → 0):** the `P²` term vanishes and `ℋ` becomes **linear in P**:

```
∂ℋ/∂P = λ_Tᵀ b_P =: s(t)      (independent of P)
```

You can no longer solve `∂ℋ/∂P = 0` for `P` — there is nothing to solve.
Minimizing a linear function over `P ∈ [0, P_max]` gives a **bang-bang** law driven
by the **switching function** `s(t) = λ_Tᵀ b_P`:

```
P*(t) = P_max   if s(t) < 0
        0       if s(t) > 0
```

Note the current law literally **divides by α₁**, so it does **not** degrade
gracefully to the time-optimal case — it blows up. That discontinuity is the heart
of why this is not a one-line config change.

---

## 3. New objects that appear in the pure time-optimal case

1. **Switching function & switch detection.** The control is set by the *sign* of
   `s(t)`, not its magnitude. You must locate the zeros of `s(t)` (switching times).
   Bang-bang optimizers typically make the **switching times** the unknowns, not a
   power-vs-time array.

2. **Singular arcs.** If `s(t) ≡ 0` on a whole interval, PMP is silent on `P` there —
   you need higher-order conditions (Kelley / generalized Legendre–Clebsch) to
   recover `P`. Quadratic regularization sidesteps this entirely.

3. **State-constrained boundary arcs.** Physically important. With energy penalized,
   the `α₁‖P‖²` term itself keeps power moderate. Remove it and nothing stops the
   solver from blasting `P_max` — except the path constraint `T ≤ T_safe` in `Ω_H`.
   The true time-optimal solution then looks like:

   > full power → ride the constraint `T = T_safe` → off.

   That middle "ride the wall" segment is a **state-constrained arc**, introducing a
   constraint multiplier and **jump (corner) conditions on the costate `λ_T`** at arc
   entry/exit. Currently `T ≤ T_safe` is a *soft penalty* (the `α₂` term), not a hard
   constraint — fine for energy-optimal, but for time-optimal you almost certainly
   want it as a **hard path constraint**, which is a separate structural change.

---

## 4. Free final time / transversality also changes meaning

Today `t_f` is pinned to the terminal constraint `Ω_d ≥ 1` because `ℋ(t_f) = −γ₂` is
infeasible — running longer only burns positive running cost, so the optimum sits on
the active terminal constraint.

In a **pure time-optimal** problem, `t_f` *is* the objective, so the transversality
condition

```
ℋ(t_f) + ∂Φ/∂t_f = 0
```

is what determines the stopping time, with the binding driver being "tumor just
reached `Ω_d ≥ 1` everywhere." Conceptually similar endpoint, but now it is the
genuine first-order condition rather than a constraint you happened to land on.

---

## 5. Why the indirect solver specifically struggles

The `indirect` mode is a gradient-projection fixed-point iteration, and it already
limit-cycles for bang-bang controls (hence `β ≈ 0.15`). That chattering *is* the
bang-bang structure leaking through. As `α₁ → 0` the control law gets steeper until
it is a step function, and the iteration oscillates between
"under-ablated → P_max" and "ablated → 0" with no damping value in between. So pure
time-optimal does not just need a new control law — it tends to break the existing
solver, which is why a switching-time / direct-collocation approach is the honest
path.

---

## 6. The practical spectrum

Two genuinely different moves, at very different costs:

| Approach | Math | Code cost |
|---|---|---|
| **Small α₁ (regularized time-optimal)** | Keep `α₁ > 0` but tiny; raise `γ₂`/`α₃`. `ℋ` stays quadratic, control stays the smooth `clip(−λ_Tᵀb_P/2α₁, …)` law. As `α₁ ↓` the smooth law *approximates* bang-bang. | **Config only.** Essentially the gain-tuning already done. A homotopy (`α₁: 1e-4 → 1e-6 → …`) approaches time-optimal while keeping the existing solver stable. |
| **Pure time-optimal (α₁ = 0)** | Linear `ℋ`, true bang-bang via `sign(s(t))`, switching-time unknowns, singular-arc handling, hard `T ≤ T_safe` path constraint with costate jumps. | **Structural.** Rewrite the control law in `adjoint.py` (the `/2α₁` is invalid), switch `indirect` to switching-time parameterization or move to direct collocation (CasADi), and promote the safety penalty to a hard constraint. |

**Summary:** energy-optimal gives a strictly convex Hamiltonian and a continuous
interior control law; time-optimal gives a Hamiltonian linear in the control, so the
optimum is bang-bang governed by the switching function `s(t) = λ_Tᵀ b_P`, plus
singular and state-constrained arcs. The first lives inside the current code; the
second changes the *type* of solution and therefore the solver.

---

## 7. How it is implemented (small-change path)

Time-optimal control is implemented **entirely inside `direct` mode** (SLSQP
single-shooting), which sidesteps the bang-bang rewrite of `control/adjoint.py`.
SLSQP optimizes the power array directly and natively supports nonlinear
inequality constraints, so the bang-bang/boundary-arc solution emerges on its own.

Activate with: `python main.py --mode direct --objective time`.

Changes (all additive; energy mode is the unchanged default):

- **`config.py`** — `CostConfig.mode` (`'energy'|'time'`), `CostConfig.alpha1_time`
  (= `1e-6`, tiny energy regularization so the SLSQP objective is not a degenerate
  LP), and `SolverConfig.enforce_safety_hard` (= `False`).
- **`control/ocp.py`** — in `'time'` mode `CostFunctional` uses `α₁ = alpha1_time`
  and `α₂ = 0` (the soft safety penalty is replaced by a hard constraint).
- **`control/constraints.py`** — `ConstraintChecker.safety_margin(T_history)` returns
  `T_safe − max_{t, Ω_H} T`; `≥ 0` ⇔ feasible (SLSQP `ineq` form).
- **`control/solver.py`** — `solve_direct` gained a per-call memoized rollout
  `_evaluate(u)` (keyed on `u.tobytes()`, caches only the two scalars `(J,
  safety_margin)`) so the objective and the hard safety constraint share **one**
  PDE integration per SLSQP evaluation. When `enforce_safety_hard` is `True`, an
  `{'type':'ineq'}` constraint enforcing `T ≤ T_safe` in healthy tissue is passed to
  `minimize`. The healthy-temperature peak is sampled on the **post-step** state so
  the final/stop instant is included (a pre-step sample would leave the constraint
  leaky by one `dt`).
- **`main.py`** — `--objective {energy,time}` flag; `time` sets `cfg.cost.mode='time'`
  and `cfg.solver.enforce_safety_hard=True`.

What is intentionally **not** done (the structural "big change" we avoided): the
linear-`ℋ` bang-bang control law / switching-time parameterization in the `indirect`
solver, singular-arc handling, and costate jump conditions for state-constrained
arcs. In `direct` mode SLSQP absorbs all of this numerically.

Validation (reduced 24×24 grid, 12 iters): time mode keeps the healthy-tissue
temperature within floating-point tolerance of `T_safe` (margin ≈ `−3×10⁻⁷ °C`,
feasible) while driving ablation; energy mode output is unchanged.

### 7.1 Solver limitation: staircase `t_f` and warm-start dependence

Direct single-shooting **cannot itself descend the minimum-time objective.**
Because the trajectory truncates at the first fully-ablated step, the optimized
time is `t_f = (k_stop+1)·dt` — a **step (staircase) function of `u`**. Small
finite-difference perturbations almost never change *which* step `k_stop` ablation
completes on, so SLSQP measures a ≈zero objective gradient and stalls at the warm
start (`Exit mode 0` after **1 iteration**). Consequences:

- **Time mode warm-starts at full power** (`P_max`), the minimum-time region;
  energy mode keeps the neutral half-power start. (`solve_direct`, `u_init=None`.)
- **Slack safety constraint** → full power is *provably* minimum-time (you cannot
  ablate faster than at `P_max`), so the 1-iteration passthrough returns the
  correct answer. Full grid, `T_safe=50°C`: full power, `t_f=92 s`, 100% ablated,
  peak healthy 47.6°C (feasible).
- **Active safety constraint** → its gradient is non-zero, so SLSQP does genuine
  work, backing power off to ride `T = T_safe` (the boundary arc). `T_safe=45°C`:
  full ablation is infeasible; best feasible ≈ 76.5%, healthy tissue parked exactly
  on 45°C.

So the gap between open-loop and `direct` in **time-optimal** mode is created by the
**constraint binding, not by cost weights** (unlike energy mode, where `α₁` tuning
creates the gap). Tuning `α₁`/`γ₂` barely moves the time-optimal solution; lowering
`T_safe` (or tightening the geometry / SAR focus) until full power overheats is what
forces a non-trivial optimal control that a fixed schedule cannot match.

A solver that truly minimizes `t_f` from an arbitrary start (no warm-start
dependence) needs the switching-time or direct-collocation formulation — the
structural change deliberately avoided in §7.

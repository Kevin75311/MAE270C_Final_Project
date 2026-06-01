# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation modes
python main.py --mode openloop      # fixed bang-bang schedule, no optimization (default)
python main.py --mode indirect      # PMP gradient projection (forward-adjoint sweeps)
python main.py --mode direct        # direct single-shooting NLP (scipy SLSQP)
python main.py --mode mpc           # receding-horizon MPC
python main.py --mode openloop --animate  # also generate GIF

# Run a single module as a script (each physics/control module has __main__)
python -m physics.bioheat
python -m physics.sar_model
python config.py                    # print config summary
```

Results are written to `results/`.

## Architecture

The simulation is a PDE-constrained optimal control problem (Bolza form, free final time) for MRI-guided microwave thermal ablation. The domain `Ω` is partitioned into tumor `Ω_T`, healthy tissue `Ω_H`, and safety margin `Ω_M`.

### State equations

**State Eq. 1 — Pennes bioheat PDE** (`physics/bioheat.py`)

```
ρc ∂T/∂t = ∇·(k∇T) − ω_b ρ_b c_b (T − T_b) + Q_met + Q_source(r, u, t)
```

Discretized to the ODE implemented in `BioHeatSolver.rhs()`:

```
ẋ_T = M⁻¹ [ K_d x_T  −  W_b (x_T − T_b·1)  +  Q_met  +  B_P·P(t) ]
```

where `M = diag(ρᵢcᵢ)`, `K_d` is the sparse discrete Laplacian weighted by `k`, and `W_b = diag(ω_b,i ρ_b c_b)`.

**State Eq. 2 — Arrhenius damage ODE** (`physics/arrhenius.py`)

```
dΩ_d/dt = A · exp(−E_a / (R · T_K)),    Ω_d(r, 0) = 0
```

`T_K = T_celsius + 273.15` — must convert to Kelvin before evaluating the exponential. Threshold `Ω_d = 1` corresponds to 63% cell death probability (irreversible necrosis).

**SAR heat source** (`physics/sar_model.py`) — Gaussian near-field approximation:

```
Q_source(r, u, t) = SAR(r) · P(t) = SAR_peak · exp(−|r − r_probe|² / (2σ_sar²)) · P(t)
```

### Cost functional (Bolza form) — `control/ocp.py`

```
J[u, t_f] = γ₁ ∫_{Ω_T} max(0, 1 − Ω_d(t_f))² dr  +  γ₂ t_f          [terminal Φ]
           + ∫₀^{t_f} [ α₁‖u‖²  +  α₂ ∫_{Ω_H} max(0, T − T_safe)² dr  +  α₃ ] dt   [running L]
```

Weights `α₁, α₂, α₃, γ₁, γ₂` are in `config.py → CostConfig`.

### Constraints — `control/constraints.py`

Path constraints (enforced at every timestep):
- `T(r,t) ≤ T_safe` in `Ω_H` (healthy tissue temperature limit)
- `0 ≤ P(t) ≤ P_max` (control bounds)
- `|dP/dt| ≤ Ṗ_max` (slew rate limit)

Terminal constraints (evaluated at `t_f`):
- `Ω_d(r, t_f) ≥ 1` in `Ω_T` (full tumor ablation)
- `t_f ≤ T_max` (max treatment duration)

### Adjoint equations and optimal control law — `control/adjoint.py`

Costates `λ = [λ_T, λ_Ω]ᵀ` satisfy backward ODEs:

```
−λ̇_T = A_dᵀ λ_T  −  W_b λ_T
       + 2α₂ max(0, T−T_safe) · 1_{Ω_H}
       + λ_Ω ⊙ [ A (E_a / R T²) exp(−E_a / RT) ]

−λ̇_Ω = 0    ⟹    λ_Ω(t) = λ_Ω(t_f) = const
```

Terminal conditions:
```
λ_T(t_f) = 0
λ_Ω(t_f) = −2γ₁ max(0, 1 − Ω_d(r, t_f)) · 1_{Ω_T}
```

Pontryagin optimal control law (projection onto control bounds):
```
P*(t) = clip( λ_Tᵀ B_P / (2α₁),  0,  P_max )
```

### Module map

| Layer | Module | Role |
|-------|--------|------|
| Config | `config.py` | Single source of truth; import `from config import cfg` everywhere |
| Physics | `physics/mesh.py` | Grid, region masks (tumor/healthy/margin), voxel area |
| Physics | `physics/discretization.py` | Assembles sparse K_d (diffusion) and W_b (perfusion) matrices |
| Physics | `physics/boundary_conditions.py` | Dirichlet/Neumann/Robin BC application |
| Physics | `physics/bioheat.py` | `BioHeatSolver` — Euler/RK4 integrator for temperature field |
| Physics | `physics/arrhenius.py` | `ArrheniusDamage` — damage accumulation integrator |
| Physics | `physics/sar_model.py` | `compute_sar_field()`, `get_control_input_vector()` |
| Control | `control/ocp.py` | `CostFunctional` — Bolza cost J with running + terminal terms |
| Control | `control/constraints.py` | `ConstraintChecker` — path and terminal constraint verification |
| Control | `control/adjoint.py` | Costate ODEs + Pontryagin optimal control law |
| Control | `control/solver.py` | `forward_simulate()`, `solve_open_loop()`, `solve_mpc()` |
| Utils | `utils/io_utils.py` | `save_trajectory()` — saves result dicts to `results/` |
| Utils | `utils/validators.py` | `check_stability_cfl()`, `check_temperature_physical()` |
| Viz | `visualization/field_plots.py` | 2D temperature/SAR field plots, optional PyVista 3D |
| Viz | `visualization/damage_plots.py` | Damage field and necrosis boundary plots |
| Viz | `visualization/control_plots.py` | Power history, temperature traces, cost breakdown |
| Viz | `visualization/animation.py` | GIF generation via Pillow |

### Data flow

`main.py` orchestrates: config → mesh → SAR precomputation → solver → constraint check → visualization → save.

All three solvers (`forward_simulate`, `solve_open_loop`, `solve_mpc`) return the same result dict:

```python
{
    'T_history':     np.ndarray,   # (n_steps+1, N)  flattened temperature
    'Omega_history': np.ndarray,   # (n_steps+1, N)  flattened damage
    'u_history':     np.ndarray,   # (n_steps,)       applied power [W]
    't_vec':         np.ndarray,   # (n_steps+1,)     time points [s]
    'cost':          dict,         # J_total, J_running, J_terminal, components
    'constraints':   dict,         # running and terminal constraint results
    'cfg':           SimConfig,
}
```

### Key design decisions

- **`cfg` is global** — `config.py` exports a singleton `cfg = SimConfig()`. Pass it explicitly to classes/functions; never hardcode numerical values — always reference `cfg.*`.
- **Flat spatial arrays** — temperature and damage are stored as 1D vectors of length `N = Nx × Ny` (row-major). Region masks from `build_region_masks(cfg)` are 2D arrays; call `.ravel()` before indexing into flat state vectors.
- **System matrices built once** — `BioHeatSolver.__init__` assembles the sparse diffusion and perfusion matrices; they are reused every timestep.
- **`direct` mode is slow** — re-integrates the full PDE at every optimizer function evaluation (single-shooting). For large grids, use `indirect` (adjoint gradients) or direct collocation (CasADi).
- **`indirect` convergence** — the gradient projection algorithm uses a relaxation parameter β (default 0.5). If it oscillates, reduce β; if it converges slowly, increase it. The algorithm is not guaranteed to converge for all cost weight combinations.
- **MPC horizon optimizer is a heuristic** — `_mpc_horizon_optimize` in `solver.py` uses a bang-off-bang rule, not a true OCP solver. Replace with `solve_direct()` or `solve_indirect()` on the horizon window for a real MPC.

## Parameter quick-reference

| Symbol | Config field | Default | Units |
|--------|-------------|---------|-------|
| ρ, c, k | `TissueConfig.rho/c/k` | 1050, 3600, 0.51 | kg/m³, J/(kg·K), W/(m·K) |
| ω_b, T_b | `TissueConfig.omega_b/T_blood` | 0.005, 37.0 | s⁻¹, °C |
| A, E_a | `ArrheniusConfig.A/E_a` | 3.1×10⁹⁸, 6.28×10⁵ | s⁻¹, J/mol |
| P_max, T_safe | `ControlConfig.P_max/T_safe` | 50.0, 45.0 | W, °C |
| α₁, α₂, α₃ | `CostConfig.alpha1/2/3` | 1e-4, 1.0, 0.01 | — |
| γ₁, γ₂ | `CostConfig.gamma1/2` | 10.0, 0.1 | — |
| dt, t_final | `SolverConfig.dt/t_final` | 1.0, 300.0 | s |

## Coding conventions

Every state equation implementation in `physics/` uses this banner:

```python
# ┌─────────────────────────────────────────────────────────────┐
# │  STATE EQUATION N — [name]                                  │
# │  [mathematical form]                                        │
# └─────────────────────────────────────────────────────────────┘
```

Import pattern:

```python
from config import cfg           # global singleton
from config import SimConfig     # type annotation only
```

## Visualization stack

| Need | Library | Location |
|------|---------|----------|
| 2D temperature/damage heatmaps | `matplotlib.pcolormesh` | `field_plots.py`, `damage_plots.py` |
| Contour overlays (tumor boundary, Ω_d=1) | `matplotlib.contour` | `damage_plots.py` |
| 3D volume render (COMSOL-like) | `pyvista` | `plot_temperature_3d_pyvista()` |
| 3D necrosis isosurface | `pyvista` | `plot_damage_3d_pyvista()` |
| GIF/MP4 animation | `matplotlib.FuncAnimation` + Pillow/ffmpeg | `animation.py` |

Install PyVista for 3D output: `pip install pyvista`

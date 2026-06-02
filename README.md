# MRI Thermal Ablation — Optimal Control Simulation

PDE-constrained optimal control of microwave thermal ablation, guided by real-time MRI thermometry.

## Environment setup (conda)

1. **Install Miniconda** (skip if already installed):
   - macOS/Linux:
     ```bash
     curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
     bash miniconda.sh -b -p "$HOME/miniconda3"
     eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
     ```
   - Windows (PowerShell):
     ```powershell
     winget install -e --id Anaconda.Miniconda3
     # then restart your terminal
     ```

2. **Create and activate the environment:**
   ```bash
   conda create -n mri_ablation python=3.11 -y
   conda activate mri_ablation
   ```

3. **Install dependencies:**
   ```bash
   conda install -c conda-forge numpy scipy matplotlib pillow -y
   pip install pyvista          # 3D interactive rendering (optional)
   ```
   Or via the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify:**
   ```bash
   python config.py             # prints config summary — should show no errors
   ```

## Quick start

```bash
conda activate mri_ablation
python main.py                 # defaults: openloop, 2D, dirichlet BC, point source
```

All four arguments (`--mode`, `--ndim`, `--bc`, `--probe`) are **freely combinable**.

## Arguments

### `--mode` — Solver / optimizer

| Value | Description |
|-------|-------------|
| `openloop` | Fixed bang-bang power schedule, no optimization (default, fast) |
| `indirect` | PMP gradient projection — forward-adjoint sweep iterations |
| `direct` | Single-shooting NLP solved with scipy SLSQP |
| `mpc` | Receding-horizon Model Predictive Control with simulated MRI noise |

```bash
python main.py --mode openloop
python main.py --mode indirect
python main.py --mode direct
python main.py --mode mpc
```

### `--ndim` — Spatial dimensionality

| Value | Grid | SAR model | Visualization |
|-------|------|-----------|---------------|
| `2` (default) | Nx × Ny | 2D slice | Single x-y heatmap |
| `3` | Nx × Ny × Nz | Full 3D volume | Axial (x-y at z=probe_z) + Meridional (x-z at y=probe_y) |

```bash
python main.py --mode openloop --ndim 2   # 2D (default)
python main.py --mode openloop --ndim 3   # 3D
```

### `--bc` — Boundary condition on ∂Ω

| Value | Equation | Parameters (config.py) |
|-------|----------|------------------------|
| `dirichlet` (default) | T = T_wall on boundary | `BoundaryConfig.T_wall` = 37 °C |
| `neumann` | k ∂T/∂n = 0 — insulated, zero flux | — |
| `robin` | k ∂T/∂n + h_c(T − T_inf) = 0 — convective cooling | `BoundaryConfig.h_c` = 50 W/(m²·K), `T_inf` = 20 °C |

```bash
python main.py --mode openloop --bc dirichlet   # fixed wall temp (default)
python main.py --mode openloop --bc neumann     # insulated boundary
python main.py --mode openloop --bc robin       # convective cooling
```

### `--probe` — SAR / antenna model

| Value | Formula | Use case |
|-------|---------|----------|
| `point` (default) | `SAR_peak · exp(−‖r−r_p‖²/2σ_r²)` (2D isotropic; 3D anisotropic with σ_z) | Baseline, fast |
| `line` | Flat cylinder of length `L_active` + Gaussian end-cap rolloff | Realistic microwave needle in r-z plane |
| `dipole` | `SAR_peak · sin²(θ) · exp(−r²/2σ²)` toroidal near-field pattern | Slot antenna approximation |

Probe position (`probe_position` in config) is always the **midpoint** of the active zone.
Needle orientation is set by `probe_direction` (default `(0,0,1)` = z-axis).

```bash
python main.py --mode openloop --probe point    # isotropic Gaussian (default)
python main.py --mode openloop --probe line     # realistic needle model
python main.py --mode openloop --probe dipole   # toroidal antenna pattern
```

### Other flags

| Flag | Description |
|------|-------------|
| `--animate` | Save a GIF animation to `results/ablation.gif` (2D only, slow) |
| `--no-save` | Skip saving the trajectory `.npz` file |

## Example combinations

All arguments are independent and freely mixed:

```bash
# Minimal defaults
python main.py

# Realistic needle + adjoint optimization, 2D
python main.py --mode indirect --probe line

# Full 3D MPC with line-source probe, Robin cooling
python main.py --mode mpc --ndim 3 --probe line --bc robin

# Direct NLP + dipole antenna + insulated boundary, with animation
python main.py --mode direct --probe dipole --bc neumann --animate

# 3D open-loop baseline, all combinations
python main.py --mode openloop --ndim 3 --bc dirichlet --probe point
python main.py --mode openloop --ndim 3 --bc robin     --probe line
python main.py --mode openloop --ndim 3 --bc neumann   --probe dipole
```

Results are written to `results/`.

## State equations

| # | Equation | File |
|---|---|---|
| 1 | Pennes bioheat PDE (discretized ODE) | `physics/bioheat.py` |
| 2 | Arrhenius damage ODE | `physics/arrhenius.py` |
| 3 | SAR heat source (control input) | `physics/sar_model.py` |

## Configuration

All parameters live in `config.py`. Key tunable values:

| Parameter | Config field | Default | Notes |
|-----------|-------------|---------|-------|
| Dimensionality | `DomainConfig.ndim` | `2` | Set to `3` for 3D |
| Domain size (xy) | `DomainConfig.Lx/Ly` | 5 × 5 cm | |
| Domain depth (z) | `DomainConfig.Lz` | 5 cm | Used when `ndim=3` |
| Grid resolution | `DomainConfig.Nx/Ny/Nz` | 50 × 50 × 50 | Nz ignored in 2D |
| Tumor center | `DomainConfig.tumor_center` | `(0.025, 0.025, 0.025)` m | Always 3-tuple |
| Probe position | `DomainConfig.probe_position` | `(0.025, 0.025, 0.025)` m | Midpoint of active zone |
| Probe model | `SARConfig.probe_model` | `'point'` | `'point'` \| `'line'` \| `'dipole'` |
| Active length | `SARConfig.L_active` | 30 mm | Used by `'line'` model |
| Needle direction | `SARConfig.probe_direction` | `(0, 0, 1)` | Unit vector; z-axis default |
| Radial SAR width | `SARConfig.sigma_sar` | 6 mm | All models (⊥ to needle) |
| Axial SAR width | `SARConfig.sigma_sar_z` | 15 mm | `'point'` full axial spread; `'line'` end-cap falloff |
| BC type | `BoundaryConfig.bc_type` | `'dirichlet'` | `'neumann'` or `'robin'` |
| Robin h_c | `BoundaryConfig.h_c` | 50 W/(m²·K) | Used when `bc_type='robin'` |
| Robin T_inf | `BoundaryConfig.T_inf` | 20 °C | Used when `bc_type='robin'` |
| Max power | `ControlConfig.P_max` | 50 W | |
| Safety temperature | `ControlConfig.T_safe` | 45 °C | |
| Cost weights | `CostConfig.alpha1/2/3, gamma1/2` | see config | |

## Visualization

| Mode | Output |
|------|--------|
| 2D | Single x-y heatmap with isotherms, tumor boundary, necrosis contour |
| 3D | **Axial view** (x-y slice at z = probe_z) + **Meridional view** (x-z slice at y = probe_y, z on vertical axis) |
| PyVista (optional) | Interactive 3D volume render + isosurfaces — `plot_temperature_3d_pyvista()`, `plot_damage_3d_pyvista()` |

```bash
pip install pyvista   # if not already installed
python -c "
from config import cfg
from physics.sar_model import compute_sar_field
from physics.bioheat import BioHeatSolver
from visualization.field_plots import plot_temperature_3d_pyvista
cfg.domain.ndim = 3
sar = compute_sar_field(cfg=cfg)
solver = BioHeatSolver(cfg=cfg, sar_field=sar)
T = solver.initialize()
plot_temperature_3d_pyvista(T, cfg=cfg)
"
```

## Running tests

```bash
python tests/run_all_tests.py          # all 9 tests × 2D + 3D = 18 runs
python tests/run_all_tests.py --2d     # 2D only
python tests/run_all_tests.py --3d     # 3D only

# Individual test files (each independently runnable):
python tests/test_01_mesh.py
python tests/test_02_sar.py            # covers point, line, dipole × 2D/3D
python tests/test_03_discretization.py
python tests/test_04_boundary_conditions.py
python tests/test_05_bioheat.py
python tests/test_06_arrhenius.py
python tests/test_07_coupled.py

# Via pytest:
pytest tests/ -v
```

Figures are saved to `results/tests/`.

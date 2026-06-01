# MRI Thermal Ablation — Optimal Control Simulation

PDE-constrained optimal control of microwave thermal ablation, guided by real-time MRI thermometry.

## Environment setup (conda)

1. **Install Miniconda** (skip if already installed):
   - Download the installer for your OS from https://docs.conda.io/en/latest/miniconda.html and run it, or use the one-liner below.
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
   pip install pyvista          # 3D rendering (optional but recommended)
   ```
   Or equivalently via the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify:**
   ```bash
   python config.py             # should print the config summary with no errors
   ```

## Quick start

```bash
conda activate mri_ablation
python main.py --mode openloop         # fixed bang-bang schedule, no optimization
python main.py --mode indirect         # PMP gradient projection (forward-adjoint sweeps)
python main.py --mode direct           # direct single-shooting NLP (scipy SLSQP)
python main.py --mode mpc              # receding-horizon MPC
python main.py --mode openloop --animate  # + generate GIF
```

Results are written to `results/`.

## State equations

| # | Equation | File |
|---|---|---|
| 1 | Pennes bioheat PDE (discretized ODE) | `physics/bioheat.py` |
| 2 | Arrhenius damage ODE | `physics/arrhenius.py` |
| 3 | SAR heat source (control input) | `physics/sar_model.py` |

## Configuration

All parameters are in `config.py`. Key tunable values:

| Parameter | Location | Default |
|---|---|---|
| Domain size | `DomainConfig.Lx/Ly` | 5 × 5 cm |
| Grid resolution | `DomainConfig.Nx/Ny` | 50 × 50 |
| Max power | `ControlConfig.P_max` | 50 W |
| Safety temperature | `ControlConfig.T_safe` | 45°C |
| Cost weights | `CostConfig.alpha1/2/3, gamma1/2` | see config |

## 3D visualization (COMSOL-like)

Requires `pyvista` (installed in step 3 above):

```bash
python -c "from visualization.field_plots import plot_temperature_3d_pyvista; ..."
```

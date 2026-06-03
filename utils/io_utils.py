"""
utils/io_utils.py — Save and load simulation results.
"""

import os
import numpy as np
import json
import yaml
from dataclasses import asdict
from datetime import datetime
from config import SimConfig, cfg as default_cfg


def save_trajectory(result: dict, run_name: str = None,
                    cfg: SimConfig = default_cfg):
    """Save full trajectory to compressed .npz + metadata JSON."""
    out_dir = cfg.viz.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    npz_path  = os.path.join(out_dir, f"{run_name}.npz")
    meta_path = os.path.join(out_dir, f"{run_name}_meta.json")

    np.savez_compressed(
        npz_path,
        T_history=result['T_history'],
        Omega_history=result['Omega_history'],
        u_history=result['u_history'],
        t_vec=result['t_vec'],
    )

    meta = {
        'run_name':   run_name,
        'timestamp':  datetime.now().isoformat(),
        'cost':       {k: float(v) for k, v in result['cost'].items()},
        'constraints': result['constraints'],
        'cfg': {
            'Nx': cfg.domain.Nx, 'Ny': cfg.domain.Ny,
            'dt': cfg.solver.dt, 't_final': cfg.solver.t_final,
            'P_max': cfg.control.P_max, 'T_safe': cfg.control.T_safe,
        }
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Saved: {npz_path}  +  {meta_path}")
    return npz_path


def load_trajectory(npz_path: str) -> dict:
    """Load a saved trajectory from .npz file."""
    data = np.load(npz_path)
    return {
        'T_history':     data['T_history'],
        'Omega_history': data['Omega_history'],
        'u_history':     data['u_history'],
        't_vec':         data['t_vec'],
    }


def save_config(cfg: SimConfig, out_dir: str):
    """
    Serialize the full SimConfig to config.yaml inside out_dir.

    Computed properties (dx, dy, dz, N) are added under a '_computed' key
    so they appear alongside the editable fields for easy comparison.
    """
    data = asdict(cfg)

    # Attach computed/derived quantities that don't live in the dataclass fields
    data['_computed'] = {
        'dx_mm':  round(cfg.domain.dx * 1e3, 4),
        'dy_mm':  round(cfg.domain.dy * 1e3, 4),
        'dz_mm':  round(cfg.domain.dz * 1e3, 4),
        'N_voxels': cfg.domain.N,
        'n_steps': cfg.solver.n_steps,
    }

    yaml_path = os.path.join(out_dir, 'config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Saved: {yaml_path}")
    return yaml_path

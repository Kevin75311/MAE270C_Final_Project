"""
run_direct3d_coarse.py — Rerun the three direct, 3-D cases at a coarser grid.

Direct single-shooting SLSQP is intractable at the default 50^3 = 125,000
voxels (>9 h, no convergence). This reruns direct / 3-D / dirichlet for each
probe at a reduced grid where SLSQP converges, so the direct_3d cells in the
report can be filled. Same gains, same BC, only the grid is coarsened.

Usage:
    python run_direct3d_coarse.py                 # all 3 probes, grid=20
    python run_direct3d_coarse.py --grid 16
    python run_direct3d_coarse.py --probe point --grid 20   # single (worker)
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(SCRIPT_DIR, 'logs')
PROBES     = ['point', 'line', 'dipole']


def _worker_code(probe: str, grid: int) -> str:
    return (
        "from config import cfg;"
        f"cfg.domain.Nx=cfg.domain.Ny=cfg.domain.Nz={grid};"
        "import main;"
        f"main.main(mode='direct', ndim=3, bc='dirichlet', probe='{probe}')"
    )


def run_one(probe: str, grid: int) -> tuple[str, int, float]:
    tag      = f"direct_3d_{probe}_g{grid}"
    log_path = os.path.join(LOG_DIR, f"{tag}.log")
    start    = datetime.now()
    print(f"[START] {tag}", flush=True)
    with open(log_path, 'w', encoding='utf-8') as fh:
        proc = subprocess.run(
            [sys.executable, '-c', _worker_code(probe, grid)],
            stdout=fh, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR,
            text=True, encoding='utf-8', errors='replace',
        )
    elapsed = (datetime.now() - start).total_seconds()
    status  = 'OK' if proc.returncode == 0 else f'FAILED rc={proc.returncode}'
    print(f"[{status:>12}] {tag:<28} {elapsed:>6.0f}s  -> logs/{tag}.log",
          flush=True)
    return tag, proc.returncode, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=20,
                    help='Cubic grid size N for Nx=Ny=Nz (default 20)')
    ap.add_argument('--probe', type=str, default=None, choices=PROBES,
                    help='Run a single probe (worker mode); default runs all 3')
    args = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.probe is not None:
        run_one(args.probe, args.grid)
        return

    print(f"Coarse direct-3D rerun  |  grid={args.grid}^3={args.grid**3} voxels"
          f"  |  bc=dirichlet  |  gains=non-aggressive")
    print("-" * 70, flush=True)
    wall = datetime.now()
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(run_one, p, args.grid): p for p in PROBES}
        for f in as_completed(futs):
            results.append(f.result())
    mins = (datetime.now() - wall).total_seconds() / 60
    print("\n" + "=" * 70)
    print(f"  Done in {mins:.1f} min")
    for tag, rc, el in sorted(results):
        print(f"  {'OK' if rc == 0 else 'FAIL':<4} {tag:<28} {el:>6.0f}s  rc={rc}")


if __name__ == '__main__':
    main()

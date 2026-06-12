"""
run_all.py — Run every mode × ndim × probe combination in parallel subprocesses.

Combinations: 4 modes × 2 dims × 3 probes = 24 runs
All runs use --bc dirichlet and the default (non-aggressive) cost gains:
  alpha1=5e-4, gamma1=1e6, gamma2=0.1

Usage:
    python run_all.py               # 6 parallel workers (default)
    python run_all.py --workers 4   # adjust parallelism
"""

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

MODES  = ['openloop', 'indirect', 'direct', 'mpc']
NDIMS  = [2, 3]
PROBES = ['point', 'line', 'dipole']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(SCRIPT_DIR, 'logs')


def run_one(mode: str, ndim: int, probe: str) -> tuple[str, int, float]:
    tag      = f"{mode}_{ndim}d_{probe}"
    log_path = os.path.join(LOG_DIR, f"{tag}.log")
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'main.py'),
        '--mode',  mode,
        '--ndim',  str(ndim),
        '--bc',    'dirichlet',
        '--probe', probe,
    ]
    start = datetime.now()
    print(f"[START] {tag}", flush=True)

    with open(log_path, 'w', encoding='utf-8') as fh:
        proc = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=SCRIPT_DIR,
            text=True,
            encoding='utf-8',
            errors='replace',
        )

    elapsed = (datetime.now() - start).total_seconds()
    status  = 'OK' if proc.returncode == 0 else f'FAILED rc={proc.returncode}'
    print(f"[{status:>12}] {tag:<30} {elapsed:>6.0f}s  -> logs/{tag}.log",
          flush=True)
    return tag, proc.returncode, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6,
                        help='Max simultaneous subprocess workers (default 6)')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    combos = list(itertools.product(MODES, NDIMS, PROBES))
    print(f"Launching {len(combos)} runs  |  workers={args.workers}  |  "
          f"bc=dirichlet  |  gains=non-aggressive")
    print(f"Logs -> {LOG_DIR}/")
    print("-" * 70, flush=True)

    wall_start = datetime.now()
    results: list[tuple[str, int, float]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, m, n, p): (m, n, p)
            for m, n, p in combos
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    wall_elapsed = (datetime.now() - wall_start).total_seconds()

    ok_runs  = sorted((t, e) for t, r, e in results if r == 0)
    bad_runs = sorted((t, r, e) for t, r, e in results if r != 0)

    print("\n" + "=" * 70)
    print(f"  All done in {wall_elapsed/60:.1f} min  |  "
          f"{len(ok_runs)} passed, {len(bad_runs)} failed")
    print("=" * 70)
    for tag, elapsed in ok_runs:
        print(f"  OK   {tag:<35} {elapsed:>6.0f}s")
    for tag, rc, elapsed in bad_runs:
        print(f"  FAIL {tag:<35} {elapsed:>6.0f}s  rc={rc}")
    if bad_runs:
        print("\nCheck the corresponding logs/<tag>.log for tracebacks.")


if __name__ == '__main__':
    main()

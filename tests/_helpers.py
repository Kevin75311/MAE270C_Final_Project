"""
tests/_helpers.py — Shared utilities for physics test suite.
"""
import os
import sys
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import cfg as _base_cfg

OUT = os.path.join(_ROOT, 'results', 'tests')
os.makedirs(OUT, exist_ok=True)


def make_cfg(ndim=2, small=False):
    """Return an independent config copy with ndim set.
    Pass small=True for heavy PDE tests to keep 3D runtimes manageable (15³ grid).
    Mesh/SAR/BC tests should use small=False (default) to preserve visual fidelity.
    """
    c = copy.deepcopy(_base_cfg)
    c.domain.ndim = ndim
    if ndim == 3 and small:
        c.domain.Nx = 15
        c.domain.Ny = 15
        c.domain.Nz = 15
    return c


def save_fig(fig, name: str, cfg=None):
    dpi = cfg.viz.dpi if cfg is not None else 100
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {path}")


def run_suite(test_fns, ndims=(2, 3)):
    """Run each test function for each requested dimensionality."""
    passed = failed = 0
    for ndim in ndims:
        for fn in test_fns:
            print(f"\n{'─'*60}")
            print(f"Running {fn.__name__}  (ndim={ndim}) ...")
            try:
                fn(ndim=ndim)
                print(f"  PASSED [OK]")
                passed += 1
            except AssertionError as e:
                print(f"  FAILED [!!]  AssertionError: {e}")
                failed += 1
            except Exception as e:
                import traceback
                print(f"  ERROR  [!!]  {type(e).__name__}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'═'*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Figures: {OUT}/")
    return failed == 0

"""
tests/run_all_tests.py — Master test runner for all physics tests.

Imports and runs each test module for both 2D and 3D simulation modes.
Figures are saved to results/tests/.

Usage:
    python tests/run_all_tests.py          # 2D + 3D
    python tests/run_all_tests.py --2d     # 2D only
    python tests/run_all_tests.py --3d     # 3D only

Pytest:
    pytest tests/ -v                       # all discovered tests
"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')

# ── Import individual test modules ────────────────────────────────────────────
from tests.test_01_mesh               import test_mesh_geometry
from tests.test_02_sar                import test_sar_field as _test_sar_field

def test_sar_point(ndim=2):    _test_sar_field(ndim=ndim, probe_model='point')
def test_sar_line(ndim=2):     _test_sar_field(ndim=ndim, probe_model='line')
def test_sar_dipole(ndim=2):   _test_sar_field(ndim=ndim, probe_model='dipole')
from tests.test_03_discretization     import test_discretization_matrices
from tests.test_04_boundary_conditions import test_boundary_conditions
from tests.test_05_bioheat            import test_bioheat_no_power, test_bioheat_heating
from tests.test_06_arrhenius          import test_arrhenius_rate, test_arrhenius_accumulation
from tests.test_07_coupled            import test_coupled_bioheat_arrhenius
from tests._helpers                   import run_suite, OUT

ALL_TESTS = [
    test_mesh_geometry,
    test_sar_point,
    test_sar_line,
    test_sar_dipole,
    test_discretization_matrices,
    test_boundary_conditions,
    test_bioheat_no_power,
    test_bioheat_heating,
    test_arrhenius_rate,
    test_arrhenius_accumulation,
    test_coupled_bioheat_arrhenius,
]


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--2d' in args:
        ndims = (2,)
    elif '--3d' in args:
        ndims = (3,)
    else:
        ndims = (2, 3)

    print(f"Running {len(ALL_TESTS)} tests × {len(ndims)} dimension(s) = "
          f"{len(ALL_TESTS) * len(ndims)} total")
    print(f"Figures → {OUT}/\n")

    ok = run_suite(ALL_TESTS, ndims=ndims)
    sys.exit(0 if ok else 1)

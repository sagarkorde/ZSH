"""Run the v2 study in order. Usage: python RUN_ALL.py [first_step] [last_step]

Steps: e00a e00b e00c e00d e01 e02 e03 e04 e05 e06 e07 e08 e09 e10 figures
Set ZSH_SMOKE=1 for a quick run on DEV data only (no evaluation data touched).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("e00a", "experiments/e00a_build_base.py"),
    ("e00b", "experiments/e00b_audit.py"),
    ("e00c", "experiments/e00c_fetch_d5.py"),
    ("e00d", "experiments/e00d_collect_future.py"),
    ("e01", "experiments/e01_primary.py"),
    ("e02", "experiments/e02_methods.py"),
    ("e03", "experiments/e03_factorial.py"),
    ("e04", "experiments/e04_stability.py"),
    ("e05", "experiments/e05_transfer.py"),
    ("e05b", "experiments/e05b_weight_drift.py"),
    ("e06", "experiments/e06_concentration.py"),
    ("e07", "experiments/e07_heuristic_validity.py"),
    ("e08", "experiments/e08_elliptic.py"),
    ("e09", "experiments/e09_atypicality.py"),
    ("e10", "experiments/e10_sensitivity.py"),
    ("figures", "experiments/make_tables_figures.py"),
]


def main():
    names = [s for s, _ in STEPS]
    first = names.index(sys.argv[1]) if len(sys.argv) > 1 else 0
    last = names.index(sys.argv[2]) if len(sys.argv) > 2 else len(STEPS) - 1
    for name, script in STEPS[first:last + 1]:
        t = time.time()
        print(f"=== {name}: {script}", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        print(f"=== {name}: exit {r.returncode} after {time.time() - t:.0f}s", flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()

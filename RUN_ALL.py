"""Run the v2 study in order. Usage: python RUN_ALL.py [first_step] [last_step]

Steps, in order:
    e00a e00b e00c e00d            data
    e01 e02 e03 e04 e05 e05b e06 e07 e08 e09 e10        pre-specified analyses
    e11 e12 e13 e14 e15 e16 e17 e18 e19 e20 e21         analyses added after the freeze
    figures tables verify                                manuscript outputs and checks

`python RUN_ALL.py` with no arguments runs every reported analysis and then
regenerates every table and figure in the article and the supplementary file.
`figures` writes results/figures and results/tables; `tables` writes the
numbered manuscript tables; `verify` re-checks every numerical claim in the
manuscript against the saved result files and fails if any disagree.

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
    # added after the analysis freeze; see DEVIATIONS.md
    ("e11", "experiments/e11_proxy_k.py"),
    ("e12", "experiments/e12_profile_support.py"),
    ("e13", "experiments/e13_actor_holdout.py"),
    ("e14", "experiments/e14_coinjoin_source.py"),
    ("e15", "experiments/e15_oracle_weights.py"),
    ("e16", "experiments/e16_profile_matching.py"),
    ("e17", "experiments/e17_benchmark.py"),
    ("e18", "experiments/e18_feature_ceiling.py"),
    ("e19", "experiments/e19_richer_features.py"),
    ("e20", "experiments/e20_warm_refit.py"),
    ("e21", "experiments/e21_shared_bound.py"),
    ("figures", "experiments/make_tables_figures.py"),
    ("tables", "experiments/make_manuscript_tables.py"),
    ("verify", "experiments/verify_manuscript_numbers.py"),
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

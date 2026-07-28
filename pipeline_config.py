# ============================================================
# pipeline_config.py — Shared path/mode configuration
#
# Every step script previously hardcoded paths from a different
# machine (C:\Users\sagar\Desktop\Q2 Paper 22326\...). This module
# centralizes those paths and adds a BALANCE_MODE switch so the
# same scripts can produce a "balanced" (stratified-upsampled) run
# and a "raw" (unbalanced) run into separate output trees, driven
# by an environment variable so RUN_PIPELINE.py's existing
# subprocess.run([...]) per-step invocation (which inherits the
# parent process's environment) can select the mode without any
# other orchestration changes.
#
# Env vars (all optional, sensible defaults below):
#   ZSH_DATA_ROOT     - folder containing Dataset.parquet
#   ZSH_OUTPUT_ROOT    - parent folder for outputs_<mode>/ trees
#   ZSH_BALANCE_MODE   - "balanced" (default) | "raw"
#   ZSH_WEIGHT_VARIANT - which zeta weight column Step 3 applies
#                         as the *primary* X_weighted.npy (default s15)
#   ZSH_PROXY_K        - MiniBatchKMeans proxy-cluster count used by
#                         Step 3's MI ranking (default 10)
#   ZSH_RUN_LAMBDA_GRID - "1" to enable Step 5's SEED_WARD_BLEND_ALPHA
#                          grid search (default off; grid is slow)
# ============================================================

import os

DATA_ROOT = os.environ.get("ZSH_DATA_ROOT", r"E:\ZSH_V2\datasets\custom")
OUTPUT_ROOT = os.environ.get("ZSH_OUTPUT_ROOT", r"D:\ZSH_V2_outputs")
ELLIPTIC_ROOT = os.environ.get("ZSH_ELLIPTIC_ROOT", r"E:\ZSH_V2\datasets\elliptic")

BALANCE_MODE = os.environ.get("ZSH_BALANCE_MODE", "balanced")
if BALANCE_MODE not in ("balanced", "raw"):
    raise ValueError(f"ZSH_BALANCE_MODE must be 'balanced' or 'raw', got {BALANCE_MODE!r}")

WEIGHT_VARIANT = os.environ.get("ZSH_WEIGHT_VARIANT", "s15")
PROXY_K = int(os.environ.get("ZSH_PROXY_K", "10"))
RUN_LAMBDA_GRID = os.environ.get("ZSH_RUN_LAMBDA_GRID", "0") == "1"
RUN_S_GRID = os.environ.get("ZSH_RUN_S_GRID", "0") == "1"
RUN_PROXY_SENSITIVITY = os.environ.get("ZSH_RUN_PROXY_SENSITIVITY", "0") == "1"

PARQUET_PATH = os.path.join(DATA_ROOT, "Dataset.parquet")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"outputs_{BALANCE_MODE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ELLIPTIC_FEATURES_CSV = os.path.join(ELLIPTIC_ROOT, "elliptic_txs_features.csv")
ELLIPTIC_CLASSES_CSV = os.path.join(ELLIPTIC_ROOT, "elliptic_txs_classes.csv")
ELLIPTIC_EDGELIST_CSV = os.path.join(ELLIPTIC_ROOT, "elliptic_txs_edgelist.csv")
ELLIPTIC_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "outputs_elliptic")
os.makedirs(ELLIPTIC_OUTPUT_DIR, exist_ok=True)

# Python executable used by RUN_PIPELINE.py to launch each step subprocess.
PIPELINE_PYTHON = os.environ.get(
    "ZSH_PYTHON", r"C:\ProgramData\anaconda3\envs\gpu-env\python.exe"
)
# Directory containing the step scripts themselves (cwd for subprocess.run).
PIPELINE_WORK_DIR = os.environ.get(
    "ZSH_WORK_DIR", r"E:\ZSH_V2\old scripts\1"
)

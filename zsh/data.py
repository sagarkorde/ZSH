"""Base table: canonical features (satoshi units) and annotations.

The same column definitions are produced from the published corpus (D1) and
from Esplora API transactions (prospective sample), so that one feature code
path serves both.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import CFG

SAT = 1e8

# ---------------------------------------------------------------------------
# Annotation codes
# ---------------------------------------------------------------------------
L1_NAMES = ["Coinbase", "ManyInManyOut", "SingleInFanOut", "FanIn", "FanOut",
            "OneInOneOut", "OpReturn", "RBF", "Unlabelled"]
# v1 names of the same count/flag rules, kept for traceability
L1_V1_NAMES = ["Coinbase", "Coinjoin_Mixer", "Batch_Payment", "Consolidation",
               "Distribution", "Standard_P2P", "OP_Return", "RBF_Enabled", "Unknown"]
L2_NAMES = ["coinbase", "P2PKH", "P2SH", "P2WPKH", "P2WSH", "P2TR", "mixed", "other"]
L3_NAMES = ["none", "runes", "omni", "other_opreturn"]
TAG_BITS = {"exchange": 1, "miner": 2, "coinjoin": 4, "other": 8, "unknown": 16}

_SCRIPT_MAP = {
    # D1 (Blockchair-style) names
    "pubkeyhash": 1, "scripthash": 2, "witness_v0_keyhash": 3,
    "witness_v0_scripthash": 4, "witness_v1_taproot": 5,
    # Esplora names
    "p2pkh": 1, "p2sh": 2, "v0_p2wpkh": 3, "v0_p2wsh": 4, "v1_p2tr": 5,
}

FEATURE_CANDIDATES = CFG["features"]["priority"]
AUDIT_ONLY = CFG["features"]["extra_audit_only"]


def l1_codes(n_in, n_out, coinbase, op_return, rbf):
    """First-match assignment of the v1 count/flag rules."""
    n_in = np.asarray(n_in)
    n_out = np.asarray(n_out)
    rules = [
        np.asarray(coinbase, bool),
        (n_in > 3) & (n_out > 3),
        (n_in == 1) & (n_out > 5),
        (n_in > n_out) & (n_in > 2),
        (n_out > n_in) & (n_out > 2),
        (n_in == 1) & (n_out == 1),
        np.asarray(op_return, bool),
        np.asarray(rbf, bool),
    ]
    code = np.full(len(n_in), 8, dtype=np.int8)
    free = np.ones(len(n_in), dtype=bool)
    for i, m in enumerate(rules):
        hit = m & free
        code[hit] = i
        free &= ~hit
    return code


def l1_rule_masks(n_in, n_out):
    """The five count rules as independent booleans (not first-match)."""
    n_in = np.asarray(n_in)
    n_out = np.asarray(n_out)
    return {
        "ManyInManyOut": (n_in > 3) & (n_out > 3),
        "SingleInFanOut": (n_in == 1) & (n_out > 5),
        "FanIn": (n_in > n_out) & (n_in > 2),
        "FanOut": (n_out > n_in) & (n_out > 2),
        "OneInOneOut": (n_in == 1) & (n_out == 1),
    }


def l2_code(types, coinbase):
    """Input script class from a list of per-input script type names."""
    if coinbase:
        return 0
    codes = {_SCRIPT_MAP.get(t, 7) for t in types if t}
    if not codes:
        return 7
    if len(codes) == 1:
        return codes.pop()
    return 6


def _first_push(script_hex):
    """Return the first data push after OP_RETURN (hex) and the opcode after 6a."""
    s = script_hex.lower()
    if not s.startswith("6a") or len(s) < 4:
        return "", ""
    op = int(s[2:4], 16)
    if 1 <= op <= 75:
        return s[4:4 + 2 * op], s[2:4]
    if op == 0x4c and len(s) >= 6:
        n = int(s[4:6], 16)
        return s[6:6 + 2 * n], s[2:4]
    if op == 0x4d and len(s) >= 8:
        n = int(s[6:8] + s[4:6], 16)
        return s[8:8 + 2 * n], s[2:4]
    return "", s[2:4]


def l3_code(has_op_return, script_hex):
    if not has_op_return:
        return 0
    s = (script_hex or "").lower()
    if s.startswith("6a5d"):
        return 1
    data, _ = _first_push(s)
    if data.startswith("6f6d6e69"):
        return 2
    return 3


def load_tagmap():
    with open(CFG["paths"]["tagmap_json"], encoding="utf-8") as fh:
        return json.load(fh)


def tag_bits(addresses, tagmap):
    bits = 0
    for a in addresses:
        c = tagmap.get(a)
        if c is None:
            continue
        bits |= TAG_BITS.get(c, TAG_BITS["other"])
    return bits


def _split_list(values):
    out = []
    if values is None:
        return out
    for v in values:
        if v is None:
            continue
        for part in str(v).replace(",", ";").split(";"):
            part = part.strip()
            if part:
                out.append(part)
    return out


# ---------------------------------------------------------------------------
# D1 -> base table
# ---------------------------------------------------------------------------
D1_COLUMNS = [
    "txid", "block_height", "block_time", "size", "vsize", "weight",
    "input_count", "output_count", "total_input_value", "total_output_value", "fee",
    "input_output_ratio", "value_difference", "avg_input_value", "avg_output_value",
    "input_address_count", "output_address_count", "total_addresses", "input_script_count",
    "has_coinbase", "has_op_return", "op_return_data", "rbf_enabled",
    "value_concentration_ratio", "input_script_types", "input_addresses", "output_addresses",
    "is_consolidation", "is_distribution", "is_peer_to_peer", "is_batch_payment", "is_coinjoin_like",
]


def split_code(block_time):
    s = CFG["splits"]
    t = pd.to_datetime(block_time, unit="s", utc=True)
    dev0 = pd.Timestamp(s["dev_start"])
    test0 = pd.Timestamp(s["test_start"])
    test1 = pd.Timestamp(s["test_end"])
    code = np.full(len(t), 9, dtype=np.int8)
    code[(t >= dev0) & (t < test0)] = 0
    code[(t >= test0) & (t < test1)] = 1
    return code


def base_from_d1_batch(df, tagmap):
    n_in = df["input_count"].to_numpy()
    n_out = df["output_count"].to_numpy()
    tin = df["total_input_value"].to_numpy(np.float64) * SAT
    tout = df["total_output_value"].to_numpy(np.float64) * SAT
    fee = df["fee"].to_numpy(np.float64) * SAT
    vsize = df["vsize"].to_numpy(np.float64)
    size = df["size"].to_numpy(np.float64)
    coinbase = df["has_coinbase"].to_numpy(bool)
    opret = df["has_op_return"].to_numpy(bool)
    rbf = df["rbf_enabled"].to_numpy(bool)

    out = pd.DataFrame({
        "txid": df["txid"].to_numpy(),
        "block_height": df["block_height"].to_numpy(np.int32),
        "block_time": df["block_time"].to_numpy(np.int64),
        "input_count": n_in.astype(np.int32),
        "output_count": n_out.astype(np.int32),
        "vsize": vsize,
        "weight": df["weight"].to_numpy(np.float64),
        "size": size,
        "total_input_value": tin,
        "total_output_value": tout,
        "fee": fee,
        "value_difference": df["value_difference"].to_numpy(np.float64) * SAT,
        "avg_input_value": df["avg_input_value"].to_numpy(np.float64) * SAT,
        "avg_output_value": df["avg_output_value"].to_numpy(np.float64) * SAT,
        "input_output_ratio": df["input_output_ratio"].to_numpy(np.float64),
        "fee_rate_sat_per_vbyte": np.where(vsize > 0, fee / np.maximum(vsize, 1), 0.0),
        "fee_rate_sat_per_byte": np.where(size > 0, fee / np.maximum(size, 1), 0.0),
        "input_address_count": df["input_address_count"].to_numpy(np.float64),
        "output_address_count": df["output_address_count"].to_numpy(np.float64),
        "total_addresses": df["total_addresses"].to_numpy(np.float64),
        "input_script_count": df["input_script_count"].to_numpy(np.float64),
        "has_coinbase": coinbase.astype(np.float64),
        "has_op_return": opret.astype(np.float64),
        "rbf_enabled": rbf.astype(np.float64),
        "value_concentration_ratio": df["value_concentration_ratio"].to_numpy(np.float64),
    })
    out["split"] = split_code(out["block_time"].to_numpy())
    out["L1"] = l1_codes(n_in, n_out, coinbase, opret, rbf)

    # consistency of stored flags with the count rules
    masks = l1_rule_masks(n_in, n_out)
    stored = {"ManyInManyOut": "is_coinjoin_like", "SingleInFanOut": "is_batch_payment",
              "FanIn": "is_consolidation", "FanOut": "is_distribution",
              "OneInOneOut": "is_peer_to_peer"}
    mism = {k: int((masks[k] != df[v].to_numpy(bool)).sum()) for k, v in stored.items()}

    types = df["input_script_types"].to_numpy()
    ins = df["input_addresses"].to_numpy()
    outs = df["output_addresses"].to_numpy()
    ops = df["op_return_data"].to_numpy()
    l2 = np.empty(len(df), dtype=np.int8)
    l3 = np.empty(len(df), dtype=np.int8)
    tb = np.empty(len(df), dtype=np.int16)
    for i in range(len(df)):
        l2[i] = l2_code(_split_list(types[i]), coinbase[i])
        l3[i] = l3_code(opret[i], ops[i] if isinstance(ops[i], str) else "")
        tb[i] = tag_bits(_split_list(ins[i]) + _split_list(outs[i]), tagmap)
    out["L2"] = l2
    out["L3"] = l3
    out["tags"] = tb
    out["opret_prefix"] = [o[:8] if isinstance(o, str) else "" for o in ops]
    return out, mism


def build_base(log):
    """Build the canonical base table from D1 (one pass, row order preserved)."""
    tagmap = load_tagmap()
    log(f"tag map: {len(tagmap):,} addresses")
    pf = pq.ParquetFile(CFG["paths"]["d1_parquet"])
    parts = []
    mism_total = Counter()
    row0 = 0
    for g in range(pf.num_row_groups):
        df = pf.read_row_group(g, columns=D1_COLUMNS).to_pandas()
        b, mism = base_from_d1_batch(df, tagmap)
        b.insert(0, "row_id", np.arange(row0, row0 + len(b), dtype=np.int64))
        row0 += len(b)
        mism_total.update(mism)
        parts.append(b)
        log(f"  row group {g + 1}/{pf.num_row_groups}: {row0:,} rows")
    base = pd.concat(parts, ignore_index=True)
    return base, dict(mism_total)


# ---------------------------------------------------------------------------
# Esplora API transaction -> base record
# ---------------------------------------------------------------------------
def eocj(values, n_in, min_value, min_equal):
    """Equal-output CoinJoin heuristic on output values (satoshis)."""
    vals = [v for v in values if v >= min_value]
    if not vals:
        return 0, 0, False
    v, k = Counter(vals).most_common(1)[0]
    return int(v), int(k), bool(k >= min_equal and n_in >= k)


def base_from_esplora(tx, tagmap=None):
    vin = tx["vin"]
    vout = tx["vout"]
    coinbase = bool(vin and vin[0].get("is_coinbase"))
    n_in = len(vin)
    n_out = len(vout)
    in_vals = [0 if coinbase else int((v.get("prevout") or {}).get("value", 0)) for v in vin]
    out_vals = [int(o.get("value", 0)) for o in vout]
    tin = float(sum(in_vals))
    tout = float(sum(out_vals))
    fee = float(tx.get("fee", 0) or 0)
    weight = float(tx["weight"])
    size = float(tx["size"])
    vsize = float(np.ceil(weight / 4.0))
    op_scripts = [o.get("scriptpubkey", "") for o in vout if o.get("scriptpubkey_type") == "op_return"]
    opret = bool(op_scripts)
    rbf = any(int(v.get("sequence", 0xFFFFFFFF)) < 0xFFFFFFFE for v in vin)
    in_types = [] if coinbase else [(v.get("prevout") or {}).get("scriptpubkey_type", "") for v in vin]
    in_addrs = [] if coinbase else [(v.get("prevout") or {}).get("scriptpubkey_address") for v in vin]
    out_addrs = [o.get("scriptpubkey_address") for o in vout]
    in_addrs = [a for a in in_addrs if a]
    out_addrs = [a for a in out_addrs if a]
    # v1 convention: in/out, with 0/0 and x/0 set to 1.0
    ratio = tin / tout if tout > 0 else 1.0
    v_star, k_star, is_cj = eocj(out_vals, n_in, CFG["annotations"]["eocj_min_value_sat"],
                                 CFG["annotations"]["eocj_min_equal"])
    status = tx.get("status", {})
    rec = {
        "txid": tx["txid"],
        "block_height": int(status.get("block_height", -1)),
        "block_time": int(status.get("block_time", 0)),
        "input_count": n_in,
        "output_count": n_out,
        "vsize": vsize,
        "weight": weight,
        "size": size,
        "total_input_value": tin,
        "total_output_value": tout,
        "fee": fee,
        "value_difference": tin - tout,
        "avg_input_value": tin / n_in if n_in else 0.0,
        "avg_output_value": tout / n_out if n_out else 0.0,
        "input_output_ratio": ratio,
        "fee_rate_sat_per_vbyte": fee / vsize if vsize else 0.0,
        "fee_rate_sat_per_byte": fee / size if size else 0.0,
        # v1 quirk: these were lengths of a list holding one joined string
        "input_address_count": 1.0 if in_addrs else 0.0,
        "output_address_count": 1.0 if out_addrs else 0.0,
        "total_addresses": float(bool(in_addrs)) + float(bool(out_addrs)),
        "input_script_count": 1.0 if in_types else 0.0,
        "has_coinbase": float(coinbase),
        "has_op_return": float(opret),
        "rbf_enabled": float(rbf),
        "value_concentration_ratio": 1.0 if tout > 0 else 0.0,
        "L1": int(l1_codes([n_in], [n_out], [coinbase], [opret], [rbf])[0]),
        "L2": l2_code(in_types, coinbase),
        "L3": l3_code(opret, op_scripts[0] if op_scripts else ""),
        "tags": tag_bits(in_addrs + out_addrs, tagmap) if tagmap is not None else 0,
        "opret_prefix": op_scripts[0][:8] if op_scripts else "",
        "eocj": int(is_cj),
        "eocj_value": v_star,
        "eocj_k": k_star,
        "input_types": ";".join(in_types),
    }
    return rec

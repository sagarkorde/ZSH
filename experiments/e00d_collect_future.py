"""E0d: collect the prospective sample D4 (Oct 2024 - Aug 2026), ANALYSIS_PLAN §2.2.

Month m covers heights (h_m, h_{m+1}], where h_m is the last block mined before the
month starts (mempool.space /api/v1/mining/blocks/timestamp/<unix time>).
Per month: 800 heights drawn without replacement; per height one page of up to 25
transactions with a uniformly drawn start index. Run only after tag `v2-frozen`.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zsh.collect import Esplora  # noqa: E402
from zsh.config import CFG, Log, out_dir, results_dir, rng_for, sha256_file, write_json  # noqa: E402
from zsh.data import base_from_esplora, load_tagmap, split_code  # noqa: E402


def month_starts(first, last):
    months = pd.period_range(first, last, freq="M")
    starts = [m.to_timestamp().tz_localize("UTC") for m in months]
    starts.append((months[-1] + 1).to_timestamp().tz_localize("UTC"))
    return [str(m) for m in months], [int(s.timestamp()) for s in starts]


def main():
    log = Log(out_dir("logs") / "e00d_collect_future.log")
    fc = CFG["future"]
    ddir = out_dir("future")
    api = Esplora(ddir / "d4_cache.sqlite", log)
    months, starts = month_starts(fc["first_month"], fc["last_month"])

    # 1. month boundaries (mempool.space only)
    bounds = []
    for ts in starts:
        status, text = api.get_one(f"/v1/mining/blocks/timestamp/{ts}", host_index=0)
        bounds.append(json.loads(text)["height"])
    manifest = {"months": []}
    plan = []
    for i, m in enumerate(months):
        lo, hi = bounds[i] + 1, bounds[i + 1]
        n_blocks = hi - lo + 1
        rng = rng_for("FUT", "heights", m)
        k = min(fc["blocks_per_month"], n_blocks)
        heights = np.sort(lo + rng.choice(n_blocks, size=k, replace=False))
        manifest["months"].append({"month": m, "first_height": lo, "last_height": hi,
                                   "blocks_in_month": int(n_blocks), "blocks_sampled": int(k)})
        plan += [(m, int(h)) for h in heights]
    log(f"{len(plan):,} blocks planned over {len(months)} months")

    # 2-3. hashes and headers of the sampled heights. mempool.space
    # /v1/blocks/<h> returns the 15 blocks h, h-1, ..., h-14 (hash, tx_count,
    # timestamp), so one request covers up to 15 sampled heights
    # (deviation logged in DEVIATIONS.md; same blocks and fields as /block/<hash>).
    need = sorted({h for _, h in plan}, reverse=True)
    tops, covered = [], set()
    for h in need:
        if h not in covered:
            tops.append(h)
            covered.update(range(h - 14, h + 1))
    api.get_many([f"/v1/blocks/{h}" for h in tops], hosts=[api.hosts[0]])
    headers = {}
    for h in tops:
        for b in api.json(f"/v1/blocks/{h}") or []:
            headers[int(b["height"])] = b
    missing = [h for h in need if h not in headers]
    if missing:
        log(f"  {len(missing)} heights missing from bulk headers; fetching singly")
        api.get_many([f"/block-height/{h}" for h in missing])
        for h in missing:
            hh = api.text(f"/block-height/{h}").strip()
            api.get_many([f"/block/{hh}"])
            headers[h] = api.json(f"/block/{hh}")
    hashes = {h: headers[h]["id"] for _, h in plan}
    info = {h: headers[h] for _, h in plan}
    log(f"  headers for {len(info):,} blocks from {len(tops):,} bulk requests")
    # 4. one page per block
    pages = {}
    for m, h in plan:
        n_tx = int(info[h]["tx_count"])
        n_pages = math.ceil(n_tx / fc["page_size"])
        r = rng_for("FUT", "page", h)
        pages[h] = (int(r.integers(n_pages)) * fc["page_size"], n_pages)
    api.get_many([f"/block/{hashes[h]}/txs/{pages[h][0]}" for _, h in plan])

    # 5. parse
    tagmap = load_tagmap()
    per_month = {x["month"]: x for x in manifest["months"]}
    recs = []
    for m, h in plan:
        start, n_pages = pages[h]
        txs = api.json(f"/block/{hashes[h]}/txs/{start}")
        if txs is None:
            log(f"  missing page for height {h}")
            continue
        mm = per_month[m]
        w = n_pages * mm["blocks_in_month"] / mm["blocks_sampled"]
        for j, tx in enumerate(txs):
            r = base_from_esplora(tx, tagmap)
            r.update({"month": m, "page_start": start, "page_index": j, "pages_in_block": n_pages,
                      "block_tx_count": int(info[h]["tx_count"]), "design_weight": w})
            recs.append(r)
    df = pd.DataFrame(recs)
    df["split"] = split_code(df.block_time.to_numpy())
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    path = ddir / "d4_future.parquet"
    df.to_parquet(path, index=False)
    manifest.update({"rows": len(df), "blocks_with_page": int(df.block_height.nunique()),
                     "month_boundary_heights": dict(zip(months + ["end"], bounds)),
                     "d4_sha256": sha256_file(path), "cache_sha256": sha256_file(ddir / "d4_cache.sqlite"),
                     "hosts": api.hosts})
    write_json(manifest, results_dir("E0") / "future_manifest.json")
    log(f"wrote {path}: {len(df):,} transactions from {df.block_height.nunique():,} blocks")


if __name__ == "__main__":
    main()

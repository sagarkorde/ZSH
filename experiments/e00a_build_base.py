"""E0a: build the canonical base table from D1 (features in satoshi units + annotations)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zsh.config import CFG, Log, out_dir, results_dir, sha256_file, write_json  # noqa: E402
from zsh.data import build_base  # noqa: E402


def main():
    log = Log(out_dir("logs") / "e00a_build_base.log")
    d1 = CFG["paths"]["d1_parquet"]
    digest = sha256_file(d1)
    log(f"D1 sha256 {digest}")
    if digest != CFG["checksums"]["d1_parquet"]:
        raise SystemExit("D1 checksum mismatch")
    base, mism = build_base(log)
    path = out_dir("base") / "base.parquet"
    base.to_parquet(path, index=False)
    log(f"wrote {path} {base.shape}")
    write_json({"rows": len(base), "flag_rule_mismatches": mism,
                "split_counts": base["split"].value_counts().sort_index().to_dict(),
                "base_sha256": sha256_file(path)},
               results_dir("E0") / "base_build.json")
    log(f"stored-flag vs count-rule mismatches: {mism}")


if __name__ == "__main__":
    main()

"""Validate user-supplied HAR-CEV input data without modifying it."""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd


def validate_raw(path: Path) -> list[str]:
    errors = []
    try:
        df = pd.read_csv(path, nrows=10)
    except Exception as exc:
        return [f"cannot read CSV ({exc})"]
    lower = {str(c).lower() for c in df.columns}
    for required in ("close", "volume"):
        if required not in lower:
            errors.append(f"missing '{required}' column")
    if len(df.columns) < 3:
        errors.append("expected timestamp, close, and volume columns")
    return errors


def validate_rv(path: Path) -> list[str]:
    errors = []
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [f"cannot read CSV ({exc})"]
    if "rv" not in df.columns:
        return ["missing 'rv' column"]
    rv = pd.to_numeric(df["rv"], errors="coerce")
    if rv.isna().any() or not np.isfinite(rv).all():
        errors.append("'rv' contains missing or non-finite values")
    if (rv <= 0).any():
        errors.append("'rv' must be strictly positive")
    if len(df) < 23:
        errors.append("fewer than 23 rows (insufficient for HAR lags)")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset_dija"))
    parser.add_argument("--rv-dir", type=Path, default=Path("dataset_rv"))
    args = parser.parse_args()

    raw_files = sorted(args.raw_dir.glob("dataset_*.csv"))
    rv_files = sorted(args.rv_dir.glob("rv_*.csv"))
    if not raw_files and not rv_files:
        print(
            "No data found. Add dataset_*.csv to dataset_dija/ and run "
            "compute_rv.py, or add rv_*.csv directly to dataset_rv/.",
            file=sys.stderr,
        )
        return 2

    failures = 0
    for kind, files, validator in (
        ("raw", raw_files, validate_raw),
        ("rv", rv_files, validate_rv),
    ):
        for path in files:
            errors = validator(path)
            if errors:
                failures += 1
                print(f"[FAIL] {kind}: {path}: {'; '.join(errors)}")
            else:
                print(f"[ OK ] {kind}: {path}")

    if failures:
        print(f"Validation failed for {failures} file(s).", file=sys.stderr)
        return 1
    print(f"Validation passed ({len(raw_files)} raw, {len(rv_files)} RV files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

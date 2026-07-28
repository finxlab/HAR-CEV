"""Compute daily annualized realized variance from 5-minute bars.

  RV_t = 252 * sum_i r_{t,i}^2   (intraday log returns; overnight excluded
                                  because returns are grouped within days)

Bar cleaning (one-bar artifacts only; economically meaningful jumps kept):
  * one-bar reversal pattern (>=15% move immediately reversed)
  * >=15% move on near-zero volume
  * >=15% move whose ratio matches a standard split factor
  * any |log return| >= 30%

Input : {data_dir}/dataset_{TICKER}.csv  (datetime index, close, volume)
Output: {out_dir}/rv_{TICKER}.csv        (date index, column 'rv')

Usage:
  python compute_rv.py --data_dir dataset_dija --out_dir dataset_rv
"""

import argparse
import os

import numpy as np
import pandas as pd

from protocol import ALL_TICKERS

DAYS_PER_YEAR = 252


def _is_split_like(ratio, tol=0.04):
    split_ratios = np.array([1 / 4, 1 / 3, 1 / 2, 2 / 3, 1.5, 2.0, 3.0, 4.0])
    vals = ratio.to_numpy(dtype=float)
    ok = np.zeros_like(vals, dtype=bool)
    for sr in split_ratios:
        ok |= np.isfinite(vals) & (np.abs(vals / sr - 1.0) <= tol)
    return pd.Series(ok, index=ratio.index)


def clean_close(close, volume):
    """Mask obvious one-bar artifacts; keep economically meaningful jumps."""
    close = close.astype(float).where(close.astype(float) > 0)
    log_ret = np.log(close).diff()
    next_log_ret = log_ret.shift(-1)
    ratio = close / close.shift(1)

    reversal = (
        (log_ret.abs() >= 0.15)
        & (next_log_ret.abs() >= 0.15)
        & (np.sign(log_ret) == -np.sign(next_log_ret))
        & ((log_ret.abs() - next_log_ret.abs()).abs()
           <= 0.35 * log_ret.abs().clip(lower=1e-12))
    )
    low_volume_jump = (log_ret.abs() >= 0.15) & (volume.fillna(np.inf) <= 100)
    split_like_jump = (log_ret.abs() >= 0.15) & _is_split_like(ratio)
    extreme_jump = log_ret.abs() >= 0.30

    bad = (reversal | low_volume_jump | split_like_jump
           | extreme_jump).fillna(False)
    cleaned = close.copy()
    cleaned.loc[bad[bad].index] = np.nan
    return cleaned, int(bad.sum())


def compute_daily_rv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for col in ["close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close, n_bad = clean_close(df["close"], df["volume"])
    trade_date = df.index.normalize()
    log_ret = np.log(close).groupby(trade_date).diff()
    rv = (log_ret ** 2).groupby(trade_date).sum() * DAYS_PER_YEAR
    rv = rv.dropna()
    rv = rv[rv > 0]
    rv.name = "rv"
    return rv, n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset_dija")
    ap.add_argument("--out_dir", default="dataset_rv")
    ap.add_argument("--tickers", default="all")
    args = ap.parse_args()

    tickers = (ALL_TICKERS if args.tickers == "all"
               else args.tickers.split(","))
    os.makedirs(args.out_dir, exist_ok=True)
    for tk in tickers:
        path = os.path.join(args.data_dir, f"dataset_{tk}.csv")
        if not os.path.exists(path):
            print(f"[{tk}] missing {path}, skipping")
            continue
        rv, n_bad = compute_daily_rv(path)
        rv.to_csv(os.path.join(args.out_dir, f"rv_{tk}.csv"))
        print(f"[{tk}] {len(rv)} days "
              f"({rv.index.min().date()} ~ {rv.index.max().date()}), "
              f"{n_bad} bad bars masked")


if __name__ == "__main__":
    main()

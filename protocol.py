"""Shared zero-shot evaluation protocol: universe, regimes, data helpers.

Protocol: for each (ticker, target regime) pair, train on the preceding
756 trading days of RV, then forecast every day of the target regime with
no adaptation. Pairs with fewer than 756 pre-target days are skipped.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ALL_TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]

# Evaluation regimes ("Bear I" serves as warm-up only and is never a target).
REGIMES = {
    "Bear I":        ("2000-01-03", "2003-01-07"),
    "Bull I":        ("2003-01-08", "2007-06-30"),
    "Crisis I":      ("2007-07-01", "2009-03-31"),
    "Recovery I":    ("2009-04-01", "2012-06-30"),
    "Bull II":       ("2012-07-01", "2015-06-30"),
    "Shock I":       ("2015-07-01", "2016-03-31"),
    "Bull III":      ("2016-04-01", "2018-12-31"),
    "Pre-Crisis":    ("2019-01-01", "2020-02-09"),
    "Crisis II":     ("2020-02-10", "2020-06-30"),
    "Recovery II":   ("2020-07-01", "2021-12-31"),
    "Tightening":    ("2022-01-01", "2022-12-31"),
    "Stabilization": ("2023-01-01", "2024-12-31"),
    "Pre-Tariff":    ("2025-01-01", "2025-03-31"),
    "Tariff Shock":  ("2025-04-02", "2025-06-30"),
    "Post-Tariff":   ("2025-07-01", "2025-12-31"),
}
TARGET_REGIMES = list(REGIMES.keys())[1:]

REGIME_TYPE = {
    "Bull I": "Bull", "Crisis I": "Crisis", "Recovery I": "Recovery",
    "Bull II": "Bull", "Shock I": "Shock", "Bull III": "Bull",
    "Pre-Crisis": "Bull", "Crisis II": "Crisis", "Recovery II": "Recovery",
    "Tightening": "Shock", "Stabilization": "Stable", "Pre-Tariff": "Shock",
    "Tariff Shock": "Shock", "Post-Tariff": "Stable",
}


def load_rv(ticker, rv_dir):
    path = os.path.join(rv_dir, f"rv_{ticker}.csv")
    return (pd.read_csv(path, index_col=0, parse_dates=True)["rv"]
            .dropna().sort_index())


def fit_clip_bounds(rv_arr, lower_pct, upper_pct):
    """Percentile clip bounds fit on the training window only.

    Returns None (no clipping) when the range covers [<=0, >=100].
    """
    if lower_pct <= 0 and upper_pct >= 100:
        return None
    lo = float(np.percentile(rv_arr, lower_pct))
    hi = float(np.percentile(rv_arr, upper_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    return lo, hi


def clip_inputs(rv_arr, clip_bounds):
    if clip_bounds is None:
        return rv_arr
    lo, hi = clip_bounds
    return np.clip(rv_arr, lo, hi)


def make_windows(rv_arr, rv_mean, look_back, clip_bounds=None):
    """Sliding windows. Inputs are causally clipped and normalized;
    targets stay on the raw RV scale."""
    seq = clip_inputs(rv_arr, clip_bounds) / rv_mean
    X, y = [], []
    for i in range(look_back, len(rv_arr)):
        X.append(seq[i - look_back:i])
        y.append(rv_arr[i])
    if not X:
        return None, None
    return np.array(X, np.float32), np.array(y, np.float32)


def make_loader(X, y, shuffle, batch):
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                      batch_size=batch, shuffle=shuffle)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def iter_pairs(rv_series, look_back, train_window):
    """Yield (tgt, rv_train, rv_tgt_values, tgt_dates) for valid pairs."""
    for tgt in TARGET_REGIMES:
        tgt_s, tgt_e = REGIMES[tgt]
        rv_before = rv_series[rv_series.index < tgt_s].values.astype(
            np.float32)
        tgt_slice = rv_series[(rv_series.index >= tgt_s)
                              & (rv_series.index <= tgt_e)]
        if len(rv_before) < train_window or len(tgt_slice) < look_back + 3:
            continue
        yield (tgt, rv_before[-train_window:],
               tgt_slice.values.astype(np.float32), tgt_slice.index)


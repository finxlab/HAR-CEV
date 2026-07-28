"""Per-pair distribution shift: Wasserstein-1 distance in log-RV space
between the 756-day training window and the target regime.

Output: outputs/wd_train.csv  (ticker, tgt, wd)

Usage:
  python compute_wd.py --rv_dir dataset_rv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from protocol import ALL_TICKERS, REGIMES, TARGET_REGIMES, load_rv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rv_dir", default="dataset_rv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--train_window", type=int, default=756)
    args = ap.parse_args()

    rows = []
    for tk in ALL_TICKERS:
        try:
            v = load_rv(tk, args.rv_dir)
        except FileNotFoundError:
            continue
        for tgt in TARGET_REGIMES:
            s, e = REGIMES[tgt]
            train = v[v.index < s].values[-args.train_window:]
            target = v[(v.index >= s) & (v.index <= e)].values
            if len(train) < args.train_window or len(target) < 5:
                continue
            rows.append({
                "ticker": tk, "tgt": tgt,
                "wd": wasserstein_distance(np.log(train), np.log(target)),
            })

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "wd_train.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out, f"({len(rows)} pairs)")


if __name__ == "__main__":
    main()


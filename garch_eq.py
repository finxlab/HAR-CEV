"""GARCH(1,1) baseline with empirical-ratio quantiles (GARCH-EQ).

For each (ticker, regime) pair:
  1. Fit GARCH(1,1) by Gaussian MLE (Nelder-Mead) on the daily
     close-to-close log returns of the 756-day training window.
  2. Run one-step recursive conditional variances h_t (fixed parameters)
     over the training window and the target regime.
  3. Training-window ratios r_t = RV_t / (252 * h_t) give an empirical
     variance-multiplier distribution; its quantiles at the 7 tau levels,
     multiplied by 252 * h_{t+1}, are the predictive RV quantiles.
The model therefore faces the same information set and quantile grid as
every other baseline.

Output: outputs/predictions_garch.parquet
    ticker, tgt, model='GARCH', date, rv_actual, q05..q95

Usage:
  python garch_eq.py --data_dir dataset_dija --rv_dir dataset_rv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from protocol import ALL_TICKERS, REGIMES, TARGET_REGIMES, load_rv

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QCOLS = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]


def daily_returns(data_dir, ticker):
    px = pd.read_csv(os.path.join(data_dir, f"dataset_{ticker}.csv"),
                     index_col=0, parse_dates=True).sort_index()
    close = pd.to_numeric(px["close"], errors="coerce").dropna()
    close = close[close > 0]
    daily = close.groupby(close.index.normalize()).last()
    ret = np.log(daily).diff().dropna()
    return ret[ret.abs() < 0.5]


def garch_fit(r):
    """GARCH(1,1) Gaussian MLE via Nelder-Mead. Returns (omega, a, b)."""
    r = r - r.mean()
    var = r.var()

    def nll(th):
        w, a, b = th
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        h = np.empty(len(r))
        h[0] = var
        for t in range(1, len(r)):
            h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
        return 0.5 * np.sum(np.log(h) + r ** 2 / h)

    best = None
    for x0 in ([var * 0.05, 0.05, 0.90], [var * 0.1, 0.1, 0.8]):
        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def garch_path(params, r_centered, h0):
    w, a, b = params
    h = np.empty(len(r_centered))
    h[0] = h0
    for t in range(1, len(r_centered)):
        h[t] = w + a * r_centered[t - 1] ** 2 + b * h[t - 1]
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset_dija")
    ap.add_argument("--rv_dir", default="dataset_rv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--train_window", type=int, default=756)
    args = ap.parse_args()

    rows = []
    for tk in ALL_TICKERS:
        try:
            ret = daily_returns(args.data_dir, tk)
            rv = load_rv(tk, args.rv_dir)
        except Exception as e:
            print(f"[{tk}] data error: {e}")
            continue
        for tgt in TARGET_REGIMES:
            s, e = pd.Timestamp(REGIMES[tgt][0]), pd.Timestamp(REGIMES[tgt][1])
            rtr = ret[ret.index < s][-args.train_window:]
            rev = ret[(ret.index >= s) & (ret.index <= e)]
            # returns lose 1-2 days to differencing; allow a small shortfall
            if len(rtr) < args.train_window - 6 or len(rev) < 30:
                continue

            params = garch_fit(rtr.values)
            mu = rtr.values.mean()
            trc = rtr.values - mu
            h_tr = garch_path(params, trc, trc.var())

            # training-window empirical ratio quantiles
            htr = pd.Series(252 * h_tr, index=rtr.index)
            ratio = (rv.reindex(htr.index) / htr).dropna()
            if len(ratio) < 100:
                continue
            rq = np.quantile(ratio.values, TAUS)

            # one-step recursive conditional variance over the target
            w0, a0, b0 = params
            evc = rev.values - mu
            h_ev = np.empty(len(evc))
            h_ev[0] = w0 + a0 * trc[-1] ** 2 + b0 * h_tr[-1]
            for t in range(1, len(evc)):
                h_ev[t] = w0 + a0 * evc[t - 1] ** 2 + b0 * h_ev[t - 1]

            base = pd.DataFrame(
                252 * h_ev[:, None] * rq[None, :], columns=QCOLS,
                index=rev.index)
            base["rv_actual"] = rv.reindex(rev.index)
            base = base.dropna(subset=["rv_actual"]).reset_index(
                names="date")
            base.insert(0, "model", "GARCH")
            base.insert(0, "tgt", tgt)
            base.insert(0, "ticker", tk)
            rows.append(base)
        print(f"[{tk}] done")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "predictions_garch.parquet")
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out)
    print("saved", out, len(df), "days,",
          df.groupby(['ticker', 'tgt']).ngroups, "pairs")


if __name__ == "__main__":
    main()

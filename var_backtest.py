"""Value-at-Risk backtest of RV forecasts on realized daily returns.

Pipeline (identical rule for every model):
  * daily close-to-close log returns from the 5-minute closes
  * sigma_hat(t+1) = sqrt(c * RVhat(t+1) / 252), where the
    intraday-to-daily scaling c = mean(r^2) / mean(RV/252) is fit on the
    756-day training window only (overnight-inclusive, no look-ahead)
  * VaR_a(t+1) = z_a * sigma_hat(t+1);  violation: r(t+1) < VaR_a
Reference limits:
  * GARCH(1,1) (Gaussian MLE on the same window, fixed-parameter
    one-step recursion) uses its own conditional variance
  * HS-250d: rolling 250-day historical-simulation quantile

Reads model q50 forecasts from outputs/predictions_*.parquet.
Output: outputs/var_backtest.csv + console summary (violation rates and
per-pair Kupiec pass shares, overall and Crisis).

Usage:
  python var_backtest.py --data_dir dataset_dija --rv_dir dataset_rv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from garch_eq import daily_returns, garch_fit, garch_path
from protocol import ALL_TICKERS, REGIMES, TARGET_REGIMES, load_rv

MODELS = ["HARCEV_Euler", "PatchTST", "iTransformer", "NBEATS", "TCN",
          "LSTM", "DLinear", "HAR", "GRULag"]


def kupiec_p(v, a):
    n, x = len(v), int(np.sum(v))
    if x in (0, n):
        return 0.0
    ph = x / n
    lr = -2 * (x * np.log(a / ph) + (n - x) * np.log((1 - a) / (1 - ph)))
    return 1 - chi2.cdf(lr, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset_dija")
    ap.add_argument("--rv_dir", default="dataset_rv")
    ap.add_argument("--pred_dir", default="outputs")
    ap.add_argument("--train_window", type=int, default=756)
    args = ap.parse_args()

    pred = pd.concat(
        [pd.read_parquet(f) for f in
         glob.glob(os.path.join(args.pred_dir, "predictions_*.parquet"))],
        ignore_index=True)
    pred = pred[pred.model.isin(MODELS)]
    pred["date"] = pd.to_datetime(pred["date"])

    rows = []
    for tk in ALL_TICKERS:
        try:
            ret = daily_returns(args.data_dir, tk)
            rv = load_rv(tk, args.rv_dir)
        except Exception as e:
            print(tk, "data error:", e)
            continue
        for tgt in TARGET_REGIMES:
            s, e = pd.Timestamp(REGIMES[tgt][0]), pd.Timestamp(REGIMES[tgt][1])
            rtr = ret[ret.index < s].values[-args.train_window:]
            rvtr = rv[rv.index < s].values[-args.train_window:]
            rev = ret[(ret.index >= s) & (ret.index <= e)]
            if len(rtr) < args.train_window - 6 or len(rvtr) < 700 \
                    or len(rev) < 30:
                continue
            c = (rtr ** 2).mean() / (rvtr.mean() / 252)

            params = garch_fit(rtr)
            mu = rtr.mean()
            trc = rtr - mu
            h_tr = garch_path(params, trc, trc.var())
            w0, a0, b0 = params
            evc = rev.values - mu
            h_ev = np.empty(len(evc))
            h_ev[0] = w0 + a0 * trc[-1] ** 2 + b0 * h_tr[-1]
            for t in range(1, len(evc)):
                h_ev[t] = w0 + a0 * evc[t - 1] ** 2 + b0 * h_ev[t - 1]

            base = pd.DataFrame({"date": rev.index, "ret": rev.values,
                                 "garch_var": h_ev, "c": c})
            # HS-250d rolling historical-simulation limits
            hist = ret[ret.index <= e]
            hs = pd.DataFrame({
                "date": hist.index,
                "hs95": hist.rolling(250).quantile(0.05).shift(1).values,
                "hs99": hist.rolling(250).quantile(0.01).shift(1).values})
            base = base.merge(hs, on="date", how="left")

            sub = pred[(pred.ticker == tk) & (pred.tgt == tgt)]
            piv = sub.pivot_table(index="date", columns="model",
                                  values="q50").reset_index()
            m = base.merge(piv, on="date", how="inner")
            m["ticker"], m["tgt"] = tk, tgt
            rows.append(m)
        print(tk, "done", flush=True)

    df = pd.concat(rows, ignore_index=True)
    out = os.path.join(args.pred_dir, "var_backtest.csv")
    df.to_csv(out, index=False)
    print("saved", out, len(df), "days,",
          df.groupby(["ticker", "tgt"]).ngroups, "pairs")

    crisis = df.tgt.isin(["Crisis I", "Crisis II"])
    z95, z99 = norm.ppf(0.05), norm.ppf(0.01)
    lims = {"HS-250d": (df.hs95, df.hs99),
            "GARCH": (z95 * np.sqrt(df.garch_var),
                      z99 * np.sqrt(df.garch_var))}
    for name in MODELS:
        if name in df.columns:
            sig = np.sqrt(df.c * df[name] / 252)
            lims[name] = (z95 * sig, z99 * sig)

    print(f"\n{'model':13s} {'95 all':>7s} {'95 crisis':>10s} "
          f"{'99 all':>7s} {'99 crisis':>10s} {'Kupiec95 crisis':>16s}")
    cd = df[crisis]
    for name, (l95, l99) in lims.items():
        v95, v99 = df.ret < l95, df.ret < l99
        kp = cd.assign(v=v95[crisis].values).groupby(
            ["ticker", "tgt"]).apply(lambda d: kupiec_p(d.v.values, 0.05))
        print(f"{name:13s} {v95.mean() * 100:6.2f}% "
              f"{v95[crisis].mean() * 100:9.2f}% "
              f"{v99.mean() * 100:6.2f}% "
              f"{v99[crisis].mean() * 100:9.2f}% "
              f"{100 * (kp > 0.05).mean():15.1f}%")


if __name__ == "__main__":
    main()

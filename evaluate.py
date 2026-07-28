"""Compute every evaluation table of the paper from saved predictions.

Reads all outputs/predictions_*.parquet (long format: ticker, tgt, model,
date, rv_actual, q05..q95) plus outputs/wd_train.csv, and prints:

  * overall metrics per model (ND, MAE, WIS, TailAE, NIS90, PS-WIS,
    PS-Bias)                                     -> paper Table 2
  * ND / WIS by WD quartile                     -> paper Table 3
  * ND / WIS by regime type                     -> paper Table 4
  * CalibErr-7 by regime type                    -> paper Table 5
  * ablation metrics (Anti-MR, No-SDE)           -> paper Table 7
  * panel-aware checks: pair/ticker sign tests, 21-day moving-block
    bootstrap CI of the daily mean WIS differential, and the pair-level
    regression of relative WIS improvement on WD + crisis dummy.

Usage:
  python evaluate.py --pred_dir outputs
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from protocol import REGIME_TYPE

TAUS = np.array([.05, .10, .25, .50, .75, .90, .95])
QC = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
MAIN_MODELS = ["HARCEV_Euler", "PatchTST", "iTransformer", "NBEATS", "TCN",
               "LSTM", "DLinear", "HAR", "GARCH"]
DL6 = ["PatchTST", "iTransformer", "NBEATS", "TCN", "LSTM", "DLinear"]
RTYPES = ["Bull", "Crisis", "Recovery", "Shock", "Stable"]
QBS = ["Q1", "Q2", "Q3", "Q4"]


def add_ps_flags(df):
    """Post-spike flag: days t+1..t+5 after RV >= within-pair 95th pct."""
    ps = np.zeros(len(df), bool)
    for _, idx in df.groupby(["ticker", "tgt"]).indices.items():
        idx = np.sort(idx)
        rv = df.rv_actual.values[idx]
        thr = np.quantile(rv, 0.95)
        for pos in np.flatnonzero(rv >= thr):
            ps[idx[pos + 1:min(pos + 6, len(idx))]] = True
    return ps


def metric_pack(df):
    y = df.rv_actual.values
    q = df[QC].values
    e = y[:, None] - q
    pb = np.where(e >= 0, TAUS * e, (TAUS - 1) * e)
    crps = 2 * pb.mean(1)
    ae = np.abs(q[:, 3] - y)
    t = np.quantile(ae, 0.95)
    is90 = (q[:, 6] - q[:, 0]) + 20 * np.clip(q[:, 0] - y, 0, None) \
        + 20 * np.clip(y - q[:, 6], 0, None)
    iqr = np.quantile(y, .75) - np.quantile(y, .25)
    ps = df["ps"].values
    return dict(ND=ae.sum() / np.abs(y).sum(), MAE=ae.mean(),
                WIS=crps.mean(), TailAE=ae[ae >= t].mean(),
                NIS90=is90.mean() / iqr, PSWIS=crps[ps].mean(),
                PSBias=(q[:, 3] - y)[ps].mean())


def calib_err(df):
    y = df.rv_actual.values
    q = df[QC].values
    cov = (y[:, None] <= q).mean(0)
    err = np.abs(cov - TAUS)
    return err.mean(), (err[0] + err[6]) / 2   # CalibErr-7, TailCalib


def wis_col(df):
    y = df.rv_actual.values
    q = df[QC].values
    e = y[:, None] - q
    pb = np.where(e >= 0, TAUS * e, (TAUS - 1) * e)
    return 2 * pb.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="outputs")
    args = ap.parse_args()

    prediction_files = glob.glob(
        os.path.join(args.pred_dir, "predictions_*.parquet"))
    if not prediction_files:
        raise SystemExit(
            f"No predictions_*.parquet files found in {args.pred_dir!r}. "
            "Run the training scripts first."
        )
    pred = pd.concat(
        [pd.read_parquet(f) for f in prediction_files], ignore_index=True)
    pred["date"] = pd.to_datetime(pred["date"])
    wd_path = os.path.join(args.pred_dir, "wd_train.csv")
    if not os.path.exists(wd_path):
        raise SystemExit(
            f"Missing {wd_path!r}. Run compute_wd.py before evaluate.py."
        )
    w = pd.read_csv(wd_path)
    w["Qb"] = pd.qcut(w.wd, 4, labels=QBS)

    frames = {}
    for m in pred.model.unique():
        d = pred[pred.model == m].merge(w, on=["ticker", "tgt"])
        d = d.sort_values(["ticker", "tgt", "date"]).reset_index(drop=True)
        d["rt"] = d.tgt.map(REGIME_TYPE)
        d["ps"] = add_ps_flags(d)
        frames[m] = d

    rows = [m for m in MAIN_MODELS if m in frames]

    print("=== Overall (Table 2) ===")
    for m in rows:
        r = metric_pack(frames[m])
        print(f"{m:13s} " + " ".join(
            f"{r[k]:.4f}" for k in ["ND", "MAE", "WIS", "TailAE", "NIS90",
                                    "PSWIS", "PSBias"]))

    print("\n=== By WD quartile: ND | WIS (Table 3) ===")
    for m in rows:
        d = frames[m]
        nd = [f"{metric_pack(d[d.Qb == q])['ND']:.4f}" for q in QBS]
        cr = [f"{metric_pack(d[d.Qb == q])['WIS']:.4f}" for q in QBS]
        print(f"{m:13s} " + " ".join(nd) + " | " + " ".join(cr))

    print("\n=== By regime type: ND | WIS (Table 4) ===")
    for m in rows:
        d = frames[m]
        nd = [f"{metric_pack(d[d.rt == t])['ND']:.4f}" for t in RTYPES]
        cr = [f"{metric_pack(d[d.rt == t])['WIS']:.4f}" for t in RTYPES]
        print(f"{m:13s} " + " ".join(nd) + " | " + " ".join(cr))

    print("\n=== CalibErr-7 by regime type (Table 5) ===")
    for m in rows:
        d = frames[m]
        print(f"{m:13s} " + " ".join(
            f"{calib_err(d[d.rt == t])[0]:.3f}" for t in RTYPES))

    print("\n=== Ablations (Table 7) ===")
    for m in ["HARCEV_Euler", "HARCEV_FixedA", "AntiMR", "GRULag"]:
        if m not in frames:
            continue
        r = metric_pack(frames[m])
        ce, tc = calib_err(frames[m])
        print(f"{m:13s} ND={r['ND']:.4f} WIS={r['WIS']:.4f} "
              f"CalibErr={ce:.4f} TailCalib={tc:.4f}")

    # ?? panel-aware inference ????????????????????????????????????????????
    print("\n=== Panel-aware checks ===")
    if "HARCEV_Euler" not in frames:
        raise SystemExit("HARCEV_Euler predictions are required for inference.")
    ours = frames["HARCEV_Euler"].assign(crps=wis_col(
        frames["HARCEV_Euler"]))
    op = ours.groupby(["ticker", "tgt"]).crps.mean()
    for m in [x for x in rows if x != "HARCEV_Euler"]:
        d = frames[m].assign(crps=wis_col(frames[m]))
        bp = d.groupby(["ticker", "tgt"]).crps.mean()
        dif = (bp - op).dropna()
        tk = dif.groupby("ticker").mean()
        p = binomtest((dif > 0).sum(), len(dif)).pvalue
        print(f"{m:13s} pairs better={100 * (dif > 0).mean():5.1f}% "
              f"(p={p:.1e})  tickers +={int((tk > 0).sum())}/{len(tk)}")

    # Moving-block bootstrap procedure (as reported in the paper):
    #   1. average per-day loss differentials cross-sectionally over all
    #      pairs active on that date -> one daily time series
    #   2. resample the daily series with 21-day moving blocks
    #   3. B = 2000 resamples; 95% CI = percentile interval [2.5, 97.5]
    rng = np.random.default_rng(0)
    best = "TCN" if "TCN" in frames else rows[1]
    d = frames[best].assign(crps=wis_col(frames[best]))
    mrg = d[["ticker", "tgt", "date", "crps"]].merge(
        ours[["ticker", "tgt", "date", "crps"]],
        on=["ticker", "tgt", "date"], suffixes=("_b", "_o"))
    daily = (mrg.crps_b - mrg.crps_o).groupby(mrg.date).mean().sort_index()
    x, n, block, B = daily.values, len(daily), 21, 2000
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(B, nb))
    idx = (starts[:, :, None]
           + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    print(f"21d moving-block bootstrap CI of daily mean WIS diff "
          f"(vs {best}): [{lo:.4f}, {hi:.4f}]")

    dl_frames = [frames[m].assign(crps=wis_col(frames[m]))
                 for m in DL6 if m in frames]
    if not dl_frames:
        print("no DL baseline predictions found; skipping WD regression")
        return
    try:
        import statsmodels.api as sm
        dl_pair = pd.concat(dl_frames).groupby(
            ["ticker", "tgt"]).crps.mean()
        tbl = pd.DataFrame({"ours": op, "dl": dl_pair}).dropna().reset_index()
        tbl = tbl.merge(w[["ticker", "tgt", "wd"]], on=["ticker", "tgt"])
        tbl["crisis"] = tbl.tgt.isin(["Crisis I", "Crisis II"]).astype(float)
        tbl["rel"] = (tbl.dl - tbl.ours) / tbl.dl
        reg = sm.OLS(tbl.rel, sm.add_constant(tbl[["wd", "crisis"]])).fit(
            cov_type="cluster", cov_kwds={"groups": tbl.tgt})
        print(f"rel improvement ~ WD + crisis (regime-clustered SE): "
              f"wd={reg.params['wd']:.2f} (p={reg.pvalues['wd']:.3f}), "
              f"crisis={reg.params['crisis']:.2f} "
              f"(p={reg.pvalues['crisis']:.2f})")
    except ImportError:
        print("statsmodels not installed; skipping WD regression")


if __name__ == "__main__":
    main()



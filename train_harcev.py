"""Train HAR-CEV and its ablations under the zero-shot protocol.

Models: HARCEV_Euler (proposed), HARCEV_FixedA, AntiMR, GRULag (No-SDE).
HARCEV_Exp is available as an optional robustness specification.

For every (ticker, target regime) pair: train on the preceding 756 trading
days (3 random seeds), forecast every day of the target regime with no
adaptation, and save the seed-ensembled quantiles.

Outputs:
  outputs/predictions_harcev.parquet
      ticker, tgt, model, date, rv_actual, q05..q95
  outputs/sde_params.parquet          (HARCEV_Euler daily SDE parameters)
      ticker, tgt, date, kappa, theta, sigma, alpha, w1, w5, w22

Usage:
  python train_harcev.py --rv_dir dataset_rv
  python train_harcev.py --ticker AAPL --models HARCEV_Euler,GRULag
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from models import (GRUWithLag, HARAttentiveCEV, HARAttentiveCEVAntiMR,
                    HARAttentiveCEVExp, HARAttentiveCEVFixedAlpha,
                    pinball_crps)
from protocol import (ALL_TICKERS, clip_inputs, fit_clip_bounds, iter_pairs,
                      load_rv, make_loader, make_windows, set_seed)

MODEL_CLASSES = {
    "HARCEV_Euler": HARAttentiveCEV,
    "HARCEV_Exp": HARAttentiveCEVExp,
    "AntiMR": HARAttentiveCEVAntiMR,
    "GRULag": GRUWithLag,
    "HARCEV_FixedA": HARAttentiveCEVFixedAlpha,
}
KAPPA_INIT, SIGMA_INIT, ALPHA_INIT = 48.123, 9.6945, 0.6339
QCOLS = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]


def make_model(cls, rv_mean, rv_std, look_back, device):
    return cls(look_back=look_back, d_ctx=64,
               kappa_init=KAPPA_INIT, kappa_min=2.0, theta_min=0.005,
               sigma_init=SIGMA_INIT, sigma_min=0.5, alpha_init=ALPHA_INIT,
               rv_mean=rv_mean, train_mean=rv_mean,
               train_std=rv_std).to(device)


def train_one(rv_train, rv_mean, rv_std, cls, args, device, seed,
              ckpt=None, clip_bounds=None):
    set_seed(seed)
    model = make_model(cls, rv_mean, rv_std, args.look_back, device)
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        model.eval()
        return model

    n_val = max(min(max(args.look_back + 5, int(len(rv_train) * 0.15)),
                    len(rv_train) - args.look_back - 2), 0)
    rv_tr = rv_train[:-n_val] if n_val > 0 else rv_train
    rv_va = rv_train[-n_val:] if n_val > 0 else None

    X_tr, y_tr = make_windows(rv_tr, rv_mean, args.look_back, clip_bounds)
    if X_tr is None or len(X_tr) < 3:
        return None
    tr_ld = make_loader(X_tr, y_tr, True, args.batch)

    va_ld = None
    if rv_va is not None:
        Xv, yv = make_windows(rv_va, rv_mean, args.look_back, clip_bounds)
        if Xv is not None and len(Xv) > 0:
            va_ld = make_loader(Xv, yv, False, args.batch)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5,
                                               min_lr=1e-6)
    best_val, no_imp, best_state = float("inf"), 0, None

    for _ in range(args.n_epochs):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            v_q, *_ = model.forward_eps(xb)
            loss = pinball_crps(v_q, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if va_ld is not None:
            model.eval()
            val, n = 0.0, 0
            with torch.no_grad():
                for xb, yb in va_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    val += pinball_crps(model(xb)[0], yb).item() * len(yb)
                    n += len(yb)
            val /= n
            sch.step(val)
            if val < best_val:
                best_val, no_imp = val, 0
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
            else:
                no_imp += 1
            if no_imp >= args.patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(model.state_dict(), ckpt)
    return model


@torch.no_grad()
def predict_regime(model, rv_train, rv_tgt, rv_mean, look_back, batch,
                   device, clip_bounds, collect_params=False):
    """One-step-ahead quantiles (and SDE parameters) for every target day."""
    full = np.concatenate([rv_train, rv_tgt])
    full_x = clip_inputs(full, clip_bounds)
    X = [full_x[i - look_back:i] / rv_mean
         for i in range(len(rv_train), len(full))]
    X_t = torch.tensor(np.array(X, np.float32))

    model.eval()
    vq, params = [], []
    for i in range(0, len(X_t), batch):
        out = model(X_t[i:i + batch].to(device))
        vq.append(out[0].cpu().numpy())
        if collect_params:
            # out = v_q, mu, s, kappa, theta, sigma, alpha, beta, weights, f_t
            kappa, theta, sigma, alpha = [out[j].cpu().numpy()
                                          for j in (3, 4, 5, 6)]
            w = out[8].cpu().numpy()
            params.append(np.column_stack([kappa, theta, sigma, alpha, w]))
    preds = np.concatenate(vq, axis=0)
    par = np.concatenate(params, axis=0) if collect_params else None
    return preds, par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="all")
    ap.add_argument("--models", default="HARCEV_Euler,HARCEV_FixedA,AntiMR,"
                                        "GRULag")
    ap.add_argument("--rv_dir", default="dataset_rv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ckpt_dir", default="ckpt_harcev")
    ap.add_argument("--look_back", type=int, default=22)
    ap.add_argument("--train_window", type=int, default=756)
    ap.add_argument("--n_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip_lower", type=float, default=0.0)
    ap.add_argument("--clip_upper", type=float, default=100.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tickers = ALL_TICKERS if args.ticker == "all" else [args.ticker]
    model_names = [name.strip() for name in args.models.split(",")]
    unknown = sorted(set(model_names) - set(MODEL_CLASSES))
    if unknown:
        ap.error(f"unknown model(s): {', '.join(unknown)}")
    os.makedirs(args.out_dir, exist_ok=True)

    pred_rows, param_rows = [], []
    for tk in tickers:
        try:
            rv_series = load_rv(tk, args.rv_dir)
        except FileNotFoundError:
            print(f"[{tk}] no RV file, skipping")
            continue
        print(f"[{tk}]")
        for tgt, rv_tr, rv_tgt, dates in iter_pairs(
                rv_series, args.look_back, args.train_window):
            clip_bounds = fit_clip_bounds(rv_tr, args.clip_lower,
                                          args.clip_upper)
            stats = clip_inputs(rv_tr, clip_bounds)
            rv_mean = float(stats.mean())
            rv_std = max(float(stats.std()), 1e-6)

            for name in model_names:
                cls = MODEL_CLASSES[name]
                seed_preds, seed_params = [], []
                for seed in range(args.n_seeds):
                    ckpt = os.path.join(
                        args.ckpt_dir, tk, name,
                        f"{tgt.replace(' ', '_')}_s{seed}.pt")
                    model = train_one(rv_tr, rv_mean, rv_std, cls, args,
                                      device, seed, ckpt, clip_bounds)
                    if model is None:
                        continue
                    preds, par = predict_regime(
                        model, rv_tr, rv_tgt, rv_mean, args.look_back,
                        args.batch, device, clip_bounds,
                        collect_params=(name == "HARCEV_Euler"))
                    seed_preds.append(preds)
                    if par is not None:
                        seed_params.append(par)
                if not seed_preds:
                    continue
                ens = np.mean(np.stack(seed_preds), axis=0)
                df = pd.DataFrame(ens, columns=QCOLS)
                df.insert(0, "rv_actual", rv_tgt)
                df.insert(0, "date", dates)
                df.insert(0, "model", name)
                df.insert(0, "tgt", tgt)
                df.insert(0, "ticker", tk)
                pred_rows.append(df)
                if seed_params:
                    par = np.mean(np.stack(seed_params), axis=0)
                    pf = pd.DataFrame(par, columns=[
                        "kappa", "theta", "sigma", "alpha",
                        "w1", "w5", "w22"])
                    pf.insert(0, "date", dates)
                    pf.insert(0, "tgt", tgt)
                    pf.insert(0, "ticker", tk)
                    param_rows.append(pf)
            print(f"  {tgt:<16} done")

    if not pred_rows:
        raise SystemExit(
            "No forecasts were produced. Check that dataset_rv contains "
            "rv_<TICKER>.csv files with enough observations for the selected "
            "training window and target regimes."
        )
    pd.concat(pred_rows, ignore_index=True).to_parquet(
        os.path.join(args.out_dir, "predictions_harcev.parquet"))
    if param_rows:
        pd.concat(param_rows, ignore_index=True).to_parquet(
            os.path.join(args.out_dir, "sde_params.parquet"))
    print("saved", args.out_dir)


if __name__ == "__main__":
    main()



"""Train the seven benchmark quantile forecasters under the zero-shot
protocol (identical optimizer schedule, windows, and seeds as HAR-CEV).

Outputs:
  outputs/predictions_baselines.parquet
      ticker, tgt, model, date, rv_actual, q05..q95

Usage:
  python train_baselines.py --rv_dir dataset_rv
  python train_baselines.py --model PatchTST --ticker AAPL
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from baselines import ALL_BENCHMARK_MODELS, build_model
from models import pinball_crps, quantile_loss
from protocol import (ALL_TICKERS, clip_inputs, fit_clip_bounds, iter_pairs,
                      load_rv, make_loader, make_windows, set_seed)

QCOLS = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]


def train_one(rv_train, model_name, rv_mean, args, device, seed,
              ckpt=None, clip_bounds=None):
    set_seed(seed)
    model = build_model(model_name, args.look_back, rv_mean).to(device)
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

    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5,
                                               min_lr=1e-6)
    best_val, no_imp, best_state = float("inf"), 0, None

    for _ in range(args.n_epochs):
        model.train()
        for xb, yb in tr_ld:
            opt.zero_grad()
            v_q, *_ = model(xb.to(device))
            quantile_loss(v_q, yb.to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if va_ld is not None:
            model.eval()
            val, n = 0.0, 0
            with torch.no_grad():
                for xb, yb in va_ld:
                    v_q, *_ = model(xb.to(device))
                    val += pinball_crps(v_q, yb.to(device)).item() * len(yb)
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
                   device, clip_bounds):
    full = np.concatenate([rv_train, rv_tgt])
    full_x = clip_inputs(full, clip_bounds)
    X = [full_x[i - look_back:i] / rv_mean
         for i in range(len(rv_train), len(full))]
    X_t = torch.tensor(np.array(X, np.float32))
    model.eval()
    vq = []
    for i in range(0, len(X_t), batch):
        v_q, *_ = model(X_t[i:i + batch].to(device))
        vq.append(v_q.cpu().numpy())
    return np.concatenate(vq, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="all")
    ap.add_argument("--model", default="all",
                    choices=ALL_BENCHMARK_MODELS + ["all"])
    ap.add_argument("--rv_dir", default="dataset_rv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ckpt_dir", default="ckpt_baselines")
    ap.add_argument("--look_back", type=int, default=22)
    ap.add_argument("--train_window", type=int, default=756)
    ap.add_argument("--n_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--clip_lower", type=float, default=0.0)
    ap.add_argument("--clip_upper", type=float, default=99.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tickers = ALL_TICKERS if args.ticker == "all" else [args.ticker]
    model_names = (ALL_BENCHMARK_MODELS if args.model == "all"
                   else [args.model])
    os.makedirs(args.out_dir, exist_ok=True)

    pred_rows = []
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
            rv_mean = float(clip_inputs(rv_tr, clip_bounds).mean())

            for name in model_names:
                seed_preds = []
                for seed in range(args.n_seeds):
                    ckpt = os.path.join(
                        args.ckpt_dir, tk, name,
                        f"{tgt.replace(' ', '_')}_s{seed}.pt")
                    model = train_one(rv_tr, name, rv_mean, args, device,
                                      seed, ckpt, clip_bounds)
                    if model is None:
                        continue
                    seed_preds.append(predict_regime(
                        model, rv_tr, rv_tgt, rv_mean, args.look_back,
                        args.batch, device, clip_bounds))
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
            print(f"  {tgt:<16} done")

    if not pred_rows:
        raise SystemExit(
            "No forecasts were produced. Check dataset_rv and the selected "
            "training window/target regimes."
        )
    pd.concat(pred_rows, ignore_index=True).to_parquet(
        os.path.join(args.out_dir, "predictions_baselines.parquet"))
    print("saved", args.out_dir)


if __name__ == "__main__":
    main()


# HAR-CEV

Official anonymous implementation for *HAR-CEV: Probabilistic Volatility Forecasting with Adaptive Mean Reversion under Distribution Shift*.

**Submission status:** Submitted to ICAIF 2026. This repository corresponds to the final submitted `main.tex`; the code, experimental protocol, and default configurations match the submitted manuscript.

HAR-CEV combines daily, weekly, and monthly HAR components with a context-dependent CEV transition and a monotone quantile head. The repository contains the proposed Euler model, the fixed-alpha and structural ablations, seven neural/econometric baselines, GARCH-EQ, distribution-shift analysis, evaluation tables, and VaR backtests. Data and trained checkpoints are intentionally excluded.

## Requirements

- Python 3.10 or later
- PyTorch 2.0 or later
- A CUDA GPU is optional

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py
```

Run every command below from the repository root.

## Data

Place raw 5-minute files at:

```text
dataset_dija/dataset_{TICKER}.csv
```

The first column must be a parseable timestamp; `close` and `volume` columns are required (case-insensitive). See `dataset_dija/README.md` for the schema. Raw data should cover the training windows and evaluation regimes defined in `protocol.py`.

Alternatively, provide realized-volatility files directly:

```text
dataset_rv/rv_{TICKER}.csv
```

Each file must have a date column/index and a strictly positive `rv` column. No data are redistributed with this repository.

Validate supplied files before training:

```bash
python validate_data.py
```

## Reproduce the study

```bash
# 1. Convert intraday bars to daily annualized realized volatility
python compute_rv.py --data_dir dataset_dija --out_dir dataset_rv

# 2. Proposed Euler model and paper ablations
python train_harcev.py --rv_dir dataset_rv

# Optional exponential-discretization robustness model
python train_harcev.py --rv_dir dataset_rv --models HARCEV_Exp

# 3. Seven benchmark models
python train_baselines.py --rv_dir dataset_rv

# 4. GARCH(1,1) empirical-ratio quantile baseline
python garch_eq.py --data_dir dataset_dija --rv_dir dataset_rv

# 5. Training-to-target Wasserstein distance
python compute_wd.py --rv_dir dataset_rv --out_dir outputs

# 6. Paper metrics, tables, and panel-aware checks
python evaluate.py --pred_dir outputs

# 7. VaR backtest
python var_backtest.py --data_dir dataset_dija --rv_dir dataset_rv
```

The default HAR-CEV run includes `HARCEV_Euler`, `HARCEV_FixedA` (alpha fixed at 0.5), `AntiMR`, and `GRULag`. Partial runs are supported:

```bash
python train_harcev.py --ticker AAPL --models HARCEV_Euler,HARCEV_FixedA
python train_baselines.py --ticker AAPL --model TCN
```

Training outputs are written to `outputs/`; checkpoints are written to `ckpt_harcev/` and `ckpt_baselines/`. These generated artifacts are ignored by Git. Training resumes from compatible checkpoints.

## Main files

| File | Purpose |
|---|---|
| `models.py` | HAR-CEV Euler, fixed-alpha, robustness, and structural ablations |
| `baselines.py` | HAR, DLinear, LSTM, TCN, N-BEATS, PatchTST, and iTransformer |
| `protocol.py` | Tickers, regimes, rolling windows, preprocessing, and seeds |
| `compute_rv.py` | Intraday bars to daily annualized realized volatility |
| `train_harcev.py` | HAR-CEV training and inference |
| `train_baselines.py` | Baseline training and inference |
| `garch_eq.py` | GARCH-EQ baseline |
| `compute_wd.py` | Pair-level Wasserstein distance |
| `evaluate.py` | Metrics, paper tables, and statistical checks |
| `var_backtest.py` | VaR-95/99 evaluation |
| `validate_data.py` | Read-only input-schema validation |
| `smoke_test.py` | Data-free forward-pass checks for all neural models |

## Protocol summary

For each valid ticker-regime pair, models train on the preceding 756 trading days and forecast the target regime without adaptation. They predict seven next-day annualized realized-volatility quantiles at 5%, 10%, 25%, 50%, 75%, 90%, and 95%. Exact tickers, dates, exclusions, clipping settings, and optimization defaults are encoded in the scripts and can be inspected through each command's `--help` option.

## License

This project is released under the MIT License. See `LICENSE`.

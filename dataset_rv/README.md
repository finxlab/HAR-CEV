# Realized-volatility data

`compute_rv.py` writes one file per asset here:

```text
dataset_rv/
  rv_AAPL.csv
  rv_MSFT.csv
```

Each file has a date index and an `rv` column containing positive daily
realized volatility. You may also supply files in this format directly.

# Raw data

Place one file per asset here:

```text
dataset_dija/
  dataset_AAPL.csv
  dataset_MSFT.csv
```

Each CSV must contain:

- a timestamp in the first column;
- `Close` (or `close`);
- `Volume` (or `volume`).

Rows should be ordered chronologically. The raw files are intentionally not
included in this repository.

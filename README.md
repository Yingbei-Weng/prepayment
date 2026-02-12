# Prepayment rate replication (Kang & Zenios, 1992)

This project replicates the core factor-model workflow from *Complete Prepayment Models for Mortgage-Backed Securities* (Kang & Zenios, 1992) using the provided, preprocessed Freddie Mac-derived dataset (`data.csv`) and a quarterly 10Y rate proxy (`10y_yahoo_quarter_avg.csv`).

## Quickstart

1. Create the virtual environment and install dependencies:

```bash
uv venv
uv sync --extra dev
uv pip install -e .
```

2. Run the replication pipeline (writes plots + fitted factors to `outputs/`):

```bash
uv run python -m prepayment.replicate --data data.csv --rates 10y_yahoo_quarter_avg.csv --out outputs
```

3. Run tests:

```bash
uv run pytest
```

## Notes

- The original paper is monthly; this implementation works on *quarterly* observations and uses quarter-of-year as the seasonality index.
- The provided 10Y series appears to be sourced from Yahoo `^TNX` (yield ×10). The loader auto-detects this and rescales to a decimal rate.

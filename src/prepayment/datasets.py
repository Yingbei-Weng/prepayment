from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RateSeries:
    df: pd.DataFrame
    scale_note: str


def load_quarterly_rates(path: str | Path) -> RateSeries:
    rates = pd.read_csv(path)
    if list(rates.columns) != ["Date", "10Y_yield_pct"]:
        raise ValueError(
            f"Unexpected columns in {path!s}: {rates.columns.tolist()} (expected Date, 10Y_yield_pct)"
        )

    rates = rates.rename(columns={"Date": "period", "10Y_yield_pct": "market_rate_raw"})
    rates["market_rate_raw"] = pd.to_numeric(rates["market_rate_raw"], errors="coerce")

    median = float(np.nanmedian(rates["market_rate_raw"].to_numpy()))
    if not np.isfinite(median):
        raise ValueError("Rate series has no finite values")

    if median > 1.0:
        rates["market_rate"] = rates["market_rate_raw"] / 100.0
        note = "Detected percent units; scaled by /100 to decimal."
    elif median > 0.2:
        rates["market_rate"] = rates["market_rate_raw"] / 10.0
        note = "Detected Yahoo ^TNX-style scale (yield×10); scaled by /10 to decimal."
    else:
        rates["market_rate"] = rates["market_rate_raw"]
        note = "Detected decimal units; no scaling applied."

    rates = rates[["period", "market_rate"]].dropna()
    return RateSeries(df=rates, scale_note=note)


def load_loan_panel(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    loans = pd.read_csv(path, na_values=["", " "], nrows=nrows, low_memory=False)

    date_cols = ["dt_fund", "dt_io_end", "dt_mty", "dt_sold", "liq_dte"]
    for col in date_cols:
        if col in loans.columns:
            loans[col] = pd.to_datetime(loans[col], errors="coerce", format="mixed")

    numeric_cols = [
        "amt_upb_endg",
        "liq_upb_amt",
        "amt_upb_pch",
        "rate_ltv",
        "rate_dcr",
        "rate_int",
        "cnt_amtn_per",
        "cnt_blln_term",
        "cnt_io_per",
        "cnt_mrtg_term",
        "cnt_rsdntl_unit",
        "cnt_yld_maint",
        "Sales_Price",
        "credit_loss",
        "year",
        "q",
        "sort_key",
        "mrtg_status",
    ]
    for col in numeric_cols:
        if col in loans.columns:
            loans[col] = pd.to_numeric(loans[col], errors="coerce")

    if "lnno" in loans.columns:
        loans["lnno"] = pd.to_numeric(loans["lnno"], errors="coerce").astype("Int64")

    required_base = {"lnno", "rate_int", "amt_upb_pch", "amt_upb_endg", "mrtg_status"}
    missing_base = sorted(required_base - set(loans.columns))
    if missing_base:
        raise ValueError(f"Missing required columns in loan data: {missing_base}")

    has_yqs = {"year", "q", "sort_key"}.issubset(loans.columns)
    if not has_yqs:
        if "quarter" not in loans.columns:
            raise ValueError("Loan data must include either (year, q, sort_key) or quarter")

        quarter_dt = pd.to_datetime(loans["quarter"], errors="coerce", format="%m/%d/%y")
        if quarter_dt.isna().mean() > 0.05:
            quarter_dt = pd.to_datetime(loans["quarter"], errors="coerce", format="%m/%d/%Y")
        if quarter_dt.isna().any():
            quarter_dt = pd.to_datetime(loans["quarter"], errors="coerce", format="mixed")

        if quarter_dt.notna().sum() == 0:
            raise ValueError("Unable to parse quarter column as dates")

        loans["year"] = quarter_dt.dt.year.astype("Int64")
        loans["q"] = quarter_dt.dt.quarter.astype("Int64")
        loans["sort_key"] = (loans["year"] * 10 + loans["q"]).astype("Int64")

    required = {"year", "q", "sort_key"}
    missing = sorted(required - set(loans.columns))
    if missing:
        raise ValueError(f"Missing required columns in loan data: {missing}")

    loans["period"] = loans["year"].astype("Int64").astype(str) + "Q" + loans["q"].astype("Int64").astype(str)

    return loans

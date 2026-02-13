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

    date_cols = ["dt_fund", "dt_io_end", "dt_mty", "dt_sold", "liq_dte", "quarter"]
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
        "market_rate",
        "is_prepaid",
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

    if "year" not in loans.columns or "q" not in loans.columns:
        if "quarter" in loans.columns:
            quarter_dt = loans["quarter"]
            if pd.api.types.is_datetime64_any_dtype(quarter_dt) and quarter_dt.notna().any():
                loans["year"] = quarter_dt.dt.year
                loans["q"] = quarter_dt.dt.quarter
            else:
                quarter_str = quarter_dt.astype(str).str.strip().str.lower()
                extracted = quarter_str.str.extract(r"^y?(?P<yy>\\d{2,4})q(?P<q>[1-4])$")
                yy = pd.to_numeric(extracted["yy"], errors="coerce")
                year = np.where(yy < 100, 2000 + yy, yy)
                loans["year"] = pd.to_numeric(year, errors="coerce")
                loans["q"] = pd.to_numeric(extracted["q"], errors="coerce")

    if "sort_key" not in loans.columns and {"year", "q"} <= set(loans.columns):
        loans["sort_key"] = loans["year"] * 10 + loans["q"]

    required = {"lnno", "year", "q", "sort_key", "rate_int", "amt_upb_pch", "amt_upb_endg", "mrtg_status"}
    missing = sorted(required - set(loans.columns))
    if missing:
        raise ValueError(f"Missing required columns in loan data: {missing}")

    loans["period"] = loans["year"].astype("Int64").astype(str) + "Q" + loans["q"].astype("Int64").astype(str)

    return loans

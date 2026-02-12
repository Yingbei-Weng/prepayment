from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PrepayDefinition:
    treat_maturity_payoff_as_prepay: bool = False
    payoff_status_code: int = 500


def add_quarter_end_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    end_month = (df["q"].astype(int) * 3).astype(int)
    df["period_end"] = pd.to_datetime(
        {"year": df["year"].astype(int), "month": end_month, "day": 1}, errors="coerce"
    ) + pd.offsets.MonthEnd(0)
    return df


def add_loan_age_quarters(df: pd.DataFrame) -> pd.DataFrame:
    df = add_quarter_end_date(df)
    if "dt_fund" not in df.columns:
        raise ValueError("Expected dt_fund to compute loan age")

    fund_month_index = df["dt_fund"].dt.year * 12 + df["dt_fund"].dt.month
    obs_month_index = df["period_end"].dt.year * 12 + df["period_end"].dt.month
    age_months = (obs_month_index - fund_month_index).astype("Int64")

    df = df.copy()
    df["age_months"] = age_months
    df["age_quarters"] = (age_months // 3).astype("Int64")
    return df


def add_beginning_upb(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["lnno", "sort_key"], kind="mergesort")
    df["begin_upb"] = df.groupby("lnno", observed=True)["amt_upb_endg"].shift(1)
    df["begin_upb"] = df["begin_upb"].fillna(df["amt_upb_pch"])
    return df


def add_market_rate_and_incentive(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(rates, how="left", on="period")
    df = df.copy()
    df["c_over_r"] = df["rate_int"] / df["market_rate"]
    return df


def add_prepayment_flags(df: pd.DataFrame, definition: PrepayDefinition) -> pd.DataFrame:
    df = df.copy()

    status = df["mrtg_status"].astype("Int64")
    is_payoff = (status == definition.payoff_status_code).fillna(False)

    liq_month = df["liq_dte"].dt.to_period("M") if "liq_dte" in df.columns else pd.Series(pd.NaT, index=df.index)
    mty_month = df["dt_mty"].dt.to_period("M") if "dt_mty" in df.columns else pd.Series(pd.NaT, index=df.index)
    is_maturity_payoff = (is_payoff & liq_month.notna() & mty_month.notna() & (liq_month == mty_month)).fillna(
        False
    )

    if definition.treat_maturity_payoff_as_prepay:
        is_prepay = is_payoff
    else:
        is_prepay = is_payoff & ~is_maturity_payoff

    credit_loss = df.get("credit_loss")
    if credit_loss is not None:
        credit_loss_num = pd.to_numeric(credit_loss, errors="coerce").fillna(0.0)
        is_prepay = is_prepay & (credit_loss_num <= 0.0)

    df["is_payoff"] = is_payoff.astype(bool)
    df["is_maturity_payoff"] = is_maturity_payoff.astype(bool)
    df["is_prepay"] = is_prepay.fillna(False).astype(bool)

    liq_upb = pd.to_numeric(df.get("liq_upb_amt"), errors="coerce")
    df["prepay_upb"] = np.where(df["is_prepay"], liq_upb.fillna(df["begin_upb"]), 0.0)

    return df


def prepare_loan_quarterly_data(
    loans: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    prepay_definition: PrepayDefinition | None = None,
) -> pd.DataFrame:
    if prepay_definition is None:
        prepay_definition = PrepayDefinition()

    df = loans
    df = add_beginning_upb(df)
    df = add_loan_age_quarters(df)
    df = add_market_rate_and_incentive(df, rates)
    df = add_prepayment_flags(df, prepay_definition)

    df = df.copy()
    df["season"] = df["q"].astype("Int64")

    return df

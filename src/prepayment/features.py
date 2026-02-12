from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PrepayDefinition:
    closed_status_codes: tuple[int, ...] = (500,)
    foreclosure_status_codes: tuple[int, ...] = (300,)
    reo_status_codes: tuple[int, ...] = (450,)
    modification_with_loss_status_codes: tuple[int, ...] = (250,)
    treat_maturity_payoff_as_prepay: bool = False


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
    df = df.sort_values(["lnno", "sort_key"], kind="mergesort").copy()

    status = df["mrtg_status"].astype("Int64")
    closed = status.isin(definition.closed_status_codes).fillna(False)
    upb_zero = pd.to_numeric(df["amt_upb_endg"], errors="coerce").fillna(np.nan).eq(0.0)
    liq_dte = df.get("liq_dte")
    liq_dte_notnull = liq_dte.notna() if liq_dte is not None else pd.Series(False, index=df.index)
    liq_dte_missing = liq_dte.isna() if liq_dte is not None else pd.Series(True, index=df.index)

    terminated = (upb_zero | closed | liq_dte_notnull).fillna(False)
    term_count = terminated.groupby(df["lnno"], observed=True).cumsum()
    term_event = terminated & term_count.eq(1)

    dt_mty = df.get("dt_mty")
    mty_sort_key = (
        (dt_mty.dt.year * 10 + dt_mty.dt.quarter).astype("Int64")
        if dt_mty is not None
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )

    if liq_dte is None or dt_mty is None:
        liq_before_mty = pd.Series(False, index=df.index)
    else:
        if definition.treat_maturity_payoff_as_prepay:
            liq_before_mty = liq_dte.notna() & dt_mty.notna() & (liq_dte <= dt_mty)
        else:
            liq_before_mty = liq_dte.notna() & dt_mty.notna() & (liq_dte < dt_mty)

    if definition.treat_maturity_payoff_as_prepay:
        term_before_mty_quarter = (
            liq_dte_missing
            & term_event
            & df["sort_key"].notna()
            & mty_sort_key.notna()
            & (df["sort_key"].astype("Int64") <= mty_sort_key)
        )
    else:
        term_before_mty_quarter = (
            liq_dte_missing
            & term_event
            & df["sort_key"].notna()
            & mty_sort_key.notna()
            & (df["sort_key"].astype("Int64") < mty_sort_key)
        )

    early_payoff = term_event & (liq_before_mty | term_before_mty_quarter)

    disqual_status_codes = (
        set(definition.modification_with_loss_status_codes)
        | set(definition.foreclosure_status_codes)
        | set(definition.reo_status_codes)
    )
    bad_status_row = status.isin(disqual_status_codes).fillna(False)
    bad_status_ever = bad_status_row.groupby(df["lnno"], observed=True).transform("max").astype(bool)

    credit_loss = pd.to_numeric(df.get("credit_loss", 0.0), errors="coerce").fillna(0.0)
    bad_credit_loss_ever = credit_loss.gt(0.0).groupby(df["lnno"], observed=True).transform("max").astype(bool)

    sale_evidence = pd.Series(False, index=df.index)
    if "dt_sold" in df.columns:
        sale_evidence = sale_evidence | df["dt_sold"].notna()
    if "Sales_Price" in df.columns:
        sale_evidence = sale_evidence | df["Sales_Price"].notna()
    bad_sale_ever = sale_evidence.groupby(df["lnno"], observed=True).transform("max").astype(bool)

    defeased = pd.to_numeric(df.get("flag_defeased", 0.0), errors="coerce").fillna(0.0).ne(0.0)
    bad_defeased_ever = defeased.groupby(df["lnno"], observed=True).transform("max").astype(bool)

    disqualified = bad_status_ever | bad_credit_loss_ever | bad_sale_ever | bad_defeased_ever
    is_prepay = (early_payoff & ~disqualified).fillna(False).astype(bool)

    liq_upb = pd.to_numeric(df.get("liq_upb_amt"), errors="coerce")
    df["is_terminated"] = terminated.astype(bool)
    df["is_early_payoff"] = early_payoff.fillna(False).astype(bool)
    df["is_prepay"] = is_prepay
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

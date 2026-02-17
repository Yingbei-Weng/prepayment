from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datasets import load_loan_panel, load_quarterly_rates
from .factors import (
    _annualized_cpr_from_smm,
    fit_burnout_curve,
    fit_refinancing_incentive,
    fit_seasonality,
    fit_seasoning_curve,
    weighted_r2,
)
from .features import PrepayDefinition, prepare_loan_quarterly_data
from .plots import save_aging_plot, save_refinancing_plot, save_roc_plot, save_seasonality_plot


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit prepayment models (factor and/or discrete-time hazard logit).")
    p.add_argument("--data", required=True, help="Path to loan panel CSV (e.g., data.csv or data_new.csv)")
    p.add_argument("--rates", required=True, help="Path to 10y_yahoo_quarter_avg.csv")
    p.add_argument("--out", default="outputs", help="Output directory")
    p.add_argument("--sample-rows", type=int, default=None, help="Load only the first N rows of the loan panel CSV")
    p.add_argument(
        "--model",
        choices=["factor", "logit", "both"],
        default="logit",
        help="Which model(s) to run (factor = paper-style factors; logit = discrete-time hazard logit)",
    )

    p.add_argument("--seasonality-age-min", type=int, default=4, help="Min age (quarters) for seasonality fit")
    p.add_argument("--seasonality-age-max", type=int, default=40, help="Max age (quarters) for seasonality fit")

    p.add_argument("--refi-age-min", type=int, default=4, help="Min age (quarters) for refinancing fit")
    p.add_argument("--refi-age-max", type=int, default=40, help="Max age (quarters) for refinancing fit")
    p.add_argument("--refi-x-min", type=float, default=0.5, help="Min C/R for refinancing curve")
    p.add_argument("--refi-x-max", type=float, default=1.6, help="Max C/R for refinancing curve")
    p.add_argument("--refi-segments", type=int, default=22, help="Number of segments for refinancing basis")

    p.add_argument("--seasoning-age-max", type=int, default=16, help="Max age (quarters) for seasoning fit")
    p.add_argument("--burnout-age-min", type=int, default=20, help="Min age (quarters) for burnout fit")
    p.add_argument("--burnout-age-max", type=int, default=None, help="Max age (quarters) for burnout fit")

    p.add_argument(
        "--treat-maturity-payoff-as-prepay",
        action="store_true",
        help="Count payoffs occurring in maturity month as prepayment events",
    )

    p.add_argument("--logit-test-size", type=float, default=0.2, help="Test size for logit train/test split")
    p.add_argument("--logit-random-state", type=int, default=42, help="Random seed for logit train/test split")
    p.add_argument(
        "--logit-class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Class weight mode for logit (balanced improves classification, none improves calibration)",
    )

    return p.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _weighted_mean_by_age(df: pd.DataFrame, *, value_col: str, weight_col: str) -> pd.DataFrame:
    work = df[["age_quarters", value_col, weight_col]].copy()
    work = work[np.isfinite(work[value_col]) & np.isfinite(work[weight_col]) & (work[weight_col] > 0)].copy()
    work["wy"] = work[value_col].to_numpy(dtype=float) * work[weight_col].to_numpy(dtype=float)
    g = work.groupby("age_quarters", observed=True)
    sum_w = g[weight_col].sum(min_count=1)
    sum_wy = g["wy"].sum(min_count=1)

    out = pd.DataFrame({"age": sum_w.index.astype(int), "value": (sum_wy / sum_w), "weight": sum_w})
    return out.sort_values("age", kind="mergesort")


def _loan_level_prepayment_metrics(df: pd.DataFrame) -> dict[str, object]:
    work = df[["lnno", "sort_key", "is_prepay"]].copy()
    optional = [c for c in ["prepay_category", "is_terminated"] if c in df.columns]
    if optional:
        work = work.join(df[optional])

    work = work[work["lnno"].notna() & work["sort_key"].notna()].copy()
    work = work.sort_values(["lnno", "sort_key"], kind="mergesort")

    per_loan_prepay = work.groupby("lnno", observed=True)["is_prepay"].max()
    n_loans = int(per_loan_prepay.shape[0])
    n_prepay = int(per_loan_prepay.sum())
    rate_all = float(n_prepay / n_loans) if n_loans > 0 else float("nan")

    last = work.groupby("lnno", observed=True).tail(1).set_index("lnno")
    terminal_counts: dict[str, int] | None = None
    terminated_mask: pd.Series | None = None

    if "prepay_category" in last.columns:
        terminal = last["prepay_category"].astype(str)
        terminal_counts = terminal.value_counts(dropna=False).astype(int).to_dict()
        terminated_mask = terminal.ne("Active")
    elif "is_terminated" in work.columns:
        per_loan_term = work.groupby("lnno", observed=True)["is_terminated"].max().astype(bool)
        terminated_mask = per_loan_term.reindex(per_loan_prepay.index).fillna(False)

    if terminated_mask is None:
        n_terminated = None
        n_active = None
        rate_terminated = None
    else:
        terminated_mask = terminated_mask.reindex(per_loan_prepay.index).fillna(False)
        n_terminated = int(terminated_mask.sum())
        n_active = int((~terminated_mask).sum())
        if n_terminated > 0:
            rate_terminated = float(per_loan_prepay[terminated_mask].mean())
        else:
            rate_terminated = float("nan")

    return {
        "loan_level_n_loans": n_loans,
        "loan_level_n_prepay": n_prepay,
        "loan_level_prepay_rate_all_loans": rate_all,
        "loan_level_n_terminated": n_terminated,
        "loan_level_n_active": n_active,
        "loan_level_prepay_rate_terminated": rate_terminated,
        "loan_level_terminal_category_counts": terminal_counts,
    }


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    v = float(value)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.4%}"


def _smm_from_annualized_cpr(cpr: np.ndarray, *, periods_per_year: int) -> np.ndarray:
    cpr = np.asarray(cpr, dtype=float)
    cpr = np.clip(cpr, 0.0, 1.0 - 1e-12)
    return 1.0 - np.power(1.0 - cpr, 1.0 / periods_per_year)


def _loan_level_predicted_prepayment_metrics(df: pd.DataFrame) -> dict[str, object]:
    work = df[["lnno", "sort_key", "cpr_pred"]].copy()
    optional = [c for c in ["prepay_category", "is_terminated"] if c in df.columns]
    if optional:
        work = work.join(df[optional])

    work = work[work["lnno"].notna() & work["sort_key"].notna()].copy()
    work = work.sort_values(["lnno", "sort_key"], kind="mergesort")
    work["smm_pred"] = _smm_from_annualized_cpr(work["cpr_pred"].to_numpy(dtype=float), periods_per_year=4)
    work["log_survive"] = np.log1p(-np.clip(work["smm_pred"].to_numpy(dtype=float), 0.0, 1.0 - 1e-12))

    log_surv = work.groupby("lnno", observed=True)["log_survive"].sum()
    p_pred = 1.0 - np.exp(log_surv.to_numpy(dtype=float))
    p_pred = pd.Series(p_pred, index=log_surv.index)

    terminated_mask: pd.Series | None = None
    if "prepay_category" in work.columns:
        last = work.groupby("lnno", observed=True).tail(1).set_index("lnno")
        terminated_mask = last["prepay_category"].astype(str).ne("Active")
    elif "is_terminated" in work.columns:
        per_loan_term = work.groupby("lnno", observed=True)["is_terminated"].max().astype(bool)
        terminated_mask = per_loan_term.reindex(p_pred.index).fillna(False)

    if terminated_mask is None:
        rate_terminated = None
    else:
        terminated_mask = terminated_mask.reindex(p_pred.index).fillna(False)
        rate_terminated = float(p_pred[terminated_mask].mean()) if int(terminated_mask.sum()) > 0 else float("nan")

    return {
        "loan_level_predicted_prepay_rate_all_loans": float(p_pred.mean()) if len(p_pred) > 0 else float("nan"),
        "loan_level_predicted_prepay_rate_terminated": rate_terminated,
    }


def _add_logit_covariates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["incentive"] = work["c_over_r"] - 1.0
    work = work.sort_values(["lnno", "sort_key"], kind="mergesort")
    incentive_pos = work["incentive"].clip(lower=0.0).to_numpy(dtype=float)
    work["burn_cum"] = pd.Series(incentive_pos, index=work.index).groupby(work["lnno"], observed=True).cumsum()
    denom = work["age_quarters"].astype(float).clip(lower=1.0)
    work["burn_avg"] = work["burn_cum"] / denom
    return work


def _fit_logit_hazard(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    class_weight: str | None,
) -> tuple[Pipeline, dict[str, object], dict[str, dict[str, object]]]:
    work = df[["lnno", "season", "age_quarters", "c_over_r", "is_prepay", "sort_key"]].copy()
    work = _add_logit_covariates(work)

    work = work[
        work["lnno"].notna()
        & work["season"].notna()
        & work["age_quarters"].notna()
        & np.isfinite(work["incentive"])
        & np.isfinite(work["burn_avg"])
    ].copy()

    loan_has_prepay = work.groupby("lnno", observed=True)["is_prepay"].max().astype(int)
    loan_ids = loan_has_prepay.index.to_numpy()
    y_loan = loan_has_prepay.to_numpy()

    if len(np.unique(y_loan)) < 2 or len(loan_ids) < 4:
        train_loans = loan_ids
        test_loans: np.ndarray = np.asarray([], dtype=loan_ids.dtype)
    else:
        train_loans, test_loans = train_test_split(
            loan_ids, test_size=test_size, random_state=random_state, stratify=y_loan
        )

    train_mask = work["lnno"].isin(train_loans)
    test_mask = work["lnno"].isin(test_loans) if len(test_loans) else pd.Series(False, index=work.index)

    feature_cols = ["season", "age_quarters", "incentive", "burn_avg"]
    X_train = work.loc[train_mask, feature_cols]
    y_train = work.loc[train_mask, "is_prepay"].astype(int).to_numpy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("season", OneHotEncoder(handle_unknown="ignore", drop="first"), ["season"]),
            ("num", StandardScaler(), ["age_quarters", "incentive", "burn_avg"]),
        ]
    )
    clf = LogisticRegression(max_iter=2000, class_weight=class_weight)
    model = Pipeline([("prep", preprocessor), ("clf", clf)])
    model.fit(X_train, y_train)

    X_all = work[feature_cols]
    p_all = model.predict_proba(X_all)[:, 1]
    work["p_prepay_hazard"] = p_all
    work["log_survive"] = np.log1p(-np.clip(p_all, 0.0, 1.0 - 1e-12))
    log_survive = work.groupby("lnno", observed=True)["log_survive"].sum()
    p_ever = 1.0 - np.exp(log_survive.to_numpy(dtype=float))
    p_ever = pd.Series(p_ever, index=log_survive.index)

    metrics: dict[str, object] = {
        "logit_n_loans": int(loan_has_prepay.shape[0]),
        "logit_n_rows": int(len(work)),
        "logit_n_events": int(work["is_prepay"].sum()),
        "logit_predicted_loan_prepay_rate_all_loans": float(p_ever.mean()) if len(p_ever) else float("nan"),
    }

    roc_artifacts: dict[str, dict[str, object]] = {}
    if len(test_loans):
        X_test = work.loc[test_mask, feature_cols]
        y_test = work.loc[test_mask, "is_prepay"].astype(int).to_numpy()
        p_test = model.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) == 2:
            auc_row = float(roc_auc_score(y_test, p_test))
            metrics["logit_row_auc_test"] = auc_row
            fpr, tpr, _ = roc_curve(y_test, p_test)
            roc_artifacts["row"] = {"fpr": fpr, "tpr": tpr, "auc": auc_row}
        else:
            metrics["logit_row_auc_test"] = float("nan")
        metrics["logit_row_logloss_test"] = float(log_loss(y_test, p_test, labels=[0, 1]))

        loan_test = pd.Series(test_loans).astype(loan_has_prepay.index.dtype)
        loan_test = loan_test[loan_test.isin(p_ever.index)]
        y_loan_test = loan_has_prepay.reindex(loan_test).to_numpy(dtype=int)
        p_loan_test = p_ever.reindex(loan_test).to_numpy(dtype=float)
        if len(np.unique(y_loan_test)) == 2:
            auc_loan = float(roc_auc_score(y_loan_test, p_loan_test))
            metrics["logit_loan_auc_test"] = auc_loan
            fpr, tpr, _ = roc_curve(y_loan_test, p_loan_test)
            roc_artifacts["loan"] = {"fpr": fpr, "tpr": tpr, "auc": auc_loan}
        else:
            metrics["logit_loan_auc_test"] = float("nan")
        metrics["logit_predicted_loan_prepay_rate_test_loans"] = float(np.mean(p_loan_test)) if len(p_loan_test) else float("nan")

    try:
        feature_names = model.named_steps["prep"].get_feature_names_out()
        coef = model.named_steps["clf"].coef_.ravel()
        metrics["logit_intercept"] = float(model.named_steps["clf"].intercept_.item())
        metrics["logit_coef"] = {str(k): float(v) for k, v in zip(feature_names, coef, strict=False)}
    except Exception:
        pass

    return model, metrics, roc_artifacts


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    plots_dir = out_dir / "plots"
    _ensure_dir(plots_dir)

    rates_series = load_quarterly_rates(args.rates)
    rates = rates_series.df

    loans = load_loan_panel(args.data, nrows=args.sample_rows)
    prepay_def = PrepayDefinition(treat_maturity_payoff_as_prepay=args.treat_maturity_payoff_as_prepay)
    df = prepare_loan_quarterly_data(loans, rates, prepay_definition=prepay_def)

    loan_level_metrics = _loan_level_prepayment_metrics(df)

    df = df[(df["begin_upb"] > 0) & np.isfinite(df["begin_upb"])].copy()
    df = df[np.isfinite(df["c_over_r"]) & np.isfinite(df["market_rate"]) & df["age_quarters"].notna()].copy()

    logit_model: Pipeline | None = None
    logit_metrics: dict[str, object] = {}
    logit_roc: dict[str, dict[str, object]] = {}
    if args.model in ("logit", "both"):
        logit_model, logit_metrics, logit_roc = _fit_logit_hazard(
            df,
            test_size=args.logit_test_size,
            random_state=args.logit_random_state,
            class_weight=None if args.logit_class_weight == "none" else "balanced",
        )

        if "row" in logit_roc:
            save_roc_plot(
                logit_roc["row"]["fpr"],
                logit_roc["row"]["tpr"],
                auc=logit_roc["row"]["auc"],
                title="Logit ROC (row-level event)",
                out_path=plots_dir / "logit_roc_row.png",
            )
        if "loan" in logit_roc:
            save_roc_plot(
                logit_roc["loan"]["fpr"],
                logit_roc["loan"]["tpr"],
                auc=logit_roc["loan"]["auc"],
                title="Logit ROC (loan-level prepay)",
                out_path=plots_dir / "logit_roc_loan.png",
            )

    smm = (df["prepay_upb"] / df["begin_upb"]).to_numpy(dtype=float)
    df["cpr_obs"] = _annualized_cpr_from_smm(smm, periods_per_year=4)

    if args.model == "logit":
        metrics = {
            "n_obs": int(len(df)),
            "prepay_events": int(df["is_prepay"].sum()),
            "rate_scale_note": rates_series.scale_note,
            **loan_level_metrics,
            **logit_metrics,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

        report = f"""# Prepayment logit model

## Summary

- Observations: {metrics['n_obs']:,}
- Prepay events: {metrics['prepay_events']:,}
- Loans: {metrics['loan_level_n_loans']:,} (terminated: {metrics['loan_level_n_terminated']}, active: {metrics['loan_level_n_active']})
- Loan-level prepay rate (terminated loans): {_format_percent(metrics['loan_level_prepay_rate_terminated'])}
- Logit predicted loan-level prepay rate (all loans): {_format_percent(metrics.get('logit_predicted_loan_prepay_rate_all_loans'))}
- Logit row AUC (test): {metrics.get('logit_row_auc_test')}
- Logit loan AUC (test): {metrics.get('logit_loan_auc_test')}

## Plots

### ROC (row-level)
![](plots/logit_roc_row.png)

### ROC (loan-level)
![](plots/logit_roc_loan.png)

## Notes

- Rates: {metrics['rate_scale_note']}
"""
        (out_dir / "report.md").write_text(report)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    x_bins = np.linspace(args.refi_x_min, args.refi_x_max, args.refi_segments + 1)
    df["c_over_r_bin"] = pd.cut(df["c_over_r"].clip(args.refi_x_min, args.refi_x_max), bins=x_bins, include_lowest=True)

    df["w_obs"] = df["begin_upb"] * df["cpr_obs"]
    df["w_x"] = df["begin_upb"] * df["c_over_r"]
    cohort_fit = (
        df.groupby(["age_quarters", "season", "c_over_r_bin"], observed=True)
        .agg(begin_upb=("begin_upb", "sum"), w_obs=("w_obs", "sum"), w_x=("w_x", "sum"))
        .reset_index()
    )
    cohort_fit["cpr_obs"] = cohort_fit["w_obs"] / cohort_fit["begin_upb"]
    cohort_fit["c_over_r"] = cohort_fit["w_x"] / cohort_fit["begin_upb"]
    cohort_fit["market_rate"] = 0.05

    seasonality = fit_seasonality(
        cohort_fit,
        age_min=args.seasonality_age_min,
        age_max=args.seasonality_age_max,
        weight_col="begin_upb",
        value_col="cpr_obs",
        periods_per_year=4,
    )

    refi = fit_refinancing_incentive(
        cohort_fit,
        seasonality,
        age_min=args.refi_age_min,
        age_max=args.refi_age_max,
        weight_col="begin_upb",
        value_col="cpr_obs",
        x_min=args.refi_x_min,
        x_max=args.refi_x_max,
        n_segments=args.refi_segments,
        periods_per_year=4,
    )

    season_factor = seasonality(df["season"].astype(int).to_numpy())
    rho_factor = refi(df["c_over_r"].to_numpy(dtype=float))
    df["aging_factor_raw"] = df["cpr_obs"] / (season_factor * rho_factor)
    df["aging_factor"] = np.clip(df["aging_factor_raw"], 0.0, 1.0)

    seasoning = fit_seasoning_curve(
        cohort_fit.assign(
            aging_factor=np.clip(
                cohort_fit["cpr_obs"].to_numpy(dtype=float)
                / (seasonality(cohort_fit["season"].to_numpy(dtype=int)) * refi(cohort_fit["c_over_r"].to_numpy(dtype=float))),
                0.0,
                1.0,
            )
        ),
        value_col="aging_factor",
        age_max=args.seasoning_age_max,
        weight_col="begin_upb",
        clip=(0.0, 1.0),
    )

    burnout_max = int(df["age_quarters"].max()) if args.burnout_age_max is None else int(args.burnout_age_max)
    burnout = fit_burnout_curve(
        cohort_fit.assign(
            aging_factor=np.clip(
                cohort_fit["cpr_obs"].to_numpy(dtype=float)
                / (seasonality(cohort_fit["season"].to_numpy(dtype=int)) * refi(cohort_fit["c_over_r"].to_numpy(dtype=float))),
                0.0,
                1.0,
            )
        ),
        value_col="aging_factor",
        age_min=args.burnout_age_min,
        age_max=burnout_max,
        weight_col="begin_upb",
        clip=(0.0, 1.0),
    )

    age_q = df["age_quarters"].to_numpy(dtype=float)
    df["cpr_pred"] = season_factor * rho_factor * seasoning(age_q) * burnout(age_q)
    loan_level_pred_metrics = _loan_level_predicted_prepayment_metrics(df)

    obs = df["cpr_obs"].to_numpy(dtype=float)
    pred = df["cpr_pred"].to_numpy(dtype=float)
    w = df["begin_upb"].to_numpy(dtype=float)

    r2_obs_level = weighted_r2(obs, pred, w)
    mean_obs = float(np.sum(w * obs) / np.sum(w))
    mean_pred = float(np.sum(w * pred) / np.sum(w))
    rmse = float(np.sqrt(np.sum(w * (obs - pred) ** 2) / np.sum(w)))

    df["w_pred"] = df["begin_upb"] * df["cpr_pred"]
    cohort = (
        df.groupby(["age_quarters", "season", "c_over_r_bin"], observed=True)
        .agg(w=("begin_upb", "sum"), w_obs=("w_obs", "sum"), w_pred=("w_pred", "sum"))
        .reset_index(drop=True)
    )
    cohort["obs"] = cohort["w_obs"] / cohort["w"]
    cohort["pred"] = cohort["w_pred"] / cohort["w"]

    r2_cohort_level = weighted_r2(cohort["obs"].to_numpy(), cohort["pred"].to_numpy(), cohort["w"].to_numpy())
    rmse_cohort = float(
        np.sqrt(
            np.sum(cohort["w"].to_numpy() * (cohort["obs"].to_numpy() - cohort["pred"].to_numpy()) ** 2)
            / np.sum(cohort["w"].to_numpy())
        )
    )

    metrics = {
        "weighted_mean_cpr_obs": mean_obs,
        "weighted_mean_cpr_pred": mean_pred,
        "weighted_rmse_obs_level": rmse,
        "weighted_rmse_cohort_level": rmse_cohort,
        "weighted_r2_obs_level": r2_obs_level,
        "weighted_r2_cohort_level": r2_cohort_level,
        "n_cohorts": int(len(cohort)),
        "n_obs": int(len(df)),
        "prepay_events": int(df["is_prepay"].sum()),
        "rate_scale_note": rates_series.scale_note,
        **loan_level_metrics,
        **loan_level_pred_metrics,
        **logit_metrics,
    }

    factors_json = {
        "seasonality": seasonality.factors,
        "refinancing": {
            "knots": refi.knots.tolist(),
            "coef": refi.coef.tolist(),
            "normalize_at": refi.normalize_at,
            "x_min": float(refi.knots[0]),
            "x_max": float(refi.knots[-1]),
        },
        "seasoning": {"t_knots": seasoning.t_knots.tolist(), "alpha": seasoning.alpha.tolist()},
        "burnout": {"t_knots": burnout.t_knots.tolist(), "alpha": burnout.alpha.tolist(), "beta": burnout.beta},
    }

    (out_dir / "factors.json").write_text(json.dumps(factors_json, indent=2, sort_keys=True))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    save_seasonality_plot(seasonality, out_path=plots_dir / "seasonality.png")
    save_refinancing_plot(refi, out_path=plots_dir / "refinancing.png")

    df_age = _weighted_mean_by_age(cohort_fit.assign(aging_factor=np.clip(
        cohort_fit["cpr_obs"].to_numpy(dtype=float)
        / (seasonality(cohort_fit["season"].to_numpy(dtype=int)) * refi(cohort_fit["c_over_r"].to_numpy(dtype=float))),
        0.0,
        1.0,
    )), value_col="aging_factor", weight_col="begin_upb")
    save_aging_plot(df_age, out_path=plots_dir / "aging.png", seasoning=seasoning, burnout=burnout)
    df_age.to_csv(out_dir / "aging_by_age.csv", index=False)

    report = f"""# Prepayment factor-model replication

## Summary

- Observations: {metrics['n_obs']:,}
- Cohorts (age × season × C/R bin): {metrics['n_cohorts']:,}
- Prepay events: {metrics['prepay_events']:,}
- Loans: {metrics['loan_level_n_loans']:,} (terminated: {metrics['loan_level_n_terminated']}, active: {metrics['loan_level_n_active']})
- Loan-level prepay rate (terminated loans): {_format_percent(metrics['loan_level_prepay_rate_terminated'])}
- Loan-level predicted prepay rate (paper model, terminated loans): {_format_percent(metrics['loan_level_predicted_prepay_rate_terminated'])}
- Logit predicted loan-level prepay rate (all loans): {_format_percent(metrics.get('logit_predicted_loan_prepay_rate_all_loans'))}
- Weighted mean CPR (obs): {metrics['weighted_mean_cpr_obs']:.6f}
- Weighted mean CPR (pred): {metrics['weighted_mean_cpr_pred']:.6f}
- Weighted RMSE (obs-level): {metrics['weighted_rmse_obs_level']:.6f}
- Weighted RMSE (cohort-level): {metrics['weighted_rmse_cohort_level']:.6f}
- Weighted R² (obs-level): {metrics['weighted_r2_obs_level']:.6f}
- Weighted R² (cohort-level): {metrics['weighted_r2_cohort_level']:.6f}

## Notes

- Rates: {metrics['rate_scale_note']}
- Maturity payoff counted as prepay: {bool(args.treat_maturity_payoff_as_prepay)}

## Plots

### Seasonality
![](plots/seasonality.png)

### Refinancing incentive
![](plots/refinancing.png)

### Aging (residual after seasonality + refi)
![](plots/aging.png)
"""
    (out_dir / "report.md").write_text(report)

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

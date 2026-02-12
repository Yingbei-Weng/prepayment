from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
from .plots import save_aging_plot, save_refinancing_plot, save_seasonality_plot


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replicate Kang & Zenios (1992) factor-style prepayment model.")
    p.add_argument("--data", required=True, help="Path to data.csv")
    p.add_argument("--rates", required=True, help="Path to 10y_yahoo_quarter_avg.csv")
    p.add_argument("--out", default="outputs", help="Output directory")
    p.add_argument("--sample-rows", type=int, default=None, help="Load only the first N rows of data.csv")

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

    df = df[(df["begin_upb"] > 0) & np.isfinite(df["begin_upb"])].copy()
    df = df[np.isfinite(df["c_over_r"]) & np.isfinite(df["market_rate"]) & df["age_quarters"].notna()].copy()

    smm = (df["prepay_upb"] / df["begin_upb"]).to_numpy(dtype=float)
    df["cpr_obs"] = _annualized_cpr_from_smm(smm, periods_per_year=4)

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

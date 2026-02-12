import numpy as np
import pandas as pd

from prepayment.factors import (
    _annualized_cpr_from_smm,
    fit_burnout_curve,
    fit_refinancing_incentive,
    fit_seasonality,
    fit_seasoning_curve,
    weighted_r2,
)


def test_factor_pipeline_recovers_signal() -> None:
    rng = np.random.default_rng(0)
    n = 15_000

    age = rng.integers(0, 61, size=n)
    season = rng.integers(1, 5, size=n)
    c_over_r = rng.uniform(0.6, 1.5, size=n)
    begin_upb = rng.lognormal(mean=16.0, sigma=0.4, size=n)

    s_true = {1: 0.95, 2: 0.9, 3: 1.1, 4: 1.05}
    season_factor = np.asarray([s_true[int(s)] for s in season], dtype=float)

    rho_true = 0.6 + 1.8 * np.clip(c_over_r - 0.9, 0.0, None)
    rho_true = np.clip(rho_true, 0.6, 2.2)

    seasoning_true = np.clip(age / 12.0, 0.0, 1.0)
    burnout_true = np.exp(-0.05 * np.clip(age - 30.0, 0.0, None))

    cpr_true = np.clip(season_factor * rho_true * seasoning_true * burnout_true, 0.0, 1.0)

    smm = 1.0 - np.power(1.0 - cpr_true, 1.0 / 4.0)
    prepay_upb = smm * begin_upb

    df = pd.DataFrame(
        {
            "age_quarters": age,
            "season": season,
            "c_over_r": c_over_r,
            "begin_upb": begin_upb,
            "prepay_upb": prepay_upb,
            "market_rate": 0.05,
        }
    )

    seasonality = fit_seasonality(df, age_min=8, age_max=50, periods_per_year=4)
    refi = fit_refinancing_incentive(
        df,
        seasonality,
        age_min=8,
        age_max=50,
        x_min=0.6,
        x_max=1.5,
        n_segments=15,
        periods_per_year=4,
    )

    smm_obs = (df["prepay_upb"] / df["begin_upb"]).to_numpy(dtype=float)
    cpr_obs = _annualized_cpr_from_smm(smm_obs, periods_per_year=4)
    s_hat = seasonality(df["season"].to_numpy())
    rho_hat = refi(df["c_over_r"].to_numpy())
    aging = np.clip(cpr_obs / (s_hat * rho_hat), 0.0, 1.0)
    df["aging"] = aging

    seasoning = fit_seasoning_curve(df, value_col="aging", age_max=16, weight_col="begin_upb")
    burnout = fit_burnout_curve(df, value_col="aging", age_min=20, age_max=60, weight_col="begin_upb")

    cpr_pred = s_hat * rho_hat * seasoning(df["age_quarters"].to_numpy()) * burnout(df["age_quarters"].to_numpy())
    score = weighted_r2(cpr_true, cpr_pred, begin_upb)

    assert score > 0.85


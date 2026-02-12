from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")
    s = float(np.sum(weights))
    if s <= 0 or not np.isfinite(s):
        return float("nan")
    return float(np.sum(weights * values) / s)


def _annualized_cpr_from_smm(smm: np.ndarray, periods_per_year: int) -> np.ndarray:
    smm = np.asarray(smm, dtype=float)
    return 1.0 - np.power(1.0 - np.clip(smm, 0.0, 1.0), periods_per_year)


def _refi_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 1 or knots.size < 2:
        raise ValueError("knots must be 1D with at least 2 values")

    deltas = knots[1:] - knots[:-1]
    if np.any(deltas <= 0):
        raise ValueError("knots must be strictly increasing")

    scaled = (x[:, None] - knots[:-1][None, :]) / deltas[None, :]
    return np.clip(scaled, 0.0, 1.0)


def _seasoning_basis(age: np.ndarray, t_knots: np.ndarray) -> np.ndarray:
    age = np.asarray(age, dtype=float)
    t_knots = np.asarray(t_knots, dtype=float)
    if np.any(t_knots <= 0):
        raise ValueError("t_knots must be positive")
    scaled = age[:, None] / t_knots[None, :]
    return np.clip(scaled, 0.0, 1.0)


def _softmax(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    u = u - np.max(u)
    ex = np.exp(u)
    return ex / np.sum(ex)


@dataclass(frozen=True)
class Seasonality:
    factors: dict[int, float]

    def __call__(self, season: Iterable[int] | int) -> np.ndarray:
        if isinstance(season, int):
            return np.asarray(self.factors.get(season, 1.0), dtype=float)
        return np.asarray([self.factors.get(int(s), 1.0) for s in season], dtype=float)


@dataclass(frozen=True)
class RefinancingIncentive:
    knots: np.ndarray
    coef: np.ndarray
    normalize_at: float | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        basis = _refi_basis(np.clip(x, self.knots[0], self.knots[-1]), self.knots)
        X = np.concatenate([np.ones((basis.shape[0], 1)), basis], axis=1)
        raw = X @ self.coef

        if self.normalize_at is None:
            return raw

        x0 = float(self.normalize_at)
        basis0 = _refi_basis(np.asarray([np.clip(x0, self.knots[0], self.knots[-1])]), self.knots)
        X0 = np.concatenate([np.ones((1, 1)), basis0], axis=1)
        denom = float((X0 @ self.coef).item())
        if denom <= 0 or not np.isfinite(denom):
            return raw
        return raw / denom


@dataclass(frozen=True)
class SeasoningCurve:
    t_knots: np.ndarray
    alpha: np.ndarray

    def __call__(self, age: np.ndarray) -> np.ndarray:
        basis = _seasoning_basis(np.asarray(age, dtype=float), self.t_knots)
        return basis @ self.alpha


def _burnout_basis(age: np.ndarray, t_knots: np.ndarray, beta: float) -> np.ndarray:
    age = np.asarray(age, dtype=float)
    t_knots = np.asarray(t_knots, dtype=float)
    dt = age[:, None] - t_knots[None, :]
    out = np.ones_like(dt, dtype=float)
    mask = dt > 0.0
    out[mask] = np.exp(-beta * dt[mask])
    return out


@dataclass(frozen=True)
class BurnoutCurve:
    t_knots: np.ndarray
    alpha: np.ndarray
    beta: float

    def __call__(self, age: np.ndarray) -> np.ndarray:
        basis = _burnout_basis(np.asarray(age, dtype=float), self.t_knots, self.beta)
        return basis @ self.alpha


def fit_seasonality(
    df: pd.DataFrame,
    *,
    age_min: int,
    age_max: int,
    weight_col: str = "begin_upb",
    value_col: str | None = None,
    periods_per_year: int = 4,
) -> Seasonality:
    data = df.copy()
    data = data[(data["age_quarters"] >= age_min) & (data["age_quarters"] <= age_max)]
    data = data[(data[weight_col] > 0) & np.isfinite(data[weight_col])].copy()

    if value_col is None:
        smm = (data["prepay_upb"] / data[weight_col]).to_numpy(dtype=float)
        y = _annualized_cpr_from_smm(smm, periods_per_year=periods_per_year)
    else:
        y = data[value_col].to_numpy(dtype=float)
    w = data[weight_col].to_numpy(dtype=float)

    overall = _weighted_mean(y, w)
    if not np.isfinite(overall) or overall <= 0:
        raise ValueError("Cannot fit seasonality: overall prepayment rate is not positive")

    factors: dict[int, float] = {}
    for season, group in data.groupby("season", observed=True):
        if value_col is None:
            smm_g = (group["prepay_upb"] / group[weight_col]).to_numpy(dtype=float)
            y_g = _annualized_cpr_from_smm(smm_g, periods_per_year=periods_per_year)
        else:
            y_g = group[value_col].to_numpy(dtype=float)
        w_g = group[weight_col].to_numpy(dtype=float)
        mean_g = _weighted_mean(y_g, w_g)
        factor = float(mean_g / overall) if np.isfinite(mean_g) else 1.0
        if not np.isfinite(factor) or factor <= 0:
            factor = 1e-6
        factors[int(season)] = factor

    return Seasonality(factors=factors)


def fit_refinancing_incentive(
    df: pd.DataFrame,
    seasonality: Seasonality,
    *,
    age_min: int,
    age_max: int,
    weight_col: str = "begin_upb",
    value_col: str | None = None,
    x_min: float = 0.5,
    x_max: float = 1.6,
    n_segments: int = 22,
    normalize_at: float | None = None,
    periods_per_year: int = 4,
) -> RefinancingIncentive:
    data = df.copy()
    data = data[(data["age_quarters"] >= age_min) & (data["age_quarters"] <= age_max)]
    data = data[(data[weight_col] > 0) & np.isfinite(data[weight_col])].copy()
    data = data[np.isfinite(data["c_over_r"]) & np.isfinite(data["market_rate"])].copy()

    if value_col is None:
        smm = (data["prepay_upb"] / data[weight_col]).to_numpy(dtype=float)
        cpr = _annualized_cpr_from_smm(smm, periods_per_year=periods_per_year)
    else:
        cpr = data[value_col].to_numpy(dtype=float)
    cpr_sa = cpr / seasonality(data["season"].to_numpy())

    x = np.clip(data["c_over_r"].to_numpy(dtype=float), x_min, x_max)
    w = data[weight_col].to_numpy(dtype=float)

    knots = np.linspace(x_min, x_max, n_segments + 1)
    basis = _refi_basis(x, knots)
    X = np.concatenate([np.ones((basis.shape[0], 1)), basis], axis=1)

    sw = np.sqrt(np.clip(w, 0.0, np.inf))
    Xw = X * sw[:, None]
    yw = cpr_sa * sw

    res = lsq_linear(Xw, yw, bounds=(0.0, np.inf), lsmr_tol="auto")
    coef = res.x
    return RefinancingIncentive(knots=knots, coef=coef, normalize_at=normalize_at)


def _weighted_mean_by_age(
    df: pd.DataFrame,
    *,
    value_col: str,
    weight_col: str,
    age_col: str = "age_quarters",
) -> pd.DataFrame:
    work = df[[age_col, value_col, weight_col]].copy()
    work["wy"] = work[value_col].to_numpy(dtype=float) * work[weight_col].to_numpy(dtype=float)

    grouped = work.groupby(age_col, observed=True)
    sum_w = grouped[weight_col].sum(min_count=1)
    sum_wy = grouped["wy"].sum(min_count=1)
    count = grouped.size()

    out = pd.DataFrame(
        {
            "age": sum_w.index.astype(int),
            "value": (sum_wy / sum_w).to_numpy(dtype=float),
            "weight": sum_w.to_numpy(dtype=float),
            "count": count.to_numpy(dtype=int),
        }
    )
    return out.sort_values("age", kind="mergesort")


def fit_seasoning_curve(
    df: pd.DataFrame,
    *,
    value_col: str,
    weight_col: str = "begin_upb",
    age_max: int,
    clip: tuple[float, float] = (0.0, 1.0),
) -> SeasoningCurve:
    data = df[["age_quarters", value_col, weight_col]].copy()
    data = data[(data["age_quarters"] >= 0) & (data["age_quarters"] <= age_max)]
    data = data[(data[weight_col] > 0) & np.isfinite(data[weight_col])].copy()
    data = data[np.isfinite(data[value_col])].copy()

    y = np.clip(data[value_col].to_numpy(dtype=float), clip[0], clip[1])
    w = data[weight_col].to_numpy(dtype=float)

    df_age = _weighted_mean_by_age(
        pd.DataFrame({"age_quarters": data["age_quarters"], "y": y, "w": w}),
        value_col="y",
        weight_col="w",
    )
    df_age = df_age[df_age["weight"] > 0].copy()

    age = df_age["age"].to_numpy(dtype=float)
    y_age = df_age["value"].to_numpy(dtype=float)
    w_age = df_age["weight"].to_numpy(dtype=float)

    t_knots = np.arange(1, int(age_max) + 1, dtype=float)
    basis = _seasoning_basis(age, t_knots)

    def obj_and_grad(u: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = _softmax(u)
        y_pred = basis @ alpha
        resid = y_pred - y_age
        f = float(np.sum(w_age * resid * resid))

        g_alpha = 2.0 * (basis.T @ (w_age * resid))
        g_u = alpha * (g_alpha - float(g_alpha @ alpha))
        return f, g_u

    u0 = np.zeros(len(t_knots), dtype=float)
    res = minimize(lambda u: obj_and_grad(u)[0], u0, jac=lambda u: obj_and_grad(u)[1], method="L-BFGS-B")
    alpha = _softmax(res.x)
    return SeasoningCurve(t_knots=t_knots, alpha=alpha)


def fit_burnout_curve(
    df: pd.DataFrame,
    *,
    value_col: str,
    weight_col: str = "begin_upb",
    age_min: int,
    age_max: int,
    clip: tuple[float, float] = (0.0, 1.0),
) -> BurnoutCurve:
    data = df[["age_quarters", value_col, weight_col]].copy()
    data = data[(data["age_quarters"] >= age_min) & (data["age_quarters"] <= age_max)]
    data = data[(data[weight_col] > 0) & np.isfinite(data[weight_col])].copy()
    data = data[np.isfinite(data[value_col])].copy()

    y = np.clip(data[value_col].to_numpy(dtype=float), clip[0], clip[1])
    w = data[weight_col].to_numpy(dtype=float)

    df_age = _weighted_mean_by_age(
        pd.DataFrame({"age_quarters": data["age_quarters"], "y": y, "w": w}),
        value_col="y",
        weight_col="w",
    )
    df_age = df_age[df_age["weight"] > 0].copy()

    age = df_age["age"].to_numpy(dtype=float)
    y_age = df_age["value"].to_numpy(dtype=float)
    w_age = df_age["weight"].to_numpy(dtype=float)

    t_knots = np.arange(int(age_min), int(age_max) + 1, dtype=float)

    def obj_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        u = theta[:-1]
        v = theta[-1]
        alpha = _softmax(u)
        beta = float(np.exp(v))

        basis = _burnout_basis(age, t_knots, beta)
        y_pred = basis @ alpha
        resid = y_pred - y_age
        f = float(np.sum(w_age * resid * resid))

        g_alpha = 2.0 * (basis.T @ (w_age * resid))
        g_u = alpha * (g_alpha - float(g_alpha @ alpha))

        dt = age[:, None] - t_knots[None, :]
        d_basis_dbeta = np.zeros_like(dt, dtype=float)
        mask = dt > 0.0
        d_basis_dbeta[mask] = -(dt[mask]) * np.exp(-beta * dt[mask])
        dy_dbeta = d_basis_dbeta @ alpha
        g_beta = float(2.0 * np.sum(w_age * resid * dy_dbeta))
        g_v = g_beta * beta

        g = np.concatenate([g_u, np.asarray([g_v])])
        return f, g

    theta0 = np.zeros(len(t_knots) + 1, dtype=float)
    theta0[-1] = np.log(0.05)
    res = minimize(
        lambda th: obj_and_grad(th)[0],
        theta0,
        jac=lambda th: obj_and_grad(th)[1],
        method="L-BFGS-B",
        bounds=[(None, None)] * len(t_knots) + [(-20.0, 20.0)],
    )
    alpha = _softmax(res.x[:-1])
    beta = float(np.exp(res.x[-1]))
    return BurnoutCurve(t_knots=t_knots, alpha=alpha, beta=beta)


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(w, dtype=float)
    y_bar = _weighted_mean(y_true, w)
    ss_res = float(np.sum(w * (y_true - y_pred) ** 2))
    ss_tot = float(np.sum(w * (y_true - y_bar) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot

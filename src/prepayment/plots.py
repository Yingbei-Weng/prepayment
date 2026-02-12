from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .factors import BurnoutCurve, RefinancingIncentive, Seasonality, SeasoningCurve


def save_seasonality_plot(seasonality: Seasonality, *, out_path: str | Path) -> None:
    seasons = sorted(seasonality.factors.keys())
    values = [seasonality.factors[s] for s in seasons]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([str(s) for s in seasons], values, color="#4C78A8")
    ax.axhline(1.0, color="black", linewidth=1, alpha=0.7)
    ax.set_title("Seasonality factor (quarter-of-year)")
    ax.set_xlabel("Season (quarter)")
    ax.set_ylabel("Factor (relative to overall mean)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_refinancing_plot(refi: RefinancingIncentive, *, out_path: str | Path) -> None:
    x = np.linspace(refi.knots[0], refi.knots[-1], 300)
    y = refi(x)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, y, color="#F58518")
    ax.axvline(1.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("Refinancing incentive vs C/R")
    ax.set_xlabel("C/R (contract / market)")
    if refi.normalize_at is None:
        ax.set_ylabel("Estimated CPR component")
    else:
        ax.set_ylabel(f"Factor (normalized at C/R={refi.normalize_at:g})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_aging_plot(
    df_age: pd.DataFrame,
    *,
    out_path: str | Path,
    seasoning: SeasoningCurve | None = None,
    burnout: BurnoutCurve | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_age["age"], df_age["value"], label="Observed (weighted mean)", color="#54A24B")

    if seasoning is not None:
        ax.plot(df_age["age"], seasoning(df_age["age"].to_numpy()), label="Seasoning fit", color="#4C78A8")
    if burnout is not None:
        ax.plot(df_age["age"], burnout(df_age["age"].to_numpy()), label="Burnout fit", color="#F58518")
    if seasoning is not None and burnout is not None:
        ax.plot(
            df_age["age"],
            seasoning(df_age["age"].to_numpy()) * burnout(df_age["age"].to_numpy()),
            label="Seasoning × burnout",
            color="#B279A2",
        )

    ax.set_title("Aging component (seasonality & refi removed)")
    ax.set_xlabel("Age (quarters)")
    ax.set_ylabel("Residual factor")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

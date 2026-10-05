"""
demand_model.py
----------------
Predicts relative delivery demand per LSOA from socio-demographic features.

Two models are trained so there's always a quantified comparison, not just
a single number with no context:
  - baseline  : demand is proportional to population alone (naive)
  - model     : a regression using population + IMD + employment rate

"Demand" itself isn't directly observable in open data, so this uses a
PROXY target built from a transparent formula (see `_synthetic_demand_proxy`)
— documented explicitly so you can discuss its limitations honestly in your
report's Evaluation & Reflection section. Once you have access to any real
delivery/footfall data (even a partial proxy, e.g. retail density), swap the
target in `train_and_evaluate` for that instead.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_sources import load_lsoa_data

FEATURES = ["population", "imd_score", "employment_rate"]


def _synthetic_demand_proxy(df: pd.DataFrame, seed: int = 7) -> pd.Series:
    """Builds a plausible 'true' demand signal for development purposes.

    Assumption encoded here (state this explicitly in your report): more
    populous, more deprived, lower-employment areas generate relatively
    higher per-capita online delivery demand (less car ownership, more
    reliance on delivery for access to goods). This is a simplification
    worth challenging/citing against real literature once you have it.
    """
    rng = np.random.default_rng(seed)
    base = (
        0.6 * (df["population"] / df["population"].max())
        + 0.3 * (df["imd_score"] / df["imd_score"].max())
        + 0.1 * (1 - df["employment_rate"])
    )
    noise = rng.normal(0, 0.05, len(df))
    demand = np.clip(base + noise, 0, None)
    return demand * 1000  # arbitrary scale: "deliveries per week" proxy


def train_and_evaluate(df: pd.DataFrame | None = None, seed: int = 7) -> dict:
    """Trains the baseline and the regression model, returns metrics + predictions."""
    if df is None:
        df = load_lsoa_data(synthetic=False)

    df = df.copy()
    df["demand"] = _synthetic_demand_proxy(df, seed=seed)

    X = df[FEATURES]
    y = df["demand"]
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=seed
    )

    # --- Baseline: demand scaled purely by population share ---
    baseline_pred = (X_test["population"] / X["population"].sum()) * y.sum()

    # --- Model: linear regression on all features ---
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)

    results = {
        "baseline_mae": mean_absolute_error(y_test, baseline_pred),
        "baseline_r2": r2_score(y_test, baseline_pred),
        "model_mae": mean_absolute_error(y_test, model_pred),
        "model_r2": r2_score(y_test, model_pred),
        "model": model,
        "feature_importance": dict(zip(FEATURES, model.coef_)),
        "df_with_demand": df,
    }
    return results


def predict_demand_all(df: pd.DataFrame, model: LinearRegression) -> pd.DataFrame:
    """Applies a trained model to every LSOA, returning df with a predicted_demand column."""
    df = df.copy()
    df["predicted_demand"] = model.predict(df[FEATURES])
    return df


if __name__ == "__main__":
    results = train_and_evaluate()
    print("Baseline (population-only):")
    print(f"  MAE = {results['baseline_mae']:.1f}   R² = {results['baseline_r2']:.3f}")
    print("Model (population + IMD + employment):")
    print(f"  MAE = {results['model_mae']:.1f}   R² = {results['model_r2']:.3f}")
    improvement = (1 - results["model_mae"] / results["baseline_mae"]) * 100
    print(f"\nModel reduces error vs. baseline by {improvement:.1f}%")
    print(f"\nFeature coefficients: {results['feature_importance']}")

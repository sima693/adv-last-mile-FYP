"""
depot_siting.py
----------------
Chooses depot/charging-station locations to minimise the demand-weighted
distance vehicles have to travel.

This is a simplified p-median problem: choose k locations from a candidate
set (here, LSOA centroids) so that the sum of (demand * distance to nearest
chosen depot) is minimised. A greedy approximation is used rather than an
exact solver — fast, easy to explain in a viva, and "good enough" is a
defensible, citable design decision (exact p-median is NP-hard; greedy is a
standard, well-documented approximation you can reference in your report).

A naive baseline (depots placed at the k highest-population LSOAs, ignoring
distance) is included so you can quantify the improvement the optimisation
actually buys you.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between arrays of points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _distance_matrix_km(df: pd.DataFrame) -> np.ndarray:
    """All-pairs haversine distance matrix (km) between LSOA centroids."""
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    n = len(df)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = _haversine_km(lat[i], lon[i], lat, lon)
    return D


def naive_baseline_depots(df: pd.DataFrame, k: int) -> list[int]:
    """Baseline: pick the k highest-population LSOAs as depots (ignores distance)."""
    return df.sort_values("population", ascending=False).index[:k].tolist()


def greedy_pmedian_depots(df: pd.DataFrame, k: int, demand_col: str = "predicted_demand") -> list[int]:
    """Greedily choose k depot locations to minimise demand-weighted distance.

    At each step, adds the candidate location that most reduces total
    demand-weighted distance to the nearest chosen depot, given what's
    already been chosen.
    """
    D = _distance_matrix_km(df)
    demand = df[demand_col].to_numpy()
    n = len(df)
    chosen: list[int] = []
    remaining = set(range(n))
    current_min_dist = np.full(n, np.inf)

    for _ in range(k):
        best_candidate, best_cost = None, np.inf
        for cand in remaining:
            new_min_dist = np.minimum(current_min_dist, D[cand])
            cost = np.sum(demand * new_min_dist)
            if cost < best_cost:
                best_cost, best_candidate = cost, cand
        chosen.append(best_candidate)
        remaining.remove(best_candidate)
        current_min_dist = np.minimum(current_min_dist, D[best_candidate])

    return chosen


def demand_weighted_avg_distance(df: pd.DataFrame, depot_indices: list[int], demand_col: str = "predicted_demand") -> float:
    """Evaluation metric: average distance each unit of demand travels to its nearest depot."""
    D = _distance_matrix_km(df)
    demand = df[demand_col].to_numpy()
    min_dist_to_depot = D[:, depot_indices].min(axis=1)
    return float(np.average(min_dist_to_depot, weights=demand))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.demand_model import train_and_evaluate, predict_demand_all

    results = train_and_evaluate()
    df = predict_demand_all(results["df_with_demand"], results["model"])

    k = 5
    baseline_depots = naive_baseline_depots(df, k)
    optimised_depots = greedy_pmedian_depots(df, k)

    baseline_dist = demand_weighted_avg_distance(df, baseline_depots)
    optimised_dist = demand_weighted_avg_distance(df, optimised_depots)
    improvement = (1 - optimised_dist / baseline_dist) * 100

    print(f"Baseline depots (top-{k} population): avg demand-weighted distance = {baseline_dist:.2f} km")
    print(f"Optimised depots (greedy p-median):   avg demand-weighted distance = {optimised_dist:.2f} km")
    print(f"\nOptimisation reduces average distance by {improvement:.1f}% vs. naive placement")

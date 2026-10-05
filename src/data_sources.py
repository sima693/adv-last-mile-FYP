"""
data_sources.py
----------------
Loads and cleans LSOA-level socio-demographic + spatial data.

Two modes:
  - synthetic=True  : generates realistic fake data so you can build/test
                       the rest of the pipeline immediately.
  - synthetic=False : loads real downloaded files from data/ (see README).

Keeping this behind one function means demand_model.py and depot_siting.py
never need to know or care which mode is active — that's the design
decision you'd justify in your Methodology section.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Columns every downstream module can rely on existing, regardless of source.
REQUIRED_COLUMNS = [
    "lsoa_code",
    "lsoa_name",
    "latitude",
    "longitude",
    "population",
    "imd_score",       # higher = more deprived (ONS/gov.uk convention)
    "employment_rate",
]


def load_lsoa_data(synthetic: bool = False, n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Return a cleaned LSOA-level DataFrame with REQUIRED_COLUMNS.

    Parameters
    ----------
    synthetic : if True, generate fake-but-realistic data for a Birmingham-
                sized area. If False, load real files from data/.
    n         : number of LSOAs to simulate (Birmingham has ~640 real LSOAs;
                150 keeps things fast while you're iterating).
    """
    if synthetic:
        return _generate_synthetic_lsoa_data(n=n, seed=seed)
    return _load_real_lsoa_data()


def _generate_synthetic_lsoa_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Roughly centre on Birmingham city centre, spread ~8km in each direction
    centre_lat, centre_lon = 52.4862, -1.8904
    lat = centre_lat + rng.normal(0, 0.045, n)
    lon = centre_lon + rng.normal(0, 0.07, n)

    # IMD and employment are anti-correlated in reality (more deprived ->
    # lower employment) — bake that relationship in so the synthetic data
    # isn't just noise; it gives the model something real to learn.
    imd_score = rng.uniform(5, 55, n)
    employment_rate = np.clip(0.85 - 0.006 * imd_score + rng.normal(0, 0.04, n), 0.3, 0.95)
    population = rng.integers(800, 3500, n)

    df = pd.DataFrame({
        "lsoa_code": [f"E01{100000 + i}" for i in range(n)],
        "lsoa_name": [f"Birmingham LSOA {i:03d}" for i in range(n)],
        "latitude": lat,
        "longitude": lon,
        "population": population,
        "imd_score": imd_score.round(1),
        "employment_rate": employment_rate.round(3),
    })
    return _validate(df)


def _load_real_lsoa_data() -> pd.DataFrame:
    pop_path = DATA_DIR / "population.csv"
    imd_path = DATA_DIR / "imd.csv"

    missing = [p for p in (pop_path, imd_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Real data mode needs these files (see README 'Swapping in real "
            f"data'): {[str(p) for p in missing]}"
        )

    pop = pd.read_csv(pop_path)
    imd = pd.read_csv(imd_path)

    # Standardize IMD columns
    imd_col_map = {
        "LSOA code (2021)": "lsoa_code",
        "LSOA code (2011)": "lsoa_code",
        "Index of Multiple Deprivation (IMD) Score": "imd_score",
        "IMD Score": "imd_score",
        "IMD_Score": "imd_score",
        "Employment Score (rate)": "employment_deprivation_rate",
        "Employment Rate": "employment_rate",
        "Employment_Rate": "employment_rate",
    }
    imd = imd.rename(columns={c: imd_col_map[c] for c in imd.columns if c in imd_col_map})
    if "employment_rate" not in imd.columns and "employment_deprivation_rate" in imd.columns:
        imd["employment_rate"] = (1.0 - imd["employment_deprivation_rate"]).round(3)

    # Standardize population columns
    pop_col_map = {
        "LSOA_name": "lsoa_name",
        "lat": "latitude",
        "lon": "longitude",
        "all_ages": "population",
    }
    pop = pop.rename(columns={c: pop_col_map[c] for c in pop.columns if c in pop_col_map})

    df = pop.merge(imd, on="lsoa_code", how="inner")
    return _validate(df)


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Loaded data is missing required columns: {missing_cols}")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).drop_duplicates(subset="lsoa_code")
    after = len(df)
    if after < before:
        print(f"[data_sources] dropped {before - after} rows with missing/duplicate data")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    data = load_lsoa_data(synthetic=False)
    print(data.head())
    print(f"\n{len(data)} LSOAs loaded. Columns: {list(data.columns)}")

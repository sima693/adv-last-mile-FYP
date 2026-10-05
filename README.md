# ADV Last-Mile FYP — Starter Project

Predicting last-mile delivery demand and siting depots/charging stations for
autonomous delivery vehicles (ADVs), using socio-demographic + spatial data.

## Structure

```
src/
  data_sources.py   # Load & clean socio-demographic + spatial data (LSOA-level)
  demand_model.py   # Predict relative delivery demand per area
  depot_siting.py   # Optimise depot locations against predicted demand
  evaluate.py       # Compare model/optimisation against baselines
  dashboard.py      # Streamlit app tying it all together
data/
  (put downloaded CSVs / shapefiles here — see "Real data" below)
notebooks/
  (scratch space for exploration — keep final logic in src/)
```

## Quick start

```bash
pip install -r requirements.txt
python -m src.demand_model        # runs end-to-end on synthetic data
streamlit run src/dashboard.py    # view the dashboard
```

Everything currently runs on **synthetic data** generated in
`data_sources.py` (`load_lsoa_data(synthetic=True)`), so you can run the
whole pipeline right now, today, with zero downloads. This is deliberate —
it lets you build and test the *logic* (features, model, optimisation)
immediately, then swap in real data once downloaded, without changing your
model/optimisation code at all.

## Swapping in real data

1. **ONS LSOA population estimates** — download the CSV from ONS, drop it
   in `data/population.csv`
2. **Index of Multiple Deprivation 2025** — download from gov.uk, drop in
   `data/imd.csv`
3. **Road network** — run `osmnx.graph_from_place("Birmingham, UK", network_type="drive")`
   locally (needs internet — won't work in a sandboxed environment) and
   cache it with `ox.save_graphml(G, "data/birmingham_roads.graphml")`
4. In `data_sources.py`, call `load_lsoa_data(synthetic=False)` instead —
   it expects the file paths above and will raise a clear error telling you
   what's missing if a file isn't there yet.

## Why this structure

- Every module has one job (data loading / modelling / optimisation /
  evaluation / display) — this maps directly onto your report's
  "Methodology & Design" section: you can point at clean module boundaries
  as a design decision and justify why (separation of concerns, testability).
- `demand_model.py` and `depot_siting.py` both expose an explicit
  **baseline** alongside the "real" model — this is what your "Evaluation &
  Reflection" section needs: a quantified comparison, not just "it seemed
  to work."

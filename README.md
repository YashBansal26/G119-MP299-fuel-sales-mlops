# MP299 — Fuel Station Daily Sales (Dehradun)

SCSE3040 MLOps mini project.

| | |
|---|---|
| Group | **G119** |
| Problem | **MP299** — predict `litres_per_day` (regression, metric: MAE) |
| Faculty | Dr. Pulkit Dwivedi — Batch 59 |

## Members

| Enrolment no. | Name |
|---|---|
| S24CSEU0449 | Sakshi |
| S24CSEU0231 | Yash Bansal |
| S24CSEU1801 | Aryan Bansal |
| S24CSEU0076 | Aaryan |

## Data (tracked with DVC)

`data/raw/train.csv` and `data/raw/later.csv` are tracked by DVC; only their
`.dvc` files are in git. `data/raw/holdout_features.csv` is committed as is.

## Stage 1 — data problems found in `train.csv`

Run from the repository root:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m mp299.data_checks     # Windows PowerShell: $env:PYTHONPATH="src"; python -m mp299.data_checks
pytest -q
```

| Problem | Present | Details |
|---|---|---|
| leaking_column | no | train has exactly the holdout columns + target |
| unit_mixup | no | every numeric column has the same range in City / Highway / Industrial |
| sentinel_value | **yes** | `staff = 999` in 66 rows (real range 3–20) |
| duplicate_rows | no | 0 exact copies, 0 copies apart from `record_id` |
| stuck_sensor | **yes** | `nearby_pumps = 6` on all 100 rows from 2026-04-06 to 2026-04-24 |
| target_typos | **yes** | 12 targets ≥ 101,170 L; next highest is 31,343 L (10× typos) |
| spelling_variants | no | `ownership` and `zone` have only clean levels |

## Stage 2 — cleaning and integration

The cleaning implementation is in `src/mp299/cleaning.py`, with its rules
and output paths in `params.yaml`. It converts the `staff = 999` sentinel
and the known stuck `nearby_pumps` window to missing values, repairs the
configured 10× target typos, and fills missing numeric values with training
medians. The function works on a copy of the input.

Place the unchanged original `train.csv` in `data/raw/`, then run:

```bash
PYTHONPATH=src python -m mp299.cleaning
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m mp299_ops.deployment
```

The cleaning command writes `data/processed/train_clean.csv` and
`data/processed/train_medians.json`; it does not overwrite the raw CSV.
The deployment checker validates the prepared Kubernetes configuration,
rather than starting an API or a cluster.

### Handoff and verified status — 23 November 2026

- **Data lead (Yash):** cleaning implementation and its in-memory tests are
  committed. The tests do not require the original CSVs.
- **Ops review (Aryan):** checked the cleaning entry point, configured output
  paths and the existing push-triggered CI. The
  [CI run for `1635f1a`](https://github.com/YashBansal26/G119-MP299-fuel-sales-mlops/actions/runs/45205615814)
  passed on GitHub. This documents an integration review, not model training.
- **Model lead:** training, three or more tracked MLflow settings, the
  registry/champion artifact and the `tracking` values in
  `reports/findings.yaml` remain to be added. Fit learned preprocessing on
  the training partition only and reuse it for validation/inference.
- **Next Ops check:** when model dependencies and tests arrive, keep library
  versions pinned in `requirements.txt` and verify the combined CI run.
  See [the Ops handoff](docs/ops-plan.md) for Stage 3 integration.

This documentation update was prepared with AI assistance.

## Layout

```
params.yaml            settings read by src/mp299/config.py
src/mp299/             config.py, data_checks.py, explore.py
tests/                 pytest tests (no CSVs needed)
reports/findings.yaml  the marking form
```

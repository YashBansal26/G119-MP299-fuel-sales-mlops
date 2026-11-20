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

## Layout

```
params.yaml            settings read by src/mp299/config.py
src/mp299/             config.py, data_checks.py, explore.py
tests/                 pytest tests (no CSVs needed)
reports/findings.yaml  the marking form
```

"""The seven data checks for train.csv (STUDENT_HANDBOOK section 7).

Every function takes a DataFrame and returns a dict in the same shape as the
`data_issues` block of reports/findings.yaml. Run:

    python -m mp299.data_checks          (from the repo root, with src on PYTHONPATH)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from mp299.config import ROOT, data_path, load_params


# 1. leaking column -------------------------------------------------------
def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame,
                         target: str, id_col: str, corr_limit: float = 0.98) -> dict:
    """A column known only after the event: missing from holdout, or ~equal to target."""
    suspects = [c for c in train.columns if c not in holdout.columns and c != target]
    for c in train.select_dtypes("number").columns:
        if c in (target, id_col) or c in suspects:
            continue
        if abs(train[c].corr(train[target])) >= corr_limit:
            suspects.append(c)
    if suspects:
        return {"present": True, "column": suspects[0]}
    return {"present": False, "column": None}


# 2. unit mix-up ------------------------------------------------------------
def check_unit_mixup(train: pd.DataFrame, numeric: list[str], group_col: str,
                     factors: list[float], tol: float = 0.15,
                     sentinels: list[float] | None = None) -> dict:
    """One group's column is ~10/12/60/100/1000 times the other groups'."""
    sentinels = sentinels or []
    best = None
    for c in numeric:
        col = train[c].where(~train[c].isin(sentinels))      # ignore sentinel typing
        med = col.groupby(train[group_col]).median()
        for g in med.index:
            others = med.drop(g).median()
            if not others or pd.isna(others) or not med[g]:
                continue
            ratio = med[g] / others
            for f in factors:
                if abs(ratio / f - 1) <= tol:
                    if best is None or abs(ratio / f - 1) < best[3]:
                        best = (c, g, f, abs(ratio / f - 1))
    if best:
        return {"present": True, "column": best[0], "group_value": best[1], "factor": best[2]}
    return {"present": False, "column": None, "group_value": None, "factor": None}


# 3. sentinel value -------------------------------------------------------
def check_sentinel_value(train: pd.DataFrame, numeric: list[str],
                         candidates: list[float], min_count: int = 5) -> dict:
    """An impossible number (-1, 999, 9999 ...) repeated many times."""
    found = []
    for c in numeric:
        counts = train[c].value_counts()
        for v in candidates:
            n = int(counts.get(v, 0))
            if n >= min_count:
                found.append((n, c, v))
    if found:
        n, c, v = max(found)
        v = int(v) if float(v).is_integer() else v
        return {"present": True, "column": c, "value": v, "count": n}
    return {"present": False, "column": None, "value": None, "count": None}


# 4. duplicate rows -------------------------------------------------------
def check_duplicate_rows(train: pd.DataFrame, id_col: str) -> dict:
    """Extra copies of a record (exact, or exact apart from record_id)."""
    exact = int(train.duplicated().sum())
    no_id = int(train.drop(columns=[id_col]).duplicated().sum())
    n = max(exact, no_id)
    return {"present": n > 0, "count": n if n > 0 else None}


# 5. stuck sensor ---------------------------------------------------------
def longest_run(train: pd.DataFrame, col: str, date_col: str) -> dict:
    df = train.sort_values([date_col], kind="stable").reset_index(drop=True)
    run_id = (df[col] != df[col].shift()).cumsum()
    runs = df.groupby(run_id).agg(value=(col, "first"), rows=(col, "size"),
                                  start=(date_col, "min"), end=(date_col, "max"))
    top = runs.loc[runs["rows"].idxmax()]
    return {"column": col, "value": top["value"], "rows": int(top["rows"]),
            "start_date": str(top["start"]), "end_date": str(top["end"])}


def check_stuck_sensor(train: pd.DataFrame, numeric: list[str], date_col: str,
                       min_rows: int = 10) -> dict:
    """Same reading on every row for a long run of days."""
    runs = [longest_run(train, c, date_col) for c in numeric]
    top = max(runs, key=lambda r: r["rows"])
    if top["rows"] >= min_rows:
        return {"present": True, "column": top["column"],
                "start_date": top["start_date"], "end_date": top["end_date"]}
    return {"present": False, "column": None, "start_date": None, "end_date": None}


# 6. target typos ---------------------------------------------------------
def check_target_typos(train: pd.DataFrame, target: str, gap_ratio: float = 2.5) -> dict:
    """A small group of targets ~10x too big, with an empty gap below them."""
    y = np.sort(train[target].dropna().to_numpy(dtype=float))
    y = y[y > 0]
    jumps = y[1:] / y[:-1]
    i = int(np.argmax(jumps))
    if jumps[i] >= gap_ratio:
        n = int(len(y) - i - 1)
        return {"present": True, "count": n}
    return {"present": False, "count": None}


# 7. spelling variants ----------------------------------------------------
def normalise_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def check_spelling_variants(train: pd.DataFrame, categorical: list[str]) -> dict:
    """Same category written in different ways ('Rain', 'rain ', 'RAI.')."""
    for c in categorical:
        raw = train[c].dropna().nunique()
        clean = train[c].dropna().map(normalise_text).nunique()
        if raw > clean:
            return {"present": True, "column": c, "true_levels": int(clean)}
    return {"present": False, "column": None, "true_levels": None}


# all seven ---------------------------------------------------------------
def run_all_checks(train: pd.DataFrame, holdout: pd.DataFrame, params: dict) -> dict:
    d, f, k = params["data"], params["features"], params["checks"]
    return {
        "leaking_column": check_leaking_column(train, holdout, d["target"], d["id_col"], k["leak_corr"]),
        "unit_mixup": check_unit_mixup(train, f["numeric"], d["group_col"], k["unit_factors"],
                                       k["unit_tolerance"], k["sentinel_candidates"]),
        "sentinel_value": check_sentinel_value(train, f["numeric"], k["sentinel_candidates"],
                                               k["sentinel_min_count"]),
        "duplicate_rows": check_duplicate_rows(train, d["id_col"]),
        "stuck_sensor": check_stuck_sensor(train, f["numeric"], d["date_col"], k["stuck_min_rows"]),
        "target_typos": check_target_typos(train, d["target"], k["typo_gap_ratio"]),
        "spelling_variants": check_spelling_variants(train, f["categorical"]),
    }


def main() -> None:
    params = load_params()
    train = pd.read_csv(data_path("train", params))
    holdout = pd.read_csv(data_path("holdout", params))
    issues = run_all_checks(train, holdout, params)
    out = ROOT / params["reports"]["data_issues"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(issues, indent=2, default=str), encoding="utf-8")
    print(json.dumps(issues, indent=2, default=str))


if __name__ == "__main__":
    main()


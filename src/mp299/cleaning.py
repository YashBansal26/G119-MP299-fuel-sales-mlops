"""Clean train.csv: fix the data problems found in Stage 1.

Fixes, in order (settings in params.yaml -> cleaning):
  1. spelling   strip spaces in text columns (no variants found, kept as a guard)
  2. duplicates drop extra copies of a record (none found, kept as a guard)
  3. sentinel   staff == 999 -> missing
  4. stuck      nearby_pumps on 2026-04-06 .. 2026-04-24 -> missing
  5. typos      litres_per_day above the threshold -> divided by 10
  6. impute     missing numeric values -> median of the cleaned training data

Run from the repo root:   python -m mp299.cleaning
"""
from __future__ import annotations

import json

import pandas as pd

from mp299.config import ROOT, data_path, load_params


def fix_text(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    for c in categorical:
        df[c] = df[c].astype(str).str.strip()
    return df


def drop_duplicates(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    return df.drop_duplicates(subset=[c for c in df.columns if c != id_col]).reset_index(drop=True)


def fix_sentinels(df: pd.DataFrame, sentinels: dict) -> pd.DataFrame:
    for col, value in (sentinels or {}).items():
        if col in df.columns:
            df[col] = df[col].mask(df[col] == value)
    return df


def fix_stuck_sensor(df: pd.DataFrame, stuck: dict, date_col: str) -> pd.DataFrame:
    if not stuck or stuck.get("column") not in df.columns:
        return df
    in_window = df[date_col].between(stuck["start_date"], stuck["end_date"])
    df[stuck["column"]] = df[stuck["column"]].mask(in_window)
    return df


def fix_target_typos(df: pd.DataFrame, target: str, threshold: float, factor: float) -> pd.DataFrame:
    if target in df.columns:
        too_big = df[target] > threshold
        df.loc[too_big, target] = df.loc[too_big, target] / factor
    return df


def impute(df: pd.DataFrame, numeric: list[str], medians: dict | None = None):
    """Fill missing numeric values. Medians come from the training data;
    pass them in to clean other files the same way."""
    if medians is None:
        medians = {c: float(df[c].median()) for c in numeric}
    for c in numeric:
        df[c] = df[c].fillna(medians[c])
    for c in ("nearby_pumps", "staff"):          # counts stay whole numbers
        if c in df.columns:
            df[c] = df[c].round().astype(int)
    return df, medians


def clean(df: pd.DataFrame, params: dict, medians: dict | None = None):
    """Return (cleaned copy of df, medians used for imputation)."""
    d, f, c = params["data"], params["features"], params["cleaning"]
    df = df.copy()
    df = fix_text(df, f["categorical"])
    df = drop_duplicates(df, d["id_col"])
    df = fix_sentinels(df, c["sentinels"])
    df = fix_stuck_sensor(df, c["stuck_sensor"], d["date_col"])
    df = fix_target_typos(df, d["target"], c["target_typo_threshold"], c["target_typo_factor"])
    return impute(df, f["numeric"], medians)


def main() -> None:
    params = load_params()
    raw = pd.read_csv(data_path("train", params))
    cleaned, medians = clean(raw, params)

    out = ROOT / params["cleaning"]["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out, index=False)
    (ROOT / params["cleaning"]["medians"]).write_text(json.dumps(medians, indent=2))

    t = params["data"]["target"]
    print(f"rows: {len(raw)} -> {len(cleaned)}")
    print(f"missing after cleaning: {int(cleaned.isna().sum().sum())}")
    print(f"{t}: max {raw[t].max():.0f} -> {cleaned[t].max():.0f}")
    print(f"staff max: {raw['staff'].max()} -> {cleaned['staff'].max()}")
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

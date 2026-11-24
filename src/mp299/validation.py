"""Data contract for MP299 input data.

Checks that a batch of feature rows (an API request, later.csv or
holdout_features.csv) is safe to send to the model:

  errors   - the data cannot be used: missing columns, blanks, unknown
             categories, impossible values (client rules, sentinels)
  warnings - the data can be used but looks different from training:
             values outside the range seen in cleaned train.csv (drift hint)

Run from the repo root:   python -m mp299.validation data/raw/holdout_features.csv
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pandas as pd

FEATURES = ["highway_distance_km", "nearby_pumps", "price_rs", "trucks_passing",
            "temp_c", "staff", "ownership", "zone"]

CATEGORIES = {
    "ownership": {"public_sector", "private", "cooperative"},
    "zone": {"Highway", "City", "Industrial"},
}

# Hard limits: a value outside these is a typing mistake, never real.
# temp_c <= 75 and highway_distance_km >= 0 are the client's rules (PROBLEM.md s5).
HARD_LIMITS = {
    "highway_distance_km": (0, None),
    "nearby_pumps": (0, None),
    "price_rs": (0, None),
    "trucks_passing": (0, None),
    "temp_c": (None, 75),
    "staff": (1, None),
}

# Sentinels found in Stage 1: typed instead of a blank.
SENTINELS = {"staff": [999]}

# Range seen in cleaned train.csv (Jan-Jun 2026). Outside -> warning only.
TRAIN_RANGE = {
    "highway_distance_km": (0.02, 14.99),
    "nearby_pumps": (0, 10),
    "price_rs": (90.0, 109.98),
    "trucks_passing": (5, 1998),
    "temp_c": (5.0, 46.0),
    "staff": (3, 20),
}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_features(df: pd.DataFrame) -> Report:
    rep = Report()

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        rep.errors.append(f"missing columns: {missing}")
        return rep

    blanks = df[FEATURES].isna().sum()
    for c, n in blanks[blanks > 0].items():
        rep.errors.append(f"{c}: {n} blank value(s)")

    for c, allowed in CATEGORIES.items():
        bad = sorted(set(df[c].dropna()) - allowed)
        if bad:
            rep.errors.append(f"{c}: unknown categories {bad}")

    for c, (lo, hi) in HARD_LIMITS.items():
        col = pd.to_numeric(df[c], errors="coerce")
        if lo is not None and (col < lo).any():
            rep.errors.append(f"{c}: {(col < lo).sum()} value(s) below {lo}")
        if hi is not None and (col > hi).any():
            rep.errors.append(f"{c}: {(col > hi).sum()} value(s) above {hi}")

    for c, values in SENTINELS.items():
        n = int(df[c].isin(values).sum())
        if n:
            rep.errors.append(f"{c}: {n} sentinel value(s) {values}")

    for c, (lo, hi) in TRAIN_RANGE.items():
        col = pd.to_numeric(df[c], errors="coerce")
        n = int(((col < lo) | (col > hi)).sum())
        if n:
            rep.warnings.append(f"{c}: {n} value(s) outside training range {lo}..{hi}")

    return rep


def main(paths: list[str]) -> int:
    worst = 0
    for path in paths:
        rep = validate_features(pd.read_csv(path))
        print(f"{path}: {'OK' if rep.ok else 'FAILED'}")
        for e in rep.errors:
            print("  ERROR  ", e)
        for w in rep.warnings:
            print("  WARNING", w)
        worst = max(worst, 0 if rep.ok else 1)
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["data/raw/holdout_features.csv"]))

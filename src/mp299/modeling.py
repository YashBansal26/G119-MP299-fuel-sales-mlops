"""Feature repair and a serializable regression pipeline for the eight inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mp299.cleaning import drop_duplicates, fix_target_typos, fix_stuck_sensor


class FeatureRepair(TransformerMixin, BaseEstimator):
    """Apply fixed rules only; learned imputation stays inside the pipeline."""

    def __init__(self, numeric, categorical, sentinels):
        self.numeric = numeric
        self.categorical = categorical
        self.sentinels = sentinels

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(self.numeric + self.categorical, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        frame = X.loc[:, self.numeric + self.categorical].copy()
        for col, value in self.sentinels.items():
            if col in frame:
                frame[col] = frame[col].mask(frame[col] == value)
        for col in self.categorical:
            frame[col] = frame[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            frame[col] = frame[col].where(frame[col].notna(), np.nan)
        return frame


def repair_training_rows(raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Repair known data errors without computing any medians or statistics."""
    data, cleaning = params["data"], params["cleaning"]
    frame = raw.copy()
    for col in params["features"]["categorical"]:
        frame[col] = frame[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    frame = drop_duplicates(frame, data["id_col"])
    frame = fix_stuck_sensor(frame, cleaning["stuck_sensor"], data["date_col"])
    return fix_target_typos(frame, data["target"], cleaning["target_typo_threshold"],
                            cleaning["target_typo_factor"])


def build_pipeline(params: dict, candidate: dict) -> Pipeline:
    numeric, categorical = params["features"]["numeric"], params["features"]["categorical"]
    numeric_steps = [("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())]
    categorical_steps = [("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline(categorical_steps), categorical),
    ], remainder="drop")
    models = {"ridge": Ridge, "random_forest": RandomForestRegressor,
              "gradient_boosting": GradientBoostingRegressor}
    settings = dict(candidate["params"])
    if candidate["model"] != "ridge":
        settings["random_state"] = params["training"]["random_state"]
    estimator = models[candidate["model"]](**settings)
    return Pipeline([
        ("repair", FeatureRepair(numeric, categorical, params["cleaning"]["sentinels"])),
        ("preprocess", preprocessor),
        ("regressor", estimator),
    ])

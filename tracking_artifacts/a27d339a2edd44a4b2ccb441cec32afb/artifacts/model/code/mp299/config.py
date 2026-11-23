"""Load project settings from params.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_path(key: str, params: dict | None = None) -> Path:
    params = params or load_params()
    return ROOT / params["data"][key]


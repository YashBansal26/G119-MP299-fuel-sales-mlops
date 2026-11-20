"""Quick EDA printout of train.csv used while hunting the data problems."""
import pandas as pd

from mp299.config import data_path, load_params


def main() -> None:
    p = load_params()
    t = pd.read_csv(data_path("train", p))
    num = p["features"]["numeric"] + [p["data"]["target"]]
    print(t.shape)
    print(t.describe().T)
    for c in p["features"]["categorical"]:
        print(t[c].value_counts(dropna=False))
    for c in num:
        print(c, "top values:", t[c].value_counts().head(3).to_dict())
    print(t.groupby(p["data"]["group_col"])[num].agg(["min", "mean", "max"]).T)


if __name__ == "__main__":
    main()

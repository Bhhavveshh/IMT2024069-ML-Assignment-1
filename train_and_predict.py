from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


ROLL_NUMBER = "IMT2024069"
CONFIGURATIONS = {
    "var1": {"features": ["x1", "x2", "x3"], "degree": 3},
    "var2": {"features": ["x1"], "degree": 4},
}


def main(data_dir: Path) -> None:
    output_dir = Path(__file__).resolve().parent
    results = []
    for variant, config in CONFIGURATIONS.items():
        train = pd.read_csv(data_dir / f"{ROLL_NUMBER}_train_{variant}.csv")
        test = pd.read_csv(data_dir / f"{ROLL_NUMBER}_test_{variant}.csv")
        features = config["features"]
        if train.empty or test.empty or "y" not in train:
            raise ValueError(f"{variant}: empty dataset or missing training target")
        if train.isna().any().any() or test.isna().any().any():
            raise ValueError(f"{variant}: datasets contain missing values")
        if not np.isfinite(train.to_numpy()).all() or not np.isfinite(test.to_numpy()).all():
            raise ValueError(f"{variant}: datasets contain non-finite values")

        model = make_pipeline(
            PolynomialFeatures(degree=config["degree"], include_bias=False),
            LinearRegression(),
        )
        scores = cross_validate(
            model, train[features], train["y"],
            cv=KFold(n_splits=5, shuffle=True, random_state=2024069),
            scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        )
        model.fit(train[features], train["y"])
        train_predictions = model.predict(train[features])
        results.append({
            "dataset": variant,
            "degree": config["degree"],
            "features": ", ".join(features),
            "cv_mse_mean": -scores["test_mse"].mean(),
            "cv_mse_std": scores["test_mse"].std(),
            "cv_r2_mean": scores["test_r2"].mean(),
            "cv_r2_std": scores["test_r2"].std(),
            "train_mse": mean_squared_error(train["y"], train_predictions),
            "train_r2": r2_score(train["y"], train_predictions),
        })

        submission = pd.DataFrame({"y": model.predict(test[features])})
        if not np.isfinite(submission["y"]).all():
            raise ValueError(f"{variant}: predictions contain non-finite values")
        submission.to_csv(output_dir / f"{ROLL_NUMBER}_pred_{variant}.csv", index=False)
    pd.DataFrame(results).to_csv(output_dir / "validation_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.data_dir)

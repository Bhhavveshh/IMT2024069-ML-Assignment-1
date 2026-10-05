import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

ROLL_NUMBER = "IMT2024069"
SEED = 2024069
MAX_DEGREES = {"var1": 10, "var2": 20}
ALPHAS = [0.0001, 0.01, 0.1, 1.0, 10.0, 100.0]


def make_model(degree, alpha):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(), Ridge(alpha=alpha, solver="cholesky"),
    )


def load_data(data_dir, variant):
    train = pd.read_csv(data_dir / f"{ROLL_NUMBER}_train_{variant}.csv")
    test = pd.read_csv(data_dir / f"{ROLL_NUMBER}_test_{variant}.csv")
    features = [f"x{i}" for i in range(1, 7 if variant == "var1" else 4)]
    if train.columns.tolist() != features + ["y"] or test.columns.tolist() != features:
        raise ValueError(f"{variant}: unexpected dataset columns")
    for frame in [train, test]:
        if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{variant}: empty dataset or non-finite values")
    return train[features], train["y"], test


def run(data_dir, output_dir):
    screening_rows, cv_rows, summaries = [], [], []
    configurations = {}
    for variant, maximum in MAX_DEGREES.items():
        x, y, test = load_data(data_dir, variant)
        development, holdout = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=SEED + 1,
        )
        fit, validation = train_test_split(development, test_size=0.25, random_state=SEED)
        candidates = []
        for degree in range(1, maximum + 1):
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            z_fit = poly.fit_transform(x.iloc[fit])
            z_valid = poly.transform(x.iloc[validation])
            scaler = StandardScaler().fit(z_fit)
            a, b = scaler.transform(z_fit), scaler.transform(z_valid)
            for alpha in ALPHAS:
                model = Ridge(alpha=alpha, solver="cholesky").fit(a, y.iloc[fit])
                predictions = model.predict(b)
                row = dict(dataset=variant, degree=degree, alpha=alpha, terms=z_fit.shape[1],
                           mse=mean_squared_error(y.iloc[validation], predictions),
                           r2=r2_score(y.iloc[validation], predictions))
                candidates.append(row)
                screening_rows.append(row)
            print(f"{variant}: screened degree {degree}", flush=True)
        best_degrees = sorted(range(1, maximum + 1), key=lambda d:
            min(r["mse"] for r in candidates if r["degree"] == d))[:4]
        shortlist = []
        for degree in best_degrees:
            shortlist.extend(sorted([r for r in candidates if r["degree"] == degree],
                                    key=lambda r:r["mse"])[:3])
        variant_cv = []
        for candidate in shortlist:
            scores = cross_validate(
                make_model(candidate["degree"], candidate["alpha"]),
                x.iloc[development], y.iloc[development],
                cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
                scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
            )
            row = dict(dataset=variant, degree=candidate["degree"], alpha=candidate["alpha"],
                       terms=candidate["terms"], cv_mse_mean=-scores["test_mse"].mean(),
                       cv_mse_std=scores["test_mse"].std(), cv_r2_mean=scores["test_r2"].mean(),
                       cv_r2_std=scores["test_r2"].std())
            variant_cv.append(row)
            cv_rows.append(row)
        selected = min(variant_cv, key=lambda r:r["cv_mse_mean"])
        model = make_model(selected["degree"], selected["alpha"])
        model.fit(x.iloc[development], y.iloc[development])
        predictions = model.predict(x.iloc[holdout])
        baseline = np.full(len(holdout), y.iloc[development].mean())
        summary = dict(selected, features=", ".join(x.columns),
                       development_rows=len(development), holdout_rows=len(holdout),
                       holdout_mse=mean_squared_error(y.iloc[holdout], predictions),
                       holdout_r2=r2_score(y.iloc[holdout], predictions),
                       baseline_holdout_mse=mean_squared_error(y.iloc[holdout], baseline),
                       baseline_holdout_r2=r2_score(y.iloc[holdout], baseline))
        summaries.append(summary)
        configurations[variant] = dict(degree=selected["degree"], alpha=selected["alpha"],
                                       features=x.columns.tolist())
        model.fit(x, y)
        submission = pd.DataFrame({"y": model.predict(test)})
        if not np.isfinite(submission.to_numpy()).all():
            raise ValueError(f"{variant}: non-finite predictions")
        submission.to_csv(output_dir / f"{ROLL_NUMBER}_pred_{variant}.csv", index=False)
        print(summary, flush=True)
    pd.DataFrame(screening_rows).to_csv(output_dir / "screening_results.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output_dir / "cv_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "validation_results.csv", index=False)
    with (output_dir / "selected_models.json").open("w") as stream:
        json.dump({"seed": SEED, "models": configurations}, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.data_dir, Path(__file__).resolve().parent)

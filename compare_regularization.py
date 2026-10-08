import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from threadpoolctl import threadpool_limits

from train_and_predict import MAX_DEGREES, SEED, load_data, make_model

GRIDS = {
    "L1": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    "L2": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
}


def compare(data_dir, base):
    rows, screening = [], []
    for variant, maximum in MAX_DEGREES.items():
        x, y, _ = load_data(data_dir, variant)
        development, _ = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=SEED + 1
        )
        fit, valid = train_test_split(development, test_size=0.25, random_state=SEED)
        variant_screen = []
        for degree in range(1, maximum + 1):
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            a, b = poly.fit_transform(x.iloc[fit]), poly.transform(x.iloc[valid])
            scaler = StandardScaler().fit(a)
            a, b = scaler.transform(a), scaler.transform(b)
            a = np.asfortranarray(a)
            for family, alphas in GRIDS.items():
                estimator = (
                    Lasso(warm_start=True, max_iter=5000, tol=0.0001)
                    if family == "L1"
                    else None
                )
                for alpha in reversed(alphas):
                    row = dict(
                        dataset=variant,
                        degree=degree,
                        regularization=family,
                        alpha=alpha,
                    )
                    if family == "L1":
                        estimator.set_params(alpha=alpha)
                    else:
                        estimator = Ridge(alpha=alpha, solver="cholesky")
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", ConvergenceWarning)
                            estimator.fit(a, y.iloc[fit])
                        predictions = estimator.predict(b)
                        row.update(
                            status="converged",
                            mse=mean_squared_error(y.iloc[valid], predictions),
                        )
                    except ConvergenceWarning:
                        row.update(status="excluded: non-convergence", mse=np.nan)
                    variant_screen.append(row)
                    screening.append(row)
            pd.DataFrame(screening).to_csv(
                base / "joint_screening_results.csv", index=False
            )
            print(f"{variant}: jointly screened degree {degree}", flush=True)
        candidates = []
        for family in GRIDS:
            converged = [
                r
                for r in variant_screen
                if r["regularization"] == family and r["status"] == "converged"
            ]
            ranked = sorted(
                range(1, maximum + 1),
                key=lambda d: min(
                    (r["mse"] for r in converged if r["degree"] == d),
                    default=float("inf"),
                ),
            )
            for degree in ranked[:4]:
                candidates.extend(
                    sorted(
                        [r for r in converged if r["degree"] == degree],
                        key=lambda r: r["mse"],
                    )[:3]
                )
        for candidate in candidates:
            degree, alpha, family = (
                candidate["degree"],
                candidate["alpha"],
                candidate["regularization"],
            )
            model = make_model(degree, alpha, family)
            row = dict(
                dataset=variant, degree=degree, regularization=family, alpha=alpha
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    scores = cross_validate(
                        model,
                        x.iloc[development],
                        y.iloc[development],
                        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
                        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                        error_score="raise",
                    )
                row.update(
                    status="converged",
                    cv_mse_mean=-scores["test_mse"].mean(),
                    cv_mse_std=scores["test_mse"].std(),
                    cv_r2_mean=scores["test_r2"].mean(),
                    cv_r2_std=scores["test_r2"].std(),
                )
            except ConvergenceWarning:
                row.update(status="excluded: non-convergence")
            rows.append(row)
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(
                base / "regularization_comparison.csv", index=False
            )


def finalize(data_dir, base):
    comparison = pd.read_csv(base / "regularization_comparison.csv")
    summaries, configurations = [], {}
    for variant, group in comparison.groupby("dataset"):
        selected = group[group.status == "converged"].sort_values("cv_mse_mean").iloc[0]
        x, y, test = load_data(data_dir, variant)
        development, holdout = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=SEED + 1
        )
        model = make_model(
            int(selected.degree), float(selected.alpha), selected.regularization
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x.iloc[development], y.iloc[development])
        prediction = model.predict(x.iloc[holdout])
        baseline = np.full(len(holdout), y.iloc[development].mean())
        summary = selected.to_dict()
        summary.update(
            features=", ".join(x.columns),
            terms=model.named_steps["polynomialfeatures"].n_output_features_,
            development_rows=len(development),
            holdout_rows=len(holdout),
            holdout_mse=mean_squared_error(y.iloc[holdout], prediction),
            holdout_r2=r2_score(y.iloc[holdout], prediction),
            baseline_holdout_mse=mean_squared_error(y.iloc[holdout], baseline),
            baseline_holdout_r2=r2_score(y.iloc[holdout], baseline),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x, y)
        estimator = model.steps[-1][1]
        summary["active_terms"] = int(np.count_nonzero(estimator.coef_))
        predictions = model.predict(test)
        if not np.isfinite(predictions).all():
            raise ValueError(f"{variant}: non-finite predictions")
        pd.DataFrame({"y": predictions}).to_csv(
            base / f"IMT2024069_pred_{variant}.csv", index=False
        )
        summaries.append(summary)
        configurations[variant] = dict(
            degree=int(selected.degree),
            alpha=float(selected.alpha),
            regularization=selected.regularization,
            features=x.columns.tolist(),
        )
        print(summary, flush=True)
    pd.DataFrame(summaries).to_csv(base / "validation_results.csv", index=False)
    (base / "selected_models.json").write_text(
        json.dumps({"seed": SEED, "models": configurations}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        base = Path(__file__).resolve().parent
        if not args.finalize_only:
            compare(args.data_dir, base)
        finalize(args.data_dir, base)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

ROLL_NUMBER = "IMT2024069"
SEED = 2024069
MAX_DEGREES = {"var1": 10, "var2": 20}


def make_model(degree, alpha, regularization="L2"):
    if regularization not in {"L1", "L2"}:
        raise ValueError("Regularization must be L1 or L2")
    estimator = (
        Ridge(alpha=alpha, solver="cholesky")
        if regularization == "L2"
        else Lasso(
            alpha=alpha,
            max_iter=30000,
            tol=0.00001,
        )
    )
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        estimator,
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


if __name__ == "__main__":
    from compare_regularization import compare, finalize

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        base = Path(__file__).resolve().parent
        compare(args.data_dir, base)
        finalize(args.data_dir, base)

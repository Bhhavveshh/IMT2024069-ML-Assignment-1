import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from train_and_predict import SEED, load_data, make_model


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_results(data_dir, base):
    directory = base / "graphs"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    results = pd.read_csv(base / "validation_results.csv").set_index("dataset")
    screening = pd.read_csv(base / "screening_results.csv")
    cv = pd.read_csv(base / "cv_results.csv")
    config = json.loads((base / "selected_models.json").read_text())["models"]
    for name, source, metric, ylabel in [
        ("degree_screening", screening, "mse", "Screening MSE (log scale)"),
        ("degree_cv", cv, "cv_mse_mean", "Mean five-fold CV MSE"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
        for ax, variant in zip(axes, ["var1", "var2"]):
            rows = source[source.dataset == variant]
            best = rows.loc[rows.groupby("degree")[metric].idxmin()].sort_values("degree")
            ax.plot(best.degree, best[metric], "o-", color="black", markersize=4,
                    label="Best alpha at each degree")
            degree = config[variant]["degree"]
            selected = best[best.degree == degree].iloc[0]
            ax.scatter([degree], [selected[metric]], marker="*", s=160, color="black",
                       label=f"Selected degree {degree}", zorder=5)
            if name == "degree_cv":
                ax.errorbar(best.degree, best[metric], yerr=best.cv_mse_std,
                            color="black", fmt="none", capsize=4)
            else:
                ax.set_yscale("log")
            ax.set(title=variant, xlabel="Polynomial degree", ylabel=ylabel)
            ax.set_xticks(best.degree if len(best) <= 10 else [1, 4, 8, 12, 16, 20])
            ax.legend(frameon=False, fontsize=8)
        save_figure(fig, directory, name)
    frames = {}
    for variant, settings in config.items():
        x, y, _ = load_data(data_dir, variant)
        development, holdout = train_test_split(np.arange(len(y)), test_size=0.2,
                                                 random_state=SEED + 1)
        model = make_model(settings["degree"], settings["alpha"])
        model.fit(x.iloc[development], y.iloc[development])
        prediction = model.predict(x.iloc[holdout])
        np.testing.assert_allclose(mean_squared_error(y.iloc[holdout], prediction),
                                   results.loc[variant, "holdout_mse"], rtol=1e-10)
        frames[variant] = pd.DataFrame({"training_row_index": holdout,
                                       "actual_y": y.iloc[holdout].to_numpy(),
                                       "predicted_y": prediction,
                                       "residual": y.iloc[holdout].to_numpy() - prediction})
        frames[variant].to_csv(directory / f"{variant}_holdout_points.csv", index=False)
    for name in ["holdout_predictions", "holdout_residuals"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
        for ax, variant in zip(axes, ["var1", "var2"]):
            frame = frames[variant]
            if name == "holdout_predictions":
                ax.scatter(frame.actual_y, frame.predicted_y, s=15, facecolors="none",
                           edgecolors="black", linewidths=0.6)
                limits = [min(frame.actual_y.min(), frame.predicted_y.min()),
                          max(frame.actual_y.max(), frame.predicted_y.max())]
                ax.plot(limits, limits, "--", color="black", label="Perfect prediction")
                ax.set(xlabel="Actual target y", ylabel="Predicted target y",
                       title=f"{variant}: R2 = {r2_score(frame.actual_y, frame.predicted_y):.4f}")
                ax.legend(frameon=False, fontsize=8)
            else:
                ax.scatter(frame.predicted_y, frame.residual, s=15, facecolors="none",
                           edgecolors="black", linewidths=0.6)
                ax.axhline(0, linestyle="--", color="black")
                ax.set(xlabel="Predicted target y", ylabel="Residual (actual - predicted)",
                       title=variant)
        save_figure(fig, directory, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        plot_results(args.data_dir, Path(__file__).resolve().parent)

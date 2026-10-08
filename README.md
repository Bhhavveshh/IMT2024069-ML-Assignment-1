# ML Assignment 1 - IMT2024069

This submission uses polynomial regression only, as required.

| Dataset | Input columns used | Polynomial degree |
| --- | --- | --- |
| var1 | `x1`, `x2`, `x3`, `x4`, `x5`, `x6` | 5 |
| var2 | `x1`, `x2`, `x3` | 12 |

## Reproduce predictions

Install dependencies, then run:

```powershell
python -m pip install -r requirements.txt
python train_and_predict.py --data-dir "path\\to\\the\\assignment\\CSVs"
python compare_regularization.py --data-dir "path\\to\\the\\assignment\\CSVs"
```

The first command performs stage-1 Ridge degree selection. The second compares
L1/L2 at those degrees and writes the final `IMT2024069_pred_var1.csv` and
`IMT2024069_pred_var2.csv`. Run both commands in that order.

The included CSV files were generated from the supplied personalized training
and test data. The PDF report documents the modelling and validation approach.

## Model selection and results

The visible assignment permits degrees up to 10 for var1 and 20 for var2.
Hidden PDF text was excluded from model selection. All supplied inputs are used.
Polynomial terms are standardized and fitted with Lasso (L1) or Ridge (L2),
retaining a polynomial prediction function. Course Lectures 7-8 motivate
validation-based penalty selection. Lecture 9 motivates residual diagnostics;
no Gaussian noise or calibrated uncertainty is assumed.

The script reserves 200 rows for holdout evaluation and screens degrees and six
regularization strengths within the other 800 rows. The four best screening
degrees and their three best alpha values are compared by five-fold CV.
Scaling is fitted within each fold. Lowest mean CV MSE determines the final model.
Seeds are 2024070 for the holdout and 2024069 for screening and CV.

| Problem | Degree | Penalty | Alpha | CV MSE | Holdout MSE | Holdout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| var1 | 5 | L1 | 0.01 | 0.3603 | 0.4062 | 0.9682 |
| var2 | 12 | L2 | 1 | 0.2782 | 0.2662 | 0.9936 |

At each stage-1 degree, Lasso alpha values 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
and 1 are compared with the selected Ridge configuration on the same five
development folds. The lowest mean CV MSE among converged candidates wins.
This is conditional on the Ridge-selected degrees, not an exhaustive joint
L1 degree search. Lasso uses max_iter=30000 and tol=0.00001. The two smallest
alphas for var2 did not converge and were excluded. L1 retains 124 of 461 terms
for var1; L2 retains all 454 terms for var2 after full-data fitting.

Ridge minimizes SSE + alpha * squared coefficient norm. Lasso minimizes
SSE/(2n) + alpha * absolute coefficient norm. Relative to the slides' MSE-based
objectives, Ridge lambda=alpha/n and Lasso lambda=2*alpha. Thus L1 and L2 alpha
values are not numerically equivalent. The intercept is not penalized.

The holdout is excluded from final model selection. Earlier exploratory work
used the training rows, so these holdout results are diagnostic rather than a
fully unseen external evaluation. CV scores are model-selection scores.
Hidden test scores cannot be computed without labels. Degree 10 and degree 12
perform almost identically for var2; the minimum-CV-MSE rule selected degree 12.

Final models are fitted on all 1000 training rows per problem. Predictions match
the supplied sample submission: one `y` column, no index, original test row order.

`screening_results.csv`, `cv_results.csv`, `validation_results.csv` and
`regularization_comparison.csv` and `selected_models.json` record configurations
and results. Run training followed by the comparison to recreate final models
and predictions. To rebuild the black PDF report, regenerate its figures first:

```powershell
python plot_results.py --data-dir "path\to\the\assignment\CSVs"
python build_report.py
```

To reproduce the five black-and-white graphs before building the report:

```powershell
python plot_results.py --data-dir "path\to\the\assignment\CSVs"
python build_report.py
```

The `graphs` directory contains degree-screening curves, CV comparisons,
holdout predicted-versus-actual plots, residual plots and the plotted holdout data.
The report includes all five figures and remains within the five-page limit.

# Polynomial Regression - IMT2024069

Two independent polynomial models predict the turbine power score (var1) and
reservoir thermal anomaly score (var2). All six var1 inputs and all three var2
inputs are retained. Hidden assignment text is excluded from model selection.

## Reproduce the complete submission

```powershell
python -m pip install -r requirements.txt
python train_and_predict.py --data-dir "path\to\the\assignment\CSVs"
python plot_results.py --data-dir "path\to\the\assignment\CSVs"
python build_report.py
```

Training delegates the joint search and final fitting to
`compare_regularization.py`. Running that script directly performs the same
workflow. To refit the selected candidates from saved CV results without
repeating the search, add `--finalize-only` to its command.

## Model selection

1. Reserve 200 training rows for holdout diagnostics (seed 2024070), leaving
   800 development rows. Split development into 600 fitting and 200 screening
   rows (seed 2024069).
2. Screen every degree 1-10 for var1 and 1-20 for var2 under both L1 and L2.
   L1 alpha values: 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.
   L2 alpha values: 0.0001, 0.001, 0.01, 0.1, 1, 10, 100.
3. For each penalty, shortlist the four best screening degrees and up to three
   converged alpha values per degree. Compare these candidates using five-fold
   shuffled development CV (seed 2024069).
4. Choose the converged candidate with the lowest mean CV MSE. Evaluate it on
   the holdout, then refit on all 1,000 training rows for test predictions.

Lasso screening uses a descending alpha warm-start path, up to 5,000 iterations
and tolerance 0.0001. CV and final fitting use up to 30,000 iterations and
tolerance 0.00001. Failed convergence is recorded and excludes the candidate.
The selected configurations are the best validated choices in this finite
search, not a guaranteed global optimum. The holdout is excluded from selection,
but earlier exploratory work used these training rows, so it is diagnostic.

## Normalization and objectives

All powers and interactions with total degree at most d are generated.
StandardScaler then applies z-score standardization to each polynomial term,
using the fitting-set mean and population standard deviation (ddof=0). Scaling
is fitted independently within each CV fold. Test data use full-training
scaling. Raw inputs are used as supplied, and the target is not normalized.

Ridge minimizes SSE + alpha * sum(w squared). Lasso minimizes
SSE/(2n) + alpha * sum(abs(w)). The intercept is not penalized. L1 and L2 alpha
values are not numerically equivalent.

## Outputs

- `selected_models.json`: final degree, penalty, alpha and features.
- `joint_screening_results.csv`: both penalties across every allowed degree.
- `regularization_comparison.csv`: CV results for the joint shortlist.
- `validation_results.csv`: final CV and holdout metrics and term counts.
- `IMT2024069_pred_var1.csv` and `IMT2024069_pred_var2.csv`: one `y` column,
  1,000 finite predictions, no index, original test row order.
- `graphs/`: five reproducible black-and-white figures and holdout-point data.
- `IMT2024069_Assignment_1_Report.pdf`: methods, selection evidence and results.

The supplied sample submission defines the prediction format. Hidden test
labels are unavailable; no test MSE or R2 is claimed. CV scores are used for
selection and should not be interpreted as an unbiased external test score.

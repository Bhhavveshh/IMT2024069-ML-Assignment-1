# ML Assignment 1 - IMT2024069

This submission uses polynomial regression only, as required.

| Dataset | Input columns used | Polynomial degree |
| --- | --- | --- |
| var1 | `x1`, `x2`, `x3` | 3 |
| var2 | `x1` | 4 |

## Reproduce predictions

Install dependencies, then run:

```powershell
python -m pip install -r requirements.txt
python train_and_predict.py --data-dir "path\\to\\the\\assignment\\CSVs"
```

The script writes `IMT2024069_pred_var1.csv` and
`IMT2024069_pred_var2.csv` in this directory.

The included CSV files were generated from the supplied personalized training
and test data. The PDF report documents the modelling and validation approach.

The script also reproduces five-fold validation in `validation_results.csv`.
The validation R2 values are low (approximately 0.155 for var1 and 0.067 for
var2); the models follow the brief's suggested configurations, but these scores
do not demonstrate optimal performance. Hidden test scores cannot be computed
without the test labels. No sample submission was supplied; predictions retain
all original test columns and append `y`.

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
```

The script writes `IMT2024069_pred_var1.csv` and
`IMT2024069_pred_var2.csv` in this directory.

The included CSV files were generated from the supplied personalized training
and test data. The PDF report documents the modelling and validation approach.

## Model selection and results

The visible assignment permits degrees up to 10 for var1 and 20 for var2.
Hidden PDF text was excluded from model selection. All supplied inputs are used.
Polynomial terms are standardized and fitted with ridge regression, retaining
a polynomial prediction function.

The script reserves 200 rows for holdout evaluation and screens degrees and six
regularization strengths within the other 800 rows. The four best screening
degrees and their three best alpha values are compared by five-fold CV.
Scaling is fitted within each fold. Lowest mean CV MSE determines the final model.
Seeds are 2024070 for the holdout and 2024069 for screening and CV.

| Problem | Degree | Ridge alpha | CV MSE | Holdout MSE | Holdout R2 |
| --- | --- | --- | --- | --- | --- |
| var1 | 5 | 10 | 0.5516 | 0.5505 | 0.9569 |
| var2 | 12 | 1 | 0.2782 | 0.2662 | 0.9936 |

The holdout is excluded from final model selection. Earlier exploratory work
used the training rows, so these holdout results are diagnostic rather than a
fully unseen external evaluation. CV scores are model-selection scores.
Hidden test scores cannot be computed without labels. Degree 10 and degree 12
perform almost identically for var2; the minimum-CV-MSE rule selected degree 12.

Final models are fitted on all 1000 training rows per problem. Predictions match
the supplied sample submission: one `y` column, no index, original test row order.

`screening_results.csv`, `cv_results.csv`, `validation_results.csv` and
`selected_models.json` record configurations and results. Running training
recreates these files and both predictions. To rebuild the black PDF report:

```powershell
python build_report.py
```

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE = Path(__file__).resolve().parent
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=7))
styles.add(ParagraphStyle(name="Heading", fontName="Helvetica-Bold", fontSize=12, leading=16,
                         spaceBefore=9, spaceAfter=5, textColor=colors.black))
styles["Title"].textColor = colors.black
styles["Title"].fontSize = 18
story = []


def body(text):
    story.append(Paragraph(text, styles["Body"]))


def heading(text):
    story.append(Paragraph(text, styles["Heading"]))


def table(rows, widths):
    item = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    item.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([item, Spacer(1, 7)])


results = pd.read_csv(BASE / "validation_results.csv")
cv = pd.read_csv(BASE / "cv_results.csv")
story.append(Paragraph("ML Assignment 1: Polynomial Regression", styles["Title"]))
body("Roll number: <b>IMT2024069</b>")
heading("1. Data and approach")
body("The two personalized problems were modelled independently. Each has 1,000 training rows with target y and 1,000 test rows without labels. All six supplied inputs were retained for turbine optimization (var1), and all three for reservoir mapping (var2). Dataset columns, missing values and finite numeric values were checked before fitting.")
body("Every monomial and interaction term with total degree at most d was generated and standardized. L1 (Lasso) and L2 (Ridge) penalize absolute and squared coefficients, respectively; both retain a polynomial prediction function. The intercept is unpenalized. Following Lectures 7-8, validation determined regularization rather than assumptions about which inputs matter.")
heading("2. Degree selection and rationale")
body("Degrees 1-10 for var1 and 1-20 for var2 were considered, following the visible assignment limits. No feature subset or degree was selected from hidden PDF text. Each dataset was split into 800 development rows and 200 holdout rows (seed 2024070). Within development, a 600/200 screening split (seed 2024069) compared each degree with alpha in {0.0001, 0.01, 0.1, 1, 10, 100}.")
body("Stage 1 used L2: the four best screening degrees and their three best alpha values were compared by five-fold shuffled CV on the 800 development rows (seed 2024069). Stage 2 compared L1 and L2 at the selected degrees using the same folds. Scaling was fitted within each fold. Lowest mean CV MSE among converged candidates selected the final penalty; the holdout was not used to choose it.")
rows = [["Problem", "Features", "Degree", "Penalty", "Alpha", "Terms"]]
for r in results.itertuples():
    rows.append([r.dataset, "x1-x6" if r.dataset == "var1" else "x1-x3", str(r.degree), r.regularization, f"{r.alpha:g}", str(r.terms)])
table(rows, [2, 3.5, 2, 2, 2, 3])
body("The selected degrees balance approximation quality against variance. Lower degrees missed nonlinear structure, while adding more terms did not consistently improve cross-validation error. Degree was chosen jointly with regularization, rather than from training fit alone. Degree 10 and degree 12 performed almost identically for var2; the minimum-CV-MSE rule selected degree 12. Term counts exclude the separately fitted intercept.")
rows = [["Problem", "Degree", "Best stage-1 L2 alpha", "L2 mean CV MSE"]]
for variant in ["var1", "var2"]:
    for degree, group in cv[cv.dataset == variant].groupby("degree"):
        r = group.loc[group.cv_mse_mean.idxmin()]
        rows.append([variant, str(degree), f"{r.alpha:g}", f"{r.cv_mse_mean:.4f}"])
table(rows, [2.5, 2, 5, 5])
story.append(PageBreak())
heading("3. Evaluation")
rows = [["Problem", "CV MSE +/- SD", "CV R2", "Holdout MSE", "Holdout R2"]]
for r in results.itertuples():
    rows.append([r.dataset, f"{r.cv_mse_mean:.4f} +/- {r.cv_mse_std:.4f}", f"{r.cv_r2_mean:.4f}", f"{r.holdout_mse:.4f}", f"{r.holdout_r2:.4f}"])
table(rows, [2, 4.5, 2.5, 3, 2.5])
body("A constant predictor using the development target mean provides a degree-zero baseline. Its holdout MSE was " + "; ".join(f"{r.dataset}: {r.baseline_holdout_mse:.4f}" for r in results.itertuples()) + ". Both selected polynomial models substantially improve on this baseline.")
body("The holdout was excluded from the final screening and CV selection procedure. Earlier exploratory analyses had already used the provided training rows, so this holdout is a diagnostic rather than a completely unseen external evaluation. CV results are also selection scores. Hidden test labels were unavailable; no test MSE or R2 is claimed.")
heading("4. Final predictions and reproducibility")
body("After evaluating the chosen configurations, each model was refitted on all 1,000 training rows. Predictions were produced for the corresponding test rows in their original order. IMT2024069_pred_var1.csv and IMT2024069_pred_var2.csv each contain exactly one y column with 1,000 finite predictions and no index, matching sample_submission.csv.")
body("Run train_and_predict.py, then compare_regularization.py to reproduce both selection stages and final inference. The latter writes the final models and predictions. The CSV results and selected_models.json record configurations and scores. requirements.txt records dependencies; README.md gives commands. build_report.py generates this report.")
body('GitHub repository: <link href="https://github.com/Bhhavveshh/IMT2024069-ML-Assignment-1" color="black">https://github.com/Bhhavveshh/IMT2024069-ML-Assignment-1</link>')
story.append(PageBreak())
heading("5. Degree-selection graphs")
body("Figure 1. Stage-1 L2 screening MSE across all allowed degrees. Each point is the lowest error among the six Ridge alpha values at that degree, evaluated on the 200-row screening split. A star marks the degree selected before the L1/L2 comparison. The logarithmic axis exposes the low-degree decline and high-degree plateau.")
story.append(Image(str(BASE / "graphs/degree_screening.png"), width=17*cm, height=6.29*cm))
story.append(Spacer(1, 14))
body("Figure 2. Stage-1 L2 cross-validation comparison of the four shortlisted degrees. Each point uses the best of three shortlisted Ridge alpha values at that degree. Error bars show one fold standard deviation, not confidence intervals. The degree-10 and degree-12 results for var2 are nearly indistinguishable. Final penalty choices appear in Figure 5.")
story.append(Image(str(BASE / "graphs/degree_cv.png"), width=17*cm, height=6.29*cm))
story.append(PageBreak())
heading("6. Holdout prediction diagnostics")
body("Figure 3. Predicted versus actual targets for the 200 holdout rows in each problem. The dashed diagonal represents perfect predictions. These plots use models fitted on the 800 development rows, before refitting on all training rows for submission.")
story.append(Image(str(BASE / "graphs/holdout_predictions.png"), width=17*cm, height=6.29*cm))
story.append(Spacer(1, 14))
body("Figure 4. Holdout residuals plotted against predicted targets. Residuals are actual minus predicted values; the dashed horizontal line marks zero error. These plots expose signed error and target-dependent patterns that aggregate MSE and R2 can obscure. The holdout limitations described in Section 3 also apply to these figures.")
story.append(Image(str(BASE / "graphs/holdout_residuals.png"), width=17*cm, height=6.29*cm))
body("plot_results.py reproduces all five figures and their holdout points from the final configurations. Full-resolution PNGs and the corresponding holdout-point CSV files are included in the graphs directory.")
story.append(PageBreak())
heading("7. L1 versus L2 regularization")
body("Following Lecture 7's validation-based penalty selection and Lecture 8's Lasso/Ridge comparison, Lasso alpha values {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1} were compared against the stage-1 Ridge choice at each selected degree. The same five development folds and scaling pipeline were used. This is a conditional comparison at degrees 5 and 12, not a joint search over every L1 degree.")
comparison = pd.read_csv(BASE / "regularization_comparison.csv")
rows = [["Problem", "Penalty", "Best alpha", "Mean CV MSE"]]
for variant in ["var1", "var2"]:
    for family in ["L1", "L2"]:
        group = comparison[(comparison.dataset == variant) & (comparison.regularization == family) & (comparison.status == "converged")]
        r = group.loc[group.cv_mse_mean.idxmin()]
        rows.append([variant, family, f"{r.alpha:g}", f"{r.cv_mse_mean:.4f}"])
table(rows, [2.5, 2.5, 4.5, 5])
body("L1 was selected for var1: alpha 0.01 reduced mean CV MSE from 0.5516 to 0.3603. After fitting all 1,000 rows, 124 of 461 terms have nonzero coefficients. L2 was retained for var2: alpha 1 achieved CV MSE 0.2782, below the best converged L1 result (0.4035). Var2 Lasso alpha 0.001 and 0.003 failed the convergence criterion and were excluded, not treated as successful fits. Maximum iterations: 30,000; tolerance: 0.00001.")
body("Figure 5. Lasso validation error versus alpha, with the chosen stage-1 Ridge CV error shown as a horizontal reference. Only converged Lasso candidates are plotted. Alpha values for L1 and L2 cannot be compared numerically as equivalent penalties because their objective normalizations differ.")
story.append(Image(str(BASE / "graphs/regularization_comparison.png"), width=17*cm, height=6.29*cm))
body("Objective conventions: Ridge minimizes SSE + alpha * sum(w squared). Lasso minimizes SSE/(2n) + alpha * sum(abs(w)). For the slides' MSE-normalized objectives, Ridge lambda = alpha/n and Lasso lambda = 2*alpha, with n the current fit size. The intercept is excluded from these penalties.")
body("Lecture 9 separates signal from random error. Residual plots are used here as diagnostics; Gaussian residuals and calibrated uncertainty intervals are not assumed. Course references: Dr. Viswanath Gopalakrishnan, IIIT Bangalore, Lecture 7 (regularized least squares), Lecture 8 (Lasso/Ridge), Lecture 9 (noise modelling), supplied annotated slides.")


def footer(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.black)
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))


SimpleDocTemplate(str(BASE / "IMT2024069_Assignment_1_Report.pdf"), pagesize=A4,
                  leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.6*cm,
                  title="ML Assignment 1 - IMT2024069", author="IMT2024069").build(
                      story, onFirstPage=footer, onLaterPages=footer)

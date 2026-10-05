from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
body("Every monomial and interaction term with total degree at most d was generated. Polynomial terms were standardized, then fitted using ridge regression with an intercept. Ridge minimizes squared residual error plus an alpha-weighted squared-coefficient penalty. The prediction function remains a polynomial in the original inputs; standardization and regularization improve numerical stability and control variance.")
heading("2. Degree selection and rationale")
body("Degrees 1-10 for var1 and 1-20 for var2 were considered, following the visible assignment limits. No feature subset or degree was selected from hidden PDF text. Each dataset was split into 800 development rows and 200 holdout rows (seed 2024070). Within development, a 600/200 screening split (seed 2024069) compared each degree with alpha in {0.0001, 0.01, 0.1, 1, 10, 100}.")
body("For each problem, the four degrees with the lowest screening MSE were retained, with their three best screening alpha values. These 12 candidates were compared using five-fold shuffled cross-validation on the 800 development rows (seed 2024069). Polynomial scaling was fitted within each fold. The configuration with the lowest mean CV MSE was selected before evaluating the 200 holdout rows.")
rows = [["Problem", "Features", "Degree", "Alpha", "Terms"]]
for r in results.itertuples():
    rows.append([r.dataset, "x1-x6" if r.dataset == "var1" else "x1-x3", str(r.degree), f"{r.alpha:g}", str(r.terms)])
table(rows, [2.5, 4, 2, 2.5, 3.5])
body("The selected degrees balance approximation quality against variance. Lower degrees missed nonlinear structure, while adding more terms did not consistently improve cross-validation error. Degree was chosen jointly with regularization, rather than from training fit alone. Degree 10 and degree 12 performed almost identically for var2; the minimum-CV-MSE rule selected degree 12. Term counts exclude the separately fitted intercept.")
rows = [["Problem", "Degree", "Best shortlisted alpha", "Mean CV MSE"]]
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
body("train_and_predict.py reproduces screening, CV, holdout evaluation, model selection and final inference. screening_results.csv, cv_results.csv, validation_results.csv and selected_models.json record the experiments. requirements.txt records dependency versions; README.md gives execution commands. build_report.py generates this report from the recorded results.")
body('GitHub repository: <link href="https://github.com/Bhhavveshh/IMT2024069-ML-Assignment-1" color="black">https://github.com/Bhhavveshh/IMT2024069-ML-Assignment-1</link>')


def footer(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.black)
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))


SimpleDocTemplate(str(BASE / "IMT2024069_Assignment_1_Report.pdf"), pagesize=A4,
                  leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.6*cm,
                  title="ML Assignment 1 - IMT2024069", author="IMT2024069").build(
                      story, onFirstPage=footer, onLaterPages=footer)

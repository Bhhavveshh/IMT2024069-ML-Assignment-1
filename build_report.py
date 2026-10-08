from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE = Path(__file__).resolve().parent
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="Body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8
    )
)
styles.add(
    ParagraphStyle(
        name="Heading",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=9,
        spaceAfter=6,
        textColor=colors.black,
    )
)
styles["Title"].textColor = colors.black
styles["Title"].fontSize = 18
story = []
results = pd.read_csv(BASE / "validation_results.csv")
comparison = pd.read_csv(BASE / "regularization_comparison.csv")
successful = comparison[comparison.status == "converged"]
screen = pd.read_csv(BASE / "joint_screening_results.csv")


def body(text):
    story.append(Paragraph(text, styles["Body"]))


def heading(text):
    story.append(Paragraph(text, styles["Heading"]))


def table(rows, widths):
    item = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    item.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([item, Spacer(1, 8)])


def figure(name, caption, height=6.29):
    story.append(
        Image(str(BASE / "graphs" / f"{name}.png"), width=17 * cm, height=height * cm)
    )
    body(caption)


story.append(
    Paragraph("Polynomial Regression: Model Selection and Results", styles["Title"])
)
body("Roll number: <b>IMT2024069</b>")
heading("1. Problem and data")
body(
    "This assignment predicts the net power score of a turbine system (var1) and the thermal anomaly score of a reservoir (var2). Each problem has 1,000 labelled training rows and 1,000 unlabelled test rows. I modelled them separately, retaining all six input variables for var1 and all three for var2. Column names, missing values and finite numeric values were checked before training."
)
heading("2. Polynomial features and scaling")
body(
    "For a candidate degree d, I included all powers and interaction terms whose total degree does not exceed d. The intercept was fitted separately. This allows the model to capture interactions between inputs while keeping the predicted response polynomial in the original variables."
)
body(
    "I used <b>z-score standardization of the expanded polynomial terms</b>: z = (term - training mean) / training standard deviation. StandardScaler uses population variance (ddof=0); a constant term receives scale 1. The supplied raw inputs were used as received, and y was not normalized. Scaling after expansion makes the coefficient penalties comparable across terms of different magnitudes."
)
body(
    "The mean and scale were fitted only on the fitting portion of each split or CV fold, then applied to its validation rows. For final predictions, they were fitted on all training rows and applied unchanged to the test data. This avoids fitting preprocessing on validation or test data."
)
heading("3. Regularization and selection procedure")
body(
    "L1 (Lasso) penalizes absolute coefficient values and can remove terms. L2 (Ridge) penalizes squared coefficients and shrinks them without deliberately producing sparsity. Both use an unpenalized intercept and a squared-error data-fitting term. I selected the degree, penalty type and penalty strength together rather than choosing a degree first and assuming it stays best after regularization."
)
body(
    "The 1,000 training rows were split into 800 development rows and 200 holdout rows (seed 2024070). Within development, a 600/200 fitting-validation split (seed 2024069) screened every degree from 1 to 10 for var1 and 1 to 20 for var2 under both penalties. L1 alpha values were {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}; L2 values were {0.0001, 0.001, 0.01, 0.1, 1, 10, 100}."
)
body(
    "For each penalty, I shortlisted its four lowest-error degrees and up to three converged alpha values per degree. Each candidate was then evaluated using the same five shuffled development folds (seed 2024069). The lowest mean CV MSE determined the final configuration. Holdout scores were computed afterwards and did not determine the choice. Models that failed the convergence criterion were recorded and excluded."
)
story.append(PageBreak())
heading("4. How the degrees were chosen")
figure(
    "degree_screening",
    "<b>Figure 1.</b> Screening error over the full allowed degree ranges. Each point shows the lowest MSE among converged alpha values for that degree and penalty. Solid lines are L1 and dashed lines are L2. The star identifies the degree and penalty later chosen by CV. The log axis reveals both the improvement from adding useful terms and the point where extra complexity stops helping.",
)
figure(
    "degree_cv",
    "<b>Figure 2.</b> Five-fold CV results for the shortlisted degrees, with the best converged alpha for each degree and penalty. Error bars show one fold standard deviation, not a confidence interval. Screening and CV use different fitting sets, so their error values need not match.",
)
for r in results.itertuples():
    family = successful[
        (successful.dataset == r.dataset)
        & (successful.regularization == r.regularization)
    ]
    winners = family.loc[family.groupby("degree").cv_mse_mean.idxmin()].sort_values(
        "cv_mse_mean"
    )
    next_best = winners[winners.degree != r.degree].iloc[0]
    body(
        f"For <b>{r.dataset}</b>, degree <b>{r.degree}</b> with {r.regularization} and alpha {r.alpha:g} gave mean CV MSE <b>{r.cv_mse_mean:.4f}</b>. The next-best shortlisted degree under the same penalty was {int(next_best.degree)}, with MSE {next_best.cv_mse_mean:.4f}. This comparison supplies the evidence for the selected degree rather than relying on training fit alone."
    )
body(
    "These are the minimum-CV-error choices among the evaluated candidates. A finite alpha grid, screening shortlist and fold variability do not establish a unique global optimum. Nearly tied degrees should be interpreted as similarly competitive models."
)
story.append(PageBreak())
heading("5. Choosing between L1 and L2")
rows = [["Problem", "Penalty", "Best degree", "Alpha", "CV MSE"]]
for variant in ["var1", "var2"]:
    for family in ["L1", "L2"]:
        group = successful[
            (successful.dataset == variant) & (successful.regularization == family)
        ]
        r = group.loc[group.cv_mse_mean.idxmin()]
        rows.append(
            [variant, family, str(r.degree), f"{r.alpha:g}", f"{r.cv_mse_mean:.4f}"]
        )
table(rows, [2.5, 2.5, 3.5, 3, 3])
figure(
    "regularization_comparison",
    "<b>Figure 3.</b> Best validated configuration for each penalty and problem. Each bar uses its own selected degree and alpha; the error bars show one fold standard deviation. Thus the comparison permits L1 and L2 to prefer different degrees.",
)
rows = [["Problem", "Degree", "Penalty", "Alpha", "Terms", "Nonzero terms"]]
for r in results.itertuples():
    rows.append(
        [
            r.dataset,
            str(r.degree),
            r.regularization,
            f"{r.alpha:g}",
            str(r.terms),
            str(r.active_terms),
        ]
    )
table(rows, [2.5, 2, 2.5, 2, 2.5, 3])
for r in results.itertuples():
    body(
        f"The final <b>{r.dataset}</b> model uses <b>{r.regularization}</b>. Fitting all 1,000 training rows leaves {r.active_terms} nonzero coefficients out of {r.terms} expanded terms, excluding the intercept. The choice was based on validation error; coefficient sparsity is a consequence of the penalty, not an assumption that some original input variables are irrelevant."
    )
body(
    "Lasso screening uses up to 5,000 iterations with tolerance 0.0001; final CV and fitting use up to 30,000 iterations with tolerance 0.00001. Nonconverged candidates are excluded. Ridge is solved directly. The search therefore compares successfully fitted models rather than accepting unreliable optimizer outputs."
)
body(
    "Objective conventions: Ridge minimizes SSE + alpha * sum(w squared). Lasso minimizes SSE/(2n) + alpha * sum(abs(w)). Their alpha values are not numerically equivalent. Standardization is identical for both, and the intercept is excluded from the penalties."
)
story.append(PageBreak())
heading("6. Holdout evaluation and final predictions")
rows = [["Problem", "CV MSE", "CV R2", "Holdout MSE", "Holdout R2"]]
for r in results.itertuples():
    rows.append(
        [
            r.dataset,
            f"{r.cv_mse_mean:.4f}",
            f"{r.cv_r2_mean:.4f}",
            f"{r.holdout_mse:.4f}",
            f"{r.holdout_r2:.4f}",
        ]
    )
table(rows, [2.5, 3, 3, 3, 3])
figure(
    "holdout_predictions",
    "<b>Figure 4.</b> Predicted versus actual values for the 200 holdout rows. The dashed diagonal is perfect agreement. These predictions were made by models fitted on the 800 development rows.",
    height=5.8,
)
figure(
    "holdout_residuals",
    "<b>Figure 5.</b> Residuals (actual minus predicted) versus predicted values. The dashed line is zero error. Signed errors provide a check for structure that an average MSE may hide.",
    height=5.8,
)
body(
    "The holdout was excluded from this selection procedure, but these training rows had been used in earlier exploratory work. The scores are therefore diagnostic, not a completely unseen external evaluation. CV scores are also selection scores. Test labels were unavailable, so no hidden-test accuracy is claimed."
)
body(
    "The selected models were refitted on all 1,000 training rows. Each submission CSV contains one y column, 1,000 finite predictions and no index, in the original test order. compare_regularization.py reproduces joint selection and inference; plot_results.py and build_report.py regenerate the figures and this report. Configuration, screening, CV and holdout results are saved alongside the code."
)
body(
    'GitHub repository: <link href="https://github.com/Bhhavveshh/IMT2024069-ML-Assignment-1" color="black">https://github.com/Bhhavveshh/IMT2024069-ML-Assignment-1</link>'
)


def footer(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.black)
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))


SimpleDocTemplate(
    str(BASE / "IMT2024069_Assignment_1_Report.pdf"),
    pagesize=A4,
    leftMargin=2 * cm,
    rightMargin=2 * cm,
    topMargin=1.6 * cm,
    bottomMargin=1.6 * cm,
    title="Polynomial Regression - IMT2024069",
    author="IMT2024069",
).build(story, onFirstPage=footer, onLaterPages=footer)

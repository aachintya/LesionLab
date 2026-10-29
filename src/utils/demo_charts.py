"""Interactive displays of recorded scores and fixed evaluation results."""
import altair as alt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from utils.screening import POSITIVE_CLASSES, screening_scores

TEAL = "#137f83"
COPPER = "#ba6749"
GREY = "#929daa"


def finish(chart):
    return chart.properties(autosize=alt.AutoSizeParams(type="fit-x", contains="padding")).configure_view(stroke=None).configure_axis(
        labelColor="#536777", titleColor="#354b5c", gridColor="#e8edf1", labelFontSize=11,
        titleFontSize=11, domain=False, tickSize=0).configure_legend(
            title=None, orient="bottom", labelColor="#536777", labelFontSize=11)


def group_balance(screening):
    data = pd.DataFrame({"group": ["Malignant / precancerous", "Benign / inflammatory"],
                         "score": [screening["positive_score"], screening["negative_score"]], "row": ["Scores", "Scores"]})
    bars = alt.Chart(data).mark_bar(size=30).encode(
        x=alt.X("score:Q", stack="zero", scale=alt.Scale(domain=[0, 1]), title="Relative grouped score", axis=alt.Axis(tickCount=5, format=".2f")),
        y=alt.Y("row:N", title=None, axis=None),
        color=alt.Color("group:N", scale=alt.Scale(domain=data.group.tolist(), range=[COPPER, TEAL]),
                        legend=alt.Legend(columns=1)),
        order=alt.Order("group:N", sort="descending"),
        tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("score:Q", title="Model score", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"cutoff": [.5]})).mark_rule(color="#243f52", strokeDash=[4, 3], size=2).encode(x="cutoff:Q")
    return finish((bars + rule).properties(height=65))


def diagnosis_bars(rows):
    data = pd.DataFrame(rows).copy()
    data["family"] = data["class"].map(lambda code: "Malignant / precancerous" if code in POSITIVE_CLASSES else "Benign / inflammatory")
    return finish(alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("score:Q", title="Relative class score", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("class:N", title=None, sort="-x", axis=alt.Axis(labelLimit=120)),
        color=alt.Color("family:N", scale=alt.Scale(domain=["Malignant / precancerous", "Benign / inflammatory"], range=[COPPER, TEAL]), legend=None),
        tooltip=[alt.Tooltip("class:N", title="Code"), alt.Tooltip("name:N", title="Category"),
                 alt.Tooltip("score:Q", title="Model score", format=".3f")]).properties(height=265))


def comparison_bars(rows):
    data = pd.DataFrame(rows)
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3, size=23).encode(
        x=alt.X("score:Q", title="Malignant / precancerous grouped score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=5, format=".2f")),
        y=alt.Y("pipeline:N", sort=data.pipeline.tolist(), title=None),
        color=alt.Color("pipeline:N", scale=alt.Scale(domain=data.pipeline.tolist(), range=[GREY, TEAL]), legend=None),
        tooltip=["pipeline:N", alt.Tooltip("score:Q", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"cutoff": [.5]})).mark_rule(color="#243f52", strokeDash=[4, 3]).encode(x="cutoff:Q")
    return finish((bars + rule).properties(height=max(80, 45*len(data))))


def modality_heatmap(branches, class_order):
    labels = {"clinical": "Clinical", "dermoscopic": "Dermoscopy", "metadata": "Metadata"}
    data = pd.DataFrame([{"modality": labels[name], "class": code, "score": float(value)}
                         for name, values in branches.items() for code, value in zip(class_order, values)])
    return finish(alt.Chart(data).mark_rect(cornerRadius=2).encode(
        x=alt.X("class:N", sort=list(class_order), title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("modality:N", sort=list(labels.values()), title=None),
        color=alt.Color("score:Q", scale=alt.Scale(domain=[0, 1], scheme="blues"), legend=alt.Legend(title="Relative score", gradientLength=150)),
        tooltip=["modality:N", "class:N", alt.Tooltip("score:Q", format=".3f")]).properties(height=125))


def evaluation_curves(frame, *, precision_recall=False):
    rows = []
    for column, label in [("saved_original_score", "Original fusion"), ("improved_ensemble_score", "Improved ensemble")]:
        if precision_recall:
            precision, recall, _ = precision_recall_curve(frame.positive_reference, frame[column])
            x, y = recall, precision
        else:
            x, y, _ = roc_curve(frame.positive_reference, frame[column])
        rows.extend({"x": float(a), "y": float(b), "pipeline": label, "point": i} for i, (a, b) in enumerate(zip(x, y)))
    chart = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2.5).encode(
        x=alt.X("x:Q", title="Recall" if precision_recall else "False-positive rate", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("y:Q", title="Precision" if precision_recall else "Sensitivity", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("pipeline:N", scale=alt.Scale(domain=["Original fusion", "Improved ensemble"], range=[GREY, TEAL])),
        order="point:Q", tooltip=["pipeline:N", alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("y:Q", format=".3f")])
    return finish(chart.properties(height=250))


def confusion_chart(counts):
    data = pd.DataFrame([
        {"reference": "Benign / inflammatory", "prediction": "Benign / inflammatory", "count": counts["true_negative"]},
        {"reference": "Benign / inflammatory", "prediction": "Malignant / precancerous", "count": counts["false_positive"]},
        {"reference": "Malignant / precancerous", "prediction": "Benign / inflammatory", "count": counts["false_negative"]},
        {"reference": "Malignant / precancerous", "prediction": "Malignant / precancerous", "count": counts["true_positive"]}])
    order = ["Benign / inflammatory", "Malignant / precancerous"]
    base = alt.Chart(data).encode(x=alt.X("prediction:N", sort=order, title="Predicted group",
                                       axis=alt.Axis(labelAngle=0, labelLimit=190, labelExpr="split(datum.label, ' / ')")),
                                  y=alt.Y("reference:N", sort=order, title="Recorded group",
                                          axis=alt.Axis(labelLimit=190, labelExpr="split(datum.label, ' / ')")))
    return finish((base.mark_rect(cornerRadius=5).encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="tealblues"), legend=None)) +
                   base.mark_text(fontSize=23, fontWeight=600).encode(text="count:Q", color=alt.condition(alt.datum.count > 250, alt.value("white"), alt.value("#203d50"))))
                  .properties(height=210))

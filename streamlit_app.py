"""Streamlit demonstration with a fixed screening policy and read-only hosted mode."""
from __future__ import annotations

import hashlib
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from utils.app_inference import CHECKPOINT_DIR, LesionPredictor, checkpoint_signature, missing_checkpoints
from utils.input_validation import InputRejected
from utils.model_artifacts import REQUIRED_FILES, install_artifacts
from utils.result_interpretation import INTERPRETATION_VERSION
from utils.demo_ensemble import ENSEMBLE_DIR, ensemble_signature, missing_ensemble
from utils.screening import SCREENING_VERSION, screening_scores, presentation_screening
from utils import demo_charts

DISPLAY_VERSION = "3.1"
HOSTED = os.environ.get("LESIONLAB_HOSTED", "0") == "1"

st.set_page_config(page_title="LesionLab · Skin lesion research", page_icon="🔬", layout="wide")
st.markdown("""<style>
  .stApp { background: linear-gradient(155deg, #f8faf9 0%, #f1f5f8 75%); }
  [data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 2.5rem; padding-bottom: 4rem; }
  h1 { letter-spacing: -.045em; font-weight: 750 !important; }
  h2, h3 { letter-spacing: -.025em; }
  [data-testid="stSidebar"] { border-right: 1px solid #e2e8ed; }
  [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 16px; background: #ffffffb8; }
  [data-testid="stFileUploader"] section { border: 1px dashed #9dbac1; border-radius: 12px; }
  [data-testid="stMetric"] { background: white; border: 1px solid #e1e8ed; border-radius: 14px; padding: 18px; }
  .eyebrow { color: #087f8c; letter-spacing: .15em; font-size: .72rem; font-weight: 750; margin-bottom: 8px; }
  .brand { font-size: 1.6rem; font-weight: 800; letter-spacing: -.05em; color: #173344; }
  .brand span { color: #087f8c; }
  .empty-result { padding: 58px 24px; text-align: center; background: #fff; border: 1px solid #e1e8ed; border-radius: 16px; }
  .empty-mark { display: inline-flex; width: 52px; height: 52px; align-items: center; justify-content: center; border-radius: 16px; background: #e7f4f4; color: #087f8c; font-size: 1.5rem; margin-bottom: 18px; }
  .empty-result h3 { font-size: 1.12rem; margin: 0 0 8px; }
  .empty-result p { color: #69808e; font-size: .93rem; }
  .step-label { color: #778c98; font-size: .72rem; letter-spacing: .10em; font-weight: 700; }
  .stButton > button { border-radius: 10px; min-height: 44px; }
  [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid #dfe7eb; margin-bottom: 1rem; }
  [data-baseweb="tab"] { font-weight: 650; padding-left: 0; padding-right: 0; }
  [data-testid="stMetricValue"] { font-size: 1.65rem; letter-spacing: -.04em; }
  .case-strip { color: #456273; font-size: .78rem; letter-spacing: .02em; padding: 8px 0 18px; }
  .screen-positive { border-left: 4px solid #ba6749; padding: 15px 20px; background: #fbf0e9; border-radius: 4px 12px 12px 4px; }
  .screen-negative { border-left: 4px solid #137f83; padding: 15px 20px; background: #eaf5f3; border-radius: 4px 12px 12px 4px; }
  .screen-positive h3, .screen-negative h3 { margin: 3px 0 6px; font-size: 1.22rem; }
  .screen-positive p, .screen-negative p { margin: 0; color: #456273; font-size: .86rem; }
  .hero-note { display: inline-block; border: 1px solid #cbdfe0; border-radius: 30px; padding: 6px 12px; color: #35636b; font-size: .78rem; background: #eff8f6; margin: 5px 0 12px; }
  .result-tag { color: #456273; font-size: .75rem; letter-spacing: .08em; font-weight: 700; margin: 12px 0 4px; }
  @media (max-width: 700px) {
    [data-testid="stMainBlockContainer"] { padding: 1.5rem 1rem 3rem; }
    h1 { font-size: 2.1rem !important; }
    [data-baseweb="tab-list"] { gap: 15px; }
  }
</style>
""", unsafe_allow_html=True)

MODES = {"Full multimodal": "fusion", "Clinical photo": "clinical",
         "Dermoscopic image": "dermoscopic", "Image check only": "validator"}


def open_setup():
    st.session_state["workspace_page"] = "Model setup"


def clear_case():
    for key in tuple(st.session_state):
        if key != "workspace_page":
            del st.session_state[key]
    st.session_state["workspace_page"] = "Analyze"


@st.cache_resource(show_spinner=False)
def predictor(directory, signature, pipeline="original", display_version=DISPLAY_VERSION, hosted=False):
    # signature intentionally participates in the key when imported weights change.
    import torch
    torch.set_num_threads(2)
    return LesionPredictor(directory, device="cpu" if hosted else "auto",
                           ensemble_dir=ENSEMBLE_DIR if pipeline == "ensemble" else None)


def read_report(name):
    path = CHECKPOINT_DIR / name
    if path.is_file() and path.stat().st_size <= 2 * 1024**2:
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            pass
    return None


def category_options(column):
    encoder = read_report("metadata_encoder.json") or {}
    return [None, *sorted(key for key in encoder.get("category_maps", {}).get(column, {})
                          if key not in ("unknown", "nan"))]


def save_upload(upload, folder, name):
    data = upload.getvalue()
    if not data or len(data) > 20 * 1024**2:
        raise ValueError("Choose an image smaller than 20 MB.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Choose a JPEG, PNG, or WebP photograph.")
            if image.width * image.height > 40_000_000:
                raise ValueError("Choose an image smaller than 40 megapixels.")
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("This file is not a readable photograph.") from exc
    path = folder / f"{name}.jpg"
    path.write_bytes(data)
    return path


def upload_widget(label, key):
    upload = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=key, max_upload_size=20)
    if upload is not None:
        try:
            data = upload.getvalue()
            if len(data) <= 20 * 1024**2:
                with Image.open(BytesIO(data)) as source:
                    if source.width * source.height <= 40_000_000:
                        preview = ImageOps.exif_transpose(source).convert("RGB")
                        preview.thumbnail((620, 330))
                        st.image(preview, caption=label)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            st.caption("The file could not be previewed. Analyze will check the upload.")
    return upload


def render_diagnoses(result):
    interpretation = result["interpretation"]
    with st.container(border=True):
        st.caption("11-CLASS INTERPRETATION")
        st.subheader("Inconclusive result" if interpretation["status"] == "inconclusive" else "Model preference")
        st.write(interpretation["summary"])
        for reason in interpretation["reasons"]:
            st.markdown(f"• {reason['text']}")
    st.altair_chart(demo_charts.diagnosis_bars(result["scores"]), width="stretch")
    st.caption("All eleven class scores and their ranking are unchanged. Scores are uncalibrated model outputs.")
    with st.expander("Scores and interpretation rules"):
        st.dataframe(pd.DataFrame(result["scores"]).sort_values("score", ascending=False), hide_index=True, width="stretch")
        st.write(interpretation["policy"]["description"])


def render_screening(result):
    screening = result.get("presentation_screening") or presentation_screening(result)
    positive = screening["positive"]
    style = "screen-positive" if positive else "screen-negative"
    st.markdown(f'<div class="{style}"><p>BROADER LESION GROUP</p>'
                f'<h3>{screening["label"]} favoured</h3>'
                '<p>Based on the model’s leading diagnostic category.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="result-tag">LEADING CATEGORY</div>', unsafe_allow_html=True)
    st.subheader(result["class_name"])
    interpretation = result["interpretation"]
    if interpretation["status"] == "inconclusive":
        st.warning("The exact diagnosis is inconclusive. This family is a model preference; review the 11-class result and image evidence.")
    else:
        st.write(interpretation["summary"])
    st.caption("One consistent rule: group the leading 11-class diagnosis. The family label is not a personal cancer-risk estimate.")
    reference = result.get("baseline_comparison")
    if reference:
        with st.expander("Compare with the original model"):
            first, second = st.columns(2)
            for column, name, label, family in [
                (first, "Original fusion", reference["class_name"], reference["screening"]["mapped_top_diagnosis"]["label"]),
                (second, "Improved ensemble", result["class_name"], screening["label"])]:
                with column, st.container(border=True):
                    st.caption(name)
                    st.markdown(f"**{label}**")
                    st.caption(family)
            st.caption("Same supplied images and metadata; same leading-diagnosis grouping rule.")
    with st.expander("What the screening groups mean"):
        st.write("Malignant / precancerous: " + ", ".join(screening["groups"]["positive"]) + ".")
        st.write("Benign / inflammatory: " + ", ".join(screening["groups"]["negative"]) + ".")
        st.caption("AKIEC includes actinic keratosis and intraepidermal carcinoma; SCCKA includes squamous cell carcinoma and keratoacanthoma. "
                   "The positive group therefore includes mixed and precancerous diagnostic families.")
        st.caption("The demo always uses the leading-diagnosis rule. Archived score-pooling experiments remain in the research details.")
    if result["mode"] != "fusion":
        st.caption("This screen uses a single image specialist. The paired multimodal validation metrics do not apply to this mode.")


def render_evidence(result):
    interpretation = result["interpretation"]
    st.write(interpretation["agreement_summary"])
    columns = st.columns(len(interpretation["modalities"]))
    for column, row in zip(columns, interpretation["modalities"]):
        with column, st.container(border=True):
            st.caption(row["label"])
            st.markdown(f"**{row['name']}**")
            st.caption("Matches combined diagnosis" if row["matches_combined"] else "Different preference")
    st.caption(interpretation["metadata_note"])
    if result["mode"] == "fusion":
        st.altair_chart(demo_charts.modality_heatmap(result["branch_scores"], [row["class"] for row in result["scores"]]), width="stretch")
        st.caption("Single-view specialist scores before ensemble combination. These are the branches explained by the image heatmaps.")
    if interpretation.get("ensemble_components"):
        st.markdown("**Inside the ensemble**")
        st.write(interpretation["ensemble_agreement"])
        for component in interpretation["ensemble_components"]:
            st.write(f"{component['label']}: **{component['name']}**")
        st.altair_chart(demo_charts.comparison_bars([
            {"pipeline": row["label"], "score": screening_scores(result["ensemble_members"][row["key"]])["positive_score"]}
            for row in interpretation["ensemble_components"]]), width="stretch")
        st.caption("Equal geometric combination of the feature classifier and six-view image average. Metadata enter through the feature classifier.")


def render_result(record):
    st.markdown('<div class="step-label">02 / RESULTS</div>', unsafe_allow_html=True)
    st.subheader("Case summary")
    if record is None:
        st.markdown('''<div class="empty-result"><div class="empty-mark">◎</div>
            <h3>Understand the model's findings</h3><p>Get a clear summary, compare the image views,<br>
            and explore visual explanations.</p></div>''', unsafe_allow_html=True)
        return
    if record["status"] == "rejected":
        st.error(record["message"], icon="🚫")
        st.caption("No disease classification was run for this request.")
        return
    if record["status"] == "error":
        st.warning(record["message"])
        return
    result = record["result"]
    st.success("Input check passed")
    if result["mode"] == "validator":
        st.write("The validator accepted this as a supported-looking lesion photograph.")
        st.caption("This check does not determine a disease or prove that a lesion is present.")
    else:
        st.caption(result.get("pipeline", {}).get("label", "Saved model"))
        screen, diagnoses, evidence = st.tabs(["Broader screening", "11-class result", "Model evidence"])
        with screen:
            render_screening(result)
        with diagnoses:
            render_diagnoses(result)
        with evidence:
            render_evidence(result)
    report = {**result, "heatmaps": {key: {k: v for k, v in value.items() if not k.endswith("_base64")}
                                   for key, value in result.get("heatmaps", {}).items()}}
    if result["mode"] != "validator":
        report["schema_version"] = DISPLAY_VERSION
        report["research_screening"] = result["screening"]
        report["screening"] = result.get("presentation_screening") or presentation_screening(result)
    st.download_button("Download analysis JSON", json.dumps(report, indent=2),
                       file_name="lesion_analysis.json", mime="application/json")


def render_heatmaps(record):
    if (record is None or record["status"] != "accepted"
            or not record["result"].get("heatmaps_requested")):
        return
    heatmaps = record["result"].get("heatmaps", {})
    if not heatmaps:
        return
    st.divider()
    st.markdown('<div class="step-label">03 / VISUAL EXPLANATIONS</div>', unsafe_allow_html=True)
    st.subheader("Where each image model looked")
    st.write("Each map highlights regions influencing that image model's score for its own leading category.")
    st.caption("Maps use the processed crop seen by the model. They explain individual image branches; they are not lesion boundaries or explanations of the combined fusion decision.")
    columns = st.columns(len(heatmaps), gap="large")
    for column, (name, explanation) in zip(columns, heatmaps.items()):
        with column, st.container(border=True):
            st.markdown("**Clinical photograph**" if name == "clinical" else "**Dermoscopic image**")
            st.caption(f"Category explained: {explanation['class_name']}")
            if explanation["status"] != "available":
                st.info(explanation["message"])
                continue
            before, after = st.columns(2)
            before.image(BytesIO(base64.b64decode(explanation["input_png_base64"])),
                         caption="Processed model input", width="stretch")
            overlay = base64.b64decode(explanation["overlay_png_base64"])
            after.image(BytesIO(overlay), caption="Grad-CAM overlay", width="stretch")
            st.caption("Overlaid warm colours highlight higher relative positive influence on this category's score. Each map has its own scale.")
            st.download_button("Download heatmap", overlay, file_name=f"{name}_gradcam.png", mime="image/png", key=f"download_heatmap_{name}")


def analyze_page():
    st.markdown('<div class="eyebrow">MULTIMODAL LESION RESEARCH</div>', unsafe_allow_html=True)
    heading, actions = st.columns([.85, .15], vertical_alignment="center")
    with heading:
        st.title("A closer look at every lesion.")
    with actions:
        st.button("Clear case", on_click=clear_case, help="Remove this session's inputs, previews and results.")
    st.write("Two image views, patient context, and one place to explore the model's findings.")
    st.caption("Research demonstration · Clinical photography + dermoscopy + patient context")
    ready = not missing_checkpoints()
    if not ready:
        if HOSTED:
            st.info("The model service is not ready. Please contact the app owner.")
            return
        if not missing_checkpoints(CHECKPOINT_DIR, "validator"):
            st.info("The input validator is available. Import your Kaggle ZIP in Model setup to unlock disease-model analysis.")
        else:
            st.info("Import your Kaggle ZIP in Model setup to enable image validation and model analysis.")
        st.button("Set up trained models", on_click=open_setup)
    with st.expander("Analysis options", expanded=not HOSTED):
        selection = st.radio("Analysis mode", list(MODES), horizontal=True,
                             index=0 if ready else 3, key="analysis_mode")
    mode = MODES[selection]
    pipeline = "original"
    if mode == "fusion":
        if HOSTED:
            pipeline = "ensemble"
        else:
            choice = st.selectbox("Multimodal pipeline", ["Improved feature/TTA ensemble", "Saved original fusion"],
                                  key="fusion_pipeline")
            pipeline = "ensemble" if choice.startswith("Improved") else "original"
    left, right = st.columns([.88, 1.2], gap="large")
    uploads, metadata, pair_confirmed, include_heatmaps = {}, {}, True, False
    with left:
        st.markdown('<div class="step-label">01 / INPUTS</div>', unsafe_allow_html=True)
        st.subheader("Add your photograph" if mode != "fusion" else "Add both views")
        with st.container(border=True):
            if mode == "fusion":
                st.caption("Use a clinical close-up and a dermoscopic image of the same lesion.")
                uploads["clinical"] = upload_widget("Clinical close-up photograph", "clinical_upload")
                uploads["dermoscopic"] = upload_widget("Dermoscopic photograph", "dermoscopic_upload")
                pair_confirmed = st.checkbox("Both photographs show the same lesion", key="same_lesion")
            else:
                key = "image" if mode == "validator" else mode
                label = "Skin lesion photograph" if mode == "validator" else selection
                uploads[key] = upload_widget(label, f"single_{mode}")
            st.caption("JPEG, PNG or WebP · Up to 20 MB per image · " +
                       ("Processed on the app server" if HOSTED else "Processed on this computer"))
        if mode == "fusion":
            with st.expander("Patient metadata · optional", expanded=True):
                st.caption("Leave fields blank when unknown. No identifying information is needed.")
                one, two = st.columns(2)
                metadata["age_approx"] = one.number_input("Approximate age", min_value=0, max_value=120,
                                                         value=None, step=5, placeholder="Unknown", key="metadata_age")
                metadata["sex"] = two.selectbox("Sex", category_options("sex"),
                                               format_func=lambda value: "Not provided" if value is None else value.title(), key="metadata_sex")
                metadata["site"] = one.selectbox("Body site", category_options("site"),
                                                format_func=lambda value: "Not provided" if value is None else value.replace("_", " ").title(), key="metadata_site")
                metadata["skin_tone_class"] = two.selectbox("Skin tone category", [None, 0, 1, 2, 3, 4, 5],
                    format_func=lambda value: "Not provided" if value is None else
                        f"{value}" + (" · darkest" if value == 0 else " · lightest" if value == 5 else ""), key="metadata_skin_tone")
                st.caption("Skin tone uses MILK10k's 0–5 categories, rather than Fitzpatrick skin types.")
        available = not missing_checkpoints(CHECKPOINT_DIR, mode)
        if pipeline == "ensemble" and missing_ensemble():
            available = False
            st.warning("The improved ensemble package is missing. Restore demo/ensemble-v1 to use this pipeline.")
        if not available:
            st.warning("This mode needs trained checkpoints. Import your ZIP in Model setup.")
        if mode != "validator":
            include_heatmaps = st.checkbox("Include model attention heatmaps", value=True, key="include_heatmaps",
                                            help="Shows which parts of the processed image influenced each image model's leading category.")
        submitted = st.button("Check image" if mode == "validator" else "Analyze lesion", type="primary",
                              width="stretch",
                              disabled=not available or not pair_confirmed or any(value is None for value in uploads.values()))
    signature = (*checkpoint_signature(), *ensemble_signature()) if pipeline == "ensemble" else checkpoint_signature()
    fingerprint = hashlib.sha256(repr((mode, pipeline, metadata, pair_confirmed, include_heatmaps, signature, INTERPRETATION_VERSION, SCREENING_VERSION, DISPLAY_VERSION)).encode())
    for name, upload in uploads.items():
        fingerprint.update(name.encode())
        if upload is not None:
            fingerprint.update(upload.getvalue())
    request_key = fingerprint.hexdigest()
    if submitted:
        with st.spinner("Checking the original image" if mode == "validator" else "Checking inputs and running the model…"):
            try:
                with tempfile.TemporaryDirectory(prefix="lesionlab-") as folder:
                    paths = {name: save_upload(upload, Path(folder), name) for name, upload in uploads.items()}
                    result = predictor(str(CHECKPOINT_DIR), signature, pipeline, DISPLAY_VERSION, HOSTED).analyze(paths, mode=mode, metadata=metadata,
                                                                             include_heatmaps=include_heatmaps)
                record = {"status": "accepted", "result": result}
            except InputRejected as exc:
                record = {"status": "rejected", "message": exc.result.message}
            except Exception as exc:
                if HOSTED:
                    import logging
                    logging.getLogger("lesionlab").error("Analysis failed (%s)", type(exc).__name__)
                record = {"status": "error", "message": "Analysis could not finish. Try a smaller image or a different photograph."
                          if HOSTED else str(exc)}
            st.session_state["analysis_record"] = (request_key, record)
    saved_key, record = st.session_state.get("analysis_record", (None, None))
    with right:
        render_result(record if saved_key == request_key else None)
    render_heatmaps(record if saved_key == request_key else None)


def results_page():
    st.markdown('<div class="eyebrow">MEASURED PERFORMANCE</div>', unsafe_allow_html=True)
    st.title("Evidence behind the demo.")
    st.write("One screening policy. A matched comparison. The full diagnostic results.")
    st.caption("Recorded development evaluation · Validation and historical holdout are reported separately")
    screening_tab, diagnosis_tab, historical_tab = st.tabs(["Broader screening", "11-class validation", "Historical holdout"])
    with screening_tab:
        path = ROOT / "demo/evaluation/screening.json"
        if not path.is_file():
            st.info("The recorded screening evaluation is unavailable.")
        else:
            evaluation = json.loads(path.read_text())["evaluations"]["risk_screening_all"]
            models = {name: values["mapped_11class_argmax"] for name, values in evaluation["models"].items()}
            improved, original = models["improved_ensemble"], models["saved_original"]
            st.markdown('<span class="hero-note">Demo policy · group the leading diagnosis</span>', unsafe_allow_html=True)
            a, b, c = st.columns(3)
            for column, label, key in [(a,"Screening accuracy","accuracy"), (b,"Positive-group sensitivity","sensitivity"), (c,"Specificity","specificity")]:
                column.metric(label, f"{improved[key]:.2%}", delta=f"{100*(improved[key]-original[key]):+.2f} pp vs original")
            st.caption("Same 786 validation lesions · 562 malignant / precancerous-family and 224 benign / inflammatory-family cases")
            st.subheader("Original model versus improved ensemble")
            rows = [{"Pipeline": label, **{title: values[key] for title,key in [
                     ("Accuracy","accuracy"),("Balanced accuracy","balanced_accuracy"),
                     ("Sensitivity","sensitivity"),("Specificity","specificity"),("Macro F1","macro_f1")]}}
                    for name,label in [("saved_original","Original fusion"),("improved_ensemble","Improved ensemble")]
                    for values in [models[name]]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                key: st.column_config.NumberColumn(format="%.3f") for key in rows[0] if key != "Pipeline"})
            st.caption("Accuracy rises from 80.41% to 82.70%. Specificity improves by 15.18 percentage points; sensitivity decreases by 2.85 points.")
            counts = improved["confusion"]
            left, right = st.columns([1.2, 1])
            with left:
                st.markdown("**Improved ensemble · decision counts**")
                st.altair_chart(demo_charts.confusion_chart(counts), width="stretch")
            with right:
                st.metric("Positive lesions detected", f"{counts['true_positive']} / 562")
                st.metric("Positive lesions missed", str(counts["false_negative"]))
                st.metric("Benign / inflammatory false alarms", str(counts["false_positive"]))
            st.caption("Melanoma maps to the positive family in 41/67 cases (61.19%); the overall family sensitivity includes the other diagnoses. "
                       "AKIEC and SCCKA contain precancerous or mixed categories.")
            with st.expander("Research details · archived score-pooling analysis"):
                st.write("A separate experiment summed positive-family scores at a fixed 0.50 cutoff. "
                         "It is preserved for reproducibility and is not the demo's decision policy.")
                st.caption("Pooled-score accuracy: original 40.97%, ensemble 83.72%. The original scores are uncalibrated; "
                           "this threshold-specific gap should not be presented as a matched binary-training improvement.")
                predictions = pd.read_csv(ROOT / "demo/evaluation/screening_predictions.csv")
                left, right = st.columns(2)
                with left:
                    st.markdown("**Continuous group-score ROC**")
                    st.altair_chart(demo_charts.evaluation_curves(predictions), width="stretch")
                with right:
                    st.markdown("**Continuous group-score precision–recall**")
                    st.altair_chart(demo_charts.evaluation_curves(predictions, precision_recall=True), width="stretch")
                st.caption("These curves assess continuous summed-score ranking, rather than the demo's category-grouping decisions.")
    comparison = ROOT / "demo/evaluation/validation_comparison.csv"
    if comparison.is_file():
        table = pd.read_csv(comparison)
        selected = table[table.method.isin(["Saved original fusion", "Feature LR + six-view ensemble"])].copy()
        ensemble = selected[selected.method == "Feature LR + six-view ensemble"].iloc[0]
        with diagnosis_tab:
            a, b, c = st.columns(3)
            a.metric("Ensemble validation accuracy", f"{ensemble.accuracy:.2%}")
            b.metric("Validation balanced accuracy", f"{ensemble.balanced_accuracy:.2%}")
            c.metric("Validation macro F1", f"{ensemble.macro_f1:.4f}")
            st.caption("Same 786 validation lesions. Eleven-class scores and evaluation are unchanged; broader grouping is a separate endpoint.")
            st.dataframe(selected[["method", "accuracy", "balanced_accuracy", "macro_f1"]].rename(
                columns={"method": "Pipeline", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1"}),
                hide_index=True, width="stretch")
    with historical_tab:
        render_historical_results()


def render_historical_results():
    st.subheader("Saved original: historical holdout")
    report = read_report("best_fusion_lr.metrics.json")
    if report is None:
        st.info("Import training_artifacts.zip in Model setup to view the recorded evaluation.")
        return
    a, b, c = st.columns(3)
    a.metric("Holdout accuracy", f"{report['accuracy']:.2%}")
    b.metric("Balanced accuracy", f"{report['balanced_accuracy']:.2%}")
    c.metric("Macro F1", f"{report['macro_f1']:.4f}")
    st.caption(f"{report.get('evaluation_lesions', 'Unknown')} lesions · {report.get('evaluation_split', 'holdout')} split · "
               "Internal evaluation; not an official leaderboard score. Image-only modes have separate performance.")
    screening_path = ROOT / "demo/evaluation/historical_holdout.json"
    if screening_path.is_file():
        screening = json.loads(screening_path.read_text())
        st.subheader("Historical broader screening · JI-ADF comparison")
        st.caption("Same 524 archived holdout lesions · Group each existing 11-class prediction · "
                   "376 malignant / precancerous and 148 benign / inflammatory cases")
        rows = [{"Pipeline": name, **{key: scores[key] for key in
                ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "macro_f1")}}
                for name, scores in screening["models"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={key: st.column_config.NumberColumn(format="%.3f")
                                    for key in rows[0] if key != "Pipeline"})
        st.caption("Our JI-ADF adaptation: 77.86% accuracy; original late fusion: 82.25%. "
                   "Sensitivity changes from 94.15% to 93.62%, while specificity rises from 36.49% to 53.38%. "
                   "These are our archived adaptation results, not the paper's published scores. "
                   "The improved ensemble's validation results use a separate cohort.")
    rows, histories = [], []
    for model in ("clinical", "dermoscopic", "metadata"):
        run = read_report(f"{model}_resume.json")
        if run:
            rows.append({"Model": model.title(), "Completed epochs": run["epoch"] + 1,
                         "Best validation score": run["best_score"], "Complete": run["complete"]})
            for entry in run.get("history", []):
                histories.append({"Epoch": entry["epoch"], "Model": model.title(), "Validation F1": entry["val_f1"]})
    if rows:
        st.subheader("Training summary")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if histories:
        st.subheader("Validation F1 by epoch")
        chart = pd.DataFrame(histories).pivot(index="Epoch", columns="Model", values="Validation F1")
        st.line_chart(chart)
    with st.expander("Evaluation details"):
        st.json(report)


def setup_page():
    if HOSTED:
        st.info("Model management is available only in the local research workspace.")
        return
    st.markdown('<div class="eyebrow">MODEL SETUP</div>', unsafe_allow_html=True)
    st.title("Bring your trained models home.")
    st.write("Import the training_artifacts.zip generated at the end of your Kaggle notebook.")
    if "import_message" in st.session_state:
        st.success(st.session_state.pop("import_message"))
    with st.container(border=True):
        st.markdown("**1. Download the output from Kaggle**")
        st.write("Click the training_artifacts.zip link under Results and download in your notebook.")
        st.markdown("**2. Select your training ZIP**")
        method = st.radio("Import method", ["Local file path", "Choose ZIP file"], horizontal=True)
        source = None
        if method == "Local file path":
            typed = st.text_input("ZIP location", placeholder="/Users/yourname/Downloads/training_artifacts.zip")
            if typed:
                source = Path(typed).expanduser()
        else:
            source = st.file_uploader("Your Kaggle training output", type=["zip"], key="artifacts_zip")
        st.caption("Import your own training output. Existing models are backed up; optimizer states and datasets are skipped.")
        if st.button("Import trained models", type="primary", disabled=source is None):
            try:
                with st.spinner("Checking and installing the model checkpoints…"):
                    installed = install_artifacts(source, CHECKPOINT_DIR)
                predictor.clear()
                st.session_state.pop("analysis_record", None)
                st.session_state.pop("analysis_mode", None)
                st.session_state["import_message"] = f"Imported {len(installed['installed'])} files. Open Analyze to use your trained models."
                st.rerun()
            except (ValueError, OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
                st.error(f"Import could not finish: {exc}")
    st.subheader("Checkpoint status")
    labels = {"input_validator.pt": "Input validator", "best_clinical.pt": "Clinical image model",
              "best_dermoscopic.pt": "Dermoscopic image model", "best_metadata.pt": "Metadata model",
              "metadata_encoder.json": "Metadata encoder", "best_fusion_lr.joblib": "Fusion model"}
    st.dataframe(pd.DataFrame([{"Component": labels[name], "Status": "Ready" if (CHECKPOINT_DIR / name).is_file() else "Awaiting import",
                               "File": name} for name in REQUIRED_FILES]), hide_index=True, width="stretch")


def about_page():
    st.markdown('<div class="eyebrow">ABOUT THE PROJECT</div>', unsafe_allow_html=True)
    st.title("Three inputs. One research pipeline.")
    st.write("LesionLab combines a clinical photograph, a dermoscopic photograph and de-identified patient context "
             "to explore eleven skin-lesion categories and their broader diagnostic families.")
    one, two, three = st.columns(3)
    for column, label, text in [(one,"Clinical image","EfficientNetV2-S close-up image features"),
                               (two,"Dermoscopy","EfficientNetV2-S dermoscopic image features"),
                               (three,"Patient context","Approximate age, sex, body site and skin-tone category")]:
        with column, st.container(border=True):
            st.markdown(f"**{label}**"); st.write(text)
    st.subheader("How the result is formed")
    st.write("An input validator checks each photograph before classification. The frozen-feature and six-view ensemble "
             "produces eleven class scores. Broader screening maps the leading category to its family. "
             "The models and saved baseline are unchanged; the demo performs no training.")
    st.caption("The positive family includes AKIEC, BCC, MAL_OTH, MEL and SCCKA. It includes precancerous and mixed diagnostic categories. "
               "Individual model scores are uncalibrated; this research demo supports exploration, not diagnosis or treatment decisions.")
    st.subheader("Evaluation")
    st.write("The Performance page reports complete, fixed evaluation cohorts. "
             "Analyze accepts your uploaded research photographs; the repository includes no preselected example gallery.")
    st.subheader("Your session")
    st.write("Uploads are processed on the app server." if HOSTED else "Images are processed on this computer.")
    st.write("Temporary analysis files are removed after each run. Previews, uploaded files and results can remain in your active "
             "Streamlit session until you clear the case or the session ends. Images are not added to the training data. "
             "Use de-identified research images for this demonstration.")
    report = ROOT / "report/LesionLab_Report.pdf"
    if report.is_file():
        st.download_button("Download research report", report.read_bytes(), file_name="MILK10k_Research_Report.pdf", mime="application/pdf")


with st.sidebar:
    st.markdown('<div class="brand">Lesion<span>Lab.</span></div>', unsafe_allow_html=True)
    st.caption("Multimodal skin lesion research")
    st.divider()
    pages = ["Analyze", "Performance", "About"] + ([] if HOSTED else ["Model setup"])
    if st.session_state.get("workspace_page") not in (None, *pages):
        st.session_state["workspace_page"] = "Analyze"
    page = st.radio("Workspace", pages,
                    label_visibility="collapsed", key="workspace_page")
    st.divider()
    st.markdown("**Model availability**")
    for label, mode in (("Input validator", "validator"), ("Clinical image", "clinical"),
                        ("Dermoscopic image", "dermoscopic"), ("Full multimodal", "fusion")):
        st.caption(("●  " if not missing_checkpoints(CHECKPOINT_DIR, mode) else "○  ") + label)
    st.divider()
    st.caption("PRESENTATION DEMO" if HOSTED else "LOCAL RESEARCH WORKSPACE")
    st.caption("Uploads are processed on the app server. Clear your case to remove session previews."
               if HOSTED else "Images are processed on this computer. Clear your case to remove session previews.")
    st.caption("Research use · Not for diagnosis")

if page == "Analyze":
    analyze_page()
elif page == "Performance":
    results_page()
elif page == "About":
    about_page()
else:
    setup_page()

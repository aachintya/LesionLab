# LesionLab

Multimodal skin-lesion research using clinical photographs, dermoscopy and patient
metadata. The main method is an EfficientNetV2-S frozen-feature/six-view ensemble.

## Project contents

| Path | Purpose |
| --- | --- |
| `src/` | Models, preprocessing, training, inference and fusion code |
| `notebooks/` | Main training, JI-ADF comparison and binary-study notebooks |
| `demo/` | Small ensemble artifacts and fixed evaluation data used by Streamlit |
| `report/LesionLab_Report.pdf` | 33-page research report |
| `report/LaTeX_Source.zip` | LaTeX source archive |
| `tests/` | Focused model, inference, screening and protocol tests |

The supplementary binary study is in the report appendix. `src/improved/` contains
shared data/training helpers required by the JI-ADF implementation. It is retained
as a code dependency. Unused DermFM experiments, duplicated references, old report
builders, audit dumps, logs, environments and raw datasets are excluded.

## Main results

The following comparison uses the **same 786 validation lesions**:

| Metric | Initial score fusion | Enhanced feature/TTA ensemble |
| --- | ---: | ---: |
| Macro ROC-AUC | 0.8627 | **0.8767** |
| Macro average precision | 0.4560 | **0.5576** |
| Mean one-versus-rest class accuracy | 0.9267 | **0.9440** |
| Macro sensitivity | 0.4282 | **0.5430** |
| Macro specificity | 0.9569 | **0.9638** |
| Mean class Dice / macro F1 | 0.3551 | **0.5530** |
| Macro PPV | 0.3233 | **0.6974** |
| Macro NPV | 0.9533 | **0.9632** |

Overall multiclass accuracy increases from **59.67%** to **69.21%**, with
75 fewer total errors (317 versus 242). Mean one-versus-rest class accuracy
includes true negatives and differs from overall multiclass accuracy.

The ensemble combines a balanced logistic classifier on a 173-dimensional frozen
feature/metadata/specialist-score representation with six-view clinical/dermoscopic
averaging using equal geometric pooling. The trained encoders remain unchanged.
The feature classifier fits 3,930 training-plus-fusion lesions, while initial score
fusion fits 524 fusion lesions; this comparison measures complete pipelines.

Published JI-ADF values use a different, 479-lesion hidden-test cohort. Numerical
cross-study differences do not establish matched-test superiority. Patient-level
independence cannot be established, and rare-class validation supports are small.
Broader screening and direct-binary studies are separate experiments.

The saved original baseline separately obtained 63.55% accuracy and 0.4026 macro F1
on 524 historical holdout lesions. That cohort is different. The original full baseline archive remains unchanged in the private research
workspace, outside this source export. Detailed comparisons and limitations are in
the report.

## Run locally

Use Python 3.12:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Large checkpoints are not included. Import your trusted Kaggle output through
**Model setup**, or run:

```sh
python src/scripts/import_training_artifacts.py /path/to/training_artifacts.zip
```

Required inference files are `best_clinical.pt`, `best_dermoscopic.pt`,
`best_metadata.pt`, `metadata_encoder.json`, `best_fusion_lr.joblib` and
`input_validator.pt`, under `src/checkpoints/`. The small downstream ensemble
artifacts are already included. Their hashes require the matching original
encoders. Missing validator weights block prediction. The Performance page works from the included evaluation data; analysis uses
your uploaded photographs. No preselected example cases are bundled.

Training notebooks require the official data and their prepared private reference
inputs where specified. This source export is not a model or
dataset backup. Kaggle notebooks retain the preinstalled CUDA PyTorch builds;
install the local app requirements only for local use.

## References and data

The JI-ADF comparison adapts [Nguyen et al., arXiv:2604.27343](https://arxiv.org/abs/2604.27343).
Source provenance for reused implementation components is retained in their source headers.

MILK10k report illustrations are credited to the MILK study team under the source
CC BY-NC 4.0 terms in `report/DATA_LICENSE.txt`. Data sources:
[MILK10k](https://api.isic-archive.com/doi/milk10k/),
[ISIC 2019](https://challenge.isic-archive.com/data/),
[PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1).
Model outputs are for research
demonstration, not clinical diagnosis.

## Streamlit Community Cloud

Deploy repository `aachintya/LesionLab`, branch `main`, entrypoint **`app.py`** and
Python **3.12**. This entrypoint uses CPU inference, fixes the improved ensemble,
and hides model setup. The ordinary local entrypoint remains `streamlit_app.py`.

The five trained binary files under `src/checkpoints/` must also be available in
the deployed repository: `best_clinical.pt`, `best_dermoscopic.pt`,
`best_metadata.pt`, `best_fusion_lr.joblib` and `input_validator.pt`.
They are ignored locally, so publishing them requires an explicit `git add -f`
for those five filenames. The metadata encoder and downstream ensemble artifacts
are already tracked. No dataset, HF token or external inference API is required.

The root dependencies are for app inference. Use the Kaggle notebook's runtime
setup when training, rather than installing CPU deployment requirements there.


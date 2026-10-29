# LesionLab

Multimodal skin-lesion research using clinical photographs, dermoscopy and patient
metadata. The main method is an EfficientNetV2-S frozen-feature/six-view ensemble.

## Project contents

| Path | Purpose |
| --- | --- |
| `src/` | Models, preprocessing, training, inference and fusion code |
| `notebooks/` | Main training, JI-ADF comparison and binary-study notebooks |
| `demo/` | Small ensemble artifacts and fixed evaluation data used by Streamlit |
| `report/LesionLab_Report.pdf` | Final 28-page research report |
| `report/LaTeX_Source.zip` | Complete portable LaTeX project |
| `tests/` | Focused model, inference, screening and protocol tests |

The supplementary binary study is in the report appendix. `src/improved/` contains
shared data/training helpers required by the JI-ADF implementation. It is retained
as a code dependency. Unused DermFM experiments, duplicated references, old report
builders, audit dumps, logs, environments and raw datasets are excluded.

## Main results

The following comparison uses the **same 786 validation lesions**:

| Method | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Original score fusion | 59.67% | 42.82% | 0.3551 |
| Feature/TTA ensemble | **69.21%** | **45.63%** | **0.4523** |

The ensemble combines a balanced logistic classifier on frozen image features,
metadata and specialist scores with six-view clinical/dermoscopic averaging using
an equal geometric combination. The trained encoders remain unchanged.
Broader lesion-family screening accuracy is 82.70% versus 80.41% for the original,
with sensitivity 88.97% and specificity 66.96%. These are exploratory validation
results, not an independent test or a state-of-the-art claim.

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

For LaTeX, upload `report/LaTeX_Source.zip` to Overleaf and select `main.tex`, or
extract it and run `latexmk -pdf main.tex`.

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

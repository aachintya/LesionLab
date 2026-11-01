# Validation results

The comparison uses the same 786 internal validation lesions across eleven
diagnoses. See `validation_results.csv` and `LesionLab_Report.pdf` for the full
metric table.

The decision rule assigns the category with the largest score. Metrics are
averaged equally across the eleven categories, with undefined ratios set to zero.
Mean one-versus-rest class accuracy includes true negatives; overall multiclass
accuracy is the proportion of lesions assigned the correct category. Class Dice
is equivalent to class F1 in this classification setting.

Overall multiclass accuracy is 59.67% for initial score fusion and 69.21% for the
feature/TTA ensemble: 469 versus 544 correct lesions, or 75 fewer total errors.
The historical 524-lesion holdout and supplementary binary studies are separate
experiments. Published JI-ADF values use a different evaluation cohort.

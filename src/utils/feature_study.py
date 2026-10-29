"""Lightweight classifiers on frozen embeddings and probability features."""
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 20261029


def candidate_specs():
    result = []
    for feature in ["pca64", "pca128", "pca64_plus_scores"]:
        for c in [.01, .1, 1.]:
            result.append(dict(id=f"{feature}_lr_balanced_C{c:g}", feature=feature, family="lr",
                               C=c, class_weight="balanced", fit_source="train_and_fusion"))
        result.append(dict(id=f"{feature}_lr_unweighted_C0.1", feature=feature, family="lr",
                           C=.1, class_weight=None, fit_source="train_and_fusion"))
    for feature in ["pca64", "pca128"]:
        for alpha in [1., 10., 100.]:
            result.append(dict(id=f"{feature}_ridge_balanced_a{alpha:g}", feature=feature, family="ridge",
                               alpha=alpha, class_weight="balanced", fit_source="train_and_fusion"))
        for c in [1., 10.]:
            result.append(dict(id=f"{feature}_rbf_balanced_C{c:g}", feature=feature, family="rbf",
                               C=c, class_weight="balanced", fit_source="train_and_fusion"))
    for leaves in [7, 15]:
        result.append(dict(id=f"scores_hist_leaves{leaves}", feature="scores", family="hist",
                           leaves=leaves, class_weight="balanced", fit_source="fusion"))
    for leaf in [2, 5]:
        result.append(dict(id=f"scores_extratrees_leaf{leaf}", feature="scores", family="extra",
                           min_leaf=leaf, class_weight="balanced", fit_source="fusion"))
    for c in [.1, 1.]:
        result.append(dict(id=f"tta_scores_lr_balanced_C{c:g}", feature="tta_scores", family="lr",
                           C=c, class_weight="balanced", fit_source="fusion"))
    for name, weights in [("all", [1/3, 1/3, 1/3]), ("images", [.5, .5, 0])]:
        result.append(dict(id=f"tta_mean_{name}", feature="tta_scores", family="pool",
                           weights=weights, fit_source="fusion"))
    return result


def fit(spec, x, y):
    if not np.isfinite(x).all():raise ValueError("Nonfinite classifier features.")
    family = spec["family"]
    if family == "pool":return None
    if family == "lr":
        model = LogisticRegression(C=spec["C"], class_weight=spec["class_weight"],
                                    max_iter=2000, tol=1e-4, solver="lbfgs", random_state=SEED)
    elif family == "ridge":
        model = RidgeClassifier(alpha=spec["alpha"], class_weight=spec["class_weight"], solver="lsqr", tol=1e-5)
    elif family == "rbf":
        model = SVC(C=spec["C"], class_weight=spec["class_weight"], kernel="rbf", gamma="scale",
                    probability=False, decision_function_shape="ovr", break_ties=True, cache_size=512)
    elif family == "hist":
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=spec["leaves"],
                    min_samples_leaf=15, l2_regularization=5., early_stopping=False, class_weight="balanced", random_state=SEED)
    elif family == "extra":
        model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=spec["min_leaf"],
                                     class_weight="balanced", max_features=.7, n_jobs=2, random_state=SEED)
    else:raise ValueError(f"Unknown classifier family: {family}")
    if family in {"lr", "ridge", "rbf"}:model=make_pipeline(StandardScaler(),model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        model.fit(x,y)
    if any(issubclass(item.category,ConvergenceWarning) for item in caught):
        raise RuntimeError("Classifier did not converge.")
    return model


def predict(spec, model, x):
    if not np.isfinite(x).all():raise ValueError("Nonfinite prediction features.")
    if spec["family"] == "pool":
        from utils.fusion_study import validate_probabilities
        p=validate_probabilities(x.reshape(-1,3,11))
        weights=np.asarray(spec["weights"])
        if weights.shape!=(3,) or (weights<0).any() or not np.isclose(weights.sum(),1):
            raise ValueError("Invalid pooling weights.")
        result=np.einsum("nmk,m->nk",p,weights)
        return result/result.sum(1,keepdims=True)
    if hasattr(model,"predict_proba"):
        observed=model.predict_proba(x)
    else:
        logits=model.decision_function(x)
        logits=logits-logits.max(1,keepdims=True)
        observed=np.exp(logits);observed/=observed.sum(1,keepdims=True)
    result=np.zeros((len(x),11))
    result[:,np.asarray(model.classes_,dtype=int)]=observed
    if not np.isfinite(result).all() or not np.allclose(result.sum(1),1):
        raise ValueError("Classifier produced invalid scores.")
    return result


def fitting_rows(spec, auxiliary_train, fusion, fold_train):
    if spec["fit_source"] == "fusion":return fusion[fold_train]
    if spec["fit_source"] == "train_and_fusion":return np.r_[auxiliary_train,fusion[fold_train]]
    raise ValueError("Unknown downstream fitting partition.")


def oof(spec, representations, labels, auxiliary_train, fusion, folds):
    predictions=np.zeros((len(fusion),11));coverage=np.zeros(len(fusion),dtype=int)
    x=representations[spec["feature"]]
    for fold_train,fold_test in folds:
        rows=fitting_rows(spec,auxiliary_train,fusion,fold_train)
        evaluation=fusion[fold_test]
        if len(np.intersect1d(rows,evaluation)):raise ValueError("An evaluation lesion entered classifier fitting.")
        model=fit(spec,x[rows],labels[rows])
        predictions[fold_test]=predict(spec,model,x[evaluation]);coverage[fold_test]+=1
    if not (coverage==1).all():raise ValueError("Incomplete or duplicated OOF coverage.")
    return predictions

"""Per-prediction feature attributions ("why did the model say that?").

Two strategies, picked automatically from the estimator type:

* **Linear models** (Logistic Regression, calibrated Linear SVM):
  ``contribution_i = tfidf_i * coef_i`` - exact for the decision function.
* **Tree ensembles** (Random Forest): Saabas decision-path decomposition.
  Walking each tree from root to leaf, every split moves the predicted
  probability; that delta is credited to the split feature. Averaged over all
  trees, ``bias + sum(contributions) == predicted probability`` exactly.

Contributions are signed toward the *real* class: negative = pushes toward
fake, positive = pushes toward real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from ml import REAL


@dataclass(frozen=True)
class FeatureContribution:
    term: str
    weight: float  # signed: <0 toward fake, >0 toward real
    share: float  # % of total absolute evidence for this prediction (0-100)
    tfidf: float  # the term's TF-IDF value in this document
    present: bool  # False when the *absence* of a term drove a tree split


@dataclass(frozen=True)
class Explanation:
    method: str
    base_value: float | None  # model's prior (RF: mean root probability of "real")
    toward_fake: list[FeatureContribution]
    toward_real: list[FeatureContribution]


def _linear_coef(model) -> np.ndarray | None:
    """Return a 1-D coefficient vector for the REAL class, or None if not linear."""
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        # coef_ is for classes_[1]; flip if that is not REAL.
        return coef if list(model.classes_).index(REAL) == 1 else -coef
    if isinstance(model, CalibratedClassifierCV):
        coefs = [
            np.asarray(cc.estimator.coef_).ravel()
            for cc in model.calibrated_classifiers_
            if hasattr(cc.estimator, "coef_")
        ]
        if coefs:
            coef = np.mean(coefs, axis=0)
            return coef if list(model.classes_).index(REAL) == 1 else -coef
    return None


def _forest_contributions(forest: RandomForestClassifier, x: sparse.csr_matrix):
    """Saabas decomposition for a single-row sparse matrix."""
    real_idx = list(forest.classes_).index(REAL)
    n_features = x.shape[1]
    contrib = np.zeros(n_features)
    bias = 0.0
    indicator, ptr = forest.decision_path(x)  # one call for all trees
    nodes_all = indicator.indices  # node ids for the single row, tree by tree
    for t, tree in enumerate(forest.estimators_):
        tr = tree.tree_
        # Leaf/node values may be counts (old sklearn) or fractions - normalise.
        values = tr.value[:, 0, :]
        probs = values[:, real_idx] / np.maximum(values.sum(axis=1), 1e-12)
        path = nodes_all[(nodes_all >= ptr[t]) & (nodes_all < ptr[t + 1])] - ptr[t]
        path = np.sort(path)  # parents always have smaller ids than children
        bias += probs[path[0]]
        parents, children = path[:-1], path[1:]
        np.add.at(contrib, tr.feature[parents], probs[children] - probs[parents])
    n = len(forest.estimators_)
    return contrib / n, bias / n


def explain(model, x: sparse.spmatrix, feature_names: np.ndarray, top_k: int = 10) -> Explanation:
    """Explain a single vectorised document ``x`` (shape ``1 x n_features``)."""
    x = sparse.csr_matrix(x)
    coef = _linear_coef(model)
    base_value: float | None = None

    if coef is not None:
        method = "linear (tfidf x coefficient)"
        contrib = np.zeros(x.shape[1])
        contrib[x.indices] = x.data * coef[x.indices]
    elif isinstance(model, RandomForestClassifier):
        method = "tree path decomposition (Saabas)"
        contrib, base_value = _forest_contributions(model, x)
    else:  # pragma: no cover - defensive: unknown estimator type
        method = "global feature importance"
        importances = getattr(model, "feature_importances_", np.zeros(x.shape[1]))
        contrib = np.zeros(x.shape[1])
        contrib[x.indices] = x.data * importances[x.indices]

    total = float(np.abs(contrib).sum()) or 1.0
    tfidf = dict(zip(x.indices.tolist(), x.data.tolist(), strict=True))

    def build(indices: np.ndarray) -> list[FeatureContribution]:
        return [
            FeatureContribution(
                term=str(feature_names[i]),
                weight=round(float(contrib[i]), 6),
                share=round(100 * abs(float(contrib[i])) / total, 2),
                tfidf=round(tfidf.get(int(i), 0.0), 4),
                present=int(i) in tfidf,
            )
            for i in indices
        ]

    order = np.argsort(contrib)  # most negative (fake) first
    fake_idx = [i for i in order[:top_k] if contrib[i] < 0]
    real_idx = [i for i in order[::-1][:top_k] if contrib[i] > 0]
    return Explanation(
        method=method,
        base_value=None if base_value is None else round(float(base_value), 4),
        toward_fake=build(np.asarray(fake_idx, dtype=int)),
        toward_real=build(np.asarray(real_idx, dtype=int)),
    )

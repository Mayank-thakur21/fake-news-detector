import joblib
import numpy as np
import pytest

from ml.explain import _forest_contributions, explain
from ml.preprocessing import prepare
from tests.conftest import _FAKE_WORDS, FAKE_TEXT, REAL_TEXT


@pytest.fixture(scope="module")
def artifacts(model_dir):
    vec = joblib.load(model_dir / "vectorizer.joblib")
    models = {k: joblib.load(model_dir / f"{k}.joblib") for k in ("rf", "lr", "svm")}
    return vec, vec.get_feature_names_out(), models


def test_forest_decomposition_is_exact(artifacts):
    vec, _, models = artifacts
    rf = models["rf"]
    for text in (FAKE_TEXT, REAL_TEXT):
        x = vec.transform([prepare(text)])
        contrib, bias = _forest_contributions(rf, x)
        p_real = rf.predict_proba(x)[0][list(rf.classes_).index(1)]
        assert bias + contrib.sum() == pytest.approx(p_real, abs=1e-9)


@pytest.mark.parametrize("key", ["rf", "lr", "svm"])
def test_explanation_directions(artifacts, key):
    vec, names, models = artifacts
    x = vec.transform([prepare(FAKE_TEXT)])
    exp = explain(models[key], x, names, top_k=5)
    assert len(exp.toward_fake) <= 5 and len(exp.toward_real) <= 5
    assert all(c.weight < 0 for c in exp.toward_fake)
    assert all(c.weight > 0 for c in exp.toward_real)
    # Words that are present in the text and push toward fake must be fake-class vocabulary.
    present_fake_terms = {c.term for c in exp.toward_fake if c.present}
    assert present_fake_terms, "expected at least one present term pushing toward fake"
    assert all(set(t.split()) & set(_FAKE_WORDS) for t in present_fake_terms)


def test_linear_explanation_only_uses_present_terms(artifacts):
    vec, names, models = artifacts
    x = vec.transform([prepare(REAL_TEXT)])
    exp = explain(models["lr"], x, names, top_k=10)
    assert exp.base_value is None
    assert all(c.present and c.tfidf > 0 for c in exp.toward_fake + exp.toward_real)


def test_shares_are_percentages(artifacts):
    vec, names, models = artifacts
    x = vec.transform([prepare(FAKE_TEXT)])
    exp = explain(models["svm"], x, names, top_k=30)
    total = sum(c.share for c in exp.toward_fake + exp.toward_real)
    assert 0 < total <= 100.01
    assert np.all([0 <= c.share <= 100 for c in exp.toward_fake])

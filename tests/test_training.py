import json

import pandas as pd
import pytest

from ml.train import TrainConfig, load_isot, load_labelled_csv, main, preprocess, train
from tests.conftest import synthetic_corpus


def test_training_writes_all_artifacts(model_dir):
    for name in ("vectorizer.joblib", "rf.joblib", "lr.joblib", "svm.joblib", "metrics.json"):
        assert (model_dir / name).exists(), name


def test_metrics_report_structure(model_dir):
    report = json.loads((model_dir / "metrics.json").read_text())
    assert set(report["models"]) == {"rf", "lr", "svm"}
    assert report["best_model"] in report["models"]
    assert report["dataset"]["rows"] == report["dataset"]["fake"] + report["dataset"]["real"]
    for m in report["models"].values():
        for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix", "train_seconds"):
            assert key in m
        assert 0.0 <= m["accuracy"] <= 1.0
        # The synthetic classes are trivially separable.
        assert m["accuracy"] >= 0.9


def test_train_subset_of_models(tmp_path):
    report = train(synthetic_corpus(20), tmp_path, TrainConfig(models=("lr",)))
    assert list(report["models"]) == ["lr"]
    assert not (tmp_path / "rf.joblib").exists()


def test_preprocess_drops_duplicates_and_empties():
    df = pd.DataFrame(
        {
            "text": ["same text here", "same text here", "!!!", "other words", "more words"],
            "label": [0, 0, 1, 1, 0],
        }
    )
    out = preprocess(df, strip_leakage=True)
    assert len(out) == 3


def test_preprocess_requires_both_classes():
    df = pd.DataFrame({"text": ["a b c", "d e f"], "label": [1, 1]})
    with pytest.raises(ValueError, match="both fake and real"):
        preprocess(df, strip_leakage=True)


def test_load_labelled_csv_accepts_string_labels(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame(
        {"title": ["T1", "T2"], "text": ["body one", "body two"], "label": ["FAKE", "real"]}
    ).to_csv(path, index=False)
    df = load_labelled_csv(path)
    assert df["label"].tolist() == [0, 1]
    assert df["text"].iloc[0] == "T1. body one"


def test_load_labelled_csv_rejects_bad_labels(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"text": ["x"], "label": ["maybe"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unrecognised label"):
        load_labelled_csv(path)


def test_load_labelled_csv_requires_columns(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"body": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required"):
        load_labelled_csv(path)


def test_load_isot(tmp_path):
    cols = {"title": ["t"], "text": ["x"], "subject": ["s"], "date": ["d"]}
    pd.DataFrame(cols).to_csv(tmp_path / "Fake.csv", index=False)
    pd.DataFrame(cols).to_csv(tmp_path / "True.csv", index=False)
    df = load_isot(tmp_path / "Fake.csv", tmp_path / "True.csv")
    assert df["label"].tolist() == [0, 1]


def test_cli_trains_from_csv(tmp_path):
    csv = tmp_path / "data.csv"
    synthetic_corpus(20).to_csv(csv, index=False)
    out = tmp_path / "models"
    assert main(["--csv", str(csv), "--model-dir", str(out), "--models", "lr,svm"]) == 0
    assert (out / "svm.joblib").exists()


def test_cli_rejects_unknown_model(tmp_path):
    assert main(["--models", "xgb", "--model-dir", str(tmp_path)]) == 2


def test_cli_reports_missing_file(tmp_path):
    assert main(["--csv", str(tmp_path / "nope.csv"), "--model-dir", str(tmp_path)]) == 1

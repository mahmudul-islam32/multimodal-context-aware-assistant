import numpy as np
import pytest

from mcaa.har.features import extract_features, feature_names
from mcaa.har.models import FeatureModel, evaluate_loso, single_window_latency_ms
from mcaa.sensors.synthetic import ACTIVITIES, make_dataset


def test_feature_shape():
    F = extract_features(np.random.default_rng(0).normal(size=(3, 128, 6)))
    assert F.shape == (3, len(feature_names()))


def test_loso_no_subject_leak_and_reasonable_f1():
    X, y, g = make_dataset(n_subjects=4, seconds_per_activity=12, seed=3)
    res = evaluate_loso(lambda: FeatureModel(n_estimators=40), X, y, g, n_splits=2)
    assert res.macro_f1 > 0.6
    assert res.confusion.shape == (len(ACTIVITIES), len(ACTIVITIES))


def test_latency_under_50ms(small_har):
    m, (X, _, _) = small_har
    assert single_window_latency_ms(m, X[0], repeats=10) < 50


def test_calibration_adds_new_class():
    base = tuple(a for a in ACTIVITIES if a != "WORKING")
    X, y, _ = make_dataset(n_subjects=3, seconds_per_activity=10, activities=base)
    m = FeatureModel(n_estimators=40).fit(X, y)
    assert "WORKING" not in m.classes_
    Xc, yc, _ = make_dataset(n_subjects=1, seconds_per_activity=10, seed=9)
    m.calibrate(Xc, yc)
    assert "WORKING" in m.classes_


def test_cnn_trains_if_torch_available():
    pytest.importorskip("torch")
    from mcaa.har.cnn import CNNModel

    X, y, _ = make_dataset(n_subjects=2, seconds_per_activity=8)
    m = CNNModel(epochs=2).fit(X, y)
    p = m.predict_proba(X[:5])
    assert p.shape == (5, len(set(y))) and np.allclose(p.sum(1), 1, atol=1e-5)

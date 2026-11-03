"""Activity classifiers, leave-subjects-out evaluation and calibration."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract_features


class ActivityClassifier(Protocol):
    classes_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> ActivityClassifier: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class FeatureModel:
    """scikit-learn baseline: hand-crafted features + random forest."""

    name = "features+random_forest"

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed
        self.pipe = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight="balanced"),
        )
        self._calib_X: np.ndarray | None = None
        self._calib_y: np.ndarray | None = None

    @property
    def classes_(self) -> np.ndarray:
        return self.pipe.classes_

    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureModel:
        self._train_F = extract_features(X)
        self._train_y = np.asarray(y)
        self.pipe[-1].set_params(n_jobs=-1)
        self.pipe.fit(self._train_F, self._train_y)
        self._single_thread()
        return self

    def _single_thread(self) -> None:
        # parallel prediction only adds overhead for single live windows
        self.pipe[-1].set_params(n_jobs=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(extract_features(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]

    def calibrate(self, X: np.ndarray, y: np.ndarray, weight: float = 5.0) -> FeatureModel:
        """Fine-tune with a short recording of the target user/task.

        Calibration windows are up-weighted and may introduce new classes
        (e.g. WORKING), which the public datasets do not contain.
        """
        F = extract_features(X)
        Xa = np.vstack([self._train_F, F])
        ya = np.concatenate([self._train_y, y])
        w = np.concatenate([np.ones(len(self._train_y)), np.full(len(y), weight)])
        self.pipe[-1].set_params(n_jobs=-1)
        self.pipe.fit(Xa, ya, randomforestclassifier__sample_weight=w)
        self._single_thread()
        return self

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> FeatureModel:
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301 - only load locally trained models


@dataclass
class LosoResult:
    model: str
    macro_f1: float
    fold_f1: list[float]
    labels: list[str]
    confusion: np.ndarray
    per_class_f1: dict[str, float]
    latency_ms_per_window: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "macro_f1": round(self.macro_f1, 4),
            "macro_f1_fold_mean": round(float(np.mean(self.fold_f1)), 4),
            "macro_f1_fold_std": round(float(np.std(self.fold_f1)), 4),
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
            "latency_ms_per_window": round(self.latency_ms_per_window, 3),
            "labels": self.labels,
            "confusion": self.confusion.tolist(),
            **self.extra,
        }


def evaluate_loso(make_model, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 4) -> LosoResult:
    """Leave-subjects-out (GroupKFold) evaluation: no subject is in train and test."""
    labels = sorted(np.unique(y).tolist())
    y_true_all, y_pred_all, folds, lat = [], [], [], []
    name = "model"
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])
        m = make_model()
        name = getattr(m, "name", type(m).__name__)
        m.fit(X[tr], y[tr])
        t0 = time.perf_counter()
        proba = m.predict_proba(X[te])
        lat.append((time.perf_counter() - t0) * 1000 / len(te))
        pred = np.asarray(m.classes_)[proba.argmax(1)]
        folds.append(f1_score(y[te], pred, average="macro", labels=labels))
        y_true_all.append(y[te])
        y_pred_all.append(pred)
    yt, yp = np.concatenate(y_true_all), np.concatenate(y_pred_all)
    per = f1_score(yt, yp, average=None, labels=labels)
    return LosoResult(
        model=name,
        macro_f1=float(f1_score(yt, yp, average="macro", labels=labels)),
        fold_f1=[float(f) for f in folds],
        labels=labels,
        confusion=confusion_matrix(yt, yp, labels=labels),
        per_class_f1=dict(zip(labels, map(float, per), strict=True)),
        latency_ms_per_window=float(np.mean(lat)),
    )


def single_window_latency_ms(model, window: np.ndarray, repeats: int = 50) -> float:
    """Latency for one live window (the real-time case, F-02 target < 50 ms)."""
    model.predict_proba(window[None])
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(window[None])
    return (time.perf_counter() - t0) * 1000 / repeats

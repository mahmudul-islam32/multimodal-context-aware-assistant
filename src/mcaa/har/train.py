"""Train and evaluate HAR models.

    python -m mcaa.har.train --config configs/har.yaml
    python -m mcaa.har.train --dataset synthetic --model rf --out models/har_rf.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..sensors.synthetic import ACTIVITIES, make_dataset
from .models import FeatureModel, evaluate_loso, single_window_latency_ms


def load_dataset(name: str, root: str = "data/raw", seed: int = 42):
    if name == "synthetic":
        return make_dataset(n_subjects=12, seconds_per_activity=45, seed=seed)
    if name == "uci-har":
        from .datasets import load_uci_har

        return load_uci_har(Path(root) / "uci-har")
    if name == "wisdm":
        from .datasets import load_wisdm

        return load_wisdm(Path(root) / "wisdm")
    raise ValueError(name)


def model_factory(kind: str, seed: int = 0):
    if kind == "rf":
        return lambda: FeatureModel(seed=seed)
    if kind == "cnn":
        from .cnn import CNNModel

        return lambda: CNNModel(seed=seed)
    raise ValueError(kind)


def train_default_model(seed: int = 0) -> FeatureModel:
    """Model used by the live API: trained on synthetic data incl. WORKING."""
    X, y, _ = make_dataset(n_subjects=8, seconds_per_activity=30, activities=ACTIVITIES, seed=seed)
    return FeatureModel(n_estimators=120, seed=seed).fit(X, y)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--model", default="rf", choices=["rf", "cnn"])
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models/har_rf.pkl")
    p.add_argument("--report", default="results/har")
    a = p.parse_args(argv)
    if a.config:
        cfg = yaml.safe_load(Path(a.config).read_text())
        for k, v in cfg.items():
            setattr(a, k, v)
    np.random.seed(a.seed)
    X, y, g = load_dataset(a.dataset, seed=a.seed)
    res = evaluate_loso(model_factory(a.model, a.seed), X, y, g, n_splits=a.folds)
    m = model_factory(a.model, a.seed)().fit(X, y)
    res.extra["single_window_latency_ms"] = round(single_window_latency_ms(m, X[0]), 3)
    res.extra["dataset"] = a.dataset
    res.extra["seed"] = a.seed
    out = Path(a.report)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.dataset}_{a.model}.json").write_text(json.dumps(res.to_dict(), indent=2))
    if a.model == "rf":
        m.save(a.out)
    print(json.dumps({k: v for k, v in res.to_dict().items() if k != "confusion"}, indent=2))


if __name__ == "__main__":
    main()

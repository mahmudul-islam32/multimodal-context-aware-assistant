"""PyTorch 1D-CNN activity model (optional dependency: ``pip install .[torch]``)."""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch
    HAVE_TORCH = False


def _require():
    if not HAVE_TORCH:
        raise ImportError("PyTorch is not installed: pip install '.[torch]'")


if HAVE_TORCH:

    class _Net(nn.Module):
        def __init__(self, n_ch: int, n_cls: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_cls))

        def forward(self, x):
            return self.head(self.features(x))


class CNNModel:
    """1D-CNN on raw (filtered) windows; per-channel standardisation."""

    name = "1d_cnn"

    def __init__(self, epochs: int = 15, lr: float = 2e-3, batch: int = 64, seed: int = 0):
        _require()
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.classes_: np.ndarray = np.array([])
        self.net = None

    def _prep(self, X: np.ndarray):
        Xn = (X - self.mu) / self.sd
        return torch.tensor(Xn.transpose(0, 2, 1), dtype=torch.float32)

    def _train(self, X, yi, epochs, params=None, lr=None):
        torch.manual_seed(self.seed)
        opt = torch.optim.Adam(params or self.net.parameters(), lr=lr or self.lr)
        Xt, yt = self._prep(X), torch.tensor(yi)
        lossf = nn.CrossEntropyLoss()
        self.net.train()
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.batch):
                idx = perm[i : i + self.batch]
                opt.zero_grad()
                loss = lossf(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        self.net.eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> CNNModel:
        torch.set_num_threads(4)
        self.classes_ = np.array(sorted(set(y)))
        self.mu = X.mean((0, 1), keepdims=True)
        self.sd = X.std((0, 1), keepdims=True) + 1e-6
        self.net = _Net(X.shape[2], len(self.classes_))
        yi = np.searchsorted(self.classes_, y)
        self._train(X, yi, self.epochs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.net(self._prep(X)), 1).numpy()

    def calibrate(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> CNNModel:
        """Fine-tune on a short target-task recording; new classes extend the head."""
        new = [c for c in sorted(set(y)) if c not in set(self.classes_)]
        if new:
            old_head = self.net.head[-1]
            self.classes_ = np.array(sorted(list(self.classes_) + new))
            lin = nn.Linear(old_head.in_features, len(self.classes_))
            self.net.head[-1] = lin
        yi = np.searchsorted(self.classes_, y)
        # freeze early conv layers, train last block + head
        params = list(self.net.features[8:].parameters()) + list(self.net.head.parameters())
        self._train(X, yi, epochs, params=params, lr=self.lr)
        return self

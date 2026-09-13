"""A minimal ANN for the chart-cloning task, in pure NumPy.

This is the "simple ANN" stage of the ANN -> SNN plan, deliberately small and
dependency-free (no PyTorch installed here yet). Its job right now is to answer
one question: **is the dataset in dataset.py actually learnable?** If a tiny MLP
cannot recover the chart labels from the observations, the observation encoding is
wrong and no amount of model complexity will save it.

Metric choice matters. Strum is positive on only ~1.3% of steps, so a model that
never strums is 98.7% accurate and completely useless. We therefore report
precision / recall / F1 on the positive class, and always alongside the
"never play anything" baseline so the comparison is honest.

Generalisation is tested *across songs* (train on two, test on the third) rather
than by random split: a random split lets near-duplicate timesteps leak between
train and test and reports a flattering number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0.0


def score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Metrics:
    pred = (y_prob >= threshold).astype(np.float32)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return Metrics(tp, fp, fn, tn)


class MLP:
    """One hidden layer, tanh, sigmoid output, binary cross-entropy.

    Scaling the loss up and the logits down keeps the initial gradients sane for
    the positive class, which is otherwise swamped by the ~99% negatives.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, pos_weight: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w1 = (rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.w2 = (rng.standard_normal((n_hidden, n_out)) * np.sqrt(1.0 / n_hidden)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)
        self.pos_weight = float(pos_weight)
        self._moments = {k: np.zeros_like(getattr(self, k)) for k in ("w1", "b1", "w2", "b2")}

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.w1 + self.b1)
        return h, h @ self.w2 + self.b2

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _, logits = self.forward(x)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 400,
        batch_size: int = 128,
        lr: float = 0.05,
        momentum: float = 0.9,
        seed: int = 0,
        verbose_every: int = 0,
    ) -> list[float]:
        rng = np.random.default_rng(seed)
        n = x.shape[0]
        history: list[float] = []

        for epoch in range(epochs):
            order = rng.permutation(n)
            total = 0.0
            batches = 0

            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb, yb = x[idx], y[idx]

                h, logits = self.forward(xb)
                p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

                # Weighted BCE; weight only the positive entries.
                w = np.where(yb > 0.5, self.pos_weight, 1.0)
                eps = 1e-7
                loss = -(w * (yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps)))
                total += float(loss.mean())
                batches += 1

                dlogits = (w * (p - yb)) / xb.shape[0]
                dw2 = h.T @ dlogits
                db2 = dlogits.sum(axis=0)
                dh = (dlogits @ self.w2.T) * (1.0 - h * h)
                dw1 = xb.T @ dh
                db1 = dh.sum(axis=0)

                for name, grad in (("w1", dw1), ("b1", db1), ("w2", dw2), ("b2", db2)):
                    m = self._moments[name]
                    m *= momentum
                    m -= lr * grad
                    getattr(self, name)[:] += m

            history.append(total / max(batches, 1))
            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"    epoch {epoch + 1:>4}  loss {history[-1]:.5f}")

        return history


TARGETS = ("green", "red", "yellow", "blue", "orange", "strum")


def train_and_evaluate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    hidden: int = 128,
    epochs: int = 400,
    pos_weight: float = 10.0,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[dict, MLP]:
    """Train on one set of songs, test on a held-out song. Returns (metrics, model)."""
    model = MLP(train_x.shape[1], hidden, train_y.shape[1], pos_weight=pos_weight, seed=seed)
    model.fit(train_x, train_y, epochs=epochs, seed=seed, verbose_every=50 if verbose else 0)

    prob = model.predict_proba(test_x)
    out = {}
    for i, name in enumerate(TARGETS):
        m = score(test_y[:, i], prob[:, i])
        base = score(test_y[:, i], np.zeros_like(prob[:, i]))  # "never play" baseline
        out[name] = {
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "accuracy": m.accuracy,
            "tp": m.tp,
            "fp": m.fp,
            "fn": m.fn,
            "baseline_accuracy": base.accuracy,
            "positives": int(test_y[:, i].sum()),
        }
    return out, model

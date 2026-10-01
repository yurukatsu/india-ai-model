"""Ridge regression on ranked features (baseline).

params:
  alpha: float            fixed ridge penalty, or
  alphas: [float, ...]    grid; the alpha with the best mean val IC is chosen
  fit_intercept: bool
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from india_ai_model.common.evaluate import mean_ic_fast


class Model:
    def __init__(self, params: dict, feature_names: list[str]):
        self.params = params
        self.feature_names = feature_names
        self.model: Ridge | None = None
        self.best_alpha: float | None = None
        self.val_ic: float | None = None

    def _fit_one(self, alpha: float, X, y, w=None) -> Ridge:
        m = Ridge(alpha=alpha, fit_intercept=self.params.get("fit_intercept", True))
        m.fit(X, y, sample_weight=w)
        return m

    def fit(self, X_tr, y_tr, d_tr, X_va, y_va, d_va, w_tr=None) -> None:
        if w_tr is not None:  # keep the effective alpha comparable to the unweighted fit
            w_tr = w_tr * (len(w_tr) / w_tr.sum())
        alphas = self.params.get("alphas")
        if alphas and X_va is not None and len(X_va):
            best = (-np.inf, None, None)
            for a in alphas:
                m = self._fit_one(float(a), X_tr, y_tr, w_tr)
                ic = mean_ic_fast(m.predict(X_va), y_va, d_va)
                if ic > best[0]:
                    best = (ic, float(a), m)
            self.val_ic, self.best_alpha, self.model = best
        else:
            a = float(self.params.get("alpha", alphas[0] if alphas else 1.0))
            self.best_alpha = a
            self.model = self._fit_one(a, X_tr, y_tr, w_tr)
            if X_va is not None and len(X_va):
                self.val_ic = mean_ic_fast(self.model.predict(X_va), y_va, d_va)

    def predict(self, X, dates=None) -> np.ndarray:
        return self.model.predict(X)

    def info(self) -> dict:
        return {
            "alpha": self.best_alpha,
            "val_ic": None if self.val_ic is None else round(self.val_ic, 4),
        }

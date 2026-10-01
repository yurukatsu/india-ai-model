"""LightGBM baseline: lambdarank (query = date, label = within-date quantile) or regression.

params:
  objective: lambdarank | regression
  n_bins: 10                     label bins for lambdarank (0 = worst .. n_bins-1 = best)
  num_boost_round: 2000
  early_stopping_rounds: 100     on mean per-date Spearman IC of the validation set
  lgb: {learning_rate: 0.02, num_leaves: 15, ...}   passed to LightGBM
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from india_ai_model.common.evaluate import mean_ic_fast


def _order_by_date(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(d, kind="stable")
    _, sizes = np.unique(d[order], return_counts=True)
    return order, sizes


def _quantile_labels(y: np.ndarray, d: np.ndarray, n_bins: int) -> np.ndarray:
    s = pd.Series(y)
    r = s.groupby(d).rank(method="first")
    n = s.groupby(d).transform("count")
    return np.floor((r - 1) / n * n_bins).astype(int).clip(0, n_bins - 1).to_numpy()


class Model:
    def __init__(self, params: dict, feature_names: list[str]):
        self.params = params
        self.feature_names = feature_names
        self.booster: lgb.Booster | None = None
        self.best_iter: int | None = None
        self.val_ic: float | None = None

    def fit(self, X_tr, y_tr, d_tr, X_va, y_va, d_va, w_tr=None) -> None:
        objective = self.params.get("objective", "lambdarank")
        lgb_params = {
            "objective": objective,
            "metric": "None",
            "verbosity": -1,
            "seed": self.params.get("seed", 0),
            "num_threads": self.params.get("num_threads", 0),
            **self.params.get("lgb", {}),
        }
        o_tr, g_tr = _order_by_date(d_tr)
        w_s = None if w_tr is None else (w_tr * (len(w_tr) / w_tr.sum()))[o_tr]
        if objective == "lambdarank":
            n_bins = int(self.params.get("n_bins", 10))
            lab_tr = _quantile_labels(y_tr, d_tr, n_bins)[o_tr]
            lgb_params.setdefault("label_gain", list(range(n_bins)))
            lgb_params.setdefault("lambdarank_truncation_level", 30)
            dtr = lgb.Dataset(
                X_tr[o_tr], lab_tr, group=g_tr, weight=w_s, feature_name=self.feature_names
            )
        else:
            dtr = lgb.Dataset(X_tr[o_tr], y_tr[o_tr], weight=w_s, feature_name=self.feature_names)

        valid_sets, callbacks = [], []
        has_val = X_va is not None and len(X_va) > 0
        if has_val:
            o_va, g_va = _order_by_date(d_va)
            X_va_s, y_va_s, d_va_s = X_va[o_va], y_va[o_va], d_va[o_va]
            if objective == "lambdarank":
                lab_va = _quantile_labels(y_va, d_va, int(self.params.get("n_bins", 10)))[o_va]
                dva = lgb.Dataset(X_va_s, lab_va, group=g_va, reference=dtr)
            else:
                dva = lgb.Dataset(X_va_s, y_va_s, reference=dtr)
            valid_sets = [dva]

            def feval(pred, _data):
                return "val_ic", mean_ic_fast(pred, y_va_s, d_va_s), True

            callbacks = [
                lgb.early_stopping(
                    int(self.params.get("early_stopping_rounds", 100)), verbose=False
                )
            ]
        else:
            feval = None

        self.booster = lgb.train(
            lgb_params,
            dtr,
            num_boost_round=int(self.params.get("num_boost_round", 2000)),
            valid_sets=valid_sets,
            feval=feval,
            callbacks=callbacks,
        )
        self.best_iter = self.booster.best_iteration or self.booster.current_iteration()
        if has_val:
            self.val_ic = mean_ic_fast(
                self.booster.predict(X_va_s, num_iteration=self.best_iter), y_va_s, d_va_s
            )

    def predict(self, X, dates=None) -> np.ndarray:
        return self.booster.predict(X, num_iteration=self.best_iter)

    def info(self) -> dict:
        return {
            "best_iter": self.best_iter,
            "val_ic": None if self.val_ic is None else round(self.val_ic, 4),
        }

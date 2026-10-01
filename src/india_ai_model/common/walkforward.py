"""Walk-forward splits on the rebalance grid (handoff §6.4).

At a retrain point T (rebalance index iT), with target horizon h (periods),
validation length k, embargo e (>= h so that no training window overlaps the
validation targets):

    usable samples : i <= iT - h          (end_date <= T)
    val            : iT - h - k <  i <= iT - h
    train          : i <= iT - h - k - e   (and i > iT - h - k - e - W for rolling W)
    predict        : iT <= i < iT + retrain_every

All quantities are in rebalance periods; dates are looked up in the calendar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    first_retrain: str  # e.g. "2012-12-31" (snapped to the rebal date of that month)
    retrain_every: int = 12
    window: str = "expanding"  # "expanding" | "rolling"
    window_months: int = 120  # rolling only
    train_start: str | None = None  # expanding: earliest training date (None = all)
    val_months: int = 12
    embargo_months: int | None = None  # None -> horizon
    last_predict: str | None = None  # None -> last rebal date
    refit_with_val: bool = False


@dataclass
class Split:
    T: pd.Timestamp
    train: list[pd.Timestamp] = field(default_factory=list)
    val: list[pd.Timestamp] = field(default_factory=list)
    predict: list[pd.Timestamp] = field(default_factory=list)

    def __repr__(self) -> str:
        f = lambda xs: f"{xs[0].date()}..{xs[-1].date()} ({len(xs)})" if xs else "-"  # noqa: E731
        return f"Split(T={self.T.date()}, train={f(self.train)}, val={f(self.val)}, predict={f(self.predict)})"


def _snap_index(rd: pd.DataFrame, date: str, *, side: str) -> int:
    d = pd.Timestamp(date)
    ym = d.strftime("%Y-%m")
    hit = rd.index[rd["ym"] == ym]
    if len(hit):
        return int(hit[0])
    if side == "left":
        return int(np.searchsorted(rd["date"].to_numpy(), np.datetime64(d), side="left"))
    return int(np.searchsorted(rd["date"].to_numpy(), np.datetime64(d), side="right")) - 1


def make_splits(cal: pd.DataFrame, cfg: WalkForwardConfig, horizon: int) -> list[Split]:
    rd = cal.loc[cal["is_rebal"], ["date", "ym"]].reset_index(drop=True)
    dates = rd["date"]
    n = len(rd)
    h = horizon
    k = cfg.val_months
    e = cfg.embargo_months if cfg.embargo_months is not None else h
    if e < h:
        raise ValueError(f"embargo_months={e} must be >= horizon={h}")
    i_start = _snap_index(rd, cfg.train_start, side="left") if cfg.train_start else 0
    i_last = _snap_index(rd, cfg.last_predict, side="right") if cfg.last_predict else n - 1
    iT = _snap_index(rd, cfg.first_retrain, side="left")

    splits: list[Split] = []
    while iT <= i_last:
        va_hi = iT - h
        va_lo = va_hi - k
        tr_hi = va_lo - e
        tr_lo = i_start if cfg.window == "expanding" else max(i_start, tr_hi - cfg.window_months)
        if cfg.window not in ("expanding", "rolling"):
            raise ValueError(cfg.window)
        if tr_hi - tr_lo < 12:
            raise ValueError(f"too little training data at T={dates[iT].date()}")
        pr_hi = min(iT + cfg.retrain_every, i_last + 1)
        splits.append(
            Split(
                T=dates[iT],
                train=list(dates[tr_lo : tr_hi + 1]),
                val=list(dates[va_lo + 1 : va_hi + 1]),
                predict=list(dates[iT:pr_hi]),
            )
        )
        iT += cfg.retrain_every
    return splits

"""Forward-return targets on the rebalance grid.

Naming: ``{kind}_{n}m[_s{skip}]``, e.g. ``drtn_1m``, ``srtn_3m``, ``drtn_1m_s1``.

For rebalance index i the window is the business-day interval
``(R_{i+skip}, R_{i+skip+months}]`` (close to close; the return on R_i itself is
*not* included). Cumulation per kind:

- compound (drtn, drtnp): C = cumsum(log1p(ret.fillna(0))), fwd = expm1(dC)
- simple   (srtn)       : C = cumsum(ret.fillna(0)),        fwd = dC

Output columns: ``date`` (= R_i), ``bid``, ``<name>``, ``n_obs`` (non-missing days in the
window), ``n_days`` (business days in the window), ``end_date`` (= R_{i+skip+months}).
Rows whose window extends beyond the calendar, or with ``n_obs == 0``, are dropped.
``end_date`` is what a model must use to mask leakage (``end_date <= T``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from india_ai_model.common import io

CUMULATION: dict[str, str] = {
    "drtn": "compound",
    "drtnp": "compound",
    "srtn": "simple",
    "gp_trtn": "compound",  # DS2 total return (adjusted), decimal
}

_NAME_RE = re.compile(r"^(?P<kind>[a-z][a-z_]*?)_(?P<months>\d+)m(?:_s(?P<skip>\d+))?$")


@dataclass(frozen=True)
class TargetSpec:
    kind: str
    months: int
    skip: int = 0

    @property
    def name(self) -> str:
        s = f"{self.kind}_{self.months}m"
        return s + (f"_s{self.skip}" if self.skip else "")

    @property
    def horizon(self) -> int:
        """Number of rebalance periods from R_i to end_date."""
        return self.skip + self.months


def parse_target_name(name: str) -> TargetSpec:
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"bad target name {name!r}; expected e.g. 'drtn_1m' or 'srtn_3m_s1'")
    kind = m["kind"]
    if kind not in CUMULATION:
        raise ValueError(f"unknown return kind {kind!r}; known: {sorted(CUMULATION)}")
    return TargetSpec(kind, int(m["months"]), int(m["skip"] or 0))


def _cumulate(mat: np.ndarray, method: str) -> np.ndarray:
    """Cumulative series with a leading zero row so that C[t] covers days 0..t-1."""
    x = np.nan_to_num(mat, nan=0.0)
    if method == "compound":
        x = np.log1p(x)
    elif method != "simple":
        raise ValueError(method)
    c = np.zeros((mat.shape[0] + 1, mat.shape[1]))
    np.cumsum(x, axis=0, out=c[1:])
    return c


def make_target(
    ret_daily: pd.DataFrame,
    cal: pd.DataFrame,
    kind: str,
    months: int,
    skip: int = 0,
) -> pd.DataFrame:
    """Forward return over ``(R_{i+skip}, R_{i+skip+months}]`` for every rebalance date R_i."""
    spec = TargetSpec(kind, months, skip)
    if months < 1 or skip < 0:
        raise ValueError("months >= 1 and skip >= 0 required")
    if kind not in ret_daily.columns:
        raise KeyError(f"ret_daily has no column {kind!r}")

    cal = cal.sort_values("td_index").reset_index(drop=True)
    n_td = len(cal)
    td_of_date = pd.Series(cal["td_index"].to_numpy(), index=cal["date"])
    rd = cal.loc[cal["is_rebal"]]
    r_td = rd["td_index"].to_numpy()
    r_date = rd["date"].to_numpy()

    r = ret_daily.loc[:, ["date", "bid", kind]]
    r = r[r["date"].isin(td_of_date.index)]
    bids = np.sort(r["bid"].unique())
    bid_pos = pd.Series(np.arange(len(bids)), index=bids)
    mat = np.full((n_td, len(bids)), np.nan)
    ti = r["date"].map(td_of_date).to_numpy()
    bi = r["bid"].map(bid_pos).to_numpy()
    mat[ti, bi] = r[kind].to_numpy()

    c = _cumulate(mat, CUMULATION[kind])
    obs = np.zeros((n_td + 1, len(bids)), dtype=np.int64)
    np.cumsum(~np.isnan(mat), axis=0, out=obs[1:])

    h = spec.horizon
    n_r = len(r_td)
    frames = []
    for i in range(n_r):
        a, b = i + skip, i + h
        if b >= n_r:
            break
        # window (R_a, R_b] -> days R_a+1 .. R_b inclusive -> C[R_b+1] - C[R_a+1]
        lo, hi = r_td[a] + 1, r_td[b] + 1
        dc = c[hi] - c[lo]
        fwd = np.expm1(dc) if CUMULATION[kind] == "compound" else dc
        n_obs = obs[hi] - obs[lo]
        keep = n_obs > 0
        if not keep.any():
            continue
        frames.append(
            pd.DataFrame(
                {
                    "date": r_date[i],
                    "bid": bids[keep],
                    spec.name: np.where(keep, fwd, np.nan)[keep],
                    "n_obs": n_obs[keep],
                    "n_days": int(hi - lo),
                    "end_date": r_date[b],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["date", "bid", spec.name, "n_obs", "n_days", "end_date"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["date"].astype("datetime64[ns]")
    out["end_date"] = out["end_date"].astype("datetime64[ns]")
    out["bid"] = out["bid"].astype("str")
    return io.sort_by_key(out)


def make_target_by_name(ret_daily: pd.DataFrame, cal: pd.DataFrame, name: str) -> pd.DataFrame:
    s = parse_target_name(name)
    return make_target(ret_daily, cal, s.kind, s.months, s.skip)


# ------------------------------------------------------------------ cache


def panel_y_dir() -> Path:
    return io.mart_dir() / "panel_y"


def get_target(name: str, *, use_cache: bool = True) -> pd.DataFrame:
    """Read ``mart/panel_y/<name>.parquet`` if present, otherwise compute (and cache)."""
    p = panel_y_dir() / f"{name}.parquet"
    if use_cache and p.exists():
        return pd.read_parquet(p)
    spec = parse_target_name(name)
    ret = io.read_ret_daily([spec.kind])
    cal = io.read_calendar()
    y = make_target(ret, cal, spec.kind, spec.months, spec.skip)
    if use_cache:
        io.write_parquet(y, p)
    return y


def build_panel_y(names: list[str] | None = None) -> None:
    """Pre-compute the commonly used targets."""
    names = names or ["drtn_1m", "drtn_1m_s1", "drtn_1m_s2", "drtn_3m", "srtn_1m", "srtn_3m"]
    for n in names:
        io.write_parquet(get_target(n, use_cache=False), panel_y_dir() / f"{n}.parquet")


def mask_low_coverage(y: pd.DataFrame, name: str, min_obs_frac: float) -> pd.DataFrame:
    """Set the target to NaN where n_obs / n_days < min_obs_frac."""
    out = y.copy()
    bad = out["n_obs"] / out["n_days"] < min_obs_frac
    out[name] = out[name].mask(bad)
    return out

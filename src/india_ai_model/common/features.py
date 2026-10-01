"""Feature construction on the rebalance grid.

Pipeline (see handoff §6.1):
1. lags and Rosenberg changes are computed on the *raw* panel_x over the union of
   names, shifting by rebalance index (not by row), so that new entrants have history
2. rows are then restricted to the universe of that date
3. each feature is ranked within the date: uniform -> (0,1) with NaN = 0.5, or
   Gaussian rank with NaN = 0. Binary factors are kept as 0/1 with NaN = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from india_ai_model.common.calendar import rebal_dates

KEY = ["date", "bid"]


@dataclass
class FeatureConfig:
    factors: list[str]
    lags: list[int] = field(default_factory=lambda: [3, 6, 12])
    change_lags: list[int] = field(default_factory=lambda: [3, 12])
    lag_factors: list[str] | None = None  # subset of factors to lag; None = all
    change_factors: list[str] | None = None  # subset for Rosenberg changes; None = all
    rank: str = "uniform"  # "uniform" | "gauss"
    binary: list[str] = field(default_factory=list)
    # peer-relative features (need a sector frame): x - median of peers in the same
    # (date, peer_level) group -> ``{f}_pr``; group median itself -> ``{f}_gm``
    peer_level: str | None = None  # sector | industry_group | industry | sub_industry
    peer_factors: list[str] = field(default_factory=list)
    group_factors: list[str] = field(default_factory=list)
    peer_min_size: int = 3
    # missing-pattern features: ``{f}_na`` (1 if the raw factor is missing) + ``n_missing``
    missing_flags: list[str] = field(default_factory=list)
    missing_count: bool = False
    # correlation-based peers (need ret_daily): x - median of the k most correlated
    # universe names over the trailing window -> ``{f}_cpr``
    corr_peer_factors: list[str] = field(default_factory=list)
    corr_peer_k: int = 10
    corr_peer_window: int = 250
    corr_peer_min_obs: int = 120
    corr_peer_kind: str = "drtn"
    corr_peer_method: str = "corr"  # "corr" (pairwise) | "lowrank" (rank-r SVD reconstruction)
    corr_peer_rank: int = 10  # lowrank only
    corr_peer_weighting: str = (
        "topk"  # "topk" (median of k peers) | "kernel" (softmax-weighted median)
    )
    corr_peer_tau: float = 0.1  # kernel temperature on similarity in [-1, 1]
    corr_peer_pool: int = 30  # kernel: number of most similar names that receive weight


# ------------------------------------------------------------- transforms


def rebal_index_map(cal: pd.DataFrame) -> pd.Series:
    rd = rebal_dates(cal)
    return pd.Series(rd["rebal_index"].to_numpy(), index=rd["date"])


def lag_by_rebal(panel: pd.DataFrame, cal: pd.DataFrame, cols: list[str], k: int) -> pd.DataFrame:
    """Value of ``cols`` ``k`` rebalance periods earlier, aligned to ``panel``'s (date, bid).

    Returns a frame with KEY + ``{c}_l{k}`` columns (NaN where no history).
    """
    ri = rebal_index_map(cal)
    src = panel[KEY + cols].copy()
    src["_ri"] = src["date"].map(ri).to_numpy() + k
    src = src.drop(columns="date").rename(columns={c: f"{c}_l{k}" for c in cols})
    dst = panel[KEY].copy()
    dst["_ri"] = dst["date"].map(ri).to_numpy()
    out = dst.merge(src, on=["bid", "_ri"], how="left").drop(columns="_ri")
    return out


def rosenberg_change(x_now: pd.Series, x_prev: pd.Series) -> pd.Series:
    """2 (x_t - x_{t-k}) / (|x_t| + |x_{t-k}|), NaN when the denominator is 0."""
    den = x_now.abs() + x_prev.abs()
    return (2.0 * (x_now - x_prev) / den.where(den > 0)).astype(float)


def rank_uniform(df: pd.DataFrame, cols: list[str], by: str = "date") -> pd.DataFrame:
    """Within-group rank scaled to (0,1): (rank - 0.5) / n. NaN stays NaN."""
    g = df.groupby(by, sort=False)[cols]
    r = g.rank(method="average")
    n = g.transform("count")
    return (r - 0.5) / n


def rank_gauss(df: pd.DataFrame, cols: list[str], by: str = "date") -> pd.DataFrame:
    u = rank_uniform(df, cols, by)
    return pd.DataFrame(stats.norm.ppf(u.to_numpy()), index=u.index, columns=u.columns)


# --------------------------------------------------------------- pipeline


def build_features(
    panel_x: pd.DataFrame,
    cal: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: FeatureConfig,
    sector: pd.DataFrame | None = None,
    ret_daily: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Return (features frame with KEY + weight + feature columns, feature names, observed mask).

    Only universe rows are returned; ranks (and peer groups) are computed within the universe.
    ``observed`` is a (rows, features) bool array, False where the raw feature was missing
    before filling (binary/flag columns count as observed).
    """
    missing = [f for f in cfg.factors if f not in panel_x.columns]
    if missing:
        raise KeyError(f"factors not in panel_x: {missing}")
    base = panel_x[KEY + cfg.factors]
    parts = [base]
    lag_cols = cfg.lag_factors if cfg.lag_factors is not None else cfg.factors
    chg_cols = cfg.change_factors if cfg.change_factors is not None else cfg.factors
    lag_cols = [c for c in lag_cols if c not in cfg.binary]
    chg_cols = [c for c in chg_cols if c not in cfg.binary]

    need_lags = sorted(set(cfg.lags) | set(cfg.change_lags))
    lagged: dict[int, pd.DataFrame] = {}
    for k in need_lags:
        cols = sorted(
            set(lag_cols if k in cfg.lags else []) | set(chg_cols if k in cfg.change_lags else [])
        )
        if cols:
            lagged[k] = lag_by_rebal(base, cal, cols, k)

    feat_cols: list[str] = list(cfg.factors)
    for k in cfg.lags:
        if k in lagged:
            cols = [f"{c}_l{k}" for c in lag_cols]
            parts.append(lagged[k][cols])
            feat_cols += cols
    for k in cfg.change_lags:
        if k in lagged:
            chg = pd.DataFrame(
                {f"{c}_d{k}": rosenberg_change(base[c], lagged[k][f"{c}_l{k}"]) for c in chg_cols}
            )
            parts.append(chg)
            feat_cols += list(chg.columns)
    full = pd.concat(parts, axis=1)

    # restrict to universe, then rank within date
    uni = universe[KEY + ["weight"]]
    df = uni.merge(full, on=KEY, how="left")
    if cfg.peer_level:
        if sector is None:
            raise ValueError("peer features need the sector frame")
        peer = peer_features(df, sector, cfg)
        df = pd.concat([df, peer], axis=1)
        feat_cols += list(peer.columns)
    if cfg.corr_peer_factors:
        if ret_daily is None:
            raise ValueError("corr peer features need ret_daily")
        cpr = corr_peer_features(df, ret_daily, cal, cfg)
        df = pd.concat([df, cpr], axis=1)
        feat_cols += list(cpr.columns)
    binary = list(cfg.binary)
    if cfg.missing_flags or cfg.missing_count:
        na = missing_features(df, cfg)
        df = pd.concat([df, na], axis=1)
        feat_cols += list(na.columns)
        binary += [c for c in na.columns if c.endswith("_na")]
    num_cols = [c for c in feat_cols if c not in binary]
    observed = df[feat_cols].notna().to_numpy()  # before any fill; used by imputers
    if cfg.rank == "uniform":
        ranked = rank_uniform(df, num_cols).fillna(0.5)
    elif cfg.rank == "gauss":
        ranked = rank_gauss(df, num_cols).fillna(0.0)
    else:
        raise ValueError(cfg.rank)
    df[num_cols] = ranked
    for c in binary:
        df[c] = df[c].fillna(0.0).astype(float)
    order = df[KEY].reset_index(drop=True).sort_values(KEY, kind="mergesort").index.to_numpy()
    df = df.iloc[order].reset_index(drop=True)
    observed = observed[order]
    return df[KEY + ["weight"] + feat_cols], feat_cols, observed


def peer_features(df: pd.DataFrame, sector: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Raw-value peer-relative (``_pr``) and group-median (``_gm``) features within (date, level)."""
    lvl = cfg.peer_level
    g = df[KEY].merge(sector[KEY + [lvl]], on=KEY, how="left")[lvl].fillna("NA").to_numpy()
    key = pd.Series(df["date"].astype(str).to_numpy() + "|" + g, index=df.index)
    cols = sorted(set(cfg.peer_factors) | set(cfg.group_factors))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"peer factors not among features: {missing}")
    grp = df[cols].groupby(key)
    med = grp.transform("median")
    n = grp.transform("count")
    med = med.where(n >= cfg.peer_min_size)
    out = {}
    for c in cfg.peer_factors:
        out[f"{c}_pr"] = df[c] - med[c]
    for c in cfg.group_factors:
        out[f"{c}_gm"] = med[c]
    return pd.DataFrame(out, index=df.index)


def missing_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Missing-value indicators of the raw factors (computed before any fill)."""
    out = {}
    for c in cfg.missing_flags:
        out[f"{c}_na"] = df[c].isna().astype(float)
    if cfg.missing_count:
        base = [c for c in cfg.factors if c in df.columns]
        out["n_missing"] = df[base].isna().sum(axis=1).astype(float)
    return pd.DataFrame(out, index=df.index)


def pairwise_corr(ret_mat: np.ndarray, min_obs: int) -> np.ndarray:
    """Pairwise correlation of the columns of (T, N) ``ret_mat`` using jointly observed days.

    Pairs with fewer than ``min_obs`` such days and the diagonal are NaN.
    """
    obs = ~np.isnan(ret_mat)
    x = np.where(obs, ret_mat, 0.0)
    o = obs.astype(float)
    n_pair = o.T @ o
    sx = x.T @ o  # sum of x_i over days where both i and j are observed
    sxx = (x * x).T @ o
    sxy = x.T @ x
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = sx / n_pair
        my = mx.T
        cov = sxy / n_pair - mx * my
        vx = sxx / n_pair - mx * mx
        vy = vx.T
        corr = cov / np.sqrt(vx * vy)
    corr[n_pair < min_obs] = np.nan
    np.fill_diagonal(corr, np.nan)
    return corr


def lowrank_similarity(ret_mat: np.ndarray, rank: int, min_obs: int) -> np.ndarray:
    """Rank-``rank`` denoised correlation: cosine similarity of the scaled SVD loadings.

    Columns are standardised over their observed days, missing days set to 0, then the
    top-``rank`` singular triplets of the (T, N) matrix give loadings V * S; similarity is
    the cosine between the rows of that (N, rank) embedding. Names observed on fewer than
    ``min_obs`` days are NaN.
    """
    obs = ~np.isnan(ret_mat)
    n_obs = obs.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(ret_mat, axis=0)
        sd = np.nanstd(ret_mat, axis=0)
    z = np.where(obs, (ret_mat - mu) / np.where(sd > 0, sd, np.nan), 0.0)
    z = np.nan_to_num(z)
    r = min(rank, min(z.shape) - 1)
    _, sv, vt = np.linalg.svd(z, full_matrices=False)
    emb = vt[:r].T * sv[:r]  # (N, r)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = emb / norm
    sim = e @ e.T
    bad = (n_obs < min_obs) | (norm[:, 0] == 0)
    sim[bad, :] = np.nan
    sim[:, bad] = np.nan
    np.fill_diagonal(sim, np.nan)
    return sim


def _top_corr_peers(ret_mat: np.ndarray, k: int, min_obs: int) -> np.ndarray:
    """Indices of the k most correlated columns for each column (-1 where unavailable)."""
    return _topk_from_similarity(pairwise_corr(ret_mat, min_obs), k)


def _topk_from_similarity(sim: np.ndarray, k: int) -> np.ndarray:
    n = sim.shape[0]
    peers = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        row = sim[i]
        valid = np.flatnonzero(~np.isnan(row))
        if len(valid) == 0:
            continue
        top = valid[np.argsort(-row[valid])[:k]]
        peers[i, : len(top)] = top
    return peers


def weighted_median(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Column-wise weighted median of (M, F) ``x`` with (M,) weights, ignoring NaN."""
    out = np.full(x.shape[1], np.nan)
    for f in range(x.shape[1]):
        v = x[:, f]
        m = ~np.isnan(v)
        if m.sum() == 0:
            continue
        order = np.argsort(v[m])
        vv, ww = v[m][order], w[m][order]
        cw = np.cumsum(ww) / ww.sum()
        out[f] = vv[np.searchsorted(cw, 0.5)]
    return out


def corr_peer_features(
    df: pd.DataFrame, ret_daily: pd.DataFrame, cal: pd.DataFrame, cfg: FeatureConfig
) -> pd.DataFrame:
    """x - median(x of the k most return-correlated universe names), per date.

    Correlations use the trailing ``corr_peer_window`` business days up to and including
    the rebalance date, so nothing after the date is used.
    """
    kind = cfg.corr_peer_kind
    td = cal.set_index("date")["td_index"]
    cols = list(cfg.corr_peer_factors)
    r = ret_daily[["date", "bid", kind]].dropna()
    r_td = r["date"].map(td).to_numpy()
    cal_dates = cal["date"].to_numpy()
    out = np.full((len(df), len(cols)), np.nan)
    for d, idx in df.groupby("date").indices.items():
        t = td.get(d)
        if t is None:
            continue
        lo = t - cfg.corr_peer_window + 1
        bids = df["bid"].to_numpy()[idx]
        win = r[(r_td >= lo) & (r_td <= t) & r["bid"].isin(bids)]
        mat = win.pivot(index="date", columns="bid", values=kind).reindex(
            index=cal_dates[max(lo, 0) : t + 1], columns=bids
        )
        rm = mat.to_numpy(dtype=float)
        if cfg.corr_peer_method == "corr":
            sim = pairwise_corr(rm, cfg.corr_peer_min_obs)
        elif cfg.corr_peer_method == "lowrank":
            sim = lowrank_similarity(rm, cfg.corr_peer_rank, cfg.corr_peer_min_obs)
        else:
            raise ValueError(cfg.corr_peer_method)
        x = df[cols].to_numpy(dtype=float)[idx]  # (N, F) raw values
        if cfg.corr_peer_weighting == "topk":
            peers = _topk_from_similarity(sim, cfg.corr_peer_k)
            for j, pr in enumerate(peers):
                pr = pr[pr >= 0]
                if len(pr) < max(3, cfg.corr_peer_k // 3):
                    continue
                out[idx[j]] = x[j] - np.nanmedian(x[pr], axis=0)
        elif cfg.corr_peer_weighting == "kernel":
            pool = _topk_from_similarity(sim, cfg.corr_peer_pool)
            for j, pr in enumerate(pool):
                pr = pr[pr >= 0]
                if len(pr) < 3:
                    continue
                w = np.exp((sim[j, pr] - sim[j, pr].max()) / cfg.corr_peer_tau)
                out[idx[j]] = x[j] - weighted_median(x[pr], w)
        else:
            raise ValueError(cfg.corr_peer_weighting)
    return pd.DataFrame(out, index=df.index, columns=[f"{c}_cpr" for c in cols])


def gauss_rank_target(
    y: pd.DataFrame,
    name: str,
    *,
    sector: pd.DataFrame | None = None,
    level: str = "sector",
) -> pd.Series:
    """Gaussian-rank the target within date (optionally within date x sector ``level``)."""
    if sector is None:
        return rank_gauss(y, [name])[name]
    s = y[KEY].merge(sector[KEY + [level]], on=KEY, how="left")[level].fillna("NA")
    tmp = y[[name]].copy()
    tmp["_g"] = y["date"].astype(str) + "|" + s.to_numpy()
    return rank_gauss(tmp, [name], by="_g")[name]

"""Evaluation of cross-sectional scores on the rebalance grid.

All functions take a long frame with ``date``, ``bid`` and the relevant columns.
The primary metric is the per-date rank IC (Spearman) of the score against a
forward return; secondary ones are ICIR / t-stat, quantile spreads on
``drtn_1m``, IC decay on skipped targets, score autocorrelation and turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from india_ai_model.common.targets import get_target


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return np.nan
    return float(stats.spearmanr(x[m], y[m]).statistic)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def ic_by_date(
    df: pd.DataFrame,
    score: str,
    target: str,
    *,
    method: str = "spearman",
    min_names: int = 10,
) -> pd.DataFrame:
    """Per-date IC and cross-section size. Columns: date, ic, n."""
    f = _spearman if method == "spearman" else _pearson
    rows = []
    for d, g in df.groupby("date", sort=True):
        x = g[score].to_numpy(dtype=float)
        y = g[target].to_numpy(dtype=float)
        n = int((~(np.isnan(x) | np.isnan(y))).sum())
        rows.append((d, f(x, y) if n >= min_names else np.nan, n))
    return pd.DataFrame(rows, columns=["date", "ic", "n"])


def ic_summary(ic: pd.Series) -> dict[str, float]:
    ic = ic.dropna()
    n = len(ic)
    mean = float(ic.mean()) if n else np.nan
    std = float(ic.std(ddof=1)) if n > 1 else np.nan
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if n > 1 and std > 0 else np.nan,
        "t_stat": mean / std * np.sqrt(n) if n > 1 and std > 0 else np.nan,
        "hit_rate": float((ic > 0).mean()) if n else np.nan,
        "n_dates": n,
    }


def quantile_returns(
    df: pd.DataFrame,
    score: str,
    ret: str,
    *,
    n_q: int = 5,
    weight: str | None = None,
) -> pd.DataFrame:
    """Mean (or weighted) forward return of each score quantile per date, plus top-bottom spread.

    Quantile 1 = lowest score, n_q = highest. Returns a frame indexed by date with
    columns q1..q{n_q} and ``spread``.
    """
    out = {}
    for d, g in df.groupby("date", sort=True):
        g = g.dropna(subset=[score, ret])
        if len(g) < n_q * 2:
            continue
        q = pd.qcut(g[score].rank(method="first"), n_q, labels=False) + 1
        if weight is None:
            means = g[ret].groupby(q).mean()
        else:
            w = g[weight]
            means = (g[ret] * w).groupby(q).sum() / w.groupby(q).sum()
        row = {f"q{i}": means.get(i, np.nan) for i in range(1, n_q + 1)}
        row["spread"] = row[f"q{n_q}"] - row["q1"]
        out[d] = row
    res = pd.DataFrame.from_dict(out, orient="index")
    res.index.name = "date"
    return res


def spread_summary(spread: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    s = spread.dropna()
    n = len(s)
    mean, std = (float(s.mean()), float(s.std(ddof=1))) if n > 1 else (np.nan, np.nan)
    return {
        "spread_mean": mean,
        "spread_ann": mean * periods_per_year,
        "spread_sharpe": mean / std * np.sqrt(periods_per_year) if n > 1 and std > 0 else np.nan,
        "spread_t_stat": mean / std * np.sqrt(n) if n > 1 and std > 0 else np.nan,
        "spread_hit_rate": float((s > 0).mean()) if n else np.nan,
    }


def score_autocorr(df: pd.DataFrame, score: str) -> pd.Series:
    """Spearman correlation between the score at consecutive rebalance dates (per date t)."""
    w = df.pivot(index="date", columns="bid", values=score).sort_index()
    out = {}
    prev = None
    for d, row in w.iterrows():
        if prev is not None:
            out[d] = _spearman(prev.to_numpy(dtype=float), row.to_numpy(dtype=float))
        prev = row
    return pd.Series(out, name="autocorr")


def turnover(
    df: pd.DataFrame, score: str, *, top_frac: float = 0.2, side: str = "top"
) -> pd.Series:
    """Fraction of the top (or bottom) ``top_frac`` basket replaced from one date to the next."""
    prev: set[str] | None = None
    out = {}
    for d, g in df.groupby("date", sort=True):
        g = g.dropna(subset=[score])
        k = max(int(round(len(g) * top_frac)), 1)
        basket = set((g.nlargest if side == "top" else g.nsmallest)(k, score)["bid"])
        if prev is not None and basket:
            out[d] = 1.0 - len(basket & prev) / len(basket)
        prev = basket
    return pd.Series(out, name=f"turnover_{side}")


def portfolio_turnover(df: pd.DataFrame, score: str) -> pd.Series:
    """One-way monthly turnover of a score-weighted long/short portfolio.

    Weights are the within-date z-score of ``score`` scaled so that the long and short
    legs each sum to 1 (gross = 2). Turnover_t = sum |w_t - w_{t-1}| / 2 over the union
    of names, so names entering or leaving the universe count as full trades.
    """
    w = df[["date", "bid", score]].dropna().copy()
    z = w.groupby("date")[score].transform(lambda s: (s - s.mean()) / s.std())
    gross = z.abs().groupby(w["date"]).transform("sum")
    w["w"] = 2.0 * z / gross
    mat = w.pivot(index="date", columns="bid", values="w").sort_index().fillna(0.0)
    diff = mat.diff().abs().sum(axis=1) / 2.0
    return diff.iloc[1:].rename("turnover_ls")


def attach_targets(pred: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Left-join the requested targets (from the panel_y cache) onto a prediction frame."""
    out = pred
    for n in names:
        y = get_target(n)[["date", "bid", n]]
        out = out.merge(y, on=["date", "bid"], how="left")
    return out


def evaluate_scores(
    pred: pd.DataFrame,
    score: str = "score",
    *,
    targets: list[str] | None = None,
    spread_target: str = "drtn_1m",
    n_q: int = 5,
    weight: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full evaluation. Returns (summary: one row per target, per-date frame).

    ``pred`` needs ``date``, ``bid``, ``score``; target columns are attached from the
    cache if missing.
    """
    targets = targets or ["drtn_1m", "drtn_1m_s1", "drtn_1m_s2", "srtn_1m", "drtn_3m"]
    need = [t for t in set(targets) | {spread_target} if t not in pred.columns]
    df = attach_targets(pred, need)

    summary_rows = []
    per_date = None
    for t in targets:
        ic = ic_by_date(df, score, t)
        row = {"target": t, **ic_summary(ic["ic"])}
        if t == spread_target:
            qr = quantile_returns(df, score, t, n_q=n_q, weight=weight)
            row.update(spread_summary(qr["spread"]))
            row.update({f"{c}_mean": float(qr[c].mean()) for c in qr.columns if c != "spread"})
            pdf = ic.set_index("date").rename(columns={"ic": f"ic_{t}", "n": "n"})
            pdf = pdf.join(qr[["spread"]])
        else:
            pdf = ic.set_index("date")[["ic"]].rename(columns={"ic": f"ic_{t}"})
        per_date = pdf if per_date is None else per_date.join(pdf, how="outer")
        summary_rows.append(row)
    ac = score_autocorr(df, score)
    to_top = turnover(df, score, side="top")
    to_bot = turnover(df, score, side="bottom")
    to_ls = portfolio_turnover(df, score)
    per_date = per_date.join(ac).join(to_top).join(to_bot).join(to_ls)
    summary = pd.DataFrame(summary_rows)
    summary["score_autocorr"] = float(ac.mean())
    summary["turnover_top20"] = float(to_top.mean())
    summary["turnover_bottom20"] = float(to_bot.mean())
    summary["turnover_ls"] = float(to_ls.mean())
    return summary, per_date.reset_index()


def ic_per_date_fast(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> pd.Series:
    """Vectorised per-date Spearman IC (for training loops / early stopping)."""
    df = pd.DataFrame({"d": dates, "p": pred, "y": y})
    g = df.groupby("d", sort=True)
    rp = g["p"].rank()
    ry = g["y"].rank()
    rp = rp - rp.groupby(df["d"]).transform("mean")
    ry = ry - ry.groupby(df["d"]).transform("mean")
    num = (rp * ry).groupby(df["d"]).sum()
    den = np.sqrt((rp * rp).groupby(df["d"]).sum() * (ry * ry).groupby(df["d"]).sum())
    return num / den.replace(0, np.nan)


def mean_ic_fast(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> float:
    return float(ic_per_date_fast(pred, y, dates).mean())


# ------------------------------------------------------------ smoothing


def smooth_scores(pred: pd.DataFrame, score: str, halflife: float) -> pd.Series:
    """EMA of the within-date z-scored score along consecutive rebalance dates, per name.

    s_t = a * z_t + (1 - a) * s_{t-1} with a = 1 - 0.5 ** (1 / halflife). The EMA restarts
    (s_t = z_t) when the name was absent at t-1, so stale scores of re-entrants are not
    reused. ``halflife <= 0`` returns the z-scored score unchanged. Only past predictions
    enter s_t, so the walk-forward stays leak-free.
    """
    z = pred.groupby("date")[score].transform(lambda s: (s - s.mean()) / s.std())
    if halflife <= 0:
        return z.rename(f"{score}_hl0")
    a = 1.0 - 0.5 ** (1.0 / halflife)
    mat = pd.DataFrame({"date": pred["date"], "bid": pred["bid"], "z": z}).pivot(
        index="date", columns="bid", values="z"
    )
    mat = mat.sort_index()
    out = np.full(mat.shape, np.nan)
    vals = mat.to_numpy()
    prev = np.full(mat.shape[1], np.nan)
    for i in range(len(mat)):
        z_i = vals[i]
        s = np.where(np.isnan(prev), z_i, a * z_i + (1 - a) * prev)
        s = np.where(np.isnan(z_i), np.nan, s)  # absent today -> no score, EMA resets
        out[i] = s
        prev = s
    sm = pd.DataFrame(out, index=mat.index, columns=mat.columns).stack()
    sm.name = f"{score}_hl{halflife:g}"
    idx = pd.MultiIndex.from_arrays([pred["date"], pred["bid"]])
    return pd.Series(sm.reindex(idx).to_numpy(), index=pred.index, name=sm.name)


def evaluate_smoothing(
    pred: pd.DataFrame,
    score: str = "score",
    *,
    halflives: list[float],
    targets: list[str] | None = None,
    spread_target: str = "drtn_1m",
    n_q: int = 5,
    weight: str | None = None,
) -> pd.DataFrame:
    """IC / spread / turnover for the raw score and for each EMA half-life (months)."""
    rows = []
    for hl in halflives:
        col = f"{score}_hl{hl:g}"
        df = pred.assign(**{col: smooth_scores(pred, score, hl)})
        summary, _ = evaluate_scores(
            df, col, targets=targets, spread_target=spread_target, n_q=n_q, weight=weight
        )
        summary.insert(0, "halflife", hl)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    """Post-hoc smoothing evaluation of a run: ``python -m india_ai_model.common.evaluate runs/x/y``."""
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--halflives", type=float, nargs="+", default=[0, 1, 2, 3, 6])
    p.add_argument("--targets", nargs="+", default=["drtn_1m", "drtn_3m"])
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir)
    pred = pd.read_parquet(run_dir / "pred.parquet")
    res = evaluate_smoothing(pred, halflives=a.halflives, targets=a.targets)
    res.to_csv(run_dir / "metrics_smooth.csv", index=False)
    cols = [
        "halflife",
        "target",
        "ic_mean",
        "icir",
        "t_stat",
        "spread_mean",
        "spread_sharpe",
        "score_autocorr",
        "turnover_top20",
        "turnover_bottom20",
        "turnover_ls",
    ]
    pd.set_option("display.width", 200)
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

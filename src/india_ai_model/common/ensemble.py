"""Combine the predictions of finished runs into an ensemble run.

Members are z-scored within date and averaged with either equal weights or weights
proportional to each member's realised IC over the trailing ``ic_window`` months
(past predictions only, so the walk-forward stays leak-free).

Usage:
  python -m india_ai_model.common.ensemble --name ens3 --runs runs/a/exp runs/b/exp [--ic-weighted]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from india_ai_model.common import io
from india_ai_model.common.evaluate import evaluate_scores, evaluate_smoothing, ic_by_date
from india_ai_model.common.targets import get_target

log = logging.getLogger("ensemble")
KEY = ["date", "bid"]


def load_members(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Inner-join member scores on (date, bid); columns named after the run's config stem."""
    out = None
    for d in run_dirs:
        d = Path(d)
        name = d.parent.name
        p = pd.read_parquet(d / "pred.parquet")[KEY + ["score"]].rename(columns={"score": name})
        out = p if out is None else out.merge(p, on=KEY, how="inner")
    return out


def zscore_within_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby("date")[cols]
    return (df[cols] - g.transform("mean")) / g.transform("std")


def trailing_ic_weights(
    df: pd.DataFrame, cols: list[str], target: str, window: int, floor: float = 0.0
) -> pd.DataFrame:
    """Per-date member weights ∝ clip(mean IC over the previous ``window`` dates, floor).

    Dates without enough history fall back to equal weights.
    """
    y = get_target(target)[KEY + [target]]
    m = df.merge(y, on=KEY, how="left")
    ics = pd.concat(
        [ic_by_date(m, c, target).set_index("date")["ic"].rename(c) for c in cols], axis=1
    ).sort_index()
    trailing = ics.shift(1).rolling(window, min_periods=window).mean()
    w = trailing.clip(lower=floor)
    w = w.div(w.sum(axis=1), axis=0)
    w = w.where(w.notna().all(axis=1), 1.0 / len(cols))
    return w


def combine(
    members: pd.DataFrame,
    cols: list[str],
    *,
    ic_weighted: bool = False,
    target: str = "drtn_1m",
    ic_window: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    z = zscore_within_date(members, cols)
    if not ic_weighted:
        score = z.mean(axis=1)
        weights = None
    else:
        weights = trailing_ic_weights(members, cols, target, ic_window)
        w_rows = weights.reindex(members["date"]).to_numpy()
        score = (z.to_numpy() * w_rows).sum(axis=1)
    pred = members[KEY].copy()
    pred["score"] = np.asarray(score, dtype=float)
    return pred, weights


def run(
    name: str,
    run_dirs: list[str],
    *,
    ic_weighted: bool = False,
    ic_window: int = 24,
    run_id: str = "exp",
    halflives: list[float] | None = None,
) -> Path:
    members = load_members(run_dirs)
    cols = [c for c in members.columns if c not in KEY]
    pred, weights = combine(members, cols, ic_weighted=ic_weighted, ic_window=ic_window)
    out = io.project_root() / "runs" / name / run_id
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "model": "ensemble",
                "target": "drtn_1m",
                "members": [str(d) for d in run_dirs],
                "ic_weighted": ic_weighted,
                "ic_window": ic_window,
            },
            f,
            sort_keys=False,
        )
    io.write_parquet(pred, out / "pred.parquet")
    if weights is not None:
        weights.reset_index().to_csv(out / "weights.csv", index=False)
    uni = io.read_universe_monthly()
    pred_w = pred.merge(uni[KEY + ["weight"]], on=KEY, how="left")
    targets = ["drtn_1m", "drtn_3m", "drtn_1m_s1", "drtn_1m_s2", "srtn_1m", "gp_trtn_1m"]
    summary, per_date = evaluate_scores(pred_w, targets=targets)
    summary.to_csv(out / "metrics.csv", index=False)
    io.write_parquet(per_date, out / "per_date.parquet")
    sm = evaluate_smoothing(pred_w, halflives=halflives or [0, 1, 2, 3], targets=["drtn_1m"])
    sm.to_csv(out / "metrics_smooth.csv", index=False)
    # member diagnostics: mean within-date rank correlation
    corr = members.groupby("date")[cols].corr(method="spearman").groupby(level=1).mean()
    corr.to_csv(out / "member_corr.csv")
    log.info(
        "%s: %s", name, summary.iloc[0][["ic_mean", "icir", "t_stat", "turnover_ls"]].to_dict()
    )
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--ic-weighted", action="store_true")
    p.add_argument("--ic-window", type=int, default=24)
    p.add_argument("--run-id", default="exp")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(a.name, a.runs, ic_weighted=a.ic_weighted, ic_window=a.ic_window, run_id=a.run_id)


if __name__ == "__main__":
    main()

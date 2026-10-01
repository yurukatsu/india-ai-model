"""Generic walk-forward runner: config.yaml -> runs/<model>/<run_id>/.

A model lives in ``india_ai_model.models.<name>.model`` and exposes ``Model`` with

    Model(params: dict, feature_names: list[str])
    .fit(X_tr, y_tr, dates_tr, X_va, y_va, dates_va, w_tr=None) -> None
    .predict(X, dates) -> np.ndarray
    .info() -> dict            (optional; logged per split)

X are float32 arrays of ranked features; y is the Gaussian-ranked target; dates are
per-row numpy datetime64 arrays (models that batch by date use them).
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from india_ai_model.common import io
from india_ai_model.common.evaluate import evaluate_scores, evaluate_smoothing
from india_ai_model.common.features import FeatureConfig, build_features, gauss_rank_target
from india_ai_model.common.impute import MaskedAEImputer
from india_ai_model.common.targets import get_target, mask_low_coverage, parse_target_name
from india_ai_model.common.walkforward import WalkForwardConfig, make_splits

log = logging.getLogger("runner")
KEY = ["date", "bid"]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_factors(fcfg: dict[str, Any], meta: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Expand ``factors`` (list, or 'all', or {categories: [...]}, minus ``exclude``)."""
    spec = fcfg.get("factors", "all")
    if spec == "all":
        factors = meta["factor"].tolist()
    elif isinstance(spec, dict) and "categories" in spec:
        factors = meta.loc[meta["category"].isin(spec["categories"]), "factor"].tolist()
    else:
        factors = list(spec)
    exclude = set(fcfg.get("exclude", []))
    max_sent = fcfg.get("max_sentinel_frac")
    if max_sent is not None:
        exclude |= set(meta.loc[meta["sentinel_frac"] > max_sent, "factor"])
    factors = [f for f in factors if f not in exclude]
    binary = meta.loc[meta["is_binary"] & meta["factor"].isin(factors), "factor"].tolist()
    return factors, binary


def build_dataset(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str], pd.DataFrame, np.ndarray]:
    """Return (frame with KEY, weight, features, y, y_raw, end_date; feature names; calendar; observed mask)."""
    cal = io.read_calendar()
    meta = io.read_mart("factor_meta")
    uni = io.read_universe_monthly()
    fcfg = dict(cfg.get("features", {}))
    factors, binary = resolve_factors(fcfg, meta)
    fc = FeatureConfig(
        factors=factors,
        lags=list(fcfg.get("lags", [3, 6, 12])),
        change_lags=list(fcfg.get("change_lags", [3, 12])),
        lag_factors=fcfg.get("lag_factors"),
        change_factors=fcfg.get("change_factors"),
        rank=fcfg.get("rank", "uniform"),
        binary=binary,
        peer_level=fcfg.get("peer_level"),
        peer_factors=list(fcfg.get("peer_factors", [])),
        group_factors=list(fcfg.get("group_factors", [])),
        peer_min_size=int(fcfg.get("peer_min_size", 3)),
        missing_flags=list(fcfg.get("missing_flags", [])),
        missing_count=bool(fcfg.get("missing_count", False)),
        corr_peer_factors=list(fcfg.get("corr_peer_factors", [])),
        corr_peer_k=int(fcfg.get("corr_peer_k", 10)),
        corr_peer_window=int(fcfg.get("corr_peer_window", 250)),
        corr_peer_min_obs=int(fcfg.get("corr_peer_min_obs", 120)),
        corr_peer_kind=fcfg.get("corr_peer_kind", "drtn"),
        corr_peer_method=fcfg.get("corr_peer_method", "corr"),
        corr_peer_rank=int(fcfg.get("corr_peer_rank", 10)),
        corr_peer_weighting=fcfg.get("corr_peer_weighting", "topk"),
        corr_peer_tau=float(fcfg.get("corr_peer_tau", 0.1)),
        corr_peer_pool=int(fcfg.get("corr_peer_pool", 30)),
    )
    px = io.read_panel_x(factors)
    sector = io.read_sector_monthly() if fc.peer_level else None
    ret_daily = io.read_ret_daily([fc.corr_peer_kind]) if fc.corr_peer_factors else None
    feats, feat_cols, observed = build_features(
        px, cal, uni, fc, sector=sector, ret_daily=ret_daily
    )
    log.info(
        "features: %d rows, %d columns (%d base factors)", len(feats), len(feat_cols), len(factors)
    )

    tname = cfg["target"]
    y = get_target(tname)
    y = mask_low_coverage(y, tname, cfg.get("min_obs_frac", 0.5))
    y = y.rename(columns={tname: "y_raw"})[KEY + ["y_raw", "end_date"]]
    df = feats.merge(y, on=KEY, how="left")
    ttf = cfg.get("target_transform", "gauss")
    if ttf == "gauss":
        sector = None
    elif ttf == "gauss_sector":
        sector = io.read_sector_monthly()
        if sector is None:
            raise FileNotFoundError(
                "target_transform=gauss_sector needs mart/sector_monthly.parquet"
            )
    else:
        raise ValueError(ttf)
    df = df.copy()
    df["y"] = gauss_rank_target(
        df, "y_raw", sector=sector, level=cfg.get("sector_level", "sector")
    ).to_numpy()
    return df, feat_cols, cal, observed


def sample_weights(df: pd.DataFrame, wcfg: dict[str, Any], T: pd.Timestamp) -> np.ndarray:
    """Per-row training weights = date weight x time decay x within-date name weight.

    wcfg:
      date: "n" (every row equal; default) | "sqrt_n" (date total = sqrt(N_d)) | "equal"
      name: "equal" (default) | "sqrt_w" | "log_w"   (w = index weight)
      time_halflife_months: null | float   (0.5 ** (months before T / halflife))
    """
    n_d = df.groupby("date")["bid"].transform("count").to_numpy(dtype=float)
    mode = wcfg.get("date", "n")
    if mode == "n":
        date_w = n_d
    elif mode == "sqrt_n":
        date_w = np.sqrt(n_d)
    elif mode == "equal":
        date_w = np.ones_like(n_d)
    else:
        raise ValueError(mode)
    name_mode = wcfg.get("name", "equal")
    w = df["weight"].to_numpy(dtype=float)
    if name_mode == "equal":
        name_w = np.ones_like(w)
    elif name_mode == "sqrt_w":
        name_w = np.sqrt(w)
    elif name_mode == "log_w":
        name_w = np.log1p(w * 1e4)  # bp scale so the log is positive and not too flat
    else:
        raise ValueError(name_mode)
    name_w = name_w / pd.Series(name_w).groupby(df["date"].to_numpy()).transform("sum").to_numpy()
    hl = wcfg.get("time_halflife_months")
    if hl:
        months = (T.year - df["date"].dt.year) * 12 + (T.month - df["date"].dt.month)
        decay = 0.5 ** (months.to_numpy(dtype=float) / float(hl))
    else:
        decay = np.ones_like(n_d)
    return (date_w * decay * name_w).astype(np.float32)


def _load_model_class(name: str):
    mod = importlib.import_module(f"india_ai_model.models.{name}.model")
    return mod.Model


def run(config_path: str | Path, run_id: str | None = None, *, overwrite: bool = False) -> Path:
    cfg = load_config(config_path)
    name = cfg["model"]
    variant = Path(config_path).stem  # runs are grouped by config name, not model class
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = io.project_root() / "runs" / variant / run_id
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pick another --run-id or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    log.info("run %s (%s)/%s -> %s", variant, name, run_id, out)

    t0 = time.time()
    df, feat_cols, cal, observed = build_dataset(cfg)
    horizon = parse_target_name(cfg["target"]).horizon
    wf = WalkForwardConfig(**cfg["walkforward"])
    splits = make_splits(cal, wf, horizon)
    with open(out / "splits.txt", "w") as f:
        f.write("\n".join(repr(s) for s in splits))
    log.info("%d splits, first %s, last %s", len(splits), splits[0], splits[-1])

    Model = _load_model_class(name)
    params = dict(cfg.get("params", {}))
    wcfg = cfg.get("sample_weight") or {}
    icfg = cfg.get("impute") or {}
    if icfg and icfg.get("method", "masked_ae") != "masked_ae":
        raise ValueError(icfg["method"])
    binary_cols = np.array(
        [
            c.endswith("_na")
            or c
            in set(resolve_factors(dict(cfg.get("features", {})), io.read_mart("factor_meta"))[1])
            for c in feat_cols
        ]
    )
    impute_cols = ~binary_cols
    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    y_all = df["y"].to_numpy(dtype=np.float32)
    d_all = df["date"].to_numpy()
    has_y = ~np.isnan(y_all)

    preds = []
    infos = []
    for s in splits:
        ts = time.time()
        tr = np.isin(d_all, np.array(s.train, dtype="datetime64[ns]")) & has_y
        va = np.isin(d_all, np.array(s.val, dtype="datetime64[ns]")) & has_y
        pr = np.isin(d_all, np.array(s.predict, dtype="datetime64[ns]"))
        w_all = sample_weights(df, wcfg, s.T) if wcfg else None
        w_tr = w_all[tr] if w_all is not None else None
        X_use = X_all
        imp_info: dict = {}
        if icfg:
            imp = MaskedAEImputer({k: v for k, v in icfg.items() if k != "method"})
            obs = observed | ~impute_cols  # never touch binary / flag columns
            imp.fit(X_all[tr], obs[tr])
            rows = tr | va | pr
            X_use = X_all.copy()
            X_use[rows] = imp.transform(X_all[rows], obs[rows])
            imp_info = imp.info_
        model = Model(params, feat_cols)
        model.fit(X_use[tr], y_all[tr], d_all[tr], X_use[va], y_all[va], d_all[va], w_tr=w_tr)
        if wf.refit_with_val:
            model = Model(params, feat_cols)
            model.fit(
                X_use[tr | va],
                y_all[tr | va],
                d_all[tr | va],
                None,
                None,
                None,
                w_tr=w_all[tr | va] if w_all is not None else None,
            )
        score = model.predict(X_all[pr], d_all[pr])
        p = df.loc[pr, KEY].copy()
        p["score"] = score.astype(np.float64)
        p["T"] = s.T
        preds.append(p)
        info = model.info() if hasattr(model, "info") else {}
        info = {
            "T": str(s.T.date()),
            "n_train": int(tr.sum()),
            "n_val": int(va.sum()),
            **imp_info,
            **info,
        }
        infos.append(info)
        log.info(
            "T=%s n_train=%d n_val=%d %.1fs %s",
            s.T.date(),
            tr.sum(),
            va.sum(),
            time.time() - ts,
            json.dumps(info, default=str),
        )

    pred = pd.concat(preds, ignore_index=True)
    pred = pred.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    io.write_parquet(pred, out / "pred.parquet")
    by_date = out / "pred_by_date"
    by_date.mkdir(exist_ok=True)
    for d, g in pred.groupby("date"):
        io.write_parquet(g[KEY + ["score"]], by_date / f"{d.strftime('%Y%m%d')}.parquet")
    with open(out / "fit_info.json", "w") as f:
        json.dump(infos, f, indent=1, default=str)

    ecfg = cfg.get("eval", {})
    pred_w = pred.merge(df[KEY + ["weight"]], on=KEY, how="left")
    summary, per_date = evaluate_scores(
        pred_w,
        targets=ecfg.get("targets"),
        spread_target=ecfg.get("spread_target", "drtn_1m"),
        n_q=ecfg.get("n_quantiles", 5),
        weight="weight" if ecfg.get("weighted_spread", False) else None,
    )
    io.write_parquet(summary, out / "metrics.parquet")
    io.write_parquet(per_date, out / "per_date.parquet")
    summary.to_csv(out / "metrics.csv", index=False)
    halflives = ecfg.get("smooth_halflives")
    if halflives:
        sm = evaluate_smoothing(
            pred_w,
            halflives=[float(h) for h in halflives],
            targets=ecfg.get("smooth_targets", [cfg["target"]]),
            spread_target=ecfg.get("spread_target", "drtn_1m"),
            n_q=ecfg.get("n_quantiles", 5),
        )
        sm.to_csv(out / "metrics_smooth.csv", index=False)
        log.info(
            "smoothing:\n%s",
            sm[["halflife", "target", "ic_mean", "icir", "turnover_ls"]].to_string(index=False),
        )
    log.info("done in %.1fs\n%s", time.time() - t0, summary.to_string())
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--run-id")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(a.config, a.run_id, overwrite=a.overwrite)


if __name__ == "__main__":
    main()

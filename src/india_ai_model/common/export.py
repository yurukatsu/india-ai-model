"""Export run predictions to the external evaluation format.

    output/<score_name>/<yyyymm>.dat     one file per rebalance month
        #bid <score_name>
        INDABI1 -0.0784833061485055
        ...
    output/scores.csv                    catalogue of every exported score

Scores are z-scored within date so that all runs share one scale. For each run two
series are written: the raw score and an EMA-smoothed one (``<name>_hl<h>``).

Usage: ``python -m india_ai_model.common.export [--runs runs/a/exp ...] [--halflives 1]``
(no --runs = every run under runs/ that has pred.parquet and metrics.csv).
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from india_ai_model.common import io
from india_ai_model.common.evaluate import smooth_scores

log = logging.getLogger("export")
KEY = ["date", "bid"]


def score_name_of(run_dir: Path) -> str:
    cfg_name, run_id = run_dir.parent.name, run_dir.name
    return cfg_name if run_id == "exp" else f"{cfg_name}__{run_id}"


def describe(cfg: dict) -> tuple[str, str]:
    """Return (variant tags, Japanese description) from a run config."""
    f = cfg.get("features", {}) or {}
    p = cfg.get("params", {}) or {}
    tags, parts = [], []
    model = cfg["model"]
    if model == "ensemble":
        members = [Path(m).parent.name for m in cfg["members"]]
        tags.append("members=" + "+".join(members))
        parts.append(
            (
                f"過去 {cfg.get('ic_window', 24)} ヶ月の IC 加重"
                if cfg.get("ic_weighted")
                else "等ウェイト"
            )
            + "の z-score 平均アンサンブル: "
            + ", ".join(members)
        )
        return ",".join(tags), "".join(parts)
    names = {"ridge": "リッジ回帰", "lgbm": "LightGBM", "dnn": "DNN"}
    base = names.get(model, model)
    if model == "lgbm":
        base += "（{}）".format(p.get("objective", "lambdarank"))
    if model == "dnn":
        arch = p.get("arch", "mlp")
        base += "（{}, シード {}）".format(
            "cross-sectional attention" if arch == "attn" else "MLP",
            p.get("seeds", 5),
        )
        if arch != "mlp":
            tags.append(arch)
        s = p.get("seeds", 5)
        if s != 5:
            tags.append(f"seeds={s}")
    parts.append(base)
    parts.append("。ターゲット {}".format(cfg["target"]))
    if cfg.get("target_transform") == "gauss_sector":
        tags.append("sector_y")
        parts.append("（{} 内で中立化）".format(cfg.get("sector_level", "sector")))
    feats = ["基本ファクター＋ラグ・変化率"]
    if f.get("peer_level"):
        tags.append("peer")
        feats.append("{} 内ピア相対".format(f["peer_level"]))
    if f.get("corr_peer_factors"):
        tags.append("cpr")
        desc = "リターン相関ピア相対（{}, {}".format(
            f.get("corr_peer_kind", "drtn"),
            f.get("corr_peer_method", "corr"),
        )
        if f.get("corr_peer_weighting", "topk") != "topk":
            desc += ", " + f["corr_peer_weighting"]
        feats.append(desc + "）")
        for k in ("corr_peer_kind", "corr_peer_method", "corr_peer_weighting"):
            v = f.get(k)
            if v and v not in ("drtn", "corr", "topk"):
                tags.append(v)
    if f.get("missing_flags"):
        tags.append("na")
        feats.append("欠損フラグ")
    parts.append("。特徴量: " + "、".join(feats))
    if cfg.get("sample_weight"):
        w = cfg["sample_weight"]
        tags.append("w:" + ",".join(f"{k}={v}" for k, v in w.items()))
        parts.append("。サンプル重み " + ", ".join(f"{k}={v}" for k, v in w.items()))
    if cfg.get("impute"):
        tags.append("ae")
        parts.append("。masked AE で欠損補完")
    return ",".join(tags), "".join(parts)


def export_run(run_dir: Path, out_root: Path, *, halflives: list[float]) -> list[dict]:
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    pred = pd.read_parquet(run_dir / "pred.parquet")[KEY + ["score"]]
    metrics = pd.read_csv(run_dir / "metrics.csv").set_index("target")
    ev = "drtn_1m" if "drtn_1m" in metrics.index else cfg["target"]
    m = metrics.loc[ev]
    base_name = score_name_of(run_dir)
    tags, desc = describe(cfg)
    rows = []
    series = {base_name: (smooth_scores(pred, "score", 0.0), 0.0)}
    for h in halflives:
        series[f"{base_name}_hl{h:g}"] = (smooth_scores(pred, "score", h), h)
    smooth = None
    if (run_dir / "metrics_smooth.csv").exists():
        smooth = pd.read_csv(run_dir / "metrics_smooth.csv")
    for name, (s, h) in series.items():
        df = pred[KEY].assign(score=s.to_numpy()).dropna()
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        for dt, g in df.groupby("date"):
            g = g.sort_values("bid")
            with open(d / f"{dt.strftime('%Y%m')}.dat", "w") as fh:
                fh.write(f"#bid {name}\n")
                fh.writelines(f"{b} {v!r}\n" for b, v in zip(g["bid"], g["score"], strict=True))
        ic, icir, to = m["ic_mean"], m["icir"], m.get("turnover_ls", np.nan)
        if h > 0 and smooth is not None:
            r = smooth[(smooth["halflife"] == h) & (smooth["target"] == ev)]
            if len(r):
                ic, icir, to = r.iloc[0]["ic_mean"], r.iloc[0]["icir"], r.iloc[0]["turnover_ls"]
        n_by_date = df.groupby("date").size()
        rows.append(
            {
                "score_name": name,
                "description": desc + ("" if h == 0 else f"。EMA 平滑化（半減期 {h:g} ヶ月）"),
                "model": cfg["model"],
                "target": cfg["target"],
                "variant": tags,
                "smooth_halflife": h,
                "value": "within-date z-score",
                "n_files": int(n_by_date.size),
                "date_min": df["date"].min().strftime("%Y-%m-%d"),
                "date_max": df["date"].max().strftime("%Y-%m-%d"),
                "n_bid_mean": round(float(n_by_date.mean()), 1),
                "ic_drtn_1m": round(float(ic), 4),
                "icir_drtn_1m": round(float(icir), 4),
                "ic_drtn_3m": round(float(metrics.loc["drtn_3m", "ic_mean"]), 4)
                if "drtn_3m" in metrics.index
                else np.nan,
                "spread_q5q1_pct": round(float(m["spread_mean"]) * 100, 3)
                if "spread_mean" in m.index
                else np.nan,
                "turnover_ls": round(float(to), 4) if pd.notna(to) else np.nan,
                "run_dir": str(run_dir),
                "config": f"configs/{run_dir.parent.name}.yaml",
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        )
    log.info("%s: %d series x %d months", base_name, len(series), int(n_by_date.size))
    return rows


def export_all(run_dirs: list[Path], out_root: Path, *, halflives: list[float]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        if not (d / "pred.parquet").exists() or not (d / "metrics.csv").exists():
            log.warning("skip %s (no pred/metrics)", d)
            continue
        rows += export_run(d, out_root, halflives=halflives)
    cat = pd.DataFrame(rows).sort_values("score_name").reset_index(drop=True)
    cat.to_csv(out_root / "scores.csv", index=False)
    return cat


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", help="run dirs; default = all runs/*/*")
    p.add_argument("--out", default=str(io.project_root() / "output"))
    p.add_argument("--halflives", type=float, nargs="*", default=[1.0])
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    runs = [Path(r) for r in a.runs] if a.runs else sorted(io.project_root().glob("runs/*/*"))
    cat = export_all(runs, Path(a.out), halflives=a.halflives)
    log.info("wrote %d score series -> %s/scores.csv", len(cat), a.out)


if __name__ == "__main__":
    main()

"""Build ``data/mart`` from ``data/raw``.

Outputs (all keyed by ``date``, ``bid`` as plain columns, sorted, unique):

- calendar.parquet          Indian business days + monthly rebalance dates
- ret_daily.parquet         wide daily returns per kind, in *decimal* units
- universe_monthly.parquet  date(rebal), bid, weight
- sector_monthly.parquet    date(rebal), bid, sector        (only if raw/sector exists)
- panel_x.parquet           date(rebal), bid, <all factor columns>, raw values, sentinels -> NaN
- factor_meta.csv           factor, source_table, category, is_binary, description, sentinel_frac
- _qc_report.parquet        long-format QC results (check, date, bid, value, note)

Usage: ``uv run build-mart [--rebal-lag 0] [--universe-join ym|asof]``
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from india_ai_model.common import io
from india_ai_model.common.calendar import (
    attach_rebal_date,
    build_calendar,
    business_days_from_returns,
    rebal_dates,
)
from india_ai_model.common.factor_meta import category_of, description_of

log = logging.getLogger("build_mart")

QC_COLUMNS = ["check", "date", "bid", "value", "note"]

# units of the raw ``ret`` column per kind; unknown kinds are inferred from the data
RETURN_UNITS = {"drtn": "pct", "srtn": "pct", "drtnp": "pct", "gp_trtn": "decimal"}
# kinds derived from the same Barra price series (masked together)
BARRA_KINDS = ["drtn", "srtn"]
SECTOR_LEVELS = ["sector", "industry_group", "industry", "sub_industry"]


def _qc(check: str, date=pd.NaT, bid=None, value=np.nan, note="") -> dict:
    return {"check": check, "date": date, "bid": bid, "value": float(value), "note": str(note)}


def _qc_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=QC_COLUMNS)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["bid"] = df["bid"].astype("str")
    return df


# ------------------------------------------------------------------ builders


def build_returns(kinds: list[str], *, min_names: int, zero_frac_max: float, rebal_lag: int):
    """Return (calendar, ret_daily wide in decimals, qc rows)."""
    qc: list[dict] = []
    longs = []
    for k in kinds:
        df = io.read_raw_return(k)
        n_sent = int((df["ret"] <= io.SENTINEL_THRESHOLD).sum())
        qc.append(_qc("ret_sentinel_rows", value=n_sent, note=k))
        df = io.clean_sentinel(df, ["ret"])
        io.assert_unique_key(df, f"raw/return kind={k}")
        df["kind"] = k
        longs.append(df)
        log.info("return kind=%s rows=%d sentinel=%d", k, len(df), n_sent)
    long = pd.concat(longs, ignore_index=True)

    if "drtn" not in kinds:
        raise RuntimeError("kind=drtn is required to build the calendar")
    days, diag = business_days_from_returns(
        long[long["kind"] == "drtn"], min_names=min_names, zero_frac_max=zero_frac_max
    )
    cal = build_calendar(days, rebal_lag=rebal_lag)
    for r in diag[~diag["is_bday"]].itertuples():
        qc.append(_qc("calendar_dropped_day", r.date, value=r.zero_frac, note=f"n_obs={r.n_obs}"))

    wide = long.pivot_table(index=["date", "bid"], columns="kind", values="ret", observed=True)
    wide = wide.reset_index()
    wide.columns.name = None
    wide = wide[wide["date"].isin(cal["date"])]
    for k in kinds:
        unit = RETURN_UNITS.get(k)
        if unit is None:  # infer: daily returns in percent have a much larger scale
            unit = "pct" if wide[k].abs().median() > 0.05 else "decimal"
            log.warning("kind=%s not in RETURN_UNITS; inferred unit=%s", k, unit)
        if unit == "pct":
            wide[k] = wide[k] / 100.0
    wide = io.sort_by_key(wide)[io.KEY + kinds]
    return cal, wide, qc


def mask_bad_returns(
    ret: pd.DataFrame,
    *,
    kinds: list[str],
    max_abs_ret: float,
    confirm_kind: str | None,
    confirm_tol: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """Set ``kinds`` to NaN on days where |drtn| > max_abs_ret and no independent source confirms it.

    Barra returns occasionally carry unadjusted corporate actions (e.g. +5,400% on a
    reverse split). A row is kept if ``confirm_kind`` (e.g. DS2 total return) is present
    and within ``confirm_tol`` of drtn; otherwise all Barra-derived kinds are masked on
    that row (srtn is derived from drtn, so it carries the same error). Masking the day
    keeps the compounded return of the surrounding period correct.
    """
    qc: list[dict] = []
    if max_abs_ret is None or max_abs_ret <= 0 or "drtn" not in ret.columns:
        return ret, qc
    big = ret["drtn"].abs() > max_abs_ret
    if confirm_kind and confirm_kind in ret.columns:
        confirmed = (ret["drtn"] - ret[confirm_kind]).abs() <= confirm_tol
        mask = big & ~confirmed.fillna(False)
    else:
        mask = big
    out = ret.copy()
    for r in out.loc[mask, io.KEY + ["drtn"]].itertuples(index=False):
        qc.append(_qc("ret_masked", r.date, r.bid, value=r.drtn, note=",".join(kinds)))
    for k in kinds:
        if k in out.columns:
            out[k] = out[k].mask(mask)
    log.info(
        "masked %d rows (|drtn|>%.2f, unconfirmed by %s) out of %d with |drtn|>%.2f",
        int(mask.sum()),
        max_abs_ret,
        confirm_kind,
        int(big.sum()),
        max_abs_ret,
    )
    return out, qc


def build_universe(cal: pd.DataFrame, how: str):
    uni = io.read_raw_universe()
    io.assert_unique_key(uni, "raw/universe")
    out = attach_rebal_date(uni, cal, how=how)
    io.assert_unique_key(out, "universe_monthly")
    return out[["date", "bid", "weight"]]


def build_sector(cal: pd.DataFrame, how: str) -> pd.DataFrame | None:
    sec = io.read_raw_sector()
    if sec is None:
        log.warning("raw/sector not found; skipping sector_monthly")
        return None
    io.assert_unique_key(sec, "raw/sector")
    out = attach_rebal_date(sec, cal, how=how)
    io.assert_unique_key(out, "sector_monthly")
    levels = [c for c in SECTOR_LEVELS if c in out.columns]
    if "sector" not in levels:
        raise ValueError("raw/sector must have a 'sector' column")
    out = out[["date", "bid"] + levels].copy()
    for c in levels:
        out[c] = out[c].astype("str")
    return out


def build_panel_x(cal: pd.DataFrame):
    """Horizontally join all factor tables on (rebal date, bid). Returns (panel, meta, qc)."""
    qc: list[dict] = []
    panel = None
    meta_rows = []
    rdates = set(rebal_dates(cal)["ym"])
    for table in io.list_factor_tables():
        raw = io.read_raw_factor(table)
        io.assert_unique_key(raw, f"raw/factor table={table}")
        fcols = [c for c in raw.columns if c not in io.KEY]
        # dates in raw that do not map to a rebalance month
        raw_ym = raw["date"].dt.strftime("%Y-%m")
        for d in sorted(raw.loc[~raw_ym.isin(rdates), "date"].unique()):
            qc.append(_qc("factor_date_unmapped", d, note=table))
        for c in fcols:
            frac = float((raw[c] <= io.SENTINEL_THRESHOLD).mean())
            vals = raw.loc[raw[c] > io.SENTINEL_THRESHOLD, c].dropna().unique()
            is_bin = bool(len(vals) > 0 and np.isin(vals, [0.0, 1.0]).all())
            meta_rows.append(
                {
                    "factor": c,
                    "source_table": table,
                    "category": category_of(c),
                    "is_binary": is_bin,
                    "description": description_of(c),
                    "sentinel_frac": round(frac, 4),
                }
            )
        df = io.clean_sentinel(raw, fcols)
        df = attach_rebal_date(df, cal, how="ym")
        log.info("factor table=%s rows=%d cols=%d", table, len(df), len(fcols))
        if panel is None:
            panel = df
        else:
            dup = [c for c in fcols if c in panel.columns]
            if dup:
                raise ValueError(f"table={table}: duplicated factor columns {dup}")
            panel = panel.merge(df, on=io.KEY, how="outer")
    if panel is None:
        raise RuntimeError("no factor tables found under raw/factor")
    panel = io.sort_by_key(panel)
    io.assert_unique_key(panel, "panel_x")
    meta = pd.DataFrame(meta_rows)
    meta["meta_source"] = "inferred"
    meta = _apply_factor_dictionary(meta)
    return panel, meta, qc


def _apply_factor_dictionary(meta: pd.DataFrame) -> pd.DataFrame:
    """Override inferred category/description with raw/factor_meta.{csv,parquet} if present.

    Expected columns: factor (required), category, description; optional source_table,
    direction, unit, formula (extra columns are carried through).
    """
    d = io.read_raw_factor_meta()
    if d is None:
        return meta
    d = d.copy()
    d["factor"] = d["factor"].astype("str").str.strip()
    over = [c for c in ("category", "description") if c in d.columns]
    extra = [c for c in d.columns if c not in meta.columns and c != "factor"]
    m = meta.merge(d[["factor"] + over + extra], on="factor", how="left", suffixes=("", "_db"))
    for c in over:
        hit = m[f"{c}_db"].notna()
        m.loc[hit, c] = m.loc[hit, f"{c}_db"]
        m = m.drop(columns=f"{c}_db")
    m["meta_source"] = np.where(m["factor"].isin(d["factor"]), "db", "inferred")
    unknown = sorted(set(d["factor"]) - set(meta["factor"]))
    if unknown:
        log.warning(
            "factor dictionary has %d factors not in raw/factor: %s", len(unknown), unknown[:10]
        )
    log.info(
        "factor dictionary applied: %d/%d factors", int((m["meta_source"] == "db").sum()), len(m)
    )
    return m


# ------------------------------------------------------------------------- QC


def run_qc(
    cal: pd.DataFrame,
    ret: pd.DataFrame,
    uni: pd.DataFrame,
    panel: pd.DataFrame,
) -> list[dict]:
    qc: list[dict] = []

    # universe size and weight sum per rebalance date
    g = uni.groupby("date")
    for d, n in g.size().items():
        qc.append(_qc("universe_n_names", d, value=n))
    for d, w in g["weight"].sum().items():
        qc.append(_qc("universe_weight_sum", d, value=w))

    # universe names without a factor row / with all-NaN factors
    fcols = [c for c in panel.columns if c not in io.KEY]
    has_any = panel[fcols].notna().any(axis=1)
    px = panel.loc[has_any, io.KEY].assign(has_x=True)
    m = uni.merge(px, on=io.KEY, how="left")
    m["has_x"] = m["has_x"].fillna(False).astype(bool)
    for d, grp in m.groupby("date"):
        miss = grp.loc[~grp["has_x"], "bid"].tolist()
        qc.append(
            _qc(
                "universe_missing_factor",
                d,
                value=len(miss) / len(grp),
                note=",".join(miss[:50]),
            )
        )

    # universe names with no drtn observation in the month after the rebalance date
    rd = rebal_dates(cal)
    cal_td = cal.set_index("date")["td_index"]
    obs = ret.loc[ret["drtn"].notna(), io.KEY].copy()
    obs["td"] = obs["date"].map(cal_td)
    obs_by_bid = obs.groupby("bid")["td"].apply(np.asarray)
    next_td = rd["td_index"].shift(-1)
    for d, td0, td1 in zip(rd["date"], rd["td_index"], next_td, strict=True):
        if np.isnan(td1):
            continue
        names = uni.loc[uni["date"] == d, "bid"]
        miss = []
        for b in names:
            arr = obs_by_bid.get(b)
            if arr is None or not ((arr > td0) & (arr <= td1)).any():
                miss.append(b)
        qc.append(
            _qc(
                "universe_missing_return",
                d,
                value=len(miss) / max(len(names), 1),
                note=",".join(miss[:50]),
            )
        )

    # rebalance dates with no factor data at all
    have = set(panel["date"].unique())
    for d in rd["date"]:
        if d not in have:
            qc.append(_qc("rebal_without_factor", d))
    # per-date factor coverage among universe names (mean non-NaN fraction)
    cov = uni.merge(panel, on=io.KEY, how="left")
    cov_by_date = cov.groupby("date")[fcols].apply(lambda x: x.notna().mean().mean())
    for d, v in cov_by_date.items():
        qc.append(_qc("universe_factor_coverage", d, value=v))
    return qc


def outlier_qc(ret: pd.DataFrame, *, outlier: float) -> list[dict]:
    """Flag daily returns with |ret| > outlier (on the unmasked values)."""
    qc: list[dict] = []
    for k in [c for c in ret.columns if c not in io.KEY]:
        bad = ret.loc[ret[k].abs() > outlier, io.KEY + [k]]
        for r in bad.itertuples(index=False):
            qc.append(_qc("ret_outlier", r.date, r.bid, value=getattr(r, k), note=k))
    return qc


def monthly_return_qc(
    cal: pd.DataFrame, ret: pd.DataFrame, uni: pd.DataFrame, *, confirm_kind: str, tol: float
) -> list[dict]:
    """Universe rows whose 1M drtn differs from the 1M ``confirm_kind`` return by more than tol."""
    from india_ai_model.common.targets import make_target

    a = make_target(ret, cal, "drtn", 1)[io.KEY + ["drtn_1m"]]
    b = make_target(ret, cal, confirm_kind, 1)[io.KEY + [f"{confirm_kind}_1m"]]
    m = uni[io.KEY].merge(a, on=io.KEY, how="inner").merge(b, on=io.KEY, how="inner")
    diff = m["drtn_1m"] - m[f"{confirm_kind}_1m"]
    bad = m[diff.abs() > tol]
    log.info(
        "monthly cross-check vs %s: %d/%d universe rows differ by > %.0f%%",
        confirm_kind,
        len(bad),
        len(m),
        tol * 100,
    )
    return [
        _qc(
            "ret_monthly_mismatch",
            r.date,
            r.bid,
            value=r.drtn_1m - getattr(r, f"{confirm_kind}_1m"),
            note=f"drtn_1m={r.drtn_1m:.4f} {confirm_kind}_1m={getattr(r, f'{confirm_kind}_1m'):.4f}",
        )
        for r in bad.itertuples(index=False)
    ]


def manifest_qc() -> list[dict]:
    """Compare raw/_manifest.parquet (latest fetch per dataset/partition) with the files on disk."""
    man = io.read_raw_manifest()
    if man is None:
        log.warning("raw/_manifest.parquet not found; skipping manifest check")
        return []
    qc: list[dict] = []
    latest = man.sort_values("fetched_at").groupby(["dataset", "partition"], as_index=False).last()
    for r in latest.itertuples(index=False):
        part = io.raw_dir() / r.dataset / r.partition
        if not part.exists():
            qc.append(
                _qc("manifest_partition_missing", r.fetched_at, note=f"{r.dataset}/{r.partition}")
            )
            continue
        n = sum(pq.ParquetFile(f).metadata.num_rows for f in part.rglob("*.parquet"))
        if n != int(r.n_rows):
            qc.append(
                _qc(
                    "manifest_row_mismatch",
                    r.fetched_at,
                    value=n - int(r.n_rows),
                    note=f"{r.dataset}/{r.partition}: disk={n} manifest={int(r.n_rows)}",
                )
            )
    # partitions on disk that no fetch recorded
    known = set(zip(latest["dataset"], latest["partition"], strict=True))
    for ds in ("universe", "sector", "factor", "return"):
        root = io.raw_dir() / ds
        if not root.exists():
            continue
        for f in root.rglob("*.parquet"):
            part = str(f.parent.relative_to(root))
            if (ds, part) not in known:
                qc.append(_qc("manifest_partition_unrecorded", note=f"{ds}/{part}"))
    return qc


# ----------------------------------------------------------------------- main


def build(
    *,
    rebal_lag: int = 0,
    universe_join: str = "ym",
    min_names: int = 50,
    zero_frac_max: float = 0.99,
    outlier: float = 0.5,
    max_abs_ret: float | None = 0.4,
    confirm_kind: str | None = "gp_trtn",
    confirm_tol: float = 0.05,
    monthly_check_tol: float = 0.1,
) -> None:
    t0 = time.time()
    out = io.mart_dir()
    out.mkdir(parents=True, exist_ok=True)
    qc: list[dict] = []

    kinds = io.list_return_kinds()
    cal, ret, q = build_returns(
        kinds, min_names=min_names, zero_frac_max=zero_frac_max, rebal_lag=rebal_lag
    )
    qc += q
    # QC on the raw values first (ret_outlier), then mask unconfirmed extremes
    qc += outlier_qc(ret, outlier=outlier)
    ret, q = mask_bad_returns(
        ret,
        kinds=BARRA_KINDS,
        max_abs_ret=max_abs_ret,
        confirm_kind=confirm_kind,
        confirm_tol=confirm_tol,
    )
    qc += q
    io.write_parquet(cal, out / "calendar.parquet")
    io.write_parquet(ret, out / "ret_daily.parquet")
    log.info(
        "calendar: %d bdays %s..%s, %d rebal dates (lag=%d)",
        len(cal),
        cal["date"].iloc[0].date(),
        cal["date"].iloc[-1].date(),
        int(cal["is_rebal"].sum()),
        rebal_lag,
    )

    uni = build_universe(cal, universe_join)
    io.write_parquet(uni, out / "universe_monthly.parquet")
    log.info("universe_monthly: %d rows, %d dates", len(uni), uni["date"].nunique())

    sec = build_sector(cal, universe_join)
    if sec is not None:
        io.write_parquet(sec, out / "sector_monthly.parquet")

    panel, meta, q = build_panel_x(cal)
    qc += q
    io.write_parquet(panel, out / "panel_x.parquet")
    meta.to_csv(out / "factor_meta.csv", index=False)
    log.info(
        "panel_x: %d rows x %d factors, %d dates",
        len(panel),
        meta.shape[0],
        panel["date"].nunique(),
    )

    qc += run_qc(cal, ret, uni, panel)
    if confirm_kind and confirm_kind in ret.columns:
        qc += monthly_return_qc(cal, ret, uni, confirm_kind=confirm_kind, tol=monthly_check_tol)
    qc += manifest_qc()
    qcdf = _qc_frame(qc)
    io.write_parquet(qcdf, out / "_qc_report.parquet")
    log.info("qc report: %s", qcdf["check"].value_counts().to_dict())
    log.info("done in %.1fs", time.time() - t0)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rebal-lag", type=int, default=0, help="business days before month-end (0 = last bday)"
    )
    p.add_argument("--universe-join", choices=["ym", "asof"], default="ym")
    p.add_argument("--min-names", type=int, default=50)
    p.add_argument("--zero-frac-max", type=float, default=0.99)
    p.add_argument(
        "--outlier", type=float, default=0.5, help="|ret| threshold for QC flag (decimal)"
    )
    p.add_argument(
        "--max-abs-ret",
        type=float,
        default=0.4,
        help="mask Barra returns with |drtn| above this unless confirmed (0 = off)",
    )
    p.add_argument(
        "--confirm-kind", default="gp_trtn", help="independent return kind for confirmation"
    )
    p.add_argument("--confirm-tol", type=float, default=0.05)
    p.add_argument("--monthly-check-tol", type=float, default=0.1)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    build(
        rebal_lag=a.rebal_lag,
        universe_join=a.universe_join,
        min_names=a.min_names,
        zero_frac_max=a.zero_frac_max,
        outlier=a.outlier,
        max_abs_ret=a.max_abs_ret or None,
        confirm_kind=a.confirm_kind or None,
        confirm_tol=a.confirm_tol,
        monthly_check_tol=a.monthly_check_tol,
    )


if __name__ == "__main__":
    main()

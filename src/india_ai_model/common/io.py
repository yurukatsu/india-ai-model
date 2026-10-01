"""I/O helpers for raw (fetched by namdb) and mart (built here) parquet files.

Conventions (see handoff doc §4.1):
- keys are plain columns, ``date`` is datetime64[ns] without time, ``bid`` is str
- rows are sorted by (date, bid) and the key is unique
- missing values stay NaN (no zero-fill / ffill)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# DB sentinel for "missing". Observed values: -999999998.0 and -1e9.
SENTINEL_THRESHOLD = -9.99e8

KEY = ["date", "bid"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return Path(os.environ.get("INDIA_AI_DATA", project_root() / "data"))


def raw_dir() -> Path:
    return data_dir() / "raw"


def mart_dir() -> Path:
    return data_dir() / "mart"


# ---------------------------------------------------------------- normalisation


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``date`` to datetime64[ns] (no time) and ``bid`` to stripped str."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize().astype("datetime64[ns]")
    # raw return files carry trailing whitespace in bid (e.g. 'INDAAA1 ')
    out["bid"] = out["bid"].astype("str").str.strip()
    return out


def clean_sentinel(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Replace DB sentinel values (<= -9.99e8) with NaN in numeric columns."""
    out = df.copy()
    if cols is None:
        cols = [c for c in out.columns if pd.api.types.is_float_dtype(out[c])]
    for c in cols:
        out[c] = out[c].mask(out[c] <= SENTINEL_THRESHOLD)
    return out


def sort_by_key(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    cols = (extra or []) + KEY
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def assert_unique_key(df: pd.DataFrame, name: str, extra: list[str] | None = None) -> None:
    cols = (extra or []) + KEY
    dup = df.duplicated(cols).sum()
    if dup:
        raise ValueError(f"{name}: {dup} duplicated rows on {cols}")


# ------------------------------------------------------------------------ raw


def _drop_partition_cols(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    drop = [c for c in ("year",) if c in df.columns and c not in keep]
    return df.drop(columns=drop)


def read_raw_universe() -> pd.DataFrame:
    """MSCI India IMI monthly constituents: date (calendar month-end), bid, weight."""
    df = pd.read_parquet(raw_dir() / "universe")
    df = _drop_partition_cols(df, keep=[])
    return sort_by_key(normalize_keys(df))


def read_raw_sector() -> pd.DataFrame | None:
    p = raw_dir() / "sector"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = _drop_partition_cols(df, keep=[])
    return sort_by_key(normalize_keys(df))


def list_factor_tables() -> list[str]:
    root = raw_dir() / "factor"
    return sorted(p.name.split("=", 1)[1] for p in root.glob("table=*") if p.is_dir())


def read_raw_factor(table: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One factor table as stored in the DB (sentinels NOT yet cleaned)."""
    df = pd.read_parquet(raw_dir() / "factor", filters=[("table", "=", table)], columns=columns)
    df = df.drop(columns=[c for c in ("table", "year") if c in df.columns])
    return sort_by_key(normalize_keys(df))


def read_raw_factor_meta() -> pd.DataFrame | None:
    """Factor dictionary from the DB (raw/factor_meta.csv or .parquet), if provided."""
    for p in (raw_dir() / "factor_meta.parquet", raw_dir() / "factor_meta.csv"):
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    return None


def read_raw_manifest() -> pd.DataFrame | None:
    """Fetch log (raw/_manifest.parquet): one row per fetch, appended."""
    p = raw_dir() / "_manifest.parquet"
    return pd.read_parquet(p) if p.exists() else None


def list_return_kinds() -> list[str]:
    root = raw_dir() / "return"
    return sorted(p.name.split("=", 1)[1] for p in root.glob("kind=*") if p.is_dir())


def read_raw_return(kind: str) -> pd.DataFrame:
    """Daily returns of one kind (long, percent units, sentinels NOT yet cleaned)."""
    df = pd.read_parquet(raw_dir() / "return", filters=[("kind", "=", kind)])
    df = df.drop(columns=[c for c in ("kind", "year") if c in df.columns])
    return sort_by_key(normalize_keys(df))


# ----------------------------------------------------------------------- mart


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_mart(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read ``data/mart/<name>.parquet`` (or ``.csv`` for factor_meta)."""
    p = mart_dir() / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p, columns=columns)
    p = mart_dir() / f"{name}.csv"
    if p.exists():
        return pd.read_csv(p)
    raise FileNotFoundError(p)


def read_calendar() -> pd.DataFrame:
    return read_mart("calendar")


def read_ret_daily(kinds: list[str] | None = None) -> pd.DataFrame:
    cols = None if kinds is None else KEY + list(kinds)
    return read_mart("ret_daily", columns=cols)


def read_panel_x(columns: list[str] | None = None) -> pd.DataFrame:
    cols = None if columns is None else KEY + [c for c in columns if c not in KEY]
    return read_mart("panel_x", columns=cols)


def read_universe_monthly() -> pd.DataFrame:
    return read_mart("universe_monthly")


def read_sector_monthly() -> pd.DataFrame | None:
    p = mart_dir() / "sector_monthly.parquet"
    return pd.read_parquet(p) if p.exists() else None

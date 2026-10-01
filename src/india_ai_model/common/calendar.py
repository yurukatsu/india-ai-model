"""Indian business-day calendar and monthly rebalance dates.

The raw ``kind=drtn`` file is on a Mon-Fri calendar where Indian exchange
holidays appear as days on which (almost) every name has a return of exactly 0.
We therefore define a business day as a date that has at least ``min_names``
non-missing returns and on which fewer than ``zero_frac_max`` of them are 0.

Rebalance date of a month = the business day ``rebal_lag`` business days before
the last business day of that month (``rebal_lag=0`` -> last business day).
Only *complete* months get a rebalance date: a month is complete if a later
month exists in the data, or if its last business day is on/after the last
weekday of the month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAL_COLUMNS = ["date", "ym", "td_index", "is_rebal", "rebal_index"]


def business_days_from_returns(
    ret_long: pd.DataFrame,
    *,
    ret_col: str = "ret",
    min_names: int = 50,
    zero_frac_max: float = 0.99,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Return (business days, per-date diagnostics) from a long daily return frame.

    ``ret_long`` must already have sentinels converted to NaN.
    """
    r = ret_long[ret_col]
    g = (
        pd.DataFrame(
            {
                "date": ret_long["date"],
                "n_obs": r.notna().astype(int),
                "n_zero": (r == 0).astype(int),
            }
        )
        .groupby("date", sort=True)
        .sum()
    )
    g["zero_frac"] = g["n_zero"] / g["n_obs"].replace(0, np.nan)
    g["is_bday"] = (g["n_obs"] >= min_names) & (g["zero_frac"] < zero_frac_max)
    days = pd.DatetimeIndex(g.index[g["is_bday"]])
    return days, g.reset_index()


def build_calendar(days: pd.DatetimeIndex, *, rebal_lag: int = 0) -> pd.DataFrame:
    """Build the calendar frame from a sorted DatetimeIndex of business days."""
    if rebal_lag < 0:
        raise ValueError("rebal_lag must be >= 0")
    days = pd.DatetimeIndex(days).sort_values().unique()
    cal = pd.DataFrame({"date": days.astype("datetime64[ns]")})
    cal["ym"] = cal["date"].dt.strftime("%Y-%m")
    cal["td_index"] = np.arange(len(cal), dtype=np.int64)

    last_in_month = cal.groupby("ym")["date"].max()
    last_weekday = last_in_month + pd.offsets.BMonthEnd(0)
    has_later = pd.Series(
        np.arange(len(last_in_month)) < len(last_in_month) - 1, index=last_in_month.index
    )
    complete = has_later | (last_in_month >= last_weekday)

    pos_from_end = cal.groupby("ym").cumcount(ascending=False)
    is_rebal = (pos_from_end == rebal_lag) & cal["ym"].map(complete).astype(bool)
    cal["is_rebal"] = is_rebal.to_numpy()
    rebal_index = np.full(len(cal), np.nan)
    rebal_index[cal["is_rebal"].to_numpy()] = np.arange(int(cal["is_rebal"].sum()))
    cal["rebal_index"] = rebal_index
    return cal[CAL_COLUMNS]


def rebal_dates(cal: pd.DataFrame) -> pd.DataFrame:
    """Subset of the calendar with rebalance dates only (rebal_index as int)."""
    r = cal.loc[cal["is_rebal"], CAL_COLUMNS].copy()
    r["rebal_index"] = r["rebal_index"].astype(np.int64)
    return r.reset_index(drop=True)


def attach_rebal_date(
    monthly: pd.DataFrame,
    cal: pd.DataFrame,
    *,
    how: str = "ym",
) -> pd.DataFrame:
    """Map a monthly frame (``date`` = calendar month-end or similar) onto rebalance dates.

    how="ym"   : join on the year-month of ``date``; a row dated 2020-01-31 is assigned
                 to the rebalance date of 2020-01. Rows in months without a rebalance
                 date are dropped.
    how="asof" : each rebalance date takes the latest monthly row with date <= rebal date.
    The returned frame has ``date`` replaced by the rebalance date.
    """
    r = rebal_dates(cal)[["date", "ym"]]
    r["date"] = r["date"].astype("datetime64[ns]")
    monthly = monthly.copy()
    monthly["date"] = pd.to_datetime(monthly["date"]).astype("datetime64[ns]")
    if how == "ym":
        m = monthly
        m["ym"] = m["date"].dt.strftime("%Y-%m")
        out = m.drop(columns="date").merge(r, on="ym", how="inner").drop(columns="ym")
    elif how == "asof":
        m = monthly.sort_values("date")
        src_dates = pd.DataFrame({"date": pd.DatetimeIndex(m["date"].unique()).sort_values()})
        src_dates["src_date"] = src_dates["date"]
        # for each rebal date, latest source date <= rebal date
        mapping = pd.merge_asof(
            r[["date"]].sort_values("date"),
            src_dates,
            on="date",
            direction="backward",
        ).dropna()
        out = (
            m.rename(columns={"date": "src_date"})
            .merge(mapping, on="src_date", how="inner")
            .drop(columns="src_date")
        )
    else:
        raise ValueError(f"unknown how={how!r}")
    cols = ["date"] + [c for c in out.columns if c != "date"]
    return out[cols].sort_values(["date", "bid"], kind="mergesort").reset_index(drop=True)

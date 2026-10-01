import numpy as np
import pandas as pd
import pytest

from india_ai_model.common.calendar import rebal_dates
from india_ai_model.common.targets import (
    TargetSpec,
    make_target,
    make_target_by_name,
    mask_low_coverage,
    parse_target_name,
)


def _window(ret_daily, cal, bid, kind, i, months, skip=0):
    rd = rebal_dates(cal)["date"]
    a, b = rd.iloc[i + skip], rd.iloc[i + skip + months]
    r = ret_daily[(ret_daily["bid"] == bid) & (ret_daily["date"] > a) & (ret_daily["date"] <= b)]
    return r[kind], b


def test_parse_target_name():
    assert parse_target_name("drtn_1m") == TargetSpec("drtn", 1, 0)
    assert parse_target_name("srtn_3m_s2") == TargetSpec("srtn", 3, 2)
    assert TargetSpec("drtn", 1, 1).name == "drtn_1m_s1"
    with pytest.raises(ValueError):
        parse_target_name("foo_1m")
    with pytest.raises(ValueError):
        parse_target_name("drtn_1")


def test_compound_matches_brute_force(ret_daily, cal):
    y = make_target(ret_daily, cal, "drtn", 1)
    assert list(y.columns) == ["date", "bid", "drtn_1m", "n_obs", "n_days", "end_date"]
    rd = rebal_dates(cal)["date"]
    for i in range(5):
        r, end = _window(ret_daily, cal, "A", "drtn", i, 1)
        exp = np.prod(1 + r.to_numpy()) - 1
        row = y[(y["date"] == rd.iloc[i]) & (y["bid"] == "A")].iloc[0]
        assert row["drtn_1m"] == pytest.approx(exp)
        assert row["end_date"] == end
        assert row["n_obs"] == len(r)
        assert row["n_days"] == len(r)
    # last rebal date (June) has no complete 1m window -> absent
    assert rd.iloc[-1] not in set(y["date"])


def test_simple_sum_and_multi_month(ret_daily, cal):
    y = make_target(ret_daily, cal, "srtn", 3)
    rd = rebal_dates(cal)["date"]
    r, end = _window(ret_daily, cal, "A", "srtn", 0, 3)
    row = y[(y["date"] == rd.iloc[0]) & (y["bid"] == "A")].iloc[0]
    assert row["srtn_3m"] == pytest.approx(r.sum())
    assert row["end_date"] == end == rd.iloc[3]
    assert set(y["date"]) == set(rd.iloc[:3])  # only i=0,1,2 have i+3 <= 5


def test_skip(ret_daily, cal):
    y = make_target_by_name(ret_daily, cal, "drtn_1m_s1")
    rd = rebal_dates(cal)["date"]
    r, end = _window(ret_daily, cal, "A", "drtn", 0, 1, skip=1)
    row = y[(y["date"] == rd.iloc[0]) & (y["bid"] == "A")].iloc[0]
    assert row["drtn_1m_s1"] == pytest.approx(np.prod(1 + r.to_numpy()) - 1)
    assert row["end_date"] == rd.iloc[2]
    # n_days is the skipped-forward month, not two months
    assert row["n_days"] == len(r)


def test_missing_days_and_absent_bids(ret_daily, cal):
    y = make_target(ret_daily, cal, "drtn", 1)
    rd = rebal_dates(cal)["date"]
    # B misses 2020-02-10 inside the Feb window (Jan31, Feb28]
    row = y[(y["date"] == rd.iloc[0]) & (y["bid"] == "B")].iloc[0]
    assert row["n_days"] - row["n_obs"] == 1
    r, _ = _window(ret_daily, cal, "B", "drtn", 0, 1)
    assert row["drtn_1m"] == pytest.approx(np.prod(1 + r.dropna().to_numpy()) - 1)
    # C has no data before March -> no row for Jan (window Feb) ...
    assert y[(y["date"] == rd.iloc[0]) & (y["bid"] == "C")].empty
    # ... but has a row for Feb (window Mar)
    assert not y[(y["date"] == rd.iloc[1]) & (y["bid"] == "C")].empty


def test_leak_mask_by_end_date(ret_daily, cal):
    y = make_target(ret_daily, cal, "drtn", 3)
    T = pd.Timestamp("2020-04-30")
    usable = y[y["end_date"] <= T]
    # only i=0 has end_date = rebal[3] = 2020-04-30
    assert usable["date"].unique().tolist() == [pd.Timestamp("2020-01-31")]


def test_mask_low_coverage(ret_daily, cal):
    y = make_target(ret_daily, cal, "drtn", 1)
    m = mask_low_coverage(y, "drtn_1m", min_obs_frac=1.0)
    assert m["drtn_1m"].isna().sum() == 1  # only B in Feb misses a day

import numpy as np
import pandas as pd

from india_ai_model.common.calendar import (
    attach_rebal_date,
    build_calendar,
    business_days_from_returns,
    rebal_dates,
)


def test_business_days_drop_holidays_and_thin_days():
    dates = pd.bdate_range("2020-01-01", "2020-01-10")
    rows = []
    for d in dates:
        for i in range(100):
            r = 0.0 if d == pd.Timestamp("2020-01-06") else 0.01 * (i % 5 - 2)
            if d == pd.Timestamp("2020-01-08") and i >= 10:
                r = np.nan  # only 10 obs -> thin day
            rows.append((d, f"b{i}", r))
    df = pd.DataFrame(rows, columns=["date", "bid", "ret"])
    days, diag = business_days_from_returns(df, min_names=50, zero_frac_max=0.99)
    assert pd.Timestamp("2020-01-06") not in days
    assert pd.Timestamp("2020-01-08") not in days
    assert len(days) == len(dates) - 2


def test_calendar_rebal_last_bday_and_complete_month(cal, bdays):
    rd = rebal_dates(cal)
    # Jan..Jun 2020, each month's last business day (Mar 31 dropped -> Mar 30)
    exp = ["2020-01-31", "2020-02-28", "2020-03-30", "2020-04-30", "2020-05-29", "2020-06-30"]
    assert rd["date"].dt.strftime("%Y-%m-%d").tolist() == exp
    assert rd["rebal_index"].tolist() == list(range(6))
    assert cal["td_index"].tolist() == list(range(len(bdays)))
    assert cal["rebal_index"].isna().sum() == len(bdays) - 6


def test_calendar_incomplete_last_month(bdays):
    days = bdays[bdays < "2020-06-20"]
    cal = build_calendar(days)
    assert rebal_dates(cal)["ym"].tolist() == [
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        "2020-05",
    ]


def test_calendar_lag(bdays):
    cal = build_calendar(bdays, rebal_lag=1)
    rd = rebal_dates(cal)
    assert rd["date"].iloc[0] == pd.Timestamp("2020-01-30")
    assert rd["date"].iloc[2] == pd.Timestamp("2020-03-27")  # Mar 31 is a holiday, 30 is last


def test_attach_rebal_date_ym_and_asof(cal):
    monthly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-02-29", "2020-07-31"]),
            "bid": ["A", "A", "B", "A"],
            "weight": [1.0, 0.5, 0.5, 1.0],
        }
    )
    ym = attach_rebal_date(monthly, cal, how="ym")
    assert ym["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-31", "2020-02-28", "2020-02-28"]
    assert list(ym.columns) == ["date", "bid", "weight"]

    asof = attach_rebal_date(monthly, cal, how="asof")
    # Feb 29 (Sat) > Feb 28 rebal -> Feb 28 uses Jan 31 data; Mar 30 uses Feb 29 data, ...
    got = asof.groupby("date")["bid"].apply(list).to_dict()
    assert got[pd.Timestamp("2020-02-28")] == ["A"]
    assert got[pd.Timestamp("2020-03-30")] == ["A", "B"]
    assert got[pd.Timestamp("2020-06-30")] == ["A", "B"]

import numpy as np
import pandas as pd
import pytest

from india_ai_model.common.calendar import build_calendar


@pytest.fixture
def bdays() -> pd.DatetimeIndex:
    # Mon-Fri 2020-01-01 .. 2020-06-30, drop two "holidays"
    d = pd.bdate_range("2020-01-01", "2020-06-30")
    return d.drop([pd.Timestamp("2020-01-27"), pd.Timestamp("2020-03-31")])


@pytest.fixture
def cal(bdays) -> pd.DataFrame:
    return build_calendar(bdays, rebal_lag=0)


@pytest.fixture
def ret_daily(bdays) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    bids = ["A", "B", "C"]
    rows = []
    for d in bdays:
        for b in bids:
            rows.append((d, b, rng.normal(0, 0.02), rng.normal(0, 0.01)))
    df = pd.DataFrame(rows, columns=["date", "bid", "drtn", "srtn"])
    df["date"] = df["date"].astype("datetime64[ns]")
    # B has a missing day inside February, C has no data before March
    df.loc[(df["bid"] == "B") & (df["date"] == "2020-02-10"), ["drtn", "srtn"]] = np.nan
    df = df[~((df["bid"] == "C") & (df["date"] < "2020-03-01"))]
    return df.reset_index(drop=True)

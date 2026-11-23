"""Shared paths, split definition, metrics and calendar helpers.

Every script imports the split from here so that all models are trained,
tuned and tested on exactly the same days.
"""
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from dateutil.easter import easter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "time_series_60min_singleindex.csv"
DAILY = ROOT / "data" / "processed" / "de_daily_load.csv"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
PRED = ROOT / "results" / "predictions"
for d in (FIG, TAB, PRED):
    d.mkdir(parents=True, exist_ok=True)

# Chronological split (no shuffling). Validation and test are each one full
# year so that every season, and one Christmas period, is seen in each.
TRAIN_END = "2018-09-30"
VAL_END = "2019-09-30"
TEST_END = "2020-09-30"

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_daily():
    df = pd.read_csv(DAILY, parse_dates=["date"], index_col="date")
    df.index.freq = "D"
    return df


def split_index(idx):
    """Boolean masks for train / validation / test on a DatetimeIndex."""
    tr = idx <= TRAIN_END
    va = (idx > TRAIN_END) & (idx <= VAL_END)
    te = (idx > VAL_END) & (idx <= TEST_END)
    return tr, va, te


def metrics(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    err = y - yhat
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / y)) * 100)
    r2 = float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def german_holidays(years):
    """Nation-wide German public holidays (regional ones are ignored)."""
    days = []
    for y in years:
        e = pd.Timestamp(easter(y))
        days += [
            pd.Timestamp(y, 1, 1),            # New Year
            e - pd.Timedelta(days=2),         # Good Friday
            e + pd.Timedelta(days=1),         # Easter Monday
            pd.Timestamp(y, 5, 1),            # Labour Day
            e + pd.Timedelta(days=39),        # Ascension
            e + pd.Timedelta(days=50),        # Whit Monday
            pd.Timestamp(y, 10, 3),           # German Unity Day
            pd.Timestamp(y, 12, 25),
            pd.Timestamp(y, 12, 26),
        ]
        if y == 2017:
            days.append(pd.Timestamp(2017, 10, 31))  # Reformation 500 yrs
    return pd.DatetimeIndex(sorted(set(days)))


def calendar_features(idx):
    """Day-of-week one-hot, public-holiday flag and Christmas-break flag."""
    hol = german_holidays(range(idx.year.min(), idx.year.max() + 1))
    f = pd.DataFrame(index=idx)
    for d in range(7):
        f[f"dow_{d}"] = (idx.dayofweek == d).astype(float)
    f["holiday"] = idx.isin(hol).astype(float)
    f["xmas"] = (((idx.month == 12) & (idx.day >= 24)) |
                 ((idx.month == 1) & (idx.day <= 1))).astype(float)
    return f


def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)

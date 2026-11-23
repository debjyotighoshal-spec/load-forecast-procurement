"""Download (if needed) the OPSD hourly file and aggregate it to daily values.

Output: data/processed/de_daily_load.csv with columns
  load_gw      daily mean of actual hourly German load (GW)
  tso_fc_gw    daily mean of the TSOs' own day-ahead load forecast (GW)
  price_eur    daily mean day-ahead price DE-LU (EUR/MWh), from Oct 2018
Days are local market days (CET/CEST). Using the daily *mean* rather than the
sum avoids the 23/25-hour days at daylight-saving changes.
"""
import urllib.request

import pandas as pd

from common import RAW, DAILY

URL = ("https://data.open-power-system-data.org/time_series/2020-10-06/"
       "time_series_60min_singleindex.csv")

COLS = {
    "DE_load_actual_entsoe_transparency": "load_gw",
    "DE_load_forecast_entsoe_transparency": "tso_fc_gw",
    "DE_LU_price_day_ahead": "price_eur",
}


def main():
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", URL)
        urllib.request.urlretrieve(URL, RAW)

    df = pd.read_csv(RAW, usecols=["cet_cest_timestamp", *COLS])
    df = df.rename(columns=COLS)
    # Timestamps carry their own +01:00/+02:00 offset; the local date is the
    # first 10 characters.
    df["date"] = pd.to_datetime(df["cet_cest_timestamp"].str[:10])
    daily = df.groupby("date").agg(
        load_gw=("load_gw", "mean"),
        tso_fc_gw=("tso_fc_gw", "mean"),
        price_eur=("price_eur", "mean"),
        hours=("load_gw", "count"),
    )
    daily[["load_gw", "tso_fc_gw"]] /= 1000.0  # MW -> GW
    daily = daily.loc["2015-01-01":"2020-09-30"]
    daily = daily.asfreq("D")
    print("days:", len(daily), "| days with <23 load hours:",
          int((daily["hours"] < 23).sum()))
    print("missing load days:", int(daily["load_gw"].isna().sum()))
    daily["load_gw"] = daily["load_gw"].interpolate()
    daily = daily.drop(columns="hours").round(4)
    DAILY.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY, index_label="date")
    print(daily.describe().round(2))


if __name__ == "__main__":
    main()

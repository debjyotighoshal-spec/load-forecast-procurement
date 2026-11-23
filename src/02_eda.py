"""Task 1, 2, 6, 7: exploratory analysis, decomposition, ACF/PACF and
unit-root tests. All statistics are computed on the training period only so
that nothing about the validation or test years leaks into model choices."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

from common import FIG, TAB, load_daily, split_index, save_json, TRAIN_END, VAL_END

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
C_TR, C_VA, C_TE = "#2a6f97", "#e09f3e", "#9e2a2b"


def main():
    df = load_daily()
    y = df["load_gw"]
    tr, va, te = split_index(y.index)
    ytr = y[tr]
    out = {"n_days": len(y), "n_train": int(tr.sum()), "n_val": int(va.sum()),
           "n_test": int(te.sum())}

    # --- Fig 1: full series with split ----------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(y[tr], color=C_TR, lw=0.7, label=f"train ({tr.sum()} d)")
    ax.plot(y[va], color=C_VA, lw=0.7, label=f"validation ({va.sum()} d)")
    ax.plot(y[te], color=C_TE, lw=0.7, label=f"test ({te.sum()} d)")
    ax.plot(y.rolling(28, center=True).mean(), color="k", lw=1.2,
            label="28-day mean")
    ax.axvspan(pd.Timestamp("2020-03-16"), pd.Timestamp("2020-05-10"),
               color="grey", alpha=0.15)
    ax.text(pd.Timestamp("2020-03-18"), 40, "COVID-19\nlockdown", fontsize=7)
    ax.set_ylabel("daily mean load (GW)")
    ax.set_title("German electricity demand, daily mean, 2015-01 to 2020-09")
    ax.legend(ncol=4, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "01_series_split.png")
    plt.close(fig)

    # --- Fig 2: weekday and month profile -------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
    dow = ytr.groupby(ytr.index.dayofweek).mean()
    axs[0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow,
               color=C_TR)
    axs[0].set_ylim(40, 62)
    axs[0].set_title("mean load by weekday (train)")
    axs[0].set_ylabel("GW")
    mon = ytr.groupby(ytr.index.month).mean()
    axs[1].bar(range(1, 13), mon, color=C_TR)
    axs[1].set_ylim(40, 62)
    axs[1].set_xticks(range(1, 13))
    axs[1].set_title("mean load by month (train)")
    fig.tight_layout()
    fig.savefig(FIG / "02_profiles.png")
    plt.close(fig)
    out["weekday_mean_gw"] = dow.round(2).tolist()
    out["month_mean_gw"] = mon.round(2).tolist()

    # --- Fig 3: MSTL decomposition (weekly + yearly) --------------------
    res = MSTL(ytr, periods=(7, 365), stl_kwargs={"robust": True}).fit()
    fig, axs = plt.subplots(5, 1, figsize=(9, 8), sharex=True)
    parts = [("observed", res.observed), ("trend", res.trend),
             ("weekly seasonal (s=7)", res.seasonal["seasonal_7"]),
             ("yearly seasonal (s=365)", res.seasonal["seasonal_365"]),
             ("remainder", res.resid)]
    for ax, (name, s) in zip(axs, parts):
        ax.plot(s, lw=0.6, color=C_TR if name != "remainder" else "grey")
        ax.set_ylabel(name, fontsize=8)
    axs[0].set_title("MSTL decomposition of training data (GW)")
    fig.tight_layout()
    fig.savefig(FIG / "03_decomposition.png")
    plt.close(fig)
    var_obs = res.observed.var()
    out["decomp_var_share"] = {
        "trend": float(res.trend.var() / var_obs),
        "weekly": float(res.seasonal["seasonal_7"].var() / var_obs),
        "yearly": float(res.seasonal["seasonal_365"].var() / var_obs),
        "remainder": float(res.resid.var() / var_obs),
    }
    out["weekly_amplitude_gw"] = float(res.seasonal["seasonal_7"].max() -
                                       res.seasonal["seasonal_7"].min())
    out["yearly_amplitude_gw"] = float(res.seasonal["seasonal_365"].max() -
                                       res.seasonal["seasonal_365"].min())
    out["trend_start_end_gw"] = [float(res.trend.iloc[0]),
                                 float(res.trend.iloc[-1])]
    lb = acorr_ljungbox(res.resid, lags=[7, 14], return_df=True)
    out["remainder_ljungbox_p"] = lb["lb_pvalue"].tolist()
    out["remainder_acf1"] = float(acf(res.resid, nlags=1)[1])
    big = res.resid.abs().sort_values(ascending=False).head(12)
    out["largest_remainder_days"] = {str(k.date()): round(float(v), 2)
                                     for k, v in big.items()}

    # --- Fig 4: ACF / PACF of levels and differenced series -------------
    d1 = ytr.diff().dropna()
    d7 = ytr.diff(7).dropna()
    d17 = ytr.diff(7).diff().dropna()
    fig, axs = plt.subplots(3, 2, figsize=(9, 7))
    for row, (name, s) in enumerate([("level", ytr), ("weekly diff (1-B^7)", d7),
                                     ("(1-B)(1-B^7)", d17)]):
        plot_acf(s, lags=60, ax=axs[row, 0], markersize=2,
                 title=f"ACF: {name}")
        plot_pacf(s, lags=60, ax=axs[row, 1], markersize=2, method="ywm",
                  title=f"PACF: {name}")
        for ax in axs[row]:
            ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(FIG / "04_acf_pacf.png")
    plt.close(fig)
    lags = [1, 2, 7, 14, 28, 60]
    out["acf_level"] = dict(zip(lags, acf(ytr, nlags=60)[lags].round(3).tolist()))
    out["pacf_level_first10"] = pacf(ytr, nlags=15, method="ywm")[1:16].round(3).tolist()
    out["acf_d17_first10"] = acf(d17, nlags=15)[1:16].round(3).tolist()
    out["pacf_d17_first10"] = pacf(d17, nlags=15, method="ywm")[1:16].round(3).tolist()
    out["acf_d7_first10"] = acf(d7, nlags=15)[1:16].round(3).tolist()
    out["acf_level_365"] = float(acf(ytr, nlags=366)[365])

    # --- Unit-root tests (task 6/7) -------------------------------------
    rows = []
    for name, s in [("level", ytr), ("first diff (1-B)", d1),
                    ("weekly diff (1-B^7)", d7), ("(1-B)(1-B^7)", d17)]:
        a = adfuller(s, autolag="AIC")
        k = kpss(s, regression="c", nlags="auto")
        rows.append({"series": name, "ADF stat": a[0], "ADF p": a[1],
                     "ADF lags": a[2], "KPSS stat": k[0], "KPSS p": k[1]})
    ur = pd.DataFrame(rows)
    ur.to_csv(TAB / "unit_root_tests.csv", index=False)
    print(ur.round(4).to_string())

    # Fig 5: differenced series
    fig, axs = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    axs[0].plot(d7, lw=0.5, color=C_TR)
    axs[0].set_title("weekly difference  y_t - y_{t-7}  (GW)")
    axs[1].plot(d17, lw=0.5, color=C_TR)
    axs[1].set_title("weekly + first difference  (1-B)(1-B^7) y_t  (GW)")
    fig.tight_layout()
    fig.savefig(FIG / "05_differenced.png")
    plt.close(fig)

    # Benchmark: price and TSO forecast quality (val + test only)
    p = df["price_eur"]
    out["price_mean_val"] = float(p[va].mean())
    out["price_mean_test"] = float(p[te].mean())
    out["price_corr_load_test"] = float(p[te].corr(y[te]))
    save_json(out, TAB / "eda_summary.json")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()

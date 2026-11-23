"""Tasks 4, 5, 8, 9: MA(q), AR(p), ARIMA, SARIMA and SARIMAX, plus naive
benchmarks and the grid operators' own day-ahead forecast.

Protocol: parameters are estimated once on the training period. The fitted
model is then run through the whole series with those parameters frozen
(statsmodels `apply(refit=False)`), which yields genuine one-step-ahead
forecasts: the forecast for day t uses observations up to day t-1 only.
Orders are chosen on the training AIC and checked on validation RMSE; the test
year is touched once, at the end.
"""
import itertools
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (FIG, TAB, PRED, load_daily, split_index, metrics,
                    calendar_features, save_json)

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def one_step(model_fn, y, tr, exog=None):
    """Fit on train, then filter the full series with frozen parameters."""
    m = model_fn(y[tr], None if exog is None else exog[tr])
    res = m.fit()
    full = res.apply(y, exog=exog, refit=False)
    pred = full.fittedvalues.copy()
    return res, pred


def arima_fn(order, seasonal=(0, 0, 0, 0), trend="c"):
    def f(y, exog):
        return SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal,
                       trend=trend, enforce_stationarity=True,
                       enforce_invertibility=True)
    return f


def main():
    df = load_daily()
    y = df["load_gw"]
    tr, va, te = split_index(y.index)
    cal = calendar_features(y.index)[["holiday", "xmas"]]
    preds = pd.DataFrame({"actual": y})
    preds["Naive (y[t-1])"] = y.shift(1)
    preds["Seasonal naive (y[t-7])"] = y.shift(7)
    preds["TSO day-ahead forecast"] = df["tso_fc_gw"]
    search = {}
    chosen = {}

    def evaluate(name, res, pred, info):
        preds[name] = pred
        info.update({"aic": float(res.aic),
                     "val_RMSE": metrics(y[va], pred[va])["RMSE"]})
        return info

    # ---- Task 4: MA(q) on the level series --------------------------------
    rows = []
    for q in [1, 2, 3, 5, 7, 10, 14]:
        t0 = time.time()
        res, p = one_step(lambda s, e: ARIMA(s, order=(0, 0, q), trend="c"), y, tr)
        rows.append({"q": q, "aic": res.aic,
                     "val_RMSE": metrics(y[va], p[va])["RMSE"],
                     "sec": time.time() - t0})
        preds[f"_MA{q}"] = p
    ma = pd.DataFrame(rows)
    print(ma.round(3))
    search["MA"] = ma.to_dict("records")
    q_best = int(ma.loc[ma["val_RMSE"].idxmin(), "q"])
    preds[f"MA({q_best})"] = preds[f"_MA{q_best}"]
    chosen["MA"] = {"q": q_best}

    # ---- Task 5: AR(p) on the level series --------------------------------
    rows = []
    for p_ in [1, 2, 7, 8, 14, 15, 21]:
        res, p = one_step(lambda s, e: ARIMA(s, order=(p_, 0, 0), trend="c"), y, tr)
        rows.append({"p": p_, "aic": res.aic,
                     "val_RMSE": metrics(y[va], p[va])["RMSE"]})
        preds[f"_AR{p_}"] = p
        if p_ == 8:
            ar8_params = res.params
    ar = pd.DataFrame(rows)
    print(ar.round(3))
    search["AR"] = ar.to_dict("records")
    p_best = int(ar.loc[ar["val_RMSE"].idxmin(), "p"])
    preds[f"AR({p_best})"] = preds[f"_AR{p_best}"]
    chosen["AR"] = {"p": p_best}
    chosen["AR8_params"] = {k: float(v) for k, v in ar8_params.items()}

    # ---- Task 8: ARIMA(p,1,q) ---------------------------------------------
    rows = []
    for p_, q in itertools.product([1, 2, 7, 8], [0, 1, 2]):
        res, p = one_step(arima_fn((p_, 1, q), trend="n"), y, tr)
        rows.append({"p": p_, "d": 1, "q": q, "aic": res.aic,
                     "val_RMSE": metrics(y[va], p[va])["RMSE"]})
        preds[f"_ARIMA{p_}1{q}"] = p
    arima = pd.DataFrame(rows).sort_values("aic")
    print(arima.round(3))
    search["ARIMA"] = arima.to_dict("records")
    b = arima.iloc[0]
    name = f"ARIMA({int(b.p)},1,{int(b.q)})"
    preds[name] = preds[f"_ARIMA{int(b.p)}1{int(b.q)}"]
    chosen["ARIMA"] = {"order": [int(b.p), 1, int(b.q)]}

    # ---- Task 9: SARIMA(p,d,q)(P,1,Q)_7 -----------------------------------
    rows = []
    for d, p_, q, P, Q in itertools.product([0, 1], [0, 1, 2], [0, 1, 2],
                                            [0, 1], [1]):
        try:
            res, p = one_step(arima_fn((p_, d, q), (P, 1, Q, 7), trend="n"), y, tr)
        except Exception as ex:  # pragma: no cover
            print("fail", d, p_, q, P, Q, ex)
            continue
        rows.append({"p": p_, "d": d, "q": q, "P": P, "D": 1, "Q": Q,
                     "aic": res.aic, "val_RMSE": metrics(y[va], p[va])["RMSE"]})
        preds[f"_S{p_}{d}{q}{P}{Q}"] = p
    sar = pd.DataFrame(rows).sort_values("aic")
    print(sar.head(10).round(3))
    search["SARIMA"] = sar.to_dict("records")
    b = sar.iloc[0]
    so = (int(b.p), int(b.d), int(b.q))
    ss = (int(b.P), 1, int(b.Q), 7)
    sname = f"SARIMA{so}{ss[:3]}7".replace(" ", "")
    preds[sname] = preds[f"_S{so[0]}{so[1]}{so[2]}{ss[0]}{ss[2]}"]
    chosen["SARIMA"] = {"order": list(so), "seasonal_order": list(ss)}

    # ---- Extension: SARIMAX with known calendar effects --------------------
    res_x, px = one_step(arima_fn(so, ss, trend="n"), y, tr, exog=cal)
    xname = sname.replace("SARIMA", "SARIMAX") + "+holidays"
    preds[xname] = px
    chosen["SARIMAX"] = {"order": list(so), "seasonal_order": list(ss),
                         "exog": list(cal.columns),
                         "params": {k: float(v) for k, v in res_x.params.items()},
                         "pvalues": {k: float(v) for k, v in res_x.pvalues.items()}}
    lb = res_x.test_serial_correlation("ljungbox", lags=[14])
    chosen["SARIMAX"]["ljungbox_p_lag14"] = float(lb[0, 1, -1])

    # Drop the search columns and store predictions ------------------------
    keep = [c for c in preds.columns if not c.startswith("_")]
    preds = preds[keep]
    preds.to_csv(PRED / "classical_predictions.csv", index_label="date")

    rows = []
    for c in keep[1:]:
        for split, m in [("val", va), ("test", te)]:
            mm = metrics(y[m], preds.loc[m, c])
            rows.append({"model": c, "split": split, **mm})
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "classical_metrics.csv", index=False)
    print(tab[tab.split == "test"].round(3).to_string())
    save_json({"search": search, "chosen": chosen},
              TAB / "classical_search.json")

    # ---- plots ------------------------------------------------------------
    models = keep[1:]
    yt = y[te]
    fig, axs = plt.subplots(len(models), 1, figsize=(9, 1.7 * len(models)),
                            sharex=True)
    for ax, c in zip(axs, models):
        m = metrics(yt, preds.loc[te, c])
        ax.plot(yt, color="k", lw=0.8, label="actual")
        ax.plot(preds.loc[te, c], color="#c0392b", lw=0.8, alpha=0.85,
                label="one-step forecast")
        ax.set_title(f"{c}   RMSE {m['RMSE']:.2f} GW | MAPE {m['MAPE']:.2f}% "
                     f"| R² {m['R2']:.3f}", fontsize=8, loc="left")
        ax.set_ylim(35, 72)
    axs[0].legend(ncol=2, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "06_classical_test_forecasts.png")
    plt.close(fig)

    ncol = 4
    nrow = int(np.ceil(len(models) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(10, 2.6 * nrow))
    for ax, c in zip(axs.flat, models):
        ax.scatter(yt, preds.loc[te, c], s=4, alpha=0.5, color="#2a6f97")
        ax.plot([38, 70], [38, 70], color="k", lw=0.7)
        ax.set_title(c, fontsize=7)
        ax.set_xlim(38, 70)
        ax.set_ylim(38, 70)
        ax.set_xlabel("actual (GW)", fontsize=7)
    for ax in list(axs.flat)[len(models):]:
        ax.axis("off")
    axs.flat[0].set_ylabel("predicted (GW)", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "07_classical_scatter.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Robustness checks for the claims made in the report.

1. Timing. The day-ahead auction closes at noon on day t-1, before day t-1's
   load is fully known. Our one-step setup uses all of day t-1, which favours
   our models over the TSO forecast. Here SARIMAX is re-scored two steps
   ahead (information up to day t-2 only), propagating the Kalman-filter
   state one extra step.
2. Bias. The TSO forecast is biased in some years; we check whether removing
   its bias changes the ranking.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (TAB, PRED, load_daily, split_index, metrics,
                    calendar_features, save_json)
from importlib import import_module

decision = import_module("06_decision")
warnings.filterwarnings("ignore")


def main():
    df = load_daily()
    y = df["load_gw"]
    tr, va, te = split_index(y.index)
    cal = calendar_features(y.index)[["holiday", "xmas"]]
    m = SARIMAX(y[tr], exog=cal[tr], order=(2, 0, 2),
                seasonal_order=(0, 1, 1, 7), trend="n").fit(disp=0)
    full = m.apply(y, exog=cal, refit=False)
    fr = full.filter_results
    T = fr.transition[:, :, 0]
    Z = fr.design[:, :, 0]
    d = np.asarray(fr.obs_intercept).reshape(-1)
    a_pred = fr.predicted_state            # a_{t|t-1}, shape (k, n+1)
    n = len(y)
    two = np.full(n, np.nan)
    for t in range(2, n):
        a2 = T @ a_pred[:, t - 1]          # a_{t|t-2}
        two[t] = (Z @ a2)[0] + d[t]
    one = full.fittedvalues.values
    # sanity check: the same algebra one step ahead reproduces fittedvalues
    chk = np.array([(Z @ a_pred[:, t])[0] + d[t] for t in range(n)])
    assert np.allclose(chk[te], one[te], atol=1e-6)
    out = {}
    for split, mk in [("val", va), ("test", te)]:
        out[f"SARIMAX_1step_{split}"] = metrics(y[mk], one[mk])
        out[f"SARIMAX_2step_{split}"] = metrics(y[mk], two[mk])
        e = (y - df["tso_fc_gw"])
        out[f"TSO_{split}"] = metrics(y[mk], df["tso_fc_gw"][mk])
        out[f"TSO_bias_{split}_GW"] = float(e[mk].mean())
        out[f"TSO_debiased_in_sample_RMSE_{split}"] = float(
            np.sqrt(((e[mk] - e[mk].mean()) ** 2).mean()))
    out["TSO_bias_train_GW"] = float((y - df["tso_fc_gw"])[tr].mean())

    # cost of the two-step SARIMAX with its own validation buffer
    tau = decision.ALPHA / (decision.ALPHA + decision.BETA)
    shift = float(np.quantile((y.values - two)[va], tau))
    P = df["price_eur"].values
    c2 = decision.daily_cost(y.values[te], two[te] + shift, P[te])
    c1 = decision.daily_cost(y.values[te], df["tso_fc_gw"].values[te], P[te])
    out["SARIMAX_2step_buffer_GW"] = shift
    out["SARIMAX_2step_buffer_test_cost_1pct_kEUR"] = float(c2.sum() * decision.SHARE / 1e3)
    out["TSO_test_cost_1pct_kEUR"] = float(c1.sum() * decision.SHARE / 1e3)

    # fair comparison: give the TSO forecast its own validation buffer, and
    # try the realistic production setup -- average the public TSO forecast
    # with our two-step SARIMAX.
    tso = df["tso_fc_gw"].values
    blend = (tso + two) / 2
    for name, f in [("TSO+buffer", tso), ("Blend_TSO_SARIMAX2", blend)]:
        sh = float(np.quantile((y.values - f)[va], tau))
        out[f"{name}_buffer_GW"] = sh
        out[f"{name}_test_cost_1pct_kEUR"] = float(
            decision.daily_cost(y.values[te], f[te] + sh, P[te]).sum() *
            decision.SHARE / 1e3)
    out["Blend_TSO_SARIMAX2_val"] = metrics(y[va], blend[va])
    out["Blend_TSO_SARIMAX2_test"] = metrics(y[te], blend[te])
    out["Blend_nobuffer_test_cost_1pct_kEUR"] = float(
        decision.daily_cost(y.values[te], blend[te], P[te]).sum() * decision.SHARE / 1e3)
    pd.DataFrame({"actual": y, "SARIMAX_2step": two}).to_csv(
        PRED / "sarimax_two_step.csv", index_label="date")
    save_json(out, TAB / "robustness.json")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()

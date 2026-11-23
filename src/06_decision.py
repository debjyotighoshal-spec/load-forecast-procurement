"""Tasks 12 and 14: turn forecasts into a procurement decision and price it.

Decision. An electricity supplier must buy tomorrow's energy in the
day-ahead auction. It buys F_t GWh at the day-ahead price P_t. If actual
demand A_t is higher, the gap is settled at the imbalance price, which we
model as P_t * (1 + ALPHA); a surplus is sold back at P_t * (1 - BETA).
Relative to perfect foresight the avoidable cost on day t is

    cost_t = 1000 * 24 * P_t+ * (ALPHA * max(A_t - F_t, 0) + BETA * max(F_t - A_t, 0))

with loads in GW (x24 -> GWh, x1000 -> MWh) and P_t+ = max(P_t, 0). This is a
newsvendor problem; the cost-minimising order is the tau* = ALPHA/(ALPHA+BETA)
quantile of tomorrow's demand, not the mean.

ALPHA and BETA are assumptions, not data: German imbalance prices (reBAP) are
not in the OPSD file. A sensitivity table over other values is produced.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, TAB, PRED, load_daily, split_index, metrics, save_json

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

ALPHA, BETA = 0.5, 0.25
SHARE = 0.01  # a supplier serving 1 % of German demand (~0.55 GW average)


def daily_cost(a, f, p, alpha=ALPHA, beta=BETA):
    p = np.clip(p, 0, None)
    return 24_000 * p * (alpha * np.clip(a - f, 0, None) +
                         beta * np.clip(f - a, 0, None))


def main():
    df = load_daily()
    tr, va, te = split_index(df.index)
    cl = pd.read_csv(PRED / "classical_predictions.csv", index_col=0,
                     parse_dates=True)
    nn_ = pd.read_csv(PRED / "neural_predictions.csv", index_col=0,
                      parse_dates=True)
    qt = pd.read_csv(PRED / "quantile_predictions.csv", index_col=0,
                     parse_dates=True)
    sx = [c for c in cl.columns if c.startswith("SARIMAX")][0]
    sa = [c for c in cl.columns if c.startswith("SARIMA(")][0]
    y = df["load_gw"]
    price = df["price_eur"]
    tau_star = ALPHA / (ALPHA + BETA)
    ENS = "Ensemble (SARIMAX + LSTM-cal)"
    ens = (cl[sx] + nn_["LSTM + calendar"]) / 2

    # ---- combined accuracy table --------------------------------------
    allp = pd.concat([cl.drop(columns="actual"),
                      nn_.drop(columns="actual")], axis=1)
    allp[ENS] = ens
    rows = []
    for c in allp.columns:
        for split, m in [("val", va), ("test", te)]:
            rows.append({"model": c, "split": split,
                         **metrics(y[m], allp.loc[m, c])})
    acc = pd.DataFrame(rows)
    acc.to_csv(TAB / "all_models_metrics.csv", index=False)
    acct = acc[acc.split == "test"].set_index("model")
    print(acct.round(3))

    # ---- procurement strategies ----------------------------------------
    # Buffers are the tau* quantile of each model's *validation* errors.
    shift = float(np.quantile((y - cl[sx])[va], tau_star))
    shift_e = float(np.quantile((y - ens)[va], tau_star))
    BUF_SX = f"SARIMAX + {shift:+.2f} GW buffer"
    BUF_EN = f"Ensemble + {shift_e:+.2f} GW buffer"

    def strategies(m):
        s = pd.DataFrame(index=df.index[m])
        s["Seasonal naive (last week)"] = cl.loc[m, "Seasonal naive (y[t-7])"]
        s["TSO public forecast"] = cl.loc[m, "TSO day-ahead forecast"]
        s[sa] = cl.loc[m, sa]
        s["LSTM + calendar"] = nn_.loc[m, "LSTM + calendar"]
        s[sx] = cl.loc[m, sx]
        s[ENS] = ens[m]
        s[f"Quantile LSTM (tau={tau_star:.2f})"] = qt.loc[m, "q_star"]
        s[BUF_SX] = cl.loc[m, sx] + shift
        s[BUF_EN] = ens[m] + shift_e
        return s

    def cost_table(m):
        s = strategies(m)
        a, p = y[m].values, price[m].values
        return s, pd.DataFrame({k: daily_cost(a, s[k].values, p) for k in s},
                               index=s.index)

    _, costs_val = cost_table(va)
    strat, costs = cost_table(te)
    # The recommended strategy is chosen on validation cost, never on test.
    # Rule fixed in advance: among strategies whose validation cost is within
    # 1 % of the best, prefer one whose risk margin is an explicit, auditable
    # buffer (a single number a trader can check and recalibrate).
    candidates = [c for c in costs_val.columns
                  if c not in ("Seasonal naive (last week)", "TSO public forecast")]
    vc = costs_val[candidates].sum()
    val_best = vc.idxmin()
    near = vc[vc <= vc.min() * 1.01].index
    buffered = [c for c in near if "buffer" in c]
    rec = min(buffered, key=lambda c: vc[c]) if buffered else val_best
    A, P = y[te].values, price[te].values
    base = costs["Seasonal naive (last week)"].sum()
    tso = costs["TSO public forecast"].sum()
    summ = pd.DataFrame({
        "val_cost_1pct_kEUR": costs_val.sum() * SHARE / 1e3,
        "annual_cost_national_MEUR": costs.sum() / 1e6,
        "annual_cost_1pct_kEUR": costs.sum() * SHARE / 1e3,
        "saving_vs_naive_pct": (1 - costs.sum() / base) * 100,
        "saving_vs_TSO_pct": (1 - costs.sum() / tso) * 100,
        "mean_bias_GW": (strat.sub(A, axis=0)).mean(),
        "short_days_pct": (strat.lt(A, axis=0)).mean() * 100,
        "p95_daily_cost_1pct_kEUR": costs.quantile(0.95) * SHARE / 1e3,
        "max_daily_cost_1pct_kEUR": costs.max() * SHARE / 1e3,
    })
    summ["recommended"] = summ.index == rec
    summ.to_csv(TAB / "decision_costs.csv")
    print(summ.round(2).to_string())
    print("lowest validation cost:", val_best, "| recommended:", rec)

    # Block bootstrap (14-day blocks) of annual cost differences, so that the
    # saving comes with an uncertainty range.
    def boot(diff, n=2000, block=14, seed=0):
        rng = np.random.default_rng(seed)
        d = np.asarray(diff)
        nb = int(np.ceil(len(d) / block))
        starts = rng.integers(0, len(d) - block, size=(n, nb))
        sums = np.array([np.concatenate([d[s:s + block] for s in row])[:len(d)].sum()
                         for row in starts])
        return np.percentile(sums, [2.5, 50, 97.5])
    bs = {}
    for other in ("Seasonal naive (last week)", "TSO public forecast", val_best):
        if other == rec:
            continue
        lo, md, hi = boot((costs[other] - costs[rec]) * SHARE / 1e3)
        bs[f"test_saving_vs_{other}_1pct_kEUR_CI95"] = [lo, md, hi]
    lo, md, hi = boot((costs_val[val_best] - costs_val[rec]) * SHARE / 1e3)
    bs["val_cost_diff_valbest_minus_rec_1pct_kEUR_CI95"] = [lo, md, hi]
    bs["val_best"] = val_best
    bs["recommended"] = rec
    save_json(bs, TAB / "decision_bootstrap.json")
    print(bs)

    # ---- sensitivity to the imbalance-penalty assumption ----------------
    sens = []
    for a_, b_ in [(0.25, 0.25), (0.5, 0.25), (1.0, 0.25), (1.0, 0.5),
                   (0.25, 0.5)]:
        ts = a_ / (a_ + b_)
        sh = float(np.quantile((y - ens)[va], ts))
        row = {"alpha": a_, "beta": b_, "tau*": ts, "buffer_GW": sh}
        for k, f in [("naive", strat["Seasonal naive (last week)"]),
                     ("TSO", strat["TSO public forecast"]),
                     ("SARIMAX", strat[sx]),
                     ("Ensemble", strat[ENS]),
                     ("Ensemble+buffer", ens[te] + sh)]:
            row[k + "_1pct_kEUR"] = daily_cost(A, f.values, P, a_, b_).sum() * SHARE / 1e3
        sens.append(row)
    sens = pd.DataFrame(sens)
    sens.to_csv(TAB / "decision_sensitivity.csv", index=False)
    print(sens.round(2).to_string())

    # ---- cost as a function of the procurement quantile ------------------
    taus = np.linspace(0.05, 0.95, 37)
    fig, ax = plt.subplots(figsize=(6, 3))
    for lbl, base_fc, col in [("SARIMAX", cl[sx], "#2a6f97"),
                              ("Ensemble", ens, "#9e2a2b")]:
        rv = (y - base_fc)[va]
        curve = [daily_cost(A, (base_fc[te] + np.quantile(rv, t)).values,
                            P).sum() * SHARE / 1e3 for t in taus]
        ax.plot(taus, curve, color=col, label=lbl + " + error quantile")
    ax.axvline(tau_star, color="k", ls="--", lw=0.9)
    ax.axvline(0.5, color="grey", ls=":", lw=0.9)
    ax.text(tau_star + 0.01, ax.get_ylim()[1] * 0.97, f"tau* = {tau_star:.2f}",
            fontsize=8, va="top")
    ax.text(0.51, ax.get_ylim()[1] * 0.97, "median", color="grey", fontsize=8,
            va="top")
    ax.set_xlabel("procurement quantile tau (buffer = tau-quantile of validation errors)")
    ax.set_ylabel("test-year imbalance cost,\n1 % supplier (kEUR)")
    ax.set_title("Buying slightly more than the point forecast pays off",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "14_cost_vs_tau.png")
    plt.close(fig)

    # ---- risk: interval width as an early-warning signal ----------------
    width = qt["q975_cal"] - qt["q025_cal"]
    thr = float(width[va].quantile(0.9))
    flag = (width[te] > thr).values
    main_cost = costs[rec]
    err = A - strat[rec].values
    risk = {
        "alpha": ALPHA, "beta": BETA, "tau_star": tau_star,
        "recommended": rec,
        "buffer_sarimax_GW": shift, "buffer_ensemble_GW": shift_e,
        "width_threshold_GW": thr,
        "flagged_days": int(flag.sum()),
        "flagged_share_of_days_pct": float(flag.mean() * 100),
        "flagged_share_of_cost_pct": float(main_cost[flag].sum() /
                                           main_cost.sum() * 100),
        "mean_cost_flagged_1pct_EUR": float(main_cost[flag].mean() * SHARE),
        "mean_cost_unflagged_1pct_EUR": float(main_cost[~flag].mean() * SHARE),
        "upper_headroom_mean_GW": float((qt.loc[te, "q975_cal"] -
                                         strat[rec]).mean()),
        "upper_headroom_mean_pct_of_load": float(
            ((qt.loc[te, "q975_cal"] - strat[rec]) / y[te]).mean() * 100),
        "mean_price_test": float(price[te].mean()),
        "negative_price_days_test": int((price[te] < 0).sum()),
        "worst_days_1pct_EUR": {str(k.date()): round(float(v) * SHARE)
                                for k, v in main_cost.nlargest(8).items()},
        "abs_err_p95_GW": float(np.quantile(np.abs(err), 0.95)),
        "abs_err_max_GW": float(np.max(np.abs(err))),
        "short_days_pct": float((err > 0).mean() * 100),
        "daily_cost_mean_1pct_EUR": float(main_cost.mean() * SHARE),
        "daily_cost_p95_1pct_EUR": float(main_cost.quantile(0.95) * SHARE),
        "daily_cost_p99_1pct_EUR": float(main_cost.quantile(0.99) * SHARE),
        "cost_top10_days_share_pct": float(main_cost.nlargest(10).sum() /
                                           main_cost.sum() * 100),
    }
    # illustrative: flagged days are re-optimised in the intraday market,
    # where we assume the penalty is halved
    adj = main_cost.copy()
    adj[flag] *= 0.5
    risk["intraday_on_flagged_saving_1pct_kEUR"] = float(
        (main_cost.sum() - adj.sum()) * SHARE / 1e3)
    save_json(risk, TAB / "risk_summary.json")
    for k, v in risk.items():
        print(k, v)

    # ---- figures for report and poster ---------------------------------
    order = ["Naive (y[t-1])", "Seasonal naive (y[t-7])", "MA(14)", "AR(21)",
             [c for c in acct.index if c.startswith("ARIMA")][0], sa, "RNN",
             "LSTM", "LSTM + calendar", "TSO day-ahead forecast", sx, ENS]
    order = [o for o in order if o in acct.index]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    vals = acct.loc[order, "RMSE"]
    cols = ["#9aa5b1"] * len(order)
    cols[order.index("TSO day-ahead forecast")] = "#e09f3e"
    cols[order.index(sx)] = "#c76b6c"
    cols[order.index(ENS)] = "#9e2a2b"
    ax.barh(range(len(order)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o.replace("+holidays", " + holidays") for o in order],
                       fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("test RMSE (GW), lower is better")
    ax.set_title("One-step-ahead accuracy on the test year (Oct 2019 - Sep 2020)",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "15_rmse_ranking.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    s = summ["annual_cost_1pct_kEUR"]
    cols = ["#9aa5b1"] * len(s)
    cols[list(s.index).index("TSO public forecast")] = "#e09f3e"
    cols[list(s.index).index(rec)] = "#9e2a2b"
    ax.barh(range(len(s)), s.values, color=cols)
    for i, v in enumerate(s.values):
        ax.text(v + s.max() * 0.01, i, f"EUR {v:,.0f}k", va="center", fontsize=7)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([(i.replace("+holidays", " + holidays") +
                         ("  (recommended)" if i == rec else "")) for i in s.index],
                       fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("imbalance cost over the test year, supplier with 1 % share (kEUR)")
    ax.set_title("What each forecast costs when used to buy energy day-ahead",
                 fontsize=9, loc="left")
    ax.set_xlim(0, s.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / "16_cost_by_strategy.png")
    plt.close(fig)

    # zoom on holidays: Christmas 2019 and Easter 2020
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    for ax, (a, b, ttl) in zip(axs, [("2019-12-14", "2020-01-08", "Christmas 2019"),
                                     ("2020-04-01", "2020-06-05", "Easter, Ascension, Whitsun 2020")]):
        ax.plot(y[a:b], "k.-", lw=1, ms=3, label="actual")
        ax.plot(cl.loc[a:b, sa], lw=1, label=sa)
        ax.plot(cl.loc[a:b, sx], lw=1, label="SARIMAX + holidays")
        ax.plot(cl.loc[a:b, "TSO day-ahead forecast"], lw=1, ls="--",
                label="TSO forecast")
        ax.set_title(ttl, fontsize=8, loc="left")
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.set_ylabel("GW")
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "17_holiday_zoom.png")
    plt.close(fig)

    # monthly cost profile of the recommended strategy
    mc = costs[["Seasonal naive (last week)", "TSO public forecast", rec]]
    mc = (mc.resample("MS").sum() * SHARE / 1e3)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    w = 8
    for i, c in enumerate(mc.columns):
        ax.bar(mc.index + pd.Timedelta(days=(i - 1) * w), mc[c], width=w,
               label=c)
    ax.set_ylabel("kEUR / month (1 % supplier)")
    ax.legend(fontsize=7)
    ax.set_title("Monthly imbalance cost on the test year", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "18_monthly_cost.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

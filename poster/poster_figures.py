"""Print-resolution figures for the A1 poster (300 dpi, large type).

Built from the saved predictions and tables, so no model is retrained.
Palette (validated for colour-vision deficiency): blue = data, amber = the
TSO's public forecast, red = our recommended model, grey = everything else.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_daily, split_index, german_holidays, PRED, TAB  # noqa: E402
from importlib import import_module  # noqa: E402

dec = import_module("06_decision")

OUT = ROOT / "poster" / "fig"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, AMBER, RED, GREY, INK, MUTED = ("#1f6fb0", "#d98a1e", "#9e2a2b",
                                      "#a3acb7", "#1f2a36", "#5b6573")
plt.rcParams.update({
    "font.family": "Calibri", "font.size": 17, "axes.titlesize": 18,
    "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
    "grid.color": "#e3e6ea", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08})

d = load_daily()
y = d["load_gw"]
tr, va, te = split_index(d.index)
cl = pd.read_csv(PRED / "classical_predictions.csv", index_col=0, parse_dates=True)
nn = pd.read_csv(PRED / "neural_predictions.csv", index_col=0, parse_dates=True)
qt = pd.read_csv(PRED / "quantile_predictions.csv", index_col=0, parse_dates=True)
acc = pd.read_csv(TAB / "all_models_metrics.csv")
acc = acc[acc.split == "test"].set_index("model")
cost = pd.read_csv(TAB / "decision_costs.csv", index_col=0)
SX = [c for c in cl.columns if c.startswith("SARIMAX")][0]
ens = (cl[SX] + nn["LSTM + calendar"]) / 2

# 1 -- the series ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 2.9))
for m, c in [(tr, BLUE), (va, BLUE), (te, BLUE)]:
    ax.plot(y[m], color=c, lw=0.5, alpha=0.55)
ax.plot(y.rolling(28, center=True).mean(), color=INK, lw=2)
for (a, b, lab) in [("2015-01-01", "2018-09-30", "train"),
                    ("2018-10-01", "2019-09-30", "validation"),
                    ("2019-10-01", "2020-09-30", "test")]:
    mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
    ax.text(mid, 70.5, lab, ha="center", fontsize=16, color=MUTED)
    ax.axvline(pd.Timestamp(a), color=MUTED, lw=0.8, ls=":")
ax.axvspan(pd.Timestamp("2020-03-16"), pd.Timestamp("2020-05-10"), color=RED,
           alpha=0.12, lw=0)
ax.text(pd.Timestamp("2020-02-20"), 36.5, "COVID-19\nlockdown", fontsize=14,
        color=RED, ha="right")
ax.set_ylim(35, 73)
ax.set_ylabel("daily mean load (GW)")
fig.savefig(OUT / "p1_series.png")
plt.close(fig)

# 2 -- the week, and what a holiday does ------------------------------------
ytr = y[tr]
hol = german_holidays(range(2015, 2019))
wk_hol = ytr[ytr.index.isin(hol) & (ytr.index.dayofweek < 5)]
vals = list(ytr[~ytr.index.isin(hol)].groupby(ytr.index[~ytr.index.isin(hol)].dayofweek).mean())
labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
vals.append(wk_hol.mean())
labels.append("Holiday\n(weekday)")
fig, ax = plt.subplots(figsize=(10.5, 3.0))
cols = [GREY] * 5 + [BLUE, BLUE, RED]
ax.bar(range(8), vals, color=cols, width=0.72)
for i, v in enumerate(vals):
    ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=15, color=INK)
ax.set_xticks(range(8))
ax.set_xticklabels(labels)
ax.set_ylim(40, 64)
ax.set_ylabel("mean load (GW)")
ax.grid(axis="x", visible=False)
fig.savefig(OUT / "p2_week_holiday.png")
plt.close(fig)

# 3 -- accuracy ranking -----------------------------------------------------
ARI = [m for m in acc.index if m.startswith("ARIMA")][0]
SA = [m for m in acc.index if m.startswith("SARIMA(")][0]
order = [("Same as last week", "Seasonal naive (y[t-7])", GREY),
         ("MA(14)", "MA(14)", GREY), ("AR(21)", "AR(21)", GREY),
         (ARI, ARI, GREY), ("SARIMA", SA, GREY), ("RNN", "RNN", GREY),
         ("LSTM", "LSTM", GREY), ("TSO public forecast", "TSO day-ahead forecast", AMBER),
         ("SARIMAX + holidays", SX, "#c98a8b"), ("LSTM + calendar", "LSTM + calendar", "#c98a8b"),
         ("Ensemble (ours)", "Ensemble (SARIMAX + LSTM-cal)", RED)]
fig, ax = plt.subplots(figsize=(10.5, 4.6))
for i, (lab, key, c) in enumerate(order):
    v = acc.loc[key, "RMSE"]
    ax.barh(i, v, color=c, height=0.7)
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=15, color=INK)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([o[0] for o in order])
ax.invert_yaxis()
ax.set_xlim(0, 4.2)
ax.set_xlabel("test-year RMSE (GW), lower is better")
ax.grid(axis="y", visible=False)
fig.savefig(OUT / "p3_rmse.png")
plt.close(fig)

# 4 -- newsvendor: cost vs procurement quantile -------------------------------
A, P = y[te].values, d["price_eur"][te].values
rv = (y - ens)[va]
taus = np.linspace(0.05, 0.95, 37)
curve = [dec.daily_cost(A, (ens[te] + np.quantile(rv, t)).values, P).sum() * 0.01 / 1e3
         for t in taus]
rv2 = (y - cl[SX])[va]
curve2 = [dec.daily_cost(A, (cl[SX][te] + np.quantile(rv2, t)).values, P).sum() * 0.01 / 1e3
          for t in taus]
fig, ax = plt.subplots(figsize=(10.5, 3.2))
ax.plot(taus, curve2, color=GREY, lw=2.5)
ax.plot(taus, curve, color=RED, lw=3)
i15, i25 = np.argmin(abs(taus - 0.15)), np.argmin(abs(taus - 0.25))
ax.text(0.16, curve2[i15] + 60, "SARIMAX", color=MUTED, fontsize=15)
ax.text(0.22, curve[i25] - 230, "Ensemble", color=RED, fontsize=15)
ax.axvline(2 / 3, color=INK, ls="--", lw=1.2)
ax.axvline(0.5, color=MUTED, ls=":", lw=1.2)
ax.text(2 / 3 + 0.01, max(curve2) * 0.95, "buy here\n$\\tau^*$ = 0.67", fontsize=15)
ax.text(0.49, max(curve2) * 0.95, "the mean\n($\\tau$ = 0.5)", fontsize=15,
        color=MUTED, ha="right")
ax.set_xlabel("which quantile of tomorrow's demand to buy ($\\tau$)")
ax.set_ylabel("cost / year (kEUR)")
fig.savefig(OUT / "p4_tau.png")
plt.close(fig)

# 5 -- money ------------------------------------------------------------------
rec = cost.index[cost["recommended"]][0]
bars = [("Buy 'same as last week'", "Seasonal naive (last week)", GREY),
        ("Buy the TSO public forecast", "TSO public forecast", AMBER),
        ("SARIMAX + holidays", SX, "#c98a8b"),
        ("Ensemble + buffer (ours)", rec, RED)]
fig, ax = plt.subplots(figsize=(10.5, 2.9))
for i, (lab, key, c) in enumerate(bars):
    v = cost.loc[key, "annual_cost_1pct_kEUR"]
    ax.barh(i, v, color=c, height=0.66)
    ax.text(v + 25, i, f"EUR {v:,.0f}k", va="center", fontsize=16, color=INK)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in bars])
ax.invert_yaxis()
ax.set_xlim(0, 2500)
ax.set_xlabel("imbalance cost over the test year, 1 % supplier (kEUR)")
ax.grid(axis="y", visible=False)
fig.savefig(OUT / "p5_cost.png")
plt.close(fig)

# 6 -- interval through the COVID shock --------------------------------------
z = qt.loc["2020-02-17":"2020-05-17"]
fig, ax = plt.subplots(figsize=(10.5, 3.1))
ax.fill_between(z.index, z["q025_cal"], z["q975_cal"], color=BLUE, alpha=0.22,
                lw=0)
ax.plot(z["actual"], color=INK, lw=1.8, marker="o", ms=3)
miss = (z["actual"] < z["q025_cal"]) | (z["actual"] > z["q975_cal"])
ax.scatter(z.index[miss], z["actual"][miss], s=70, color=RED, zorder=3,
           edgecolor="white", linewidth=1.5)
ax.axvline(pd.Timestamp("2020-03-16"), color=MUTED, ls="--", lw=1)
ax.text(pd.Timestamp("2020-03-17"), 36.2, "lockdown", fontsize=15, color=MUTED)
ax.text(pd.Timestamp("2020-02-18"), 69.5, "95 % interval", fontsize=15, color=BLUE)
ax.text(pd.Timestamp("2020-03-28"), 69.5, "actual", fontsize=15, color=INK)
ax.text(pd.Timestamp("2020-04-18"), 69.5, "outside interval", fontsize=15, color=RED)
ax.set_ylim(35, 72)
ax.set_ylabel("GW")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
fig.savefig(OUT / "p6_interval.png")
plt.close(fig)
print("poster figures ->", OUT)

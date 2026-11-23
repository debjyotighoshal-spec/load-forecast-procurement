"""Task 13: probabilistic forecast with a pair of quantile LSTMs.

Two LSTMs are trained by back-propagating the pinball loss, one for the 2.5 %
and one for the 97.5 % quantile, giving a nominal 95 % prediction interval.
A third LSTM estimates the cost-optimal procurement quantile tau* used in the
decision analysis (see 06_decision.py). The architecture is the one tuned for
the point LSTM in 04_neural.py, with calendar inputs.

Because a quantile network trained on 3.7 years can be over- or under-confident
on new data, we also report a split-conformal recalibration (Romano et al.,
2019): the interval is widened (or narrowed) by the 95 % quantile of the
validation-year conformity scores. Note that the validation year is also used
for early stopping, so this calibration is slightly optimistic.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, TAB, PRED, load_daily, split_index, save_json
from nnutils import make_windows, train_model, predict
from importlib import import_module

neural = import_module("04_neural")

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

ALPHA_SHORT, BETA_LONG = 0.5, 0.25          # see 06_decision.py
TAU_STAR = ALPHA_SHORT / (ALPHA_SHORT + BETA_LONG)
TAUS = {"q025": 0.025, "q975": 0.975, "q_star": TAU_STAR}
SEEDS = [0, 1, 2]


def picp_mpiw(y, lo, hi):
    inside = (y >= lo) & (y <= hi)
    return float(inside.mean() * 100), float(np.mean(hi - lo))


def main():
    df = load_daily()
    y, z, cal, masks, (mu, sd) = neural.prepare(df)
    tr, va, te = masks
    cfg = json.load(open(TAB / "neural_summary.json"))["LSTM"]["config"]
    (trd, _), (vad, vidx), (ted, tidx) = neural.datasets(z, cal, masks,
                                                        cfg["lag"], True)
    out = pd.DataFrame({"actual": y}, index=df.index)
    for name, tau in TAUS.items():
        pv, pt = [], []
        for s in SEEDS:
            m, _, _ = train_model(cfg, trd, vad, loss="pinball", tau=tau, seed=s)
            pv.append(predict(m, vad) * sd + mu)
            pt.append(predict(m, ted) * sd + mu)
        col = np.full(len(y), np.nan)
        col[vidx] = np.mean(pv, axis=0)
        col[tidx] = np.mean(pt, axis=0)
        out[name] = col
        print(name, "done")

    # guard against quantile crossing
    lo = np.minimum(out["q025"], out["q975"])
    hi = np.maximum(out["q025"], out["q975"])
    crossings = int((out["q025"] > out["q975"]).sum())
    out["q025"], out["q975"] = lo, hi

    # split-conformal recalibration on the validation year
    yv = out.loc[va, "actual"].values
    score = np.maximum(out.loc[va, "q025"].values - yv,
                       yv - out.loc[va, "q975"].values)
    n = len(score)
    qhat = float(np.quantile(score, min(1, np.ceil((n + 1) * 0.95) / n)))
    out["q025_cal"] = out["q025"] - qhat
    out["q975_cal"] = out["q975"] + qhat
    out.to_csv(PRED / "quantile_predictions.csv", index_label="date")

    res = {"tau_star": TAU_STAR, "conformal_qhat_gw": qhat,
           "crossings": crossings, "config": cfg}
    covid = (df.index >= "2020-03-16") & (df.index <= "2020-05-10")
    for split, m in [("val", va), ("test", te),
                     ("test_excl_lockdown", te & ~covid),
                     ("test_lockdown", te & covid)]:
        yy = out.loc[m, "actual"].values
        for tag, a, b in [("raw", "q025", "q975"),
                          ("conformal", "q025_cal", "q975_cal")]:
            p, w = picp_mpiw(yy, out.loc[m, a].values, out.loc[m, b].values)
            res[f"{split}_{tag}_PICP"] = p
            res[f"{split}_{tag}_MPIW"] = w
        res[f"{split}_tau_star_coverage"] = float(
            (yy <= out.loc[m, "q_star"].values).mean() * 100)
        res[f"{split}_q025_below"] = float((yy < out.loc[m, "q025"]).mean() * 100)
        res[f"{split}_q975_above"] = float((yy > out.loc[m, "q975"]).mean() * 100)
    save_json(res, TAB / "quantile_summary.json")
    for k, v in res.items():
        print(k, v)

    # ---- plots ------------------------------------------------------------
    t = out[te]
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.6))
    for ax, (a, b, ttl) in zip(axs, [
            ("q025", "q975", "raw quantile-LSTM interval"),
            ("q025_cal", "q975_cal", "after split-conformal recalibration")]):
        p, w = picp_mpiw(t["actual"].values, t[a].values, t[b].values)
        ax.fill_between(t.index, t[a], t[b], color="#2a6f97", alpha=0.3,
                        label="95 % prediction interval")
        ax.plot(t["actual"], color="k", lw=0.7, label="actual")
        miss = (t["actual"] < t[a]) | (t["actual"] > t[b])
        ax.scatter(t.index[miss], t["actual"][miss], color="#c0392b", s=10,
                   zorder=3, label="outside interval")
        ax.set_title(f"Test year, {ttl}: PICP {p:.1f} %, MPIW {w:.2f} GW",
                     fontsize=8, loc="left")
        ax.set_ylabel("GW")
        ax.set_ylim(33, 75)
    axs[0].legend(fontsize=7, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "12_interval_test.png")
    plt.close(fig)

    z_ = t.loc["2020-02-15":"2020-05-15"]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(z_.index, z_["q025"], z_["q975"], color="#2a6f97",
                    alpha=0.3, label="95 % interval (raw)")
    ax.plot(z_["actual"], "k.-", lw=0.8, ms=3, label="actual")
    ax.plot(z_["q_star"], color="#e09f3e", lw=1, label=f"q{TAU_STAR:.3f} (procure)")
    ax.axvline(pd.Timestamp("2020-03-16"), color="grey", ls="--", lw=0.8)
    ax.text(pd.Timestamp("2020-03-17"), 36, "lockdown begins", fontsize=7)
    ax.set_title("Zoom: interval forecast through the COVID-19 demand shock",
                 fontsize=9, loc="left")
    ax.set_ylabel("GW")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "13_interval_zoom_covid.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Tasks 10 and 11: RNN and LSTM one-step-ahead forecasters.

Hyper-parameters (lag window, batch size, learning rate, layers, neurons) are
tuned by random search on the validation year. The chosen configuration is
then retrained with three seeds; the reported forecast is the seed average and
the seed-to-seed spread is reported as well. An extra LSTM variant also sees
the calendar of the target day (weekday, public holiday, Christmas break),
which is known in advance and therefore legitimate input.
"""
import itertools
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FIG, TAB, PRED, load_daily, split_index, metrics,
                    calendar_features, save_json)
from nnutils import make_windows, train_model, predict

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

GRID = {"lag": [7, 14, 28, 56], "batch": [16, 32, 64],
        "lr": [1e-3, 3e-3, 1e-2], "layers": [1, 2],
        "hidden": [16, 32, 64, 128]}
N_TRIALS = 24
SEEDS = [0, 1, 2]


def prepare(df):
    y = df["load_gw"].values
    tr, va, te = split_index(df.index)
    mu, sd = y[tr].mean(), y[tr].std()
    z = (y - mu) / sd
    cal = calendar_features(df.index).values
    return y, z, cal, (tr, va, te), (mu, sd)


def datasets(z, cal, masks, lag, use_cal):
    c = cal if use_cal else None
    return [make_windows(z, c, lag, m) for m in masks]


def run_search(cell, z, cal, masks, mu, sd, y, use_cal=False):
    combos = list(itertools.product(*GRID.values()))
    random.Random(7).shuffle(combos)
    rows = []
    for combo in combos[:N_TRIALS]:
        cfg = dict(zip(GRID.keys(), combo), cell=cell)
        (trd, _), (vad, vidx), _ = datasets(z, cal, masks, cfg["lag"], use_cal)
        t0 = time.time()
        model, _, hist = train_model(cfg, trd, vad, seed=0)
        pv = predict(model, vad) * sd + mu
        m = metrics(y[vidx], pv)
        rows.append({**cfg, "epochs": len(hist), "val_RMSE": m["RMSE"],
                     "val_MAPE": m["MAPE"], "sec": round(time.time() - t0, 1)})
        print(cell, rows[-1])
    return pd.DataFrame(rows).sort_values("val_RMSE")


def final_fit(cfg, z, cal, masks, mu, sd, y, n, use_cal):
    (trd, _), (vad, vidx), (ted, tidx) = datasets(z, cal, masks, cfg["lag"],
                                                  use_cal)
    pv, pt, per_seed, hists = [], [], [], []
    for s in SEEDS:
        model, _, hist = train_model(cfg, trd, vad, seed=s)
        a = predict(model, vad) * sd + mu
        b = predict(model, ted) * sd + mu
        pv.append(a)
        pt.append(b)
        hists.append(hist)
        per_seed.append(metrics(y[tidx], b)["RMSE"])
    full = np.full(n, np.nan)
    full[vidx] = np.mean(pv, axis=0)
    full[tidx] = np.mean(pt, axis=0)
    return full, per_seed, hists


def main():
    df = load_daily()
    y, z, cal, masks, (mu, sd) = prepare(df)
    tr, va, te = masks
    out, preds = {}, pd.DataFrame({"actual": y}, index=df.index)
    curves = {}
    tables = []
    for cell, label, use_cal in [("rnn", "RNN", False), ("lstm", "LSTM", False)]:
        s = run_search(cell, z, cal, masks, mu, sd, y, use_cal)
        s.to_csv(TAB / f"search_{cell}.csv", index=False)
        best = s.iloc[0][list(GRID) + ["cell"]].to_dict()
        best = {k: (int(v) if k in ("lag", "batch", "layers", "hidden") else v)
                for k, v in best.items()}
        full, per_seed, hists = final_fit(best, z, cal, masks, mu, sd, y,
                                          len(y), use_cal)
        preds[label] = full
        curves[label] = hists[0]
        out[label] = {"config": best, "test_RMSE_per_seed": per_seed}
        tables.append(s.assign(model=label))

    # LSTM with calendar inputs, using the tuned LSTM architecture
    cfg = out["LSTM"]["config"]
    full, per_seed, hists = final_fit(cfg, z, cal, masks, mu, sd, y, len(y),
                                      True)
    preds["LSTM + calendar"] = full
    curves["LSTM + calendar"] = hists[0]
    out["LSTM + calendar"] = {"config": cfg, "test_RMSE_per_seed": per_seed}

    preds.to_csv(PRED / "neural_predictions.csv", index_label="date")
    rows = []
    for c in ["RNN", "LSTM", "LSTM + calendar"]:
        for split, m in [("val", va), ("test", te)]:
            rows.append({"model": c, "split": split,
                         **metrics(y[m], preds.loc[m, c])})
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "neural_metrics.csv", index=False)
    print(tab.round(3).to_string())
    save_json(out, TAB / "neural_summary.json")

    # ---- plots ------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.6))
    for ax, (k, h) in zip(axs, curves.items()):
        h = np.array(h)
        ax.plot(h[:, 0], label="train MSE")
        ax.plot(h[:, 1], label="validation MSE")
        ax.set_yscale("log")
        ax.set_title(f"{k}: learning curve", fontsize=8)
        ax.set_xlabel("epoch")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "08_nn_learning_curves.png")
    plt.close(fig)

    yt = preds.loc[te, "actual"]
    fig, axs = plt.subplots(3, 1, figsize=(9, 5.4), sharex=True)
    for ax, c in zip(axs, ["RNN", "LSTM", "LSTM + calendar"]):
        m = metrics(yt, preds.loc[te, c])
        ax.plot(yt, color="k", lw=0.8, label="actual")
        ax.plot(preds.loc[te, c], color="#c0392b", lw=0.8, label="forecast")
        ax.set_title(f"{c}   RMSE {m['RMSE']:.2f} GW | MAE {m['MAE']:.2f} | "
                     f"MAPE {m['MAPE']:.2f}% | R² {m['R2']:.3f}", fontsize=8,
                     loc="left")
        ax.set_ylim(35, 72)
    axs[0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "09_nn_test_forecasts.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, c in zip(axs, ["RNN", "LSTM", "LSTM + calendar"]):
        ax.scatter(yt, preds.loc[te, c], s=4, alpha=0.5, color="#2a6f97")
        ax.plot([38, 70], [38, 70], color="k", lw=0.7)
        ax.set_xlim(38, 70)
        ax.set_ylim(38, 70)
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("actual (GW)")
    axs[0].set_ylabel("predicted (GW)")
    fig.tight_layout()
    fig.savefig(FIG / "10_nn_scatter.png")
    plt.close(fig)

    plot_tuning()


def plot_tuning():
    """Validation RMSE of every random-search trial against each setting."""
    allt = pd.concat([pd.read_csv(TAB / f"search_{c}.csv").assign(model=c.upper())
                      for c in ("rnn", "lstm")])
    hps = ["lag", "batch", "lr", "layers", "hidden"]
    fig, axs = plt.subplots(1, len(hps), figsize=(11, 2.6), sharey=True)
    for ax, hp in zip(axs, hps):
        levels = GRID[hp]
        for k, (lbl, g) in enumerate(allt.groupby("model")):
            x = [levels.index(v) + (k - 0.5) * 0.18 for v in g[hp]]
            ax.scatter(x, g["val_RMSE"], s=12, label=lbl, alpha=0.75)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(v) for v in levels], fontsize=7)
        ax.set_xlabel(hp)
    axs[0].set_ylabel("validation RMSE (GW)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Random search: 24 trials per network", fontsize=9, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "11_nn_tuning.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

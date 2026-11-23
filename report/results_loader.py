"""Collect every number quoted in the report and poster from the result files,
so that text and results cannot drift apart."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

# Change this to your repository URL and rebuild.
REPO_URL = "https://github.com/debjyotighoshal-spec/load-forecast-procurement"


def _j(name):
    return json.load(open(TAB / name))


def load():
    R = {"eda": _j("eda_summary.json"), "cls": _j("classical_search.json"),
         "nn": _j("neural_summary.json"), "q": _j("quantile_summary.json"),
         "risk": _j("risk_summary.json"), "boot": _j("decision_bootstrap.json"),
         "rob": _j("robustness.json")}
    acc = pd.read_csv(TAB / "all_models_metrics.csv")
    R["acc_test"] = acc[acc.split == "test"].set_index("model")
    R["acc_val"] = acc[acc.split == "val"].set_index("model")
    R["ur"] = pd.read_csv(TAB / "unit_root_tests.csv")
    R["cost"] = pd.read_csv(TAB / "decision_costs.csv", index_col=0)
    R["sens"] = pd.read_csv(TAB / "decision_sensitivity.csv")
    R["search_rnn"] = pd.read_csv(TAB / "search_rnn.csv")
    R["search_lstm"] = pd.read_csv(TAB / "search_lstm.csv")
    R["rec"] = R["risk"]["recommended"]
    R["SX"] = [m for m in R["acc_test"].index if m.startswith("SARIMAX")][0]
    R["SA"] = [m for m in R["acc_test"].index if m.startswith("SARIMA(")][0]
    R["AR"] = [m for m in R["acc_test"].index if m.startswith("ARIMA")][0]
    R["ENS"] = "Ensemble (SARIMAX + LSTM-cal)"
    return R

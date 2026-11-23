# From forecast to purchase: day-ahead electricity procurement

Course project, Time-Series Forecasting.
**Debjyoti Ghoshal, Ajit Kumar**

We forecast German daily electricity demand one day ahead and use the
forecast to decide how much energy a supplier should buy in the day-ahead
market. The project compares classical linear models (MA, AR, ARIMA, SARIMA,
SARIMAX) with recurrent neural networks (RNN, LSTM). It builds a 95 %
prediction interval from a pair of quantile LSTMs and puts a euro value on
each forecast through an imbalance-cost model.

The full report is in [`report/`](report/) and the poster is in
[`poster/`](poster/).

## Main results (test year, Oct 2019 - Sep 2020, one-step-ahead)

| Model | RMSE (GW) | MAPE (%) | R² |
|---|---|---|---|
| Seasonal naive (same day last week) | 3.72 | 4.08 | 0.696 |
| MA(14) | 3.32 | 5.03 | 0.758 |
| AR(21) | 2.41 | 3.11 | 0.873 |
| ARIMA(7,1,1) | 2.64 | 3.24 | 0.847 |
| SARIMA(2,0,2)(0,1,1)7 | 2.25 | 2.65 | 0.889 |
| SARIMAX + holiday dummies | 1.54 | 2.00 | 0.948 |
| RNN | 2.23 | 2.78 | 0.891 |
| LSTM | 2.22 | 2.84 | 0.892 |
| LSTM + calendar inputs | 1.54 | 2.06 | 0.948 |
| **Ensemble (SARIMAX + LSTM-calendar)** | **1.40** | **1.87** | **0.957** |
| TSO public day-ahead forecast (benchmark) | 1.76 | 2.59 | 0.932 |

Decision: buying the ensemble forecast plus a 0.48 GW buffer (the 67th
percentile, because shortfalls cost twice as much as surpluses) gives an
imbalance cost of EUR 826k a year for a supplier with 1 % of German demand.
The alternatives cost EUR 1,469k with the public TSO forecast and EUR 2,034k
with "same as last week". Under realistic bidding timing (bids due at noon the
day before) the saving against the public forecast is about EUR 255k a year.
The quantile-LSTM 95 % interval covers 95.1 % of validation days and 92.3 % of
test days.

## Data

* Source: Open Power System Data, *Time series* package, version 2020-10-06
  (ENTSO-E Transparency Platform data), CC-BY 4.0.
  <https://data.open-power-system-data.org/time_series/2020-10-06/>
* Series used: `DE_load_actual_entsoe_transparency` (target),
  `DE_load_forecast_entsoe_transparency` (the grid operators' own day-ahead
  forecast, used only as a benchmark), `DE_LU_price_day_ahead` (used to price
  forecast errors).
* `data/processed/de_daily_load.csv` (2,100 days, 2015-01-01 to 2020-09-30)
  is committed. The 130 MB hourly raw file is not committed because it is
  over GitHub's file-size limit. `src/01_prepare_data.py` downloads it again.

Split (chronological, no shuffling, used for every model):

| set        | period                   | days |
|------------|--------------------------|------|
| train      | 2015-01-01 to 2018-09-30 | 1369 |
| validation | 2018-10-01 to 2019-09-30 | 365  |
| test       | 2019-10-01 to 2020-09-30 | 366  |

## How to reproduce

```bash
pip install -r requirements.txt
cd src
python 01_prepare_data.py   # download + daily aggregation
python 02_eda.py            # decomposition, ACF/PACF, ADF/KPSS
python 03_classical.py      # MA, AR, ARIMA, SARIMA, SARIMAX (~5 min)
python 04_neural.py         # RNN / LSTM tuning (~15 min on a laptop CPU)
python 05_quantile.py       # quantile LSTMs, PICP / MPIW
python 06_decision.py       # procurement cost, risk analysis
python 07_robustness.py     # two-day-ahead timing check, TSO bias check
cd ../report && python make_report.py     # DOCX + PDF (PDF export needs MS Word)
cd ../poster && python poster_figures.py && python build_poster.py
```

The GitHub link printed in the report and poster is set by `REPO_URL` in
`report/results_loader.py`.

Seeds are fixed (`common.SEED`, plus seeds 0-2 for the networks). Small
differences in the neural results can still appear across PyTorch versions
and CPUs.

## Repository layout

```
data/processed/      daily dataset used by all scripts
src/                 numbered pipeline scripts + shared helpers
results/figures/     every figure used in the report and poster
results/tables/      metrics, search grids, decision and risk tables (CSV/JSON)
results/predictions/ one-step-ahead predictions of every model
report/              report (DOCX + PDF) and the script that builds it
poster/              poster (PPTX + PDF + PNG) and its build script
```

## AI use

An AI assistant (Claude, by Anthropic) helped write the code and draft the
report and poster. See the AI disclaimer in the report.

## Licence

Code: MIT. Data: OPSD / ENTSO-E under their original licence terms (CC-BY 4.0
for the OPSD package).

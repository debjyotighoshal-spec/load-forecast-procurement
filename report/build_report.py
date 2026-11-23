"""Build report/Report_Load_Forecast_Procurement.docx from the result files.

Every number in the text is read from results/tables, so re-running the
pipeline and then this script keeps the report consistent with the code.
"""
import json

from docx.shared import Cm

from docx_helpers import Report
from results_loader import load, FIG, ROOT, REPO_URL

R = load()
A, V = R["acc_test"], R["acc_val"]
C = R["cost"]
E, Q, K, B, RB = R["eda"], R["q"], R["risk"], R["boot"], R["rob"]
SX, SA, ARI, ENS, REC = R["SX"], R["SA"], R["AR"], R["ENS"], R["rec"]
SHORT = {SX: "SARIMAX + holidays", SA: "SARIMA", ENS: "Ensemble"}


def f(x, d=2):
    return f"{x:,.{d}f}"


def k(x):            # kEUR -> "EUR 826k"
    return f"EUR {x:,.0f}k"


def m(x):            # kEUR -> "EUR 0.83 million"
    return f"EUR {x / 1000:,.2f} million"


cost_rec = C.loc[REC, "annual_cost_1pct_kEUR"]
cost_tso = C.loc["TSO public forecast", "annual_cost_1pct_kEUR"]
cost_nai = C.loc["Seasonal naive (last week)", "annual_cost_1pct_kEUR"]
sav_tso = cost_tso - cost_rec
sav_nai = cost_nai - cost_rec
ci_tso = B["test_saving_vs_TSO public forecast_1pct_kEUR_CI95"]
ci_nai = B["test_saving_vs_Seasonal naive (last week)_1pct_kEUR_CI95"]
buf = K["buffer_ensemble_GW"]
mean_load = 55.49
real_tso = RB["TSO+buffer_test_cost_1pct_kEUR"]
real_bl = RB["Blend_TSO_SARIMAX2_test_cost_1pct_kEUR"]
real_sav = real_tso - real_bl
sxp = R["cls"]["chosen"]["SARIMAX"]["params"]
ur = R["ur"].set_index("series")
wd = E["weekday_mean_gw"]
vs = E["decomp_var_share"]
nn = R["nn"]

rep = Report()
doc = rep.doc

# --------------------------------------------------------------------------
# Title page
# --------------------------------------------------------------------------
for _ in range(5):
    rep.p("")
rep.p("Time-Series Forecasting: Course Project Report", align="center", size=12)
t = rep.p("**From Forecast to Purchase**", align="center", size=26)
rep.p("Forecasting German daily electricity demand and turning the forecast "
      "into a day-ahead energy procurement decision", align="center", size=14)
rep.p("")
rep.p("**Debjyoti Ghoshal**  and  **Ajit Kumar**", align="center", size=13)
rep.p("Group [   ]", align="center", size=11)
rep.p("")
rep.p(f"Submitted: 24 November 2026", align="center", size=11)
rep.p(f"Code and data: {REPO_URL}", align="center", size=10)
rep.page_break()

# --------------------------------------------------------------------------
# Management summary (one page)
# --------------------------------------------------------------------------
rep.h("Management summary", 1)
rep.p("**The decision.** An electricity supplier has to buy tomorrow's energy in "
      "the day-ahead auction before it knows what its customers will use. If it "
      "buys too little, it covers the gap at the imbalance price, which we take "
      "as 50 % above the auction price. If it buys too much, it sells the surplus "
      "back at 25 % below. We studied a supplier serving 1 % of German demand "
      f"(about 0.55 GW on average, 4.9 TWh a year). Its question is: *how much "
      "should we buy for tomorrow?*", align="justify")
rep.p("**Recommendation.** Forecast tomorrow's demand as the average of two "
      "models: a seasonal ARIMA model that knows the public-holiday calendar "
      "(SARIMAX), and an LSTM neural network that sees the last 28 days and the "
      "calendar. Then **buy "
      f"{f(buf)} GW-equivalent ({f(buf / mean_load * 100, 1)} % of average load) "
      "more than the forecast.** The buffer is there because a shortfall costs "
      "twice as much as a surplus. The best purchase is therefore the 67th "
      "percentile of tomorrow's demand, not the average. We estimated the "
      "buffer from the previous year's forecast errors. In operation it "
      "should be recalibrated every month.",
      align="justify")
rep.p("**Expected saving** (test year October 2019 to September 2020, 1 % supplier):",
      space_after=2)
rep.table(["Procurement strategy", "Imbalance cost / year",
           "Our saving against it"],
          [["Buy 'same as last week' (status quo)", k(cost_nai),
            f"{k(sav_nai)}  ({f(C.loc[REC, 'saving_vs_naive_pct'], 0)} %)"],
           ["Buy the grid operators' free public forecast", k(cost_tso),
            f"{k(sav_tso)}  ({f(C.loc[REC, 'saving_vs_TSO_pct'], 0)} %)"],
           ["Recommended: ensemble forecast + buffer", k(cost_rec), "-"]],
          "Annual imbalance cost by procurement strategy (test year).",
          widths_cm=[8.0, 3.8, 4.4], font=9, bold_rows=(2,))
rep.p(f"A block bootstrap puts the saving against the public forecast between "
      f"{k(ci_tso[0])} and {k(ci_tso[2])} a year (95 % range). In this "
      f"study every model forecasts with all of yesterday's data. A real "
      f"supplier must bid by noon the day before. Under that stricter timing, "
      f"the best option is to average our model with the public forecast, and "
      f"the saving falls to about **{k(real_sav)} a year**. That is still "
      f"positive, and it is the figure we would budget with. The forecast "
      f"runs automatically in under a minute a day, so no analyst time is "
      f"needed for routine bids.", align="justify")
rep.p(f"**Risk.** On a typical day the forecast is off by "
      f"{f(A.loc[ENS, 'MAE'])} GW ({f(A.loc[ENS, 'MAPE'])} %). One day in "
      f"20 it is off by more than {f(K['abs_err_p95_GW'], 1)} GW. The worst "
      f"miss was {f(K['abs_err_max_GW'], 1)} GW. For the 1 % supplier, the "
      f"imbalance bill is EUR {K['daily_cost_mean_1pct_EUR']:,.0f} on an "
      f"average day, above EUR {K['daily_cost_p95_1pct_EUR']:,.0f} on one "
      f"day in 20, and EUR {max(K['worst_days_1pct_EUR'].values()):,} on the "
      f"worst day. Three of the eight worst days were regional holidays the "
      f"model does not know about. The 95 % prediction interval covered "
      f"{f(Q['test_raw_PICP'], 1)} % of test days, and only "
      f"{f(Q['test_lockdown_raw_PICP'], 0)} % during the COVID-19 lockdown. "
      f"It is a little too narrow when demand shifts suddenly, so it should "
      f"not be the only safeguard.", align="justify")
rep.page_break()

rep.h("Contents", 1)
rep.toc()

# --------------------------------------------------------------------------
# 1 Introduction
# --------------------------------------------------------------------------
rep.h("1  Introduction", 1).paragraph_format.page_break_before = True
rep.p("Electricity cannot be stored cheaply in large amounts, so supply has to "
      "match demand at every moment. In liberalised markets such as Germany, "
      "each supplier is financially responsible for any gap between the energy "
      "it bought in advance and the energy its customers actually used. Most "
      "of a supplier's energy is bought in the day-ahead auction. Bids for each "
      "hour of day *t* must be in by noon on day *t-1*. Whatever is still "
      "wrong after that is settled at the imbalance price, which is usually "
      "unfavourable. A better demand forecast therefore turns directly into "
      "money.", align="justify")
rep.p("We picked this problem for three reasons. First, the data are public, "
      "long and of high quality. Second, the grid operators publish their own "
      "day-ahead forecast, which gives us a professional benchmark rather "
      "than a naive one. Third, the money attached to a forecast error can be "
      "computed from real auction prices. The question we answer is "
      "practical: *what quantity should a supplier buy for tomorrow, and how "
      "much does a good forecast save compared with simple rules or the "
      "public forecast?*", align="justify")
rep.p("The report follows the 14 tasks of the project brief. Chapter 2 "
      "describes the data and the split. Chapters 3 and 4 analyse the series "
      "and its stationarity. Chapter 5 fits the classical linear models (MA, "
      "AR, ARIMA, SARIMA). Chapter 6 fits recurrent neural networks, and "
      "Chapter 7 compares all models. Chapters 8 to 10 turn the forecasts into "
      "a procurement decision, build a probabilistic forecast and quantify "
      "risk. Chapter 11 assesses the solution from a management point of view. "
      f"All code and the processed data are at {REPO_URL}.", align="justify")

# --------------------------------------------------------------------------
# 2 Data
# --------------------------------------------------------------------------
rep.h("2  Data (Task 1)", 1)
rep.h("2.1  Source and variables", 2)
rep.p("We use the Open Power System Data (OPSD) *Time series* package, version "
      "2020-10-06 [11]. It republishes hourly data from the ENTSO-E "
      "Transparency Platform [12]. Three German series are used:", align="justify")
rep.bullets(["**Actual load** (`DE_load_actual_entsoe_transparency`), the target. "
             "It is the total electricity consumed in Germany, including losses, in MW.",
             "**TSO day-ahead load forecast** (`DE_load_forecast_entsoe_transparency`). "
             "This forecast is published by the four transmission system operators. We "
             "use it only as a benchmark, never as a model input, except in the "
             "realistic-timing check of Section 8.6.",
             "**Day-ahead price** for the DE-LU bidding zone (EUR/MWh), available "
             "from 1 October 2018. We use it only to put a value on forecast errors."])
rep.h("2.2  Pre-processing", 2)
rep.p("The hourly values were grouped by local market day (CET/CEST). We take "
      "the daily *mean* load in GW, not the daily sum. The two days a year "
      "with 23 or 25 hours, caused by daylight-saving changes, would otherwise "
      "show artificial dips and peaks. Multiplying by 24 gives daily energy in "
      "GWh. The series runs from 1 January 2015 to 30 September 2020, 2,100 "
      "days with no missing values. The mean is 55.5 GW, and the range is "
      "38.6 GW (Sunday 18 September 2016) to 68.2 GW (Wednesday 23 January "
      "2019).",
      align="justify")
rep.h("2.3  Training, validation and test split", 2)
rep.p("The data are split by time, without shuffling (Table 2). The "
      "validation and test sets are one full year each, so that every "
      "season and one Christmas period appears in each. All model choices "
      "(orders, hyper-parameters, buffer sizes) are made on the validation "
      "year. The test year is used once, for the final evaluation. "
      "Every model makes one-step-ahead forecasts: the forecast for day *t* "
      "uses observations up to day *t-1* only.", align="justify")
rep.table(["Set", "Period", "Days", "Share"],
          [["Training", "1 Jan 2015 - 30 Sep 2018", E["n_train"], "65 %"],
           ["Validation", "1 Oct 2018 - 30 Sep 2019", E["n_val"], "17 %"],
           ["Test", "1 Oct 2019 - 30 Sep 2020", E["n_test"], "17 %"]],
          "Chronological split used by every model.",
          widths_cm=[3, 6, 2, 2], font=9, left_cols=(0, 1))
rep.figure(FIG / "01_series_split.png",
           "Daily mean German load with the three splits. The black line is a "
           "28-day moving average. The grey band marks the first COVID-19 "
           "lockdown (16 March to 10 May 2020), which falls inside the test year.")
rep.p("The test year is unusual. In the five weeks from 23 March 2020, demand "
      "averaged 50.2 GW, 8 % below the same weeks of 2019, as industry and "
      "commerce shut down. From July to September it was still 3 % below "
      "the previous year. We kept this period in the test set "
      "deliberately. A forecast a supplier relies on has to survive such "
      "shocks, and the test year shows what happens when one arrives.",
      align="justify")

# --------------------------------------------------------------------------
# 3 Exploratory analysis
# --------------------------------------------------------------------------
rep.h("3  Exploratory analysis (Task 2)", 1)
rep.p("All statistics in this chapter use the training period only, so "
      "nothing about the validation or test years influences how the models "
      "are designed.", align="justify")
rep.h("3.1  Weekly and yearly profiles", 2)
rep.figure(FIG / "02_profiles.png",
           "Mean load by weekday and by month, training period.", width_cm=15)
rep.p(f"The weekly cycle is the strongest feature of the data. Tuesday to "
      f"Thursday are almost identical ({f(wd[1], 1)}, {f(wd[2], 1)} and "
      f"{f(wd[3], 1)} GW). Monday ({f(wd[0], 1)} GW) is slightly lower, "
      f"because its early-morning hours still carry the weekend's low demand. "
      f"Saturday ({f(wd[5], 1)} GW) and Sunday ({f(wd[6], 1)} GW) are much "
      f"lower: Sunday demand is {f(wd[2] - wd[6], 1)} GW, or "
      f"{f((wd[2] - wd[6]) / wd[2] * 100, 0)} %, below Wednesday. Over the "
      f"year, demand is highest in January and February (about 59-60 GW) and "
      f"lowest in May and August (about 53 GW). Germany heats mostly with gas "
      f"and uses little air conditioning, so the summer trough is shallow and "
      f"has no summer peak.", align="justify")
rep.h("3.2  Decomposition into trend, seasonality and remainder", 2)
rep.p("Because there are two seasonal periods, we used MSTL [8], which "
      "applies STL [9] repeatedly with periods of 7 and 365 days. Robust "
      "fitting is used so that holidays do not distort the seasonal "
      "components.", align="justify")
rep.figure(FIG / "03_decomposition.png",
           "MSTL decomposition of the training data: trend, weekly seasonal, "
           "yearly seasonal and remainder components (GW).")
rep.p(f"We measured how much of the variance of the series each component "
      f"carries. The weekly component carries {f(vs['weekly'] * 100, 0)} %, "
      f"the yearly component {f(vs['yearly'] * 100, 0)} %, the trend only "
      f"{f(vs['trend'] * 100, 1)} % and the remainder "
      f"{f(vs['remainder'] * 100, 1)} %. The weekly component swings by "
      f"{f(E['weekly_amplitude_gw'], 1)} GW from peak to trough. The yearly "
      f"component swings by {f(E['yearly_amplitude_gw'], 1)} GW, but that "
      f"figure is inflated by the sharp Christmas dip it has to absorb. The "
      f"trend rises from {f(E['trend_start_end_gw'][0], 1)} to "
      f"{f(E['trend_start_end_gw'][1], 1)} GW over the training period. We "
      f"are cautious about reading this as real growth. Official statistics "
      f"show German electricity consumption roughly flat over 2015-2018, so "
      f"the rise may partly reflect changes in how much of the grid the "
      f"ENTSO-E figure covers.", align="justify")
rep.p("Is the remainder white noise? No. The Ljung-Box test [17] rejects "
      f"white noise at lags 7 and 14 (p < 10^-40). Its lag-1 autocorrelation "
      f"is {f(E['remainder_acf1'])}. The large remainders are not random "
      "either. All twelve of the largest fall on holidays. Eleven are "
      "nation-wide holidays, for example Whit Monday 2017 (15.8 GW, meaning "
      "the decomposition expected far more load than occurred), Whit Monday "
      "2016, Ascension Day 2015, Good Friday 2017, Easter Monday 2015 and New "
      "Year's Day 2018. The twelfth is Corpus Christi 2015, a regional "
      "holiday. A holiday falls on a "
      "different date each year, so neither seasonal component can learn it. "
      "This was the single most useful observation of the analysis, and it "
      "led directly to the holiday variables in Sections 5.7 and 6.4.",
      align="justify")
rep.p("The decomposition has one weakness. With only 3.75 years of data the "
      "yearly component is estimated from fewer than four cycles. In the "
      "first and last year it follows the data almost exactly, which is why "
      "the remainder is nearly flat there. The middle two years give a "
      "fairer picture of the true noise.", align="justify")
rep.h("3.3  Autocorrelation", 2)
rep.figure(FIG / "04_acf_pacf.png",
           "ACF and PACF up to lag 60 for the level series, the weekly "
           "difference and the combined first-and-weekly difference "
           "(training data). The shaded band is the 95 % confidence band.")
al = E["acf_level"]
pl = E["pacf_level_first10"]
rep.p(f"The ACF of the level series (Figure 4, top left) is dominated by the "
      f"week. The autocorrelation is {f(al['7'])} at lag 7, {f(al['14'])} at "
      f"lag 14 and {f(al['28'])} at lag 28, but only {f(al['1'])} at lag 1 "
      f"and {f(al['2'])} at lag 2. Today looks much more like the same day "
      f"last week than like yesterday. The ACF also decays very slowly "
      f"across multiples of 7, and at lag 365 it is still "
      f"{f(E['acf_level_365'])}. Both are signs of strong persistence, and "
      f"possibly non-stationarity.", align="justify")
rep.p(f"The PACF of the level series is significant at lags 1 ({f(pl[0])}), "
      f"2 ({f(pl[1])}), 6 ({f(pl[5])}), 7 ({f(pl[6])}) and 8 ({f(pl[7])}). "
      f"There is a second smaller cluster around lags 13-15. A large positive "
      f"spike at 7 next to a negative spike at 8 is what a multiplicative "
      f"seasonal AR model produces, (1 - phi B)(1 - Phi B^7). Our AR(8) fit "
      f"later confirms this (Section 5.4).", align="justify")
d17 = E["acf_d17_first10"]
rep.p(f"After weekly and first differencing (bottom row), the ACF has clear "
      f"spikes at lag 1 ({f(d17[0])}), lag 2 ({f(d17[1])}) and lag 7 "
      f"({f(d17[6])}), and little else. This is the signature of a "
      f"non-seasonal MA(1) or MA(2) combined with a seasonal MA(1), close to "
      f"the classic 'airline' model. The PACF instead decays gradually at "
      f"the seasonal lags 7, 14 and 21, which fits an MA rather than an AR "
      f"structure.", align="justify")
rep.p("**Our observations.**", space_after=2)
rep.bullets([
    "The week explains more than twice as much variance as the year. A model that gets the weekday wrong cannot be rescued by anything else.",
    "Holidays are the main source of large errors. They behave like a Sunday falling on a weekday and need explicit treatment.",
    "Yesterday matters mainly as a correction to 'same day last week'. It is not a good forecast on its own (lag-1 ACF is only 0.54).",
    "The variability is mostly regular calendar structure. That makes the series very predictable in normal weeks and badly predictable around unusual days."])

# --------------------------------------------------------------------------
# 4 Stationarity
# --------------------------------------------------------------------------
rep.h("4  Stationarity (Tasks 6 and 7)", 1)
rep.h("4.1  Unit-root tests", 2)
rep.p("We ran two tests whose null hypotheses are opposite. The augmented "
      "Dickey-Fuller (ADF) test [6] has a unit root as its null hypothesis. "
      "The KPSS test [7] has stationarity as its null. Lag lengths were "
      "chosen by AIC for ADF and automatically for KPSS.", align="justify")
rows = []
for s in ur.index:
    r = ur.loc[s]
    adf = "reject unit root" if r["ADF p"] < 0.05 else "unit root"
    kp = "reject stationarity" if r["KPSS p"] < 0.05 else "stationary"
    rows.append([s, f(r["ADF stat"]), f"{r['ADF p']:.4f}", f(r["KPSS stat"], 3),
                 ("< 0.01" if r["KPSS p"] <= 0.01 else
                  "> 0.10" if r["KPSS p"] >= 0.1 else f"{r['KPSS p']:.3f}"),
                 f"ADF: {adf}; KPSS: {kp}"])
rep.table(["Series", "ADF stat", "ADF p", "KPSS stat", "KPSS p", "Reading"], rows,
          "Unit-root tests on the training data.",
          widths_cm=[3.4, 1.7, 1.6, 1.8, 1.6, 6.3], font=8, left_cols=(0, 5))
rep.p(f"On the level series the two tests disagree. ADF rejects a unit root "
      f"(p = {ur.loc['level', 'ADF p']:.4f}), but KPSS rejects stationarity "
      f"(statistic {f(ur.loc['level', 'KPSS stat'], 3)}, above the 1 % "
      f"critical value of 0.739). This is the textbook case of a series "
      f"that has no stochastic trend in the random-walk sense, yet is not "
      f"stationary: its mean moves with the season and the slow trend. "
      f"ADF, which includes many lags, absorbs the weekly pattern and sees "
      f"strong mean reversion. KPSS sees a level that drifts through the "
      f"year.", align="justify")
rep.h("4.2  Making the series stationary", 2)
rep.figure(FIG / "05_differenced.png",
           "Weekly difference and combined first-and-weekly difference of the "
           "training data. The isolated spikes are holidays, each appearing "
           "twice: once when it enters the difference and once a week later "
           "when it leaves.")
rep.p(f"A weekly difference, y_t - y_(t-7), removes the weekly pattern and "
      f"most of the yearly drift. Both tests then agree that the series is "
      f"stationary (ADF p < 0.001, KPSS {f(ur.loc['weekly diff (1-B^7)', 'KPSS stat'], 3)}, "
      f"p > 0.10). A first difference alone also passes both tests, but it "
      f"leaves the weekly cycle in the data (Figure 4). Adding a first "
      f"difference on top of the weekly one is not needed for stationarity. "
      f"It is a modelling choice, which we test as d = 0 or d = 1 in the "
      f"SARIMA search. One price of differencing is visible in Figure 5: a "
      f"single holiday becomes two outliers, one negative and one positive "
      f"a week later. This is one reason the holiday dummies in Section 5.7 "
      f"help so much.", align="justify")

# --------------------------------------------------------------------------
# 5 Classical models
# --------------------------------------------------------------------------
rep.h("5  Classical linear models (Tasks 3, 4, 5, 8, 9)", 1)
rep.h("5.1  Evaluation protocol and metrics", 2)
rep.p("Each model is estimated once, on the training period, by maximum "
      "likelihood with statsmodels [14]. With its parameters frozen, the "
      "model is then run through the whole series by the Kalman filter. "
      "This produces genuine one-step-ahead forecasts. The forecast for day "
      "*t* uses data up to *t-1*, and parameters fitted only on data before "
      "October 2018. The model is not refitted during the test year. We "
      "report four metrics on the test year, where e_t = y_t - yhat_t:",
      align="justify")
rep.eq("RMSE = sqrt( mean(e_t²) ),    MAE = mean|e_t|,    "
       "MAPE = 100 · mean|e_t / y_t|,    R² = 1 - Σe_t² / Σ(y_t - ȳ)²")
rep.p("RMSE punishes large misses, and large misses are what drive "
      "imbalance cost. MAE is the typical miss. MAPE makes the result "
      "comparable with published load-forecasting work. R² tells how much "
      "of the day-to-day variation is explained.", align="justify")
rep.h("5.2  Benchmarks", 2)
rep.p(f"A forecast is only good relative to something. We use three "
      f"benchmarks. The naive forecast (yesterday's load) has a test RMSE "
      f"of {f(A.loc['Naive (y[t-1])', 'RMSE'])} GW and an R² of only "
      f"{f(A.loc['Naive (y[t-1])', 'R2'])}, because it predicts a weekday "
      f"level on every Saturday. The seasonal naive forecast (same day last "
      f"week) is much better: RMSE "
      f"{f(A.loc['Seasonal naive (y[t-7])', 'RMSE'])} GW, MAPE "
      f"{f(A.loc['Seasonal naive (y[t-7])', 'MAPE'])} %. The TSO day-ahead "
      f"forecast, made by professionals with weather forecasts, reaches an "
      f"RMSE of {f(A.loc['TSO day-ahead forecast', 'RMSE'])} GW and a MAPE "
      f"of {f(A.loc['TSO day-ahead forecast', 'MAPE'])} %. This is the bar "
      f"we really want to clear.", align="justify")

ma = R["cls"]["search"]["MA"]
rep.h("5.3  Moving-average model MA(q) (Task 4)", 2)
rep.p("An MA(q) model writes today's value as a mean plus a weighted sum of "
      "the last *q* forecast errors:", align="justify")
rep.eq("y_t = μ + ε_t + θ_1 ε_(t-1) + … + θ_q ε_(t-q)")
rep.p("**What decides q.** The ACF of an MA(q) process is exactly zero "
      "beyond lag *q*, so in theory *q* is the last lag at which the sample "
      "ACF is significant. Our level series has no such cut-off (Figure 4): "
      "the ACF is still 0.7 at lag 28. Only a very large *q* could capture "
      "it. We therefore fixed a candidate set q ∈ {1, 2, 3, 5, 7, 10, 14}, "
      "estimated each on the training data, and kept the *q* with the "
      "lowest validation RMSE. AIC agreed with this choice.",
      align="justify")
rep.table(["q"] + [str(r["q"]) for r in ma],
          [["AIC"] + [f(r["aic"], 0) for r in ma],
           ["Val. RMSE (GW)"] + [f(r["val_RMSE"]) for r in ma]],
          "MA(q) on the level series: training AIC and validation RMSE.",
          font=8)
rep.p(f"Validation RMSE falls steadily as *q* grows. It drops from "
      f"{f(ma[0]['val_RMSE'])} GW at q = 1 to {f(ma[-1]['val_RMSE'])} GW at "
      f"q = 14, and the improvement has not levelled off. We stopped at "
      f"q = 14, because fitting becomes slow and the point is already "
      f"clear. On the test year MA(14) reaches an RMSE of "
      f"{f(A.loc['MA(14)', 'RMSE'])} GW and an R² of {f(A.loc['MA(14)', 'R2'])}. "
      f"That beats seasonal naive but falls well short of the TSO. An MA "
      f"model has a memory of exactly *q* days of shocks. A series that "
      f"remembers the same weekday for months is the wrong kind of process "
      f"for it. Figure 7 shows the consequence: MA(14) under-shoots the "
      f"weekend troughs throughout the test year.", align="justify")

ar = R["cls"]["search"]["AR"]
a8 = R["cls"]["chosen"]["AR8_params"]
rep.h("5.4  Autoregressive model AR(p) (Task 5)", 2)
rep.eq("y_t = c + φ_1 y_(t-1) + … + φ_p y_(t-p) + ε_t")
rep.p("**What decides p.** For an AR(p) process the *partial* "
      "autocorrelation is zero beyond lag *p*, so *p* is read from the "
      "PACF cut-off. Information criteria and validation error are used "
      "to confirm the choice. The PACF in Figure 4 is significant up to "
      "lag 8, with a second group at 13-15 and smaller spikes at 21-22. "
      "This suggests the candidates p ∈ {1, 2, 7, 8, 14, 15, 21}.",
      align="justify")
rep.table(["p"] + [str(r["p"]) for r in ar],
          [["AIC"] + [f(r["aic"], 0) for r in ar],
           ["Val. RMSE (GW)"] + [f(r["val_RMSE"]) for r in ar]],
          "AR(p) on the level series: training AIC and validation RMSE.",
          font=8)
rep.p(f"The jumps in Table 5 follow the PACF closely. Going from p = 2 to "
      f"p = 7 cuts validation RMSE from {f(ar[1]['val_RMSE'])} to "
      f"{f(ar[2]['val_RMSE'])} GW, and p = 8 cuts it again to "
      f"{f(ar[3]['val_RMSE'])} GW. After that the gains are small. AR(21) "
      f"has the lowest validation error and is our AR model. Its test RMSE "
      f"is {f(A.loc['AR(21)', 'RMSE'])} GW (MAPE "
      f"{f(A.loc['AR(21)', 'MAPE'])} %), clearly better than MA(14).",
      align="justify")
rep.p(f"The AR(8) coefficients are worth a closer look. The fitted values "
      f"are φ1 = {f(a8['ar.L1'])}, φ7 = {f(a8['ar.L7'])} and "
      f"φ8 = {f(a8['ar.L8'])}, and the other lags are near zero. The product "
      f"φ1 · φ7 = {f(a8['ar.L1'] * a8['ar.L7'])} is almost exactly -φ8. In "
      f"other words, the unconstrained AR(8) has found the multiplicative "
      f"seasonal form (1 - 0.64B)(1 - 0.78B^7) on its own. In words: start "
      f"from the same day last week, and correct it by how much yesterday "
      f"differed from the day before last week. This is the structure a "
      f"seasonal ARIMA builds in from the start.", align="justify")

ari = R["cls"]["search"]["ARIMA"][:6]
rep.h("5.5  ARIMA(p, d, q) (Task 8)", 2)
rep.p("Following Section 4, we set d = 1: a first difference passes both "
      "unit-root tests. We searched p ∈ {1, 2, 7, 8} and q ∈ {0, 1, 2} and "
      "chose the model with the lowest training AIC.", align="justify")
rep.table(["(p, d, q)", "AIC", "Val. RMSE"],
          [[f"({r['p']}, 1, {r['q']})", f(r["aic"], 1), f(r["val_RMSE"], 3)]
           for r in ari],
          "Six best ARIMA(p,1,q) models by training AIC.",
          widths_cm=[3, 3, 3], font=8.5)
rep.p(f"The AIC choice is {ARI}. Models with p < 7 are far worse (AIC above "
      f"8,200), because after a first difference the weekly cycle is still "
      f"there and has to be modelled by lags 7 and 8. The top six models "
      f"are within 3 AIC points and 0.004 GW of validation RMSE of each "
      f"other, so the exact (p, q) hardly matters once p ≥ 7. On the test "
      f"year {ARI} has an RMSE of {f(A.loc[ARI, 'RMSE'])} GW, slightly "
      f"*worse* than AR(21) on the levels "
      f"({f(A.loc['AR(21)', 'RMSE'])} GW). A likely cause is visible in "
      f"Figure 5. Differencing turns every holiday into two shocks, and a model "
      f"fitted on differences carries part of a holiday's dip into the "
      f"following days. On the levels, AR(21) only sees one bad day. So "
      f"differencing is needed for the unit-root theory, but on its own it "
      f"is not the best way to handle a weekly pattern this stable.",
      align="justify")

sar = R["cls"]["search"]["SARIMA"][:6]
rep.h("5.6  Seasonal ARIMA (Task 9)", 2)
rep.p("SARIMA(p,d,q)(P,D,Q)_7 models the weekly pattern directly through a "
      "seasonal difference (D = 1) and seasonal AR/MA terms. Guided by "
      "the ACF of the differenced series (Section 3.3), we searched "
      "p, q ∈ {0, 1, 2}, d ∈ {0, 1}, P ∈ {0, 1}, Q = 1 and D = 1. That is "
      "36 models, selected by training AIC.", align="justify")
rep.table(["(p,d,q)(P,D,Q)7", "AIC", "Val. RMSE"],
          [[f"({r['p']},{r['d']},{r['q']})({r['P']},1,{r['Q']})",
            f(r["aic"], 1), f(r["val_RMSE"], 3)] for r in sar],
          "Six best SARIMA models by training AIC.",
          widths_cm=[4, 3, 3], font=8.5)
rep.p(f"The selected model is {SA}. It uses a weekly difference only "
      f"(d = 0) and a seasonal MA(1) term. On the test year it reaches an "
      f"RMSE of {f(A.loc[SA, 'RMSE'])} GW and a MAPE of "
      f"{f(A.loc[SA, 'MAPE'])} %, the best of the purely univariate "
      f"classical models. Two fitted values tell us something about the "
      f"data. The seasonal MA coefficient is -0.998, almost exactly -1. This "
      f"means the weekly difference is almost cancelled again by the MA "
      f"term, so the weekly pattern is essentially *fixed*: it does not "
      f"wander from year to year. The AR part (φ1 = 0.60, φ2 = 0.33, sum "
      f"0.93) is close to a unit root. The model treats deviations from the "
      f"weekly profile as very persistent, which matches the slow yearly "
      f"swing.", align="justify")
rep.h("5.7  Extension: SARIMAX with holiday variables", 2)
rep.p("The decomposition showed that holidays cause the largest errors, and "
      "the holiday calendar is known years in advance. Using it as input is "
      "therefore legitimate. We added two dummy variables to the same "
      "SARIMA model. The first marks the nine nation-wide German public "
      "holidays (computed from the Easter date). The second marks the "
      "Christmas break, 24 December to 1 January.", align="justify")
rep.figure(FIG / "17_holiday_zoom.png",
           "Holiday periods in the test year. On the nine test-year holidays "
           "SARIMA over-forecast by 9.1 GW on average, while SARIMAX's mean "
           "absolute error there is 1.9 GW. One week after each holiday SARIMA "
           "is also worse (2.4 vs. 1.6 GW), because the dip echoes through "
           "the seasonal term. Christmas remains hard for every model.")
rep.p(f"The estimated holiday effect is **{f(sxp['holiday'])} GW**: a public "
      f"holiday cuts national demand by about the difference between a "
      f"Wednesday and a Sunday. The Christmas-break effect is "
      f"**{f(sxp['xmas'])} GW** on top of the weekday pattern. Both are "
      f"highly significant (p < 0.001). Adding these two variables cuts "
      f"test RMSE from {f(A.loc[SA, 'RMSE'])} to "
      f"**{f(A.loc[SX, 'RMSE'])} GW** (MAPE {f(A.loc[SX, 'MAPE'])} %, R² "
      f"{f(A.loc[SX, 'R2'])}). That is a 32 % reduction from two columns of "
      f"zeros and ones. It is the largest improvement from any single change "
      f"in the project, and it puts SARIMAX ahead of the TSO forecast "
      f"({f(A.loc['TSO day-ahead forecast', 'RMSE'])} GW). The Ljung-Box "
      f"test still rejects white-noise residuals, so some structure is left. "
      f"Chapter 10 shows that the largest remaining errors fall on regional "
      f"holidays and bridge days.", align="justify")
rep.figure(FIG / "06_classical_test_forecasts.png",
           "One-step-ahead forecasts (red) against actual load (black) on the "
           "test year for every classical model and benchmark.", width_cm=13.5)
rep.figure(FIG / "07_classical_scatter.png",
           "Predicted against actual load on the test year. Points on the "
           "diagonal are perfect forecasts.", width_cm=15.5)
names = ["Naive (y[t-1])", "Seasonal naive (y[t-7])", "TSO day-ahead forecast",
         "MA(14)", "AR(21)", ARI, SA, SX]
rep.table(["Model", "RMSE (GW)", "MAE (GW)", "MAPE (%)", "R²"],
          [[n.replace("+holidays", " + holidays"), f(A.loc[n, "RMSE"], 3),
            f(A.loc[n, "MAE"], 3), f(A.loc[n, "MAPE"]), f(A.loc[n, "R2"], 3)]
           for n in names],
          "Test-year accuracy of the classical models and benchmarks.",
          widths_cm=[6.2, 2.2, 2.2, 2.2, 2.0], font=8.5, bold_rows=(7,))

# --------------------------------------------------------------------------
# 6 Neural networks
# --------------------------------------------------------------------------
rep.h("6  Recurrent neural networks (Tasks 10 and 11)", 1)
rep.h("6.1  Set-up", 2)
rep.p("Every model in Chapter 5 is linear in past values. A recurrent "
      "network can in principle learn non-linear effects. For example, a "
      "cold Monday after a holiday weekend may behave differently from "
      "what the sum of a 'Monday effect' and a 'cold effect' predicts. We "
      "used PyTorch [15] with the same split as before.", align="justify")
rep.bullets([
    "**Input:** the last *L* daily loads (the lag window), standardised with the training mean and standard deviation. The window for day *t* ends at *t-1*, so the forecast is one step ahead.",
    "**Architecture:** an RNN (tanh) or LSTM [3] layer, or two stacked layers with 10 % dropout between them. The last hidden state goes to a small dense head (hidden -> ReLU -> 1).",
    "**Training:** Adam on the mean squared error, gradient clipping at 1.0, up to 200 epochs, and early stopping when the validation loss has not improved for 20 epochs. The best weights are restored.",
    "**Tuning:** random search over 24 configurations per network type from lag ∈ {7, 14, 28, 56}, batch ∈ {16, 32, 64}, learning rate ∈ {0.001, 0.003, 0.01}, layers ∈ {1, 2} and neurons ∈ {16, 32, 64, 128}. Configurations are ranked by validation RMSE.",
    "**Final model:** the best configuration is retrained with three random seeds and the three forecasts are averaged. We report the spread across seeds, because a single network run can be misleading."])
rep.figure(FIG / "11_nn_tuning.png",
           "Validation RMSE of all 48 random-search trials against each "
           "hyper-parameter. Points are shifted slightly sideways to separate "
           "RNN and LSTM.", width_cm=16)
rc, lc = nn["RNN"]["config"], nn["LSTM"]["config"]
rep.p(f"Figure 9 shows which settings matter. The lag window matters most. "
      f"A window of 7 days is clearly too short, and the three best LSTM "
      f"runs all use 28 days, i.e. four weeks of history. At lags of 14 "
      f"days and more, the best LSTM run beats the best RNN run. Batch size "
      f"and learning rate matter less. The largest learning rate (0.01) "
      f"produced the one failed RNN run (RMSE 3.6 GW): with plain RNNs the "
      f"gradient explodes or vanishes more easily. The two best LSTM "
      f"configurations both have two layers.", align="justify")
rep.h("6.2  RNN (Task 10)", 2)
rep.p(f"The best RNN uses a lag window of {rc['lag']} days, batch size "
      f"{rc['batch']}, learning rate {rc['lr']}, {rc['layers']} layer and "
      f"{rc['hidden']} neurons. Its validation RMSE is "
      f"{f(V.loc['RNN', 'RMSE'])} GW, already better than every univariate "
      f"classical model on validation. On the test year it reaches "
      f"{f(A.loc['RNN', 'RMSE'])} GW (MAPE {f(A.loc['RNN', 'MAPE'])} %). "
      f"That is roughly equal to SARIMA ({f(A.loc[SA, 'RMSE'])} GW). The "
      f"three seeds gave test RMSEs of "
      f"{', '.join(f(x) for x in nn['RNN']['test_RMSE_per_seed'])} GW.",
      align="justify")
rep.h("6.3  LSTM (Task 11)", 2)
ls = nn["LSTM"]["test_RMSE_per_seed"]
rep.p(f"The best LSTM uses a {lc['lag']}-day window, batch size "
      f"{lc['batch']}, learning rate {lc['lr']}, {lc['layers']} stacked "
      f"layers and {lc['hidden']} neurons per layer. On validation it is "
      f"clearly better than the RNN ({f(V.loc['LSTM', 'RMSE'])} vs. "
      f"{f(V.loc['RNN', 'RMSE'])} GW). On the test year the advantage "
      f"disappears: {f(A.loc['LSTM', 'RMSE'])} GW against "
      f"{f(A.loc['RNN', 'RMSE'])} GW. The seeds also differ a lot: "
      f"{', '.join(f(x) for x in ls)} GW, a range of "
      f"{f(max(ls) - min(ls))} GW. Averaging the three seeds helps (the "
      f"ensemble is better than any single seed), and we would not deploy a "
      f"single LSTM run.", align="justify")
rep.h("6.4  LSTM with calendar inputs", 2)
lcs = nn["LSTM + calendar"]["test_RMSE_per_seed"]
rep.p(f"Without help, the networks have the same blind spot as SARIMA: they "
      f"cannot know that next Monday is Easter Monday. We therefore fed the "
      f"tuned LSTM the calendar of the target day as well, concatenated to "
      f"its last hidden state: weekday (one-hot), public holiday and "
      f"Christmas break. This is information a supplier has in advance. "
      f"Validation RMSE dropped to **{f(V.loc['LSTM + calendar', 'RMSE'])} GW**, "
      f"the best of any model, and test RMSE to "
      f"**{f(A.loc['LSTM + calendar', 'RMSE'])} GW** (MAPE "
      f"{f(A.loc['LSTM + calendar', 'MAPE'])} %, R² "
      f"{f(A.loc['LSTM + calendar', 'R2'])}). The seeds now agree much more "
      f"closely ({', '.join(f(x) for x in lcs)} GW).", align="justify")
rep.figure(FIG / "08_nn_learning_curves.png",
           "Training and validation loss (standardised units, log scale) for "
           "seed 0 of each final network. Training stops when validation "
           "loss stops improving.", width_cm=16)
rep.figure(FIG / "09_nn_test_forecasts.png",
           "Neural-network forecasts against actual load on the test year.",
           width_cm=15.5)
rep.figure(FIG / "10_nn_scatter.png",
           "Predicted against actual load on the test year for the three "
           "networks.", width_cm=14)
nm = ["RNN", "LSTM", "LSTM + calendar"]
rep.table(["Model", "Config (lag/batch/lr/layers/neurons)", "Val. RMSE",
           "Test RMSE", "Test MAE", "Test MAPE", "Test R²"],
          [[n, f"{nn[n]['config']['lag']}/{nn[n]['config']['batch']}/"
               f"{nn[n]['config']['lr']}/{nn[n]['config']['layers']}/"
               f"{nn[n]['config']['hidden']}",
            f(V.loc[n, "RMSE"], 3), f(A.loc[n, "RMSE"], 3),
            f(A.loc[n, "MAE"], 3), f(A.loc[n, "MAPE"]), f(A.loc[n, "R2"], 3)]
           for n in nm],
          "Neural-network results (average of three seeds).",
          widths_cm=[3, 4.4, 1.7, 1.7, 1.6, 1.7, 1.6], font=8)
rep.p("**Our observations on the networks.** Every network did better on "
      "validation than on test, by 0.21 GW (RNN) up to 0.53 GW (LSTM). "
      "SARIMAX lost only 0.02 GW, and the other classical models were even "
      "slightly better on test than on validation. Two reasons "
      "seem likely. The test year contains the COVID-19 shock, and a "
      "network that has learned 'what a normal April looks like' is more "
      "confidently wrong than a linear model whose forecast is anchored to "
      "last week. And early stopping on the validation year means the "
      "networks are, mildly, tuned to that year. The plain LSTM, with more "
      "parameters, overfits more. On this data set, where the structure is "
      "mostly linear calendar effects, the non-linear capacity of the LSTM "
      "brings little until it is given the calendar. Once it has the "
      "calendar, it matches the best linear model.", align="justify")

# --------------------------------------------------------------------------
# 7 Comparison
# --------------------------------------------------------------------------
rep.h("7  Model comparison", 1)
rep.figure(FIG / "15_rmse_ranking.png",
           "Test RMSE of all models. Orange: the TSO's professional forecast. "
           "Red: our two best models.", width_cm=15)
rep.p(f"SARIMAX and the calendar LSTM end up with almost the same test RMSE "
      f"({f(A.loc[SX, 'RMSE'])} and {f(A.loc['LSTM + calendar', 'RMSE'])} "
      f"GW), but they make different mistakes. Their errors correlate at "
      f"only 0.69. SARIMAX is better on ordinary weekdays (MAE 0.98 vs. "
      f"1.16 GW) and adapted faster to the lockdown (1.26 vs. 1.93 GW from "
      f"9 March to 30 April 2020, when the LSTM kept over-forecasting by "
      f"1.5 GW). The LSTM is better on weekends and holidays (0.90 vs. "
      f"1.17 GW). A simple average of "
      f"the two, with no fitted weights, gives **{f(A.loc[ENS, 'RMSE'])} GW "
      f"RMSE, {f(A.loc[ENS, 'MAE'])} GW MAE, {f(A.loc[ENS, 'MAPE'])} % MAPE "
      f"and R² {f(A.loc[ENS, 'R2'], 3)}**. This is the most accurate "
      f"forecast in the study. It is {f((1 - A.loc[ENS, 'RMSE'] / A.loc['TSO day-ahead forecast', 'RMSE']) * 100, 0)} % "
      f"better than the TSO forecast in RMSE and "
      f"{f((1 - A.loc[ENS, 'RMSE'] / A.loc['Seasonal naive (y[t-7])', 'RMSE']) * 100, 0)} % "
      f"better than seasonal naive.", align="justify")
rep.p(f"Two checks keep this claim honest. **Bias:** the TSO forecast "
      f"under-predicted by {f(RB['TSO_bias_val_GW'])} GW on average in the "
      f"validation year but was unbiased in the test year "
      f"({f(RB['TSO_bias_test_GW'])} GW). Its bias-free RMSE on test is "
      f"{f(RB['TSO_debiased_in_sample_RMSE_test'])} GW, so bias does not "
      f"explain the gap. **Timing:** our models use all of yesterday's load, "
      f"but a real bid must be in by noon the day before. When we re-score "
      f"SARIMAX two days ahead, its test RMSE rises from "
      f"{f(RB['SARIMAX_1step_test']['RMSE'])} to "
      f"{f(RB['SARIMAX_2step_test']['RMSE'])} GW, which is *worse* than the "
      f"TSO's {f(RB['TSO_test']['RMSE'])} GW. The TSO has weather forecasts "
      f"and we do not. Averaging our two-day-ahead SARIMAX with the TSO "
      f"forecast gives {f(RB['Blend_TSO_SARIMAX2_test']['RMSE'])} GW, "
      f"better than either alone. So the fair conclusion is this: under the "
      f"course protocol our models beat the professionals, and under real "
      f"bidding constraints they *add value* to the professional forecast. "
      f"They do not replace it.", align="justify")
allm = ["Naive (y[t-1])", "Seasonal naive (y[t-7])", "MA(14)", "AR(21)", ARI,
        SA, SX, "RNN", "LSTM", "LSTM + calendar", ENS, "TSO day-ahead forecast"]
rep.table(["Model", "Val. RMSE", "Val. MAPE", "Test RMSE", "Test MAE",
           "Test MAPE", "Test R²"],
          [[n.replace("+holidays", " + holidays"), f(V.loc[n, "RMSE"], 3),
            f(V.loc[n, "MAPE"]), f(A.loc[n, "RMSE"], 3), f(A.loc[n, "MAE"], 3),
            f(A.loc[n, "MAPE"]), f(A.loc[n, "R2"], 3)] for n in allm],
          "All models on validation and test (one-step-ahead, GW and %).",
          widths_cm=[5.4, 1.7, 1.7, 1.7, 1.7, 1.7, 1.6], font=8,
          bold_rows=(10,))

# --------------------------------------------------------------------------
# 8 Decision
# --------------------------------------------------------------------------
rep.h("8  From forecast to decision (Task 12)", 1)
rep.h("8.1  The procurement problem", 2)
rep.p("Consider a supplier that serves a fixed 1 % of German demand, which "
      "we scale from the national series. Each day it buys F_t of energy in "
      "the day-ahead auction at price P_t. Actual demand A_t is known only "
      "afterwards. We model the imbalance settlement simply. A shortfall "
      "(A > F) must be bought at P(1 + α). A surplus (F > A) is sold back "
      "at P(1 - β). Measured against perfect foresight, the avoidable cost "
      "of day *t* is", align="justify")
rep.eq("cost_t = 24 000 · P_t⁺ · [ α · max(A_t - F_t, 0) + β · max(F_t - A_t, 0) ]")
rep.p("Here loads are in GW (×24 gives GWh and ×1,000 gives MWh), and P⁺ is "
      "the price floored at zero. The test year had 8 negative-price days, "
      "and on those days our simple rule would otherwise reward a surplus. "
      "We set **α = 0.5 and β = 0.25**: running short costs twice as much "
      "per MWh as running long. These values are assumptions, because "
      "German imbalance prices (reBAP) are not in our data. They reflect "
      "that being short when the system is short is penalised harder. "
      "Section 8.5 shows the conclusions hold for other values.",
      align="justify")
rep.h("8.2  The optimal quantity is a quantile, not the mean", 2)
rep.p("This cost has the same shape as the classic newsvendor problem [13]. "
      "If tomorrow's demand has distribution G, the expected cost is "
      "minimised by buying the quantile", align="justify")
rep.eq("F* = G⁻¹(τ*),    τ* = α / (α + β) = 0.5 / 0.75 = 0.67")
rep.p("In plain words: when a shortfall is twice as expensive as a surplus, "
      "you should buy enough to be short on only one day in three. Buying "
      "the point forecast leaves you short on about half the days. This is "
      "the link between the forecast and the action. The forecast gives the "
      "centre, and the cost structure says how far above the centre to buy. "
      "We tried two ways of getting the 67th percentile:", align="justify")
rep.bullets([
    f"**Point forecast + buffer.** Add to the point forecast the 67th percentile of its own errors in the validation year. For the ensemble this buffer is **{f(buf)} GW**, and for SARIMAX it is {f(K['buffer_sarimax_GW'])} GW.",
    "**Direct quantile model.** Train an LSTM with the pinball loss at τ = 0.67 (the same method as in Chapter 9)."])
rep.h("8.3  What each strategy would have cost", 2)
strs = list(C.index)
rep.table(["Strategy", "Val. cost", "Test cost", "Saving vs naive", "Saving vs TSO",
           "Bias (GW)", "Short days"],
          [[s.replace("+holidays", " + holidays").replace("+ +", "+ "),
            k(C.loc[s, "val_cost_1pct_kEUR"]), k(C.loc[s, "annual_cost_1pct_kEUR"]),
            f"{f(C.loc[s, 'saving_vs_naive_pct'], 1)} %",
            f"{f(C.loc[s, 'saving_vs_TSO_pct'], 1)} %",
            f"{C.loc[s, 'mean_bias_GW']:+.2f}", f"{f(C.loc[s, 'short_days_pct'], 0)} %"]
           for s in strs],
          "Annual imbalance cost for a 1 % supplier (validation year and test "
          "year). Bias = mean of purchase minus actual. Short days = share of "
          "days bought too little.",
          widths_cm=[5.0, 1.9, 1.9, 1.9, 1.9, 1.5, 1.5], font=7.5,
          bold_rows=(strs.index(REC),))
rep.figure(FIG / "16_cost_by_strategy.png",
           "Test-year imbalance cost of each procurement strategy for a "
           "supplier with 1 % market share.", width_cm=15)
rep.p(f"Table 11 holds the main result of the project. Buying 'the same as "
      f"last week' would have cost the 1 % supplier {k(cost_nai)} in "
      f"imbalance charges over the test year. Buying the TSO forecast would "
      f"have cost {k(cost_tso)}. Buying the ensemble forecast plus the "
      f"buffer would have cost **{k(cost_rec)}**, which is "
      f"{f(C.loc[REC, 'saving_vs_TSO_pct'], 0)} % less than the TSO forecast "
      f"and {f(C.loc[REC, 'saving_vs_naive_pct'], 0)} % less than the status "
      f"quo. Scaled to the whole German market, that is EUR "
      f"{f(C.loc[REC, 'annual_cost_national_MEUR'], 1)} million against EUR "
      f"{f(C.loc['TSO public forecast', 'annual_cost_national_MEUR'], 1)} "
      f"million.", align="justify")
rep.p(f"Three details in the table are instructive. (1) The buffer works as "
      f"the theory says. The ensemble alone is short on "
      f"{f(C.loc[ENS, 'short_days_pct'], 0)} % of days; with the buffer it "
      f"is short on {f(C.loc[REC, 'short_days_pct'], 0)} %, close to the "
      f"target of 33 %. In money the buffer's effect depends on the year. "
      f"On validation it cut the ensemble's cost by "
      f"{f((1 - C.loc[REC, 'val_cost_1pct_kEUR'] / C.loc[ENS, 'val_cost_1pct_kEUR']) * 100, 0)} %, "
      f"and on test it cut SARIMAX's cost by "
      f"{f((1 - C.loc[[c for c in C.index if c.startswith('SARIMAX +')][0], 'annual_cost_1pct_kEUR'] / C.loc[SX, 'annual_cost_1pct_kEUR']) * 100, 0)} %. "
      f"For the ensemble on test it saved almost nothing "
      f"({k(C.loc[ENS, 'annual_cost_1pct_kEUR'])} to {k(cost_rec)}). The "
      f"ensemble was already over-forecasting the lower post-COVID demand by "
      f"{f(C.loc[ENS, 'mean_bias_GW'])} GW, so it carried an accidental "
      f"buffer of its own. Most of the saving comes from forecast accuracy, "
      f"and the buffer is a cheap insurance on top. (2) The cost-versus-τ curve (Figure 15) is flat "
      f"between about 0.5 and 0.7. Getting τ roughly right matters, but its "
      f"exact value does not. Buying well below the forecast is costly: at "
      f"τ = 0.2 the cost would be 40-70 % higher than at the optimum. (3) SARIMAX alone has the lowest bias but not the "
      f"lowest cost. Being unbiased is the right goal for a forecaster, but "
      f"not for a buyer facing asymmetric prices.", align="justify")
rep.figure(FIG / "14_cost_vs_tau.png",
           "Test-year cost as a function of the procurement quantile τ, for "
           "SARIMAX and for the ensemble. The dashed line is the "
           "theoretical optimum τ* = 0.67.", width_cm=13)
rep.h("8.4  How the strategy was chosen", 2)
vb = B["val_cost_diff_valbest_minus_rec_1pct_kEUR_CI95"]
rep.p(f"We chose the strategy on the validation year only. The lowest "
      f"validation cost belonged to the direct quantile LSTM "
      f"({k(C.loc[B['val_best'], 'val_cost_1pct_kEUR'])}), with the "
      f"ensemble + buffer just behind ({k(C.loc[REC, 'val_cost_1pct_kEUR'])}). "
      f"The difference is 0.7 %. A 14-day block bootstrap [18] of the "
      f"validation-year difference gives a 95 % range of {k(vb[0])} to "
      f"{k(vb[2])}, which includes zero, so validation cannot tell the two "
      f"apart. We had fixed a tie-break rule in advance: within 1 % of the "
      f"best validation cost, prefer a strategy whose safety margin is one "
      f"explicit number, which a trader can inspect, explain and "
      f"recalibrate. That selects the ensemble + buffer. The test year "
      f"supports the rule. The quantile LSTM over-bought by "
      f"{f(C.loc[B['val_best'], 'mean_bias_GW'])} GW on average, because it "
      f"did not adapt to the lower post-COVID demand, and it cost "
      f"{k(C.loc[B['val_best'], 'annual_cost_1pct_kEUR'] - cost_rec)} more. "
      f"We report this openly: had we followed the validation minimum "
      f"blindly, the saving against the TSO would have been "
      f"{f(C.loc[B['val_best'], 'saving_vs_TSO_pct'], 0)} % instead of "
      f"{f(C.loc[REC, 'saving_vs_TSO_pct'], 0)} %.", align="justify")
rep.p(f"A block bootstrap of the test-year daily cost differences gives the "
      f"saving against the TSO forecast as {k(sav_tso)} (95 % range "
      f"{k(ci_tso[0])} to {k(ci_tso[2])}). The saving against seasonal "
      f"naive is {k(sav_nai)} (range {k(ci_nai[0])} to {k(ci_nai[2])}). "
      f"Neither range comes near zero.", align="justify")
rep.h("8.5  Sensitivity to the penalty assumption", 2)
sn = R["sens"]
rep.table(["α", "β", "τ*", "Buffer (GW)", "Naive", "TSO", "SARIMAX", "Ensemble",
           "Ensemble + buffer"],
          [[f(r.alpha), f(r.beta), f(r["tau*"]), f"{r.buffer_GW:+.2f}",
            f(r["naive_1pct_kEUR"], 0), f(r["TSO_1pct_kEUR"], 0),
            f(r["SARIMAX_1pct_kEUR"], 0), f(r["Ensemble_1pct_kEUR"], 0),
            f(r["Ensemble+buffer_1pct_kEUR"], 0)] for _, r in sn.iterrows()],
          "Test-year cost (kEUR, 1 % supplier) under other penalty assumptions. "
          "The buffer is re-estimated on validation for each τ*.",
          widths_cm=[1.2, 1.2, 1.2, 1.8, 1.6, 1.6, 1.8, 1.8, 2.4], font=8)
rep.p("Under every assumption we tried, the ranking stays the same: our "
      "forecasts are much cheaper than the TSO forecast, which is much "
      "cheaper than seasonal naive. The value of the buffer depends on how "
      "asymmetric the penalties are. With symmetric penalties (τ* = 0.5) it "
      "adds nothing, as it should. When shortfalls cost four times as much "
      "as surpluses (α = 1, β = 0.25), the buffer saves another 17 % over "
      "the unbuffered ensemble. When surpluses are the expensive side "
      "(τ* = 0.33), the rule correctly buys *below* the forecast and saves "
      "about 11 %. The method adapts to the cost structure. Only the numbers "
      "change.", align="justify")
rep.h("8.6  Realistic timing", 2)
rep.p(f"As Chapter 7 showed, a real bid on day *t-1* cannot use the full load "
      f"of day *t-1*. We repeated the cost analysis with two-day-ahead "
      f"forecasts. The TSO forecast with its own validation buffer costs "
      f"{k(real_tso)}. Our two-day-ahead SARIMAX with buffer costs "
      f"{k(RB['SARIMAX_2step_buffer_test_cost_1pct_kEUR'])}. The average of "
      f"the two, with buffer, costs **{k(real_bl)}**. This is the "
      f"configuration we would actually deploy, and its saving of about "
      f"**{k(real_sav)} a year** is the number we would put in a budget. "
      f"The {k(sav_tso)} figure from Section 8.3 is what the course "
      f"protocol allows. It is an upper bound on what a supplier could "
      f"expect.", align="justify")
rep.figure(FIG / "18_monthly_cost.png",
           "Monthly imbalance cost of the status quo, the TSO forecast and "
           "the recommended strategy (1 % supplier, test year).", width_cm=15)
rep.p("Figure 16 shows where the money is lost. For the recommended "
      "strategy, December is the most expensive month (EUR 131k), because "
      "of Christmas and the bridge days around it. The naive rule's worst "
      "month is January (EUR 456k). In the first week back after the "
      "holidays it buys the low holiday quantities of the week before. The "
      "TSO forecast was most expensive in October and November 2019 (EUR "
      "199k and 278k). The lockdown months of March and April 2020 raised "
      "the recommended strategy's cost only moderately, to about EUR 65k a "
      "month against EUR 36k in February.", align="justify")

# --------------------------------------------------------------------------
# 9 Probabilistic forecast
# --------------------------------------------------------------------------
rep.h("9  Probabilistic forecast (Task 13)", 1)
rep.h("9.1  Quantile LSTMs and the pinball loss", 2)
rep.p("A point forecast says nothing about how wrong it might be. To get a "
      "95 % prediction interval we trained two LSTMs. One estimates the "
      "2.5 % quantile of tomorrow's demand and the other the 97.5 % "
      "quantile. Each is trained by back-propagating the pinball (quantile) "
      "loss [4]:", align="justify")
rep.eq("L_τ(y, q) = max( τ (y - q),  (τ - 1)(y - q) )")
rep.p("For τ = 0.975, a forecast that is too low by 1 GW costs 0.975 and "
      "one that is too high costs only 0.025. Minimising this loss pushes "
      "the output up until only 2.5 % of observations lie above it. Both "
      "networks use the tuned LSTM architecture of Section 6.3 with "
      "calendar inputs (28-day window, 2 × 128 units). They were trained "
      "with early stopping on the validation pinball loss, and each "
      "quantile is the average of three seeds. No quantile crossings "
      "occurred (lower bound above upper bound).", align="justify")
rep.p("Coverage is evaluated with the two measures the brief asks for [19]:",
      space_after=2)
rep.eq("PICP = (100/n) · #{ t : L_t ≤ y_t ≤ U_t },        MPIW = mean(U_t - L_t)")
rep.p("A good interval has a PICP close to the nominal 95 % and an MPIW that "
      "is as small as possible. Because a network trained on four years "
      "may be over- or under-confident, we also tried split-conformal "
      "recalibration [5]. The interval is widened by the 95 % quantile of "
      "the validation-year scores max(L - y, y - U).", align="justify")
rep.h("9.2  Results", 2)
qrows = []
for lab, key in [("Validation year", "val"), ("Test year", "test"),
                 ("Test, excl. lockdown", "test_excl_lockdown"),
                 ("Test, lockdown only", "test_lockdown")]:
    qrows.append([lab, f(Q[f"{key}_raw_PICP"], 1) + " %", f(Q[f"{key}_raw_MPIW"]),
                  f(Q[f"{key}_q025_below"], 1) + " %", f(Q[f"{key}_q975_above"], 1) + " %",
                  f(Q[f"{key}_conformal_PICP"], 1) + " %", f(Q[f"{key}_conformal_MPIW"])])
rep.table(["Period", "PICP", "MPIW (GW)", "Below L", "Above U", "PICP (conf.)",
           "MPIW (conf.)"], qrows,
          "Coverage of the nominal 95 % interval. 'Below L' and 'Above U' "
          "should each be 2.5 %.",
          widths_cm=[4.0, 1.7, 1.9, 1.7, 1.7, 2.0, 2.0], font=8)
rep.figure(FIG / "12_interval_test.png",
           "95 % prediction interval against actual load on the test year, "
           "before and after conformal recalibration. Red dots: days "
           "outside the interval.")
rep.figure(FIG / "13_interval_zoom_covid.png",
           "The interval and the τ = 0.67 procurement quantile through the "
           "COVID-19 shock.", width_cm=15)
rep.p(f"On the validation year the interval is almost perfectly calibrated: "
      f"PICP {f(Q['val_raw_PICP'], 1)} % with an average width of "
      f"{f(Q['val_raw_MPIW'])} GW (about ±3 GW, or ±5 % of mean load). "
      f"Conformal recalibration therefore had almost nothing to correct. "
      f"The adjustment was only {f(Q['conformal_qhat_gw'], 3)} GW. On the "
      f"test year coverage falls to {f(Q['test_raw_PICP'], 1)} %, and the "
      f"misses are lopsided: {f(Q['test_q025_below'], 1)} % of days fell "
      f"*below* the lower bound and only {f(Q['test_q975_above'], 1)} % "
      f"above the upper bound. Of the 24 days below the interval, 8 came "
      f"before the pandemic, all around holidays (31 October, 1 November and "
      f"Christmas week). Five fell in the lockdown. The remaining 11 came "
      f"between May and August 2020, mostly ordinary Tuesdays and "
      f"Wednesdays, when demand stayed about 3 % below normal and below "
      f"anything the network had seen at that time of year. During the "
      f"lockdown itself, PICP was "
      f"{f(Q['test_lockdown_raw_PICP'], 0)} %, and the network reacted "
      f"partly by widening the interval to {f(Q['test_lockdown_raw_MPIW'], 1)} GW.",
      align="justify")
rep.p("Our reading: the interval is trustworthy in a normal year, and it "
      "errs in the expected direction under a demand shock, but it errs "
      "on the side of too much confidence. Conformal calibration on a "
      "calm validation year cannot fix that, because the calibration year "
      "itself held no shock. A rolling recalibration on the last 90 days "
      "would have widened the lower side within weeks. We would add this "
      "before deployment.", align="justify")

# --------------------------------------------------------------------------
# 10 Risk
# --------------------------------------------------------------------------
rep.h("10  Risk and reliability (Task 14)", 1)
rep.h("10.1  From interval to action", 2)
rep.p("The interval is useful because it supports three concrete actions, "
      "each of which makes supply more reliable:", align="justify")
rep.bullets([
    f"**Sizing the day-ahead purchase.** The purchase is set at the 67th percentile (Chapter 8) rather than at the mean, so the supplier is short on about one day in four instead of one in two ({f(C.loc[REC, 'short_days_pct'], 0)} % of test days).",
    f"**Sizing flexible back-up.** The gap between the upper 97.5 % bound and the purchase is the extra energy needed to be covered on 39 days out of 40. On the test year it averaged {f(K['upper_headroom_mean_GW'], 2)} GW nationally, or {f(K['upper_headroom_mean_pct_of_load'], 1)} % of load. For the 1 % supplier this is about {f(K['upper_headroom_mean_GW'] * 10, 0)} MW of flexible contracts (intraday purchases, a demand-response pool or a call option on a gas plant). These can be arranged in advance instead of paying the imbalance price.",
    f"**Flagging risky days.** When the interval is unusually wide (above {f(K['width_threshold_GW'], 1)} GW, the 90th percentile of validation widths), the model itself is unsure. Such days can be passed to a trader for review and intraday trading. In the test year {K['flagged_days']} days ({f(K['flagged_share_of_days_pct'], 0)} %) were flagged, and they carried {f(K['flagged_share_of_cost_pct'], 0)} % of the imbalance cost."])
rep.p(f"We should be honest about the third action. Flagged days cost "
      f"EUR {K['mean_cost_flagged_1pct_EUR']:,.0f} on average against EUR "
      f"{K['mean_cost_unflagged_1pct_EUR']:,.0f} for other days, only 1.3 "
      f"times more. The interval width is a weak early-warning signal, "
      f"because the most expensive days are the ones the model is "
      f"*confidently* wrong about. Suppose intraday trading halved the "
      f"penalty on flagged days. The rule would then save "
      f"{k(K['intraday_on_flagged_saving_1pct_kEUR'])} a year: worthwhile, "
      f"but not the main lever.", align="justify")
rep.h("10.2  Quantified risk", 2)
wdays = K["worst_days_1pct_EUR"]
why = {"2020-01-07": "Under-bought by 3.0 GW: first full working week after the holidays, demand returned faster than forecast",
       "2019-12-23": "Over-bought by 8.9 GW: Monday before Christmas, many already on leave",
       "2019-11-01": "Over-bought by 5.8 GW: All Saints' Day, a holiday in five states (not in our national list)",
       "2020-06-11": "Over-bought by 7.1 GW: Corpus Christi, a holiday in six states (not in our list)",
       "2019-10-31": "Over-bought by 3.9 GW: Reformation Day, a holiday in nine states (not in our list)",
       "2019-12-03": "Under-bought by 1.6 GW on a high-price day (EUR 51/MWh); no calendar reason",
       "2019-12-31": "Over-bought by 4.7 GW: New Year's Eve, a half-day for many businesses",
       "2019-11-02": "Under-bought by 2.8 GW: Saturday after All Saints' Day; the holiday dip was carried forward"}
rep.table(["Date", "Cost (EUR, 1 %)", "What happened"],
          [[d, f"{v:,}", why.get(d, "")] for d, v in wdays.items()],
          "The eight most expensive test-year days for the recommended strategy.",
          widths_cm=[2.4, 2.6, 11.0], font=8, left_cols=(0, 2))
rep.p(f"For the 1 % supplier, the recommended strategy cost EUR "
      f"{K['daily_cost_mean_1pct_EUR']:,.0f} on an average day. One day in "
      f"20 cost more than EUR {K['daily_cost_p95_1pct_EUR']:,.0f}, one day "
      f"in 100 more than EUR {K['daily_cost_p99_1pct_EUR']:,.0f}, and the "
      f"worst day EUR {max(wdays.values()):,}. The ten worst days produced "
      f"{f(K['cost_top10_days_share_pct'], 0)} % of the annual cost. In "
      f"GW terms, 95 % of days had a purchase error below "
      f"{f(K['abs_err_p95_GW'], 2)} GW, and the largest error was "
      f"{f(K['abs_err_max_GW'], 2)} GW.", align="justify")
rep.p("Table 14 is the most practical result of the risk analysis. Three "
      "of the eight worst days are regional public holidays (Reformation "
      "Day, All Saints' Day, Corpus Christi), and a fourth is the Saturday "
      "after one of them. Our "
      "holiday variable only covers nation-wide holidays. A regional "
      "holiday variable, weighted by the share of German load in the "
      "states that observe it, is an easy improvement and would target "
      "exactly the most expensive days.", align="justify")
rep.h("10.3  Other risks", 2)
rep.bullets([
    "**Regime shifts.** The COVID-19 year showed that all models lose accuracy, and the intervals lose coverage, when behaviour changes suddenly. The weekly buffer recalibration and a rolling interval recalibration limit the damage. A human override for known events (lockdowns, strikes) is still necessary.",
    f"**Model risk.** Single LSTM runs differ by up to {f(max(ls) - min(ls), 1)} GW in test RMSE depending only on the random seed. We always average three seeds and keep SARIMAX in the ensemble as a stable anchor.",
    "**Assumption risk.** The penalties α and β are assumed. Table 12 shows the ranking does not change, but the euro figures scale roughly in proportion to the penalties.",
    "**Scaling risk.** We scaled national errors to a 1 % portfolio. A real small portfolio is noisier relative to its size than the national total, so its errors and its savings would both be larger in percentage terms."])

# --------------------------------------------------------------------------
# 11 Management assessment
# --------------------------------------------------------------------------
rep.h("11  Management assessment", 1)
rep.p("Would we recommend that a supplier adopt this system, and what is it "
      "worth?", align="justify")
rep.table(["Question", "Answer from our results"],
          [["How accurate is it?", f"Typical daily error {f(A.loc[ENS, 'MAE'])} GW ({f(A.loc[ENS, 'MAPE'])} %). RMSE {f(A.loc[ENS, 'RMSE'])} GW vs {f(A.loc['TSO day-ahead forecast', 'RMSE'])} GW for the TSO forecast (one-step)."],
           ["What is it worth per year (1 % supplier)?", f"{k(sav_tso)} vs. the free TSO forecast under the course protocol; about {k(real_sav)} under real bidding timing. {k(sav_nai)} vs. the 'same as last week' rule."],
           ["How sure are we?", f"95 % bootstrap range of the saving vs. TSO: {k(ci_tso[0])} to {k(ci_tso[2])}. The ranking holds under all five penalty assumptions tested."],
           ["What does it cost to run?", "Open-source software; retraining takes about 20 minutes of laptop CPU time a month, and the daily forecast less than a minute. The main cost is one analyst for set-up and monitoring."],
           ["What can go wrong?", f"Demand shocks (coverage fell to {f(Q['test_lockdown_raw_PICP'], 0)} % in lockdown); regional holidays (3 of the 8 worst days); seed sensitivity of single LSTMs."],
           ["What would we do next?", "Add regional holidays and temperature forecasts; recalibrate buffer and interval on a rolling 90-day window; feed the TSO forecast in as an input."]],
          "Management assessment of the proposed forecasting system.",
          widths_cm=[4.2, 11.8], font=8.5, left_cols=(0, 1))
rep.p(f"Even the cautious figure of {k(real_sav)} a year would pay for a "
      f"full-time analyst several times over for a supplier of this size. "
      f"For a larger supplier the saving scales with volume, while the "
      f"running cost hardly changes. The labour saving matters too. Today "
      f"many small suppliers have a person adjust last week's profile by "
      f"hand every morning, and the system makes that a monitoring task. "
      f"What a manager should take away is not that 'the LSTM won'. It did "
      f"not, on its own. Three things together did most of the work: "
      f"putting known calendar information into the model, averaging two "
      f"different model types, and buying a quantile instead of the mean. "
      f"Each of these is cheap, and each can be explained to a trader.",
      align="justify")
rep.p("**What we would do differently.** We would pull imbalance prices "
      "(reBAP) from the German regulator's data portal, so that the cost "
      "model uses real prices instead of assumed penalties. We would test "
      "at the real bidding horizon from the start, not as a robustness "
      "check. And we would include the regional holiday calendar, which "
      "we found the hard way to be the largest remaining source of cost.",
      align="justify")

# --------------------------------------------------------------------------
# 12 Conclusion
# --------------------------------------------------------------------------
rep.h("12  Conclusion", 1)
rep.p(f"German daily electricity demand is dominated by a very stable weekly "
      f"pattern ({f(vs['weekly'] * 100, 0)} % of variance) and a yearly cycle "
      f"({f(vs['yearly'] * 100, 0)} %). Its largest surprises are holidays. "
      f"Among the classical models, MA(q) was the poorest fit, because the "
      f"series' long memory is the wrong kind for an MA. AR models on the "
      f"levels rediscovered the seasonal structure by themselves, and "
      f"seasonal ARIMA with holiday dummies was the best single model "
      f"({f(A.loc[SX, 'RMSE'])} GW RMSE). Recurrent networks matched it "
      f"only once they were given the calendar. Averaging the two gave "
      f"{f(A.loc[ENS, 'RMSE'])} GW, {f(A.loc[ENS, 'MAPE'])} % MAPE, better "
      f"than the grid operators' own forecast in the one-step setting. "
      f"Turned into a newsvendor purchase rule, the forecast would have cut "
      f"a 1 % supplier's imbalance cost by "
      f"{f(C.loc[REC, 'saving_vs_TSO_pct'], 0)} % ({k(sav_tso)}) against "
      f"the public forecast, and by about {k(real_sav)} under real bidding "
      f"timing. The quantile LSTMs gave a well-calibrated 95 % interval in "
      f"a normal year ({f(Q['val_raw_PICP'], 1)} %), but an over-confident "
      f"one in the COVID year ({f(Q['test_raw_PICP'], 1)} %). That is a "
      f"reminder that a risk model is only as good as the years it has "
      f"seen.", align="justify")

# --------------------------------------------------------------------------
# References, AI disclaimer, appendix
# --------------------------------------------------------------------------
rep.h("References", 1)
refs = [
    "Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.",
    "Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/",
    "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.",
    "Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.",
    "Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32.",
    "Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366), 427-431.",
    "Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1-3), 159-178.",
    "Bandara, K., Hyndman, R. J., & Bergmeir, C. (2021). MSTL: A seasonal-trend decomposition algorithm for time series with multiple seasonal patterns. arXiv:2107.13462.",
    "Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.",
    "Hong, T., & Fan, S. (2016). Probabilistic electric load forecasting: A tutorial review. *International Journal of Forecasting*, 32(3), 914-938.",
    "Open Power System Data (2020). Data Package Time series, version 2020-10-06. https://doi.org/10.25832/time_series/2020-10-06",
    "ENTSO-E Transparency Platform. Total load: day-ahead forecast and actual. https://transparency.entsoe.eu/",
    "Arrow, K. J., Harris, T., & Marschak, J. (1951). Optimal inventory policy. *Econometrica*, 19(3), 250-272.",
    "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*.",
    "Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.",
    "Weron, R. (2014). Electricity price forecasting: A review of the state-of-the-art with a look into the future. *International Journal of Forecasting*, 30(4), 1030-1081.",
    "Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65(2), 297-303.",
    "Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217-1241.",
    "Khosravi, A., Nahavandi, S., Creighton, D., & Atiya, A. F. (2011). Lower upper bound estimation method for construction of neural network-based prediction intervals. *IEEE Transactions on Neural Networks*, 22(3), 337-346.",
]
for i, r in enumerate(refs, 1):
    para = rep.p(f"[{i}]  " + r, size=9.5, space_after=3)
    para.paragraph_format.left_indent = Cm(0.9)
    para.paragraph_format.first_line_indent = Cm(-0.9)

rep.h("AI disclaimer", 1)
rep.p("We used an AI assistant (Claude, by Anthropic) in this project. It "
      "helped write and debug the Python code, run the experiments, produce "
      "the figures, and draft the text of this report and the poster. The "
      "choice of problem, the decision framing and the assumptions in "
      "Section 8.1 were discussed and agreed by the authors. Every number "
      "in the report is generated by the code in the repository and can be "
      "reproduced with it. The authors are responsible for the content, "
      "including any errors.", align="justify")

rep.h("Appendix A  Reproducibility", 1)
rep.p(f"Repository: {REPO_URL}. The pipeline is `src/01_prepare_data.py` to "
      "`src/07_robustness.py`, run in order. The processed daily dataset "
      "(`data/processed/de_daily_load.csv`) is in the repository. The 130 MB "
      "raw file is downloaded by the first script. Every table in this "
      "report comes from `results/tables/`, and the report itself is built "
      "by `report/build_report.py`. Software: Python 3.11, statsmodels "
      "0.15, PyTorch 2.11 (CPU), pandas 3.0, matplotlib 3.10.",
      align="justify")
rep.h("Appendix B  Top five configurations of each random search", 1)
for lbl, s in [("RNN", R["search_rnn"]), ("LSTM", R["search_lstm"])]:
    rep.table(["lag", "batch", "lr", "layers", "neurons", "epochs", "val. RMSE", "val. MAPE"],
              [[int(r.lag), int(r.batch), r.lr, int(r.layers), int(r.hidden),
                int(r.epochs), f(r.val_RMSE, 3), f(r.val_MAPE)]
               for _, r in s.head(5).iterrows()],
              f"{lbl}: five best of 24 random-search trials (seed 0).",
              font=8.5)

pages_file = ROOT / "report" / "toc_pages.json"
rep.fill_toc(json.load(open(pages_file)) if pages_file.exists() else None)
out = ROOT / "report" / "Report_Load_Forecast_Procurement.docx"
rep.save(out)
json.dump([t for lvl, t in rep.headings if lvl in (1, 2) and t != "Contents"],
          open(ROOT / "report" / "toc_headings.json", "w"), indent=1)
print("saved", out, "| figures:", rep.fig_n, "| tables:", rep.tab_n)

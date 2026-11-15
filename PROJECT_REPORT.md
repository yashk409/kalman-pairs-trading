# Kalman Pairs-Trading Project — Complete Report

**Status: CLOSED — null result.** *(Reopened 2026-07-14; re-closed. Verdict unchanged.)*
**Verdict: the strategy has no exploitable edge. Its apparent profits were selection bias, its deep-learning branch had zero gross alpha — and the spread reversion it is built on does not happen out of sample.**

Final honest out-of-sample result (pairs re-screened **point-in-time at every cutoff**, US equities, 4 × 126-day windows, 30 pairs, post-fix):

| Strategy | Gross | **Net** | Sharpe | Windows positive | p |
|---|---|---|---|---|---|
| Kalman Filter | +0.66% | **−0.12%** | **−0.26** | 1/4 | 0.781 |
| KF + VolFilter | +0.48% | −0.13% | −0.40 | 2/4 | 0.796 |
| Static-Z (control, no Kalman) | +0.00% | −0.37% | −0.30 | 2/4 | 0.763 |
| **SPY (just buy the index)** | — | **+9.30%** | **+1.30** | **4/4** | 0.003 |

> ### The root cause — and it is **not** transaction costs
>
> | | |
> |---|---|
> | Spread amplitude vs cost of trading it (**theoretical** coverage) | **3.2× – 17.9×** |
> | What the strategy actually **realizes** | **0.85×** |
>
> **The spreads have ample room to pay. The strategy captures ~10% of its own theoretical move.**
> Entering at z = 1.5σ does not reliably lead to reverting to 0.5σ. The screener finds pairs
> that *have* reverted; out of sample they *stop*. Gross is +0.66% at **p = 0.78** —
> indistinguishable from zero **before costs are even applied.**

This is not a coding failure. [Do & Faff (2012)](https://onlinelibrary.wiley.com/doi/10.1111/j.1475-6803.2012.01317.x), testing pairs trading on US equities 1963–2009 with realistic costs, found it *"largely unprofitable in the period post-2002."* We independently reproduced the published result — **three times, by three different routes.**

---

> ## ⚠️ READ THIS BEFORE RUNNING ANYTHING IN THIS REPO
>
> **`config.yaml` is a fossil, not a validated configuration.** It was last written by the
> *selection-biased* screen and still contains:
>
> - **Pairs chosen with lookahead** — whatever list is in there was written by a single-date
>   screen. It is *not* a validated pair list, and it is what produced the fake Sharpe 1.55.
> - **`entry_z: 1.5`** — a value I explicitly **REFUTED** as overfit (p = 0.40; the ranking
>   *inverts* across markets). It is still in the file only because nothing overwrote it.
> - **`capital_per_pair: 3000000`** — **the whole book, despite the name.** `portfolio.py` and
>   `backtest.py` both divide by `len(pairs)`. **Do not "fix" that divisor** — removing it puts
>   `n_pairs`× leverage on live orders. It already happened once (§22.1).
> - **`holdings.yaml`** — stale, from a long-dead US ETF configuration.
>
> **Do not treat any of it as a result.** The only trustworthy artifacts in this repo are
> `walkforward.py` (the honest harness), `walkforward_results.csv` / `WALKFORWARD_REPORT.md`
> (the real answer), and this report.
>
> **The volatility-regime filter is still wired into `main.py` / `backtest.py` / `signals.py`.
> It was MEASURED and it HURTS** (Sharpe −0.26 → −0.40). It should be removed. See §22.3.

---

## Chronology — what happened, in order

| # | Stage | Outcome |
|---|---|---|
| 0 | **University course project** | Brief: can DL beat a Kalman filter at predicting spread reversion? Pipeline built to *produce a number*, not to *test* one. |
| 1 | Audit of the initial 30-epoch run | **6 bugs**, several fatal. KF was trading whitened noise → flat $0. |
| 2 | Fixes applied | KF trades the spread; DL signals un-lagged; leverage and cost-churn fixed. |
| 3 | Built `screener.py` + clean-OOS protocol | Cointegration screen; `--asof` == `--train-end`; rolling `coint_lookback`. |
| 4 | Multi-market (India) | `UNIVERSE_IN`, market profiles; **yfinance SQLite cache bug** found and fixed. |
| 5 | Built `experiments.py` | 8-suite ablation harness. `seed_variance` = the luck test. |
| 6 | India ablation | **DL is a coin flip** — 14-point swing from the random seed alone. N-BEATS dropped. |
| 7 | US universe expanded 117 → 475 | ~112,575 candidate pairs. |
| 8 | US ablation | **DL is *reliably negative*** (p = 0.0026). Its losses ≈ its transaction costs. |
| 9 | Gross/net decomposition | KF gross **is** positive (+0.70–0.90%); costs eat 40–80%. Break-even 10 bps/leg. |
| 10 | Breadth test (8 → 31 pairs) | **Sharpe 0.30 → 1.55.** Looked like the answer. It wasn't. |
| 11 | DL breadth test | **Seed dispersion did not shrink.** Error is *coherent*. DL definitively dead. |
| 12 | **Phase 0: `walkforward.py`** | Rolling re-screen → **Sharpe 1.55 → −0.20. NO-GO.** The 1.55 was selection bias. |
| 13 | Literature check | Do & Faff (2012): pairs trading *"largely unprofitable post-2002."* We reproduced it. |
| 14 | Successor project scaffolded | `~/Desktop/Projects/StatArb` — factor-residual stat arb. See §17. |
| 15 | **Salvage phase** | 3 alternative strategies written (`alt_*.py`). One **never traded**, one had **no units**, one made **+4.6% on 6 trades** — and became the vol filter. See §22.6. |
| 16 | **Parallelization** (M4, 10-core) | `ProcessPoolExecutor` + `spawn`-safe globals + BLAS thread capping. **The one unambiguous win** — and it touches no strategy logic. §22.7. |
| 17 | **Revival attempt** (2026-07-14) | 5 changes: DL removal, cost barrier, vol filter, "sizing fix", parallelization. |
| 18 | **Audit + honest re-run** | Sizing "fix" was an **8× leverage bug**. Verdict **unchanged: Sharpe −0.26**. But it revealed *why* — see §22.5. |

---

## Table of contents

1. [What the project was — including its origin as a course assignment](#1-what-the-project-was)
2. [Stage 1 — Correctness bugs found and fixed](#2-stage-1--correctness-bugs-found-and-fixed)
3. [Stage 2 — Pair screener and clean out-of-sample protocol](#3-stage-2--pair-screener-and-clean-out-of-sample-protocol)
4. [Stage 3 — Multi-market support (India)](#4-stage-3--multi-market-support-india)
5. [Stage 4 — The ablation harness](#5-stage-4--the-ablation-harness)
6. [Stage 5 — India results: DL is luck](#6-stage-5--india-results-dl-is-luck)
7. [Stage 6 — US results: DL is worse than luck](#7-stage-6--us-results-dl-is-worse-than-luck)
8. [Stage 7 — Gross vs net: the cost diagnosis](#8-stage-7--gross-vs-net-the-cost-diagnosis)
9. [Stage 8 — The breadth hypothesis (and why it was a mirage)](#9-stage-8--the-breadth-hypothesis-and-why-it-was-a-mirage)
10. [Stage 9 — The test that killed deep learning](#10-stage-9--the-test-that-killed-deep-learning)
11. [Stage 10 — Phase 0: the honest harness and the NO-GO](#11-stage-10--phase-0-the-honest-harness-and-the-no-go)
12. [Architectural defects in the codebase](#12-architectural-defects-in-the-codebase)
13. [Falsification ledger — including my own false positives](#13-falsification-ledger--including-my-own-false-positives)
14. [What went well](#14-what-went-well)
15. [What went wrong](#15-what-went-wrong)
16. [Literature review — what the field says](#16-literature-review--what-the-field-says)
17. [The one remaining idea — and the successor project](#17-the-one-remaining-idea--and-the-successor-project)
18. [What is reusable](#18-what-is-reusable)
19. [File inventory](#19-file-inventory)
20. [Tooling built along the way](#20-tooling-built-along-the-way)
21. [Session artifacts](#21-session-artifacts)
22. [**The revival attempt (2026-07-14) — and the finding that finally explains everything**](#22-the-revival-attempt-2026-07-14--and-the-finding-that-finally-explains-everything)
    - 22.6 [The salvage phase — and where the vol-regime idea was born](#226-the-salvage-phase--three-alternative-strategies-and-where-the-vol-regime-idea-was-born)
    - 22.7 [Parallelization (Apple Silicon M4)](#227-parallelization-apple-silicon-m4--the-one-unambiguous-engineering-win)
23. [Reconciliation with the parallel Antigravity log](#23-reconciliation-with-the-parallel-antigravity-log)

---

## 1. What the project was

### 1.0 Origin — a university course assignment

This began as a **course project**, not a trading desk. The original brief was academic and the
hypothesis was a natural one for a student assignment:

> *Take classical econometric pairs trading (Engle-Granger cointegration + a Kalman filter) and
> ask whether modern deep learning — LSTM, Transformer, N-BEATS, CNN-Transformer — can predict
> spread mean-reversion better than the classical filter can.*

That framing is worth stating plainly, because **it is the source of almost every error that
followed.** A course assignment is graded on whether the pipeline runs, the models train, and
the backtest produces a curve. It is *not* graded on whether the curve is real. So the initial
codebase was built to **produce a number**, and it did — with no seed-variance test, no gross/net
split, no point-in-time selection discipline, no cost model worth the name, and no control
strategy to beat. Every one of those omissions later turned out to be load-bearing.

The project stopped being a course assignment the moment we tried to scale the universe and
point it at live money. Everything from Stage 1 onward is the story of a sandbox experiment
colliding with the actual constraints of trading: transaction costs, selection bias, API limits,
and the brutally low signal-to-noise ratio of real financial data.

**The single most important thing to understand about this repo:** the academic version of this
project *succeeded*. It ran, it trained, it produced a Sharpe of 1.55. The professional version
of the same project is a **null result**. The difference between those two outcomes is not code
— it is methodology.

### 1.1 The thesis

A market-neutral **pairs-trading / statistical-arbitrage system**. Goal: make a profit.

**Core thesis.** Find two cointegrated assets, model the spread between them, and trade its mean
reversion — long the cheap leg, short the rich leg, flat market exposure.

**Two competing signal generators:**
- **ECM Kalman Filter** — dynamic hedge ratio β, spread z-score, threshold entry/exit.
- **Deep learning** — LSTM, Transformer, N-BEATS, CNN-Transformer, each forecasting the forward change in the cointegration spread.

**Stack:** Python, yfinance, statsmodels, pykalman, TensorFlow/Keras, pandas.
(`requirements.txt`: `yfinance`, `statsmodels`, `scikit-learn`, `pykalman`, `pyyaml`, `tensorflow`.)

---

## 2. Stage 1 — Correctness bugs found and fixed

The initial 30-epoch run produced a flat $0 P&L for the Kalman filter and erratic DL results. Audit found **six** bugs, several fatal.

### 2.1 The Kalman filter was trading noise (fatal)

```python
innov[t] = v          # one-step innovation
# ...and the strategy traded the z-score of THIS
```

A Kalman filter's *entire purpose* is to make the innovation **white noise**. If it succeeds, the innovation has no autocorrelation and no mean reversion — nothing to trade. The KF was therefore producing ~0 trades and a flat $0 P&L, **by construction**.

**Fix** — trade the dynamic-β spread *level*, which is the persistent, mean-reverting series:
```python
spread[t] = y_arr[t] - theta[0] * x_arr[t]
```

### 2.2 DL signals were lagged by the full forecast horizon (fatal)

Test signals were indexed by the **label date** (`DY = D + horizon`) rather than the **feature date** (`D`). The backtest therefore entered a position only *after* the predicted move had already fully elapsed — systematically trading 5 days late.

**Fix:** `test_dates = D[is_test]`.

### 2.3 DL was predicting the wrong target

Models predicted the two individual leg returns and then differenced them (`y_A − β·y_B`). Individual 5-day equity returns are near-random-walk noise; differencing two noisy forecasts **amplifies** the noise.

**Fix:** predict the forward change in the spread directly — a single `Dense(1)` head on `d_spread = spread.shift(-horizon) − spread`. The spread is stationary; its forward change is the genuinely learnable quantity.

### 2.4 Capital leverage bug

Each pair was sized against the **full** book, so an N-pair portfolio ran at **N× leverage** and its returns were not comparable to the benchmark.

**Fix:** `position_block(sig, risk_capital / n_pairs, ...)`.

### 2.5 Transaction-cost churn

Positions were re-sized on **every bar** as `risk_capital` drifted, paying `t_cost` on every single day regardless of whether the signal changed — a pure cost leak.

**Fix:** trade only when the signal actually changes; size is locked in at entry.

### 2.6 Live/backtest divergence

The live path (`main.py`) and the vectorized backtest implemented the trading rule separately and could disagree.

**Fix:** a single shared `kf_decide(z, prev_signal, stop_z, entry_z, exit_z)` in `signals.py`, with hysteresis, used by both.

**Result:** the KF began trading and producing sane P&L. But — as everything below shows — fixing the bugs revealed the strategy, rather than saving it.

---

## 3. Stage 2 — Pair screener and clean out-of-sample protocol

### 3.1 The screener (`screener.py`)

Built a cointegration screener that scans a large universe and emits a diversified shortlist. Filters, in order:

| Filter | Threshold | Why |
|---|---|---|
| Return correlation | \|ρ\| ≥ 0.4 | legs must co-move to hedge |
| Engle-Granger cointegration | p < 0.05 (later 0.01) | tested **both directions**, stronger kept |
| Hedge ratio sanity | 0.1 ≤ \|β\| ≤ 10 | |
| Half-life | 5–60 days | fast enough to round-trip, slow enough to not be noise |
| Zero crossings | ≥ 8 | trade frequency proxy |
| **Persistence** | cointegrates in **both halves** | rejects "one lucky stretch" pairs |
| β drift | < 0.5 | rejects unstable hedge ratios |
| ADF on spread | p < 0.10 | final stationarity confirmation |
| Diversification | each ticker used once; ≤2 pairs per sector-combo | |

**Design note learned the hard way:** cointegration is tested on a *recent rolling window* (~3y), never deep history. Relationships decay — EWA/EWC was Chan's 2006–2012 textbook pair and is broken today.

### 3.2 Clean OOS protocol

Established a strict discipline:
- Screener `--asof` **must equal** backtest `--train-end`.
- `coint_lookback` in the backtest **must equal** the screener's `--lookback` (756 days), so the backtest trades exactly the relationship that was screened (verified: `MATCH: True, 756 days`).
- Everything — β, α, λ, Q, R, scalers, z-score normalization — fitted only on data ≤ cutoff.
- Benchmark against **real downloaded prices** (SPY / ^NSEI), not a synthetic proxy.

**This protocol was correct as far as it went — but it contained a hole that took until Stage 10 to find.** See §11.

---

## 4. Stage 3 — Multi-market support (India)

Generalized the codebase away from US-only assumptions:
- `MARKET_PROFILES` — benchmark, timezone, market hours, currency per market.
- `UNIVERSE_IN` — 161 NSE tickers across 21 groups (`.NS` suffix), benchmark `^NSEI`.
- Market-aware `get_current_prices` / `is_market_open_now`.

### The data-integrity bug (user-diagnosed)

The India expansion immediately produced **22 failed downloads**. Two independent causes were
tangled together, and separating them was the whole job:

| Symptom | Real cause | Fix |
|---|---|---|
| *"no price data"* on **valid** tickers | yfinance's **concurrent writes corrupted its own SQLite timezone cache** (`unable to open database file`) → **false negatives** | **`threads=False`** on every `yf.download` |
| `KeyError` on the price frame | **`Adj Close` vs `Close`** — yfinance changed which columns it returns depending on `auto_adjust`, so the column being indexed sometimes did not exist | Pin the column selection explicitly |
| *"no price data"* on **dead** tickers | Genuinely delisted (`TATAMOTORS.NS`, `LTIM.NS` — corporate actions) | Removed from the universe |

**We traded download speed for data stability, deliberately.** `threads=False` makes the
screener's data pull slower, and that was the correct call: a *silently wrong* price frame
poisons every downstream number, and a false-negative ticker biases the universe in a way no
later test can detect. Verified 9/9 batch success afterwards.

*(This is one of the few genuinely reusable artifacts in the repo — the fix was ported into the
successor project's `data.py` verbatim.)*

*(Note: `Rs` is used rather than `₹` because matplotlib's default fonts lack the rupee glyph.)*

---

## 5. Stage 4 — The ablation harness

Built `experiments.py` — an independent harness that reuses the data/model/portfolio code but re-implements the windowing so it can test configurations the main pipeline cannot express.

**Suites:**

| Suite | Question |
|---|---|
| `seed_variance` | **Is the P&L edge or a coin flip?** Same config, N seeds, per-model. |
| `data_start` | 2015 / 2018 / 2020 / 2022 / 2023 |
| `coint_lookback` | full / 1008 / 756 / 504 / 378 / 252 |
| `split_mode` | `[base|val|ft]` vs `[base|ft|val]` vs no fine-tune |
| `finetune_days` | 0 / 63 / 126 / 252 |
| `entry_thresholds` | KF `entry_z` 1.0 / 1.25 / 1.5 / 2.0 |
| `walkforward` | multiple cutoffs, each tested on the next ~6 months |
| `kf_edge` | KF-only (entry_z × stop_z) grid **across every** walk-forward window |

**The decisive design choice:** `seed_variance` runs each model per-seed **without ensembling**. If a model's P&L standard deviation swamps its mean, the "profit" is a coin flip, not an edge. This one suite ended up settling the entire deep-learning question.

---

## 6. Stage 5 — India results: DL is luck

3 pairs (GAIL/POWERGRID, AMBUJACEM/JUNIORBEES, JINDALSTEL/TATASTEEL), 8 seeds.

### Seed variance — same data, same config, only the random seed changed

| Model | mean | std | min | max | #positive | Sharpe | avg trades |
|---|---|---|---|---|---|---|---|
| LSTM | −0.50% | 1.40 | −2.02 | +2.24 | 3/8 | −0.21 | 16 |
| Transformer | −0.71% | 4.59 | **−8.74** | **+5.39** | 4/8 | −0.11 | 14 |
| CNN-Transformer | −2.07% | 2.69 | −7.00 | +1.16 | 2/8 | −0.98 | 12 |
| **N-BEATS** | **−3.03%** | 4.08 | −9.52 | +2.42 | 2/8 | **−1.56** | **128** |

**Finding:** a **14-point swing** (−8.74% → +5.39%) from nothing but the random seed. The "profits" seen in earlier screenshots were coin flips.

**N-BEATS was the throwaway** — worst mean, worst Sharpe, and ~**8× the turnover** of every other model (128 trades vs 12–16). A cost-churning machine. Dropped from all subsequent runs.

### Fine-tuning ablation (DL-Ensemble)

| `fine_tune_days` | Return |
|---|---|
| **0 (off)** | **+3.65%** |
| 63 | −3.91% |
| 126 (then-current config) | **−5.83%** |
| 252 | +0.66% |

Set `fine_tune_days: 0`. *(This conclusion was later shown to be **directionally right but magnitude-inflated** — see §13.)*

---

## 7. Stage 6 — US results: DL is worse than luck

Expanded `UNIVERSE_US` from 117 → **475 tickers across 40 groups** (85 ETFs). ~112,575 candidate pairs. Screened 8 pairs, ran the full ablation.

### Seed variance (8 pairs, 8 seeds, N-BEATS excluded)

| Model | mean | std | #positive | Sharpe | t | p |
|---|---|---|---|---|---|---|
| LSTM | −0.33% | 1.00 | 2/8 | −0.19 | −0.92 | 0.388 |
| Transformer | −1.75% | 1.67 | **1/8** | −0.87 | −2.97 | **0.021** |
| CNN-Transformer | −1.38% | 2.04 | 2/8 | −0.82 | −1.92 | 0.097 |
| **Combined** | **−1.15%** | | **5/24** | | **−3.37** | **0.0026** |

In India, DL was a *coin flip*. In the US it is **reliably negative** — 5 of 24 seed-runs profitable where a coin flip predicts ~12 (binomial **p = 0.0066**).

A statistically *significant negative* result is not bad luck. It is a system that reliably destroys money.

### The cost decomposition — the smoking gun

| Model | avg trades | **estimated** cost drag | **actual** mean return |
|---|---|---|---|
| CNN-Transformer | 78 | **−1.47%** | **−1.38%** |
| Transformer | 61 | −1.15% | −1.75% |
| LSTM | 60 | −1.13% | −0.33% |

**The models' losses are almost exactly their transaction costs.** They have zero gross predictive power: they emit essentially random signals, pay the spread on each one, and hand back the bill.

### Why — the R² ceiling

For a mean-reverting (Ornstein-Uhlenbeck) spread, the theoretically optimal forecast **is a linear function of `z_spread`** — which is already one of the five input features:

$$E[\Delta S] \approx -\kappa \cdot z \cdot h$$

Measured on the 8 US pairs, a **single-variable linear regression** of the 5-day forward spread change on `z_spread` captures:

| | R² |
|---|---|
| Mean across 8 pairs | **0.0199 (2%)** |
| Best (MAR/AME) | 0.038 |
| Worst (DVN/LNG) | **0.0005** — and *wrong-signed* (ρ = +0.023) |

**That 2% is the entire learnable structure in the target.** An LSTM with ~15,000 parameters ingesting a 60×5 tensor is being asked to rediscover one coefficient buried in 98% noise. It cannot beat the linear rule; it can only add estimation variance — which becomes random signal flips, which become trades, which become the cost bill.

**The DL branch was never complementary to the KF. It was redundant with it, and strictly worse** — both extract the same 2% from `z_spread`, but the KF does it with one threshold rule and zero estimation noise.

*(DVN/LNG having wrong-signed, near-zero reversion is itself a finding: the screener selected a pair whose relationship had decayed.)*

---

## 8. Stage 7 — Gross vs net: the cost diagnosis

Stripping transaction costs out of the KF revealed a **genuine positive gross edge** — and one that improves *per trade* with selectivity.

| entry_z | Gross | Costs | Net | Trades | edge/trade | cost/trade | **coverage** |
|---|---|---|---|---|---|---|---|
| 1.5 | +0.90% | −0.72% | +0.18% | 103 | 0.87 bps | 0.70 bps | **1.25×** |
| 2.0 | +0.70% | −0.37% | +0.33% | 55 | 1.27 bps | 0.67 bps | **1.89×** |
| 3.0 | +0.21% | −0.07% | +0.13% | 14 | 1.50 bps | 0.50 bps | **3.00×** |

**Break-even transaction cost: 10 bps/leg** (against 5 bps modelled).

Cost per trade is **flat**; edge per trade **rises** with selectivity. This is a mechanical relationship, not a fitted one — and it correctly pointed at turnover, not signal quality, as the problem.

**But total profit collapses as you get selective** (+0.21% at `entry_z=3.0`), because you only get 14 trades. This produced the (ultimately wrong) diagnosis in Stage 8.

---

## 9. Stage 8 — The breadth hypothesis (and why it was a mirage)

**The reasoning (which was sound):** the fundamental law of active management says

$$\text{IR} = \text{IC} \times \sqrt{\text{breadth}}$$

Our IC ≈ 0.14 (√0.02) — a perfectly *normal* stat-arb information coefficient. Nobody has a high one. But breadth = **8 pairs** is catastrophic. We had a decent signal and nowhere to spend it. The screener scans 112,575 candidates and we were throwing away all but 8.

**The test:** re-screen at `--top 60`, `--coint-pmax 0.01` (tightened for multiple testing). Only **31 pairs** survived. Book scaled 500K → 3M so integer-share rounding wouldn't swamp a 1.5 bps edge.

**The result — same code, same windows, only pair count changed:**

| entry_z | mean | **Sharpe** | max DD | #positive | (8-pair Sharpe) |
|---|---|---|---|---|---|
| 1.5 | +1.27% | **1.55** | −1.3% | 3/4 | *0.30* |
| 2.0 | +0.95% | **1.64** | −1.1% | 3/4 | *0.68* |
| 2.5 | +0.84% | 1.53 | −0.8% | 3/4 | — |
| 3.0 | +0.58% | 1.36 | −0.5% | 3/4 | — |

**Sharpe 0.30 → 1.55.** And — the part that made it convincing — **Sharpe was flat across every threshold** (1.36–1.64). At 8 pairs it swung wildly with the knob (0.30 → 0.68), which is what a *tuned, fragile* number looks like. A result that barely moves when you turn the knob is what a *structural* one looks like.

**It was wrong.** See §11. The flatness was real; the level was fake. The signal I trusted as "robustness" was robustness to a *parameter*, not to the *protocol*.

---

## 10. Stage 9 — The test that killed deep learning

After the KF breadth result, I argued the DL models might be salvageable: their implied gross alpha (actual return minus estimated cost) suggested **LSTM ≈ +0.80%**, comparable to the KF. If their losses were *real signal buried under estimation noise*, and noise diversifies away, then breadth should rescue them too.

**That is a falsifiable prediction, so I tested it.** If DL error were idiosyncratic noise, going 8 → 31 pairs must shrink seed-to-seed dispersion by ≈ √(31/8) = **2×**.

| Model | σ @ 8 pairs | σ **predicted** @ 31 | σ **actual** @ 31 | |
|---|---|---|---|---|
| LSTM | 1.00 | 0.50 | **1.04** | no shrinkage |
| Transformer | 1.67 | 0.83 | **2.18** | **grew** |
| CNN-Transformer | 2.04 | 1.02 | **2.21** | **grew** |

**Dispersion did not shrink at all. For two of three models it grew.**

Errors that do not cancel across 31 independent pairs **are not noise — they are coherent.** Each seed learns a *different wrong function* and applies it confidently to every pair, so mistakes add rather than average out.

> **These models are not noisy estimators of a real signal. They are confident estimators of nothing.**

Meanwhile LSTM became **significantly negative** (−1.06%, t = −2.88, **p = 0.024**) and got *worse* with diversification. On the identical change, the KF's Sharpe went 0.30 → 1.55.

**Deep learning: definitively dead.** My "+0.80% implied gross alpha" hypothesis was refuted by the very test I designed to check it.

---

## 11. Stage 10 — Phase 0: the honest harness and the NO-GO

### The hole in the protocol

Every result up to this point — **including the Sharpe 1.55** — selected its pairs **once**, as-of the most recent date (2026-01-09), and then backtested them on windows that **preceded** that date (2024-07, 2025-01, 2025-07).

**The pairs were therefore chosen partly *because* they survived the test period.** That is selection lookahead. The clean-OOS protocol from Stage 2 rigorously prevented leakage in *fitting* (β, λ, Q, scalers) — and completely missed it in *selection*.

### The fix — `walkforward.py`

At **every** cutoff, re-screen the entire 475-ticker universe using **only data available at that moment**, then trade the resulting pairs forward into a window nothing has seen. Plus two controls that were never run:

- **Static-Z** — a plain rolling z-score on a *static* OLS-β spread, no Kalman filter at all. *If the Kalman machinery can't beat this, the Kalman machinery is decoration.*
- **Gross vs net**, reported side by side, in every run.

**Pre-committed go/no-go, written into the script before it was run:**
> KF Sharpe < 0.30 after the rolling re-screen → the 1.55 was selection bias. Stop.

### The result

| Cutoff | Candidates | Passed | Selected | KF gross | KF net | KF Sharpe | Static-Z net | SPY |
|---|---|---|---|---|---|---|---|---|
| 2024-07-01 | 30,965 | 89 | 30 | +0.35% | −0.37% | −0.40 | −2.67% | +8.56% |
| 2025-01-02 | 28,351 | 134 | 30 | +1.83% | **+1.03%** | +0.52 | +0.07% | +6.74% |
| 2025-07-01 | 27,665 | 54 | 30 | +0.25% | −0.49% | −0.63 | −0.42% | +11.87% |
| 2026-01-02 | 17,514 | 64 | 27 | +0.47% | −0.36% | −0.27 | +2.16% | +10.03% |
| **Aggregate** | | | | **+0.73%** | **−0.05%** | **−0.20** | −0.21% | **+9.30%** |

**Sharpe 1.55 → −0.20.** Net −0.05%, **1 of 4** windows positive, p = 0.903. The criterion fired.

```
NO-GO: KF Sharpe -0.20 < 0.30. The earlier 1.55 was SELECTION BIAS.
```

### What else the honest run revealed

**1. The lookahead's fingerprint is visible.** The cutoff that *nearly* held up (2026-01-02, −0.36%) is the one closest to the old screen date of 2026-01-09 — almost no lookahead available. The windows furthest back are where the fake edge lived. Textbook signature.

**2. The screener selected GOOG/GOOGL.** Two share classes of the *same company*. It cointegrates beautifully — and its spread amplitude is far below the round-trip cost. It is **mathematically incapable of paying**, and the screener happily selected it, because it filters on p-value, half-life and β-stability and **never once asks whether the reversion is bigger than the fee.**

**3. The Kalman filter is not useless — just not enough.** It beats the static-β control on gross (+0.73% vs +0.16%), so the filter genuinely adds signal. Gross is even marginally positive (t = 1.95, p = 0.146). But it's ~1.5%/yr gross against a cost structure that consumes 100% of it.

**4. The arithmetic, finally stated plainly:**

> **edge/trade = 0.17 bps · cost/trade = 0.18 bps · coverage = 0.94×**

---

## 12. Architectural defects in the codebase

Found by audit. **None of the top-line result depends on fixing these** — which is itself informative about where the leverage wasn't.

### 12.1 λ is a category error *(real bug, never fixed)*

`estimate_lambda_ecm` returns the **error-correction coefficient**: how fast *the spread* reverts to its mean. It is then used as the state-transition decay for **θ = [β, α]**:

```python
F = (1.0 + lam) * np.eye(2)     # pulls BETA back toward theta_star
c = -lam * theta_star
```

These are **unrelated quantities**. "The spread reverts with half-life 14 days" is being reinterpreted as "the hedge ratio reverts to its OLS value with half-life 14 days." No theory connects them.

Worse: `estimate_QR_em` fits `Q` under `transition_matrices=np.eye(2)` (a random walk), but the filter then **runs** with `F=(1+λ)I`. **You estimate one model and execute a different one.**

### 12.2 The filter fights its own signal *(conceptual, never fixed)*

The traded spread uses the **posterior** β — updated with today's observation:

```python
spread[t] = y_arr[t] - theta[0] * x_arr[t]   # theta = POSTERIOR
```

A Kalman filter exists to drive `y − βx − α → 0`. Every update **actively explains away the mispricing you want to trade.** The tension is fundamental: *the better the filter, the less there is to trade.* This is the same wall hit in §2.1 — trading the innovation gave zero trades because the filter whitens it; retreating to the posterior spread is the same problem in weaker form.

A pairs hedge ratio should be **slow and stable**. This one is optimized to be **fast and adaptive** — precisely backwards.

### 12.3 Horizon/holding mismatch *(never fixed)*

The nets forecast a **5-day** move, but the backtest re-evaluates and can flip **daily**. You predict a 5-day reversion, then abandon it after 1 day when noise flips the sign. Manufactures turnover without ever capturing the forecast.

*(The literature independently confirms this matters: Guijarro-Ordonez et al. find "around half of the Sharpe ratio can persist for a holding period of one week." The 5-day horizon was right; re-deciding daily destroyed it.)*

### 12.4 The screener never checks whether a pair can pay *(never fixed)*

No cost-adjusted filter. Should reject any pair where `spread_sd × entry_z < 2 × round_trip_cost`. **GOOG/GOOGL is the proof this is missing.**

### 12.5 Label overlap was unpurged *(FIXED in Phase 0)*

With `horizon=5`, the sample at *t* shares label information with *t+1 … t+4*. The bare index split left the last 5 base samples overlapping the first validation samples — meaning **early stopping was selecting the model on data it had partially trained on.**

**Fix:** purge `horizon` samples either side of every split boundary (`data_loader.prepare_dl_data`).

### 12.6 DL optimized the wrong objective *(never fixed — and probably the deepest DL flaw)*

The nets minimize **Huber loss on a near-noise target**, and *then* a threshold is bolted on top to make trades. **The trading objective was never optimized.** The state of the art ([Guijarro-Ordonez et al.](https://arxiv.org/abs/2106.04028)) is explicit that the trading objective *"is different from a prediction objective"* and optimizes **Sharpe directly**. This is very likely why each seed converged to a different confident fiction.

### 12.7 Models grossly oversized

~15,000 parameters to extract a 2% R² signal from ~2,000 samples. Should be ~500.

### 12.8 Universe survivorship bias *(unfixable without point-in-time data)*

`UNIVERSE_US` lists tickers liquid **today**. The bankrupt and delisted are absent. (The run itself surfaced 12 dead tickers: CTRA, MRO, CMA, ANSS, X, K, SKX, PARA, FI, DFS, HES, SNV.)

---

## 13. Falsification ledger — including my own false positives

A report that only records its successes is worthless. **Two of my confident claims were wrong, and one I got right for the wrong reason.**

| Claim | Verdict | Note |
|---|---|---|
| KF trades the whitened innovation → ~0 trades | ✅ **Confirmed** | Root-cause bug. Fixed. |
| DL signals lagged by the full horizon | ✅ **Confirmed** | Root-cause bug. Fixed. |
| ~~`entry_z = 1.5` is the defensive sweet spot~~ | ❌ **REFUTED** | **Overfit.** Never significant (p = 0.40), and the ranking **inverts across markets** — India ranked `2.0` *worst*, the US ranked it *best*. I sold this as "evidence-backed." It wasn't. |
| ~~Fine-tuning hurts: +3.65% vs −5.83%~~ | ⚠️ **Partial** | Direction replicated (4/4 in the US), but magnitude was **noise-inflated ~5×**. Read off a 2-seed draw when seed σ was 1–2%. |
| ~~Breadth fixes it — Sharpe 1.55~~ | ❌ **REFUTED** | **That was selection lookahead, not breadth.** Collapses to −0.20 under an honest protocol. |
| ~~LSTM has ~+0.80% hidden gross alpha~~ | ❌ **REFUTED** | My own retraction of "DL is dead" — and *it* was wrong too. The breadth test refuted it. |
| DL has no exploitable gross alpha | ✅ **Confirmed** | Right conclusion, but I first reached it via a crude cost estimate, not evidence. The breadth/dispersion test is what actually proves it. |
| The strategy has no edge after honest evaluation | ✅ **Confirmed** | The final, and only, robust conclusion. Re-confirmed 2026-07-14: Sharpe **−0.26**. |
| **— Revival attempt, 2026-07-14 —** | | |
| ~~Sizing: "the `/len(pairs)` divisor is a bug"~~ | ❌ **REFUTED (user's)** | It is **not** a bug — it is diversification. Removing it put **8× leverage** on live orders (§22.1). Caught from the order sheet, not the backtest. |
| ~~The cost barrier gives 3× coverage~~ | ❌ **REFUTED (user's)** | It gave **1.00×** — break-even. Used `entry_z` instead of `entry_z − exit_z` (1.5×), and ignored the `1/(1+\|β\|)` divisor (~2×). |
| ~~Mean reversion needs high vol; block the quiet 25%~~ | ❌ **REFUTED (user's)** | Correctly *implemented* (trailing quantile, no lookahead) but it **HURTS**: Sharpe −0.26 → **−0.40**. Cut gross 27% while cutting cost only 22%. |
| Deep learning should be stripped entirely | ✅ **Confirmed (user's)** | The best-supported change of the five. Zero gross alpha across 2 markets, 3 architectures, 8 seeds. |
| The Kalman filter is decoration vs a static-β z-score | ❌ **REFUTED** | KF gross **+0.66%** vs Static-Z **+0.00%**. The dynamic β earns its keep. Control finally run. |
| ~~Coverage (edge/cost) is the binding constraint for pairs~~ | ❌ **REFUTED (mine)** | **My framing error.** Theoretical coverage is **3.2–17.9×**; realized is **0.85×**. Amplitude is ample — **the reversion doesn't happen** (§22.5). Right lens for the residual project, wrong one here. |
| **— Salvage phase (`salvage_results.md`) —** | | |
| ~~Factor-residual salvage: "Net PnL 0.0000"~~ | ❌ **VOID** | **Not a null result — a null *by construction*.** It placed **zero trades in every period**: OU was fitted on residual *returns* instead of the cumulative residual, so the AR(1) coefficient came back ≤ 0 and the κ filter rejected every stock. A table of eleven zeros, read as evidence. |
| ~~Options-dispersion salvage: "Net Vol PnL −0.1916"~~ | ❌ **VOID** | **No options, no implied vol, no vega** — realized-vs-realized has no variance risk premium to capture. And it adds *volatility points* to a *cost in returns*: **the number has no units** and cannot be interpreted at all. |
| ~~Vol-regime salvage: "+4.61% — this rescues pairs trading"~~ | ❌ **REFUTED** | **6 trades, one hand-picked pair (KO/PEP), no screener.** This is the *entire* evidentiary basis on which the vol filter entered production. Measured properly at 30 pairs / 352 trades it made the Sharpe **worse** (§22.3). **The project's root methodological error, committed one last time.** |
| Parallelization is correct and safe | ✅ **Confirmed** | No shared mutable state (no races), `spawn`-safe globals via `initializer` (no IPC blow-up), BLAS threads capped (no oversubscription). **The only one of the five changes that was simply right** — and the only one that doesn't touch the strategy. |

### The root methodological error, stated so it isn't repeated

**The axis suites (`data_start`, `coint_lookback`, `split_mode`, `fine_tune_days`) run only 2 seeds, while seed-to-seed σ is 1–2%.** Every ±2–4% swing across those scenarios is *pure noise*. `coint_lookback` shows Transformer swinging +2.47% → −2.05% → +2.35% across settings — that is not a finding, that is a random number generator.

**I read conclusions off those tables anyway.** That is how the fine-tuning magnitude and the `entry_z` "sweet spot" were manufactured.

### The second lesson, from the 2026-07-14 revival

**A green backtest is not a safety check.** The 8× leverage bug (§22.1) was invisible in
`backtest.py` — which happily printed *"Sharpe 2.81"* — because the backtest still divided by
`n_pairs` while the live path no longer did. It was caught only by **reading the live order
sheet and adding up the notionals**. Before any capital moves: sum the gross notional across
the order sheet and check it against the book. A backtest cannot catch a bug that lives in the
code the backtest doesn't run.

**And: a filter that "saves costs" must beat the breadth it destroys.** The vol-regime filter
cut costs by 22% and looked sensible. It cut *gross* by 27% and made the Sharpe worse. Cutting
turnover is not free — `IR = IC × √breadth`, and every trade you skip is breadth you gave up.

---

## 14. What went well

- **The bug hunt was real and correct.** Six genuine bugs, several fatal (innovation-trading, horizon lag, N× leverage). All fixed and verified.
- **The harness proved itself.** The KF returned an *identical* −2.11% across every `split_mode` and `fine_tune_days` scenario — exactly as it must, since those knobs don't touch it. That internal consistency is why the negative results are trustworthy.
- **`seed_variance` was the right instrument.** One suite settled the entire deep-learning question, permanently.
- **The gross/net decomposition** correctly identified turnover — not signal quality — as the binding constraint, and it was right.
- **Pre-committing the falsification criterion worked.** The NO-GO threshold was written into `walkforward.py` *before* it ran. When the number came in at −0.20, there was no room to rationalize.
- **`walkforward.py` caught its own project's biggest lie.** A harness that kills the hypothesis it was built to support is the most valuable thing here.
- **The negative result is independently corroborated** by Do & Faff (2012). We reproduced a published finding without knowing it.

**A rigorous experiment that returns a null result is a success, not a failure.** You now know something true instead of believing something false — and you found out before risking capital.

---

## 15. What went wrong

- **Selection bias survived nine stages of "clean OOS" discipline.** The protocol rigorously prevented leakage in *fitting* and was completely blind to leakage in *selection*. This is the single most important lesson in the project.
- **I over-read small samples repeatedly.** Two false positives (`entry_z`, fine-tuning magnitude) generated from 2-seed draws and 4-window t-tests. Both were presented with more confidence than the data supported.
- **Robustness to a *parameter* was mistaken for robustness to the *protocol*.** The flat Sharpe (1.36–1.64 across all `entry_z`) genuinely was a good sign — and it was flatness in the wrong dimension. It said nothing about selection bias.
- **The DL branch was architecturally doomed from the start** and it took the whole project to see why: it was assigned a task (predict spread direction) whose answer was already sitting in one of its own input features, with a 2% R² ceiling.
- **The nets never optimized the trading objective.** Minimizing Huber loss and then thresholding is not the same as maximizing risk-adjusted return, and the gap is where all the incoherent per-seed fictions were born.
- **Enormous compute was spent on the wrong axis.** Overnight 300-epoch runs to squeeze more IC out of 8 pairs, when the binding constraints were breadth and — ultimately — costs and selection bias.

---

## 16. Literature review — what the field says

| Source | Finding |
|---|---|
| [Do & Faff (2012), *Are Pairs Trading Profits Robust to Trading Costs?*](https://onlinelibrary.wiley.com/doi/10.1111/j.1475-6803.2012.01317.x) | US equities 1963–2009 with realistic commissions, market impact and short fees. Pairs trading is **"largely unprofitable in the period post-2002."** **This is exactly our result.** |
| [Avellaneda & Lee (2010), *Statistical Arbitrage in the U.S. Equities Market*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1153505) | Abandons pairs. **PCA factor residuals** + OU mean reversion + s-score. Sharpe **1.44** (1997–2007) after costs — but only **0.9** in 2003–2007. Decaying. |
| [Guijarro-Ordonez, Pelger & Zanotti (2022), *Deep Learning Statistical Arbitrage*](https://arxiv.org/abs/2106.04028) (*Management Science*) | ~550 large-cap US stocks. Residuals vs Fama-French 5 / PCA / IPCA. **CNN+Transformer**, and critically **optimizes Sharpe directly, not prediction error** — the trading objective *"is different from a prediction objective."* Claims Sharpe **> 4**, profitable under realistic costs. Notes ~**half the Sharpe survives a 1-week holding period**, and that the **factor model barely matters — signal extraction is the separating element.** |
| [A Deep Learning Approach for Trading Factor Residuals (2024)](https://arxiv.org/html/2412.11432v1) + [replication](https://bsiranosian.com/blog/deep-learning-statistical-arbitrage-part-1/) | Replicates on 2016–2024, gets Sharpe **> 10** — and the authors themselves warn this **"has not yet incorporated realistic transaction costs, market impact, or short-sale fees."** **Treat the architecture as credible and the headline numbers as fantasy.** |

**Conclusion:** cointegration pairs trading on liquid large-caps is an efficiently-arbitraged, **sub-transaction-cost** phenomenon. This is exactly what thirty years of everyone knowing the strategy would predict.

---

## 17. The one remaining idea — and the successor project

**Stop trading pairs. Trade factor residuals.**

Every failure mode we diagnosed is *structurally* eliminated by the Avellaneda-Lee / DLSA architecture:

| What killed us | What the residual approach fixes |
|---|---|
| **Pair-selection lookahead** (1.55 → −0.20) | **There is no pair selection.** Structurally impossible to have this bias. |
| **Breadth = 30 pairs**; IC 0.14 → IR hopeless | **~475 residual signals**, one per stock. |
| Loss = Huber on `d_spread`, *then* threshold | **Loss maximizes Sharpe directly.** |
| edge/trade 0.17 bps **<** cost/trade 0.18 bps | **Turnover is a constraint inside the objective.** |
| 2% R² ceiling — `z_spread` already had the signal | Factor residual is a cleaner, higher-amplitude reverting series. |
| Screener picked **GOOG/GOOGL** | No screener exists. |

**Proposed Phase A (cheap go/no-go, no DL):** rolling 252d PCA on the 475-ticker return matrix → ~15 factors → per-stock residual → OU fit → s-score → dollar-neutral book across all names, run through the **existing honest walk-forward harness**, gross and net.

> **Pre-committed criterion: gross Sharpe < 0.5 → stop the project for real.**
> If the residual signal has no edge *before* costs, no neural net can manufacture one.

**Phase B (only if A passes):** replace the OU s-score with the CNN+Transformer, and the threshold rule with a network mapping signals → portfolio weights, trained on **negative Sharpe with a turnover penalty**. This finally gives the DL models a job they can actually do.

**Honest probability of ending profitable after realistic costs: ~25–30%.** Better than patching the pairs strategy (~15%), because it is structurally sound and independently evidenced — but the cost problem does not vanish just because the signal is cleaner, and 2024–2026 is a far more arbitraged sample than their 1998–2016.

**Recommendation: start a NEW project.** Every function here is shaped like `f(A, B, …)` — a *pair*. Residual stat arb's fundamental object is an N×T matrix → factors → N residuals → one weight vector. Different data model, not an extension. And leaving this codebase intact preserves it as a documented null result.

### 17.1 The successor was built — `~/Desktop/Projects/StatArb`

A **separate git repo**, outside this folder. This codebase is frozen.

| Commit | Contents |
|---|---|
| `09d7a17` | Scaffold factor-residual stat-arb project (Phase A) — 11 files, 764 lines |
| `be85faa` | Master prompt covering Phases A–D |

**Ported (proven code):** `universe.py` (463 tickers — the 12 delisted names this project's walk-forward exposed are stripped), `data.py` (the `threads=False` yfinance fix + parquet cache), `metrics.py` (`compute_metrics`).

**Rewritten:** `backtest.py` — a **weight-vector** engine, not pair share-blocks. Weights at *t* earn returns at *t+1*; costs scale with turnover; gross/net always reported together.

**New:** `residuals.py` — the Avellaneda-Lee engine (PCA eigenportfolios → factor regression → OU fit → s-score → factor-neutral weights). `run_phase_a.py` — the runner, with the go/no-go **hard-coded before it was ever run**.

**The structural difference:** there is no train/test split, no screener, and no fitted hyperparameter. Every quantity at time *t* comes from a trailing window ending at *t*. The entire post-warmup period is one continuous out-of-sample backtest with **no lookahead available anywhere to exploit** — which is precisely the hole that invalidated nine stages of this project.

`PROMPT.md` in that repo carries the post-mortem, the six working rules, and the pre-committed criterion for every phase.

> **Phase A go/no-go: gross Sharpe < 0.5 → stop.** Gross, not net — if there is no edge *before* costs, no network can manufacture one.

**Phase A has not been run yet.** That is the next action.

---

## 18. What is reusable

| Asset | Verdict |
|---|---|
| **`walkforward.py`** | **Port the pattern.** The crown jewel — point-in-time re-screening, gross/net, pre-committed go/no-go. It caught the lie. |
| **`UNIVERSE_US`** (in `screener.py`) | **Port the dict.** 475 curated tickers, 40 sectors, delisted names weeded out. Exactly the PCA universe. |
| **`download_prices`** (in `data_loader.py`) | **Port.** Contains the `threads=False` SQLite-cache fix. |
| **`compute_metrics`** (in `experiments.py`) | **Port as-is.** Sharpe / max-DD / vol / turnover, correct. |
| **`models/cnn_transformer.py`** | **Port for Phase B.** Literally the right architecture — needs a new head and the Sharpe loss. |
| `models/kalman_filter.py` | **Leave.** No role in residual stat arb. |
| `screener.py` (`evaluate_pair`, cointegration filters) | **Leave.** No pairs, no screener. |
| `signals.py`, `portfolio.py`, `main.py`, `experiments.py` (suites) | **Leave.** All pair-shaped. |
| `backtest_pnl` | **Rewrite.** Holdings/pair-based; the new one takes a weight vector. |

**~40% of the infrastructure carries; ~0% of the strategy logic.**

**The most valuable thing to carry is not code — it is the discipline:** pre-committed falsification criteria before every run, seed-variance luck tests, point-in-time *everything* (selection included), and gross-vs-net always. Without that methodology you would simply generate another Sharpe 1.55 and believe it.

---

## 19. File inventory

### Core pipeline
| File | Role | Status |
|---|---|---|
| `config.yaml` | pairs, market profile, dates, KF params, DL hyperparams | ⚠️ **FOSSIL** — biased pairs + refuted `entry_z`. Not a validated config. |
| `holdings.yaml` | live position state | ⚠️ **STALE** — from a long-dead US ETF configuration |
| `data_loader.py` | download (`threads=False`), cointegration params, DL windowing (**purge/embargo**) | |
| `models/kalman_filter.py` | ECM Kalman, λ/Q/R estimation, rolling z-score | contains the **λ category error** (§12.1) |
| `models/{lstm,transformer,nbeats,cnn_transformer}.py` | DL architectures, all `Dense(1)` | zero gross alpha (§10) |
| `signals.py` | shared `kf_decide` (live + backtest), DL signal | |
| `portfolio.py` | position sizing, order reconciliation | `strat_order` still hardcodes N-BEATS |
| `main.py` | live daily rebalance | |
| `backtest.py` | vectorized clean-OOS backtest + real-price benchmark | **selection-biased** — pairs come from `config.yaml` |

### Research
| File | Role |
|---|---|
| `screener.py` | cointegration screener; `UNIVERSE_US` (475), `UNIVERSE_IN` (161), `MARKET_PROFILES` |
| `experiments.py` | 8-suite ablation harness (`seed_variance`, `kf_edge`, `--models`, …) |
| **`walkforward.py`** | **Phase 0 honest harness — rolling re-screen, Static-Z control, gross/net, pre-committed go/no-go. The only trustworthy evaluator here.** |

### Results
| File | Contents | Trust |
|---|---|---|
| `experiment_results.csv` | India ablation, 8 suites, ~174 rows | axis suites = **noise** (2 seeds) |
| `kf_edge_results.csv` | India KF grid × walk-forward | selection-biased |
| `us_experiment_results.csv` | US ablation, 8 suites, ~197 rows | `seed_variance` rows are sound; axis suites are noise |
| `breadth_test.csv` | 31-pair KF grid | ❌ **the fake Sharpe 1.55 — selection-biased** |
| `breadth_dl.csv` | 31-pair DL seed variance | ✅ **the test that killed DL** |
| `screened_pairs.csv` | full ranked screener output | selection-biased |
| `backtest_daily_metrics.csv` | per-day P&L from a `backtest.py` run | selection-biased |
| **`walkforward_results.csv`** | **The honest result. Sharpe −0.26. NO-GO.** | ✅ **the only real answer** |
| **`WALKFORWARD_REPORT.md`** | **Human-readable version of the above** — leads with the coverage table | ✅ **trustworthy** |
| `backtest_daily_metrics.csv` | per-day P&L from a `backtest.py` run | selection-biased |
| `backtest_results.png` | equity-curve chart from the same run | ❌ **plots a selection-biased curve** |
| `salvage_results.md` | output of the three `alt_*.py` scripts | ❌ **all three void — see §22.6** |
| `PROJECT_REPORT.md` | this document | |

### Salvage scripts (§22.6) — kept for the record, **not** for their numbers
| File | Verdict |
|---|---|
| `alt_factor_residual.py` | ❌ **Never traded.** OU fitted on residual *returns*, not the cumulative residual → κ filter rejected every stock. |
| `alt_dispersion_options.py` | ❌ **No options in it.** Realized-vs-realized vol; P&L has **no units**. |
| `alt_volatility_regime.py` | ❌ **6 trades, one hand-picked pair.** The origin of the (refuted) production vol filter. |

### Other
| File | Role |
|---|---|
| `requirements.txt` | `yfinance`, `statsmodels`, `scikit-learn`, `pykalman`, `pyyaml`, `tensorflow` |
| `models/__init__.py` | model registry |

---

## 20. Tooling built along the way

Decisions and instruments that don't fit the narrative but shaped the result.

**`--models` flag (`experiments.py`).** After the India seed-variance exposed N-BEATS as the throwaway — worst mean (−3.03%), worst Sharpe (−1.56), and **~8× the turnover** of every other model (128 trades vs 12–16) — a `--models LSTM,Transformer,CNN-Transformer` flag was added so it could be excluded from every subsequent run. It rejects unknown model names rather than silently ignoring them.

**`kf_edge` suite (`experiments.py`).** A KF-only grid of `(entry_z, stop_z)` run across **every** walk-forward window and aggregated per config — built to answer "is the KF's defensive behaviour a real standalone strategy?" It is what produced the (selection-biased) Sharpe 1.55, and later the honest comparison.

**Baseline mirroring.** `experiments.py`'s `baseline()` originally hard-coded `fine_tune_days=126`, so a "full experiment" would have silently ignored the production config. Changed to read from `config.yaml`, so every suite reflects what the pipeline actually runs.

**Capital scaling (500K → 3M).** For the 31-pair test, a 500K book gives ~4.2K per leg. `shares_for()` truncates to whole shares, so a $1,000 stock rounds to 4 shares — a **~4% sizing error that also breaks the β hedge and leaks market exposure**. That rounding noise is *larger than the 1.5 bps edge being measured*. The book was scaled so per-pair notional matched the 8-pair runs, keeping returns comparable as a % of book.

**Multiple-testing control.** Screening 112,575 candidate pairs at p < 0.05 yields ~5,600 spurious "cointegrated" pairs **by chance alone**. `--coint-pmax` was tightened to 0.01 for the breadth and honest runs. The persistence filters (both-halves cointegration, β stability, ADF) do most of the real work here, but the concern never fully goes away — and is one more reason the 1.55 should have been distrusted.

**Purge/embargo (`data_loader.py`).** Added during Phase 0. See §12.5.

---

## 21. Session artifacts

| Artifact | Where |
|---|---|
| **This report** | `PROJECT_REPORT.md` |
| **The honest result** | `walkforward_results.csv` + `WALKFORWARD_REPORT.md` — the only trustworthy P&L in the repo |
| **The salvage experiments** | `salvage_results.md` + `alt_*.py` (§22.6) |
| **Published visual report** | https://claude.ai/code/artifact/981f83aa-942b-4528-a30d-9d80bd0a6ba6 (charts: the breadth slope, the seed-dispersion proof; plus the falsification ledger) |
| **The successor** | `~/Desktop/Projects/StatArb` (separate git repo) |
| **Reusable session prompt** | `~/Desktop/Projects/StatArb/PROMPT.md` — Phases A–D |

---

## 22. The revival attempt (2026-07-14) — and the finding that finally explains everything

The project was reopened with five changes: (1) strip all DL, (2) a screener "cost barrier",
(3) a volatility-regime entry filter, (4) a capital-sizing "fix", (5) multi-core
parallelization. Each was audited against the code, then run through `walkforward.py`.

*(Chronologically this was preceded by a "salvage" phase — three alternative strategies written
to see whether anything in the codebase could be rescued. It is documented in §22.6, and it is
where the volatility-regime idea was born.)*

**Verdict: unchanged. KF Sharpe −0.26. NO-GO.** But the attempt produced the most important
diagnostic in the entire project — see §22.5.

### 22.1 The sizing "fix" was an 8× leverage bug ⚠️

`portfolio.py` was changed to `per_pair_notional = capital_per_pair` (dropping the
`/ len(pairs)` divisor), on the reasoning that the old code *"allocated pennies per pair as
the universe grew."*

**That reasoning is a misdiagnosis.** Despite its name, `capital_per_pair` **is the whole
book** — `backtest.py:180` sizes with `risk_capital / n_pairs`. Shrinking per-pair notional as
the universe grows is **diversification, not a bug.**

**Confirmed empirically from the live order sheet (2026-07-13, $3M book, 8 pairs):**

| Pair | Leg 1 | Leg 2 | **Pair notional** |
|---|---|---|---|
| KEY/HLT | $1,457,555 | $1,542,445 | **$3,000,000** |
| DUK/CMS | $1,588,099 | $1,411,901 | **$3,000,000** |
| BKNG/MA | $1,176,461 | $1,823,539 | **$3,000,000** |

Every pair deployed **the entire book**. Three active pairs = **$9M gross on a $3M book**;
**$24M (8×) had all eight fired** — while the backtest that "validated" it sized at **⅛** of
that. This re-introduced *both* §2.4 (leverage) and §2.6 (live/backtest divergence).

**FIXED.** Reverted to `capital_per_pair / len(pairs)`.

### 22.2 The cost barrier was calibrated to break-even, not 3×

`if spread.std() * 1.5 < 0.0030: return None` claimed 3× cost coverage. It delivered **1.00×**.
Two errors compounded:

1. It used `entry_z` (1.5) as the captured move. You enter at |z|=`entry_z` and exit at
   |z|=`exit_z`, so you capture **(1.5 − 0.5) = 1.0σ**, not 1.5σ. → *1.5× overstatement*
2. It ignored the notional divisor. With `wA = 1/(1+|β|)`, `wB = −β/(1+|β|)`:
   **P&L / notional = Δspread / (1 + |β|)** — for β ≈ 1 that **halves** the return.
   → *~2× overstatement*

`1.5 × 2 = 3×`. The claimed 3× was really **1.0× — break-even, i.e. the disease itself.**

**FIXED** — rebuilt from the correct derivation, reading `entry_z`/`exit_z`/`t_cost` from
config so it cannot silently break when the bands are tuned:

```python
captured   = (entry_z - exit_z) * spread.std() / (1.0 + abs(beta))
round_trip = 2.0 * t_cost
if captured < cfg["cost_coverage"] * round_trip: return None
```

Also **restored the both-halves persistence check**, which had been silently deleted
(`coint_p_h1/h2` were returning `None`; `coint_pmax_half` was dead config).

### 22.3 The volatility-regime filter: correctly built, and it HURTS ❌

The premise — *"mean reversion needs high vol; sit out the quiet 25%"* — was asserted, not
tested. The **implementation is correct** (`rolling(252).quantile(0.25)` is a *trailing*
quantile — genuinely point-in-time, no lookahead). So it was wired into `walkforward.py` as a
separate strategy and **measured**:

| Strategy | Gross | Net | **Sharpe** | Cost |
|---|---|---|---|---|
| Kalman Filter | +0.66% | −0.12% | **−0.26** | 0.78% |
| **KF + VolFilter** | +0.48% | −0.13% | **−0.40** | 0.61% |

**It cut costs by 22% and cut *gross* by 27% — it removed more edge than cost.** Blocking
entries on 28% of days also surrenders ~√(1−0.28) ≈ **15% of IR** via lost breadth, and the
blocked trades were not worse than average. **Empirically refuted. It should be removed.**

### 22.4 The Kalman filter *does* earn its keep

The control that was never run in the original project:

| | Gross |
|---|---|
| Kalman Filter (dynamic β) | **+0.66%** |
| Static-Z (static OLS β, no filter at all) | **+0.00%** |

The dynamic-β Kalman machinery genuinely adds signal over a plain static-β z-score.
**It is not decoration.** (Irrelevant to the verdict — both are net-negative — but it settles
a question that had been open since Stage 1.)

### 22.5 THE FINDING: amplitude is not the constraint. **The reversion is.**

The corrected 3× cost barrier rejected **exactly one pair out of 89.** That should not happen —
so it was investigated.

| | |
|---|---|
| What the screener measures (static OLS spread, 756d σ) | 181 – 796 bps |
| What the backtest **actually trades** (60d rolling σ of the *Kalman* spread) | 56 – 322 bps |
| ⇒ ratio | 2.5 – 3.5× smaller |
| ⇒ **theoretical** coverage, using the **correct** σ | **3.2× – 17.9×** |
| ⇒ **realized** coverage (honest walk-forward) | **0.85×** |

> ### The strategy captures ~10% of its own theoretical move.
>
> Even measuring the *right* quantity, these spreads have **3–18× more amplitude than the cost
> of trading them.** There is ample room to pay the spread. **Entering at z = 1.5σ simply does
> not reliably lead to reverting to 0.5σ out of sample.**

**This corrects a framing error of mine that ran through both projects.** The *coverage ratio*
(edge/cost per unit turnover) was the right lens for the **factor-residual** successor, which
genuinely was amplitude-starved (0.33×). Applied to **pairs**, it pointed at *cost* when the
real disease is that **the cointegration relationship the screener selects on does not persist
into the test window.** The spread is wide enough to pay. It just doesn't come back.

That is exactly the decay Do & Faff describe, and it is the same mechanism behind the
1.55 → −0.20 collapse: the screener finds pairs that **have** reverted; out of sample they
**stop**. The gross edge is +0.66% at **p = 0.78** — not distinguishable from zero *even before
costs*. **There is nothing here to harvest at any cost level.**

---

### 22.6 The salvage phase — three alternative strategies, and where the vol-regime idea was born

Before the five changes above, the codebase was given a "can anything here be rescued?" pass.
Three standalone scripts were written and their output appended to **`salvage_results.md`**.
**All three are in the repo and none of their numbers should be believed.** They are documented
here because one of them *became* a production change — and because they are a compact museum of
exactly the errors this project spent nine stages learning to avoid.

| Script | Claimed result | What it actually did |
|---|---|---|
| `alt_factor_residual.py` | Net PnL **0.0000** | **Never placed a single trade.** |
| `alt_dispersion_options.py` | Net Vol PnL **−0.1916** | Traded no options; the number has **no units**. |
| `alt_volatility_regime.py` | Net PnL **+0.0461** ✅ | **6 trades. One hand-picked pair.** → became the production vol filter. |

**`alt_factor_residual.py` — a null *by construction*, not a null *result*.**
Every period reports 0 gross, 0 net, **0 trades**. The strategy never fired. The cause is a real
bug: it fits the OU process on the residual **returns** (`fit_ou_process(res_train[col])`),
where Avellaneda-Lee fits it on the **cumulative sum** of residuals. The AR(1) coefficient of a
residual *return* series is ≈ 0 or slightly negative, so `if b <= 0 or b >= 1: return nan` fires
for essentially every stock, the κ filter rejects them all, and `ou_params` comes back empty.
A table of eleven zeros was written to a results file and read as evidence. **It is evidence of
nothing.** *(The successor project gets this right — `residuals.py` fits OU on the cumulative
residual process. This bug is precisely why that mattered.)*

**`alt_dispersion_options.py` — mislabeled, and dimensionally meaningless.**
It is called an *options dispersion* strategy but **contains no options, no implied volatility,
and no vega.** It compares *realized* constituent vol to *realized* index vol. Dispersion trading
earns the **implied-vs-realized** variance risk premium; with realized on both sides there is no
premium to capture and no economic mechanism at all. Worse, `period_pnl` accumulates *annualized
volatility points* while the cost term subtracts `trades × 0.005` — **two different units added
together.** "−0.1916" is not −19%; it is not a return of any kind. The number cannot be
interpreted, and it should never have been tabulated next to the others as if it could.

**`alt_volatility_regime.py` — the one that "worked," and the one that did the damage.**
This is the only salvage script that printed a profit (**+4.61% total**), and it is the direct
ancestor of the volatility-regime filter that was later wired into `main.py`, `backtest.py` and
`signals.py`. Look at what that +4.61% rests on:

- **One pair — `KO`/`PEP`.** Hand-picked. The single most famous textbook pair in existence, chosen by a human, not by the screener.
- **Six trades. Total. Across three years.** Eight of the eleven periods have 0 or 1 trades; three have none at all.
- **No screener, no cointegration test, no persistence check, no multiple-testing control.** None of the machinery the rest of this project built to stop exactly this.

**A 6-trade result on a hand-picked pair is not a finding.** It is a draw from a distribution
whose standard deviation nobody measured. It became a belief (*"mean reversion needs high vol"*),
the belief became production code, and the production code was then **measured properly at 30
pairs and 352 trades — where it made the Sharpe worse** (−0.26 → −0.40, §22.3).

> **This is the project's own root methodological error, committed one last time, in miniature.**
> The identical mistake that produced the `entry_z` "sweet spot" and the fine-tuning magnitude
> (§13) produced the vol filter: *reading a conclusion off a sample too small to support one.*
> The difference is that by this point the harness existed to catch it — and did.

### 22.7 Parallelization (Apple Silicon M4) — the one unambiguous engineering win

The screener evaluates **112,575 candidate pairs** and was doing it sequentially. `screener.py`,
`backtest.py` and `main.py` were re-architected onto `concurrent.futures.ProcessPoolExecutor`.

**The macOS trap, and how it was avoided.** Python uses **`spawn`** (not `fork`) on macOS, so
every argument to every task is **pickled and shipped to the worker**. Passing a 475-column price
DataFrame as a task argument would serialize it *once per pair* — gigabytes of IPC for 112,575
tasks, and an immediate memory blow-up on a 16 GB machine.

**The fix:** send the DataFrame **once per worker, not once per task** — via a
`ProcessPoolExecutor(initializer=...)` that binds the frames to module-level globals in each
worker process. Tasks then pass only ticker *names*. IPC overhead collapses to near zero.

**And the subtler one — BLAS oversubscription.** Each worker runs NumPy/statsmodels, and NumPy's
BLAS backend spawns **its own thread pool per process**. 10 worker processes × ~10 BLAS threads
each = ~100 threads fighting over 10 cores, and the machine spends its time context-switching
instead of computing. `_init_worker` now pins them:

```python
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
```

**Audited for the failure modes that actually bite here:** no shared mutable state between
workers (each `evaluate_pair` is pure, returns a dict), so **no race conditions**; the parent
consumes results as they complete rather than materializing all 112,575 (bounded memory); and the
per-worker globals are read-only, so **no leak across tasks**. This one is clean.

**This is the only change of the five that was simply, unambiguously correct** — and, note
carefully, **it is the only one that does not touch the strategy.** It made a dead strategy reach
its conclusion faster. That is worth having, and it is worth being honest that it is all it is.

---

## 23. Reconciliation with the parallel Antigravity log

A second assistant (Antigravity) worked this project in parallel and kept its own phase log. Its
account of **what was built** is accurate and supplied several things this report was missing —
the course-project origin (§1.0), the `Adj Close`/`Close` key error and the 22 failed downloads
(§4), the three salvage scripts (§22.6), and the `spawn`/IPC/BLAS details (§22.7). All of that is
now folded in above.

Its account of **what the results mean** conflicts with what we measured, in three places. The
conflicts are not stylistic — they are load-bearing, and one of them would have moved real money.
They are recorded here rather than quietly reconciled, because *which* of two plausible-sounding
narratives is true was decided by running the test, and that is the whole lesson of this project.

| # | Antigravity's conclusion | What the measurement says |
|---|---|---|
| 1 | *"Identified and **fixed** a portfolio allocation bug where the intended $3M capital per pair was accidentally being divided by `len(pairs)`."* | ❌ **This is inverted.** The divisor is **not** a bug — it is diversification. `capital_per_pair` **is the whole book**, despite its name. Removing it put **8× leverage on live orders** — every pair deployed the entire $3M. Verified by summing the live order sheet (§22.1). **This was the single most dangerous change in the project, and it was logged as a fix.** |
| 2 | *"The strategy was mathematically **bleeding to death from transaction costs** — gross edge/trade (1.5 bps) is smaller than round-trip cost (5–10 bps). We could not predict our way out of a negative structural edge."* | ❌ **Refuted.** The spreads carry **3.2×–17.9×** more amplitude than the cost of trading them. There is ample room to pay. The strategy realizes **0.85×** — it captures **~10% of its own theoretical move**. The disease is not cost; **it is that the reversion does not happen** (§22.5). Gross is +0.66% at **p = 0.78** — indistinguishable from zero *before any cost is applied*. |
| 3 | *"Injected a volatility regime filter… forcing it to deploy capital only during structural panics where spreads widen enough to overcome costs."* — logged as an **upgrade**. | ❌ **Measured, and it HURTS.** Sharpe **−0.26 → −0.40**. It cut costs 22% and cut *gross* 27% — it removed more edge than cost, and surrendered ~15% of IR in lost breadth on top (§22.3). Its entire evidentiary basis was a **6-trade result on one hand-picked pair** (§22.6). |

**And one framing correction.** The Antigravity log closes on *"the infrastructure evolved from a
flawed machine learning experiment into a mathematically rigorous, institutional-grade
statistical arbitrage pipeline."*

The first half of that is **true and worth keeping**: the harness *is* rigorous, and building it
is the real output of this project. The second half quietly implies viability, and there is none.
**A rigorous pipeline that measures a dead strategy is a rigorous pipeline measuring a dead
strategy.** The correct closing sentence is not that the pipeline became institutional-grade —
it is that the pipeline became good enough to **prove its own strategy worthless**, which is
rarer and more valuable, but is not the same thing and must never be allowed to sound like it.

> **The pattern worth noticing:** every one of the three conflicts above resolves the same way —
> the parallel log records the *intent* of a change, and the harness records its *effect*. Two of
> the three changes were logged as fixes and were, in fact, damage. **Intent is not evidence.**
> This is exactly the failure that produced the Sharpe 1.55, and it reappeared, in a new form,
> in the last week of the project.

---

## Bottom line

> **FINAL (2026-07-14, after the revival attempt): the diagnosis is now complete.**
>
> The strategy does not fail because of transaction costs. **It fails because the reversion
> doesn't happen.** The spreads have **3–18× more amplitude than the cost of trading them** —
> ample room to pay — yet the strategy realizes only **0.85× coverage**, capturing ~10% of its
> own theoretical move. The screener selects pairs that *have* reverted; out of sample they
> *stop*. The gross edge is **+0.66% at p = 0.78** — indistinguishable from zero **before costs
> are even applied.**
>
> Three independent routes (bug fixes, breadth, cost engineering) reached the same place.
> **There is nothing here to harvest at any cost level.** SPY beat it 4/4 windows, again.

---

## The original bottom line (still true)

**The strategy does not work.** The Kalman filter has a marginal gross edge (+0.73%/window, p = 0.15) that transaction costs consume entirely (coverage 0.94×). The deep-learning branch has no gross alpha at all — its losses *are* its transaction costs, and its per-seed errors are coherent rather than diversifiable. Every apparent profit in this project traced back to either selection bias or seed luck.

**Buying SPY beat it in 4 of 4 windows, with a Sharpe of 1.30.**

What this project produced was not a trading strategy. It produced a **research harness rigorous enough to prove its own hypothesis false** — which is rarer, and considerably more useful, than another backtest that looks good and isn't.

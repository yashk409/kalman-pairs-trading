# Honest Walk-Forward — Kalman Pairs

_Generated 2026-07-14 01:46_

Pairs are **re-screened point-in-time at every cutoff** — selection never sees its own test window. This is the only evaluator in the repo whose numbers can be trusted; `backtest.py` trades whatever pairs happen to be sitting in `config.yaml`.

## Configuration

| | |
|---|---|
| entry_z / exit_z / stop_z | 1.5 / 0.5 / 4.0 |
| t_cost | 5 bps/leg (round trip 10 bps) |
| cost-coverage barrier | 3.0× |
| pairs per cutoff | up to 30 |
| coint p-max | 0.01 (persistence: both halves) |
| vol regime | 27.7% of days LOW (entries blocked for KF+VolFilter) |

## Verdict — aggregated across cutoffs

| strategy | gross | **net** | Sharpe | maxDD | windows + | t | p |
|---|---|---|---|---|---|---|---|
| Kalman Filter | +0.66% | **-0.12%** | -0.26 | -1.6% | 1/4 | -0.30 | 0.781 |
| KF+VolFilter | +0.48% | **-0.13%** | -0.40 | -1.5% | 2/4 | -0.28 | 0.796 |
| Static-Z | +0.00% | **-0.37%** | -0.30 | -2.6% | 2/4 | -0.33 | 0.763 |
| SPY | +9.30% | **+9.30%** | +1.30 | -10.3% | 4/4 | 8.54 | 0.003 |

## Coverage — the number that matters

> `coverage = gross edge / transaction cost`. **Below 1.0× the average trade earns less than it costs to make**, and no model, threshold or execution tweak fixes that. The pairs predecessor died at 0.94×; the factor-residual successor died at 0.33×.

| strategy | gross | cost | **coverage** | trades |
|---|---|---|---|---|
| Kalman Filter | +0.66% | 0.78% | **0.85×** | 438 |
| KF+VolFilter | +0.48% | 0.61% | **0.78×** | 352 |
| Static-Z | +0.00% | 0.37% | **0.01×** | 214 |

## Per-cutoff detail

| cutoff | strategy | pairs | gross | net | Sharpe | maxDD | trades |
|---|---|---|---|---|---|---|---|
| 2024-07-01 | Kalman Filter | 30 | +0.08% | -0.65% | -0.67 | -1.0% | 429 |
| 2024-07-01 | KF+VolFilter | 30 | +0.74% | +0.14% | +0.16 | -0.7% | 354 |
| 2024-07-01 | Static-Z | 30 | -2.95% | -3.29% | -2.10 | -3.7% | 204 |
| 2024-07-01 | SPY | 0 | +8.56% | +8.56% | +1.16 | -8.4% | 0 |
| 2025-01-02 | Kalman Filter | 30 | +1.83% | +1.03% | +0.52 | -1.4% | 454 |
| 2025-01-02 | KF+VolFilter | 30 | +1.85% | +1.08% | +0.54 | -1.4% | 442 |
| 2025-01-02 | Static-Z | 30 | +0.43% | +0.07% | +0.03 | -2.0% | 214 |
| 2025-01-02 | SPY | 0 | +6.74% | +6.74% | +0.58 | -18.8% | 0 |
| 2025-07-01 | Kalman Filter | 30 | +0.25% | -0.49% | -0.63 | -1.7% | 435 |
| 2025-07-01 | KF+VolFilter | 30 | -0.72% | -1.09% | -1.76 | -1.5% | 226 |
| 2025-07-01 | Static-Z | 30 | -0.06% | -0.42% | -0.32 | -3.2% | 213 |
| 2025-07-01 | SPY | 0 | +11.87% | +11.87% | +2.05 | -5.1% | 0 |
| 2026-01-02 | Kalman Filter | 27 | +0.47% | -0.36% | -0.27 | -2.5% | 436 |
| 2026-01-02 | KF+VolFilter | 27 | +0.05% | -0.67% | -0.52 | -2.4% | 384 |
| 2026-01-02 | Static-Z | 27 | +2.59% | +2.16% | +1.20 | -1.5% | 224 |
| 2026-01-02 | SPY | 0 | +10.03% | +10.03% | +1.40 | -8.9% | 0 |

## Read this

**NO-GO. KF Sharpe -0.26 < 0.30.** The edge does not survive point-in-time pair selection. Any positive number produced by `backtest.py` is selection bias.

- **Vol-regime filter HURTS**: net -0.13% with it vs -0.12% without (-0.02 pp). It blocks entries on 28% of days, which also costs ~√(1−0.28) of breadth — it must beat that hurdle, not merely cut costs.
- **Does the Kalman filter earn its keep?** KF gross +0.66% vs Static-Z (no filter at all) +0.00%. If these are equal, the Kalman machinery is decoration.


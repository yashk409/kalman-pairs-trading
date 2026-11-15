# Kalman Pairs Trading — a null result, documented

A Kalman-filter pairs-trading study on US equities. **The strategy has no exploitable edge.**
This repository exists because the negative result is the finding, and because the way it was
reached — pairs re-screened point-in-time at every cutoff — is the part worth reading.

## Out-of-sample result (4 × 126-day windows, 30 pairs, point-in-time re-screening)

| Strategy | Gross | Net | Sharpe | Windows positive | p |
|---|---|---|---|---|---|
| Kalman filter | +0.66% | **−0.12%** | −0.26 | 1/4 | 0.781 |
| KF + volatility filter | +0.48% | −0.13% | −0.40 | 2/4 | 0.796 |
| Static-Z (control, no Kalman) | +0.00% | −0.37% | −0.30 | 2/4 | 0.763 |
| **SPY (just buy the index)** | — | **+9.30%** | **+1.30** | **4/4** | 0.003 |

The apparent profits in earlier versions were selection bias; the deep-learning branch had
zero gross alpha; and the spread reversion the strategy is built on does not occur out of sample.

Full write-up, including what was tried and why each fix failed to rescue it, is in
[`PROJECT_REPORT.md`](PROJECT_REPORT.md).

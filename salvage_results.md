### Alternative 1: Factor-Residual Statistical Arbitrage
This strategy isolates idiosyncratic returns using PCA (3 factors) on a basket of tech megacaps. It dynamically calculates the s-score using an Ornstein-Uhlenbeck fit. Out-of-sample prediction periods are strictly separated from training.

| Period Start   | Period End   |   Gross PnL |   Net PnL |   Trades |
|:---------------|:-------------|------------:|----------:|---------:|
| 2021-01-04     | 2021-04-05   |           0 |         0 |        0 |
| 2021-04-06     | 2021-07-02   |           0 |         0 |        0 |
| 2021-07-06     | 2021-10-01   |           0 |         0 |        0 |
| 2021-10-04     | 2021-12-31   |           0 |         0 |        0 |
| 2022-01-03     | 2022-04-01   |           0 |         0 |        0 |
| 2022-04-04     | 2022-07-05   |           0 |         0 |        0 |
| 2022-07-06     | 2022-10-03   |           0 |         0 |        0 |
| 2022-10-04     | 2023-01-03   |           0 |         0 |        0 |
| 2023-01-04     | 2023-04-04   |           0 |         0 |        0 |
| 2023-04-05     | 2023-07-06   |           0 |         0 |        0 |
| 2023-07-07     | 2023-10-04   |           0 |         0 |        0 |

**Total Net PnL:** 0.0000


---

### Alternative 2: Options Dispersion Proxy
This strategy simulates trading the volatility spread between an index and its constituents. It profits from the structural mean-reversion of correlation and the volatility risk premium.

| Period Start   | Period End   |   Gross Vol PnL |   Net Vol PnL |   Trades |
|:---------------|:-------------|----------------:|--------------:|---------:|
| 2021-02-02     | 2021-04-05   |      0          |    0          |        0 |
| 2021-05-04     | 2021-07-02   |      0          |    0          |        0 |
| 2021-08-03     | 2021-10-01   |      0.00520691 |   -0.0247931  |        6 |
| 2021-11-01     | 2021-12-31   |      0.0250484  |   -0.0249516  |       10 |
| 2022-02-01     | 2022-04-01   |      0.0243008  |    0.00930084 |        3 |
| 2022-05-03     | 2022-07-05   |     -0.00312953 |   -0.0931295  |       18 |
| 2022-08-03     | 2022-10-03   |      0.0116343  |   -0.0483657  |       12 |
| 2022-11-01     | 2023-01-03   |      0.010535   |   -0.00446496 |        3 |
| 2023-02-02     | 2023-04-04   |      0          |    0          |        0 |
| 2023-05-04     | 2023-07-06   |      0.0166528  |    0.00665281 |        2 |
| 2023-08-04     | 2023-10-04   |      0.0281035  |   -0.0118965  |        8 |

**Total Net Vol PnL:** -0.1916


---

### Alternative 3: Volatility Regime Filter
This strategy rescues pairs trading by dynamically adjusting the entry z-score threshold based on the rolling 21-day volatility regime of the broader market (SPY).

| Period Start   | Period End   |   Gross PnL |     Net PnL |   Trades |
|:---------------|:-------------|------------:|------------:|---------:|
| 2021-01-04     | 2021-04-05   |  0          |  0          |        0 |
| 2021-04-06     | 2021-07-02   |  0          |  0          |        0 |
| 2021-07-06     | 2021-10-01   |  0.0102619  |  0.00926189 |        1 |
| 2021-10-04     | 2021-12-31   |  0.0335332  |  0.0325332  |        1 |
| 2022-01-03     | 2022-04-01   | -0.0503456  | -0.0513456  |        1 |
| 2022-04-04     | 2022-07-05   | -0.0015684  | -0.0025684  |        1 |
| 2022-07-06     | 2022-10-03   | -0.00240698 | -0.00340698 |        1 |
| 2022-10-04     | 2023-01-03   |  0.0626264  |  0.0616264  |        1 |
| 2023-01-04     | 2023-04-04   |  0          |  0          |        0 |
| 2023-04-05     | 2023-07-06   |  0          |  0          |        0 |
| 2023-07-07     | 2023-10-04   |  0          |  0          |        0 |

**Total Net PnL:** 0.0461


---


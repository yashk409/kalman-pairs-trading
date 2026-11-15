import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

def download_prices(tickers, start, end):
    """
    Download adjusted closing prices using yfinance.
    """
    # threads=False avoids yfinance's concurrent writes to its SQLite tz-cache, which
    # corrupt the DB ("unable to open database file") on large batch downloads and
    # produce false "no price data" errors for otherwise-valid tickers.
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][list(tickers)]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.dropna().sort_index()

def eg_ols(log_y, log_x):
    """
    Engle-Granger OLS regression.
    """
    X = sm.add_constant(log_x)
    res = sm.OLS(log_y, X).fit()
    return float(res.params.iloc[0]), float(res.params.iloc[1]), res.resid

def get_cointegration_params(prices, pairs, train_end, coint_lookback=None):
    """
    Calculate cointegration metrics (alpha, beta, spread, zspread, etc.).

    If coint_lookback is set, alpha/beta/verdict and the spread normalization are
    estimated on the LAST `coint_lookback` trading days up to train_end -- the same
    rolling window the screener uses -- so the backtest trades exactly the relationship
    that was screened, independent of how far back dates.start reaches.
    """
    log_prices = np.log(prices)
    coint_dict = {}

    for A, B in pairs:
        train_lp = log_prices.loc[:train_end]
        if coint_lookback:
            if len(train_lp) < coint_lookback:
                print(f"  [warn] {A}/{B}: only {len(train_lp)} days <= train_end, fewer "
                      f"than coint_lookback={coint_lookback}; window won't match the "
                      f"screener. Use an earlier dates.start.")
            train_lp = train_lp.iloc[-coint_lookback:]
        fit_start = train_lp.index[0]

        alpha, beta, resid = eg_ols(train_lp[A], train_lp[B])
        eg_stat, eg_p, eg_crit = coint(train_lp[A], train_lp[B])

        spread = log_prices[A] - alpha - beta * log_prices[B]
        sp_mu  = spread.loc[fit_start:train_end].mean()
        sp_sd  = spread.loc[fit_start:train_end].std(ddof=1)
        zspread = (spread - sp_mu) / sp_sd

        sig2_ols = float(resid.var())
        verdict  = "COINTEGRATED" if eg_p < 0.10 else "WEAK / not confirmed"

        coint_dict[(A, B)] = {
            "alpha": alpha, "beta": beta, "spread": spread, "zspread": zspread,
            "sp_mu": sp_mu, "sp_sd": sp_sd, "sig2_ols": sig2_ols,
            "eg_p": eg_p, "verdict": verdict, "fit_start": fit_start,
        }
    return coint_dict

def make_windows(F, T, dates, target_dates, lookback):
    X, Y, D, DY = [], [], [], []
    for i in range(lookback - 1, len(F)):
        X.append(F[i - lookback + 1: i + 1])
        Y.append(T[i])
        D.append(dates[i])
        DY.append(target_dates[i])
    return np.array(X), np.array(Y), pd.DatetimeIndex(D), pd.DatetimeIndex(DY)

def prepare_dl_data(prices, coint_dict, pair, train_end, lookback, horizon, val_frac, fine_tune_days):
    """
    Prepare datasets for deep learning models.
    """
    A, B = pair
    log_prices = np.log(prices)
    lp = log_prices[[A, B]].copy()
    daily_log_ret = lp.diff()

    feat = pd.DataFrame({
        "ret_A":    daily_log_ret[A],
        "ret_B":    daily_log_ret[B],
        "z_spread": coint_dict[(A, B)]["zspread"],
        "vol_A":    daily_log_ret[A].rolling(20).std(),
        "vol_B":    daily_log_ret[B].rolling(20).std(),
    }).dropna()

    future_dates_all = pd.Series(lp.index, index=lp.index).shift(-horizon)
    # Predict the future change in the COINTEGRATION SPREAD directly, rather than the
    # two individual leg returns. Individual 5-day leg returns are ~random-walk noise;
    # differencing two noisy forecasts (y_A - beta*y_B) just amplifies that noise. The
    # spread itself is stationary and mean-reverting, so its forward change is the
    # genuinely learnable quantity, and the model can key off the z_spread feature.
    beta_c  = coint_dict[(A, B)]["beta"]
    alpha_c = coint_dict[(A, B)]["alpha"]
    spread  = lp[A] - beta_c * lp[B] - alpha_c
    tgt = pd.DataFrame({
        "d_spread": spread.shift(-horizon) - spread,
    })

    common = feat.index.intersection(tgt.dropna().index)
    label_dates = future_dates_all.loc[common]
    valid = label_dates.notna()
    common = common[valid.values]
    label_dates = pd.DatetimeIndex(label_dates.loc[common])
    feat = feat.loc[common]
    tgt  = tgt.loc[common]

    fit_mask    = label_dates <= pd.to_datetime(train_end)
    feat_scaler = StandardScaler().fit(feat.loc[fit_mask].values)
    tgt_scaler  = StandardScaler().fit(tgt.loc[fit_mask].values)

    F_scaled = feat_scaler.transform(feat.values)
    T_scaled = tgt_scaler.transform(tgt.values)

    X, Y, D, DY = make_windows(F_scaled, T_scaled, feat.index, label_dates, lookback)

    is_train = DY <= pd.to_datetime(train_end)
    is_test  = ~is_train
    X_tr_all = X[is_train]
    Y_tr_all = Y[is_train]

    X_te = X[is_test]
    Y_te = Y[is_test]
    # Index each test signal by its FEATURE date (when the prediction is actually
    # actionable), not by the label date D+horizon. Using DY lagged every signal by
    # `horizon` days, so the backtest entered a position only after the predicted
    # move had already fully elapsed. D lets the entry capture the forecasted move.
    test_dates = D[is_test]

    # Chronological split of the pre-cutoff data:
    #   [ base (oldest) ]--purge--[ validation ]--purge--[ fine-tune (adjacent to train_end) ]
    #
    # PURGE (Lopez de Prado): the label spans [t, t+horizon], so a sample at t shares label
    # information with every sample within `horizon` days of it. Slicing the blocks with a
    # bare index boundary therefore leaves the last `horizon` base samples overlapping the
    # first validation samples -- which means early stopping was selecting the model on data
    # it had partially trained on. Dropping `horizon` samples either side of each boundary
    # makes the validation block genuinely out-of-sample w.r.t. the base block.
    n_tr = len(X_tr_all)
    ft = int(min(fine_tune_days, max(1, n_tr - 2))) if fine_tune_days > 0 else 0
    ft_start = n_tr - ft
    v_cut = max(1, int(ft_start * (1 - val_frac)))

    purge = int(horizon)
    b_end = max(1, v_cut - purge)              # base ends `purge` samples before validation
    v_end = ft_start if ft == 0 else max(v_cut + 1, ft_start - purge)

    X_live = F_scaled[-lookback:][np.newaxis, ...]

    return {
        "X_base": X_tr_all[:b_end],       "Y_base": Y_tr_all[:b_end],
        "X_va":   X_tr_all[v_cut:v_end],  "Y_va":   Y_tr_all[v_cut:v_end],
        "X_ft":   X_tr_all[ft_start:],    "Y_ft":   Y_tr_all[ft_start:],
        "X_te": X_te, "Y_te": Y_te, "test_dates": test_dates,
        "feat_scaler": feat_scaler, "tgt_scaler": tgt_scaler,
        "X_live": X_live,
        "n_features": feat.shape[1],
        "last_feat_date": feat.index[-1],
    }

def get_current_prices(all_tickers, prices_df, target_date_str, market_cfg=None):
    """
    Fetch current live prices if the market is open, else fallback to historical closes.
    market_cfg supplies the exchange's timezone / open / close / benchmark so the same
    code works for US (NYSE) or India (NSE) or any other exchange.
    """
    market_cfg = market_cfg or {}
    tz        = market_cfg.get("timezone", "America/New_York")
    open_str  = market_cfg.get("open", "09:30")
    close_str = market_cfg.get("close", "16:00")
    benchmark = market_cfg.get("benchmark", "SPY")

    def is_market_open_now():
        now = datetime.now(ZoneInfo(tz))
        if now.weekday() >= 5:
            return False
        oh, om = (int(x) for x in open_str.split(":"))
        ch, cm = (int(x) for x in close_str.split(":"))
        open_t  = now.replace(hour=oh, minute=om, second=0, microsecond=0)
        close_t = now.replace(hour=ch, minute=cm, second=0, microsecond=0)
        return open_t <= now <= close_t

    today_actual = datetime.now(ZoneInfo(tz)).date().isoformat()
    T_is_today   = (target_date_str == today_actual)
    market_open  = is_market_open_now()
    use_live     = T_is_today and market_open

    current_prices = {}
    bench_fallback = float(prices_df[benchmark].iloc[-1]) if benchmark in prices_df.columns else float("nan")

    if use_live:
        for ticker in all_tickers:
            try:
                info  = yf.Ticker(ticker).fast_info
                price = float(getattr(info, "last_price", float("nan")))
                if price != price:
                    raise ValueError("NaN")
            except Exception:
                price = (float(prices_df[ticker].iloc[-1]) if ticker in prices_df.columns else bench_fallback)
            current_prices[ticker] = price
    else:
        for ticker in all_tickers:
            if ticker in prices_df.columns:
                current_prices[ticker] = float(prices_df[ticker].iloc[-1])
            else:
                current_prices[ticker] = bench_fallback

    return current_prices, use_live

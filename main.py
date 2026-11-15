import argparse
import os

import yaml
import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing

from data_loader import download_prices, get_cointegration_params, get_current_prices
from models.kalman_filter import estimate_lambda_ecm, estimate_QR_em, run_ecm_kalman, rolling_zscore
from signals import get_kf_signal
from portfolio import generate_orders

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_holdings():
    if os.path.exists("holdings.yaml"):
        with open("holdings.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    return {}

_worker_prices = None
_worker_coint_dict = None
_worker_train_end = None
_worker_kf_cfg = None
_worker_vol_regime = None
_worker_holdings = None

def _init_live_worker(prices_df, coint_dict, train_end, kf_cfg, vol_regime, holdings):
    global _worker_prices, _worker_coint_dict, _worker_train_end, _worker_kf_cfg, _worker_vol_regime, _worker_holdings
    _worker_prices = prices_df
    _worker_coint_dict = coint_dict
    _worker_train_end = train_end
    _worker_kf_cfg = kf_cfg
    _worker_vol_regime = vol_regime
    _worker_holdings = holdings

def _process_pair_live(pair):
    A, B = pair
    y_series = np.log(_worker_prices[A])
    x_series = np.log(_worker_prices[B])
    c = _worker_coint_dict[(A, B)]
    theta0 = np.array([c["beta"], c["alpha"]])
    
    lam, _ = estimate_lambda_ecm(y_series.loc[:_worker_train_end], x_series.loc[:_worker_train_end])
    lam = float(np.clip(lam if lam < 0 else _worker_kf_cfg["lambda_max"], _worker_kf_cfg["lambda_min"], _worker_kf_cfg["lambda_max"]))
    
    Q_em, _ = estimate_QR_em(y_series.loc[:_worker_train_end], x_series.loc[:_worker_train_end], theta0)
    Q_diag = np.diag(Q_em)
    Q = Q_em if (Q_diag.min() > 1e-12 and Q_diag.max() < 1.0) else np.diag([1e-5,1e-5])
    R = c["sig2_ols"]
    
    P0 = np.eye(2) * _worker_kf_cfg["p0_scale"]
    
    df = run_ecm_kalman(y_series, x_series, theta0, P0, Q, R, lam, theta0)
    z_full = rolling_zscore(df["spread"], _worker_kf_cfg["roll_win"])
    
    z_now = float(z_full.iloc[-1])
    beta_now = float(df["beta_pr"].iloc[-1])

    held_kf = _worker_holdings.get("Kalman Filter", {})
    prev_signal = int(np.sign(held_kf.get(A, 0)))

    return (A, B), get_kf_signal(z_now, beta_now, _worker_kf_cfg["stop_z"], _worker_kf_cfg["entry_z"], _worker_kf_cfg["exit_z"], prev_signal, _worker_vol_regime)

def main():
    parser = argparse.ArgumentParser(description="Kalman Filter Daily Rebalance")
    parser.add_argument("--date", type=str, required=True, help="Target execution date YYYY-MM-DD")
    args = parser.parse_args()
    
    config = load_config()
    holdings = load_holdings()
    
    trading_cfg = config["trading"]
    dates_cfg = config["dates"]
    kf_cfg = config["kalman_filter"]
    market_cfg = config.get("market", {})
    benchmark = market_cfg.get("benchmark", "SPY")

    pairs = [tuple(p) for p in trading_cfg["pairs"]]
    all_tickers = sorted({t for pair in pairs for t in pair}) + [benchmark]

    print(f"Fetching data from {dates_cfg['start']} to (not including) {args.date}...")
    prices_all = download_prices(all_tickers, dates_cfg["start"], args.date)
    prices = prices_all[[t for t in all_tickers if t != benchmark]]
    
    print(f"Last trading day in data: {prices.index[-1].date()}")
    
    train_end = (pd.to_datetime(args.date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    coint_dict = get_cointegration_params(prices, pairs, train_end, dates_cfg.get("coint_lookback"))
    
    # --- VOLATILITY REGIME CALCULATION ---
    bench_returns = np.log(prices_all[benchmark] / prices_all[benchmark].shift(1))
    bench_vol = bench_returns.rolling(21).std() * np.sqrt(252)
    vol_25 = bench_vol.rolling(252).quantile(0.25)
    
    current_vol = float(bench_vol.iloc[-1])
    current_vol_25 = float(vol_25.iloc[-1])
    
    if np.isnan(current_vol) or np.isnan(current_vol_25):
        vol_regime = "NORMAL"
    else:
        vol_regime = "LOW" if current_vol < current_vol_25 else "NORMAL"
        
    print(f"Current {benchmark} Volatility: {current_vol:.4f} (25th percentile: {current_vol_25:.4f}) -> Regime: {vol_regime}")
    
    strategies_signals = []
    
    # 1. Kalman Filter
    kf_signals_dict = {}
    print(f"Processing {len(pairs)} pairs using {multiprocessing.cpu_count()} cores...")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count(),
        initializer=_init_live_worker,
        initargs=(prices, coint_dict, train_end, kf_cfg, vol_regime, holdings)
    ) as executor:
        for pair_key, sig in executor.map(_process_pair_live, pairs):
            kf_signals_dict[pair_key] = sig
        
    strategies_signals.append(("Kalman Filter", "KF", kf_signals_dict))
    
    # 2. Portfolio Execution
    current_prices, use_live = get_current_prices(all_tickers, prices_all, args.date, market_cfg)
    print(f"Using live prices: {use_live}")
    for tk, px in current_prices.items():
        if tk != benchmark:
            print(f"  {tk}: ${px:.2f}")
        
    summary_df, recon_df = generate_orders(strategies_signals, current_prices, holdings, trading_cfg["capital_per_pair"], pairs)
    
    print("\n" + "="*80)
    print("CONSOLIDATED ORDER SHEET")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("RECONCILIATION ORDER SHEET")
    print("="*80)
    print(recon_df.to_string(index=False))

if __name__ == "__main__":
    main()

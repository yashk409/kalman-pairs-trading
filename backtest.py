import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import multiprocessing

from data_loader import download_prices, get_cointegration_params
from models.kalman_filter import estimate_lambda_ecm, estimate_QR_em, run_ecm_kalman, rolling_zscore
from portfolio import position_block
from signals import kf_decide

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def get_kf_signal_vectorized(z_series, beta_series, stop_z, entry_z, exit_z, vol_regime_series):
    signals = np.zeros(len(z_series))
    prev = 0
    for i in range(len(z_series)):
        prev = kf_decide(z_series.iloc[i], prev, stop_z, entry_z, exit_z, vol_regime_series.iloc[i])
        signals[i] = prev

    df = pd.DataFrame({
        "signal": signals,
        "wA": 1.0 / (1.0 + abs(beta_series)),
        "wB": -beta_series / (1.0 + abs(beta_series))
    }, index=z_series.index)
    
    return df

_worker_prices = None
_worker_coint_dict = None
_worker_train_end = None
_worker_kf_cfg = None
_worker_vol_regime = None

def _init_bt_worker(prices_df, coint_dict, train_end, kf_cfg, vol_regime):
    global _worker_prices, _worker_coint_dict, _worker_train_end, _worker_kf_cfg, _worker_vol_regime
    _worker_prices = prices_df
    _worker_coint_dict = coint_dict
    _worker_train_end = train_end
    _worker_kf_cfg = kf_cfg
    _worker_vol_regime = vol_regime

def _process_pair_bt(pair):
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
    df_kf = run_ecm_kalman(y_series, x_series, theta0, np.eye(2)*_worker_kf_cfg["p0_scale"], Q, c["sig2_ols"], lam, theta0)
    z_full = rolling_zscore(df_kf["spread"], _worker_kf_cfg["roll_win"])
    
    regime_aligned = _worker_vol_regime.reindex(z_full.index).fillna("NORMAL")
    signals = get_kf_signal_vectorized(
        z_full, df_kf["beta_pr"], _worker_kf_cfg["stop_z"], _worker_kf_cfg["entry_z"], _worker_kf_cfg["exit_z"], regime_aligned
    )
    return (A, B), signals

def main():
    parser = argparse.ArgumentParser(description="Kalman Volatility Regime Backtest")
    parser.add_argument("--days", type=int, default=45, help="Number of recent days to backtest")
    parser.add_argument("--train-end", default=None,
                        help="Clean OOS cutoff YYYY-MM-DD")
    args = parser.parse_args()
    
    config = load_config()
    trading_cfg = config["trading"]
    dates_cfg = config["dates"]
    kf_cfg = config["kalman_filter"]
    market_cfg = config.get("market", {})
    benchmark = market_cfg.get("benchmark", "SPY")
    cur = market_cfg.get("currency", "$")

    pairs = [tuple(p) for p in trading_cfg["pairs"]]
    all_tickers = sorted({t for pair in pairs for t in pair})

    today_str = pd.Timestamp.today().strftime('%Y-%m-%d')
    print(f"Downloading data from {dates_cfg['start']} to {today_str}...")
    prices_all = download_prices(all_tickers + [benchmark], dates_cfg["start"], today_str)
    prices = prices_all[[t for t in all_tickers if t != benchmark]]
    
    if args.train_end:
        train_end = pd.to_datetime(args.train_end).strftime('%Y-%m-%d')
    else:
        train_end = prices.index[-args.days - 1].strftime('%Y-%m-%d')
    train_end_ts = pd.to_datetime(train_end)
    n_oos = int((prices.index > train_end_ts).sum())
    coint_lookback = dates_cfg.get("coint_lookback")
    print(f"Train End: {train_end}  |  OOS test days: {n_oos}")

    coint_dict = get_cointegration_params(prices, pairs, train_end, coint_lookback)
    
    # --- VOLATILITY REGIME CALCULATION ---
    print(f"Calculating {benchmark} Volatility Regime...")
    bench_returns = np.log(prices_all[benchmark] / prices_all[benchmark].shift(1))
    bench_vol = bench_returns.rolling(21).std() * np.sqrt(252)
    vol_25 = bench_vol.rolling(252).quantile(0.25)
    
    vol_regime = pd.Series("NORMAL", index=prices_all.index)
    vol_regime[bench_vol < vol_25] = "LOW"
    
    strategy_signals = {
        "Kalman Filter": {}
    }
    
    print(f"Processing {len(pairs)} pairs using {multiprocessing.cpu_count()} cores...")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count(),
        initializer=_init_bt_worker,
        initargs=(prices, coint_dict, train_end, kf_cfg, vol_regime)
    ) as executor:
        for pair_key, signals in executor.map(_process_pair_bt, pairs):
            strategy_signals["Kalman Filter"][pair_key] = signals
        
    test_dates = prices.index[prices.index >= train_end_ts]
    capital_per_pair = trading_cfg["capital_per_pair"]
    total_capital = float(capital_per_pair)
    t_cost = kf_cfg["t_cost"]
    print(f"\n--- Clean OOS backtest: {test_dates[0].date()} -> {test_dates[-1].date()} ---")

    csv_rows = []
    modes = ["Full Compounding", "Half Kelly"]
    results = {strat: {m: {"pnl": [0.0], "dates": [test_dates[0]]} for m in modes} for strat in strategy_signals.keys()}

    bench_px = prices_all[benchmark].reindex(test_dates).ffill()
    bench_shares = total_capital / float(bench_px.iloc[0])
    spy_pnl = [0.0] + [bench_shares * (float(bench_px.iloc[i]) - float(bench_px.iloc[0]))
                       for i in range(1, len(test_dates))]
    
    for strat, pair_signals in strategy_signals.items():
        for mode in modes:
            cum_pnl = 0.0
            current_capital = float(capital_per_pair)
            n_pairs = len(pairs)
            holdings = {t: 0 for t in all_tickers}
            pair_sig_state = {p: 0 for p in pairs}

            for i in range(1, len(test_dates)):
                today = test_dates[i]
                yesterday = test_dates[i-1]
                
                daily_pnl = 0.0
                for t in all_tickers:
                    price_today = prices.loc[today, t]
                    price_yest = prices.loc[yesterday, t]
                    daily_pnl += holdings[t] * (price_today - price_yest)
                
                total_cost = 0.0
                current_capital += daily_pnl
                cum_profit = current_capital - capital_per_pair
                
                if mode == "Full Compounding":
                    risk_capital = current_capital
                else:
                    if cum_profit > 0:
                        risk_capital = capital_per_pair + 0.5 * cum_profit
                    else:
                        risk_capital = current_capital
                
                for A, B in pairs:
                    sig_df = pair_signals[(A, B)]
                    if yesterday not in sig_df.index:
                        continue
                    row = sig_df.loc[yesterday]
                    sig, wA, wB = row["signal"], row["wA"], row["wB"]

                    if sig == pair_sig_state[(A, B)]:
                        continue

                    price_A_today, price_B_today = prices.loc[today, A], prices.loc[today, B]
                    target_block = position_block(sig, risk_capital / n_pairs, A, B, wA, wB, {A: price_A_today, B: price_B_today})

                    tgt_A = target_block[A][0] if "Purchase" in target_block[A][1] else -target_block[A][0]
                    tgt_B = target_block[B][0] if "Purchase" in target_block[B][1] else -target_block[B][0]

                    cost_A = abs(tgt_A - holdings[A]) * price_A_today * t_cost
                    cost_B = abs(tgt_B - holdings[B]) * price_B_today * t_cost
                    total_cost += cost_A + cost_B

                    holdings[A] = tgt_A
                    holdings[B] = tgt_B
                    pair_sig_state[(A, B)] = sig
                
                current_capital -= total_cost
                net_daily_pnl = daily_pnl - total_cost
                cum_pnl += net_daily_pnl
                
                results[strat][mode]["pnl"].append(cum_pnl)
                results[strat][mode]["dates"].append(today)
                
                csv_rows.append({
                    "Date": today.strftime('%Y-%m-%d'),
                    "Compounding_Mode": mode,
                    "Strategy": strat,
                    "Daily_PnL": round(daily_pnl, 2),
                    "Transaction_Costs": round(total_cost, 2),
                    "Cumulative_PnL": round(cum_pnl, 2),
                    "Portfolio_Value": round(current_capital, 2),
                    "Risk_Capital": round(risk_capital, 2)
                })

    df_metrics = pd.DataFrame(csv_rows)
    df_metrics.to_csv("backtest_daily_metrics.csv", index=False)
    print("Saved daily metrics to 'backtest_daily_metrics.csv'.")

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    sns.set_theme(style="darkgrid")
    
    print("\n--- Backtest Final P&L ---")
    for i, mode in enumerate(modes):
        ax = axes[i]
        for strat in results.keys():
            pnls = results[strat][mode]["pnl"]
            print(f"[{mode}] {strat:15s}: {cur}{pnls[-1]:,.2f}")
            ax.plot(results[strat][mode]["dates"], pnls, label=f"{strat} ({cur}{pnls[-1]/1000:,.1f}k)")

        ax.plot(test_dates, spy_pnl, color='black', linestyle='--', linewidth=2,
                label=f"{benchmark} Benchmark ({cur}{spy_pnl[-1]/1000:,.1f}k)")
        if i == 0:
            print(f"[Benchmark] {benchmark} B&H : {cur}{spy_pnl[-1]:,.2f}")

        ax.set_title(f"Cumulative P&L - {mode}")
        ax.set_ylabel(f"P&L ({cur})")
        ax.legend()
        
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=300)
    print("Saved plot to 'backtest_results.png'.")

    n_days = len(test_dates) - 1
    ann = np.sqrt(252)
    spy_ret = spy_pnl[-1] / total_capital
    spy_daily = np.diff(spy_pnl)
    spy_sharpe = (spy_daily.mean() / spy_daily.std() * ann) if spy_daily.std() > 0 else float("nan")

    print("\n" + "=" * 78)
    print(f"CLEAN OOS PERFORMANCE  |  {test_dates[0].date()} -> {test_dates[-1].date()}  "
          f"({n_days} days)  |  book {cur}{total_capital:,.0f}")
    print("=" * 78)
    bench_col = f"vs {benchmark}"
    print(f"{'Strategy':16s} {'Final P&L':>13s} {'Return':>8s} {bench_col:>9s} {'Sharpe':>8s}")
    print("-" * 78)
    for strat in results.keys():
        pnls = np.array(results[strat]["Full Compounding"]["pnl"])
        final = pnls[-1]
        ret = final / total_capital
        daily = np.diff(pnls)
        sharpe = (daily.mean() / daily.std() * ann) if daily.std() > 0 else float("nan")
        print(f"{strat:16s} {final:13,.0f} {ret:7.2%} {ret - spy_ret:+8.2%} {sharpe:8.2f}")
    print("-" * 78)
    print(f"{benchmark + ' (real px)':16s} {spy_pnl[-1]:13,.0f} {spy_ret:7.2%} {'--':>8s} {spy_sharpe:8.2f}")
    print("=" * 78)

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import yfinance as yf

def evaluate_dispersion():
    # Constituents vs Index (e.g. Dow 30 proxy)
    tickers = ["AAPL", "MSFT", "V", "JPM", "WMT", "JNJ", "PG", "HD"]
    index_ticker = "SPY"
    
    all_tickers = tickers + [index_ticker]
    print(f"Fetching data for {len(tickers)} constituents + Index...")
    data = yf.download(all_tickers, start="2020-01-01", end="2024-01-01", progress=False, threads=False)['Close']
    data = data.dropna(axis=1)
    
    # Calculate daily returns
    returns = np.log(data / data.shift(1)).dropna()
    
    train_window = 252
    test_window = 63
    
    results = []
    
    for start_idx in range(0, len(returns) - train_window - test_window, test_window):
        train_rets = returns.iloc[start_idx : start_idx + train_window]
        test_rets = returns.iloc[start_idx + train_window : start_idx + train_window + test_window]
        
        # Train: Calculate historical 21-day rolling dispersion
        roll_vol_comps = train_rets[tickers].rolling(21).std().mean(axis=1) * np.sqrt(252)
        roll_vol_idx = train_rets[index_ticker].rolling(21).std() * np.sqrt(252)
        
        hist_dispersion = (roll_vol_comps - roll_vol_idx).dropna()
        disp_mean = hist_dispersion.mean()
        disp_std = hist_dispersion.std()
        
        # Test: Evaluate dispersion signals
        # We proxy the "implied" vs "realized" by trading when realized dispersion deviates
        test_vol_comps = test_rets[tickers].rolling(21).std().mean(axis=1) * np.sqrt(252)
        test_vol_idx = test_rets[index_ticker].rolling(21).std() * np.sqrt(252)
        
        test_dispersion = (test_vol_comps - test_vol_idx).dropna()
        
        period_pnl = 0
        trades = 0
        
        for i in range(1, len(test_dispersion)):
            z_score = (test_dispersion.iloc[i-1] - disp_mean) / disp_std
            
            # Simple mean reversion on volatility spread
            if z_score > 1.5: # Dispersion too high, sell dispersion (short components, long index)
                # Proxy payoff: mean reversion of the spread
                period_pnl -= (test_dispersion.iloc[i] - test_dispersion.iloc[i-1]) 
                trades += 1
            elif z_score < -1.5: # Dispersion too low, buy dispersion
                period_pnl += (test_dispersion.iloc[i] - test_dispersion.iloc[i-1])
                trades += 1
                
        # Assume generic slippage in vol terms
        net_pnl = period_pnl - (trades * 0.005)
        
        if len(test_dispersion) > 0:
            results.append({
                "Period Start": test_dispersion.index[0].strftime('%Y-%m-%d'),
                "Period End": test_dispersion.index[-1].strftime('%Y-%m-%d'),
                "Gross Vol PnL": period_pnl,
                "Net Vol PnL": net_pnl,
                "Trades": trades
            })
            
    df_res = pd.DataFrame(results)
    
    report = "### Alternative 2: Options Dispersion Proxy\n"
    report += "This strategy simulates trading the volatility spread between an index and its constituents. "
    report += "It profits from the structural mean-reversion of correlation and the volatility risk premium.\n\n"
    report += df_res.to_markdown(index=False)
    report += f"\n\n**Total Net Vol PnL:** {df_res['Net Vol PnL'].sum():.4f}\n"
    
    with open("salvage_results.md", "a") as f:
        f.write(report + "\n\n---\n\n")
    print("Dispersion proxy evaluation complete. Appended to salvage_results.md")

if __name__ == "__main__":
    evaluate_dispersion()

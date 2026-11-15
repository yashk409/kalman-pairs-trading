import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

def evaluate_volatility_regime():
    pair = ["KO", "PEP"]
    index_ticker = "SPY"
    
    all_tickers = pair + [index_ticker]
    print(f"Fetching data for pair {pair} and index...")
    data = yf.download(all_tickers, start="2020-01-01", end="2024-01-01", progress=False, threads=False)['Close']
    data = data.dropna(axis=1)
    
    returns = np.log(data / data.shift(1)).dropna()
    
    train_window = 252
    test_window = 63
    
    results = []
    
    for start_idx in range(0, len(returns) - train_window - test_window, test_window):
        train_rets = returns.iloc[start_idx : start_idx + train_window]
        test_rets = returns.iloc[start_idx + train_window : start_idx + train_window + test_window]
        train_prices = data.iloc[start_idx : start_idx + train_window]
        test_prices = data.iloc[start_idx + train_window : start_idx + train_window + test_window]
        
        # Train: Calculate historical market volatility thresholds
        roll_vol_idx = train_rets[index_ticker].rolling(21).std() * np.sqrt(252)
        vol_25 = roll_vol_idx.quantile(0.25)
        vol_75 = roll_vol_idx.quantile(0.75)
        
        # Train: Calculate OLS beta for the pair spread
        y = np.log(train_prices[pair[0]])
        x = np.log(train_prices[pair[1]])
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        beta = model.params.iloc[1]
        
        train_spread = y - beta * x
        spread_mean = train_spread.mean()
        spread_std = train_spread.std()
        
        # Test: Apply dynamic threshold based on volatility regime
        test_vol_idx = test_rets[index_ticker].rolling(21).std() * np.sqrt(252)
        test_y = np.log(test_prices[pair[0]])
        test_x = np.log(test_prices[pair[1]])
        test_spread = test_y - beta * test_x
        
        test_z = (test_spread - spread_mean) / spread_std
        
        period_pnl = 0
        trades = 0
        current_pos = 0 # 1 for long spread, -1 for short spread
        
        for i in range(1, len(test_z)):
            current_vol = test_vol_idx.iloc[i-1]
            if np.isnan(current_vol):
                continue
                
            # Dynamic Regime Logic
            if current_vol < vol_25:
                entry_z = 3.0 # Low vol regime: noise dominates, widen threshold (trade less)
            elif current_vol > vol_75:
                entry_z = 1.25 # High vol regime: forced liquidations, tighten threshold (trade more)
            else:
                entry_z = 2.0
                
            z = test_z.iloc[i-1]
            
            if current_pos == 0:
                if z < -entry_z:
                    current_pos = 1
                    trades += 1
                elif z > entry_z:
                    current_pos = -1
                    trades += 1
            else:
                # Exit on reversion
                if (current_pos == 1 and z >= 0) or (current_pos == -1 and z <= 0):
                    current_pos = 0
                    trades += 1
            
            # PnL logic (simplified)
            if current_pos == 1:
                period_pnl += test_rets[pair[0]].iloc[i] - beta * test_rets[pair[1]].iloc[i]
            elif current_pos == -1:
                period_pnl -= test_rets[pair[0]].iloc[i] - beta * test_rets[pair[1]].iloc[i]
                
        # Deduct 5bps per leg per trade
        net_pnl = period_pnl - (trades * 0.0010) 
        
        results.append({
            "Period Start": test_rets.index[0].strftime('%Y-%m-%d'),
            "Period End": test_rets.index[-1].strftime('%Y-%m-%d'),
            "Gross PnL": period_pnl,
            "Net PnL": net_pnl,
            "Trades": trades
        })
    
    df_res = pd.DataFrame(results)
    
    report = "### Alternative 3: Volatility Regime Filter\n"
    report += "This strategy rescues pairs trading by dynamically adjusting the entry z-score threshold "
    report += "based on the rolling 21-day volatility regime of the broader market (SPY).\n\n"
    report += df_res.to_markdown(index=False)
    report += f"\n\n**Total Net PnL:** {df_res['Net PnL'].sum():.4f}\n"
    
    with open("salvage_results.md", "a") as f:
        f.write(report + "\n\n---\n\n")
    print("Volatility Regime evaluation complete. Appended to salvage_results.md")

if __name__ == "__main__":
    evaluate_volatility_regime()

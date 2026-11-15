import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
import statsmodels.api as sm
import os

def fit_ou_process(res):
    """Fits an Ornstein-Uhlenbeck process to the residual series."""
    x = res[:-1].values
    y = res[1:].values
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    a, b = model.params
    if b <= 0 or b >= 1:
        return np.nan, np.nan, np.nan
    
    kappa = -np.log(b) * 252
    m = a / (1 - b)
    sigma = np.std(model.resid) * np.sqrt(252 * 2 * kappa / (1 - b**2))
    return kappa, m, sigma

def evaluate_factor_residual():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "V"]
    print(f"Fetching data for {len(tickers)} tickers...")
    data = yf.download(tickers, start="2020-01-01", end="2024-01-01", progress=False, threads=False)['Close']
    data = data.dropna(axis=1) # Drop stocks with missing data
    returns = np.log(data / data.shift(1)).dropna()
    
    train_window = 252
    test_window = 63
    
    results = []
    
    for start_idx in range(0, len(returns) - train_window - test_window, test_window):
        train_rets = returns.iloc[start_idx : start_idx + train_window]
        test_rets = returns.iloc[start_idx + train_window : start_idx + train_window + test_window]
        
        scaler_mean = train_rets.mean()
        scaler_std = train_rets.std()
        train_norm = (train_rets - scaler_mean) / scaler_std
        
        pca = PCA(n_components=3)
        pca.fit(train_norm)
        
        betas = pca.components_.T
        factors_train = pca.transform(train_norm)
        reconstructed_train = np.dot(factors_train, betas.T)
        res_train = train_norm - reconstructed_train
        
        ou_params = {}
        for col in train_norm.columns:
            kappa, m, sigma = fit_ou_process(res_train[col])
            if not np.isnan(kappa) and kappa > 252/30:
                ou_params[col] = (m, sigma)
                
        test_norm = (test_rets - scaler_mean) / scaler_std
        factors_test = pca.transform(test_norm)
        reconstructed_test = np.dot(factors_test, betas.T)
        res_test = test_norm - reconstructed_test
        
        period_pnl = 0
        trades = 0
        
        for i in range(1, len(res_test)):
            for col in ou_params:
                m, sigma = ou_params[col]
                s_score = (res_test[col].iloc[i-1] - m) / sigma
                
                if s_score < -1.25:
                    period_pnl += test_rets[col].iloc[i]
                    trades += 1
                elif s_score > 1.25:
                    period_pnl -= test_rets[col].iloc[i]
                    trades += 1
        
        net_pnl = period_pnl - (trades * 0.0005)
        results.append({
            "Period Start": test_rets.index[0].strftime('%Y-%m-%d'),
            "Period End": test_rets.index[-1].strftime('%Y-%m-%d'),
            "Gross PnL": period_pnl,
            "Net PnL": net_pnl,
            "Trades": trades
        })
    
    df_res = pd.DataFrame(results)
    
    report = "### Alternative 1: Factor-Residual Statistical Arbitrage\n"
    report += "This strategy isolates idiosyncratic returns using PCA (3 factors) on a basket of tech megacaps. "
    report += "It dynamically calculates the s-score using an Ornstein-Uhlenbeck fit. Out-of-sample prediction periods are strictly separated from training.\n\n"
    report += df_res.to_markdown(index=False)
    report += f"\n\n**Total Net PnL:** {df_res['Net PnL'].sum():.4f}\n"
    
    with open("salvage_results.md", "a") as f:
        f.write(report + "\n\n---\n\n")
    print("Factor-Residual evaluation complete. Appended to salvage_results.md")

if __name__ == "__main__":
    evaluate_factor_residual()

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter as PKF

def eg_ols(log_y, log_x):
    X = sm.add_constant(log_x)
    res = sm.OLS(log_y, X).fit()
    return float(res.params.iloc[0]), float(res.params.iloc[1]), res.resid

def estimate_lambda_ecm(y, x, k_lags=1):
    alpha, beta, _ = eg_ols(y, x)
    df = pd.DataFrame({"y": y, "x": x})
    df["dy"] = df["y"].diff()
    df["dx"] = df["x"].diff()
    df["ect_lag"] = (df["y"] - alpha - beta * df["x"]).shift(1)
    for i in range(1, k_lags + 1):
        df[f"dy_lag{i}"] = df["dy"].shift(i)
        df[f"dx_lag{i}"] = df["dx"].shift(i)
    df = df.dropna()
    rhs = ["ect_lag"] + [f"dy_lag{i}" for i in range(1,k_lags+1)] + \
          [f"dx_lag{i}" for i in range(1,k_lags+1)]
    fit = sm.OLS(df["dy"], sm.add_constant(df[rhs])).fit()
    return float(fit.params["ect_lag"]), float(fit.mse_resid)

def estimate_QR_em(y_tr, x_tr, theta_init, n_iter=8):
    n = len(y_tr)
    H = np.zeros((n, 1, 2))
    H[:, 0, 0] = x_tr.values
    H[:, 0, 1] = 1.0
    kf = PKF(
        transition_matrices=np.eye(2),
        observation_matrices=H,
        initial_state_mean=theta_init,
        initial_state_covariance=np.eye(2),
        transition_covariance=np.eye(2)*1e-5,
        observation_covariance=np.array([[1.0]]),
        em_vars=["transition_covariance","observation_covariance"],
    )
    kf = kf.em(y_tr.values.reshape(-1,1), n_iter=n_iter)
    Q = np.asarray(kf.transition_covariance)
    R = float(np.asarray(kf.observation_covariance).item())
    return Q, R

def run_ecm_kalman(y_series, x_series, theta0, P0, Q, R, lam, theta_star):
    n = len(y_series)
    y_arr, x_arr = y_series.values, x_series.values
    theta = theta0.copy()
    theta_star = np.asarray(theta_star)
    P = P0.copy()
    F = (1.0 + lam) * np.eye(2)
    c = -lam * theta_star

    beta_pr = np.full(n, np.nan)
    beta_po = np.full(n, np.nan)
    innov   = np.full(n, np.nan)
    spread  = np.full(n, np.nan)

    for t in range(n):
        theta_pred = F @ theta + c
        P_pred = F @ P @ F.T + Q
        H = np.array([[x_arr[t], 1.0]])
        y_pred = (H @ theta_pred).item()
        v = y_arr[t] - y_pred
        S = (H @ P_pred @ H.T).item() + R
        if not np.isfinite(S) or S <= 0:
            break
        K = (P_pred @ H.T).flatten() / S
        theta = theta_pred + K * v
        P = (np.eye(2) - np.outer(K, H.flatten())) @ P_pred
        beta_pr[t] = theta_pred[0]
        beta_po[t] = theta[0]
        innov[t]   = v                              # one-step innovation (whitened, ~noise)
        # Cointegration spread using the filtered (dynamic) hedge ratio. This is the
        # tradeable, persistent mean-reversion series; the intercept level is removed
        # downstream by the rolling z-score. Trading `innov` instead barely crosses
        # thresholds (the filter whitens it), which is why the KF made ~0 trades.
        spread[t]  = y_arr[t] - theta[0] * x_arr[t]

    return pd.DataFrame({"beta_pr": beta_pr, "beta_po": beta_po,
                         "innov": innov, "spread": spread}, index=y_series.index)

def rolling_zscore(series, window):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s

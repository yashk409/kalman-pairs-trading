import numpy as np

def kf_decide(z, prev_signal, stop_z, entry_z, exit_z, vol_regime="NORMAL"):
    """
    Single source of truth for the Kalman z-score trading rule, shared by the live
    snapshot (get_kf_signal) and the vectorized backtest (get_kf_signal_vectorized) so
    they behave identically. Includes hysteresis: between the exit and entry bands the
    existing position is held rather than flattened.
    """
    if np.isnan(z) or abs(z) > stop_z:
        return 0                      # no data / stop-out
    if z > entry_z and vol_regime != "LOW":
        return -1                     # spread rich -> short spread
    if z < -entry_z and vol_regime != "LOW":
        return +1                     # spread cheap -> long spread
    if -exit_z < z < exit_z:
        return 0                      # reverted to mean -> exit
    return prev_signal                # between exit and entry -> hold (hysteresis)


def get_kf_signal(z_now, beta_now, stop_z, entry_z, exit_z, prev_signal=0, vol_regime="NORMAL"):
    signal = kf_decide(z_now, prev_signal, stop_z, entry_z, exit_z, vol_regime)
    if np.isnan(z_now):
        reason = "insufficient data"
    elif abs(z_now) > stop_z:
        reason = f"STOP |z|={z_now:.2f} > {stop_z}"
    elif z_now > entry_z:
        reason = f"SHORT spread (z={z_now:.2f} > +{entry_z})"
    elif z_now < -entry_z:
        reason = f"LONG spread  (z={z_now:.2f} < -{entry_z})"
    elif -exit_z < z_now < exit_z:
        reason = f"FLAT/EXIT z near zero ({z_now:.2f})"
    else:
        reason = f"HOLD prev={prev_signal} between bands ({z_now:.2f})"

    wA =  1.0 / (1.0 + abs(beta_now))
    wB = -beta_now / (1.0 + abs(beta_now))

    return {
        "signal": signal,
        "z": z_now,
        "beta": beta_now,
        "wA": wA,
        "wB": wB,
        "reason": reason
    }



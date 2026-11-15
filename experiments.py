"""
experiments.py  --  Comprehensive ablation / robustness harness for the pairs strategy.

Independent of main.py / backtest.py: it reuses the data, cointegration, model and
portfolio code but re-implements the windowing + train/val/fine-tune split so it can test
configurations the main pipeline cannot express (old vs new split ordering, full-window
beta, no fine-tune, etc.). It NEVER touches config.yaml or the saved backtest artifacts.

The point is to answer: is the strategy's P&L an EDGE or NOISE, and which knobs move it?

--------------------------------------------------------------------------------------
USAGE
  python3 experiments.py --suite seed_variance   --seeds 8   --epochs 80
  python3 experiments.py --suite data_start       --epochs 80
  python3 experiments.py --suite coint_lookback   --epochs 80
  python3 experiments.py --suite split_mode       --epochs 80
  python3 experiments.py --suite finetune_days    --epochs 80
  python3 experiments.py --suite entry_thresholds --kf-only          # fast, KF only
  python3 experiments.py --suite walkforward      --epochs 80
  python3 experiments.py --suite all              --epochs 60        # everything (slow!)

Every run appends tidy rows to experiment_results.csv (one row per strategy per scenario)
so you can slice/plot afterwards. Each suite also prints a formatted table.

--------------------------------------------------------------------------------------
SUITES (each varies ONE axis, holding the rest at the BASELINE below)
  seed_variance   : same config, N seeds, PER-SEED per-model P&L. The decisive luck test:
                    if a model's P&L std >> |mean|, its "profit" is a coin flip, not edge.
  data_start      : 2015 / 2018 / 2020 / 2022 / 2023  (DL base size + KF lambda/Q/warm-up)
  coint_lookback  : full / 1008 / 756 / 504 / 378 / 252  (hedge-ratio estimation window)
  split_mode      : new [base|val|ft]  vs  old [base|ft|val]  vs  no_finetune
  finetune_days   : 0 / 63 / 126 / 252
  entry_thresholds: KF entry_z 1.0/1.25/1.5/2.0 (KF-only, fast)
  walkforward     : several non-overlapping train_ends, each tested on the next ~6 months
  kf_edge         : KF-ONLY grid of (entry_z, stop_z) run across EVERY walk-forward window,
                    aggregated per config -- the decisive test of whether the Kalman filter's
                    defensive behavior is a real standalone strategy or break-even noise.

--------------------------------------------------------------------------------------
RUNTIME: DL training dominates. One DL config = 4 models x n_seeds x n_pairs trainings.
Use --epochs 60-80 for sweeps (early stopping caps it anyway) and run suites one at a
time. Use --kf-only to sanity-check the harness structure in seconds.
"""
import argparse
import warnings
import os

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from data_loader import download_prices, get_cointegration_params, make_windows
from models.kalman_filter import (estimate_lambda_ecm, estimate_QR_em,
                                   run_ecm_kalman, rolling_zscore)
from models.lstm import build_lstm
from models.transformer import build_transformer
from models.nbeats import build_nbeats
from models.cnn_transformer import build_cnn_transformer
from portfolio import position_block
from signals import kf_decide

DL_NAMES = ["LSTM", "Transformer", "N-BEATS", "CNN-Transformer"]


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# ======================================================================================
# Metrics
# ======================================================================================
def compute_metrics(pnl_series, capital, n_trades):
    pnl = np.asarray(pnl_series, dtype=float)
    final = float(pnl[-1])
    daily = np.diff(pnl)
    sd = daily.std()
    sharpe = float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    ann_vol = float(sd * np.sqrt(252) / capital) if capital else 0.0
    equity = capital + pnl
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return {
        "final_pnl": round(final, 0),
        "ret_pct": round(100 * final / capital, 2),
        "sharpe": round(sharpe, 2),
        "ann_vol_pct": round(100 * ann_vol, 1),
        "max_dd_pct": round(100 * float(dd.min()), 1),
        "n_trades": int(n_trades),
        "days": len(pnl) - 1,
    }


# ======================================================================================
# Backtest P&L loop (equal-weight book, trade only on signal change) -- mirrors backtest.py
# ======================================================================================
def backtest_pnl(pair_signals, test_dates, prices, all_tickers, pairs, capital, t_cost):
    n_pairs = len(pairs)
    cum, cap = 0.0, float(capital)
    holdings = {t: 0 for t in all_tickers}
    state = {p: 0 for p in pairs}
    pnl = [0.0]
    n_trades = 0
    for i in range(1, len(test_dates)):
        today, yst = test_dates[i], test_dates[i - 1]
        dpnl = sum(holdings[t] * (prices.loc[today, t] - prices.loc[yst, t]) for t in all_tickers)
        cap += dpnl
        cost = 0.0
        risk = cap
        for A, B in pairs:
            sdf = pair_signals[(A, B)]
            if yst not in sdf.index:
                continue
            row = sdf.loc[yst]
            sig = int(row["signal"])
            if sig == state[(A, B)]:
                continue
            pxA, pxB = prices.loc[today, A], prices.loc[today, B]
            blk = position_block(sig, risk / n_pairs, A, B, row["wA"], row["wB"], {A: pxA, B: pxB})
            tA = blk[A][0] if "Purchase" in blk[A][1] else -blk[A][0]
            tB = blk[B][0] if "Purchase" in blk[B][1] else -blk[B][0]
            cost += abs(tA - holdings[A]) * pxA * t_cost + abs(tB - holdings[B]) * pxB * t_cost
            holdings[A], holdings[B] = tA, tB
            state[(A, B)] = sig
            n_trades += 1
        cap -= cost
        cum += dpnl - cost
        pnl.append(cum)
    return pnl, n_trades


# ======================================================================================
# Kalman signal (depends on data_start via lambda/Q/warm-up, and coint beta)
# ======================================================================================
def kf_signal_df(prices, A, B, c, kf, train_end, vol_regime=None):
    """vol_regime: optional Series of "LOW"/"NORMAL" per date. When supplied, kf_decide blocks
    NEW entries on LOW-vol days. Default None => "NORMAL" everywhere, i.e. the filter is OFF.
    walkforward.py MUST pass this, or the honest harness silently evaluates a different
    strategy than the one main.py/backtest.py would trade."""
    y, x = np.log(prices[A]), np.log(prices[B])
    theta0 = np.array([c["beta"], c["alpha"]])
    lam, _ = estimate_lambda_ecm(y.loc[:train_end], x.loc[:train_end])
    lam = float(np.clip(lam if lam < 0 else kf["lambda_max"], kf["lambda_min"], kf["lambda_max"]))
    Qem, _ = estimate_QR_em(y.loc[:train_end], x.loc[:train_end], theta0)
    Qd = np.diag(Qem)
    Q = Qem if (Qd.min() > 1e-12 and Qd.max() < 1.0) else np.diag([1e-5, 1e-5])
    df = run_ecm_kalman(y, x, theta0, np.eye(2) * kf["p0_scale"], Q, c["sig2_ols"], lam, theta0)
    z = rolling_zscore(df["spread"], kf["roll_win"])
    beta_s = df["beta_pr"]
    reg = (vol_regime.reindex(z.index).fillna("NORMAL") if vol_regime is not None
           else pd.Series("NORMAL", index=z.index))
    sig = np.zeros(len(z))
    prev = 0
    for i, zz in enumerate(z):
        prev = kf_decide(zz, prev, kf["stop_z"], kf["entry_z"], kf["exit_z"], reg.iloc[i])
        sig[i] = prev
    return pd.DataFrame({"signal": sig,
                         "wA": 1.0 / (1.0 + beta_s.abs()),
                         "wB": -beta_s / (1.0 + beta_s.abs())}, index=z.index)


# ======================================================================================
# DL dataset with CONFIGURABLE split ordering
# ======================================================================================
def make_dl_dataset(prices, coint, pair, train_end, lookback, horizon, val_frac,
                    fine_tune_days, split_mode):
    A, B = pair
    lp = np.log(prices[[A, B]]).copy()
    dlr = lp.diff()
    feat = pd.DataFrame({
        "ret_A": dlr[A], "ret_B": dlr[B],
        "z_spread": coint[pair]["zspread"],
        "vol_A": dlr[A].rolling(20).std(), "vol_B": dlr[B].rolling(20).std(),
    }).dropna()

    fut = pd.Series(lp.index, index=lp.index).shift(-horizon)
    beta, alpha = coint[pair]["beta"], coint[pair]["alpha"]
    spread = lp[A] - beta * lp[B] - alpha
    tgt = pd.DataFrame({"d_spread": spread.shift(-horizon) - spread})

    common = feat.index.intersection(tgt.dropna().index)
    ld = fut.loc[common]
    common = common[ld.notna().values]
    ld = pd.DatetimeIndex(ld.loc[common])
    feat, tgt = feat.loc[common], tgt.loc[common]

    fit = ld <= pd.to_datetime(train_end)
    fs = StandardScaler().fit(feat.loc[fit].values)
    ts = StandardScaler().fit(tgt.loc[fit].values)
    F, T = fs.transform(feat.values), ts.transform(tgt.values)

    X, Y, D, DY = make_windows(F, T, feat.index, ld, lookback)
    is_tr = DY <= pd.to_datetime(train_end)
    Xtr, Ytr = X[is_tr], Y[is_tr]
    Xte, Yte = X[~is_tr], Y[~is_tr]
    test_dates = D[~is_tr]

    n = len(Xtr)
    ft = int(min(fine_tune_days, max(1, n - 2)))
    if split_mode == "new":            # [ base ][ val ][ fine-tune (adjacent to cutoff) ]
        fts = n - ft
        vc = max(1, int(fts * (1 - val_frac)))
        Xb, Yb, Xv, Yv, Xf, Yf = Xtr[:vc], Ytr[:vc], Xtr[vc:fts], Ytr[vc:fts], Xtr[fts:], Ytr[fts:]
    elif split_mode == "old":          # [ base ][ fine-tune ][ val (most recent) ]
        vc = max(ft + 1, int(n * (1 - val_frac)))
        Xv, Yv = Xtr[vc:], Ytr[vc:]
        Xf, Yf = Xtr[vc - ft:vc], Ytr[vc - ft:vc]
        Xb, Yb = Xtr[:vc - ft], Ytr[:vc - ft]
    else:                              # no_finetune: [ base ][ val ], no fine-tune pass
        vc = max(1, int(n * (1 - val_frac)))
        Xb, Yb, Xv, Yv = Xtr[:vc], Ytr[:vc], Xtr[vc:], Ytr[vc:]
        Xf, Yf = Xtr[:0], Ytr[:0]

    return dict(X_base=Xb, Y_base=Yb, X_va=Xv, Y_va=Yv, X_ft=Xf, Y_ft=Yf,
                X_te=Xte, Y_te=Yte, test_dates=test_dates, tgt_scaler=ts,
                n_features=feat.shape[1])


def build_any(name, lookback, nfeat, lr, cfg):
    if name == "LSTM":
        return build_lstm(lookback, nfeat, lr)
    if name == "Transformer":
        return build_transformer(lookback, nfeat, lr)
    if name == "N-BEATS":
        return build_nbeats(lookback, nfeat, lr, num_blocks=cfg["nbeats"]["num_blocks"])
    if name == "CNN-Transformer":
        return build_cnn_transformer(lookback, nfeat, lr,
                                     d_model=cfg["cnn_transformer"]["d_model"],
                                     num_blocks=cfg["cnn_transformer"]["num_blocks"])
    raise ValueError(name)


def train_one(name, ds, seed, epochs, batch, lr, cfg, do_finetune):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    m = build_any(name, cfg["deep_learning"]["lookback"], ds["n_features"], lr, cfg)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    m.fit(ds["X_base"], ds["Y_base"], validation_data=(ds["X_va"], ds["Y_va"]),
          epochs=epochs, batch_size=batch, verbose=0, callbacks=[es])
    if do_finetune and len(ds["X_ft"]) > 0:
        m.optimizer.learning_rate.assign(lr / 10.0)
        m.fit(ds["X_ft"], ds["Y_ft"], epochs=max(1, epochs // 2), batch_size=batch, verbose=0)
    return m


def dl_signal_df(models_by_seed, ds, beta, entry_mult):
    dte, dva = [], []
    for m in models_by_seed.values():
        dte.append(ds["tgt_scaler"].inverse_transform(m.predict(ds["X_te"], verbose=0))[:, 0])
        dva.append(ds["tgt_scaler"].inverse_transform(m.predict(ds["X_va"], verbose=0))[:, 0])
    delta = np.mean(dte, axis=0)
    thr = float(np.mean([entry_mult * np.std(v, ddof=1) for v in dva]))
    sig = np.where(delta > thr, 1, np.where(delta < -thr, -1, 0))
    wA, wB = 1.0 / (1.0 + abs(beta)), -beta / (1.0 + abs(beta))
    return pd.DataFrame({"signal": sig, "wA": wA, "wB": wB}, index=ds["test_dates"])


# ======================================================================================
# Run one configuration -> rows for KF, each DL model, DL-Ensemble, benchmark
# ======================================================================================
def run_single(prices_all, cfg, P, benchmark, capital, t_cost):
    pairs = [tuple(p) for p in cfg["trading"]["pairs"]]
    all_tk = sorted({t for pr in pairs for t in pr})
    prices = prices_all.loc[P["data_start"]:]
    px = prices[all_tk]
    te = P["train_end"]
    te_ts = pd.to_datetime(te)

    coint = get_cointegration_params(px, pairs, te, P["coint_lookback"])

    test_dates = px.index[px.index >= te_ts]
    if P.get("oos_days"):
        test_dates = test_dates[:P["oos_days"] + 1]

    kf = dict(cfg["kalman_filter"])
    kf["entry_z"] = P.get("entry_z", kf["entry_z"])
    kf["stop_z"]  = P.get("stop_z", kf["stop_z"])

    strategies = {"Kalman Filter": {(A, B): kf_signal_df(px, A, B, coint[(A, B)], kf, te)
                                    for A, B in pairs}}

    if P["include_dl"]:
        per_model = {n: {} for n in DL_NAMES}
        for A, B in pairs:
            ds = make_dl_dataset(px, coint, (A, B), te,
                                 cfg["deep_learning"]["lookback"], cfg["deep_learning"]["horizon"],
                                 cfg["deep_learning"]["val_frac"], P["fine_tune_days"], P["split_mode"])
            beta = coint[(A, B)]["beta"]
            do_ft = P["split_mode"] != "no_finetune" and P["fine_tune_days"] > 0
            for name in DL_NAMES:
                models = {s: train_one(name, ds, s, P["epochs"], cfg["deep_learning"]["batch_size"],
                                       cfg["deep_learning"]["learning_rate"], cfg, do_ft)
                          for s in P["seeds"]}
                per_model[name][(A, B)] = dl_signal_df(models, ds, beta,
                                                       P.get("entry_mult", cfg["deep_learning"]["entry_mult"]))
        for name in DL_NAMES:
            strategies[name] = per_model[name]
        # majority-vote ensemble across the 4 architectures
        ens = {}
        for A, B in pairs:
            base = per_model[DL_NAMES[0]][(A, B)]
            net = sum(per_model[n][(A, B)]["signal"] for n in DL_NAMES)
            ens[(A, B)] = pd.DataFrame({"signal": np.sign(net), "wA": base["wA"], "wB": base["wB"]},
                                       index=base.index)
        strategies["DL-Ensemble"] = ens

    rows = []
    meta = {k: P[k] for k in ("data_start", "train_end", "coint_lookback", "split_mode",
                              "fine_tune_days", "epochs", "entry_z", "entry_mult")}
    meta["n_seeds"] = len(P["seeds"])
    for sname, psig in strategies.items():
        pnl, ntr = backtest_pnl(psig, test_dates, px, all_tk, pairs, capital, t_cost)
        rows.append({"strategy": sname, **meta, **compute_metrics(pnl, capital, ntr)})

    bpx = prices_all[benchmark].reindex(test_dates).ffill()
    bsh = capital / float(bpx.iloc[0])
    bpnl = [0.0] + [bsh * (float(bpx.iloc[i]) - float(bpx.iloc[0])) for i in range(1, len(test_dates))]
    rows.append({"strategy": benchmark, **meta, **compute_metrics(bpnl, capital, 0)})
    return rows


# ======================================================================================
# Seed-variance: per-seed, per-model (NOT ensembled) -> the decisive luck test
# ======================================================================================
def run_seed_variance(prices_all, cfg, P, benchmark, capital, t_cost):
    pairs = [tuple(p) for p in cfg["trading"]["pairs"]]
    all_tk = sorted({t for pr in pairs for t in pr})
    px = prices_all.loc[P["data_start"]:][all_tk]
    te = P["train_end"]
    coint = get_cointegration_params(px, pairs, te, P["coint_lookback"])
    test_dates = px.index[px.index >= pd.to_datetime(te)]

    ds_by_pair = {(A, B): make_dl_dataset(px, coint, (A, B), te,
                                          cfg["deep_learning"]["lookback"], cfg["deep_learning"]["horizon"],
                                          cfg["deep_learning"]["val_frac"], P["fine_tune_days"], P["split_mode"])
                  for A, B in pairs}
    rows = []
    do_ft = P["split_mode"] != "no_finetune" and P["fine_tune_days"] > 0
    for name in DL_NAMES:
        for s in P["seeds"]:
            psig = {}
            for A, B in pairs:
                ds = ds_by_pair[(A, B)]
                m = train_one(name, ds, s, P["epochs"], cfg["deep_learning"]["batch_size"],
                              cfg["deep_learning"]["learning_rate"], cfg, do_ft)
                psig[(A, B)] = dl_signal_df({s: m}, ds, coint[(A, B)]["beta"],
                                            cfg["deep_learning"]["entry_mult"])
            pnl, ntr = backtest_pnl(psig, test_dates, px, all_tk, pairs, capital, t_cost)
            met = compute_metrics(pnl, capital, ntr)
            rows.append({"strategy": name, "seed": s, **met})
            print(f"    {name:16s} seed={s:<4d} ret={met['ret_pct']:+6.2f}%  sharpe={met['sharpe']:+.2f}")
    return rows


# ======================================================================================
# Suites
# ======================================================================================
def baseline(cfg, args):
    # Baseline mirrors the PRODUCTION config where it can: fine_tune_days and coint_lookback
    # come from config.yaml so every suite (except the ones that deliberately vary those axes)
    # reflects what main.py / backtest.py actually run. With fine_tune_days=0 the "new" split
    # yields an empty fine-tune block and train_one skips the pass, so no fine-tuning occurs.
    return dict(data_start="2015-01-01", train_end=args.train_end,
                coint_lookback=cfg.get("dates", {}).get("coint_lookback", 756),
                split_mode="new", fine_tune_days=cfg.get("dates", {}).get("fine_tune_days", 126),
                seeds=[42, 123], epochs=args.epochs,
                entry_z=cfg["kalman_filter"]["entry_z"], entry_mult=cfg["deep_learning"]["entry_mult"],
                include_dl=not args.kf_only, oos_days=None)


def print_table(rows, title, cols=("scenario", "strategy", "ret_pct", "sharpe", "max_dd_pct", "n_trades", "days")):
    print(f"\n===== {title} =====")
    df = pd.DataFrame(rows)
    show = [c for c in cols if c in df.columns]
    print(df[show].to_string(index=False))


def suite_axis(prices_all, cfg, args, benchmark, capital, t_cost, axis, values):
    """Vary one axis of the baseline across `values`."""
    all_rows = []
    for v in values:
        P = baseline(cfg, args)
        P[axis] = v
        if axis == "fine_tune_days" and v == 0:
            P["split_mode"] = "no_finetune"
        label = f"{axis}={v}"
        print(f"\n[ {label} ]  (start={P['data_start']} coint_lb={P['coint_lookback']} "
              f"split={P['split_mode']} ft={P['fine_tune_days']})")
        try:
            rows = run_single(prices_all, cfg, P, benchmark, capital, t_cost)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        for r in rows:
            r["suite"] = axis
            r["scenario"] = label
        all_rows.extend(rows)
        for r in rows:
            print(f"    {r['strategy']:16s} ret={r['ret_pct']:+6.2f}%  sharpe={r['sharpe']:+.2f}  "
                  f"maxDD={r['max_dd_pct']:+.1f}%  trades={r['n_trades']}")
    return all_rows


def suite_walkforward(prices_all, cfg, args, benchmark, capital, t_cost):
    cutoffs = ["2024-07-01", "2025-01-02", "2025-07-01", "2026-01-02"]
    all_rows = []
    for te in cutoffs:
        P = baseline(cfg, args)
        P["train_end"] = te
        P["oos_days"] = 126           # test ~6 months, non-overlapping-ish
        print(f"\n[ walkforward train_end={te}  (OOS ~126d) ]")
        try:
            rows = run_single(prices_all, cfg, P, benchmark, capital, t_cost)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        for r in rows:
            r["suite"] = "walkforward"
            r["scenario"] = f"te={te}"
        all_rows.extend(rows)
        for r in rows:
            print(f"    {r['strategy']:16s} ret={r['ret_pct']:+6.2f}%  sharpe={r['sharpe']:+.2f}  days={r['days']}")
    return all_rows


def suite_kf_edge(prices_all, cfg, args, benchmark, capital, t_cost):
    """KF-ONLY edge hunt: is the Kalman filter's defensive behavior a REAL standalone
    strategy? Runs the KF across EVERY walk-forward window for a grid of (entry_z, stop_z)
    and reports, per config, how it does ACROSS windows -- not on a single lucky cutoff.
    A genuine defensive edge should hold up in every window (positive-ish mean, small worst
    case, beats the index when the index falls). KF-only, so it's fast."""
    cutoffs = ["2024-07-01", "2025-01-02", "2025-07-01", "2026-01-02"]
    # Selectivity ladder. The gross/net decomposition showed cost coverage per trade rises
    # with entry_z (1.25x at 1.5 -> 3.0x at 3.0) while total gross falls, so the profitable
    # regime should be HIGH entry_z x HIGH breadth. This grid walks that ladder.
    grid = [(1.5, 4.0), (2.0, 4.0), (2.5, 4.0), (3.0, 4.0), (2.5, 3.0), (2.5, 5.0)]
    all_rows = []
    for ez, sz in grid:
        for te in cutoffs:
            P = baseline(cfg, args)
            P["include_dl"] = False          # KF only
            P["train_end"] = te
            P["oos_days"] = 126
            P["entry_z"] = ez
            P["stop_z"] = sz
            label = f"ez={ez}_sz={sz}_te={te}"
            try:
                rows = run_single(prices_all, cfg, P, benchmark, capital, t_cost)
            except Exception as e:
                print(f"    ERROR ({label}): {e}")
                continue
            # keep KF + benchmark; tag entry/stop in scenario (schema-stable, no new cols)
            for r in rows:
                r["suite"] = "kf_edge"
                r["scenario"] = label
            all_rows.extend(rows)
            kfr = next(r for r in rows if r["strategy"] == "Kalman Filter")
            bnr = next(r for r in rows if r["strategy"] == benchmark)
            print(f"    ez={ez:<4} sz={sz:<4} te={te}  KF ret={kfr['ret_pct']:+6.2f}%  "
                  f"sharpe={kfr['sharpe']:+.2f}  maxDD={kfr['max_dd_pct']:+.1f}%  "
                  f"| idx={bnr['ret_pct']:+6.2f}%  {'BEAT' if kfr['ret_pct']>bnr['ret_pct'] else 'lag '}")
    summarize_kf_edge(all_rows, benchmark)
    return all_rows


def summarize_kf_edge(rows, benchmark):
    """Aggregate KF performance ACROSS the walk-forward windows for each (entry_z, stop_z)."""
    df = pd.DataFrame(rows)
    kf = df[df["strategy"] == "Kalman Filter"].copy()
    idx = df[df["strategy"] == benchmark].set_index("scenario")["ret_pct"]
    kf["idx_ret"] = kf["scenario"].map(idx)
    # config = entry_z + stop_z (strip the te= part of the scenario label)
    kf["config"] = kf["scenario"].str.replace(r"_te=.*", "", regex=True)
    print("\n===== KF-EDGE SUMMARY: KF as a standalone strategy, ACROSS walk-forward windows =====")
    print(f"{'config':16s} {'mean ret%':>9s} {'worst%':>7s} {'best%':>7s} {'mean Shrp':>9s} "
          f"{'#win>0':>7s} {'beat idx':>8s}")
    for cfg_name, g in kf.groupby("config"):
        r = g["ret_pct"]
        beats = (g["ret_pct"] > g["idx_ret"]).sum()
        print(f"{cfg_name:16s} {r.mean():9.2f} {r.min():7.2f} {r.max():7.2f} "
              f"{g['sharpe'].mean():9.2f} {int((r>0).sum())}/{len(r):<5} {beats}/{len(r)}")
    print("\nRead: a REAL defensive edge shows mean ret > 0, a small (near-zero) worst case,")
    print("mean Sharpe > 0, and 'beat idx' high in the windows where the index fell. If mean")
    print("is ~0 and #win>0 is ~half, the KF is break-even noise, not a tradeable strategy.")


def summarize_seed_variance(rows):
    df = pd.DataFrame(rows)
    print("\n===== SEED-VARIANCE SUMMARY (is it edge or a coin flip?) =====")
    print(f"{'model':16s} {'mean ret%':>9s} {'std ret%':>9s} {'min':>7s} {'max':>7s} "
          f"{'mean/std':>9s} {'%profit':>8s} {'mean Sharpe':>11s}")
    for name, g in df.groupby("strategy"):
        r = g["ret_pct"]
        snr = r.mean() / r.std() if r.std() > 0 else float("nan")
        print(f"{name:16s} {r.mean():9.2f} {r.std():9.2f} {r.min():7.2f} {r.max():7.2f} "
              f"{snr:9.2f} {100*(r>0).mean():7.0f}% {g['sharpe'].mean():11.2f}")
    print("\nRead: if |mean/std| < ~0.5 and %profit ~50%, the P&L is NOISE, not an edge.")


# ======================================================================================
# Main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description="Strategy ablation / robustness harness")
    ap.add_argument("--suite", default="seed_variance",
                    choices=["seed_variance", "data_start", "coint_lookback", "split_mode",
                             "finetune_days", "entry_thresholds", "walkforward", "kf_edge", "all"])
    ap.add_argument("--train-end", default="2026-01-09", help="Cutoff for single-window suites")
    ap.add_argument("--epochs", type=int, default=None, help="DL epoch ceiling (default: config)")
    ap.add_argument("--seeds", type=int, default=8, help="Number of seeds for seed_variance")
    ap.add_argument("--models", default=None,
                    help="Comma-separated DL models to test (default all 4). "
                         "N-BEATS is the recommended drop (worst mean/Sharpe + ~8x turnover). "
                         "e.g. --models LSTM,Transformer")
    ap.add_argument("--kf-only", action="store_true", help="Skip DL (fast structural check)")
    ap.add_argument("--out", default="experiment_results.csv")
    args = ap.parse_args()

    cfg = load_config()
    if args.epochs is None:
        args.epochs = cfg["deep_learning"]["epochs"]

    # Restrict the DL model set (e.g. drop the cost-churning N-BEATS). The DL suites and the
    # ensemble read the module-level DL_NAMES, so narrowing it here narrows the whole run.
    global DL_NAMES
    if args.models:
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        bad = [m for m in want if m not in DL_NAMES]
        if bad:
            ap.error(f"unknown model(s) {bad}; choose from {DL_NAMES}")
        DL_NAMES = want
        print(f"DL models restricted to: {DL_NAMES}")
    market = cfg.get("market", {})
    benchmark = market.get("benchmark", "SPY")
    capital = float(cfg["trading"]["capital_per_pair"])
    t_cost = cfg["kalman_filter"]["t_cost"]
    pairs = [tuple(p) for p in cfg["trading"]["pairs"]]
    tickers = sorted({t for pr in pairs for t in pr}) + [benchmark]

    print(f"Pairs: {[f'{a}/{b}' for a, b in pairs]}  |  benchmark: {benchmark}  |  epochs<= {args.epochs}")
    print("Downloading full history once (2013-01-01 -> today)...")
    prices_all = download_prices(tickers, "2013-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    print(f"  data: {prices_all.index[0].date()} -> {prices_all.index[-1].date()}  ({len(prices_all)} rows)")

    results = []

    def do(suite):
        if suite == "seed_variance":
            P = baseline(cfg, args)
            P["seeds"] = list(range(args.seeds))
            print(f"\n########## SEED VARIANCE ({args.seeds} seeds, per-model, no ensembling) ##########")
            rows = run_seed_variance(prices_all, cfg, P, benchmark, capital, t_cost)
            for r in rows:
                r["suite"] = "seed_variance"
                r["scenario"] = f"seed={r['seed']}"
            summarize_seed_variance(rows)
            results.extend(rows)
        elif suite == "data_start":
            results.extend(suite_axis(prices_all, cfg, args, benchmark, capital, t_cost,
                                      "data_start", ["2015-01-01", "2018-01-01", "2020-01-01",
                                                     "2022-01-01", "2023-01-01"]))
        elif suite == "coint_lookback":
            results.extend(suite_axis(prices_all, cfg, args, benchmark, capital, t_cost,
                                      "coint_lookback", [None, 1008, 756, 504, 378, 252]))
        elif suite == "split_mode":
            results.extend(suite_axis(prices_all, cfg, args, benchmark, capital, t_cost,
                                      "split_mode", ["new", "old", "no_finetune"]))
        elif suite == "finetune_days":
            results.extend(suite_axis(prices_all, cfg, args, benchmark, capital, t_cost,
                                      "fine_tune_days", [0, 63, 126, 252]))
        elif suite == "entry_thresholds":
            args.kf_only = True   # KF-only sweep (fast); entry_z only affects the KF
            results.extend(suite_axis(prices_all, cfg, args, benchmark, capital, t_cost,
                                      "entry_z", [1.0, 1.25, 1.5, 2.0]))
        elif suite == "walkforward":
            results.extend(suite_walkforward(prices_all, cfg, args, benchmark, capital, t_cost))
        elif suite == "kf_edge":
            results.extend(suite_kf_edge(prices_all, cfg, args, benchmark, capital, t_cost))

    if args.suite == "all":
        for s in ["seed_variance", "data_start", "coint_lookback", "split_mode",
                  "finetune_days", "entry_thresholds", "walkforward", "kf_edge"]:
            do(s)
    else:
        do(args.suite)

    if results:
        df = pd.DataFrame(results)
        header = not os.path.exists(args.out)
        df.to_csv(args.out, mode="a", header=header, index=False)
        print(f"\nAppended {len(df)} rows to {args.out}")
        print("Tip: load it in pandas and pivot by scenario/strategy to compare ret_pct & sharpe.")


if __name__ == "__main__":
    main()

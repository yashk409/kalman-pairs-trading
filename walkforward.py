"""
walkforward.py -- Phase 0: the HONEST evaluation harness.

Every earlier result (including the Sharpe 1.55) selected its pairs ONCE, as-of the most
recent date, and then backtested them on WINDOWS THAT PRECEDED THAT DATE. The pairs were
therefore chosen partly *because* they survived the test period. That is selection
lookahead, and it inflates results.

This harness removes it. At every cutoff it RE-SCREENS the universe using only data
available at that moment, then trades the resulting pairs forward into a period that no
part of selection, fitting or scaling has seen. Nothing about the test window can leak
backwards into which pairs get picked.

It also adds the two controls that were missing:

  * Static-Z  -- a plain rolling z-score on a STATIC OLS-beta spread, no Kalman filter at
                all. If the Kalman machinery cannot beat this, the Kalman machinery is
                decoration. We never once ran this control.
  * Gross vs net -- P&L with costs switched off and on, reported side by side, so we can
                see whether an edge exists at all before execution eats it.

GO / NO-GO (fixed in advance, before seeing any number):
  KF Sharpe after the rolling re-screen < 0.3  ->  the 1.55 was selection bias. Stop.

USAGE
  python3 walkforward.py                                    # defaults: US, 4 cutoffs, 126d OOS
  python3 walkforward.py --top 30 --oos-days 126
  python3 walkforward.py --cutoffs 2024-07-01,2025-01-02,2025-07-01,2026-01-02
"""
import argparse
import itertools
import warnings
import os

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import yaml

from screener import (MARKET_PROFILES, build_sector_map, all_tickers,
                      download_universe, evaluate_pair, score_row, select_diversified)
from data_loader import get_cointegration_params
from experiments import kf_signal_df, backtest_pnl, compute_metrics
from models.kalman_filter import rolling_zscore
from signals import kf_decide


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# ======================================================================================
# Point-in-time screen: uses ONLY data <= asof.
# ======================================================================================
def screen_asof(close, asof, universe, sector_map, scfg, top, max_per_sectorpair):
    """Re-run the full screener using only prices up to `asof`. Returns selected pairs."""
    upto = close.loc[:asof].dropna(how="all", axis=1)
    win = upto.iloc[-scfg["lookback"]:]
    # Keep tickers with a full history over the screening window -- a name that only listed
    # halfway through cannot be judged for cointegration.
    win = win.dropna(axis=1)
    cols = list(win.columns)
    if len(cols) < 2:
        return []

    # Vectorized return-correlation prefilter. Engle-Granger is the expensive step, so we
    # only pay for it on pairs whose legs actually co-move. Same rule evaluate_pair applies,
    # just computed once as a matrix instead of O(n^2) times in Python.
    rets = np.log(win).diff()
    C = rets.corr().values
    idx = {c: i for i, c in enumerate(cols)}
    cands = [(a, b) for a, b in itertools.combinations(cols, 2)
             if abs(C[idx[a], idx[b]]) >= scfg["corr_min"]]

    passed = []
    for tA, tB in cands:
        m = evaluate_pair(tA, tB, upto, scfg)
        if m is not None:
            m["score"] = score_row(m)
            passed.append(m)
    passed.sort(key=lambda r: r["score"], reverse=True)
    sel = select_diversified(passed, sector_map, top, max_per_sectorpair)
    return [(r["y"], r["x"]) for r in sel], len(cands), len(passed)


# ======================================================================================
# Static-beta z-score control (no Kalman filter anywhere).
# ======================================================================================
def build_vol_regime(close, benchmark):
    """
    Point-in-time volatility regime of the benchmark: 21d realized vol vs its own TRAILING
    252d 25th percentile. `rolling(252).quantile(.25)` is trailing, so there is no lookahead
    -- that part of the implementation is correct as written in backtest.py.

    NOTE: this is evaluated here, not assumed. The premise "mean reversion needs high vol" is
    UNTESTED and our own factor-residual data points the other way (the edge died in 2022, the
    highest-vol year, and was strongest in the low-vol 2016-2019 bull). The honest harness must
    measure the filter's effect rather than take it on faith -- hence VolFilter runs as its own
    strategy alongside the unfiltered KF, so the two are directly comparable.
    """
    r = np.log(close[benchmark] / close[benchmark].shift(1))
    vol = r.rolling(21).std() * np.sqrt(252)
    q25 = vol.rolling(252).quantile(0.25)
    regime = pd.Series("NORMAL", index=close.index)
    regime[vol < q25] = "LOW"
    return regime


def static_signal_df(prices, A, B, c, kf, index):
    """Plain rolling z-score on a spread built from the STATIC OLS beta/alpha fitted on the
    screening window. Identical entry/exit/stop rule as the KF, so the ONLY difference is
    whether beta is dynamic (Kalman) or fixed. This isolates what the filter contributes."""
    lp = np.log(prices[[A, B]])
    beta, alpha = c["beta"], c["alpha"]
    spread = lp[A] - beta * lp[B] - alpha
    z = rolling_zscore(spread, kf["roll_win"])
    sig, prev = np.zeros(len(z)), 0
    for i, zz in enumerate(z):
        prev = kf_decide(zz, prev, kf["stop_z"], kf["entry_z"], kf["exit_z"])
        sig[i] = prev
    wA = 1.0 / (1.0 + abs(beta))
    wB = -beta / (1.0 + abs(beta))
    return pd.DataFrame({"signal": sig, "wA": wA, "wB": wB}, index=z.index)


def main():
    ap = argparse.ArgumentParser(description="Phase 0: honest rolling-rescreen walk-forward")
    ap.add_argument("--market", choices=list(MARKET_PROFILES), default="us")
    ap.add_argument("--cutoffs", default="2024-07-01,2025-01-02,2025-07-01,2026-01-02")
    ap.add_argument("--oos-days", type=int, default=126)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--lookback", type=int, default=756, help="screening window (days)")
    ap.add_argument("--coint-pmax", type=float, default=0.01)
    ap.add_argument("--corr-min", type=float, default=0.4)
    ap.add_argument("--max-per-sectorpair", type=int, default=2)
    ap.add_argument("--entry-z", type=float, default=None, help="override config entry_z")
    ap.add_argument("--cost-coverage", type=float, default=3.0,
                    help="screener cost barrier: required gross/round-trip ratio")
    ap.add_argument("--out", default="walkforward_results.csv")
    ap.add_argument("--report", default=None, help="also write a markdown report to this path")
    args = ap.parse_args()

    cfg = load_config()
    kf = dict(cfg["kalman_filter"])
    if args.entry_z is not None:
        kf["entry_z"] = args.entry_z
    t_cost = kf["t_cost"]
    capital = float(cfg["trading"]["capital_per_pair"])
    prof = MARKET_PROFILES[args.market]
    universe, bench = prof["universe"], prof["config"]["benchmark"]
    sector_map = build_sector_map(universe)
    tickers = all_tickers(universe)

    scfg = {
        "lookback": args.lookback, "min_obs": min(500, args.lookback),
        "coint_pmax": args.coint_pmax, "coint_pmax_half": max(0.10, args.coint_pmax * 2),
        "adf_pmax": 0.10, "corr_min": args.corr_min,
        "halflife_min": 5.0, "halflife_max": 60.0,
        "beta_min": 0.1, "beta_max": 10.0, "beta_drift_max": 0.5,
        "min_crossings": 8,
        # Cost-coverage barrier, using the SAME bands/cost the backtest trades with.
        "entry_z": kf["entry_z"], "exit_z": kf["exit_z"], "t_cost": t_cost,
        "cost_coverage": args.cost_coverage,
    }
    _rt = 2.0 * t_cost
    print(f"Cost barrier: gross >= {args.cost_coverage}x round-trip ({_rt*1e4:.0f}bps) "
          f"-> spread_sd >= {args.cost_coverage*_rt*2.0/(kf['entry_z']-kf['exit_z'])*1e4:.0f}bps "
          f"at beta=1")

    cutoffs = [c.strip() for c in args.cutoffs.split(",")]
    print("=" * 96)
    print("PHASE 0 -- ROLLING RE-SCREEN WALK-FORWARD (no selection lookahead)")
    print("=" * 96)
    print(f"market={args.market}  universe={len(tickers)} tickers  top={args.top} pairs/cutoff")
    print(f"entry_z={kf['entry_z']}  t_cost={t_cost*1e4:.0f}bps/leg  book={capital:,.0f}")
    print(f"GO/NO-GO: KF Sharpe < 0.30 after re-screen => edge was selection bias.\n")

    print("Downloading full universe once...")
    close = download_universe(tickers + [bench], "2013-01-01",
                             pd.Timestamp.today().strftime("%Y-%m-%d"))
    print(f"  {close.shape[1]} tickers, {close.index[0].date()} -> {close.index[-1].date()}")

    vol_regime = build_vol_regime(close, bench)
    _low = float((vol_regime == "LOW").mean())
    print(f"  vol regime: {_low:.1%} of days are LOW (entries blocked for KF+VolFilter)\n")

    rows = []
    for te in cutoffs:
        print("-" * 96)
        print(f"CUTOFF {te}  -- screening on data <= {te} ONLY")
        pairs, n_cand, n_pass = screen_asof(close, te, universe, sector_map, scfg,
                                            args.top, args.max_per_sectorpair)
        if not pairs:
            print("  no pairs survived; skipping")
            continue
        print(f"  {n_cand:,} candidates -> {n_pass} passed filters -> {len(pairs)} selected")
        print(f"  e.g. {', '.join(f'{a}/{b}' for a, b in pairs[:6])}{' ...' if len(pairs) > 6 else ''}")

        tk = sorted({t for p in pairs for t in p})
        px = close[tk].dropna()
        coint = get_cointegration_params(px, pairs, te, args.lookback)

        # Test window: strictly AFTER the cutoff. Selection, beta, lambda, Q, R and the
        # z-score normalization have all seen only data <= te.
        td = px.index[px.index >= pd.to_datetime(te)][: args.oos_days + 1]
        if len(td) < 20:
            print("  insufficient OOS data; skipping")
            continue

        strategies = {
            "Kalman Filter": {(A, B): kf_signal_df(px, A, B, coint[(A, B)], kf, te) for A, B in pairs},
            "KF+VolFilter":  {(A, B): kf_signal_df(px, A, B, coint[(A, B)], kf, te, vol_regime)
                              for A, B in pairs},
            "Static-Z":      {(A, B): static_signal_df(px, A, B, coint[(A, B)], kf, td) for A, B in pairs},
        }

        bpx = close[bench].reindex(td).ffill()
        bsh = capital / float(bpx.iloc[0])
        bpnl = [0.0] + [bsh * (float(bpx.iloc[i]) - float(bpx.iloc[0])) for i in range(1, len(td))]

        print(f"  OOS {td[0].date()} -> {td[-1].date()} ({len(td)-1}d, {len(pairs)} pairs)")
        print(f"    {'strategy':16s} {'GROSS':>8s} {'NET':>8s} {'cost':>7s} {'Sharpe':>7s} {'maxDD':>7s} {'trades':>7s}")
        for name, sig in strategies.items():
            gp, ntr = backtest_pnl(sig, td, px, tk, pairs, capital, 0.0)
            npl, _  = backtest_pnl(sig, td, px, tk, pairs, capital, t_cost)
            mg = compute_metrics(gp, capital, ntr)
            mn = compute_metrics(npl, capital, ntr)
            print(f"    {name:16s} {mg['ret_pct']:+7.2f}% {mn['ret_pct']:+7.2f}% "
                  f"{mg['ret_pct']-mn['ret_pct']:6.2f}% {mn['sharpe']:+7.2f} {mn['max_dd_pct']:+6.1f}% {ntr:7d}")
            rows.append({"cutoff": te, "strategy": name, "n_pairs": len(pairs),
                         "gross_pct": mg["ret_pct"], "net_pct": mn["ret_pct"],
                         "cost_pct": round(mg["ret_pct"] - mn["ret_pct"], 2),
                         "sharpe": mn["sharpe"], "max_dd_pct": mn["max_dd_pct"],
                         "ann_vol_pct": mn["ann_vol_pct"], "n_trades": ntr, "days": mn["days"]})
        mb = compute_metrics(bpnl, capital, 0)
        print(f"    {bench:16s} {'--':>8s} {mb['ret_pct']:+7.2f}% {'--':>7s} {mb['sharpe']:+7.2f} {mb['max_dd_pct']:+6.1f}%")
        rows.append({"cutoff": te, "strategy": bench, "n_pairs": 0,
                     "gross_pct": mb["ret_pct"], "net_pct": mb["ret_pct"], "cost_pct": 0.0,
                     "sharpe": mb["sharpe"], "max_dd_pct": mb["max_dd_pct"],
                     "ann_vol_pct": mb["ann_vol_pct"], "n_trades": 0, "days": mb["days"]})

    if not rows:
        print("\nNo results.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 96)
    print("VERDICT -- aggregated across cutoffs (each window's pairs chosen blind to it)")
    print("=" * 96)
    from scipy import stats
    print(f"{'strategy':16s} {'GROSS':>8s} {'NET':>8s} {'Sharpe':>7s} {'maxDD':>7s} {'win':>5s} {'t':>6s} {'p':>6s}")
    for name, g in df.groupby("strategy", sort=False):
        r = g["net_pct"].values
        t, p = stats.ttest_1samp(r, 0) if len(r) > 1 else (np.nan, np.nan)
        print(f"{name:16s} {g['gross_pct'].mean():+7.2f}% {r.mean():+7.2f}% {g['sharpe'].mean():+7.2f} "
              f"{g['max_dd_pct'].mean():+6.1f}% {int((r>0).sum())}/{len(r):<3} {t:6.2f} {p:6.3f}")

    kfr = df[df.strategy == "Kalman Filter"]
    sh = kfr["sharpe"].mean()
    print("\n" + "=" * 96)
    if sh < 0.30:
        print(f"NO-GO: KF Sharpe {sh:.2f} < 0.30. The edge does not survive point-in-time selection.")
    else:
        print(f"GO: KF Sharpe {sh:.2f} >= 0.30 with pair selection blind to the test window.")
    print("=" * 96)
    print(f"Saved {len(df)} rows to {args.out}")

    if args.report:
        write_report(args.report, df, args, kf, t_cost, bench, _low, sh)
        print(f"Wrote markdown report to {args.report}")


def write_report(path, df, args, kf, t_cost, bench, low_frac, sh):
    """
    Markdown summary. COVERAGE (gross edge / cost, per unit turnover) is reported FIRST,
    because it is the single number that killed both predecessor projects and it was legible
    from day one in each of them.
    """
    from scipy import stats
    L = []
    A = L.append
    A("# Honest Walk-Forward — Kalman Pairs")
    A("")
    A(f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}_")
    A("")
    A("Pairs are **re-screened point-in-time at every cutoff** — selection never sees its own "
      "test window. This is the only evaluator in the repo whose numbers can be trusted; "
      "`backtest.py` trades whatever pairs happen to be sitting in `config.yaml`.")
    A("")
    A("## Configuration")
    A("")
    A(f"| | |")
    A(f"|---|---|")
    A(f"| entry_z / exit_z / stop_z | {kf['entry_z']} / {kf['exit_z']} / {kf['stop_z']} |")
    A(f"| t_cost | {t_cost*1e4:.0f} bps/leg (round trip {2*t_cost*1e4:.0f} bps) |")
    A(f"| cost-coverage barrier | {args.cost_coverage}× |")
    A(f"| pairs per cutoff | up to {args.top} |")
    A(f"| coint p-max | {args.coint_pmax} (persistence: both halves) |")
    A(f"| vol regime | {low_frac:.1%} of days LOW (entries blocked for KF+VolFilter) |")
    A("")

    A("## Verdict — aggregated across cutoffs")
    A("")
    A("| strategy | gross | **net** | Sharpe | maxDD | windows + | t | p |")
    A("|---|---|---|---|---|---|---|---|")
    for name, g in df.groupby("strategy", sort=False):
        r = g["net_pct"].values
        t, p = stats.ttest_1samp(r, 0) if len(r) > 1 else (float("nan"), float("nan"))
        A(f"| {name} | {g['gross_pct'].mean():+.2f}% | **{r.mean():+.2f}%** | "
          f"{g['sharpe'].mean():+.2f} | {g['max_dd_pct'].mean():+.1f}% | "
          f"{int((r>0).sum())}/{len(r)} | {t:.2f} | {p:.3f} |")
    A("")

    # The coverage ratio: gross edge vs cost, per unit of turnover.
    A("## Coverage — the number that matters")
    A("")
    A("> `coverage = gross edge / transaction cost`. **Below 1.0× the average trade earns "
      "less than it costs to make**, and no model, threshold or execution tweak fixes that. "
      "The pairs predecessor died at 0.94×; the factor-residual successor died at 0.33×.")
    A("")
    A("| strategy | gross | cost | **coverage** | trades |")
    A("|---|---|---|---|---|")
    for name, g in df.groupby("strategy", sort=False):
        if name == bench:
            continue
        gr, co = g["gross_pct"].mean(), g["cost_pct"].mean()
        cov = (gr / co) if co > 0 else float("nan")
        A(f"| {name} | {gr:+.2f}% | {co:.2f}% | **{cov:.2f}×** | {g['n_trades'].mean():.0f} |")
    A("")

    A("## Per-cutoff detail")
    A("")
    A("| cutoff | strategy | pairs | gross | net | Sharpe | maxDD | trades |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        A(f"| {r['cutoff']} | {r['strategy']} | {r['n_pairs']} | {r['gross_pct']:+.2f}% | "
          f"{r['net_pct']:+.2f}% | {r['sharpe']:+.2f} | {r['max_dd_pct']:+.1f}% | {r['n_trades']} |")
    A("")

    A("## Read this")
    A("")
    kfr = df[df.strategy == "Kalman Filter"]
    vfr = df[df.strategy == "KF+VolFilter"]
    szr = df[df.strategy == "Static-Z"]
    if sh < 0.30:
        A(f"**NO-GO. KF Sharpe {sh:.2f} < 0.30.** The edge does not survive point-in-time "
          f"pair selection. Any positive number produced by `backtest.py` is selection bias.")
    else:
        A(f"**GO. KF Sharpe {sh:.2f} ≥ 0.30** with selection blind to the test window.")
    A("")
    if len(vfr) and len(kfr):
        d = vfr["net_pct"].mean() - kfr["net_pct"].mean()
        verdict = "HELPS" if d > 0 else "HURTS"
        A(f"- **Vol-regime filter {verdict}**: net {vfr['net_pct'].mean():+.2f}% with it vs "
          f"{kfr['net_pct'].mean():+.2f}% without ({d:+.2f} pp). It blocks entries on "
          f"{low_frac:.0%} of days, which also costs ~√(1−{low_frac:.2f}) of breadth — it must "
          f"beat that hurdle, not merely cut costs.")
    if len(szr) and len(kfr):
        A(f"- **Does the Kalman filter earn its keep?** KF gross {kfr['gross_pct'].mean():+.2f}% "
          f"vs Static-Z (no filter at all) {szr['gross_pct'].mean():+.2f}%. If these are equal, "
          f"the Kalman machinery is decoration.")
    A("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

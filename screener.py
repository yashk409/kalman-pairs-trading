"""
Cointegration-based pairs screener.

Scans a diversified universe of ETFs and equities, tests every candidate pair for a
*currently* tradeable mean-reverting relationship, and emits a diversified,
non-overlapping shortlist that the existing backtest / live pipeline can consume.

Design notes (learned the hard way from EWA/EWC):
  * Cointegration is tested on a RECENT window (default ~3y), not deep history, because
    relationships decay (EWA/EWC was Chan's 2006-2012 pair and is broken today).
  * Persistence is required: the pair must cointegrate in BOTH halves of the window and
    keep a stable hedge ratio, so we don't select a spread that only *looks* stationary
    over one lucky stretch.
  * Half-life must sit in a tradeable band (fast enough to round-trip, slow enough that
    it isn't just microstructure noise).
  * Selection is diversified: each ticker is used at most once and repeated sector
    combinations are capped, so the portfolio isn't 8 flavors of the same bet.
"""
import argparse
import os
import itertools
import concurrent.futures
import multiprocessing

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# --------------------------------------------------------------------------------------
# Universe: grouped by theme/sector. Groups are used both to tag pairs for diversification
# and (optionally) to bias the search toward economically sensible relationships.
# --------------------------------------------------------------------------------------
# Deep, sector-diversified US universe (~270 liquid names with long history). Cointegration
# is found within sectors, so each group is intentionally deep. Acquired/renamed tickers are
# avoided (e.g. PXD->XOM, ATVI->MSFT, FISV->FI, FB->META, ANTM->ELV, DISCA->WBD). Every "_etf"
# group is honored by --etf-only. First group wins for a ticker listed in several (sector map).
UNIVERSE_US = {
    "country_etf":   ["EWA", "EWC", "EWG", "EWU", "EWJ", "EWZ", "EWW", "EWH", "EWL", "EWP", "EWQ", "EWY", "EWT", "EWS", "EWM", "EWD", "EWI", "EWN", "INDA", "FXI", "EZA", "TUR"],
    "broad_etf":     ["QQQ", "DIA", "IWM", "MDY", "VTI", "RSP"],
    "sector_etf":    ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLY", "XLC"],
    "industry_etf":  ["SMH", "SOXX", "KBE", "KRE", "XBI", "IBB", "ITB", "XHB", "OIH", "XOP", "KIE", "XRT", "IYT", "JETS", "TAN", "ITA"],
    "commodity_etf": ["GLD", "SLV", "GDX", "GDXJ", "USO", "XME", "DBC", "UNG", "DBA", "CORN", "WEAT", "PPLT", "PALL", "COPX", "URA", "DBB"],
    "bond_etf":      ["TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND", "TIP", "MBB", "EMB", "JNK", "VCIT", "VCSH", "BIL"],
    "money_banks":   ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "BK", "STT"],
    "regional_banks":["RF", "KEY", "CFG", "HBAN", "FITB", "MTB", "CMA", "ZION", "ALLY", "FHN", "SNV"],
    "oil_gas":       ["XOM", "CVX", "COP", "EOG", "OXY", "PSX", "VLO", "MPC", "HES", "DVN", "FANG", "MRO", "APA", "CTRA"],
    "oilfield_svcs": ["SLB", "HAL", "BKR", "NOV", "FTI"],
    "midstream":     ["KMI", "WMB", "OKE", "ET", "EPD", "MPLX", "LNG", "TRGP"],
    "big_tech":      ["MSFT", "AAPL", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA"],
    "software":      ["ORCL", "CRM", "ADBE", "NOW", "INTU", "IBM", "CSCO", "SNOW", "WDAY", "TEAM", "DDOG", "PANW", "FTNT", "CRWD", "ZS", "NET", "ADSK", "ANSS"],
    "internet":      ["NFLX", "DIS", "CMCSA", "WBD", "PARA", "SPOT", "PINS", "SNAP", "UBER", "LYFT", "ABNB", "BKNG", "EXPE"],
    "semis":         ["AMD", "INTC", "TXN", "ADI", "MU", "QCOM", "AVGO", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "MCHP", "NXPI", "TSM", "ASML", "SWKS", "TER", "MPWR"],
    "payments":      ["V", "MA", "AXP", "PYPL", "FI", "FIS", "GPN", "COF", "DFS", "SYF"],
    "retail":        ["WMT", "TGT", "COST", "HD", "LOW", "DG", "DLTR", "KR", "BJ", "ROST", "TJX", "BURL", "ULTA", "BBY", "TSCO"],
    "ecommerce":     ["EBAY", "ETSY", "W", "CHWY", "SHOP", "MELI", "CVNA"],
    "pharma":        ["JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "GILD", "AMGN", "VRTX", "REGN", "BIIB", "ZTS"],
    "biotech":       ["MRNA", "INCY", "BMRN", "ALNY", "EXEL", "SRPT", "NBIX"],
    "medical_dev":   ["MDT", "ABT", "SYK", "BSX", "ISRG", "EW", "ZBH", "BDX", "DHR", "TMO", "A", "IDXX", "RMD", "DXCM", "ALGN"],
    "health_ins":    ["UNH", "CI", "ELV", "HUM", "CNC", "CVS", "MOH"],
    "telecom":       ["VZ", "T", "TMUS"],
    "staples":       ["KO", "PEP", "PG", "CL", "MO", "PM", "GIS", "K", "HSY", "KHC", "CAG", "MKC", "CLX", "KMB", "SYY", "STZ", "TAP", "KDP", "MDLZ", "ADM"],
    "autos":         ["GM", "F", "TM", "STLA", "HMC"],
    "airlines":      ["DAL", "UAL", "AAL", "LUV", "ALK"],
    "travel":        ["MAR", "HLT", "RCL", "CCL", "NCLH", "MGM", "LVS", "WYNN", "H"],
    "utilities":     ["DUK", "SO", "NEE", "D", "AEP", "EXC", "XEL", "ED", "WEC", "ES", "PEG", "SRE", "EIX", "AEE", "CMS", "DTE", "PPL", "FE", "ETR"],
    "industrials":   ["CAT", "DE", "HON", "GE", "MMM", "EMR", "ETN", "ITW", "PH", "ROK", "DOV", "CMI", "PCAR", "GWW", "FAST", "IR", "AME"],
    "aero_defense":  ["BA", "LMT", "RTX", "NOC", "GD", "LHX", "TDG", "HWM", "HII", "TXT"],
    "transports":    ["UNP", "CSX", "NSC", "ODFL", "JBHT", "CHRW", "FDX", "UPS", "EXPD", "XPO"],
    "insurance":     ["MET", "PRU", "AIG", "ALL", "TRV", "PGR", "CB", "HIG", "AFL", "CINF", "WRB", "ACGL", "L"],
    "asset_mgmt":    ["BLK", "BX", "KKR", "APO", "BEN", "TROW", "IVZ", "AMP", "SCHW", "NTRS", "ICE", "CME", "NDAQ", "MCO", "SPGI", "MSCI"],
    "homebuilders":  ["DHI", "LEN", "PHM", "NVR", "TOL", "KBH", "MHO", "BLDR", "MAS", "MHK"],
    "reits":         ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "DLR", "WELL", "VICI", "EQR", "AVB", "EXR", "MAA", "ESS", "ARE", "INVH", "KIM", "REG", "BXP"],
    "chemicals":     ["LIN", "APD", "SHW", "ECL", "DD", "DOW", "PPG", "ALB", "LYB", "CE", "IFF", "EMN", "RPM"],
    "metals_mining": ["FCX", "NUE", "STLD", "CLF", "X", "AA", "NEM", "GOLD", "RIO", "BHP", "VALE", "SCCO", "AEM"],
    "restaurants":   ["MCD", "SBUX", "CMG", "YUM", "DRI", "DPZ", "QSR", "WEN", "TXRH", "CAKE"],
    "apparel":       ["NKE", "LULU", "VFC", "RL", "TPR", "CPRI", "PVH", "UAA", "DECK", "CROX", "SKX"],
    "agriculture":   ["BG", "MOS", "CF", "NTR", "FMC", "CTVA"],
}

# Indian universe: liquid NSE names + ETFs (yfinance uses the ".NS" suffix for the NSE).
# NOTE: TATAMOTORS.NS and LTIM.NS are omitted - Yahoo currently 404s their history after
# corporate actions (Tata Motors demerger, LTI/Mindtree merger). Re-add if Yahoo restores.
UNIVERSE_IN = {
    "it_services":   ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS"],
    "priv_banks":    ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS", "RBLBANK.NS"],
    "psu_banks":     ["SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS", "INDIANB.NS", "BANKINDIA.NS"],
    "nbfc_finance":  ["BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS", "MUTHOOTFIN.NS", "LICHSGFIN.NS", "PFC.NS", "RECLTD.NS", "SBICARD.NS", "HDFCAMC.NS"],
    "insurance":     ["HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS", "LICI.NS", "GICRE.NS", "NIACL.NS"],
    "auto":          ["MARUTI.NS", "M&M.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS", "MOTHERSON.NS", "BOSCHLTD.NS"],
    "energy_oil":    ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "GAIL.NS", "HINDPETRO.NS", "PETRONET.NS", "OIL.NS", "IGL.NS", "MGL.NS"],
    "power_util":    ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIPOWER.NS", "JSWENERGY.NS", "ADANIGREEN.NS", "NHPC.NS", "SJVN.NS", "TORNTPOWER.NS"],
    "metals":        ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS", "NMDC.NS", "SAIL.NS", "COALINDIA.NS", "HINDZINC.NS", "NATIONALUM.NS", "APLAPOLLO.NS"],
    "fmcg":          ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "TATACONSUM.NS", "COLPAL.NS", "VBL.NS", "UBL.NS"],
    "pharma":        ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS", "LUPIN.NS", "TORNTPHARM.NS", "ALKEM.NS", "ZYDUSLIFE.NS", "BIOCON.NS", "GLENMARK.NS"],
    "healthcare":    ["APOLLOHOSP.NS", "MAXHEALTH.NS", "FORTIS.NS", "LALPATHLAB.NS", "METROPOLIS.NS"],
    "cement":        ["ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS", "DALBHARAT.NS", "JKCEMENT.NS", "RAMCOCEM.NS"],
    "telecom":       ["BHARTIARTL.NS", "IDEA.NS", "INDUSTOWER.NS"],
    "capital_goods": ["LT.NS", "SIEMENS.NS", "ABB.NS", "CGPOWER.NS", "BEL.NS", "HAL.NS", "BHEL.NS", "THERMAX.NS"],
    "infra_logistics": ["ADANIPORTS.NS", "ADANIENT.NS", "CONCOR.NS", "INDIGO.NS"],
    "consumer_dur":  ["TITAN.NS", "HAVELLS.NS", "VOLTAS.NS", "DIXON.NS", "CROMPTON.NS", "WHIRLPOOL.NS", "BLUESTARCO.NS"],
    "retail":        ["DMART.NS", "TRENT.NS", "ABFRL.NS", "JUBLFOOD.NS", "PAGEIND.NS"],
    "paints_chem":   ["ASIANPAINT.NS", "BERGEPAINT.NS", "PIDILITIND.NS", "SRF.NS", "PIIND.NS", "DEEPAKNTR.NS", "AARTIIND.NS", "TATACHEM.NS", "UPL.NS"],
    "realty":        ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS"],
    "etf":           ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "JUNIORBEES.NS", "PSUBNKBEES.NS"],
}

# Market profiles written into config.yaml on --write-config so the backtest / live
# scripts use the right benchmark, exchange hours and currency.
MARKET_PROFILES = {
    "us": {
        "universe": UNIVERSE_US,
        "config":   {"benchmark": "SPY", "timezone": "America/New_York",
                     "open": "09:30", "close": "16:00", "currency": "$"},
    },
    "india": {
        "universe": UNIVERSE_IN,
        "config":   {"benchmark": "^NSEI", "timezone": "Asia/Kolkata",
                     "open": "09:15", "close": "15:30", "currency": "Rs"},
    },
}


def build_sector_map(universe):
    """ticker -> sector (first group wins for tickers that appear in multiple)."""
    sec = {}
    for group, tickers in universe.items():
        for t in tickers:
            sec.setdefault(t, group)
    return sec


def all_tickers(universe):
    seen = []
    for tickers in universe.values():
        for t in tickers:
            if t not in seen:
                seen.append(t)
    return seen


def download_universe(tickers, start, end):
    """Adjusted closes for the whole universe; failed/empty tickers are dropped."""
    # threads=False: prevents yfinance's SQLite tz-cache from being corrupted by
    # concurrent writes during large batch downloads (the cause of the "unable to open
    # database file" errors and the resulting false "no price data" negatives).
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, threads=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all", axis=1).sort_index()
    return close


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def hedge_ratio(log_y, log_x):
    """OLS: log_y = alpha + beta*log_x. Returns (alpha, beta, resid)."""
    X = sm.add_constant(log_x)
    res = sm.OLS(log_y, X).fit()
    return float(res.params.iloc[0]), float(res.params.iloc[1]), res.resid


def half_life(spread):
    """Mean-reversion half-life (days) via AR(1) / Ornstein-Uhlenbeck fit."""
    s = spread.dropna()
    lag = s.shift(1)
    delta = s - lag
    d = pd.concat([delta, lag], axis=1).dropna()
    d.columns = ["delta", "lag"]
    if len(d) < 20:
        return np.inf
    kappa = sm.OLS(d["delta"], sm.add_constant(d["lag"])).fit().params.iloc[1]
    if kappa >= 0:
        return np.inf
    return float(-np.log(2) / kappa)


def zero_crossings(spread):
    """How many times the (de-meaned) spread crosses zero — a proxy for trade frequency."""
    s = (spread - spread.mean()).dropna().values
    return int(np.sum(np.diff(np.sign(s)) != 0))


def coint_best_direction(log_a, log_b):
    """
    Engle-Granger is order-dependent; test both directions and keep the stronger one.
    Returns (p_value, y_ticker_is_a, alpha, beta, resid).
    """
    p_ab = coint(log_a, log_b)[1]
    p_ba = coint(log_b, log_a)[1]
    if p_ab <= p_ba:
        a, b, resid = hedge_ratio(log_a, log_b)
        return p_ab, True, a, b, resid
    else:
        a, b, resid = hedge_ratio(log_b, log_a)
        return p_ba, False, a, b, resid


# --------------------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------------------
_worker_close = None
_worker_cfg = None

def _init_worker(close_df, cfg):
    # Cap BLAS threads inside workers. Apple's Accelerate (and OpenBLAS) already multithread,
    # so N processes x M BLAS threads oversubscribes the 10 cores and is frequently SLOWER
    # than serial. One thread per worker; the process pool provides the parallelism.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
    global _worker_close, _worker_cfg
    _worker_close = close_df
    _worker_cfg = cfg

def _evaluate_wrapper(combo):
    tA, tB = combo
    return evaluate_pair(tA, tB, _worker_close, _worker_cfg)

def evaluate_pair(tA, tB, close, cfg):
    """Return a metrics dict if the pair passes all filters, else None."""
    sub = close[[tA, tB]].dropna()
    if len(sub) < cfg["min_obs"]:
        return None
    sub = sub.iloc[-cfg["lookback"]:]
    if len(sub) < cfg["min_obs"]:
        return None

    la, lb = np.log(sub[tA]), np.log(sub[tB])

    # Return-correlation prefilter: legs must actually move together to hedge well.
    ret_corr = la.diff().corr(lb.diff())
    if not np.isfinite(ret_corr) or abs(ret_corr) < cfg["corr_min"]:
        return None

    # Primary cointegration test on the full recent window.
    p_full, a_is_y, alpha, beta, resid = coint_best_direction(la, lb)
    if not np.isfinite(p_full) or p_full > cfg["coint_pmax"]:
        return None
    if abs(beta) < cfg["beta_min"] or abs(beta) > cfg["beta_max"]:
        return None

    # Orient so y = the dependent leg, x = the hedge leg.
    y_tkr, x_tkr = (tA, tB) if a_is_y else (tB, tA)
    ly = la if a_is_y else lb
    lx = lb if a_is_y else la
    spread = ly - alpha - beta * lx

    hl = half_life(spread)
    if not (cfg["halflife_min"] <= hl <= cfg["halflife_max"]):
        return None

    crossings = zero_crossings(spread)
    if crossings < cfg["min_crossings"]:
        return None

    # Persistence: require a stable hedge ratio across halves.
    half = len(sub) // 2
    ly1, lx1 = ly.iloc[:half], lx.iloc[:half]
    ly2, lx2 = ly.iloc[half:], lx.iloc[half:]

    # PERSISTENCE: require cointegration in BOTH halves, not just over the full window. This
    # was silently dropped (coint_p_h1/h2 were being returned as None and coint_pmax_half was
    # dead config). It is the primary defense against the EWA/EWC trap -- a pair that
    # cointegrates over one lucky stretch and nowhere else. With ~112k candidates screened at
    # p<0.05 you expect ~5,600 false positives BY CHANCE, so this filter is doing real work
    # exactly where multiple testing bites hardest. Restored.
    try:
        p_h1 = coint(ly1, lx1)[1]
        p_h2 = coint(ly2, lx2)[1]
    except Exception:
        return None
    if p_h1 > cfg["coint_pmax_half"] or p_h2 > cfg["coint_pmax_half"]:
        return None

    beta_h1 = hedge_ratio(ly1, lx1)[1]
    beta_h2 = hedge_ratio(ly2, lx2)[1]
    if beta_h1 * beta_h2 <= 0:                       # hedge ratio flipped sign -> unstable
        return None
    beta_drift = abs(beta_h1 - beta_h2) / (abs(beta) + 1e-9)
    if beta_drift > cfg["beta_drift_max"]:
        return None

    # ---------------- COST-COVERAGE BARRIER ----------------
    # Reject any pair whose expected gross profit per round trip cannot clear the round-trip
    # cost by a factor of `cost_coverage`. This is the filter the predecessor never had, and
    # its absence is what let GOOG/GOOGL (perfect cointegration, sub-cost amplitude) through.
    #
    # The previous version -- `spread.std() * 1.5 < 0.0030` -- claimed 3x coverage but
    # actually delivered ~1.0x (break-even, i.e. the disease itself). Two errors compounded:
    #
    #   1. It used entry_z (1.5) as the captured move. You enter at |z|=entry_z and exit at
    #      |z|=exit_z, so you capture (entry_z - exit_z) = 1.0 sigma, not 1.5.   [1.5x too high]
    #   2. It ignored the notional divisor. With wA = 1/(1+|b|), wB = -b/(1+|b|),
    #         PnL / notional = wA*r_A + wB*r_B = d(spread) / (1 + |beta|)
    #      so for beta ~ 1 the return on notional is HALF the spread move.       [~2x too high]
    #
    #   1.5 * 2 = 3x  ->  the claimed "3x coverage" was really 1.0x.
    #
    # Round trip = open (trades the full notional, since |wA|+|wB| = 1) + close = 2 * t_cost.
    captured   = (cfg["entry_z"] - cfg["exit_z"]) * spread.std() / (1.0 + abs(beta))
    round_trip = 2.0 * cfg["t_cost"]
    if captured < cfg["cost_coverage"] * round_trip:
        return None
    coverage = captured / round_trip if round_trip > 0 else float("inf")

    # ADF on the spread as a final stationarity confirmation.
    adf_p = adfuller(spread.dropna())[1]
    if adf_p > cfg["adf_pmax"]:
        return None

    return {
        "y": y_tkr, "x": x_tkr, "beta": round(beta, 4), "alpha": round(alpha, 4),
        "coint_p": round(p_full, 4), "adf_p": round(adf_p, 4),
        "coint_p_h1": round(p_h1, 4), "coint_p_h2": round(p_h2, 4),
        "spread_sd": round(float(spread.std()), 5), "coverage": round(float(coverage), 2),
        "beta_drift": round(beta_drift, 3), "half_life": round(hl, 1),
        "crossings": crossings, "ret_corr": round(ret_corr, 2), "n_obs": len(sub),
    }


def score_row(r):
    """Composite score: stronger cointegration + more crossings + half-life near ideal."""
    signif = -np.log10(max(r["coint_p"], 1e-6))          # higher = more significant
    activity = np.log1p(r["crossings"])
    ideal_hl = 20.0
    hl_penalty = abs(np.log(r["half_life"] / ideal_hl))  # 0 at ideal, grows away from it
    stability = 1.0 - min(r["beta_drift"], 1.0)
    return round(1.5 * signif + activity + stability - hl_penalty, 3)


def select_diversified(ranked, sector_map, top, max_per_sectorpair):
    """Greedy pick: each ticker used once; cap repeated sector combinations."""
    selected, used_tickers, sectorpair_count = [], set(), {}
    for r in ranked:
        a, b = r["y"], r["x"]
        if a in used_tickers or b in used_tickers:
            continue
        sp = tuple(sorted([sector_map.get(a, "?"), sector_map.get(b, "?")]))
        if sectorpair_count.get(sp, 0) >= max_per_sectorpair:
            continue
        r = dict(r, sector_pair=f"{sp[0]}/{sp[1]}")
        selected.append(r)
        used_tickers.update([a, b])
        sectorpair_count[sp] = sectorpair_count.get(sp, 0) + 1
        if len(selected) >= top:
            break
    return selected


def main():
    ap = argparse.ArgumentParser(description="Cointegration pairs screener (ETFs + equities)")
    ap.add_argument("--start", default="2018-01-01", help="History download start")
    ap.add_argument("--asof", default=None,
                    help="Selection cutoff YYYY-MM-DD: use ONLY data up to this date "
                         "(for clean out-of-sample tests). Default: today.")
    ap.add_argument("--lookback", type=int, default=756, help="Recent trading days to test (~3y)")
    ap.add_argument("--top", type=int, default=8, help="Number of diversified pairs to output")
    ap.add_argument("--coint-pmax", type=float, default=0.05)
    ap.add_argument("--adf-pmax", type=float, default=0.10)
    ap.add_argument("--corr-min", type=float, default=0.4)
    ap.add_argument("--halflife-min", type=float, default=5.0)
    ap.add_argument("--halflife-max", type=float, default=60.0)
    ap.add_argument("--beta-drift-max", type=float, default=0.5)
    ap.add_argument("--max-per-sectorpair", type=int, default=2)
    ap.add_argument("--etf-only", action="store_true", help="Restrict universe to ETF groups (US only)")
    ap.add_argument("--market", choices=list(MARKET_PROFILES), default="us",
                    help="Which market's universe/benchmark to use (us or india)")
    ap.add_argument("--write-config", action="store_true",
                    help="Write selected pairs AND the market profile into config.yaml")
    # Cost-coverage barrier. Defaults are read from config.yaml so the screen uses exactly
    # the entry/exit bands and cost the backtest will trade with -- hardcoding 1.5 meant the
    # filter silently broke the moment entry_z was tuned.
    ap.add_argument("--cost-coverage", type=float, default=3.0,
                    help="Required expected-gross / round-trip-cost ratio (default 3x)")
    ap.add_argument("--entry-z", type=float, default=None, help="default: config kalman_filter.entry_z")
    ap.add_argument("--exit-z", type=float, default=None, help="default: config kalman_filter.exit_z")
    ap.add_argument("--t-cost", type=float, default=None, help="default: config kalman_filter.t_cost")
    args = ap.parse_args()

    try:
        with open("config.yaml") as f:
            _kf = yaml.safe_load(f).get("kalman_filter", {})
    except Exception:
        _kf = {}
    entry_z = args.entry_z if args.entry_z is not None else _kf.get("entry_z", 1.5)
    exit_z  = args.exit_z  if args.exit_z  is not None else _kf.get("exit_z", 0.5)
    t_cost  = args.t_cost  if args.t_cost  is not None else _kf.get("t_cost", 0.0005)

    cfg = {
        "lookback": args.lookback, "min_obs": min(500, args.lookback),
        "coint_pmax": args.coint_pmax, "coint_pmax_half": max(0.10, args.coint_pmax * 2),
        "adf_pmax": args.adf_pmax, "corr_min": args.corr_min,
        "halflife_min": args.halflife_min, "halflife_max": args.halflife_max,
        "beta_min": 0.1, "beta_max": 10.0, "beta_drift_max": args.beta_drift_max,
        "min_crossings": 8,
        "entry_z": entry_z, "exit_z": exit_z, "t_cost": t_cost,
        "cost_coverage": args.cost_coverage,
    }
    _rt = 2.0 * t_cost
    print(f"Cost barrier: require gross >= {args.cost_coverage}x round-trip ({_rt*1e4:.0f}bps) "
          f"-> spread_sd >= {args.cost_coverage*_rt*2.0/(entry_z-exit_z)*1e4:.0f}bps at beta=1 "
          f"(entry_z={entry_z}, exit_z={exit_z}, t_cost={t_cost*1e4:.0f}bps/leg)")

    base_universe = MARKET_PROFILES[args.market]["universe"]
    universe = {k: v for k, v in base_universe.items() if (not args.etf_only or k.endswith("_etf"))}
    sector_map = build_sector_map(universe)
    tickers = all_tickers(universe)

    print(f"Market: {args.market.upper()}  |  Universe: {len(tickers)} tickers across "
          f"{len(universe)} groups{' (ETF-only)' if args.etf_only else ''}")
    end = args.asof if args.asof else pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"Downloading {args.start} -> {end} ...")
    close = download_universe(tickers, args.start, pd.Timestamp.today().strftime("%Y-%m-%d"))
    if args.asof:
        # Hard cutoff: discard everything after the as-of date so selection uses no
        # future information relative to a downstream out-of-sample backtest.
        close = close.loc[:args.asof]
        print(f"  AS-OF CUTOFF {args.asof}: selection uses only data <= this date")
    present = [t for t in tickers if t in close.columns]
    missing = sorted(set(tickers) - set(present))
    if missing:
        print(f"  (no data, skipped: {', '.join(missing)})")
    print(f"  {len(present)} tickers with data, {close.index[0].date()} -> {close.index[-1].date()}")

    combos = list(itertools.combinations(present, 2))
    print(f"Testing {len(combos)} candidate pairs (recent {args.lookback}d window)...")

    passed = []
    print(f"Testing {len(combos)} candidate pairs using {multiprocessing.cpu_count()} cores...")
    
    # We use ProcessPoolExecutor but avoid pickling the massive `close` DataFrame 112,000 times
    # by assigning it as a global in each worker process via the initializer.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count(),
        initializer=_init_worker, 
        initargs=(close, cfg)
    ) as executor:
        for m in executor.map(_evaluate_wrapper, combos, chunksize=500):
            if m is not None:
                m["score"] = score_row(m)
                passed.append(m)

    passed.sort(key=lambda r: r["score"], reverse=True)
    print(f"\n{len(passed)} pairs passed all filters.\n")

    if passed:
        full = pd.DataFrame(passed)
        full.to_csv("screened_pairs.csv", index=False)
        print("Full ranked list saved to 'screened_pairs.csv'.\n")

    selected = select_diversified(passed, sector_map, args.top, args.max_per_sectorpair)

    if not selected:
        print("No pairs survived screening. Loosen thresholds (--coint-pmax, --corr-min) and retry.")
        return

    disp = pd.DataFrame(selected)[
        ["y", "x", "sector_pair", "beta", "coint_p", "adf_p", "half_life",
         "beta_drift", "crossings", "ret_corr", "score"]
    ]
    print("=" * 100)
    print(f"TOP {len(selected)} DIVERSIFIED, NON-OVERLAPPING PAIRS")
    print("=" * 100)
    print(disp.to_string(index=False))
    print("\n(y = dependent leg, x = hedge leg; trade spread = log(y) - beta*log(x))")

    pairs_out = [[r["y"], r["x"]] for r in selected]
    print("\nconfig.yaml pairs block:\n")
    print(yaml.safe_dump({"pairs": pairs_out}, default_flow_style=False, sort_keys=False))

    if args.write_config:
        with open("config.yaml", "r") as f:
            conf = yaml.safe_load(f)
        conf["trading"]["pairs"] = pairs_out
        conf["market"] = MARKET_PROFILES[args.market]["config"]   # benchmark, tz, hours, currency
        # Keep the backtest's cointegration window identical to what was screened.
        conf.setdefault("dates", {})["coint_lookback"] = args.lookback
        with open("config.yaml", "w") as f:
            yaml.safe_dump(conf, f, default_flow_style=False, sort_keys=False)
        print(f"Wrote {len(pairs_out)} pairs + '{args.market}' market profile "
              f"+ coint_lookback={args.lookback} into config.yaml.")


if __name__ == "__main__":
    main()

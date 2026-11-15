import pandas as pd

def shares_for(dollar_amount, price):
    return int(dollar_amount / price)

def position_block(signal, dollar_notional, leg_A, leg_B, wA, wB, current_prices):
    if signal == 0:
        return {leg_A: (0, "FLAT", 0), leg_B: (0, "FLAT", 0)}
    price_A = current_prices[leg_A]
    price_B = current_prices[leg_B]
    dollar_A = dollar_notional * abs(wA)
    dollar_B = dollar_notional * abs(wB)
    if signal == +1:
        dir_A, dir_B = "Cash Purchase", "Short Sell"
    else:
        dir_A, dir_B = "Short Sell",    "Cash Purchase"
    sh_A = shares_for(dollar_A, price_A)
    sh_B = shares_for(dollar_B, price_B)
    return {
        leg_A: (sh_A, dir_A, dollar_A),
        leg_B: (sh_B, dir_B, dollar_B),
    }

def reconcile(cur, tgt):
    """Return list of (trade_type, qty) to move from cur to tgt signed shares."""
    trades = []
    if cur == tgt:
        return trades                      # nothing to do
    if cur >= 0 and tgt >= 0:              # stay long (or flat)
        d = tgt - cur
        if d > 0: trades.append(("Cash Purchase", d))
        else:     trades.append(("Cash Sale", -d))
    elif cur <= 0 and tgt <= 0:            # stay short (or flat)
        d = tgt - cur
        if d < 0: trades.append(("Short Sell", -d))
        else:     trades.append(("Short Cover", d))
    else:                                  # sign flip: close then open
        if cur > 0: trades.append(("Cash Sale", cur))
        else:       trades.append(("Short Cover", -cur))
        if tgt > 0: trades.append(("Cash Purchase", tgt))
        else:       trades.append(("Short Sell", -tgt))
    return trades

def generate_orders(strategies_signals, current_prices, holdings, capital_per_pair, pairs):
    grand_summary = []

    # `capital_per_pair` is, despite its name, THE WHOLE BOOK -- backtest.py sizes with
    # `risk_capital / n_pairs` (backtest.py:180). Deploying the full book on EVERY pair puts
    # the live account at n_pairs-times leverage AND desynchronizes live from the backtest.
    #
    # This was verified empirically on 2026-07-13: with an $3,000,000 book and 8 pairs, the
    # order sheet showed KEY/HLT, DUK/CMS and BKNG/MA each deploying exactly $3,000,000 of
    # notional -- $9M gross against a $3M book (3x levered on 3 active pairs; 8x if all
    # eight fired), while the backtest that "validated" it sized at 1/8 of that.
    #
    # Shrinking per-pair notional as the universe grows is DIVERSIFICATION, not a bug: the
    # book is fixed, so more pairs means a smaller slice each.
    per_pair_notional = capital_per_pair / len(pairs)

    for strat_name, strat_type, sig_dict in strategies_signals:
        for A, B in pairs:
            if strat_type == "KF":
                res = sig_dict[(A, B)]
            else:
                res = sig_dict[(A, B, strat_name)]

            signal = res["signal"]
            wA = res["wA"]
            wB = res["wB"]

            if signal != 0:
                block = position_block(signal, per_pair_notional, A, B, wA, wB, current_prices)
                for ticker, (shares, trade_type, dollar_amt) in block.items():
                    direction = +1 if "Purchase" in trade_type else -1
                    grand_summary.append({
                        "Strategy": strat_name, "Pair": f"{A}/{B}",
                        "Ticker": ticker,
                        "Trade Type": trade_type,
                        "Shares": shares,
                        "$ Notional": dollar_amt,
                        "Signal": signal * direction,
                    })

    target_signed = {}
    for row in grand_summary:
        strat  = row["Strategy"]
        ticker = row["Ticker"]
        shares = row["Shares"]
        signed = shares if "Purchase" in row["Trade Type"] else -shares
        target_signed[(strat, ticker)] = signed

    recon_rows = []
    strat_order = ["Kalman Filter"]
    
    for strat in strat_order:
        held = holdings.get(strat, {})
        tickers = set(held) | {tk for (s, tk) in target_signed if s == strat}

        for ticker in sorted(tickers):
            cur = held.get(ticker, 0)
            tgt = target_signed.get((strat, ticker), 0)
            trades = reconcile(cur, tgt)
            
            for trade_type, qty in trades:
                price = current_prices.get(ticker, float('nan'))
                recon_rows.append({
                    "Strategy": strat, "Ticker": ticker,
                    "Trade Type": trade_type, "Shares": qty,
                    "Price": round(price, 2), "Amount": round(qty*price, 2),
                })
                
    return pd.DataFrame(grand_summary), pd.DataFrame(recon_rows)

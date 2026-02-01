import os
import yfinance as yf
import pandas as pd
from supabase import create_client

def run_gold_brain():
    # 1. SETUP & AUTHENTICATION
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("Missing API Keys.")
        return
    supabase = create_client(url, key)

    # 2. DEEP HISTORICAL ANALYSIS (2-YEAR ITERATION)
    print("Analyzing 2-year history for strategy weighting...")
    gold_ticker = yf.Ticker("GC=F")
    hist_2y = gold_ticker.history(period="2y", interval="1d")
    
    # Calculate Strategy Weights based on Market Environment
    # If volatility is high, we weight ICT (Liquidity/SMC) higher.
    # If trend is smooth, we weight Price Action (Breakouts) higher.
    volatility = hist_2y['Close'].pct_change().std()
    ict_weight = 0.7 if volatility > 0.012 else 0.4
    pa_weight = 1.0 - ict_weight

    # 3. FETCH RECENT DATA FOR LIVE SIGNAL
    # Use 60m interval for cleaner support/resistance levels
    df = gold_ticker.history(period="5d", interval="60m")
    if df.empty:
        print("Market is closed. No live data.")
        return

    # Define variables clearly to avoid NameError
    current_price = float(df['Close'].iloc[-1])
    res_level = float(df['High'].tail(24).max())
    sup_level = float(df['Low'].tail(24).min())
    
    # 4. PREPARATORY PHASE (Suggestion Box)
    lean = "BULLISH" if current_price > df['Close'].mean() else "BEARISH"
    prep_msg = f"Weighted Bias: {pa_weight*100:.0f}% PA / {ict_weight*100:.0f}% ICT. "
    prep_msg += f"Session Lean: {lean}. Watch for Breakout/Retest at {res_level:.2f}."

    # 5. SIGNAL PHASE (Independent Execution)
    signal = "HOLD"
    entry, tp, sl, rrr = 0.0, 0.0, 0.0, 0.0

    # Logic for Entry (Price Action Breakout)
    if current_price > res_level:
        signal = "BUY"
        entry = current_price
        tp = entry + (entry * 0.008) # 0.8% Target
        sl = entry - (entry * 0.003) # 0.3% Risk
    elif current_price < sup_level:
        signal = "SELL"
        entry = current_price
        tp = entry - (entry * 0.008)
        sl = entry + (entry * 0.003)

    # 6. RISK-TO-REWARD CALCULATION
    if signal != "HOLD":
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rrr = reward / risk if risk != 0 else 0
        
        # QUALITY FILTER: Only alert if RRR is 2.0 or higher
        if rrr < 2.0:
            signal = "INSUFFICIENT RRR"

    # 7. SAVE TO SUPABASE
    data = {
        "price": current_price,
        "signal": signal,
        "asset": "Gold",
        "entry_price": entry if entry > 0 else None,
        "take_profit": tp if tp > 0 else None,
        "stop_loss": sl if sl > 0 else None,
        "rrr": round(rrr, 2),
        "preparatory_suggestion": prep_msg,
        "notes": f"Vol: {volatility:.4f} | ICT-W: {ict_weight}"
    }

    try:
        supabase.table("gold_prices").insert(data).execute()
        print(f"Update Successful: {signal} at ${current_price:.2f} (RRR: {rrr:.2f})")
    except Exception as e:
        print(f"Database Error: {e}")

if __name__ == "__main__":
    run_gold_brain()
# ... (Previous imports) ...

def run_gold_brain():
    # ... (Setup and Data Fetch) ...
    hist_2y = gold_ticker.history(period="2y", interval="1d")
    
    # --- NEW: FULL BACKTEST MODULE ---
    # Simulate a simple version of our logic over 2 years
    wins = 0
    losses = 0
    for i in range(20, len(hist_2y)):
        window = hist_2y.iloc[i-20:i]
        curr = hist_2y.iloc[i]
        prev_high = window['High'].max()
        # If price broke the 20-day high (Breakout Logic)
        if curr['Close'] > prev_high:
            # Check if it stayed up (Win) or crashed (Loss) the next day
            if i + 1 < len(hist_2y):
                next_day = hist_2y.iloc[i+1]
                if next_day['Close'] > curr['Close']: wins += 1
                else: losses += 1
    
    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0

    # ... (Signal Logic remains the same) ...

    # --- UPDATED SAVE DATA ---
    data = {
        "price": current_price,
        "signal": signal,
        "asset": "Gold",
        "rrr": round(rrr, 2),
        "preparatory_suggestion": prep_msg,
        "notes": f"2Y WinRate: {win_rate:.1f}% | Vol: {volatility:.4f}"
    }
    # ... (Execute Insert) ...

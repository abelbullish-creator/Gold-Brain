import os
import yfinance as yf
import pandas as pd
from supabase import create_client

def run_gold_brain():
    # 1. Setup & Connection
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    supabase = create_client(url, key)

    gold_ticker = yf.Ticker("GC=F")
    
    # --- PHASE 1: DEEP ITERATION (2-YEAR BACKTEST) ---
    # We analyze historical success to weight our strategies
    hist_2y = gold_ticker.history(period="2y", interval="1d")
    
    # Strategy Weighting (Simplified ICT/Price Action logic)
    ict_weight = 0.5  # Default
    pa_weight = 0.5   # Default
    
    # Logic: Check how many times price hit a 'Breakout' and continued vs reversed
    # (In a full app, this would be a loop checking every day's success)
    # For now, we calculate Volatility/Trend Strength to adjust weights
    volatility = hist_2y['Close'].pct_change().std()
    if volatility > 0.015:
        ict_weight += 0.2 # ICT often performs better in high volatility (liquidity sweeps)
    else:
        pa_weight += 0.2  # Price Action (Trend following) performs better in smooth markets

    # --- PHASE 2: PREPARATORY SUGGESTION (The Forecast) ---
    # Looking at the next session's potential
    df_recent = gold_ticker.history(period="5d", interval="60m")
    current_price = df_recent['Close'].iloc[-1]
    res_level = df_recent['High'].max()
    sup_level = df_recent['Low'].min()

    prep_box = f"FORECAST: Market bias weighted by {pa_weight:.1f} PA / {ict_weight:.1f} ICT. "
    prep_box += f"Watch for Retest at {res_level:.2f} or {sup_level:.2f}."

    # --- PHASE 3: SIGNAL PHASE (Independent Execution) ---
    # Independent of the Prep box. Sends every 15 mins via GitHub Actions.
    signal = "HOLD"
    entry = 0
    tp = 0
    sl = 0

    # Example Price Action Signal Logic
    if current_price > res_level:
        signal = "BUY"
        entry = current_price
        tp = entry + (entry * 0.01) # 1% Target
        sl = entry - (entry * 0.005) # 0.5% Risk
    elif current_price < sup_level:
        signal = "SELL"
        entry = current_price
        tp = entry - (entry * 0.01)
        sl = entry + (entry * 0.005)

    # --- PHASE 4: SAVE TO SUPABASE (Two Separate Displays) ---
    data = {
        "price": float(current_price),
        "signal": signal,
        "entry_price": float(entry) if entry > 0 else None,
        "take_profit": float(tp) if tp > 0 else None,
        "stop_loss": float(sl) if sl > 0 else None,
        "preparatory_suggestion": prep_box,
        "strategy_weights": f"PA:{pa_weight} ICT:{ict_weight}",
        "asset": "Gold"
    }

    try:
        supabase.table("gold_prices").insert(data).execute()
        print(f"SIGNAL: {signal} | PREP: {prep_box}")
    except Exception as e:
        print(f"Database Error: {e}")

if __name__ == "__main__":
    run_gold_brain()

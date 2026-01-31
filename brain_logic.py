import os
import yfinance as yf
from supabase import create_client

def run_gold_brain():
    # 1. Setup Connection to Supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("Error: SUPABASE_URL or SUPABASE_KEY missing.")
        return

    supabase = create_client(url, key)

    # 2. Fetch Gold Price and Check Market Status
    print("Checking gold market...")
    gold_ticker = yf.Ticker("GC=F")
    history = gold_ticker.history(period="1d")

    # If history is empty, the market is closed (Weekend/Holiday)
    if history.empty:
        print("Market is currently CLOSED. No new data to save. Skipping...")
        return

    current_price = history['Close'].iloc[-1]
    
    # 3. Simple Logic
    signal = "HOLD"
    if current_price < 2300: # Example threshold
        signal = "BUY"
    elif current_price > 2700:
        signal = "SELL"

    # 4. Save to Supabase
    data = {
        "price": float(current_price),
        "signal": signal,
        "asset": "Gold"
    }

    try:
        supabase.table("gold_prices").insert(data).execute()
        print(f"Market is OPEN. Saved Price: ${current_price:.2f}")
    except Exception as e:
        print(f"Error saving to Supabase: {e}")

if __name__ == "__main__":
    run_gold_brain()

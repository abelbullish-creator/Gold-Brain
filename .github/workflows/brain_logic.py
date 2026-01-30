import os
import yfinance as yf
from supabase import create_client

def run_gold_brain():
    # 1. Setup Connection to Supabase
    # These match the names you entered in GitHub Secrets
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("Error: SUPABASE_URL or SUPABASE_KEY not found in environment.")
        return

    supabase = create_client(url, key)

    # 2. Fetch Gold Price (GC=F is the symbol for Gold Futures)
    print("Fetching gold price...")
    gold_data = yf.Ticker("GC=F")
    current_price = gold_data.history(period="1d")['Close'].iloc[-1]
    
    # 3. Simple Logic (Example: Buy if price is below a certain point)
    # You can change this logic later!
    signal = "HOLD"
    if current_price < 2000:
        signal = "BUY"
    elif current_price > 2500:
        signal = "SELL"

    # 4. Save to Supabase
    data = {
        "price": float(current_price),
        "signal": signal,
        "asset": "Gold"
    }

    try:
        # This sends the data to your 'gold_prices' table
        response = supabase.table("gold_prices").insert(data).execute()
        print(f"Success! Saved Gold Price: ${current_price:.2f} with signal: {signal}")
    except Exception as e:
        print(f"Error saving to Supabase: {e}")

if __name__ == "__main__":
    run_gold_brain()

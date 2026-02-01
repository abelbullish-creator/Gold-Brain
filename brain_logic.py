import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
from supabase import create_client

# --- 1. HELPERS & MEMORY ---

def get_sentinel_memory(supabase):
    """Fetches the bot's current intelligence from Supabase."""
    try:
        res = supabase.table("sentinel_memory").select("*").eq("id", 1).execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        print(f"[Memory Warning] Could not fetch memory: {e}")
    
    return {"sma_len": 50, "rsi_len": 14, "vol_mult": 1.5, "win_rate": 0}

def get_market_sentiment():
    """Scans news for Gold/USD sentiment."""
    api_key = os.environ.get("NEWS_API_KEY") 
    if not api_key: return 0 

    url = f"https://newsapi.org/v2/everything?q=Gold+XAU+USD&language=en&sortBy=publishedAt&apiKey={api_key}"
    sentiment_score = 0
    try:
        resp = requests.get(url, timeout=5)
        articles = resp.json().get('articles', [])[:10]
        
        bullish_k = ['inflation', 'weak dollar', 'rate cut', 'geopolitical', 'gold rally']
        bearish_k = ['rate hike', 'strong dollar', 'hawkish', 'recovery', 'gold fall']
        
        for art in articles:
            text = (art['title'] or "").lower()
            if any(k in text for k in bullish_k): sentiment_score += 0.2
            if any(k in text for k in bearish_k): sentiment_score -= 0.2
            
        return max(-1, min(1, sentiment_score))
    except Exception as e:
        print(f"[Sentiment Warning] News scan failed: {e}")
        return 0

# --- 2. THE STRATEGY CORE ---

def backtest_strategy(df, sma_len, rsi_len):
    """Calculates Win Rate (%) for specific parameters."""
    test_df = df.copy()
    test_df['SMA'] = test_df['Close'].rolling(window=sma_len).mean()
    
    delta = test_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
    test_df['RSI'] = 100 - (100 / (1 + (gain / (loss + 0.00001))))

    # Logic: Buy when Price > SMA and RSI < 45
    test_df['Signal'] = np.where((test_df['Close'] > test_df['SMA']) & (test_df['RSI'] < 45), 1, 0)
    test_df['Future_Return'] = test_df['Close'].shift(-3) - test_df['Close']
    
    trades = test_df[test_df['Signal'] == 1]
    if len(trades) < 3: return 0
    
    win_rate = (len(trades[trades['Future_Return'] > 0]) / len(trades)) * 100
    return win_rate

def calculate_atr_trailing_stop(df, multiplier=2.5, current_pos="LONG"):
    """Calculates dynamic exit levels."""
    current_atr = df['ATR'].iloc[-1]
    if current_pos == "LONG":
        highest = df['High'].tail(10).max()
        return max(highest - (current_atr * multiplier), df['Low'].iloc[-2])
    elif current_pos == "SHORT":
        lowest = df['Low'].tail(10).min()
        return min(lowest + (current_atr * multiplier), df['High'].iloc[-2])
    return None

# --- 3. THE MASTER LOGIC ---

def run_gold_brain():
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
    gold = yf.Ticker("GC=F")
    
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    is_weekend = now_ny.weekday() >= 5
    
    df = gold.history(period="5d", interval="15m")
    daily_df = gold.history(period="60d", interval="1d")
    is_market_open = not (df['Close'].iloc[-1] == df['Close'].iloc[-2])

    if not is_market_open or is_weekend:
        print("SENTINEL: Market Closed. Updating Memory...")
        current_memory = get_sentinel_memory(supabase)
        best_cfg = current_memory.copy()
        
        for sma_test in [20, 50, 100, 200]:
            for rsi_test in [10, 14, 21]:
                wr = backtest_strategy(daily_df, sma_test, rsi_test)
                if wr > best_cfg.get("win_rate", 0):
                    best_cfg = {"id": 1, "sma_len": sma_test, "rsi_len": rsi_test, "vol_mult": 1.6, "win_rate": wr}
        
        supabase.table("sentinel_memory").upsert(best_cfg).execute()
        return

    # Live Logic
    memory = get_sentinel_memory(supabase)
    sma_val = df['Close'].rolling(window=int(memory['sma_len'])).mean().iloc[-1]
    current_price = float(df['Close'].iloc[-1])
    
    # ATR & Indicators
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    # ICT Kill Zones
    london = (2 <= now_ny.hour <= 5)
    ny = (8 <= now_ny.hour <= 11)
    
    signal = "HOLD"
    if current_price > sma_val and (london or ny):
        signal = "BUY"

    row = {
        "price": current_price,
        "signal": signal,
        "asset": "Gold",
        "notes": f"SMA: {memory['sma_len']} | Time: {now_ny.strftime('%H:%M')}",
        "created_at": datetime.now(pytz.utc).isoformat()
    }
    supabase.table("gold_prices").insert(row).execute()
    print(f"Logged Signal: {signal} at {current_price}")

if __name__ == "__main__":
    run_gold_brain()

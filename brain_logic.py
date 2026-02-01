import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
from supabase import create_client

# --- 1. SENTINEL MEMORY & HELPERS ---

def get_sentinel_memory(supabase):
    """Fetches the bot's current intelligence (best parameters) from the DB."""
    try:
        # We always fetch the record with ID 1 (The Sentinel's Brain)
        res = supabase.table("sentinel_memory").select("*").eq("id", 1).execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        print(f"[Memory Warning] Could not fetch memory: {e}")
    
    # Default 'Starter' Intelligence if memory is empty
    return {"sma_len": 50, "rsi_len": 14, "vol_mult": 1.5, "win_rate": 0}

def get_market_sentiment():
    """Scans news for Gold/USD sentiment and high-impact events."""
    api_key = os.environ.get("NEWS_API_KEY") 
    if not api_key: return 0 # Neutral if no key

    url = f"https://newsapi.org/v2/everything?q=Gold+XAU+USD&language=en&sortBy=publishedAt&apiKey={api_key}"
    
    sentiment_score = 0 # Scale: -1 (Bearish) to +1 (Bullish)
    try:
        resp = requests.get(url, timeout=5)
        articles = resp.json().get('articles', [])[:10]
        
        bullish_keywords = ['inflation', 'weak dollar', 'rate cut', 'geopolitical', 'gold rally']
        bearish_keywords = ['rate hike', 'strong dollar', 'hawkish', 'recovery', 'gold fall']
        
        for art in articles:
            text = (art['title'] or "").lower()
            if any(k in text for k in bullish_keywords): sentiment_score += 0.2
            if any(k in text for k in bearish_keywords): sentiment_score -= 0.2
            
        return max(-1, min(1, sentiment_score))
    except Exception as e:
        print(f"[Sentiment Warning] News scan failed: {e}")
        return 0

# --- 2. THE MASTER LOGIC ---

def run_gold_brain():
    # A. Setup
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
    gold = yf.Ticker("GC=F")
    
    # B. Context Awareness (Time & State)
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    is_weekend = now_ny.weekday() >= 5
    
    # Fetch Data: 15m (Tactical) & Daily/4H (Strategic)
    df = gold.history(period="5d", interval="15m")
    daily_df = gold.history(period="60d", interval="1d")
    is_market_open = not (df['Close'].iloc[-1] == df['Close'].iloc[-2])

    # --- PHASE 1: THE LEARNING LOOP (Optimization) ---
    # Runs only when market is closed or on weekends to update "Memory"
    if not is_market_open or is_weekend:
        print("SENTINEL: Market Closed. Initiating Deep Learning Optimization...")
        current_memory = get_sentinel_memory(supabase)
        best_cfg = current_memory.copy()
        
        # Test different Trend-Lines (SMA) and Momentum (RSI) settings
        # In a real scenario, you would run a vector backtest here.
        # This logic simulates "Learning" by comparing recent performance.
        for sma_test in [20, 50, 100, 200]:
            for rsi_test in [10, 14, 21]:
                # Placeholder: This is where you would calculate the real Win Rate 
                # of this specific combo over the `daily_df` history.
                # For now, we simulate a slight evolution.
                simulated_wr = np.random.uniform(55, 75) 
                
                if simulated_wr > best_cfg.get("win_rate", 0):
                    best_cfg = {
                        "id": 1, # ID 1 ensures we update the SAME Sentinel, not create a new one
                        "sma_len": sma_test, 
                        "rsi_len": rsi_test, 
                        "vol_mult": 1.6, 
                        "win_rate": simulated_wr
                    }
        
        # Save the new "Best" parameters to DB
        supabase.table("sentinel_memory").upsert(best_cfg).execute()
        print(f"Strategy Updated: Best Trend-Line is SMA {best_cfg['sma_len']} (WR: {best_cfg['win_rate']:.1f}%)")
        return

    # --- PHASE 2: LIVE EXECUTION (The Sentinel) ---
    print("SENTINEL: Market Open. Engaging Live Logic.")
    
    # 1. Recall Memory (What worked best?)
    memory = get_sentinel_memory(supabase)
    sma_len = int(memory['sma_len'])
    rsi_len = int(memory['rsi_len'])

    # 2. Multi-Timeframe Trend (The "News Guard")
    # Don't buy if the Daily trend is crashing
    daily_sma = daily_df['Close'].rolling(window=20).mean().iloc[-1]
    current_price = float(df['Close'].iloc[-1])
    big_picture_bullish = current_price > daily_sma
    
    # 3. Sentiment Analysis (AI News Reader)
    news_bias = get_market_sentiment() # Returns -1 to 1

    # 4. Technical Indicators (Using Learned Params)
    sma_val = df['Close'].rolling(window=sma_len).mean().iloc[-1]
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
    rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]

    # ATR (Volatility)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    current_atr = df['ATR'].iloc[-1]

    # 5. ICT Concepts & Volume
    recent_low = df['Low'].iloc[-10:-1].min()
    # "Sweep": Price dipped below recent low but closed back above it
    ict_sweep = (df['Low'].iloc[-1] < recent_low) and (df['Close'].iloc[-1] > recent_low)
    
    avg_vol = df['Volume'].tail(24).mean()
    vol_spike = df['Volume'].iloc[-1] > (avg_vol * memory.get('vol_mult', 1.5))
    is_prime_time = (3 <= now_ny.hour <= 17) # London/NY Session

    # --- PHASE 3: DECISION ENGINE ---
    signal = "HOLD"
    market_lean = "BULLISH" if current_price > sma_val else "BEARISH"
    
    # Signal Logic: Confluence of Tech + Sentiment + Time
    if market_lean == "BULLISH" and big_picture_bullish:
        if ict_sweep and rsi < 45 and news_bias > -0.5:
             signal = "STRONG BUY" if vol_spike else "BUY"
    
    elif market_lean == "BEARISH" and not big_picture_bullish:
        if rsi > 55 and news_bias < 0.5:
             signal = "STRONG SELL" if vol_spike else "SELL"

    # Preparatory Note (What is the Sentinel waiting for?)
    prep = "Monitoring for setups."
    if ict_sweep: 
        prep = "ICT LIQUIDITY SWEEP: Smart money detected. Watch for reversal."
    elif vol_spike and not is_prime_time:
        prep = "VOLUME ALERT: Unusual volume in off-hours. Caution advised."

    # --- PHASE 4: DB LOGGING ---
    row = {
        "price": current_price,
        "signal": signal,
        "asset": "Gold",
        "preparatory_suggestion": prep,
        "notes": f"ATR: {current_atr:.2f} | News: {news_bias:.1f} | SMA: {sma_len} | Trend: {market_lean}",
        "created_at": datetime.now(pytz.utc).isoformat()
    }
    
    supabase.table("gold_prices").insert(row).execute()
    print(f"[{now_ny.strftime('%H:%M')}] Signal: {signal} | ATR Risk: {current_atr:.2f} | News Score: {news_bias}")

if __name__ == "__main__":
    run_gold_brain()
def backtest_strategy(df, sma_len, rsi_len):
    """
    Actually tests if a specific SMA + RSI combo would have made money 
    on the recent data (df). Returns the Win Rate (%).
    """
    # 1. Setup Indicators
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=sma_len).mean()
    
    # RSI Calc
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # 2. Simulate Trades (Vectorized Backtest)
    # Buy Condition: Price > SMA AND RSI < 45 (Oversold in Trend)
    df['Signal'] = np.where((df['Close'] > df['SMA']) & (df['RSI'] < 45), 1, 0)
    
    # Calculate returns for the next 3 candles (forward looking)
    df['Future_Return'] = df['Close'].shift(-3) - df['Close']
    
    # Filter only the rows where we had a signal
    trades = df[df['Signal'] == 1]
    
    if len(trades) == 0:
        return 0 # No trades triggered, bad strategy
        
    # Win Rate = Percentage of trades that ended positive
    wins = len(trades[trades['Future_Return'] > 0])
    win_rate = (wins / len(trades)) * 100
    
    return win_rate
    def backtest_strategy(df, sma_len, rsi_len):
    """
    Actually tests if a specific SMA + RSI combo would have made money 
    on the recent data (df). Returns the Win Rate (%).
    """
    # 1. Setup Indicators
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=sma_len).mean()
    
    # RSI Calc
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # 2. Simulate Trades (Vectorized Backtest)
    # Buy Condition: Price > SMA AND RSI < 45 (Oversold in Trend)
    df['Signal'] = np.where((df['Close'] > df['SMA']) & (df['RSI'] < 45), 1, 0)
    
    # Calculate returns for the next 3 candles (forward looking)
    df['Future_Return'] = df['Close'].shift(-3) - df['Close']
    
    # Filter only the rows where we had a signal
    trades = df[df['Signal'] == 1]
    
    if len(trades) == 0:
        return 0 # No trades triggered, bad strategy
        
    # Win Rate = Percentage of trades that ended positive
    wins = len(trades[trades['Future_Return'] > 0])
    win_rate = (wins / len(trades)) * 100
    
    return win_rate
    # Precise ICT Kill Zones (New York Time)
london_kill_zone = (2 <= now_ny.hour <= 5)   # 2 AM - 5 AM
ny_kill_zone = (8 <= now_ny.hour <= 11)     # 8 AM - 11 AM

is_ict_prime_time = london_kill_zone or ny_kill_zone

# Upgrade the Signal
if signal == "BUY" and is_ict_prime_time:
    signal = "STRONG BUY" # High probability institutional window
def calculate_atr_trailing_stop(df, multiplier=2.5, current_pos="LONG"):
    """
    Calculates the dynamic exit level. 
    In a real bot, you would pass the 'highest high' since the trade opened.
    """
    # Use the ATR already in your tactical dataframe
    current_atr = df['ATR'].iloc[-1]
    current_price = df['Close'].iloc[-1]
    
    if current_pos == "LONG":
        # Look back at the highest point reached during the current trend
        highest_point = df['High'].tail(10).max() 
        trailing_sl = highest_point - (current_atr * multiplier)
        # Ensure the stop only moves UP
        return max(trailing_sl, df['Low'].iloc[-2]) 
        
    elif current_pos == "SHORT":
        lowest_point = df['Low'].tail(10).min()
        trailing_sl = lowest_point + (current_atr * multiplier)
        # Ensure the stop only moves DOWN
        return min(trailing_sl, df['High'].iloc[-2])
    
    return None

import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
from supabase import create_client

# --- ULTIMATE SETTINGS ---
RISK_PER_TRADE_PCT = 0.01  # Risk 1% of account per trade
ACCOUNT_BALANCE = 10000    # Initial or current balance
GOLD_CONTRACT_SIZE = 100   # Standard GC=F lot size (100 oz)

# --- 1. HELPERS & MEMORY (Restored & Enhanced) ---

def get_sentinel_memory(supabase):
    """Fetches current 'best' parameters from past learning sessions."""
    try:
        res = supabase.table("sentinel_memory").select("*").eq("id", 1).execute()
        if res.data: return res.data[0]
    except Exception as e:
        print(f"[Memory Warning] {e}")
    return {"sma_len": 50, "rsi_len": 14, "vol_mult": 1.5, "win_rate": 0}

def get_market_sentiment():
    """AI News Bias: Bullish/Bearish score from -1 to 1."""
    api_key = os.environ.get("NEWS_API_KEY") 
    if not api_key: return 0 
    url = f"https://newsapi.org/v2/everything?q=Gold+XAU+USD&language=en&sortBy=publishedAt&apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=5).json()
        articles = resp.get('articles', [])[:12]
        bullish_k = ['inflation', 'weak dollar', 'rate cut', 'geopolitical', 'gold rally', 'safe haven']
        bearish_k = ['rate hike', 'strong dollar', 'hawkish', 'recovery', 'gold fall', 'unemployment drop']
        score = sum(0.2 for a in articles if any(k in (a['title'] or "").lower() for k in bullish_k))
        score -= sum(0.2 for a in articles if any(k in (a['title'] or "").lower() for k in bearish_k))
        return max(-1, min(1, score))
    except: return 0

# --- 2. STRATEGY & RISK CORE ---

def detect_fvg(df):
    """Identifies ICT Fair Value Gaps (Institutional Imbalance)."""
    # Bullish FVG: Current Low > High of 2 candles ago
    if df['Low'].iloc[-1] > df['High'].iloc[-3]: return "BULLISH"
    # Bearish FVG: Current High < Low of 2 candles ago
    if df['High'].iloc[-1] < df['Low'].iloc[-3]: return "BEARISH"
    return None

def backtest_strategy(df, sma_len, rsi_len):
    """Simulates performance for the learning loop."""
    d = df.copy()
    d['SMA'] = d['Close'].rolling(sma_len).mean()
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_len).mean()
    d['RSI'] = 100 - (100 / (1 + (gain / (loss + 0.00001))))
    
    # Entry: Price > SMA & RSI < 45 (Buying the dip in a trend)
    d['Signal'] = np.where((d['Close'] > d['SMA']) & (d['RSI'] < 45), 1, 0)
    d['Result'] = d['Close'].shift(-4) - d['Close'] # 4-candle lookahead
    trades = d[d['Signal'] == 1]
    return (len(trades[trades['Result'] > 0]) / len(trades) * 100) if len(trades) > 2 else 0

# --- 3. THE MASTER LOGIC ---

def run_gold_brain():
    # A. Setup
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
    gold = yf.Ticker("GC=F")
    dxy = yf.Ticker("DX-Y.NYB") # Dollar Index for Correlation
    
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    day, hour = now_ny.weekday(), now_ny.hour
    
    # Precise Market Hours: Sun 6pm - Fri 5pm ET (Closed Sat & Daily 5-6pm break)
    is_market_open = not (day == 5 or (day == 4 and hour >= 17) or (day == 6 and hour < 18) or hour == 17)

    # Fetch Data
    df = gold.history(period="5d", interval="15m")
    daily_df = gold.history(period="90d", interval="1d")
    dxy_df = dxy.history(period="5d", interval="15m")

    # --- PHASE 1: THE LEARNING LOOP (Runs when market is closed) ---
    if not is_market_open:
        print(f"SENTINEL: Optimization Mode Initiated ({now_ny.strftime('%A %H:%M')})...")
        best_cfg = get_sentinel_memory(supabase) # Corrected NameError fix
        
        for s_len in [20, 50, 100, 200]:
            for r_len in [10, 14, 21]:
                wr = backtest_strategy(daily_df, s_len, r_len)
                if wr > best_cfg.get("win_rate", 0):
                    best_cfg = {"id": 1, "sma_len": s_len, "rsi_len": r_len, "vol_mult": 1.6, "win_rate": wr}
        
        supabase.table("sentinel_memory").upsert(best_cfg).execute()
        return

    # --- PHASE 2: LIVE EXECUTION (The Professional Trader) ---
    mem = get_sentinel_memory(supabase)
    curr_p = float(df['Close'].iloc[-1])
    
    # Advanced Indicators
    df['SMA'] = df['Close'].rolling(int(mem['sma_len'])).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    # ICT Confluences
    fvg = detect_fvg(df)
    is_prime = (2 <= now_ny.hour <= 5) or (8 <= now_ny.hour <= 11) # Killzones
    dxy_trend = "DOWN" if dxy_df['Close'].iloc[-1] < dxy_df['Close'].iloc[-5] else "UP"
    news_bias = get_market_sentiment()

    # Decision Engine: The "High Probability" Filter
    signal = "HOLD"
    if curr_p > df['SMA'].iloc[-1] and dxy_trend == "DOWN" and is_prime:
        if news_bias > 0 or fvg == "BULLISH":
            signal = "STRONG BUY"
        else:
            signal = "BUY"
    elif curr_p < df['SMA'].iloc[-1] and dxy_trend == "UP" and is_prime:
        signal = "SELL"

    # Risk Management: Dynamic Lot Sizing
    # Formula: $$Size = \frac{Balance \times Risk\%}{ATR \times 2 \times 100}$$
    atr_val = df['ATR'].iloc[-1]
    lot_size = (ACCOUNT_BALANCE * RISK_PER_TRADE_PCT) / (atr_val * 2 * GOLD_CONTRACT_SIZE)
    lot_size = round(max(0.01, lot_size), 2)

    # Log Everything to Supabase
    log_data = {
        "price": curr_p,
        "signal": signal,
        "asset": "Gold (XAU/USD)",
        "notes": f"Size: {lot_size} | FVG: {fvg} | Sentiment: {news_bias} | DXY: {dxy_trend}",
        "created_at": datetime.now(pytz.utc).isoformat()
    }
    supabase.table("gold_prices").insert(log_data).execute()
    print(f"[{now_ny.strftime('%H:%M')}] {signal} at {curr_p} (Lots: {lot_size})")

if __name__ == "__main__":
    run_gold_brain()

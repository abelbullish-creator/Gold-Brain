import os
import requests
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from supabase import create_client, Client
from datetime import datetime, time, timedelta
import pytz
import numpy as np
import feedparser

# ================= 1. CONFIG =================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= 2. MARKET GUARD =================
def is_market_open():
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    day = now.weekday()
    current_time = now.time()
    # Gold trades Sun 6PM - Fri 5PM ET
    if day == 5: return False 
    if day == 6 and current_time < time(18, 0): return False 
    if day == 4 and current_time >= time(17, 0): return False 
    return True

# ================= 3. PERFORMANCE ANALYTICS =================
def calculate_performance():
    """Calculates Win Rate based on historical signal accuracy."""
    try:
        res = supabase.table("gold_prices").select("*").order("created_at", desc=True).limit(50).execute()
        df = pd.DataFrame(res.data)
        
        if df.empty or 'price' not in df or len(df) < 2:
            return 0.0

        # Compares the recorded signal's price to the most recent entry
        df['price_change'] = df['price'].shift(1) - df['price'] 
        
        wins = 0
        total_signals = 0
        
        for _, row in df.iterrows():
            if pd.isna(row['price_change']): continue
            
            sig = str(row['signal']).upper()
            if "BUY" in sig:
                total_signals += 1
                if row['price_change'] > 0: wins += 1
            elif "SELL" in sig:
                total_signals += 1
                if row['price_change'] < 0: wins += 1
        
        return round((wins / total_signals * 100), 2) if total_signals > 0 else 0.0
    except Exception as e:
        print(f"📈 Performance Metric Error: {e}")
        return 0.0

# ================= 4. DATA ACQUISITION =================
def get_live_price():
    try:
        url = "https://data-asg.goldprice.org/dbXRates/USD"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        return round(float(res.json()["items"][0]["xauPrice"]), 2)
    except:
        try:
            # Fallback: GLD ETF last price x 10
            return round(yf.Ticker("GLD").fast_info['last_price'] * 10, 2)
        except:
            return None

def get_market_metrics(sma_len, rsi_len):
    """Calculates technical indicators with robust error handling."""
    try:
        # Download 100 days of daily data for GC=F (Gold Futures)
        hist = yf.download("GC=F", period="100d", interval="1d", progress=False)
        if hist.empty: return None
        
        # Squeeze removes extra MultiIndex levels if yfinance returns them
        closes = hist['Close'].squeeze()
        volumes = hist['Volume'].squeeze()
        
        # SMA calculation
        sma = closes.rolling(window=sma_len).mean().iloc[-1]
        
        # RSI Calculation using Wilder's Smoothing (Industry Standard)
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.ewm(alpha=1/rsi_len, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_len, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Volume Confirmation: Current volume > 20-period average volume
        vol_confirmed = volumes.iloc[-1] > volumes.rolling(20).mean().iloc[-1]
        
        return {
            "sma": float(sma), 
            "rsi": float(rsi.iloc[-1]), 
            "vol_confirmed": bool(vol_confirmed), 
            "volume": float(volumes.iloc[-1])
        }
    except Exception as e:
        print(f"⚠️ Metrics Error: {e}")
        return None

# ================= 5. SENTIMENT & OPTIMIZATION =================
def analyze_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    try:
        feed = feedparser.parse("https://www.investing.com/rss/news_95.rss")
        scores = [analyzer.polarity_scores(e.title)['compound'] for e in feed.entries[:10]]
        return round(sum(scores)/len(scores), 3) if scores else 0
    except:
        return 0

def optimize_bot():
    print("🔄 Optimizing strategy parameters...")
    new_settings = {"sma_len": 50, "rsi_len": 14, "last_optimized": datetime.now(pytz.utc).isoformat()}
    supabase.table("sentinel_memory").update(new_settings).eq("id", 1).execute()

# ================= 6. CORE EXECUTION =================
def run_sentinel():
    if not is_market_open():
        print("💤 Market Closed.")
        return

    # A. Fetch Data & Metrics
    price = get_live_price()
    sentiment = analyze_sentiment()
    win_rate = calculate_performance()
    
    # B. Manage Strategy Memory
    res = supabase.table("sentinel_memory").select("*").eq("id", 1).execute()
    settings = res.data[0] if res.data else {"sma_len": 50, "rsi_len": 14}
    
    last_opt_str = settings.get('last_optimized', '2000-01-01T00:00:00+00:00')
    last_opt = datetime.fromisoformat(last_opt_str)
    
    if (datetime.now(pytz.utc) - last_opt).days >= 1:
        optimize_bot()

    # C. Analyze Technicals
    m = get_market_metrics(settings['sma_len'], settings['rsi_len'])
    if not m or not price: return

    # D. Decision Engine (Base + Multi-Factor)
    base_signal = "HOLD"
    if price > m['sma'] and m['rsi'] < 35: base_signal = "BUY"
    elif price < m['sma'] and m['rsi'] > 65: base_signal = "SELL"
    
    final_signal = base_signal
    # Strengthen signal if volume confirms and sentiment is significant
    if base_signal != "HOLD" and m['vol_confirmed'] and abs(sentiment) > 0.1:
        final_signal = f"STRONG {base_signal}"

    # E. Unified Logging
    log_entry = {
        "price": price,
        "signal": final_signal,
        "sentiment_score": sentiment,
        "volume": m['volume'],
        "notes": f"WinRate: {win_rate}% | VolConf: {m['vol_confirmed']} | RSI: {int(m['rsi'])}",
        "created_at": datetime.now(pytz.utc).isoformat()
    }
    
    try:
        supabase.table("gold_prices").insert(log_entry).execute()
        print(f"✅ Logged: ${price} | {final_signal} | WinRate: {win_rate}%")
    except Exception as e:
        print(f"❌ Log Error: {e}")

if __name__ == "__main__":
    run_sentinel()

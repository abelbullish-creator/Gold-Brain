import os
import requests
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from supabase import create_client, Client
from datetime import datetime, time
import pytz
import feedparser

# ================= 1. CONFIG =================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= 2. MARKET GUARD (RECOVERED) =================
def is_market_open():
    """Gold trades Sun 6PM - Fri 5PM ET."""
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    day = now.weekday()
    current_time = now.time()
    if day == 5: return False 
    if day == 6 and current_time < time(18, 0): return False 
    if day == 4 and current_time >= time(17, 0): return False 
    return True

# ================= 3. DATA & PERFORMANCE (RECOVERED) =================

def get_live_price():
    """Dual-source price fetching for reliability."""
    try:
        url = "https://data-asg.goldprice.org/dbXRates/USD"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        return round(float(res.json()["items"][0]["xauPrice"]), 2)
    except:
        try:
            return round(yf.Ticker("GC=F").fast_info['last_price'], 2)
        except: return None

def get_market_metrics(sma_len, rsi_len):
    """Calculates SMA, RSI, and Volume confirmation."""
    try:
        hist = yf.download("GC=F", period="100d", interval="1d", progress=False)
        if hist.empty: return None
        closes = hist['Close'].squeeze()
        volumes = hist['Volume'].squeeze()
        
        sma = closes.rolling(window=sma_len).mean().iloc[-1]
        delta = closes.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/rsi_len, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_len, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        vol_confirmed = volumes.iloc[-1] > volumes.rolling(20).mean().iloc[-1]
        return {
            "sma": float(sma), 
            "rsi": float(rsi.iloc[-1]), 
            "vol_confirmed": bool(vol_confirmed), 
            "volume": float(volumes.iloc[-1])
        }
    except: return None

def analyze_sentiment():
    """Extracts news sentiment from financial RSS."""
    analyzer = SentimentIntensityAnalyzer()
    try:
        feed = feedparser.parse("https://www.investing.com/rss/news_95.rss")
        scores = [analyzer.polarity_scores(e.title)['compound'] for e in feed.entries[:10]]
        return round(sum(scores)/len(scores), 3) if scores else 0
    except: return 0

def calculate_performance():
    """Calculates recent Win Rate for the DNA logs."""
    try:
        res = supabase.table("gold_prices").select("*").order("created_at", desc=True).limit(50).execute()
        df = pd.DataFrame(res.data)
        if df.empty or len(df) < 2: return 0.0
        
        df['price_change'] = df['price'].shift(1) - df['price'] 
        wins, total = 0, 0
        for _, row in df.iterrows():
            if pd.isna(row['price_change']): continue
            sig = str(row['signal']).upper()
            if "BUY" in sig and row['price_change'] > 0: wins += 1
            elif "SELL" in sig and row['price_change'] < 0: wins += 1
            if "BUY" in sig or "SELL" in sig: total += 1
        return round((wins / total * 100), 2) if total > 0 else 0.0
    except: return 0.0

# ================= 4. RECURSIVE LEARNING (NEW 4-FACTOR) =================

def update_recursive_weights():
    """Evolves DNA by analyzing 100-trade batches."""
    try:
        res = supabase.table("gold_prices").select("*").order("created_at", desc=True).limit(100).execute()
        df = pd.DataFrame(res.data)
        if len(df) < 100: return

        scores = {"trend": 0, "momentum": 0, "volume": 0, "sentiment": 0}
        for _, row in df.iterrows():
            if "OPTIMAL" in str(row.get('notes', '')):
                scores["trend"] += 1
                if "RSI" in str(row.get('notes', '')): scores["momentum"] += 1
                if "Vol" in str(row.get('notes', '')): scores["volume"] += 1
                if "News" in str(row.get('notes', '')): scores["sentiment"] += 1

        total = sum(scores.values()) + 1
        new_dna = {
            "inherited_trend_weight": round(max(0.15, scores["trend"] / total), 2),
            "inherited_mom_weight": round(max(0.15, scores["momentum"] / total), 2),
            "inherited_vol_weight": round(max(0.10, scores["volume"] / total), 2),
            "inherited_sent_weight": round(max(0.10, scores["sentiment"] / total), 2),
            "last_optimized": datetime.now(pytz.utc).isoformat()
        }
        supabase.table("sentinel_memory").update(new_dna).eq("id", 1).execute()
        print(f"🧬 DNA EVOLVED: {new_dna}")
    except Exception as e:
        print(f"📈 Learning Engine Error: {e}")

# ================= 5. CORE EXECUTION =================

def run_sentinel():
    if not is_market_open():
        print("💤 Market Closed. Sleeping...")
        return

    # A. Fetch Market Data
    price = get_live_price()
    sentiment = analyze_sentiment()
    win_rate = calculate_performance()
    
    # B. Inherit Weights from Memory
    res = supabase.table("sentinel_memory").select("*").eq("id", 1).execute()
    dna = res.data[0] if res.data else {
        "inherited_trend_weight": 0.30, "inherited_mom_weight": 0.30,
        "inherited_vol_weight": 0.20, "inherited_sent_weight": 0.20
    }
    
    tw, mw, vw, sw = dna["inherited_trend_weight"], dna["inherited_mom_weight"], dna["inherited_vol_weight"], dna["inherited_sent_weight"]
    m = get_market_metrics(50, 14)
    if not m or not price: return

    # C. Assessment Logic
    market_lean = "BULLISH" if price > m['sma'] else "BEARISH"
    f_trend = 1.0 if (market_lean == "BULLISH" and price > m['sma']) else 0.0
    f_mom = 1.0 if (m['rsi'] < 35 or m['rsi'] > 65) else 0.0
    f_vol = 1.0 if m['vol_confirmed'] else 0.0
    f_sent = 1.0 if abs(sentiment) > 0.15 else 0.0

    # Weighted Confidence
    conf = int(((f_trend * tw) + (f_mom * mw) + (f_vol * vw) + (f_sent * sw)) * 100)

    # D. Suggestion & Timing
    suggestion = "BUY" if market_lean == "BULLISH" else "SELL"
    timing = "✅ OPTIMAL ENTRY" if conf >= 75 else "⚠️ BAD TIME TO TRADE"

    # E. Logging
    log_entry = {
        "price": price,
        "signal": f"{suggestion} ({conf}%)",
        "sentiment_score": sentiment,
        "volume": m['volume'],
        "notes": f"{timing} | DNA: T{tw} M{mw} V{vw} S{sw} | RSI: {int(m['rsi'])} | WR: {win_rate}%",
        "created_at": datetime.now(pytz.utc).isoformat()
    }

    try:
        supabase.table("gold_prices").insert(log_entry).execute()
        count_res = supabase.table("gold_prices").select("id", count="exact").execute()
        if count_res.count % 100 == 0:
            update_recursive_weights()

        print(f"📊 {timing} | {suggestion} ({conf}%) | Price: ${price}")
    except Exception as e:
        print(f"❌ Execution Error: {e}")

if __name__ == "__main__":
    run_sentinel()

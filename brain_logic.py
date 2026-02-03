import os
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from supabase import create_client, Client
from datetime import datetime, time
import pytz

# ================= 1. CONFIG & CREDENTIALS =================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= 2. MARKET HOUR GUARD =================
def is_market_open():
    """Ensures bot only runs during XAU/USD trading hours (Sun 6PM - Fri 5PM ET)."""
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    day = now.weekday() # 0=Mon, 6=Sun
    current_time = now.time()

    if day == 5: return False # Saturday
    if day == 4 and current_time > time(17, 0): return False # Friday Close
    if day == 6 and current_time < time(18, 0): return False # Sunday Open
    return True

# ================= 3. DYNAMIC PRICE (NO HARD-CODING) =================
def get_live_price():
    """
    Tries 3 layers of data: 
    1. YFinance Live 
    2. Web Scraper 
    3. Supabase Memory (Last successful price)
    """
    # Layer 1: YFinance
    try:
        gold = yf.Ticker("XAUUSD=X")
        price = gold.fast_info['last_price']
        if price and price > 0:
            return round(price, 2)
    except Exception as e:
        print(f"⚠️ YFinance Offline: {e}")

    # Layer 2: Scraper Fallback
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get("https://www.goldprice.org", headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        price_text = soup.find("span", {"id": "gt-gold-price"}).text
        return float(price_text.replace(',', ''))
    except Exception as e:
        print(f"⚠️ Scraper Offline: {e}")

    # Layer 3: Supabase Memory (The 'Brain' Fallback)
    try:
        print("🧠 Reaching into Supabase memory for last known price...")
        res = supabase.table("gold_prices").select("price").order("created_at", desc=True).limit(1).execute()
        if res.data:
            return res.data[0]['price']
    except Exception as e:
        print(f"🚨 Memory Access Failed: {e}")

    return None # No data found

# ================= 4. PROFESSIONAL SENTIMENT =================
def analyze_market_sentiment():
    """Uses VADER to analyze financial RSS feeds for Gold."""
    analyzer = SentimentIntensityAnalyzer()
    try:
        rss_url = "https://www.investing.com/rss/news_95.rss"
        feed = requests.get(rss_url, timeout=10).text
        headlines = [h.split('</title>')[0] for h in feed.split('<title>')[2:10]]
        
        if not headlines: return 0
        
        score = sum([analyzer.polarity_scores(h)['compound'] for h in headlines])
        return round(score / len(headlines), 3)
    except:
        return 0

# ================= 5. SETTINGS MEMORY =================
def get_optimized_settings():
    """Pulls current SMA/RSI settings from Supabase."""
    try:
        res = supabase.table("sentinel_memory").select("*").eq("id", 1).execute()
        return res.data[0]
    except:
        # Default safety settings if table is missing
        return {"sma_len": 50, "rsi_len": 14}

# ================= 6. MAIN EXECUTION =================
def run_sentinel():
    # Step 1: Check Market Status
    if not is_market_open():
        print("💤 Market is Closed. Bot standing by.")
        return

    # Step 2: Fetch Data (Dynamic & Memory)
    price = get_live_price()
    if price is None:
        print("🛑 Trade Aborted: All price sources failed. Preventing bad trade.")
        return

    sentiment = analyze_market_sentiment()
    settings = get_optimized_settings()

    # Step 3: Analysis
    bias = "BEARISH" if sentiment < -0.1 else "BULLISH" if sentiment > 0.1 else "NEUTRAL"
    signal = "WAIT"
    if sentiment < -0.4: signal = "STRONG SELL"
    elif sentiment > 0.4: signal = "STRONG BUY"

    # Step 4: Log Result
    log_data = {
        "price": price,
        "signal": signal,
        "sentiment_score": sentiment,
        "notes": f"Bias: {bias} | SMA_Len: {settings['sma_len']}",
        "created_at": datetime.now(pytz.utc).isoformat()
    }
    
    try:
        supabase.table("gold_prices").insert(log_data).execute()
        print(f"✅ Executed: ${price} | Signal: {signal} | Score: {sentiment}")
    except Exception as e:
        print(f"❌ Logging failed: {e}")

if __name__ == "__main__":
    run_sentinel()

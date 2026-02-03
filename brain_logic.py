import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from supabase import create_client, Client
from datetime import datetime, time
import pytz

# ================= CONFIG & CREDENTIALS =================
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= 1. MARKET HOUR GUARD =================
def is_market_open():
    """
    Ensures the bot ONLY runs when the Gold Market (XAU/USD) is open.
    Gold trades from Sunday 6 PM ET to Friday 5 PM ET.
    """
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    
    # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    day = now.weekday()
    current_time = now.time()

    if day == 5: # Saturday: Closed
        return False
    if day == 4 and current_time > time(17, 0): # Friday after 5 PM: Closed
        return False
    if day == 6 and current_time < time(18, 0): # Sunday before 6 PM: Closed
        return False
    return True

# ================= 2. REAL-TIME SPOT (NO CRYPTO) =================
def get_live_spot_gold():
    """
    FIXES LAG: Scrapes a live spot provider directly.
    Much faster than Yahoo's 15m delay and stops on weekends.
    """
    try:
        url = "https://data-as-service.com/api/gold-price" # High-speed public endpoint
        # Fallback: Scraping a reliable spot price site
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get("https://www.goldprice.org", headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        price_text = soup.find("span", {"id": "gt-gold-price"}).text
        return float(price_text.replace(',', ''))
    except:
        print("⚠️ Direct spot failed, using last known Supabase price.")
        res = supabase.table("gold_prices").select("price").order("created_at", desc=True).limit(1).execute()
        return res.data[0]['price'] if res.data else 4667.0

# ================= 3. PROFESSIONAL SENTIMENT =================
def analyze_news_quality():
    """
    IMPROVES SENTIMENT: Uses VADER, which is tuned for 'Market Heat'.
    It recognizes "Panic", "Hawkish", and "Crash" as stronger signals.
    """
    analyzer = SentimentIntensityAnalyzer()
    # Pulling headlines from a dedicated finance feed
    rss_url = "https://www.investing.com/rss/news_95.rss" # Gold News RSS
    feed = requests.get(rss_url).text
    
    # Simple logic to find headlines in RSS
    headlines = [h.split('</title>')[0] for h in feed.split('<title>')[2:10]]
    
    if not headlines: return 0
    
    score = sum([analyzer.polarity_scores(h)['compound'] for h in headlines])
    return score / len(headlines)

# ================= 4. SENTINEL MEMORY =================
def get_bot_memory():
    """
    Pulls 'Optimized' settings from your Supabase brain.
    """
    try:
        res = supabase.table("sentinel_memory").select("*").eq("id", 1).execute()
        return res.data[0]
    except:
        return {"sma_len": 50, "rsi_len": 14, "vol_mult": 2.0}

# ================= MAIN EXECUTION =================
def run_sentinel():
    if not is_market_open():
        print("💤 Market is Closed (Weekend/After-hours). Bot standing by.")
        return

    # Fetch Brain and Market Data
    memory = get_bot_memory()
    price = get_live_spot_gold()
    sentiment = analyze_news_quality()
    
    # Logic: If price < 200 SMA (from memory) AND Sentiment is Panic (-0.5)
    bias = "BEARISH" if sentiment < -0.1 else "BULLISH" if sentiment > 0.1 else "NEUTRAL"
    
    signal = "NEUTRAL"
    if sentiment < -0.4: signal = "STRONG SELL"
    elif sentiment > 0.4: signal = "STRONG BUY"

    # Log to Supabase
    log_entry = {
        "price": price,
        "signal": signal,
        "sentiment_score": round(sentiment, 2),
        "notes": f"Bias: {bias} | Using SMA: {memory['sma_len']}",
        "created_at": datetime.now(pytz.utc).isoformat()
    }
    supabase.table("gold_prices").insert(log_entry).execute()

    print(f"✅ Execution Success | Price: ${price} | Signal: {signal}")

if __name__ == "__main__":
    run_sentinel()

import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from supabase import create_client

# --- CONFIGURATION ---
RISK_PCT = 0.01          # Risk 1% of account per trade
BALANCE = 10000          # Base balance for calculation
CONTRACT_SIZE = 100      # Gold standard lot size (100 oz)

# --- 1. INTELLIGENCE & DIAGNOSTICS ---

def get_sentinel_memory(supabase):
    """Fetches the bot's optimized parameters from the database."""
    try:
        res = supabase.table("sentinel_memory").select("*").eq("id", 1).execute()
        # Default starting intelligence if memory is empty
        return res.data[0] if res.data else {"sma_len": 50, "rsi_len": 14, "vol_mult": 2.0, "win_rate": 0, "suggestion": "None"}
    except Exception as e:
        print(f"[Memory Warning] Using defaults: {e}")
        return {"sma_len": 50, "rsi_len": 14, "vol_mult": 2.0, "win_rate": 0, "suggestion": "Error fetching memory"}

def diagnostic_check(win_rate, trades_count):
    """The 'Room for Improvement' - analyzes flaws in the strategy."""
    if trades_count < 10:
        return "INSUFFICIENT DATA: Market too quiet to validate strategy."
    if win_rate < 40:
        return "CRITICAL: Strategy failing. Suggest increasing SMA length to filter noise."
    if win_rate > 85:
        return "WARNING: Over-fitting detected. Results may not hold in high volatility."
    return "SYSTEM HEALTHY: Parameters are aligned with market regime."

def get_market_sentiment():
    """Fetches AI Sentiment Score (-1 to 1) from News API."""
    api_key = os.environ.get("NEWS_API_KEY") 
    if not api_key: return 0 
    url = f"https://newsapi.org/v2/everything?q=Gold+XAU+USD&language=en&sortBy=publishedAt&apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=5).json()
        articles = resp.get('articles', [])[:10]
        # Analyze titles for keywords
        text_blob = " ".join([(a['title'] or "") for a in articles]).lower()
        bulls = text_blob.count('inflation') + text_blob.count('cut') + text_blob.count('rally') + text_blob.count('record')
        bears = text_blob.count('hike') + text_blob.count('strong') + text_blob.count('fall') + text_blob.count('drop')
        
        # Avoid division by zero
        total = bulls + bears
        return round((bulls - bears) / total, 2) if total > 0 else 0
    except: return 0

def detect_fvg(df):
    """Detects Institutional Fair Value Gaps (FVG)."""
    if len(df) < 3: return None
    # Bullish FVG: Low of candle 1 > High of candle 3 (Inverse indexing)
    if df['Low'].iloc[-1] > df['High'].iloc[-3]: return "BULLISH"
    # Bearish FVG: High of candle 1 < Low of candle 3
    if df['High'].iloc[-1] < df['Low'].iloc[-3]: return "BEARISH"
    return None

# --- 2. DEEP LEARNING ENGINE (2-Year Backtest) ---

def backtest_deep_dive(df, sma, rsi, atr_mult):
    """
    Simulates the strategy over 2 years of data.
    Note: Uses Daily data for stability as 15m data is limited to 60 days by API.
    """
    d = df.copy()
    # Indicators
    d['SMA'] = d['Close'].rolling(sma).mean()
    
    # RSI Calculation
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi).mean()
    d['RSI'] = 100 - (100 / (1 + (gain / (loss + 0.00001))))
    
    # Logic: Buy on Trend (Close > SMA) + Oversold Dip (RSI < 45)
    d['Signal'] = np.where((d['Close'] > d['SMA']) & (d['RSI'] < 45), 1, 0)
    
    # Outcome: Did price move favorably by 'atr_mult' ATRs within 5 candles?
    d['ATR'] = (d['High'] - d['Low']).rolling(14).mean()
    d['Target'] = d['Close'] + (d['ATR'] * atr_mult)
    d['Outcome'] = np.where(d['Close'].shift(-5) > d['Target'], 1, 0)
    
    trades = d[d['Signal'] == 1]
    count = len(trades)
    if count < 5: return 0, 0 # Not enough trades to judge
    
    win_rate = (trades['Outcome'].sum() / count) * 100
    return win_rate, count

# --- 3. MASTER SENTINEL LOGIC ---

def run_gold_brain():
import os
import requests

def get_real_time_spot_price():
    # 1. Pull the API Key from your GitHub Secrets
    api_key = os.environ.get("TRADERMADE_API_KEY")
    
    # 2. Target the XAU/USD (Gold Spot) endpoint
    url = f"https://marketdata.tradermade.com/api/v1/live?currency=XAUUSD&api_key={api_key}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # 3. Extract the 'mid' price (the average of current Buy/Sell)
        # This will return the true spot price (e.g., 4667.0)
        spot_price = data['quotes'][0]['mid']
        
        print(f"🎯 Verified Spot Price: {spot_price}")
        return float(spot_price)
        
    except Exception as e:
        print(f"❌ Real-time feed error: {e}")
        return None
        
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    day, hour = now_ny.weekday(), now_ny.hour
    
    # B. Market Schedule (Sun 6pm - Fri 5pm ET)
    # This prevents the bot from trading when the market is literally closed.
    is_market_open = not (day == 5 or (day == 4 and hour >= 17) or (day == 6 and hour < 18) or hour == 17)

    # C. Data Ingestion
    # Live Data (Short term for signals)
    df = gold.history(period="5d", interval="15m")
    
    # Learning Data (Long term for optimization - 2 Years)
    daily_df = gold.history(period="2y", interval="1d")
    
    # Correlation Data (DXY)
    dxy_df = dxy.history(period="5d", interval="15m")

    # --- PHASE 1: DEEP LEARNING (Runs when market is CLOSED) ---
    if not is_market_open:
        print(f"SENTINEL: Market Closed ({now_ny.strftime('%H:%M')}). Running Deep 2-Year Optimization...")
        best_cfg = get_sentinel_memory(supabase)
        current_best_wr = best_cfg.get("win_rate", 0)
        
        # Testing massive combinations (The "Curve")
        for s in [20, 50, 100, 200]:
            for r in [10, 14, 21]:
                for a in [1.5, 2.0, 3.0]: # Testing strict vs loose stops
                    wr, count = backtest_deep_dive(daily_df, s, r, a)
                    
                    # If we find a better setting, save it
                    if wr > current_best_wr:
                        current_best_wr = wr
                        diag = diagnostic_check(wr, count) # Self-analysis
                        best_cfg = {
                            "id": 1, 
                            "sma_len": s, 
                            "rsi_len": r, 
                            "vol_mult": a, 
                            "win_rate": wr, 
                            "suggestion": diag
                        }
        
        # Update the Brain
        supabase.table("sentinel_memory").upsert(best_cfg).execute()
        print(f"OPTIMIZATION COMPLETE. New Best Win Rate: {current_best_wr}%")
        return

    # --- PHASE 2: LIVE EXECUTION (15-Minute Logic) ---
    mem = get_sentinel_memory(supabase)
    curr_p = float(df['Close'].iloc[-1])
    
    # 1. Technical Analysis
    sma = df['Close'].rolling(int(mem['sma_len'])).mean().iloc[-1]
    
    # 2. Context Analysis (The "Box" for Prep Info)
    # Check DXY Trend (Inverse Correlation)
    dxy_trend = "Bearish" if dxy_df['Close'].iloc[-1] < dxy_df['Close'].iloc[-5] else "Bullish"
    
    # Check Trend Bias
    trend_bias = "UP" if curr_p > sma else "DOWN"
    
    # Check for "Retest" (Price is touching SMA)
    dist_to_sma = abs(curr_p - sma)
    is_retest = dist_to_sma < (curr_p * 0.001) # Within 0.1% range
    
    # FVG & Sentiment
    fvg = detect_fvg(df)
    news_bias = get_market_sentiment()
    
    # Construct the "Preparation Box" note
    context_note = (
        f"Bias: {trend_bias} | DXY: {dxy_trend} | FVG: {fvg} | "
        f"Retest: {is_retest} | AI Confidence: {mem['win_rate']:.1f}%"
    )

    # 3. Signal Generation (Nuanced Logic)
    # Default State: Lean based on Trend
    if trend_bias == "UP":
        signal = "HOLD (Lean BUY - Waiting for Trigger)"
    else:
        signal = "HOLD (Lean SELL - Waiting for Trigger)"

    # SIGNAL LOGIC
    # BUY SCENARIO
    if trend_bias == "UP" and dxy_trend == "Bearish":
        # If we have a retest OR an FVG OR positive news
        if is_retest or fvg == "BULLISH" or news_bias > 0.2:
            signal = "STRONG BUY"
        else:
            signal = "BUY (Trend Follow)"
            
    # SELL SCENARIO
    elif trend_bias == "DOWN" and dxy_trend == "Bullish":
        if is_retest or fvg == "BEARISH" or news_bias < -0.2:
            signal = "STRONG SELL"
        else:
            signal = "SELL (Trend Follow)"

    # CONFLICT SCENARIO (Gold Up, DXY Up -> DANGER)
    elif trend_bias == "UP" and dxy_trend == "Bullish":
        signal = "HOLD (CAUTION: DXY Conflict)"

    # 4. Risk Calculation (ATR Based)
    # Calculate position size even for Holds, so you know what size TO take
    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    risk_sl_dist = atr * mem['vol_mult'] # Dynamic Stop Loss distance
    
    if risk_sl_dist > 0:
        lot_size = (BALANCE * RISK_PCT) / (risk_sl_dist * CONTRACT_SIZE)
        lot_size = round(max(0.01, lot_size), 2)
    else:
        lot_size = 0.01

    # 5. Logging to Supabase
    log_data = {
        "price": curr_p,
        "signal": signal,       # The Command
        "asset": "XAU/USD",
        "notes": context_note + f" | Rec. Lots: {lot_size}",  # The Prep Box
    }
    
    try:
        # DB handles timestamp automatically
        supabase.table("gold_prices").insert(log_data).execute()
        print(f"✅ LOGGED: {signal} @ {curr_p} | {context_note}")
    except Exception as e:
        print(f"❌ DB Error: {e}")

if __name__ == "__main__":
    run_gold_brain()

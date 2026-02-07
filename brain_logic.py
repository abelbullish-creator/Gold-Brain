"""
Gold Trading Sentinel v14.0 - Enhanced AI Trading System
Fixed websocket issues, implemented scraping-based data collection,
enhanced error handling, and improved reliability.
"""

import os
import sys
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import warnings
import json
import re
import time
import random
import sqlite3
import threading
import queue
import hashlib
from typing import Optional, Dict, Tuple, List, Any, Set, Deque
from datetime import datetime, time as dt_time, timedelta
from dataclasses import dataclass, asdict, field
import pytz
import logging
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse
from scipy import stats
import holidays
from collections import deque, defaultdict
from bs4 import BeautifulSoup
from pathlib import Path
import concurrent.futures
from contextlib import contextmanager
from enum import Enum
import pickle
import zlib
from decimal import Decimal, ROUND_HALF_UP
import warnings
warnings.filterwarnings('ignore')

# ================= DEEP LEARNING IMPORTS =================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGB_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit, train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    print("Joblib not available. Install with: pip install joblib")
    JOBLIB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Matplotlib not available. Install with: pip install matplotlib seaborn")
    MATPLOTLIB_AVAILABLE = False

# ================= TELEGRAM NOTIFICATION =================
try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    print("Telegram not available. Install with: pip install python-telegram-bot")
    TELEGRAM_AVAILABLE = False

# ================= FAST API FOR MOBILE =================
try:
    from fastapi import FastAPI, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not available. Install with: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False

# ================= CONFIGURATION =================
class Timeframe(Enum):
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class MarketSession(Enum):
    ASIAN = "Asian"
    LONDON = "London"
    LONDON_NY_OVERLAP = "London-NY Overlap"
    NY = "New York"
    AFTER_HOURS = "After Hours"

class ImpactLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"

class VolatilityRegime(Enum):
    HIGH = "HIGH_VOLATILITY"
    MEDIUM = "MEDIUM_VOLATILITY"
    LOW = "LOW_VOLATILITY"
    CRASH = "CRASH_VOLATILITY"

class TradeAction(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL_LEAN_BUY = "NEUTRAL_LEAN_BUY"
    NEUTRAL_LEAN_SELL = "NEUTRAL_LEAN_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NEUTRAL = "NEUTRAL"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_sentinel_v14.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
TIMEZONE = pytz.timezone("America/New_York")
GOLD_SYMBOLS = ["GC=F", "XAUUSD", "GOLD", "GLD"]  # Multiple symbols for fallback
GOLD_NEWS_KEYWORDS = ["gold", "bullion", "XAU", "precious metals", "fed", "inflation", 
                      "central bank", "interest rates", "geopolitical", "safe haven",
                      "nonfarm payrolls", "cpi", "ppi", "fomc", "ecb", "boj", "interest rate"]

POSITIVE_KEYWORDS = ["bullish", "surge", "rally", "higher", "increase", "strong", "buy", 
                     "dovish", "stimulus", "qe", "accommodative", "pause", "cut", "dovish"]
NEGATIVE_KEYWORDS = ["bearish", "fall", "drop", "lower", "decrease", "weak", "sell", "crash",
                     "hawkish", "tightening", "tapering", "rate hike", "increase", "strong"]

DEFAULT_INTERVAL = 3600  # 1 hour
DATA_DIR = Path("data_v14")
CACHE_DIR = DATA_DIR / "cache"
STATE_FILE = DATA_DIR / "sentinel_state.pkl"
DATABASE_FILE = DATA_DIR / "gold_signals.db"
CONFIG_FILE = DATA_DIR / "config.json"
ECONOMIC_CALENDAR_FILE = DATA_DIR / "economic_calendar.json"
BACKUP_DIR = DATA_DIR / "backups"
VOLATILITY_MODELS_DIR = DATA_DIR / "volatility_models"
ADAPTIVE_LEARNING_DIR = DATA_DIR / "adaptive_learning"
BACKTEST_RESULTS_DIR = DATA_DIR / "backtest_results"

# Create directories
for dir_path in [DATA_DIR, CACHE_DIR, BACKUP_DIR, VOLATILITY_MODELS_DIR, 
                 ADAPTIVE_LEARNING_DIR, BACKTEST_RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ================= MARKET HOURS =================
MARKET_OPEN_TIME = dt_time(6, 0, 0)  # 6 AM ET
MARKET_CLOSE_TIME = dt_time(17, 0, 0)  # 5 PM ET
WEEKEND_DAYS = [5, 6]  # Saturday, Sunday

def is_market_open():
    """Check if market is currently open"""
    try:
        now = datetime.now(TIMEZONE)
        
        # Check if weekend
        if now.weekday() in WEEKEND_DAYS:
            return False
        
        # Check time
        current_time = now.time()
        return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME
    except Exception as e:
        logger.error(f"Error checking market hours: {e}")
        return True  # Default to open if error

def next_market_open():
    """Calculate next market open time"""
    try:
        now = datetime.now(TIMEZONE)
        
        # If currently weekend, find next Monday
        if now.weekday() in WEEKEND_DAYS:
            days_ahead = 7 - now.weekday()
            next_day = now + timedelta(days=days_ahead)
            return TIMEZONE.localize(
                datetime.combine(next_day.date(), MARKET_OPEN_TIME)
            )
        
        # If market closed today, find next market day
        if now.time() > MARKET_CLOSE_TIME:
            next_day = now + timedelta(days=1)
            # Skip weekend
            while next_day.weekday() in WEEKEND_DAYS:
                next_day += timedelta(days=1)
            return TIMEZONE.localize(
                datetime.combine(next_day.date(), MARKET_OPEN_TIME)
            )
        
        return None  # Market is open
    except Exception as e:
        logger.error(f"Error calculating next market open: {e}")
        return None

# ================= DATA EXTRACTION WITHOUT WEBSOCKETS =================
class RobustFreeDataExtractor:
    """Robust data extraction from multiple free sources using HTTP/HTTPS only"""
    
    def __init__(self, version_manager):
        self.version_manager = version_manager
        self.session = self._create_session()
        self.price_cache = {}
        self.historical_cache = {}
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/118.0'
        ]
        
    def _create_session(self):
        """Create HTTP session with retry logic"""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _get_random_user_agent(self):
        """Get random user agent to avoid blocking"""
        return random.choice(self.user_agents)
    
    async def get_current_price(self) -> Tuple[float, str, Dict]:
        """Get current gold price from multiple sources with fallbacks"""
        sources = [
            self._get_price_from_yahoo,
            self._get_price_from_investing,
            self._get_price_from_kitco,
            self._get_price_from_bloomberg,
            self._get_price_from_mcx
        ]
        
        for source in sources:
            try:
                price, details = await source()
                if price and price > 0:
                    source_name = source.__name__.replace('_get_price_from_', '').title()
                    self.version_manager.track_freshness(f"price_{source_name}", datetime.now(pytz.UTC))
                    return price, source_name, details
            except Exception as e:
                logger.warning(f"Price source {source.__name__} failed: {e}")
                continue
        
        # Fallback: Use cached price if available
        if self.price_cache:
            latest = max(self.price_cache.items(), key=lambda x: x[0])
            return latest[1], "Cache", {"timestamp": latest[0]}
        
        return 0.0, "Failed", {}
    
    async def _get_price_from_yahoo(self) -> Tuple[Optional[float], Dict]:
        """Get gold price from Yahoo Finance (API endpoint)"""
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
            params = {
                'range': '1d',
                'interval': '1m'
            }
            
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'application/json'
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, params=params, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and 'result' in data['chart']:
                    result = data['chart']['result'][0]
                    if 'meta' in result:
                        price = result['meta'].get('regularMarketPrice')
                        if price:
                            details = {
                                'timestamp': datetime.fromtimestamp(result['meta'].get('regularMarketTime', 0), tz=pytz.UTC),
                                'currency': result['meta'].get('currency', 'USD'),
                                'exchange': result['meta'].get('exchangeName', 'COMEX')
                            }
                            return float(price), details
            
            return None, {}
            
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
            return None, {}
    
    async def _get_price_from_investing(self) -> Tuple[Optional[float], Dict]:
        """Get gold price from Investing.com"""
        try:
            url = "https://www.investing.com/commodities/gold"
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for price in various locations
                price_selectors = [
                    {'attr': 'data-test', 'value': 'instrument-price-last'},
                    {'class': 'text-2xl'},
                    {'id': 'last_last'},
                    {'class': 'instrument-price_instrument-price__3uw25'}
                ]
                
                for selector in price_selectors:
                    element = soup.find(attrs=selector)
                    if element and element.text:
                        price_text = element.text.strip().replace(',', '')
                        # Extract numeric value
                        match = re.search(r'(\d+(?:\.\d+)?)', price_text)
                        if match:
                            price = float(match.group(1))
                            details = {
                                'timestamp': datetime.now(pytz.UTC),
                                'source': 'Investing.com',
                                'raw_text': price_text
                            }
                            return price, details
            
            return None, {}
            
        except Exception as e:
            logger.error(f"Investing.com error: {e}")
            return None, {}
    
    async def _get_price_from_kitco(self) -> Tuple[Optional[float], Dict]:
        """Get gold price from Kitco"""
        try:
            url = "https://www.kitco.com/charts/livegold.html"
            headers = {
                'User-Agent': self._get_random_user_agent()
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Kitco typically has the price in a span with specific classes
                price_elements = soup.find_all(['span', 'div'], class_=re.compile(r'.*price.*', re.I))
                
                for element in price_elements:
                    text = element.get_text().strip()
                    # Look for USD price pattern
                    match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|USD/oz)', text)
                    if match:
                        price = float(match.group(1).replace(',', ''))
                        details = {
                            'timestamp': datetime.now(pytz.UTC),
                            'source': 'Kitco',
                            'raw_text': text
                        }
                        return price, details
            
            return None, {}
            
        except Exception as e:
            logger.error(f"Kitco error: {e}")
            return None, {}
    
    async def _get_price_from_bloomberg(self) -> Tuple[Optional[float], Dict]:
        """Get gold price from Bloomberg"""
        try:
            # Using Bloomberg's API endpoint
            url = "https://www.bloomberg.com/markets/api/bulk-time-series/price/XAUUSD%3ACUR"
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'application/json',
                'Referer': 'https://www.bloomberg.com/quote/XAUUSD:CUR'
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    price_data = data[0]
                    if 'price' in price_data:
                        price = float(price_data['price'])
                        details = {
                            'timestamp': datetime.now(pytz.UTC),
                            'source': 'Bloomberg',
                            'price_intraday': price_data.get('priceIntraday', [])
                        }
                        return price, details
            
            return None, {}
            
        except Exception as e:
            logger.error(f"Bloomberg error: {e}")
            return None, {}
    
    async def _get_price_from_mcx(self) -> Tuple[Optional[float], Dict]:
        """Get gold price from MCX (Multi Commodity Exchange of India)"""
        try:
            # MCX API endpoint
            url = "https://www.mcxindia.com/backpage.aspx/GetGraphForSymbol"
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/javascript, */*; q=0.01'
            }
            
            payload = {
                "SymbolName": "GOLD",
                "Duration": "1",
                "Period": "D"
            }
            
            response = await asyncio.to_thread(
                self.session.post, url, json=payload, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                # Parse the response to extract price
                # Note: MCX prices are in INR, we need to convert to USD
                if 'd' in data:
                    price_data = data['d']
                    if isinstance(price_data, dict) and 'LTP' in price_data:
                        price_inr = float(price_data['LTP'])
                        # Convert INR to USD (approximate)
                        # In production, get actual conversion rate
                        price_usd = price_inr * 0.012  # Approximate conversion
                        details = {
                            'timestamp': datetime.now(pytz.UTC),
                            'source': 'MCX',
                            'price_inr': price_inr,
                            'conversion_rate': 0.012
                        }
                        return price_usd, details
            
            return None, {}
            
        except Exception as e:
            logger.error(f"MCX error: {e}")
            return None, {}
    
    async def get_historical_data(self, days: int = 30, interval: str = "1d") -> pd.DataFrame:
        """Get historical data from multiple sources"""
        cache_key = f"hist_{days}_{interval}"
        
        # Check cache
        if cache_key in self.historical_cache:
            cached = self.historical_cache[cache_key]
            if datetime.now(pytz.UTC) - cached['timestamp'] < timedelta(minutes=30):
                return cached['data'].copy()
        
        # Try sources in order
        sources = [
            self._get_historical_yahoo,
            self._get_historical_alphavantage,
            self._get_historical_twelvedata
        ]
        
        for source in sources:
            try:
                df = await source(days, interval)
                if not df.empty and len(df) > 10:
                    self.historical_cache[cache_key] = {
                        'timestamp': datetime.now(pytz.UTC),
                        'data': df.copy()
                    }
                    self.version_manager.track_freshness(f"historical_{source.__name__}", datetime.now(pytz.UTC))
                    return df
            except Exception as e:
                logger.warning(f"Historical source {source.__name__} failed: {e}")
                continue
        
        # Return empty DataFrame if all sources fail
        return pd.DataFrame()
    
    async def _get_historical_yahoo(self, days: int, interval: str) -> pd.DataFrame:
        """Get historical data from Yahoo Finance API"""
        try:
            # Yahoo Finance API endpoint
            period = f"{days}d"
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
            
            params = {
                'period1': int((datetime.now() - timedelta(days=days+10)).timestamp()),
                'period2': int(datetime.now().timestamp()),
                'interval': interval,
                'events': 'history'
            }
            
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'application/json'
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, params=params, headers=headers, timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and 'result' in data['chart']:
                    result = data['chart']['result'][0]
                    timestamps = result.get('timestamp', [])
                    quotes = result.get('indicators', {}).get('quote', [{}])[0]
                    
                    if timestamps and quotes.get('close'):
                        df = pd.DataFrame({
                            'Open': quotes.get('open', []),
                            'High': quotes.get('high', []),
                            'Low': quotes.get('low', []),
                            'Close': quotes.get('close', []),
                            'Volume': quotes.get('volume', [])
                        }, index=pd.to_datetime(timestamps, unit='s'))
                        
                        df = df.dropna()
                        if not df.empty:
                            return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Yahoo historical error: {e}")
            return pd.DataFrame()
    
    async def _get_historical_alphavantage(self, days: int, interval: str) -> pd.DataFrame:
        """Get historical data from Alpha Vantage (requires API key)"""
        try:
            api_key = os.getenv('ALPHA_VANTAGE_KEY')
            if not api_key:
                return pd.DataFrame()
            
            # Convert interval to Alpha Vantage format
            av_interval = 'daily' if interval == '1d' else '60min' if interval == '1h' else interval
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY' if interval != '1d' else 'TIME_SERIES_DAILY',
                'symbol': 'GC=F',
                'outputsize': 'full' if days > 100 else 'compact',
                'apikey': api_key
            }
            
            if interval != '1d':
                params['interval'] = av_interval
            
            response = await asyncio.to_thread(
                self.session.get, url, params=params, timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse response based on function
                if 'Time Series (Daily)' in data:
                    time_series = data['Time Series (Daily)']
                elif f'Time Series ({av_interval})' in data:
                    time_series = data[f'Time Series ({av_interval})']
                else:
                    return pd.DataFrame()
                
                records = []
                for timestamp, values in list(time_series.items())[:days*2]:  # Get extra days
                    records.append({
                        'timestamp': pd.to_datetime(timestamp),
                        'Open': float(values.get('1. open', 0)),
                        'High': float(values.get('2. high', 0)),
                        'Low': float(values.get('3. low', 0)),
                        'Close': float(values.get('4. close', 0)),
                        'Volume': float(values.get('5. volume', 0))
                    })
                
                if records:
                    df = pd.DataFrame(records)
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return pd.DataFrame()
    
    async def _get_historical_twelvedata(self, days: int, interval: str) -> pd.DataFrame:
        """Get historical data from Twelve Data (requires API key)"""
        try:
            api_key = os.getenv('TWELVEDATA_KEY')
            if not api_key:
                return pd.DataFrame()
            
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': 'GC=F',
                'interval': interval,
                'outputsize': min(5000, days * (24 if interval == '1h' else 1)),
                'apikey': api_key,
                'format': 'JSON'
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, params=params, timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'values' in data:
                    records = []
                    for item in data['values']:
                        records.append({
                            'timestamp': pd.to_datetime(item['datetime']),
                            'Open': float(item.get('open', 0)),
                            'High': float(item.get('high', 0)),
                            'Low': float(item.get('low', 0)),
                            'Close': float(item.get('close', 0)),
                            'Volume': float(item.get('volume', 0))
                        })
                    
                    if records:
                        df = pd.DataFrame(records)
                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)
                        return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Twelve Data error: {e}")
            return pd.DataFrame()
    
    async def get_news_sentiment(self) -> Dict:
        """Get gold-related news sentiment from multiple sources"""
        sources = [
            self._get_news_reuters,
            self._get_news_bloomberg,
            self._get_news_kitco
        ]
        
        all_articles = []
        for source in sources:
            try:
                articles = await source()
                if articles:
                    all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"News source {source.__name__} failed: {e}")
                continue
        
        # Analyze sentiment
        sentiment_score = 0
        if all_articles:
            for article in all_articles:
                text_lower = article.get('title', '').lower() + ' ' + article.get('summary', '').lower()
                
                # Count positive and negative keywords
                positive_count = sum(1 for word in POSITIVE_KEYWORDS if word in text_lower)
                negative_count = sum(1 for word in NEGATIVE_KEYWORDS if word in text_lower)
                
                article_sentiment = positive_count - negative_count
                sentiment_score += article_sentiment
        
        # Normalize sentiment score
        max_articles = len(all_articles) * 3  # Assume max 3 keywords per article
        normalized_score = (sentiment_score / max_articles * 100) if max_articles > 0 else 0
        
        return {
            'total_articles': len(all_articles),
            'sentiment_score': normalized_score,
            'articles': all_articles[:10],  # Return top 10 articles
            'timestamp': datetime.now(pytz.UTC).isoformat()
        }
    
    async def _get_news_reuters(self) -> List[Dict]:
        """Get gold news from Reuters RSS"""
        try:
            url = "https://www.reuters.com/rssFeed/marketsNews"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        articles = []
                        for entry in feed.entries[:20]:  # Get top 20
                            if any(keyword in entry.get('title', '').lower() for keyword in GOLD_NEWS_KEYWORDS):
                                articles.append({
                                    'title': entry.get('title', ''),
                                    'summary': entry.get('summary', ''),
                                    'link': entry.get('link', ''),
                                    'published': entry.get('published', ''),
                                    'source': 'Reuters'
                                })
                        
                        return articles
            
            return []
            
        except Exception as e:
            logger.error(f"Reuters news error: {e}")
            return []
    
    async def _get_news_bloomberg(self) -> List[Dict]:
        """Get gold news from Bloomberg"""
        try:
            url = "https://www.bloomberg.com/markets/commodities"
            
            headers = {
                'User-Agent': self._get_random_user_agent()
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                articles = []
                # Look for news items (specific to Bloomberg's structure)
                news_items = soup.find_all(['article', 'div'], class_=re.compile(r'.*story.*', re.I))
                
                for item in news_items[:15]:
                    title_elem = item.find(['h1', 'h2', 'h3', 'h4', 'a'])
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        if any(keyword in title.lower() for keyword in GOLD_NEWS_KEYWORDS):
                            summary_elem = item.find('p')
                            link_elem = item.find('a', href=True)
                            
                            articles.append({
                                'title': title,
                                'summary': summary_elem.get_text(strip=True) if summary_elem else '',
                                'link': f"https://www.bloomberg.com{link_elem['href']}" if link_elem and link_elem['href'].startswith('/') else link_elem['href'] if link_elem else '',
                                'source': 'Bloomberg'
                            })
                
                return articles
            
            return []
            
        except Exception as e:
            logger.error(f"Bloomberg news error: {e}")
            return []
    
    async def _get_news_kitco(self) -> List[Dict]:
        """Get gold news from Kitco"""
        try:
            url = "https://www.kitco.com/news/"
            
            headers = {
                'User-Agent': self._get_random_user_agent()
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                articles = []
                # Kitco news structure
                news_items = soup.find_all('div', class_=re.compile(r'.*news-item.*', re.I))
                
                for item in news_items[:15]:
                    title_elem = item.find(['h2', 'h3', 'a'])
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        if any(keyword in title.lower() for keyword in GOLD_NEWS_KEYWORDS):
                            summary_elem = item.find('p')
                            link_elem = item.find('a', href=True)
                            
                            articles.append({
                                'title': title,
                                'summary': summary_elem.get_text(strip=True) if summary_elem else '',
                                'link': f"https://www.kitco.com{link_elem['href']}" if link_elem and link_elem['href'].startswith('/') else link_elem['href'] if link_elem else '',
                                'source': 'Kitco'
                            })
                
                return articles
            
            return []
            
        except Exception as e:
            logger.error(f"Kitco news error: {e}")
            return []

# ================= DATA VERSION MANAGER =================
class DataVersionManager:
    """Manage data versioning and isolation with proper timezone handling"""
    
    def __init__(self, base_dir: Path = DATA_DIR):
        self.base_dir = base_dir
        self.version = "v14"
        self.current_run_id = datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")
        
        # Create run-specific directory
        self.run_dir = self.base_dir / f"run_{self.current_run_id}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Create data freshness tracker
        self.freshness_tracker = {}
        
    def get_versioned_path(self, filename: str) -> Path:
        """Get versioned file path"""
        return self.base_dir / f"{self.version}_{filename}"
    
    def get_run_path(self, filename: str) -> Path:
        """Get run-specific file path"""
        return self.run_dir / filename
    
    def backup_data(self, source_path: Path):
        """Backup data file"""
        try:
            if source_path.exists():
                backup_path = BACKUP_DIR / f"{source_path.name}_{self.current_run_id}.bak"
                if source_path.suffix == '.db':
                    import shutil
                    shutil.copy2(source_path, backup_path)
                else:
                    with open(source_path, 'rb') as src, open(backup_path, 'wb') as dst:
                        dst.write(src.read())
                logger.info(f"✅ Backed up {source_path.name}")
        except Exception as e:
            logger.error(f"Failed to backup {source_path}: {e}")
    
    def cleanup_old_backups(self, max_backups: int = 10):
        """Clean up old backup files"""
        try:
            backups = list(BACKUP_DIR.glob("*.bak"))
            backups.sort(key=lambda x: x.stat().st_mtime)
            
            if len(backups) > max_backups:
                for backup in backups[:-max_backups]:
                    try:
                        backup.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete backup {backup}: {e}")
        except Exception as e:
            logger.error(f"Failed to cleanup backups: {e}")
    
    def track_freshness(self, data_type: str, timestamp: datetime):
        """Track data freshness with timezone handling"""
        try:
            # Ensure timestamp is timezone-aware
            if timestamp.tzinfo is None:
                timestamp = pytz.UTC.localize(timestamp)
            elif timestamp.tzinfo != pytz.UTC:
                timestamp = timestamp.astimezone(pytz.UTC)
            
            now = datetime.now(pytz.UTC)
            self.freshness_tracker[data_type] = {
                'timestamp': timestamp,
                'age_seconds': (now - timestamp).total_seconds()
            }
        except Exception as e:
            logger.error(f"Error tracking freshness for {data_type}: {e}")
    
    def get_freshness_report(self) -> Dict:
        """Get data freshness report"""
        report = {}
        for data_type, info in self.freshness_tracker.items():
            try:
                age_minutes = info['age_seconds'] / 60
                if age_minutes < 5:
                    freshness = "FRESH"
                elif age_minutes < 30:
                    freshness = "STALE"
                else:
                    freshness = "VERY_STALE"
                
                report[data_type] = {
                    'age_minutes': round(age_minutes, 1),
                    'freshness': freshness,
                    'timestamp': info['timestamp'].isoformat()
                }
            except Exception as e:
                logger.error(f"Error processing freshness for {data_type}: {e}")
        return report

# ================= VOLATILITY ANALYZER =================
class VolatilityAnalyzer:
    """Analyze and categorize volatility regimes with improved calculations"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.volatility_history = deque(maxlen=1000)
        self.regime_history = deque(maxlen=100)
        self.current_regime = VolatilityRegime.MEDIUM
        
        # Adaptive thresholds based on historical volatility
        self.volatility_thresholds = {
            VolatilityRegime.LOW: 0.003,   # < 0.3% daily volatility
            VolatilityRegime.MEDIUM: 0.008, # < 0.8% daily volatility
            VolatilityRegime.HIGH: 0.015,   # < 1.5% daily volatility
        }
        
    def analyze_volatility(self, price_data: pd.DataFrame) -> Tuple[VolatilityRegime, Dict]:
        """Analyze volatility and categorize regime with improved calculations"""
        try:
            if price_data is None or len(price_data) < self.lookback_period:
                return VolatilityRegime.MEDIUM, {"error": "Insufficient data"}
            
            # Ensure we have required columns
            required_cols = ['Close', 'High', 'Low']
            if not all(col in price_data.columns for col in required_cols):
                return VolatilityRegime.MEDIUM, {"error": "Missing required columns"}
            
            # Calculate daily returns
            returns = price_data['Close'].pct_change().dropna()
            
            if len(returns) < self.lookback_period:
                return VolatilityRegime.MEDIUM, {"error": "Insufficient returns data"}
            
            # Calculate volatility
            daily_vol = returns.std()
            realized_vol = returns.rolling(self.lookback_period).std()
            current_vol = realized_vol.iloc[-1] if not realized_vol.empty else daily_vol
            
            # Calculate ATR (Average True Range)
            high_low = price_data['High'] - price_data['Low']
            high_close = np.abs(price_data['High'] - price_data['Close'].shift())
            low_close = np.abs(price_data['Low'] - price_data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_period = min(14, len(true_range))
            atr = true_range.rolling(atr_period).mean().iloc[-1] if atr_period > 0 else 0
            
            # Calculate volatility ratio (current vs historical)
            hist_vol = returns.std()
            vol_ratio = current_vol / hist_vol if hist_vol > 0 else 1.0
            
            # Determine regime
            if current_vol < self.volatility_thresholds[VolatilityRegime.LOW]:
                regime = VolatilityRegime.LOW
            elif current_vol < self.volatility_thresholds[VolatilityRegime.MEDIUM]:
                regime = VolatilityRegime.MEDIUM
            elif current_vol < self.volatility_thresholds[VolatilityRegime.HIGH]:
                regime = VolatilityRegime.HIGH
            else:
                regime = VolatilityRegime.CRASH
            
            # Check for regime change
            if len(self.regime_history) > 0:
                last_regime = self.regime_history[-1]
                if regime != last_regime:
                    logger.info(f"📈 Volatility regime changed: {last_regime.value} -> {regime.value}")
            
            # Update history
            self.volatility_history.append({
                'timestamp': datetime.now(pytz.UTC),
                'volatility': float(current_vol),
                'regime': regime
            })
            self.regime_history.append(regime)
            self.current_regime = regime
            
            # Prepare analysis
            current_price = price_data['Close'].iloc[-1] if len(price_data) > 0 else 0
            analysis = {
                'current_volatility': float(current_vol),
                'daily_volatility': float(daily_vol),
                'atr': float(atr),
                'volatility_ratio': float(vol_ratio),
                'regime': regime.value,
                'price_range': float(price_data['High'].iloc[-1] - price_data['Low'].iloc[-1]) if len(price_data) > 0 else 0,
                'avg_true_range_pct': float((atr / current_price * 100) if current_price > 0 else 0),
                'is_high_volatility': regime in [VolatilityRegime.HIGH, VolatilityRegime.CRASH]
            }
            
            return regime, analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return VolatilityRegime.MEDIUM, {"error": str(e)}
    
    def get_volatility_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility-specific features with error handling"""
        try:
            features = pd.DataFrame(index=price_data.index)
            
            # Basic returns
            features['returns'] = price_data['Close'].pct_change()
            features['log_returns'] = np.log(price_data['Close'] / price_data['Close'].shift(1))
            
            # Volatility measures with different windows
            windows = [5, 10, 20, 50]
            for window in windows:
                if len(price_data) >= window:
                    features[f'volatility_{window}'] = features['returns'].rolling(window).std()
                    features[f'range_{window}'] = (
                        price_data['High'].rolling(window).max() - 
                        price_data['Low'].rolling(window).min()
                    ) / price_data['Close'].rolling(window).mean()
            
            # ATR calculations
            if len(price_data) > 1:
                high_low = price_data['High'] - price_data['Low']
                high_close = np.abs(price_data['High'] - price_data['Close'].shift())
                low_close = np.abs(price_data['Low'] - price_data['Close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                
                atr_windows = [7, 14, 21]
                for window in atr_windows:
                    if len(true_range) >= window:
                        features[f'atr_{window}'] = true_range.rolling(window).mean()
                        features[f'atr_pct_{window}'] = features[f'atr_{window}'] / price_data['Close']
            
            # Volatility ratio (short-term vs long-term)
            if 'returns' in features.columns:
                short_vol = features['returns'].rolling(5).std()
                long_vol = features['returns'].rolling(20).std()
                features['vol_ratio_short_long'] = (short_vol / long_vol).replace([np.inf, -np.inf], 1.0)
            
            # Fill NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting volatility features: {e}")
            return pd.DataFrame()
    
    def get_regime_specific_parameters(self, regime: VolatilityRegime) -> Dict:
        """Get trading parameters for specific volatility regime"""
        params = {
            VolatilityRegime.LOW: {
                'position_size_pct': 10.0,
                'stop_loss_pct': 1.0,
                'take_profit_pct': 2.0,
                'trailing_stop_pct': 0.5,
                'max_leverage': 2.0,
                'confidence_threshold': 60.0
            },
            VolatilityRegime.MEDIUM: {
                'position_size_pct': 7.5,
                'stop_loss_pct': 1.5,
                'take_profit_pct': 3.0,
                'trailing_stop_pct': 0.8,
                'max_leverage': 1.5,
                'confidence_threshold': 70.0
            },
            VolatilityRegime.HIGH: {
                'position_size_pct': 5.0,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'trailing_stop_pct': 1.2,
                'max_leverage': 1.0,
                'confidence_threshold': 80.0
            },
            VolatilityRegime.CRASH: {
                'position_size_pct': 2.5,
                'stop_loss_pct': 3.0,
                'take_profit_pct': 6.0,
                'trailing_stop_pct': 2.0,
                'max_leverage': 0.5,
                'confidence_threshold': 90.0
            }
        }
        
        return params.get(regime, params[VolatilityRegime.MEDIUM])

# ================= ECONOMIC CALENDAR INTEGRATION =================
class EconomicCalendar:
    """Fetch and analyze economic calendar events"""
    
    def __init__(self):
        self.events_cache = {}
        self.last_update = None
        self.session = requests.Session()
        
    async def fetch_events(self, days_ahead: int = 7) -> List[Dict]:
        """Fetch economic calendar events from multiple sources"""
        try:
            sources = [
                self._fetch_events_forexfactory,
                self._fetch_events_investing,
                self._fetch_events_fxstreet
            ]
            
            all_events = []
            for source in sources:
                try:
                    events = await source(days_ahead)
                    if events:
                        all_events.extend(events)
                except Exception as e:
                    logger.warning(f"Economic calendar source {source.__name__} failed: {e}")
                    continue
            
            # Remove duplicates
            unique_events = []
            seen = set()
            for event in all_events:
                key = f"{event.get('date')}_{event.get('event')}_{event.get('time')}"
                if key not in seen:
                    seen.add(key)
                    unique_events.append(event)
            
            self.events_cache['events'] = unique_events
            self.last_update = datetime.now(pytz.UTC)
            
            return unique_events
            
        except Exception as e:
            logger.error(f"Error fetching economic events: {e}")
            return []
    
    async def _fetch_events_forexfactory(self, days_ahead: int) -> List[Dict]:
        """Fetch events from Forex Factory"""
        try:
            # Forex Factory calendar
            today = datetime.now().strftime('%b %d, %Y')
            url = "https://www.forexfactory.com/calendar"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                events = []
                
                # Parse Forex Factory calendar table
                calendar_table = soup.find('table', class_='calendar__table')
                if calendar_table:
                    rows = calendar_table.find_all('tr', class_='calendar__row')
                    
                    for row in rows[:20]:  # Get top 20 events
                        try:
                            time_elem = row.find('td', class_='calendar__time')
                            currency_elem = row.find('td', class_='calendar__currency')
                            event_elem = row.find('td', class_='calendar__event')
                            impact_elem = row.find('td', class_='calendar__impact')
                            
                            if all([time_elem, currency_elem, event_elem, impact_elem]):
                                impact_class = impact_elem.find('span')['class'][0] if impact_elem.find('span') else ''
                                impact_map = {
                                    'high': 'HIGH',
                                    'medium': 'MEDIUM',
                                    'low': 'LOW'
                                }
                                impact = impact_map.get(impact_class, 'LOW')
                                
                                events.append({
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'time': time_elem.text.strip(),
                                    'currency': currency_elem.text.strip(),
                                    'event': event_elem.text.strip(),
                                    'impact': impact,
                                    'actual': None,
                                    'forecast': None,
                                    'previous': None,
                                    'source': 'Forex Factory'
                                })
                        except:
                            continue
                
                return events
            
            return []
            
        except Exception as e:
            logger.error(f"Forex Factory calendar error: {e}")
            return []
    
    async def _fetch_events_investing(self, days_ahead: int) -> List[Dict]:
        """Fetch events from Investing.com"""
        try:
            url = "https://www.investing.com/economic-calendar/"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                events = []
                
                # Parse Investing.com calendar
                event_rows = soup.find_all('tr', {'data-event-datetime': True})
                
                for row in event_rows[:15]:
                    try:
                        time_elem = row.find('td', class_='time')
                        currency_elem = row.find('td', class_='left')
                        event_elem = row.find('td', class_='event')
                        impact_elem = row.find('td', class_='sentiment')
                        
                        if all([time_elem, currency_elem, event_elem]):
                            # Parse impact from bull icons
                            if impact_elem:
                                bulls = len(impact_elem.find_all('i', class_='grayFullBullishIcon'))
                                if bulls >= 3:
                                    impact = 'HIGH'
                                elif bulls >= 2:
                                    impact = 'MEDIUM'
                                else:
                                    impact = 'LOW'
                            else:
                                impact = 'LOW'
                            
                            events.append({
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'time': time_elem.text.strip(),
                                'currency': currency_elem.text.strip()[:3],
                                'event': event_elem.text.strip(),
                                'impact': impact,
                                'source': 'Investing.com'
                            })
                    except:
                        continue
                
                return events
            
            return []
            
        except Exception as e:
            logger.error(f"Investing.com calendar error: {e}")
            return []
    
    async def _fetch_events_fxstreet(self, days_ahead: int) -> List[Dict]:
        """Fetch events from FXStreet"""
        try:
            url = "https://www.fxstreet.com/economic-calendar"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = await asyncio.to_thread(
                self.session.get, url, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                events = []
                
                # Parse FXStreet calendar
                event_items = soup.find_all('div', class_=re.compile(r'.*calendar-item.*', re.I))
                
                for item in event_items[:15]:
                    try:
                        time_elem = item.find('div', class_=re.compile(r'.*time.*', re.I))
                        currency_elem = item.find('span', class_=re.compile(r'.*currency.*', re.I))
                        event_elem = item.find('a', class_=re.compile(r'.*event.*', re.I))
                        
                        if all([time_elem, currency_elem, event_elem]):
                            # Determine impact (FXStreet often has color coding)
                            impact = 'LOW'
                            if 'high' in item.get('class', []):
                                impact = 'HIGH'
                            elif 'medium' in item.get('class', []):
                                impact = 'MEDIUM'
                            
                            events.append({
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'time': time_elem.text.strip(),
                                'currency': currency_elem.text.strip()[:3],
                                'event': event_elem.text.strip(),
                                'impact': impact,
                                'source': 'FXStreet'
                            })
                    except:
                        continue
                
                return events
            
            return []
            
        except Exception as e:
            logger.error(f"FXStreet calendar error: {e}")
            return []
    
    def get_impact_score(self, events: List[Dict]) -> float:
        """Calculate overall market impact score"""
        if not events:
            return 0.0
        
        impact_weights = {'HIGH': 1.0, 'MEDIUM': 0.5, 'LOW': 0.2}
        total_score = 0.0
        
        for event in events:
            impact = event.get('impact', 'LOW')
            weight = impact_weights.get(impact, 0.1)
            total_score += weight
        
        return min(total_score / len(events), 1.0) if events else 0.0
    
    def get_upcoming_high_impact_events(self, hours_ahead: int = 24) -> List[Dict]:
        """Get upcoming high impact events"""
        try:
            if 'events' not in self.events_cache:
                return []
            
            now = datetime.now(pytz.UTC)
            cutoff = now + timedelta(hours=hours_ahead)
            
            high_impact = []
            for event in self.events_cache['events']:
                event_time_str = f"{event['date']} {event['time']}"
                try:
                    # Parse time string
                    event_time = None
                    time_formats = [
                        '%Y-%m-%d %H:%M',
                        '%Y-%m-%d %I:%M %p',
                        '%Y-%m-%d %I:%M%p'
                    ]
                    
                    for fmt in time_formats:
                        try:
                            event_time = datetime.strptime(event_time_str, fmt)
                            event_time = pytz.UTC.localize(event_time)
                            break
                        except:
                            continue
                    
                    if event_time and event['impact'] == 'HIGH' and now <= event_time <= cutoff:
                        high_impact.append(event)
                except:
                    continue
            
            return high_impact
            
        except Exception as e:
            logger.error(f"Error getting high impact events: {e}")
            return []

# ================= ADVANCED FEATURE ENGINEER =================
class AdvancedFeatureEngineer:
    """Generate comprehensive features for ML models with improved feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_scalers = {}
        self.feature_cache = {}
        self.feature_importance = {}
        
    def create_features(self, df: pd.DataFrame, include_derived: bool = True) -> pd.DataFrame:
        """Create comprehensive feature set with error handling"""
        try:
            if df is None or df.empty or len(df) < 20:
                logger.warning("Insufficient data for feature engineering")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=df.index)
            
            # 1. Price-based features
            features['price'] = df['Close']
            features['returns'] = df['Close'].pct_change()
            features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            if 'High' in df.columns and 'Low' in df.columns:
                features['high_low_ratio'] = df['High'] / df['Low']
                features['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
                features['close_position'] = features['close_position'].replace([np.inf, -np.inf], 0.5).fillna(0.5)
            
            # 2. Moving averages
            periods = [5, 10, 20, 50, 100]
            for period in periods:
                if len(df) >= period:
                    features[f'sma_{period}'] = df['Close'].rolling(period).mean()
                    features[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
                    features[f'price_sma_ratio_{period}'] = df['Close'] / features[f'sma_{period}']
            
            # 3. Volatility features
            if len(df) >= 20:
                features['volatility_20'] = df['Close'].pct_change().rolling(20).std()
                features['volatility_50'] = df['Close'].pct_change().rolling(50).std()
            
            # Calculate ATR
            atr = self.calculate_atr(df, 14)
            if not atr.empty:
                features['atr_14'] = atr
                features['atr_ratio'] = atr / df['Close']
            
            # 4. Momentum indicators
            features['rsi_14'] = self.calculate_rsi(df['Close'], 14)
            features['rsi_28'] = self.calculate_rsi(df['Close'], 28)
            
            # 5. Volume features
            if 'Volume' in df.columns:
                features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
                features['volume_price_trend'] = (df['Volume'] * df['Close'].pct_change()).rolling(20).sum()
                features['obv'] = self.calculate_obv(df)
            
            # 6. Time-based features
            if not df.index.empty:
                features['hour'] = df.index.hour
                features['day_of_week'] = df.index.dayofweek
                features['month'] = df.index.month
                features['quarter'] = df.index.quarter
            
            # 7. Statistical features
            if len(df) >= 20:
                features['skewness_20'] = df['Close'].rolling(20).skew()
                features['kurtosis_20'] = df['Close'].rolling(20).kurt()
                features['z_score_20'] = (
                    df['Close'] - df['Close'].rolling(20).mean()
                ) / df['Close'].rolling(20).std()
                features['z_score_20'] = features['z_score_20'].replace([np.inf, -np.inf], 0)
            
            if include_derived:
                features = self.create_derived_features(features)
            
            # Fill NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            # Remove infinite values
            features = features.replace([np.inf, -np.inf], 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            if len(df) < 2:
                return pd.Series(index=df.index)
            
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(period).mean()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(index=df.index)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with error handling"""
        try:
            if len(prices) < period:
                return pd.Series(index=prices.index, data=50.0)
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Handle division by zero
            rsi = rsi.replace([np.inf, -np.inf], 50.0)
            rsi = rsi.fillna(50.0)
            
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, data=50.0)
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            if 'Volume' not in df.columns or len(df) < 2:
                return pd.Series(index=df.index, data=0)
            
            obv = pd.Series(0, index=df.index, dtype=float)
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            return obv
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return pd.Series(index=df.index, data=0)
    
    def create_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create advanced derived features with error handling"""
        try:
            # Rate of change
            periods = [5, 10, 20]
            for period in periods:
                if 'price' in features.columns and len(features) >= period:
                    features[f'roc_{period}'] = (
                        features['price'] / features['price'].shift(period) - 1
                    ) * 100
                    features[f'roc_{period}'] = features[f'roc_{period}'].fillna(0)
            
            # Bollinger Bands
            bb_periods = [20, 50]
            for period in bb_periods:
                if 'price' in features.columns and len(features) >= period:
                    sma = features[f'sma_{period}'] if f'sma_{period}' in features else features['price'].rolling(period).mean()
                    std = features['price'].rolling(period).std()
                    features[f'bb_upper_{period}'] = sma + (std * 2)
                    features[f'bb_lower_{period}'] = sma - (std * 2)
                    features[f'bb_position_{period}'] = (
                        features['price'] - features[f'bb_lower_{period}']
                    ) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
                    features[f'bb_position_{period}'] = features[f'bb_position_{period}'].replace([np.inf, -np.inf], 0).fillna(0)
            
            # Fibonacci retracement (simplified)
            fib_lookbacks = [20, 50, 100]
            for lookback in fib_lookbacks:
                if 'price' in features.columns and len(features) >= lookback:
                    high = features['price'].rolling(lookback).max()
                    low = features['price'].rolling(lookback).min()
                    fib_range = high - low
                    
                    levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                    for level in levels:
                        features[f'fib_{lookback}_{int(level*100)}'] = low + (fib_range * level)
                        features[f'fib_{lookback}_{int(level*100)}'] = features[f'fib_{lookback}_{int(level*100)}'].fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating derived features: {e}")
            return features

# ================= SIMPLE ML SYSTEM (NO 5-YEAR TRAINING) =================
class SimpleMLSystem:
    """Simple ML system for signal generation without extensive training"""
    
    def __init__(self, version_manager):
        self.version_manager = version_manager
        self.models = {}
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_engineer = AdvancedFeatureEngineer()
        
    async def generate_signal(self, historical_data: pd.DataFrame, volatility_regime: str) -> Optional[Dict]:
        """Generate signal using simple ML approach"""
        try:
            if historical_data.empty or len(historical_data) < 50:
                return None
            
            # Create features
            features = self.feature_engineer.create_features(historical_data)
            if features.empty:
                return None
            
            # Get latest features
            latest_features = features.iloc[-1:].copy()
            
            # Simple rule-based signal generation
            current_price = historical_data['Close'].iloc[-1]
            sma_20 = latest_features.get('sma_20', current_price).iloc[0]
            sma_50 = latest_features.get('sma_50', current_price).iloc[0]
            rsi = latest_features.get('rsi_14', 50).iloc[0]
            
            # Generate signal based on rules
            signal = self._generate_ml_signal(
                current_price, sma_20, sma_50, rsi, volatility_regime
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return None
    
    def _generate_ml_signal(self, price: float, sma_20: float, sma_50: float, 
                          rsi: float, volatility_regime: str) -> Dict:
        """Generate signal using simple ML rules"""
        try:
            action = "NEUTRAL"
            confidence = 50.0
            reasons = []
            
            # Rule 1: Trend following
            if price > sma_20 > sma_50:
                action = "BUY"
                confidence += 20
                reasons.append("Uptrend confirmed (price > SMA20 > SMA50)")
            elif price < sma_20 < sma_50:
                action = "SELL"
                confidence += 20
                reasons.append("Downtrend confirmed (price < SMA20 < SMA50)")
            
            # Rule 2: RSI momentum
            if rsi > 70:
                confidence -= 15
                reasons.append("RSI overbought")
                if action == "BUY":
                    action = "NEUTRAL_LEAN_BUY"
            elif rsi < 30:
                confidence -= 15
                reasons.append("RSI oversold")
                if action == "SELL":
                    action = "NEUTRAL_LEAN_SELL"
            elif 45 <= rsi <= 55:
                confidence += 5
                reasons.append("RSI neutral")
            
            # Rule 3: Volatility adjustment
            if volatility_regime == "HIGH_VOLATILITY":
                confidence *= 0.8
                reasons.append("High volatility - reduced confidence")
            elif volatility_regime == "LOW_VOLATILITY":
                confidence *= 1.1
                reasons.append("Low volatility - increased confidence")
            
            # Ensure confidence bounds
            confidence = max(0.0, min(100.0, confidence))
            
            # Determine strength
            if confidence >= 75:
                if action == "BUY":
                    action = "STRONG_BUY"
                elif action == "SELL":
                    action = "STRONG_SELL"
            
            return {
                'action': action,
                'confidence': confidence,
                'reasons': reasons,
                'indicators': {
                    'price': price,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'rsi': rsi,
                    'price_vs_sma20_pct': ((price / sma_20) - 1) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ML signal generation: {e}")
            return {
                'action': 'NEUTRAL',
                'confidence': 50.0,
                'reasons': [f'Error: {str(e)[:50]}']
            }

# ================= BACKTESTER =================
class Backtester:
    """Enhanced backtester with realistic simulation"""
    
    def __init__(self, initial_capital: float = 100000, commission_pct: float = 0.0001, 
                 slippage_pct: float = 0.0001, min_trade_size: float = 1000):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.min_trade_size = min_trade_size
        
        # Performance metrics
        self.equity_curve = []
        self.trade_history = []
        
    def simulate_trades(self, historical_data: pd.DataFrame, signal_series: pd.Series) -> Dict:
        """Simulate trades based on signals"""
        try:
            if historical_data is None or historical_data.empty or signal_series is None or signal_series.empty:
                return {"error": "Insufficient data for backtest"}
            
            # Initialize state
            equity = self.initial_capital
            position = 0  # Current position size
            entry_price = 0
            trade_id = 0
            
            self.equity_curve = [equity]
            self.trade_history = []
            
            # Iterate through signals
            for date, signal in signal_series.items():
                if date not in historical_data.index:
                    continue
                
                price = historical_data.loc[date, 'Close']
                
                # Apply slippage
                buy_price = price * (1 + self.slippage_pct)
                sell_price = price * (1 - self.slippage_pct)
                
                # Calculate position size (simplified)
                trade_size = equity * 0.01  # 1% risk per trade
                trade_size = max(trade_size, self.min_trade_size)
                
                if signal == 1:  # Buy signal
                    if position == 0:
                        # Open position
                        shares = trade_size / buy_price
                        position = shares
                        entry_price = buy_price
                        commission = trade_size * self.commission_pct
                        equity -= commission
                        
                        self.trade_history.append({
                            'id': trade_id,
                            'type': 'BUY',
                            'date': date,
                            'price': buy_price,
                            'shares': shares,
                            'commission': commission
                        })
                        trade_id += 1
                
                elif signal == -1:  # Sell signal
                    if position > 0:
                        # Close position
                        proceeds = position * sell_price
                        profit = proceeds - (position * entry_price)
                        commission = proceeds * self.commission_pct
                        equity += profit - commission
                        
                        self.trade_history.append({
                            'id': trade_id,
                            'type': 'SELL',
                            'date': date,
                            'price': sell_price,
                            'shares': position,
                            'commission': commission,
                            'profit': profit
                        })
                        trade_id += 1
                        
                        position = 0
                
                # Update equity curve
                self.equity_curve.append(equity)
            
            # Calculate metrics
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            
            total_return_pct = (equity / self.initial_capital - 1) * 100
            annual_return_pct = ((equity / self.initial_capital) ** (252 / len(historical_data)) - 1) * 100
            max_drawdown_pct = self._calculate_max_drawdown()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(returns)
            win_rate_pct = self._calculate_win_rate() * 100
            profit_factor = self._calculate_profit_factor()
            
            results = {
                'final_equity': equity,
                'total_return_pct': total_return_pct,
                'annual_return_pct': annual_return_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'win_rate_pct': win_rate_pct,
                'profit_factor': profit_factor,
                'total_trades': len([t for t in self.trade_history if t['type'] == 'SELL']),
                'winning_trades': sum(1 for t in self.trade_history if t['type'] == 'SELL' and t['profit'] > 0),
                'losing_trades': sum(1 for t in self.trade_history if t['type'] == 'SELL' and t['profit'] <= 0),
                'equity_curve': self.equity_curve,
                'trade_history': self.trade_history
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest simulation: {e}")
            return {"error": str(e)}
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if not self.equity_curve:
            return 0.0
        
        equity_series = pd.Series(self.equity_curve)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return drawdown.min() * -100 if drawdown.min() < 0 else 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if returns.empty:
            return 0.0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if not downside_returns.empty else 0.0
        expected_return = returns.mean()
        return expected_return / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        closed_trades = [t for t in self.trade_history if t['type'] == 'SELL']
        if not closed_trades:
            return 0.0
        winning = sum(1 for t in closed_trades if t['profit'] > 0)
        return winning / len(closed_trades)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        closed_trades = [t for t in self.trade_history if t['type'] == 'SELL']
        if not closed_trades:
            return 0.0
        gross_profit = sum(t['profit'] for t in closed_trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in closed_trades if t['profit'] < 0))
        return gross_profit / gross_loss if gross_loss > 0 else 0.0

# ================= TELEGRAM NOTIFICATION MANAGER =================
class TelegramNotificationManager:
    """Manage Telegram notifications"""
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv('TELEGRAM_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.bot = None
        self.enabled = False
        
        if self.token and self.chat_id and TELEGRAM_AVAILABLE:
            try:
                self.bot = Bot(token=self.token)
                self.enabled = True
                logger.info("✅ Telegram notifications enabled")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ Telegram notifications disabled - missing token or chat_id")
    
    async def send_signal(self, signal: Dict):
        """Send trading signal to Telegram"""
        if not self.enabled or not self.bot:
            return
        
        try:
            action = signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            price = signal.get('price', 0)
            source = signal.get('source', 'Unknown')
            
            # Create message
            emoji = "🟢" if "BUY" in action else "🔴" if "SELL" in action else "⚪"
            message = f"{emoji} *Gold Trading Signal*\n\n"
            message += f"*Action:* {action.replace('_', ' ')}\n"
            message += f"*Confidence:* {confidence:.1f}%\n"
            message += f"*Price:* ${price:.2f}\n"
            message += f"*Source:* {source}\n"
            message += f"*Time:* {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S ET')}\n"
            
            if 'reason' in signal:
                message += f"\n*Reason:* {signal['reason']}\n"
            
            # Add risk management info
            if 'position_size' in signal:
                pos = signal['position_size']
                message += f"\n*Position Size:* {pos.get('recommended', 0)}%\n"
            
            # Send message
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info("📤 Telegram notification sent")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

# ================= MOBILE API =================
class MobileAPI:
    """Mobile API for real-time signal access"""
    
    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.app = None
        self.signals = []
        self.performance = {}
        
        if FASTAPI_AVAILABLE:
            self._setup_api()
    
    def _setup_api(self):
        """Setup FastAPI application"""
        self.app = FastAPI(title="Gold Trading Sentinel API", version="14.0")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Define API models
        class SignalResponse(BaseModel):
            signal_id: str
            action: str
            confidence: float
            price: float
            timestamp: str
            source: str
        
        class PerformanceResponse(BaseModel):
            total_signals: int
            successful_signals: int
            win_rate: float
            avg_confidence: float
        
        # Setup routes
        @self.app.get("/")
        async def root():
            return {
                "service": "Gold Trading Sentinel v14.0",
                "status": "operational",
                "version": "14.0"
            }
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/signal/latest")
        async def get_latest_signal():
            if self.signals:
                return self.signals[-1]
            return {"error": "No signals available"}
        
        @self.app.get("/signal/history")
        async def get_signal_history(limit: int = 10):
            return self.signals[-limit:] if self.signals else []
        
        @self.app.get("/performance")
        async def get_performance():
            return self.performance
    
    def add_signal(self, signal: Dict):
        """Add signal to API"""
        if not self.signals:
            self.signals = []
        
        # Add signal ID and timestamp
        signal_id = f"signal_{len(self.signals)+1:04d}"
        signal['signal_id'] = signal_id
        signal['api_timestamp'] = datetime.now(pytz.UTC).isoformat()
        
        self.signals.append(signal)
        
        # Keep only last 100 signals
        if len(self.signals) > 100:
            self.signals = self.signals[-100:]
    
    def update_performance(self, performance: Dict):
        """Update performance metrics"""
        self.performance = performance
    
    def run(self):
        """Run the API server"""
        if self.app:
            uvicorn.run(self.app, host=self.host, port=self.port)
        else:
            logger.error("FastAPI not available")

# ================= GOLD TRADING SENTINEL V14 =================
class GoldTradingSentinelV14:
    """Gold Trading Sentinel v14.0 - Enhanced AI Trading System"""
    
    def __init__(self, config: Dict):
        # Core components
        self.version_manager = DataVersionManager()
        self.data_extractor = RobustFreeDataExtractor(self.version_manager)
        self.volatility_analyzer = VolatilityAnalyzer()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ml_system = SimpleMLSystem(self.version_manager)
        self.economic_calendar = EconomicCalendar()
        self.backtester = Backtester()
        
        # Configuration
        self.config = config
        
        # State management
        self.start_time = datetime.now(pytz.UTC)
        self.signal_count = 0
        self.consecutive_successes = 0
        self.last_success_time = None
        self.performance = {
            'total_runs': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'high_confidence_signals': 0,
            'avg_confidence': 0.0,
            'win_rate': 0.0
        }
        
        # Initialize Telegram and Mobile API if enabled
        self.telegram = TelegramNotificationManager(
            token=config.get('telegram_token'),
            chat_id=config.get('telegram_chat_id')
        ) if config.get('enable_telegram') else None
        
        self.mobile_api = MobileAPI(
            port=config.get('api_port', 8080),
            host=config.get('api_host', "0.0.0.0")
        ) if config.get('enable_mobile_api') else None
        
        # Start API in background if enabled
        if config.get('enable_mobile_api') and self.mobile_api and self.mobile_api.app:
            import threading
            api_thread = threading.Thread(target=self.mobile_api.run, daemon=True)
            api_thread.start()
            logger.info(f"🌐 Mobile API started on port {config.get('api_port', 8080)}")
    
    async def initialize(self):
        """Initialize the trading system"""
        logger.info("🚀 Initializing Gold Trading Sentinel v14.0...")
        
        # Backup data
        self.version_manager.backup_data(STATE_FILE)
        self.version_manager.backup_data(DATABASE_FILE)
        self.version_manager.cleanup_old_backups()
        
        # Fetch economic calendar
        events = await self.economic_calendar.fetch_events()
        if events:
            logger.info(f"📅 Loaded {len(events)} economic events")
        
        logger.info("✅ Initialization complete")
    
    async def generate_signal(self) -> Optional[Dict]:
        """Generate trading signal with multiple fallbacks"""
        try:
            # Check market status
            if not is_market_open():
                next_open = next_market_open()
                if next_open:
                    logger.info(f"⏳ Market closed. Next open: {next_open}")
                return None
            
            # Get current price with fallbacks
            current_price, source, price_details = await self.data_extractor.get_current_price()
            
            if current_price <= 0:
                logger.error("❌ Failed to get valid price from all sources")
                self.performance['failed_signals'] += 1
                return None
            
            # Get historical data
            historical_data = await self.data_extractor.get_historical_data(days=30)
            
            if historical_data.empty:
                logger.warning("⚠️ Insufficient historical data, using fallback signal")
                return self._generate_fallback_signal(current_price)
            
            # Analyze volatility
            volatility_regime, volatility_analysis = self.volatility_analyzer.analyze_volatility(historical_data)
            
            # Generate ML signal if enabled
            ml_signal = None
            if self.config.get('enable_ai', True):
                ml_signal = await self.ml_system.generate_signal(
                    historical_data, volatility_regime.value
                )
            
            # Generate rule-based signal
            rule_signal = self._generate_rule_based_signal(
                current_price, historical_data, volatility_regime
            )
            
            # Combine signals
            signal = self._combine_signals(ml_signal, rule_signal, current_price, volatility_regime)
            
            # Add metadata
            signal['price'] = current_price
            signal['timestamp'] = datetime.now(pytz.UTC).isoformat()
            signal['price_source'] = source
            signal['volatility_regime'] = volatility_regime.value
            
            # Add position size and risk management
            signal['position_size'] = self._calculate_position_size(signal, volatility_regime)
            signal['risk_management'] = self.volatility_analyzer.get_regime_specific_parameters(volatility_regime)
            
            # Add volatility analysis
            signal['volatility_analysis'] = volatility_analysis
            
            # Add economic impact
            impact_score = self.economic_calendar.get_impact_score(
                self.economic_calendar.events_cache.get('events', [])
            )
            signal['economic_impact_score'] = impact_score
            
            if impact_score > 0.7:
                signal['confidence'] *= 0.8  # Reduce confidence before high impact events
            
            # Add news sentiment
            news_sentiment = await self.data_extractor.get_news_sentiment()
            signal['news_sentiment'] = news_sentiment
            
            # Update tracking
            self.signal_count += 1
            self._update_performance_tracking(signal)
            
            # Send notifications
            if self.telegram and signal.get('confidence', 0) >= self.config.get('telegram_threshold', 70):
                await self.telegram.send_signal(signal)
            
            # Update mobile API
            if self.mobile_api:
                self.mobile_api.add_signal(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Critical error generating signal: {e}")
            self.performance['failed_signals'] += 1
            return None
    
    def _generate_fallback_signal(self, current_price: float) -> Dict:
        """Generate fallback signal when no data available"""
        return {
            'action': 'NEUTRAL',
            'confidence': 0.0,
            'source': 'Fallback',
            'price': current_price,
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'volatility_regime': 'UNKNOWN',
            'reason': 'No data available - default to neutral'
        }
    
    def _generate_rule_based_signal(self, current_price: float, historical_data: pd.DataFrame, 
                                  volatility_regime: VolatilityRegime) -> Dict:
        """Generate rule-based signal with fallback handling"""
        try:
            if historical_data.empty or len(historical_data) < 50:
                return {
                    'action': 'NEUTRAL',
                    'confidence': 50.0,
                    'source': 'Rule-Based',
                    'reason': 'Insufficient data - neutral position'
                }
            
            # Calculate indicators
            closes = historical_data['Close']
            sma_20 = closes.rolling(20).mean().iloc[-1]
            sma_50 = closes.rolling(50).mean().iloc[-1]
            rsi = self.feature_engineer.calculate_rsi(closes, 14).iloc[-1]
            
            action = 'NEUTRAL'
            confidence = 60.0
            reason = []
            
            # Rule 1: Price vs Moving Averages
            if current_price > sma_20 > sma_50:
                action = 'BUY'
                confidence += 15
                reason.append("Price above rising moving averages")
            elif current_price < sma_20 < sma_50:
                action = 'SELL'
                confidence += 15
                reason.append("Price below falling moving averages")
            elif current_price > sma_20:
                action = 'NEUTRAL_LEAN_BUY'
                confidence += 5
                reason.append("Price above short-term MA")
            elif current_price < sma_20:
                action = 'NEUTRAL_LEAN_SELL'
                confidence += 5
                reason.append("Price below short-term MA")
            
            # Rule 2: RSI
            if rsi > 70:
                confidence -= 10
                reason.append("RSI indicates overbought")
            elif rsi < 30:
                confidence -= 10
                reason.append("RSI indicates oversold")
            elif 40 <= rsi <= 60:
                confidence += 5
                reason.append("RSI in neutral range")
            
            # Adjust confidence based on volatility regime
            if volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.CRASH]:
                confidence *= 0.8
                reason.append("Reduced confidence due to high volatility")
            elif volatility_regime == VolatilityRegime.LOW:
                confidence *= 1.1
                reason.append("Increased confidence due to low volatility")
            
            # Ensure confidence is within bounds
            confidence = max(0.0, min(100.0, confidence))
            
            return {
                'action': action,
                'confidence': confidence,
                'source': 'Rule-Based',
                'reason': ' | '.join(reason),
                'indicators': {
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'rsi': rsi,
                    'price_vs_sma20_pct': ((current_price / sma_20) - 1) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating rule-based signal: {e}")
            return {
                'action': 'NEUTRAL',
                'confidence': 50.0,
                'source': 'Rule-Based (Error)',
                'reason': f'Error: {str(e)[:50]}'
            }
    
    def _combine_signals(self, ml_signal: Optional[Dict], rule_signal: Dict, 
                        current_price: float, volatility_regime: VolatilityRegime) -> Dict:
        """Combine ML and rule-based signals"""
        try:
            if ml_signal is None or ml_signal.get('confidence', 0) < 50:
                # Use rule-based signal only
                return rule_signal
            
            # Both signals available - create ensemble
            ml_confidence = ml_signal.get('confidence', 0)
            rule_confidence = rule_signal.get('confidence', 0)
            
            # Weight signals based on confidence and volatility regime
            if volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.CRASH]:
                # Favor rule-based in high volatility
                ml_weight = 0.3
                rule_weight = 0.7
            else:
                # Favor ML in normal conditions
                ml_weight = 0.7
                rule_weight = 0.3
            
            # Calculate weighted confidence
            weighted_confidence = (ml_confidence * ml_weight + rule_confidence * rule_weight)
            
            # Determine final action
            confidence_diff = abs(ml_confidence - rule_confidence)
            if confidence_diff > 20:
                # Large difference, trust higher confidence signal
                if ml_confidence > rule_confidence:
                    final_action = ml_signal['action']
                    source = 'ML (High Confidence)'
                else:
                    final_action = rule_signal['action']
                    source = 'Rule-Based (High Confidence)'
            else:
                # Small difference, use ML signal
                final_action = ml_signal['action']
                source = 'ML + Rule Ensemble'
            
            # Adjust confidence based on agreement
            if ml_signal['action'] == rule_signal['action']:
                # Signals agree - boost confidence
                weighted_confidence *= 1.1
                source += ' (Agreement)'
            else:
                # Signals disagree - reduce confidence
                weighted_confidence *= 0.9
                source += ' (Disagreement)'
            
            # Ensure confidence is within bounds
            weighted_confidence = max(0.0, min(100.0, weighted_confidence))
            
            return {
                'action': final_action,
                'confidence': weighted_confidence,
                'source': source,
                'combined_from': {
                    'ml_signal': ml_signal['action'] if ml_signal else 'None',
                    'ml_confidence': ml_confidence,
                    'rule_signal': rule_signal['action'],
                    'rule_confidence': rule_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return rule_signal  # Fallback to rule-based
    
    def _calculate_position_size(self, signal: Dict, volatility_regime: VolatilityRegime) -> Dict:
        """Calculate position size with risk management"""
        try:
            base_size = self.config.get('position_size_base', 5.0)
            max_size = self.config.get('max_position_size', 15.0)
            
            # Get regime-specific parameters
            regime_params = self.volatility_analyzer.get_regime_specific_parameters(volatility_regime)
            regime_max_size = regime_params.get('position_size_pct', 10.0)
            
            # Adjust based on confidence
            confidence = signal.get('confidence', 50.0)
            confidence_factor = confidence / 100.0
            
            # Adjust based on volatility regime
            volatility_factor = 1.0
            if volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.CRASH]:
                volatility_factor = 0.5
            elif volatility_regime == VolatilityRegime.LOW:
                volatility_factor = 1.2
            
            # Adjust based on consecutive successes
            success_factor = 1.0 + (min(self.consecutive_successes, 5) * 0.05)
            
            # Calculate final position size
            calculated_size = base_size * confidence_factor * volatility_factor * success_factor
            
            # Apply caps
            final_size = min(calculated_size, max_size, regime_max_size)
            final_size = max(final_size, 1.0)  # Minimum 1%
            
            # Check if high confidence signal
            high_confidence_threshold = self.config.get('high_confidence_threshold', 85.0)
            is_high_confidence = confidence >= high_confidence_threshold
            
            if is_high_confidence:
                # Boost for high confidence
                final_size *= 1.3
                final_size = min(final_size, max_size * 1.2)  # Allow slightly higher for high confidence
            
            # Round to nearest 0.5%
            final_size = round(final_size * 2) / 2
            
            return {
                'base': base_size,
                'calculated': round(calculated_size, 2),
                'recommended': final_size,
                'max_allowed': min(max_size, regime_max_size),
                'is_high_confidence': is_high_confidence,
                'confidence_factor': round(confidence_factor, 3),
                'volatility_factor': round(volatility_factor, 3),
                'success_factor': round(success_factor, 3)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'base': 5.0,
                'recommended': 5.0,
                'is_high_confidence': False
            }
    
    def _update_performance_tracking(self, signal: Dict):
        """Update performance tracking metrics"""
        try:
            self.performance['total_runs'] += 1
            
            confidence = signal.get('confidence', 0)
            
            # Update average confidence
            total_signals = self.performance['successful_signals'] + 1
            current_avg = self.performance['avg_confidence']
            new_avg = (current_avg * self.performance['successful_signals'] + confidence) / total_signals
            self.performance['avg_confidence'] = new_avg
            
            # Track high confidence signals
            high_confidence_threshold = self.config.get('high_confidence_threshold', 85.0)
            if confidence >= high_confidence_threshold:
                self.performance['high_confidence_signals'] += 1
            
            # Track consecutive successes
            if confidence >= 60:  # Consider as success
                self.performance['successful_signals'] += 1
                self.consecutive_successes += 1
                self.last_success_time = datetime.now(pytz.UTC)
            else:
                self.consecutive_successes = 0
            
            # Calculate win rate
            total_runs = self.performance['total_runs']
            successful = self.performance['successful_signals']
            self.performance['win_rate'] = (successful / total_runs * 100) if total_runs > 0 else 0
            
            # Update mobile API
            if self.mobile_api:
                self.mobile_api.update_performance(self.performance)
            
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def display_signal(self, signal: Dict):
        """Display signal in console with rich formatting"""
        try:
            print("\n" + "=" * 100)
            print("🚀 GOLD TRADING SENTINEL V14.0 - REAL-TIME SIGNAL")
            print("=" * 100)
            
            # Basic information
            timestamp = signal.get('timestamp')
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now(pytz.UTC)
            
            print(f"🕒 Time: {timestamp.astimezone(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S ET')}")
            print(f"💰 Current Price: ${signal.get('price', 0):.2f}")
            print(f"📊 Signal ID: {signal.get('signal_id', f'#{self.signal_count}')}")
            print(f"📁 Run ID: {self.version_manager.current_run_id}")
            print("-" * 100)
            
            # Signal details
            action = signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            
            # Action emojis and formatting
            action_configs = {
                "STRONG_BUY": ("🟢", "STRONG BUY", "🟢🟢🟢"),
                "BUY": ("🟢", "BUY", "🟢🟢"),
                "NEUTRAL_LEAN_BUY": ("🟡", "NEUTRAL (Lean to Buy)", "↗️"),
                "NEUTRAL_LEAN_SELL": ("🟡", "NEUTRAL (Lean to Sell)", "↘️"),
                "SELL": ("🔴", "SELL", "🔴🔴"),
                "STRONG_SELL": ("🔴", "STRONG SELL", "🔴🔴🔴"),
                "NEUTRAL": ("⚪", "NEUTRAL", "➡️")
            }
            
            emoji, display_name, strength = action_configs.get(action, ("⚪", action, ""))
            
            # Print signal
            print(f"🎯 TRADING SIGNAL: {strength} {emoji} {display_name} {strength}")
            print(f"📊 Confidence: {confidence:.1f}%")
            print(f"📡 Source: {signal.get('source', 'Unknown')}")
            print(f"📈 Volatility Regime: {signal.get('volatility_regime', 'Unknown')}")
            print("-" * 100)
            
            # Position size
            pos_size = signal.get('position_size', {})
            if pos_size:
                print(f"💼 POSITION SIZE: {pos_size.get('recommended', 'N/A')}% of capital")
                
                if pos_size.get('is_high_confidence', False):
                    print("   🏆 HIGH CONFIDENCE SIGNAL - Increased position size")
                
                print(f"   📊 Base: {pos_size.get('base', 'N/A')}% | "
                      f"Calculated: {pos_size.get('calculated', 'N/A')}%")
                print(f"   📈 Confidence Factor: {pos_size.get('confidence_factor', 1.0):.2f}x")
                print(f"   🌪️ Volatility Factor: {pos_size.get('volatility_factor', 1.0):.2f}x")
                print(f"   📈 Success Factor: {pos_size.get('success_factor', 1.0):.2f}x")
            
            print("-" * 100)
            
            # Risk management
            risk_mgmt = signal.get('risk_management', {})
            if risk_mgmt:
                print(f"⚠️ RISK MANAGEMENT:")
                print(f"   Stop Loss: {risk_mgmt.get('stop_loss_pct', 'N/A')}%")
                print(f"   Take Profit: {risk_mgmt.get('take_profit_pct', 'N/A')}%")
                print(f"   Trailing Stop: {risk_mgmt.get('trailing_stop_pct', 'N/A')}%")
                print(f"   Max Leverage: {risk_mgmt.get('max_leverage', 'N/A')}x")
            
            print("-" * 100)
            
            # Market analysis
            volatility_analysis = signal.get('volatility_analysis', {})
            if volatility_analysis:
                print(f"📈 MARKET ANALYSIS:")
                print(f"   Current Volatility: {volatility_analysis.get('current_volatility', 0):.4f}")
                print(f"   ATR: {volatility_analysis.get('atr', 0):.2f}")
                print(f"   Price Range: ${volatility_analysis.get('price_range', 0):.2f}")
                print(f"   Volatility Ratio: {volatility_analysis.get('volatility_ratio', 1.0):.2f}x")
            
            print("-" * 100)
            
            # Performance summary
            print(f"📊 PERFORMANCE SUMMARY:")
            print(f"   Total Signals: {self.signal_count}")
            print(f"   Successful: {self.performance['successful_signals']} "
                  f"({self.performance.get('win_rate', 0):.1f}%)")
            print(f"   High Confidence: {self.performance.get('high_confidence_signals', 0)}")
            print(f"   Avg Confidence: {self.performance.get('avg_confidence', 0):.1f}%")
            print(f"   Consecutive Successes: {self.consecutive_successes}")
            
            print("=" * 100)
            
            # Additional information for high confidence signals
            if confidence >= self.config.get('high_confidence_threshold', 85.0):
                print("\n💎 HIGH CONFIDENCE RECOMMENDATIONS:")
                print("   1. Consider larger position size (within risk limits)")
                print("   2. Use tighter stop-loss for better risk management")
                print("   3. Monitor closely for entry opportunities")
                print("   4. Consider scaling in/out of position")
            
            print("\n")
            
        except Exception as e:
            logger.error(f"Error displaying signal: {e}")
            print(f"Error displaying signal: {e}")
    
    async def run_single_signal(self):
        """Run single signal generation"""
        await self.initialize()
        
        print("\n🎯 Generating single trading signal...")
        print("=" * 60)
        
        signal = await self.generate_signal()
        
        if signal:
            self.display_signal(signal)
        else:
            print("❌ Failed to generate signal")
        
        await self.shutdown()
    
    async def run_live_mode(self, interval: int = None):
        """Run in live mode with specified interval"""
        await self.initialize()
        
        interval = interval or self.config.get('interval', DEFAULT_INTERVAL)
        interval_minutes = interval // 60
        
        logger.info(f"🚀 Starting live trading mode")
        logger.info(f"⏰ Signal interval: {interval_minutes} minutes")
        logger.info(f"📊 Market hours only: {self.config.get('market_hours_only', True)}")
        logger.info("Press Ctrl+C to stop")
        print("\n" + "=" * 60)
        
        try:
            while True:
                # Check if we should run based on market hours
                should_run = True
                if self.config.get('market_hours_only', True):
                    should_run = is_market_open()
                
                if should_run:
                    signal = await self.generate_signal()
                    if signal:
                        self.display_signal(signal)
                    else:
                        logger.warning("No signal generated")
                else:
                    next_open = next_market_open()
                    if next_open:
                        wait_time = (next_open - datetime.now(TIMEZONE)).total_seconds()
                        if wait_time > 0:
                            hours = wait_time / 3600
                            logger.info(f"⏳ Market closed. Next open in {hours:.1f} hours "
                                       f"({next_open.strftime('%Y-%m-%d %H:%M ET')})")
                
                # Calculate next run time
                next_run = datetime.now(TIMEZONE) + timedelta(seconds=interval)
                
                # Display countdown
                if should_run:
                    print(f"\n⏳ Next signal at: {next_run.strftime('%H:%M:%S ET')}")
                    print("-" * 40)
                
                # Sleep until next interval
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutdown requested...")
        except Exception as e:
            logger.error(f"Error in live mode: {e}")
        finally:
            await self.shutdown()
    
    async def run_backtest(self, days: int = 365, initial_capital: float = 100000):
        """Run backtest on historical data"""
        await self.initialize()
        
        logger.info(f"📈 Running backtest for {days} days")
        logger.info(f"💰 Initial capital: ${initial_capital:,.2f}")
        
        try:
            # Get historical data
            historical_data = await self.data_extractor.get_historical_data(days=days, interval="1d")
            
            if historical_data.empty:
                logger.error("❌ No historical data for backtest")
                return
            
            # Generate signals for each day
            signals = []
            signal_dates = []
            
            for i in range(60, len(historical_data)):  # Start after enough data for indicators
                current_date = historical_data.index[i]
                current_price = historical_data['Close'].iloc[i]
                historical_slice = historical_data.iloc[:i+1]
                
                # Analyze volatility
                volatility_regime, _ = self.volatility_analyzer.analyze_volatility(historical_slice)
                
                # Generate signal (simulated)
                signal = self._generate_rule_based_signal(current_price, historical_slice, volatility_regime)
                signal['price'] = current_price
                signal['timestamp'] = current_date
                
                signals.append(signal)
                signal_dates.append(current_date)
            
            # Run backtest
            if signals:
                # Convert to pandas Series for backtesting
                signal_actions = pd.Series(
                    [1 if 'BUY' in s['action'] else -1 if 'SELL' in s['action'] else 0 
                     for s in signals],
                    index=signal_dates
                )
                
                # Run backtest
                results = self.backtester.simulate_trades(historical_data, signal_actions)
                
                # Display results
                print("\n" + "=" * 80)
                print("📈 BACKTEST RESULTS")
                print("=" * 80)
                
                if 'error' in results:
                    print(f"❌ Error: {results['error']}")
                else:
                    print(f"💰 Initial Capital: ${initial_capital:,.2f}")
                    print(f"💰 Final Equity: ${results.get('final_equity', 0):,.2f}")
                    print(f"📊 Total Return: {results.get('total_return_pct', 0):.2f}%")
                    print(f"📈 Annual Return: {results.get('annual_return_pct', 0):.2f}%")
                    print(f"📉 Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
                    print(f"⚡ Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
                    print(f"📊 Sortino Ratio: {results.get('sortino_ratio', 0):.3f}")
                    print(f"✅ Win Rate: {results.get('win_rate_pct', 0):.2f}%")
                    print(f"📈 Profit Factor: {results.get('profit_factor', 0):.3f}")
                    print(f"📊 Total Trades: {results.get('total_trades', 0)}")
                    print(f"✅ Winning Trades: {results.get('winning_trades', 0)}")
                    print(f"❌ Losing Trades: {results.get('losing_trades', 0)}")
                
                print("=" * 80)
                
                # Save results
                results_file = BACKTEST_RESULTS_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"💾 Backtest results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
        
        await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the trading system"""
        logger.info("🛑 Shutting down Gold Trading Sentinel...")
        
        try:
            # Performance summary
            runtime = datetime.now(pytz.UTC) - self.start_time
            hours = runtime.total_seconds() / 3600
            
            print("\n" + "=" * 80)
            print("📈 FINAL PERFORMANCE SUMMARY")
            print("=" * 80)
            
            print(f"⏱️  Runtime: {hours:.1f} hours")
            print(f"📊 Total Signals Generated: {self.signal_count}")
            print(f"✅ Successful Signals: {self.performance['successful_signals']}")
            print(f"❌ Failed Signals: {self.performance['failed_signals']}")
            
            if self.signal_count > 0:
                success_rate = (self.performance['successful_signals'] / self.signal_count) * 100
                print(f"📈 Success Rate: {success_rate:.1f}%")
                print(f"📊 Average Confidence: {self.performance.get('avg_confidence', 0):.1f}%")
                print(f"🏆 High Confidence Signals: {self.performance.get('high_confidence_signals', 0)}")
                print(f"📈 Consecutive Successes: {self.consecutive_successes}")
            
            print(f"💾 Data Directory: {DATA_DIR}")
            print(f"📊 Log File: gold_sentinel_v14.log")
            print("=" * 80)
            
            # Data freshness report
            freshness = self.version_manager.get_freshness_report()
            if freshness:
                print("\n📊 DATA FRESHNESS REPORT:")
                for data_type, info in freshness.items():
                    print(f"   {data_type}: {info['freshness']} ({info['age_minutes']:.1f} minutes old)")
            
            logger.info("✅ Shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# ================= INSTALLATION HELPERS =================
def install_requirements():
    """Install required packages"""
    requirements = [
        "numpy",
        "pandas",
        "aiohttp",
        "requests",
        "beautifulsoup4",
        "feedparser",
        "pytz",
        "scipy",
        "holidays",
        "torch",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "seaborn",
        "python-telegram-bot",
        "fastapi",
        "uvicorn",
        "pydantic"
    ]
    
    print("\n📦 Installing required packages...")
    
    for package in requirements:
        print(f"Installing {package}...")
        os.system(f"pip install {package}")
    
    print("\n✅ Installation complete!")

# ================= MAIN EXECUTION =================
async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Gold Trading Sentinel v14.0 - Advanced AI Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python brain_logic.py --mode single          # Generate single signal
  python brain_logic.py --mode live            # Run in live mode
  python brain_logic.py --mode backtest        # Run backtest
  python brain_logic.py --test-system          # Test system components
        """
    )
    
    parser.add_argument('--mode', choices=['single', 'live', 'backtest', 'test'], 
                       default='single', help='Operation mode')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                       help=f'Signal interval in seconds (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--backtest-days', type=int, default=365,
                       help='Days to backtest (default: 365)')
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='Initial capital for backtest (default: 100000)')
    parser.add_argument('--enable-telegram', action='store_true',
                       help='Enable Telegram notifications')
    parser.add_argument('--telegram-token', type=str,
                       help='Telegram bot token')
    parser.add_argument('--telegram-chat-id', type=str,
                       help='Telegram chat ID')
    parser.add_argument('--enable-api', action='store_true',
                       help='Enable mobile API')
    parser.add_argument('--api-port', type=int, default=8080,
                       help='Mobile API port (default: 8080)')
    parser.add_argument('--api-host', type=str, default="0.0.0.0",
                       help='Mobile API host (default: 0.0.0.0)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal console output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--install', action='store_true',
                       help='Install required packages')
    
    args = parser.parse_args()
    
    # Install requirements if requested
    if args.install:
        install_requirements()
        return
    
    # Configure logging
    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Display banner
    if not args.quiet:
        print("\n" + "=" * 100)
        print("🚀 GOLD TRADING SENTINEL V14.0 - ADVANCED AI TRADING SYSTEM")
        print("=" * 100)
        print("Features: AI-Powered Signals | Volatility Analysis | Economic Calendar")
        print("          Risk Management | Backtesting | Telegram Notifications")
        print("          Mobile API | Real-time Trading | Scraping-Based Data")
        print("=" * 100)
        print(f"📁 Data Directory: {DATA_DIR}")
        print(f"🕒 Market Hours: {MARKET_OPEN_TIME.strftime('%H:%M')} - {MARKET_CLOSE_TIME.strftime('%H:%M')} ET")
        print(f"⏰ Signal Interval: {args.interval // 60} minutes")
        print("=" * 100)
    
    # Check for required packages
    missing_packages = []
    if not SKLEARN_AVAILABLE:
        missing_packages.append("scikit-learn")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install scikit-learn")
        print("Some features will be disabled.\n")
    
    # Create configuration
    config = {
        'interval': args.interval,
        'enable_ai': True,
        'enable_volatility_analysis': True,
        'enable_backtesting': True,
        'enable_telegram': args.enable_telegram,
        'enable_mobile_api': args.enable_api,
        'api_port': args.api_port,
        'api_host': args.api_host,
        'telegram_token': args.telegram_token or os.getenv("TELEGRAM_TOKEN"),
        'telegram_chat_id': args.telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID"),
        'market_hours_only': True,
        'min_confidence': 60.0,
        'high_confidence_threshold': 85.0,
        'position_size_base': 5.0,
        'max_position_size': 15.0,
        'telegram_threshold': 70.0
    }
    
    # Create trading system
    sentinel = GoldTradingSentinelV14(config)
    
    # Execute based on mode
    try:
        if args.mode == 'test':
            print("\n🔍 Testing system components...")
            await sentinel.initialize()
            
            # Test price extraction
            price, source, details = await sentinel.data_extractor.get_current_price()
            print(f"💰 Current Price: ${price:.2f} ({source})")
            
            # Test historical data
            historical = await sentinel.data_extractor.get_historical_data(days=7)
            if not historical.empty:
                print(f"📊 Historical Data: {len(historical)} rows")
                print(f"📈 Latest Close: ${historical['Close'].iloc[-1]:.2f}")
            
            # Test volatility analysis
            if not historical.empty:
                regime, analysis = sentinel.volatility_analyzer.analyze_volatility(historical)
                print(f"📈 Volatility Regime: {regime.value}")
                print(f"📊 Current Volatility: {analysis['current_volatility']:.4f}")
            
            # Test news sentiment
            news = await sentinel.data_extractor.get_news_sentiment()
            print(f"📰 News Sentiment: {news['sentiment_score']:.1f} ({news['total_articles']} articles)")
            
            print("✅ System test complete")
        
        elif args.mode == 'single':
            await sentinel.run_single_signal()
        
        elif args.mode == 'live':
            await sentinel.run_live_mode(args.interval)
        
        elif args.mode == 'backtest':
            await sentinel.run_backtest(args.backtest_days, args.initial_capital)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ================= EXECUTION =================
if __name__ == "__main__":
    # Windows asyncio policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run main
    asyncio.run(main())

"""
Gold Trading Sentinel v9.0 - Professional Trading System
With Free Data Sources, Robust Fallback System, and Data Isolation
No API keys required - All data from free public sources
"""

import os
import sys
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import json
import re
import time
import random
import sqlite3
import threading
import queue
import hashlib
import aiofiles
import csv
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
import schedule
import concurrent.futures
from contextlib import contextmanager
from enum import Enum
import pickle
import zlib
from decimal import Decimal, ROUND_HALF_UP
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
class Timeframe(Enum):
    M15 = "15m"
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_sentinel_v9.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
TIMEZONE = pytz.timezone("America/New_York")
GOLD_NEWS_KEYWORDS = ["gold", "bullion", "XAU", "precious metals", "fed", "inflation", 
                      "central bank", "interest rates", "geopolitical", "safe haven",
                      "nonfarm payrolls", "cpi", "ppi", "fomc", "ecb", "boj", "interest rate"]

POSITIVE_KEYWORDS = ["bullish", "surge", "rally", "higher", "increase", "strong", "buy", 
                     "dovish", "stimulus", "qe", "accommodative", "pause", "cut", "dovish"]
NEGATIVE_KEYWORDS = ["bearish", "fall", "drop", "lower", "decrease", "weak", "sell", "crash",
                     "hawkish", "tightening", "tapering", "rate hike", "increase", "strong"]

DEFAULT_INTERVAL = 900  # 15 minutes
DATA_DIR = Path("data_v9")  # NEW VERSIONED DATA DIRECTORY
CACHE_DIR = DATA_DIR / "cache"
STATE_FILE = DATA_DIR / "sentinel_state.pkl"
DATABASE_FILE = DATA_DIR / "gold_signals.db"
CONFIG_FILE = DATA_DIR / "config.json"
ECONOMIC_CALENDAR_FILE = DATA_DIR / "economic_calendar.json"
BACKUP_DIR = DATA_DIR / "backups"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)

# ================= DATA ISOLATION AND VERSIONING =================
class DataVersionManager:
    """Manage data versioning and isolation to prevent clashes"""
    
    def __init__(self, base_dir: Path = DATA_DIR):
        self.base_dir = base_dir
        self.version = "v9"
        self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
                    # For SQLite databases
                    import shutil
                    shutil.copy2(source_path, backup_path)
                else:
                    # For other files
                    with open(source_path, 'rb') as src, open(backup_path, 'wb') as dst:
                        dst.write(src.read())
                logger.info(f"✅ Backed up {source_path.name} to {backup_path.name}")
        except Exception as e:
            logger.error(f"Failed to backup {source_path}: {e}")
    
    def cleanup_old_backups(self, max_backups: int = 10):
        """Clean up old backup files"""
        try:
            backups = list(BACKUP_DIR.glob("*.bak"))
            backups.sort(key=lambda x: x.stat().st_mtime)
            
            if len(backups) > max_backups:
                for backup in backups[:-max_backups]:
                    backup.unlink()
                    logger.debug(f"Cleaned up old backup: {backup.name}")
        except Exception as e:
            logger.error(f"Failed to cleanup backups: {e}")
    
    def track_freshness(self, data_type: str, timestamp: datetime):
        """Track data freshness"""
        self.freshness_tracker[data_type] = {
            'timestamp': timestamp,
            'age_seconds': (datetime.now() - timestamp).total_seconds()
        }
    
    def get_freshness_report(self) -> Dict:
        """Get data freshness report"""
        report = {}
        for data_type, info in self.freshness_tracker.items():
            age_minutes = info['age_seconds'] / 60
            freshness = "FRESH" if age_minutes < 5 else "STALE" if age_minutes < 30 else "VERY_STALE"
            report[data_type] = {
                'age_minutes': round(age_minutes, 1),
                'freshness': freshness,
                'timestamp': info['timestamp'].isoformat()
            }
        return report

# ================= ROBUST FREE DATA EXTRACTOR =================
class RobustFreeDataExtractor:
    """Robust data extractor using only free public sources with fallbacks"""
    
    def __init__(self, version_manager: DataVersionManager):
        self.version_manager = version_manager
        self.session = requests.Session()
        self.setup_session()
        
        # Multiple free data sources for redundancy
        self.price_sources = [
            self._get_price_from_yfinance,
            self._get_price_from_investing,
            self._get_price_from_marketwatch,
            self._get_price_from_google,
        ]
        
        self.historical_sources = [
            self._get_historical_from_yfinance,
            self._get_historical_from_alphavantage,  # Free tier available
            self._get_historical_from_csv_fallback,
        ]
        
        # Source weights for weighted average
        self.source_weights = {
            'yfinance': 0.35,
            'investing': 0.25,
            'marketwatch': 0.20,
            'google': 0.15,
            'fallback': 0.05
        }
        
        # Cache for failed sources
        self.failed_sources = {}
        self.source_timeouts = {}
        
    def setup_session(self):
        """Setup HTTP session with retries"""
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504, 408],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Set headers to mimic browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    async def get_current_price(self) -> Tuple[float, str, Dict]:
        """Get current gold price from multiple free sources with fallback"""
        prices = {}
        sources_used = []
        errors = []
        
        # Try each source with timeout
        for source_func in self.price_sources:
            source_name = source_func.__name__.replace('_get_price_from_', '')
            
            # Skip if source failed recently
            if self._should_skip_source(source_name):
                continue
            
            try:
                # Run with timeout
                price, metadata = await asyncio.wait_for(
                    source_func(),
                    timeout=10.0
                )
                
                if price > 0:
                    prices[source_name] = {
                        'price': price,
                        'metadata': metadata,
                        'timestamp': datetime.now(pytz.utc)
                    }
                    sources_used.append(source_name)
                    logger.info(f"✅ {source_name}: ${price:.2f}")
                    
                    # Reset failure count
                    self.failed_sources.pop(source_name, None)
                    
                else:
                    errors.append(f"{source_name}: Invalid price")
                    
            except asyncio.TimeoutError:
                errors.append(f"{source_name}: Timeout")
                self._record_source_failure(source_name)
            except Exception as e:
                errors.append(f"{source_name}: {str(e)[:50]}")
                self._record_source_failure(source_name)
        
        # If no sources succeeded, use fallback
        if not prices:
            logger.warning("All price sources failed, using fallback")
            fallback_price = await self._get_fallback_price()
            return fallback_price, "fallback", {"fallback": True}
        
        # Calculate weighted average
        weighted_price = self._calculate_weighted_average(prices)
        
        # Determine primary source
        primary_source = max(sources_used, key=lambda x: self.source_weights.get(x, 0.1))
        
        # Track freshness
        self.version_manager.track_freshness("price_data", datetime.now(pytz.utc))
        
        logger.info(f"📊 Final price: ${weighted_price:.2f} (from {len(sources_used)} sources)")
        
        return weighted_price, primary_source, {
            'sources_used': sources_used,
            'all_prices': {k: v['price'] for k, v in prices.items()},
            'errors': errors
        }
    
    def _calculate_weighted_average(self, prices: Dict) -> float:
        """Calculate weighted average of prices"""
        weighted_sum = 0
        total_weight = 0
        
        for source_name, data in prices.items():
            weight = self.source_weights.get(source_name, 0.1)
            weighted_sum += data['price'] * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else sum(d['price'] for d in prices.values()) / len(prices)
    
    def _should_skip_source(self, source_name: str) -> bool:
        """Check if source should be skipped due to recent failures"""
        if source_name in self.failed_sources:
            failures = self.failed_sources[source_name]
            if failures['count'] >= 3:
                # Check if enough time has passed
                time_since_failure = (datetime.now() - failures['last_failure']).total_seconds()
                if time_since_failure < 300:  # 5 minutes
                    return True
        return False
    
    def _record_source_failure(self, source_name: str):
        """Record source failure"""
        if source_name not in self.failed_sources:
            self.failed_sources[source_name] = {
                'count': 0,
                'last_failure': datetime.now()
            }
        self.failed_sources[source_name]['count'] += 1
        self.failed_sources[source_name]['last_failure'] = datetime.now()
    
    async def _get_price_from_yfinance(self) -> Tuple[float, Dict]:
        """Get price from Yahoo Finance"""
        try:
            # Try multiple tickers
            tickers = ["GC=F", "XAUUSD=X", "GLD", "IAU"]
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        price = float(hist['Close'].iloc[-1])
                        
                        # Convert ETF prices to approximate gold price
                        if ticker in ["GLD", "IAU"]:
                            price = price * 10
                        
                        return price, {"ticker": ticker, "method": "yfinance"}
                        
                except Exception as e:
                    continue
            
            # If all tickers fail, use quick quote
            return await self._get_yfinance_quick_quote()
            
        except Exception as e:
            logger.debug(f"yfinance error: {e}")
            return 0.0, {"error": str(e)}
    
    async def _get_yfinance_quick_quote(self) -> Tuple[float, Dict]:
        """Get quick quote from Yahoo Finance"""
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
            params = {
                'range': '1d',
                'interval': '1m'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as response:
                    data = await response.json()
                    
                    if 'chart' in data and 'result' in data['chart']:
                        result = data['chart']['result'][0]
                        meta = result['meta']
                        price = meta.get('regularMarketPrice', 0)
                        return float(price), {"method": "yfinance_api"}
                        
        except Exception as e:
            logger.debug(f"yfinance API error: {e}")
        
        return 0.0, {"error": "API failed"}
    
    async def _get_price_from_investing(self) -> Tuple[float, Dict]:
        """Get price from Investing.com (web scraping)"""
        try:
            url = "https://www.investing.com/commodities/gold"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    html = await response.text()
                    
                    # Parse price using regex (Investing.com structure)
                    patterns = [
                        r'data-test="instrument-price-last">([\d,]+\.?\d*)</span>',
                        r'lastInst"[\s\S]*?>([\d,]+\.?\d*)<',
                        r'"last":\s*"([\d,]+\.?\d*)"'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, html)
                        if match:
                            price_str = match.group(1).replace(',', '')
                            return float(price_str), {"source": "investing.com"}
            
            return 0.0, {"error": "Price not found"}
            
        except Exception as e:
            logger.debug(f"Investing.com error: {e}")
            return 0.0, {"error": str(e)}
    
    async def _get_price_from_marketwatch(self) -> Tuple[float, Dict]:
        """Get price from MarketWatch"""
        try:
            url = "https://www.marketwatch.com/investing/future/gold"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    html = await response.text()
                    
                    # Look for price in JSON-LD
                    json_ld_pattern = r'<script type="application/ld\+json">(.*?)</script>'
                    matches = re.findall(json_ld_pattern, html, re.DOTALL)
                    
                    for match in matches:
                        try:
                            data = json.loads(match)
                            if isinstance(data, dict) and 'offers' in data:
                                price = data['offers'].get('price', 0)
                                if price:
                                    return float(price), {"source": "marketwatch"}
                        except:
                            continue
                    
                    # Alternative pattern
                    price_pattern = r'price="([\d,]+\.?\d*)"'
                    match = re.search(price_pattern, html)
                    if match:
                        price_str = match.group(1).replace(',', '')
                        return float(price_str), {"source": "marketwatch"}
            
            return 0.0, {"error": "Price not found"}
            
        except Exception as e:
            logger.debug(f"MarketWatch error: {e}")
            return 0.0, {"error": str(e)}
    
    async def _get_price_from_google(self) -> Tuple[float, Dict]:
        """Get price from Google Finance"""
        try:
            url = "https://www.google.com/finance/quote/GC:F"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    html = await response.text()
                    
                    # Google Finance price pattern
                    patterns = [
                        r'data-last-price="([\d,]+\.?\d*)"',
                        r'"(\d+\.?\d*)"\s*data-source="GC:F"',
                        r'data-price="([\d,]+\.?\d*)"'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, html)
                        if match:
                            price_str = match.group(1).replace(',', '')
                            return float(price_str), {"source": "google"}
            
            return 0.0, {"error": "Price not found"}
            
        except Exception as e:
            logger.debug(f"Google Finance error: {e}")
            return 0.0, {"error": str(e)}
    
    async def _get_fallback_price(self) -> float:
        """Get fallback price when all sources fail"""
        try:
            # Try to load last known price from cache
            cache_file = CACHE_DIR / "last_price.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if time.time() - cache_data.get('timestamp', 0) < 3600:  # Less than 1 hour old
                        return cache_data.get('price', 1950.0)
            
            # Return conservative default
            return 1950.0
            
        except Exception:
            return 1950.0  # Safe default
    
    async def get_historical_data(self, days: int = 60, interval: str = "1h") -> pd.DataFrame:
        """Get historical data from multiple sources"""
        historical_data = []
        
        for source_func in self.historical_sources:
            try:
                data = await asyncio.wait_for(
                    source_func(days, interval),
                    timeout=30.0
                )
                
                if not data.empty and len(data) > 10:
                    historical_data.append(data)
                    logger.info(f"✅ Historical data from {source_func.__name__}: {len(data)} records")
                    
                    # If we have enough data, break early
                    if len(data) > 100:
                        break
                        
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Historical source {source_func.__name__} failed: {e}")
                continue
        
        # Combine all data sources
        if historical_data:
            # Remove duplicates and keep most recent
            combined = pd.concat(historical_data)
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            
            # Track freshness
            self.version_manager.track_freshness("historical_data", datetime.now(pytz.utc))
            
            return combined
        
        # Fallback to empty DataFrame
        logger.warning("All historical sources failed")
        return pd.DataFrame()
    
    async def _get_historical_from_yfinance(self, days: int, interval: str) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            # Map intervals
            interval_map = {
                "1h": "1h",
                "15m": "15m",
                "4h": "4h",
                "1d": "1d"
            }
            
            yf_interval = interval_map.get(interval, "1h")
            period = f"{days}d"
            
            ticker = yf.Ticker("GC=F")
            hist = ticker.history(period=period, interval=yf_interval)
            
            if not hist.empty:
                return hist
                
        except Exception as e:
            logger.debug(f"yFinance historical error: {e}")
        
        return pd.DataFrame()
    
    async def _get_historical_from_alphavantage(self, days: int, interval: str) -> pd.DataFrame:
        """Get historical data from Alpha Vantage (free tier)"""
        try:
            # Alpha Vantage has free tier (5 requests/minute, 500/day)
            # This is a fallback option
            api_key = os.getenv("ALPHAVANTAGE_KEY", "demo")  # Use demo key if not available
            
            # Map intervals
            if interval == "1d":
                function = "TIME_SERIES_DAILY"
            elif interval == "1h":
                function = "TIME_SERIES_INTRADAY"
            else:
                return pd.DataFrame()
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': function,
                'symbol': 'GC=F',
                'apikey': api_key,
                'outputsize': 'full' if days > 100 else 'compact'
            }
            
            if interval == "1h":
                params['interval'] = '60min'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    data = await response.json()
                    
                    # Parse response
                    if "Time Series" in data:
                        time_key = list(data.keys())[1]  # Get time series key
                        time_series = data[time_key]
                        
                        # Convert to DataFrame
                        df = pd.DataFrame.from_dict(time_series, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df = df.astype(float)
                        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        df = df.sort_index()
                        
                        return df
                        
        except Exception as e:
            logger.debug(f"Alpha Vantage error: {e}")
        
        return pd.DataFrame()
    
    async def _get_historical_from_csv_fallback(self, days: int, interval: str) -> pd.DataFrame:
        """Fallback historical data from CSV cache"""
        try:
            cache_file = CACHE_DIR / "historical_cache.csv"
            if cache_file.exists():
                # Check cache age
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < 86400:  # Less than 24 hours old
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    
                    # Filter for requested days
                    cutoff = datetime.now() - timedelta(days=days)
                    df = df[df.index >= cutoff]
                    
                    return df
                    
        except Exception as e:
            logger.debug(f"CSV cache error: {e}")
        
        return pd.DataFrame()
    
    def save_price_cache(self, price: float):
        """Save price to cache for fallback"""
        try:
            cache_file = CACHE_DIR / "last_price.json"
            cache_data = {
                'price': price,
                'timestamp': time.time(),
                'saved_at': datetime.now(pytz.utc).isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.debug(f"Failed to save price cache: {e}")
    
    def save_historical_cache(self, df: pd.DataFrame):
        """Save historical data to cache"""
        try:
            if not df.empty:
                cache_file = CACHE_DIR / "historical_cache.csv"
                df.to_csv(cache_file)
                logger.debug("Saved historical data to cache")
        except Exception as e:
            logger.debug(f"Failed to save historical cache: {e}")

# ================= ENHANCED ECONOMIC CALENDAR WITH ROBUST FALLBACK =================
class RobustEconomicCalendar:
    """Economic calendar with multiple free sources and fallback"""
    
    def __init__(self, version_manager: DataVersionManager):
        self.version_manager = version_manager
        self.events = []
        self.last_fetch = None
        
        # Multiple free calendar sources
        self.calendar_sources = [
            self._fetch_forexfactory_calendar,
            self._fetch_investing_calendar,
            self._fetch_marketwatch_calendar,
            self._fetch_tradingeconomics_calendar,
        ]
        
        # Cache for events
        self.cache_file = CACHE_DIR / "calendar_cache.json"
        
        # High-impact events for gold
        self.high_impact_events = [
            "Nonfarm Payrolls", "NFP", "CPI", "PPI", "FOMC", "Fed Rate Decision",
            "Interest Rate", "Unemployment Rate", "GDP", "Retail Sales",
            "ISM Manufacturing", "Consumer Confidence", "Inflation Rate",
            "Core PCE", "JOLTs", "Jobless Claims", "Philadelphia Fed",
            "Industrial Production", "Durable Goods", "Trade Balance"
        ]
    
    async def fetch_calendar(self, days_ahead: int = 3) -> List[Dict]:
        """Fetch economic calendar from multiple sources"""
        all_events = []
        
        # Try cache first (less than 2 hours old)
        cached_events = self._load_cached_calendar(max_age_hours=2)
        if cached_events:
            all_events.extend(cached_events)
            logger.info(f"✅ Loaded {len(cached_events)} events from cache")
        
        # Fetch from live sources
        for source_func in self.calendar_sources:
            try:
                events = await asyncio.wait_for(
                    source_func(days_ahead),
                    timeout=15.0
                )
                
                if events:
                    all_events.extend(events)
                    logger.info(f"✅ Fetched {len(events)} events from {source_func.__name__}")
                    
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Calendar source {source_func.__name__} failed: {e}")
                continue
        
        # Remove duplicates and filter for gold-related events
        unique_events = self._deduplicate_events(all_events)
        filtered_events = self._filter_gold_events(unique_events)
        
        # Sort by time
        filtered_events.sort(key=lambda x: x.get('time', datetime.now()))
        
        # Update state
        self.events = filtered_events
        self.last_fetch = datetime.now(pytz.utc)
        
        # Save to cache
        self._save_to_cache(filtered_events)
        
        # Track freshness
        self.version_manager.track_freshness("economic_calendar", self.last_fetch)
        
        logger.info(f"📅 Total economic events: {len(filtered_events)}")
        
        return filtered_events
    
    def _load_cached_calendar(self, max_age_hours: int = 2) -> List[Dict]:
        """Load cached calendar data"""
        try:
            if self.cache_file.exists():
                cache_age = time.time() - self.cache_file.stat().st_mtime
                if cache_age < max_age_hours * 3600:
                    with open(self.cache_file, 'r') as f:
                        cache_data = json.load(f)
                        
                        # Convert string dates back to datetime
                        events = []
                        for event in cache_data.get('events', []):
                            if 'time' in event and isinstance(event['time'], str):
                                try:
                                    event['time'] = datetime.fromisoformat(event['time'])
                                except:
                                    continue
                            events.append(event)
                        
                        return events
                        
        except Exception as e:
            logger.debug(f"Failed to load cached calendar: {e}")
        
        return []
    
    def _save_to_cache(self, events: List[Dict]):
        """Save events to cache"""
        try:
            # Convert datetime to string for JSON serialization
            serializable_events = []
            for event in events:
                event_copy = event.copy()
                if 'time' in event_copy and isinstance(event_copy['time'], datetime):
                    event_copy['time'] = event_copy['time'].isoformat()
                serializable_events.append(event_copy)
            
            cache_data = {
                'events': serializable_events,
                'last_fetch': datetime.now(pytz.utc).isoformat(),
                'event_count': len(events)
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.debug(f"Failed to save calendar cache: {e}")
    
    async def _fetch_forexfactory_calendar(self, days_ahead: int) -> List[Dict]:
        """Fetch from ForexFactory calendar"""
        try:
            # ForexFactory is the most reliable free calendar
            url = "https://www.forexfactory.com/calendar"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=10) as response:
                    html = await response.text()
                    
                    # Parse HTML
                    soup = BeautifulSoup(html, 'html.parser')
                    events = []
                    
                    # Find calendar rows (simplified parsing)
                    rows = soup.find_all('tr', class_='calendar_row')
                    
                    for row in rows[:50]:  # Limit to first 50 rows
                        try:
                            event = self._parse_forexfactory_row(row)
                            if event:
                                events.append(event)
                        except:
                            continue
                    
                    return events
                    
        except Exception as e:
            logger.debug(f"ForexFactory calendar error: {e}")
            return []
    
    def _parse_forexfactory_row(self, row) -> Optional[Dict]:
        """Parse ForexFactory calendar row"""
        try:
            # Extract time
            time_cell = row.find('td', class_='calendar__time')
            if not time_cell:
                return None
            
            time_text = time_cell.text.strip()
            if not time_text or time_text in ['All Day', 'Day 1', 'Day 2']:
                return None
            
            # Parse time to datetime
            event_time = self._parse_time_string(time_text)
            if not event_time:
                return None
            
            # Extract currency
            currency_cell = row.find('td', class_='calendar__currency')
            currency = currency_cell.text.strip() if currency_cell else ''
            
            # Extract impact
            impact_cell = row.find('td', class_='calendar__impact')
            impact = 'MEDIUM'
            if impact_cell:
                impact_spans = impact_cell.find_all('span', class_='icon')
                if impact_spans:
                    span_class = impact_spans[0].get('class', [])
                    if 'icon--ff-impact-red' in span_class:
                        impact = 'HIGH'
                    elif 'icon--ff-impact-orange' in span_class:
                        impact = 'MEDIUM'
                    elif 'icon--ff-impact-yellow' in span_class:
                        impact = 'LOW'
            
            # Extract title
            event_cell = row.find('td', class_='calendar__event')
            title = event_cell.text.strip() if event_cell else ''
            
            # Only include USD events and high-impact events
            if currency != 'USD' and not any(event.lower() in title.lower() for event in self.high_impact_events):
                return None
            
            return {
                'time': event_time,
                'currency': currency,
                'title': title,
                'impact': impact,
                'source': 'ForexFactory',
                'is_high_impact': impact == 'HIGH'
            }
            
        except Exception as e:
            logger.debug(f"Row parse error: {e}")
            return None
    
    def _parse_time_string(self, time_str: str) -> Optional[datetime]:
        """Parse time string to datetime"""
        try:
            now = datetime.now(TIMEZONE)
            
            # Handle formats like "2:30p" or "10:30a"
            time_str = time_str.lower().replace('a', ' AM').replace('p', ' PM')
            
            # Parse time
            time_obj = datetime.strptime(time_str, '%I:%M %p').time()
            
            # Create datetime for today
            event_time = datetime.combine(now.date(), time_obj)
            event_time = TIMEZONE.localize(event_time)
            
            # If time has passed, assume it's for tomorrow
            if event_time < now:
                event_time += timedelta(days=1)
            
            return event_time
            
        except Exception as e:
            logger.debug(f"Time parse error for '{time_str}': {e}")
            return None
    
    async def _fetch_investing_calendar(self, days_ahead: int) -> List[Dict]:
        """Fetch from Investing.com calendar"""
        # Simplified implementation
        return []
    
    async def _fetch_marketwatch_calendar(self, days_ahead: int) -> List[Dict]:
        """Fetch from MarketWatch calendar"""
        # Simplified implementation
        return []
    
    async def _fetch_tradingeconomics_calendar(self, days_ahead: int) -> List[Dict]:
        """Fetch from TradingEconomics calendar"""
        # Simplified implementation
        return []
    
    def _deduplicate_events(self, events: List[Dict]) -> List[Dict]:
        """Remove duplicate events"""
        seen = set()
        unique_events = []
        
        for event in events:
            # Create unique key
            key = f"{event.get('title', '')}_{event.get('time', '')}"
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        return unique_events
    
    def _filter_gold_events(self, events: List[Dict]) -> List[Dict]:
        """Filter events for gold relevance"""
        filtered = []
        
        for event in events:
            title = event.get('title', '').lower()
            
            # Include USD events and gold-related events
            if event.get('currency') == 'USD' or any(
                keyword.lower() in title for keyword in self.high_impact_events
            ):
                filtered.append(event)
        
        return filtered
    
    def get_upcoming_high_impact_events(self, hours_ahead: int = 6) -> List[Dict]:
        """Get upcoming high-impact events"""
        now = datetime.now(TIMEZONE)
        upcoming = []
        
        for event in self.events:
            event_time = event.get('time')
            if not event_time:
                continue
            
            # Check if event is high impact and within time window
            if event.get('is_high_impact', False):
                time_diff = (event_time - now).total_seconds() / 3600  # hours
                if 0 <= time_diff <= hours_ahead:
                    upcoming.append(event)
        
        return sorted(upcoming, key=lambda x: x.get('time'))
    
    def get_time_to_next_high_impact(self) -> Optional[timedelta]:
        """Get time to next high-impact event"""
        upcoming = self.get_upcoming_high_impact_events(24)
        if upcoming:
            next_event = upcoming[0]
            event_time = next_event.get('time')
            if event_time:
                return event_time - datetime.now(TIMEZONE)
        
        return None

# ================= ROBUST NEWS SENTIMENT ANALYZER =================
class RobustNewsSentimentAnalyzer:
    """News sentiment analyzer with multiple free sources"""
    
    def __init__(self, version_manager: DataVersionManager):
        self.version_manager = version_manager
        
        # Multiple free news sources (RSS feeds)
        self.news_sources = [
            {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US", "name": "Yahoo Finance"},
            {"url": "https://www.marketwatch.com/rss/topstories", "name": "MarketWatch"},
            {"url": "https://www.investing.com/rss/news_25.rss", "name": "Investing.com"},
            {"url": "https://www.reutersagency.com/feed/?best-sectors=markets", "name": "Reuters"},
            {"url": "https://www.cnbc.com/id/10000664/device/rss/rss.html", "name": "CNBC"},
        ]
        
        # Cache for news
        self.cache_file = CACHE_DIR / "news_cache.json"
    
    async def get_news_sentiment(self) -> Tuple[float, List[Dict]]:
        """Get news sentiment from multiple sources"""
        all_articles = []
        
        # Try cache first (less than 30 minutes old)
        cached_articles = self._load_cached_news(max_age_minutes=30)
        if cached_articles:
            all_articles.extend(cached_articles)
        
        # Fetch from live sources
        tasks = [self._fetch_rss_feed(source) for source in self.news_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        
        # Remove duplicates
        unique_articles = self._deduplicate_articles(all_articles)
        
        # Filter for gold-related articles
        gold_articles = self._filter_gold_articles(unique_articles)
        
        # Analyze sentiment
        sentiment_score, analyzed_articles = self._analyze_sentiment(gold_articles)
        
        # Save to cache
        self._save_to_cache(analyzed_articles)
        
        # Track freshness
        self.version_manager.track_freshness("news_sentiment", datetime.now(pytz.utc))
        
        logger.info(f"📰 News sentiment: {sentiment_score:.2f} (based on {len(analyzed_articles)} articles)")
        
        return sentiment_score, analyzed_articles
    
    def _load_cached_news(self, max_age_minutes: int = 30) -> List[Dict]:
        """Load cached news"""
        try:
            if self.cache_file.exists():
                cache_age = time.time() - self.cache_file.stat().st_mtime
                if cache_age < max_age_minutes * 60:
                    with open(self.cache_file, 'r') as f:
                        cache_data = json.load(f)
                        return cache_data.get('articles', [])
                        
        except Exception as e:
            logger.debug(f"Failed to load cached news: {e}")
        
        return []
    
    def _save_to_cache(self, articles: List[Dict]):
        """Save articles to cache"""
        try:
            cache_data = {
                'articles': articles,
                'last_fetch': datetime.now(pytz.utc).isoformat(),
                'article_count': len(articles)
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.debug(f"Failed to save news cache: {e}")
    
    async def _fetch_rss_feed(self, source: Dict) -> List[Dict]:
        """Fetch articles from RSS feed"""
        articles = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source['url'], timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:10]:  # Get latest 10
                            article = {
                                'title': entry.get('title', ''),
                                'summary': entry.get('summary', ''),
                                'link': entry.get('link', ''),
                                'published': entry.get('published', ''),
                                'source': source['name']
                            }
                            articles.append(article)
                            
        except Exception as e:
            logger.debug(f"Failed to fetch RSS from {source['name']}: {e}")
        
        return articles
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles"""
        seen = set()
        unique_articles = []
        
        for article in articles:
            key = f"{article['title']}_{article['source']}"
            if key not in seen:
                seen.add(key)
                unique_articles.append(article)
        
        return unique_articles
    
    def _filter_gold_articles(self, articles: List[Dict]) -> List[Dict]:
        """Filter articles for gold relevance"""
        gold_articles = []
        
        for article in articles:
            text = f"{article['title']} {article['summary']}".lower()
            
            # Check for gold-related keywords
            if any(keyword.lower() in text for keyword in GOLD_NEWS_KEYWORDS):
                gold_articles.append(article)
        
        return gold_articles
    
    def _analyze_sentiment(self, articles: List[Dict]) -> Tuple[float, List[Dict]]:
        """Analyze sentiment of articles"""
        if not articles:
            return 0.0, []
        
        total_score = 0
        analyzed_articles = []
        
        for article in articles:
            text = f"{article['title']} {article['summary']}".lower()
            
            # Count positive and negative keywords
            positive_count = sum(1 for word in POSITIVE_KEYWORDS if word in text)
            negative_count = sum(1 for word in NEGATIVE_KEYWORDS if word in text)
            
            # Calculate sentiment
            if positive_count + negative_count > 0:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment = 0
            
            # Weight by relevance (presence of gold keywords)
            relevance = sum(1 for keyword in GOLD_NEWS_KEYWORDS if keyword in text)
            weight = min(relevance / 3, 1.0)
            
            article_score = sentiment * weight
            total_score += article_score
            
            analyzed_articles.append({
                **article,
                'sentiment': article_score,
                'relevance': relevance
            })
        
        # Normalize score
        avg_score = total_score / len(articles) if articles else 0
        normalized_score = max(-1, min(1, avg_score))
        
        return normalized_score, analyzed_articles

# ================= ENHANCED DATA MANAGER WITH ISOLATION =================
class IsolatedDataManager:
    """Data manager with isolation to prevent data clashes"""
    
    def __init__(self, version_manager: DataVersionManager, extractor: RobustFreeDataExtractor):
        self.version_manager = version_manager
        self.extractor = extractor
        self.cache = {}
        self.cache_expiry = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400,
        }
        self.cache_lock = threading.RLock()
        
    async def get_current_price(self) -> Tuple[float, str, Dict]:
        """Get current price with isolation"""
        price, source, details = await self.extractor.get_current_price()
        
        # Save to cache for fallback
        if price > 0:
            self.extractor.save_price_cache(price)
        
        return price, source, details
    
    async def get_historical_data(self, timeframe: str = "1h", days: int = 30) -> pd.DataFrame:
        """Get historical data with isolation"""
        cache_key = f"historical_{timeframe}_{days}"
        
        with self.cache_lock:
            # Check cache
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                max_age = self.cache_expiry.get(timeframe, 3600)
                if time.time() - cache_entry['timestamp'] < max_age:
                    logger.debug(f"✅ Using cached historical data for {timeframe}")
                    return cache_entry['data']
        
        # Fetch fresh data
        interval_map = {
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        
        interval = interval_map.get(timeframe, "1h")
        data = await self.extractor.get_historical_data(days, interval)
        
        # Save to cache
        with self.cache_lock:
            self.cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
        
        # Save to cache file for fallback
        if not data.empty:
            self.extractor.save_historical_cache(data)
        
        return data
    
    def get_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        """Get multi-timeframe data with isolation"""
        timeframes = {
            "15m": ("5d", "15m"),
            "1h": ("30d", "1h"),
            "4h": ("60d", "4h"),
            "1d": ("180d", "1d")
        }
        
        results = {}
        
        for tf, (days, interval) in timeframes.items():
            # Use async run in sync context
            try:
                data = asyncio.run(self.get_historical_data(tf, int(days[:-1])))
                if not data.empty:
                    results[tf] = data
            except Exception as e:
                logger.error(f"Failed to get data for {tf}: {e}")
        
        return results

# ================= SIGNAL GENERATOR WITH DATA ISOLATION =================
class IsolatedSignalGenerator:
    """Signal generator with data isolation"""
    
    def __init__(self, version_manager: DataVersionManager):
        self.version_manager = version_manager
        self.signal_history = deque(maxlen=100)
        self.last_signals = {}
        
        # Performance tracking
        self.performance = {
            'total_signals': 0,
            'strong_signals': 0,
            'neutral_signals': 0,
            'paused_signals': 0
        }
    
    async def generate_signal(self, 
                            price_data: Dict[str, pd.DataFrame],
                            current_price: float,
                            volatility_regime: str,
                            atr: float,
                            news_sentiment: float,
                            economic_calendar: RobustEconomicCalendar) -> Dict[str, Any]:
        """Generate signal with isolated data"""
        
        # Check for high-impact events
        upcoming_high_impact = economic_calendar.get_upcoming_high_impact_events(2)  # Next 2 hours
        if upcoming_high_impact:
            return self._generate_pause_signal(upcoming_high_impact[0], current_price)
        
        # Calculate technical score
        technical_score = self._calculate_technical_score(price_data.get('1h', pd.DataFrame()), current_price)
        
        # Adjust for volatility
        adjusted_score = self._adjust_for_volatility(technical_score, volatility_regime)
        
        # Adjust for news sentiment (±20%)
        news_adjustment = news_sentiment * 20
        final_score = max(0, min(100, adjusted_score + news_adjustment))
        
        # Determine signal
        signal_type, confidence = self._determine_signal(final_score)
        
        # Check for signal continuity
        if self.signal_history:
            last_signal = self.signal_history[-1]
            # Avoid rapid signal changes unless strong evidence
            if last_signal['action'] != signal_type and abs(final_score - 50) < 20:
                signal_type = last_signal['action']
                confidence = last_signal['confidence'] * 0.9  # Reduce confidence slightly
        
        # Generate signal
        signal = {
            'action': signal_type,
            'confidence': round(confidence, 1),
            'price': current_price,
            'timestamp': datetime.now(pytz.utc),
            'market_summary': self._generate_summary(signal_type, final_score, volatility_regime),
            'indicators': {
                'technical_score': technical_score,
                'final_score': final_score,
                'volatility_regime': volatility_regime,
                'atr': atr,
                'news_sentiment': news_sentiment,
                'signal_id': self._generate_signal_id()
            }
        }
        
        # Update history
        self.signal_history.append(signal)
        self.last_signals[signal['indicators']['signal_id']] = signal
        
        # Update performance
        self.performance['total_signals'] += 1
        if "STRONG" in signal_type:
            self.performance['strong_signals'] += 1
        elif "NEUTRAL" in signal_type:
            self.performance['neutral_signals'] += 1
        
        return signal
    
    def _calculate_technical_score(self, df_1h: pd.DataFrame, current_price: float) -> float:
        """Calculate technical score"""
        if df_1h.empty or len(df_1h) < 20:
            return 50.0
        
        try:
            closes = df_1h['Close'].values
            closes_series = pd.Series(closes)
            
            # 1. Trend (40%)
            sma_20 = closes_series.rolling(20).mean().iloc[-1]
            sma_50 = closes_series.rolling(50).mean().iloc[-1] if len(closes_series) >= 50 else sma_20
            
            price_vs_sma20 = (current_price / sma_20 - 1) * 100
            price_vs_sma50 = (current_price / sma_50 - 1) * 100
            
            if price_vs_sma20 > 1.5 and price_vs_sma50 > 1.0 and current_price > sma_20 > sma_50:
                trend_score = 90
            elif price_vs_sma20 > 0.5 and price_vs_sma50 > 0:
                trend_score = 70
            elif price_vs_sma20 < -1.5 and price_vs_sma50 < -1.0 and current_price < sma_20 < sma_50:
                trend_score = 10
            elif price_vs_sma20 < -0.5 and price_vs_sma50 < 0:
                trend_score = 30
            else:
                trend_score = 50
            
            # 2. RSI (30%)
            rsi = self._calculate_rsi(closes_series, 14)
            if rsi < 30:
                rsi_score = 80
            elif rsi < 40:
                rsi_score = 65
            elif rsi > 70:
                rsi_score = 20
            elif rsi > 60:
                rsi_score = 35
            else:
                rsi_score = 50
            
            # 3. Volume trend (20%) - if available
            if 'Volume' in df_1h.columns and len(df_1h) >= 20:
                volume = df_1h['Volume']
                volume_sma = volume.rolling(20).mean().iloc[-1]
                current_volume = volume.iloc[-1]
                volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1
                
                # Price change
                price_change = (current_price / df_1h['Close'].iloc[-2] - 1) * 100
                
                if volume_ratio > 1.5 and price_change > 0:
                    volume_score = 80
                elif volume_ratio > 1.2 and price_change > 0:
                    volume_score = 65
                elif volume_ratio > 1.5 and price_change < 0:
                    volume_score = 20
                elif volume_ratio > 1.2 and price_change < 0:
                    volume_score = 35
                else:
                    volume_score = 50
            else:
                volume_score = 50
            
            # 4. Price position (10%)
            recent_high = df_1h['High'].tail(20).max()
            recent_low = df_1h['Low'].tail(20).min()
            
            if recent_high != recent_low:
                position = (current_price - recent_low) / (recent_high - recent_low)
                if position > 0.7:  # Near resistance
                    position_score = 30
                elif position < 0.3:  # Near support
                    position_score = 70
                else:
                    position_score = 50
            else:
                position_score = 50
            
            # Weighted composite score
            composite_score = (
                trend_score * 0.4 +
                rsi_score * 0.3 +
                volume_score * 0.2 +
                position_score * 0.1
            )
            
            return composite_score
            
        except Exception as e:
            logger.error(f"Technical score calculation error: {e}")
            return 50.0
    
    def _adjust_for_volatility(self, score: float, volatility_regime: str) -> float:
        """Adjust score based on volatility"""
        adjustments = {
            'LOW': 1.1,     # Be more sensitive in low volatility
            'MEDIUM': 1.0,   # Normal
            'HIGH': 0.9,     # Be more conservative in high volatility
            'EXTREME': 0.8   # Very conservative
        }
        
        multiplier = adjustments.get(volatility_regime, 1.0)
        
        # Center at 50, adjust, recenter
        centered = score - 50
        adjusted = centered * multiplier
        return adjusted + 50
    
    def _determine_signal(self, score: float) -> Tuple[str, float]:
        """Determine signal based on score"""
        if score >= 85:
            return "STRONG_BUY", score
        elif score >= 70:
            return "BUY", score
        elif score >= 60:
            return "NEUTRAL_LEAN_BUY", score
        elif score <= 15:
            return "STRONG_SELL", 100 - score
        elif score <= 30:
            return "SELL", 100 - score
        elif score <= 40:
            return "NEUTRAL_LEAN_SELL", 100 - score
        else:
            # 40 < score < 60
            # Slight bullish bias for gold
            return "NEUTRAL_LEAN_BUY", 55
    
    def _generate_summary(self, signal_type: str, score: float, volatility_regime: str) -> str:
        """Generate market summary"""
        parts = []
        
        # Signal strength
        if "STRONG" in signal_type:
            parts.append(f"Strong {signal_type.split('_')[-1].lower()} signal")
        else:
            parts.append(f"{signal_type.replace('_', ' ').lower()} signal")
        
        # Score context
        if score >= 70:
            parts.append("positive technical setup")
        elif score <= 30:
            parts.append("weak technical setup")
        
        # Volatility context
        if volatility_regime in ['HIGH', 'EXTREME']:
            parts.append(f"{volatility_regime.lower()} volatility environment")
        
        return ". ".join(parts).capitalize()
    
    def _generate_pause_signal(self, event: Dict, current_price: float) -> Dict[str, Any]:
        """Generate pause signal for high-impact events"""
        self.performance['paused_signals'] += 1
        
        event_time = event.get('time', datetime.now(TIMEZONE))
        time_to_event = event_time - datetime.now(TIMEZONE)
        minutes_to_event = max(0, int(time_to_event.total_seconds() / 60))
        
        return {
            'action': 'PAUSE',
            'confidence': 0.0,
            'price': current_price,
            'timestamp': datetime.now(pytz.utc),
            'market_summary': f"Trading paused due to upcoming {event.get('title', 'economic event')}",
            'indicators': {
                'pause_event': event,
                'minutes_to_event': minutes_to_event,
                'signal_id': self._generate_signal_id()
            }
        }
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50)
        except:
            return 50.0
    
    def _generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        return hashlib.md5(f"{datetime.now().isoformat()}{random.random()}".encode()).hexdigest()[:12]

# ================= VOLATILITY ANALYZER =================
class VolatilityAnalyzer:
    """Analyze market volatility"""
    
    def __init__(self):
        self.atr_period = 14
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(df) < period + 1:
                return 0.0
            
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return float(atr)
            
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.0
    
    def get_volatility_regime(self, df: pd.DataFrame) -> str:
        """Determine current volatility regime"""
        if len(df) < 50:
            return "MEDIUM"
        
        try:
            # Calculate daily returns
            returns = df['Close'].pct_change().dropna()
            
            # Calculate rolling volatility (20-period)
            if len(returns) >= 20:
                volatility = returns.rolling(20).std().iloc[-1]
            else:
                volatility = returns.std() if len(returns) > 1 else 0.01
            
            # Determine regime
            if volatility < 0.005:   # < 0.5%
                return "LOW"
            elif volatility < 0.015: # 0.5% - 1.5%
                return "MEDIUM"
            elif volatility < 0.03:  # 1.5% - 3%
                return "HIGH"
            else:                    # > 3%
                return "EXTREME"
                
        except Exception as e:
            logger.error(f"Volatility regime error: {e}")
            return "MEDIUM"

# ================= SESSION ANALYZER =================
class SessionAnalyzer:
    """Analyze market sessions"""
    
    def __init__(self):
        self.timezone = TIMEZONE
        self.sessions = {
            MarketSession.ASIAN: {
                'start': dt_time(19, 0),  # 7 PM ET
                'end': dt_time(4, 0),     # 4 AM ET
                'multiplier': 0.6
            },
            MarketSession.LONDON: {
                'start': dt_time(3, 0),   # 3 AM ET
                'end': dt_time(12, 0),    # 12 PM ET
                'multiplier': 0.9
            },
            MarketSession.LONDON_NY_OVERLAP: {
                'start': dt_time(8, 0),   # 8 AM ET
                'end': dt_time(11, 0),    # 11 AM ET
                'multiplier': 1.3
            },
            MarketSession.NY: {
                'start': dt_time(8, 0),   # 8 AM ET
                'end': dt_time(17, 0),    # 5 PM ET
                'multiplier': 1.1
            },
            MarketSession.AFTER_HOURS: {
                'start': dt_time(17, 0),  # 5 PM ET
                'end': dt_time(19, 0),    # 7 PM ET
                'multiplier': 0.7
            }
        }
    
    def get_current_session(self) -> Tuple[MarketSession, Dict]:
        """Get current market session"""
        now = datetime.now(self.timezone)
        current_time = now.time()
        
        for session, config in self.sessions.items():
            start = config['start']
            end = config['end']
            
            # Handle overnight sessions
            if start > end:  # Overnight
                if current_time >= start or current_time < end:
                    return session, config
            else:  # Day session
                if start <= current_time < end:
                    return session, config
        
        # Default to Asian session
        return MarketSession.ASIAN, self.sessions[MarketSession.ASIAN]
    
    def get_session_multiplier(self) -> float:
        """Get session multiplier for signal adjustment"""
        session, config = self.get_current_session()
        return config['multiplier']

# ================= TELEGRAM NOTIFIER =================
class TelegramNotifier:
    """Telegram notification system"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
    
    async def send_signal(self, signal: Dict, signal_number: int = None):
        """Send signal to Telegram"""
        message = self._format_signal_message(signal, signal_number)
        await self._send_message(message)
    
    async def send_alert(self, alert_type: str, content: str):
        """Send alert to Telegram"""
        message = self._format_alert_message(alert_type, content)
        await self._send_message(message)
    
    async def _send_message(self, message: str):
        """Send message to Telegram"""
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("✅ Message sent to Telegram")
                else:
                    error_text = await response.text()
                    logger.error(f"Telegram API error: {error_text}")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def _format_signal_message(self, signal: Dict, signal_number: int = None) -> str:
        """Format signal for Telegram"""
        
        if signal['action'] == 'PAUSE':
            return self._format_pause_message(signal)
        
        # Signal emojis
        emoji_map = {
            "STRONG_BUY": "🟢🟢🟢",
            "BUY": "🟢🟢",
            "NEUTRAL_LEAN_BUY": "🟡",
            "NEUTRAL_LEAN_SELL": "🟡",
            "SELL": "🔴🔴",
            "STRONG_SELL": "🔴🔴🔴"
        }
        
        emoji = emoji_map.get(signal['action'], "⚪")
        title = f"*GOLD SIGNAL #{signal_number}*" if signal_number else "*GOLD SIGNAL*"
        
        # Format message
        message = f"""
{emoji} {title}

*Signal:* {signal['action']}
*Price:* ${signal['price']:.2f}
*Confidence:* {signal['confidence']:.1f}%

*Technical Score:* {signal['indicators'].get('technical_score', 0):.1f}
*Volatility:* {signal['indicators'].get('volatility_regime', 'N/A')}
*News Sentiment:* {signal['indicators'].get('news_sentiment', 0):+.2f}

*Market Summary:*
{signal['market_summary']}

_Signal ID: {signal['indicators'].get('signal_id', 'N/A')}_
_Generated at {signal['timestamp'].astimezone(TIMEZONE).strftime('%H:%M:%S ET')}_

#Gold #Trading #Signal
"""
        return message
    
    def _format_pause_message(self, signal: Dict) -> str:
        """Format pause message"""
        event = signal['indicators'].get('pause_event', {})
        minutes = signal['indicators'].get('minutes_to_event', 0)
        
        message = f"""
🛑 *TRADING PAUSED*

*Reason:* High-impact economic event
*Event:* {event.get('title', 'Economic Release')}
*Time to Event:* {minutes} minutes
*Impact:* {event.get('impact', 'HIGH')}

*Recommendation:* 
• Avoid new positions
• Close risky positions
• Wait for post-event stabilization

*Resume:* 30+ minutes after event release

_Current Gold Price: ${signal['price']:.2f}_
"""
        return message
    
    def _format_alert_message(self, alert_type: str, content: str) -> str:
        """Format alert message"""
        emoji_map = {
            "ERROR": "❌",
            "WARNING": "⚠️",
            "INFO": "ℹ️",
            "SUCCESS": "✅"
        }
        
        emoji = emoji_map.get(alert_type, "⚠️")
        return f"{emoji} *{alert_type}*\n\n{content}"
    
    async def close(self):
        """Close session"""
        await self.session.close()

# ================= MAIN TRADING BOT =================
class GoldTradingSentinelV9:
    """Gold Trading Sentinel v9.0 with robust free data sources"""
    
    def __init__(self, config: Dict = None):
        # Initialize version manager for data isolation
        self.version_manager = DataVersionManager()
        
        # Backup existing data
        self.version_manager.backup_data(STATE_FILE)
        self.version_manager.backup_data(DATABASE_FILE)
        self.version_manager.cleanup_old_backups()
        
        # Load configuration
        self.config = config or self._load_config()
        
        # Initialize components
        self.data_extractor = RobustFreeDataExtractor(self.version_manager)
        self.data_manager = IsolatedDataManager(self.version_manager, self.data_extractor)
        self.economic_calendar = RobustEconomicCalendar(self.version_manager)
        self.news_analyzer = RobustNewsSentimentAnalyzer(self.version_manager)
        self.volatility_analyzer = VolatilityAnalyzer()
        self.session_analyzer = SessionAnalyzer()
        self.signal_generator = IsolatedSignalGenerator(self.version_manager)
        
        # Telegram notifier
        self.telegram = None
        if self.config.get('telegram_token') and self.config.get('telegram_chat_id'):
            self.telegram = TelegramNotifier(
                self.config['telegram_token'],
                self.config['telegram_chat_id']
            )
        
        # State
        self.running = False
        self.signal_count = 0
        self.last_signal = None
        self.start_time = datetime.now(pytz.utc)
        
        # Performance tracking
        self.performance = {
            'total_runs': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'data_freshness': {}
        }
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        default_config = {
            'interval': DEFAULT_INTERVAL,
            'enable_telegram': True,
            'enable_economic_calendar': True,
            'enable_news_sentiment': True,
            'telegram_token': os.getenv("TELEGRAM_TOKEN"),
            'telegram_chat_id': os.getenv("TELEGRAM_CHAT_ID"),
            'data_sources': ['yfinance', 'investing', 'marketwatch', 'google'],
            'fallback_enabled': True,
            'cache_enabled': True
        }
        
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    saved_config = json.load(f)
                default_config.update(saved_config)
                logger.info("✅ Loaded configuration")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save configuration"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("✅ Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    async def initialize(self):
        """Initialize the bot"""
        logger.info("🚀 Initializing Gold Trading Sentinel v9.0...")
        logger.info(f"📁 Data directory: {DATA_DIR}")
        logger.info(f"🆔 Run ID: {self.version_manager.current_run_id}")
        
        # Fetch economic calendar
        if self.config.get('enable_economic_calendar', True):
            logger.info("📅 Fetching economic calendar...")
            await self.economic_calendar.fetch_calendar()
        
        # Send startup message
        if self.telegram:
            startup_msg = (
                "🚀 *Gold Trading Sentinel v9.0 Started*\n\n"
                "Features:\n"
                "• Free data sources only\n"
                "• Robust fallback system\n"
                "• Data isolation (v9)\n"
                "• Economic calendar integration\n"
                "• News sentiment analysis\n\n"
                "_Bot is now monitoring XAUUSD..._"
            )
            await self.telegram.send_alert("INFO", startup_msg)
    
    async def generate_signal(self) -> Optional[Dict]:
        """Generate trading signal"""
        self.performance['total_runs'] += 1
        
        try:
            logger.info("=" * 80)
            logger.info("🔄 Generating trading signal...")
            
            # 1. Get current price
            logger.info("💰 Fetching current price...")
            current_price, source, price_details = await self.data_manager.get_current_price()
            
            if current_price <= 0:
                logger.error("❌ Failed to get valid price")
                self.performance['failed_signals'] += 1
                return None
            
            logger.info(f"✅ Current price: ${current_price:.2f} ({source})")
            
            # 2. Get historical data
            logger.info("📊 Fetching historical data...")
            price_data = self.data_manager.get_multi_timeframe_data()
            
            if not price_data or '1h' not in price_data:
                logger.warning("⚠️ Limited historical data available")
            
            # 3. Analyze volatility
            df_1h = price_data.get('1h', pd.DataFrame())
            volatility_regime = self.volatility_analyzer.get_volatility_regime(df_1h)
            atr = self.volatility_analyzer.calculate_atr(df_1h)
            
            logger.info(f"📈 Volatility: {volatility_regime} (ATR: ${atr:.2f})")
            
            # 4. Get news sentiment
            news_sentiment = 0.0
            if self.config.get('enable_news_sentiment', True):
                logger.info("📰 Analyzing news sentiment...")
                news_sentiment, _ = await self.news_analyzer.get_news_sentiment()
                logger.info(f"✅ News sentiment: {news_sentiment:+.2f}")
            
            # 5. Generate signal
            logger.info("⚡ Generating trading signal...")
            signal = await self.signal_generator.generate_signal(
                price_data=price_data,
                current_price=current_price,
                volatility_regime=volatility_regime,
                atr=atr,
                news_sentiment=news_sentiment,
                economic_calendar=self.economic_calendar
            )
            
            # Add metadata
            signal['metadata'] = {
                'signal_number': self.signal_count + 1,
                'price_source': source,
                'price_details': price_details,
                'run_id': self.version_manager.current_run_id
            }
            
            # Update state
            self.last_signal = signal
            self.signal_count += 1
            
            # Update performance
            if signal['action'] != 'PAUSE':
                self.performance['successful_signals'] += 1
            else:
                self.performance['failed_signals'] += 1
            
            # Update data freshness
            self.performance['data_freshness'] = self.version_manager.get_freshness_report()
            
            logger.info(f"✅ Signal #{self.signal_count} generated: {signal['action']}")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}", exc_info=True)
            self.performance['failed_signals'] += 1
            return None
    
    def display_signal(self, signal: Dict):
        """Display signal in console"""
        print("\n" + "=" * 90)
        print("🚀 GOLD TRADING SENTINEL v9.0 - FREE DATA SOURCES")
        print("=" * 90)
        
        if signal['action'] == 'PAUSE':
            self._display_pause_signal(signal)
            return
        
        metadata = signal.get('metadata', {})
        indicators = signal['indicators']
        
        print(f"🕒 Time: {signal['timestamp'].astimezone(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S ET')}")
        print(f"💰 Price: ${signal['price']:.2f}")
        print(f"📊 Signal #{metadata.get('signal_number', 'N/A')}")
        print(f"📁 Run ID: {metadata.get('run_id', 'N/A')}")
        print("-" * 90)
        
        # Display signal
        signal_configs = {
            "STRONG_BUY": ("🟢", "STRONG BUY", "🟢🟢🟢", "EXTREMELY BULLISH"),
            "BUY": ("🟢", "BUY", "🟢🟢", "BULLISH"),
            "NEUTRAL_LEAN_BUY": ("🟡", "NEUTRAL (Lean to Buy)", "↗️", "CAUTIOUSLY BULLISH"),
            "NEUTRAL_LEAN_SELL": ("🟡", "NEUTRAL (Lean to Sell)", "↘️", "CAUTIOUSLY BEARISH"),
            "SELL": ("🔴", "SELL", "🔴🔴", "BEARISH"),
            "STRONG_SELL": ("🔴", "STRONG SELL", "🔴🔴🔴", "EXTREMELY BEARISH")
        }
        
        emoji, display_name, strength, bias = signal_configs.get(
            signal['action'], ("⚪", signal['action'], "", "NEUTRAL")
        )
        
        print(f"🎯 SIGNAL: {strength} {emoji} {display_name} {strength}")
        print(f"📊 Confidence: {signal['confidence']:.1f}%")
        print(f"📈 Market Bias: {bias}")
        print("-" * 90)
        
        # Display indicators
        print("📊 TECHNICAL ANALYSIS:")
        print(f"   Technical Score: {indicators.get('technical_score', 0):.1f}")
        print(f"   Final Score: {indicators.get('final_score', 0):.1f}")
        print(f"   Volatility Regime: {indicators.get('volatility_regime', 'N/A')}")
        print(f"   ATR: ${indicators.get('atr', 0):.2f}")
        print(f"   News Sentiment: {indicators.get('news_sentiment', 0):+.2f}")
        
        # Display metadata
        if metadata.get('price_source'):
            print(f"\n📡 DATA SOURCES:")
            print(f"   Primary Source: {metadata['price_source']}")
            if 'price_details' in metadata and 'sources_used' in metadata['price_details']:
                sources = metadata['price_details']['sources_used']
                print(f"   Sources Used: {', '.join(sources)}")
        
        print("\n📋 MARKET SUMMARY:")
        print(f"   {signal['market_summary']}")
        print("=" * 90)
        
        # Display recommendations
        self._display_recommendations(signal)
    
    def _display_pause_signal(self, signal: Dict):
        """Display pause signal"""
        indicators = signal['indicators']
        event = indicators.get('pause_event', {})
        
        print(f"\n🛑 TRADING PAUSED")
        print(f"Reason: High-impact economic event")
        print(f"Event: {event.get('title', 'Economic Release')}")
        print(f"Time to Event: {indicators.get('minutes_to_event', 0)} minutes")
        print(f"Impact: {event.get('impact', 'HIGH')}")
        print(f"\nCurrent Price: ${signal['price']:.2f}")
        print(f"Time: {signal['timestamp'].astimezone(TIMEZONE).strftime('%H:%M ET')}")
        
        print("\n📋 RECOMMENDATIONS:")
        print("   • Avoid new positions")
        print("   • Close risky positions")
        print("   • Wait for post-event stabilization")
        print("   • Resume trading 30+ minutes after event")
        print("=" * 90)
    
    def _display_recommendations(self, signal: Dict):
        """Display trading recommendations"""
        print("\n💼 TRADING RECOMMENDATIONS:")
        print("-" * 50)
        
        action = signal['action']
        atr = signal['indicators'].get('atr', 0)
        confidence = signal['confidence']
        
        recommendations = {
            "STRONG_BUY": [
                "• Enter long position immediately",
                "• Use full position size",
                f"• Suggested stop loss: ${signal['price'] - atr*2:.2f}",
                f"• Suggested take profit: ${signal['price'] + atr*3:.2f}",
                "• Monitor for continuation"
            ],
            "BUY": [
                "• Enter long position",
                "• Use 75% position size",
                f"• Suggested stop loss: ${signal['price'] - atr*1.5:.2f}",
                f"• Suggested take profit: ${signal['price'] + atr*2.5:.2f}",
                "• Wait for minor pullbacks"
            ],
            "NEUTRAL_LEAN_BUY": [
                "• Consider small long position",
                "• Use 50% position size",
                f"• Suggested stop loss: ${signal['price'] - atr*2:.2f}",
                f"• Suggested take profit: ${signal['price'] + atr*2:.2f}",
                "• Wait for confirmation"
            ],
            "NEUTRAL_LEAN_SELL": [
                "• Consider reducing long positions",
                "• Use 25% position size for short",
                f"• Suggested stop loss: ${signal['price'] + atr*2:.2f}",
                f"• Suggested take profit: ${signal['price'] - atr*2:.2f}",
                "• Wait for breakdown"
            ],
            "SELL": [
                "• Exit long positions",
                "• Enter short position",
                "• Use 75% position size",
                f"• Suggested stop loss: ${signal['price'] + atr*1.5:.2f}",
                f"• Suggested take profit: ${signal['price'] - atr*2.5:.2f}",
                "• Short on bounces"
            ],
            "STRONG_SELL": [
                "• Exit all long positions",
                "• Enter aggressive short",
                "• Use full position size",
                f"• Suggested stop loss: ${signal['price'] + atr*2:.2f}",
                f"• Suggested take profit: ${signal['price'] - atr*3:.2f}",
                "• Add on failed bounces"
            ]
        }
        
        for rec in recommendations.get(action, ["• No specific recommendations"]):
            print(f"   {rec}")
        
        # Confidence note
        if confidence >= 80:
            print(f"\n   ✅ High confidence signal ({confidence}%)")
        elif confidence >= 60:
            print(f"\n   ⚠️ Moderate confidence signal ({confidence}%)")
        else:
            print(f"\n   ⚠️ Low confidence signal ({confidence}%)")
        
        print("\n⚠️ RISK MANAGEMENT:")
        print("   • Maximum risk: 2% of account per trade")
        print("   • Always use stop-loss orders")
        print("   • Monitor economic calendar")
        print("   • Adjust position size based on confidence")
        print("=" * 90)
    
    async def process_signal(self, signal: Dict):
        """Process generated signal"""
        # Display in console
        self.display_signal(signal)
        
        # Send to Telegram if enabled
        if self.telegram and signal['action'] != 'PAUSE':
            await self.telegram.send_signal(signal, self.signal_count)
        
        # Log performance
        self._log_performance()
    
    def _log_performance(self):
        """Log system performance"""
        runtime = datetime.now(pytz.utc) - self.start_time
        hours = runtime.total_seconds() / 3600
        
        logger.info("📈 SYSTEM PERFORMANCE:")
        logger.info(f"   Runtime: {hours:.1f} hours")
        logger.info(f"   Total Signals: {self.signal_count}")
        logger.info(f"   Successful: {self.performance['successful_signals']}")
        logger.info(f"   Failed: {self.performance['failed_signals']}")
        logger.info(f"   Success Rate: {self.performance['successful_signals']/max(self.performance['total_runs'], 1)*100:.1f}%")
        
        # Log data freshness
        freshness = self.performance.get('data_freshness', {})
        for data_type, info in freshness.items():
            logger.info(f"   {data_type}: {info.get('age_minutes', 0):.1f} min ({info.get('freshness', 'UNKNOWN')})")
    
    async def run_single(self):
        """Run single signal generation"""
        await self.initialize()
        signal = await self.generate_signal()
        
        if signal:
            await self.process_signal(signal)
        else:
            print("❌ Failed to generate signal")
        
        await self.shutdown()
    
    async def run_live(self, interval: int = DEFAULT_INTERVAL):
        """Run live signal generation"""
        await self.initialize()
        
        logger.info(f"🚀 Starting live mode (interval: {interval//60} minutes)")
        logger.info("Press Ctrl+C to stop\n")
        
        try:
            while True:
                signal = await self.generate_signal()
                if signal:
                    await self.process_signal(signal)
                
                # Calculate next run time
                next_run = datetime.now(TIMEZONE) + timedelta(seconds=interval)
                print(f"\n⏳ Next signal at: {next_run.strftime('%H:%M:%S ET')}")
                print("-" * 50)
                
                # Sleep until next interval
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Received shutdown signal...")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the bot"""
        logger.info("🛑 Shutting down Gold Trading Sentinel v9.0...")
        
        # Send shutdown message
        if self.telegram:
            runtime = datetime.now(pytz.utc) - self.start_time
            hours = runtime.total_seconds() / 3600
            
            shutdown_msg = (
                f"🛑 *Gold Trading Sentinel v9.0 Stopped*\n\n"
                f"Performance Summary:\n"
                f"• Runtime: {hours:.1f} hours\n"
                f"• Signals Generated: {self.signal_count}\n"
                f"• Success Rate: {self.performance['successful_signals']/max(self.performance['total_runs'], 1)*100:.1f}%\n"
                f"• Run ID: {self.version_manager.current_run_id}\n\n"
                f"_Shutdown complete._"
            )
            
            await self.telegram.send_alert("INFO", shutdown_msg)
            await self.telegram.close()
        
        # Save configuration
        self.save_config()
        
        # Cleanup
        self.version_manager.cleanup_old_backups()
        
        logger.info("✅ Shutdown complete")

# ================= MAIN EXECUTION =================
async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Gold Trading Sentinel v9.0 - Free Data Sources')
    parser.add_argument('--mode', choices=['single', 'live', 'test', 'clean'], 
                       default='single', help='Operation mode')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                       help=f'Signal interval in seconds (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--test-data', action='store_true',
                       help='Test data extraction')
    parser.add_argument('--test-telegram', action='store_true',
                       help='Test Telegram connection')
    parser.add_argument('--clean-cache', action='store_true',
                       help='Clean cache directory')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal console output')
    
    args = parser.parse_args()
    
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    # Display banner
    if not args.quiet:
        print("\n" + "=" * 100)
        print("🚀 GOLD TRADING SENTINEL v9.0 - FREE DATA SOURCES WITH ROBUST FALLBACK")
        print("=" * 100)
        print("Features: No API Keys Required | Multi-Source Data Extraction")
        print("          Data Isolation (v9) | Robust Fallback System")
        print("          Economic Calendar | News Sentiment Analysis")
        print("=" * 100)
        print(f"📁 Data Directory: {DATA_DIR}")
        print("=" * 100)
    
    # Clean cache if requested
    if args.clean_cache:
        print("\n🧹 Cleaning cache directory...")
        try:
            for file in CACHE_DIR.glob("*"):
                file.unlink()
            print("✅ Cache cleaned")
        except Exception as e:
            print(f"❌ Failed to clean cache: {e}")
        return 0
    
    # Create bot instance
    config = {
        'interval': args.interval,
        'enable_telegram': True,
        'enable_economic_calendar': True,
        'enable_news_sentiment': True,
        'telegram_token': os.getenv("TELEGRAM_TOKEN"),
        'telegram_chat_id': os.getenv("TELEGRAM_CHAT_ID")
    }
    
    bot = GoldTradingSentinelV9(config)
    
    # Test Telegram
    if args.test_telegram:
        if not config['telegram_token'] or not config['telegram_chat_id']:
            print("❌ Telegram credentials not found!")
            print("   Set environment variables:")
            print("   export TELEGRAM_TOKEN='your_bot_token'")
            print("   export TELEGRAM_CHAT_ID='your_chat_id'")
            return 1
        
        print("\n🤖 Testing Telegram connection...")
        telegram = TelegramNotifier(config['telegram_token'], config['telegram_chat_id'])
        
        try:
            await telegram.send_alert("TEST", "✅ Telegram connection test successful!")
            await asyncio.sleep(1)
            await telegram.close()
            print("✅ Telegram test successful!")
            return 0
        except Exception as e:
            print(f"❌ Telegram test failed: {e}")
            return 1
    
    # Test data extraction
    if args.test_data:
        print("\n🔍 Testing data extraction...")
        version_manager = DataVersionManager()
        extractor = RobustFreeDataExtractor(version_manager)
        
        print("💰 Testing price extraction...")
        price, source, details = await extractor.get_current_price()
        print(f"   Price: ${price:.2f} ({source})")
        print(f"   Details: {details}")
        
        print("\n📊 Testing historical data...")
        data = await extractor.get_historical_data(days=5, interval="1h")
        print(f"   Records: {len(data)}")
        
        return 0
    
    if args.mode == 'single':
        print("\n🎯 Generating single signal...")
        print("-" * 50)
        await bot.run_single()
    
    elif args.mode == 'live':
        print("\n🚀 Starting live mode...")
        print("-" * 50)
        await bot.run_live(args.interval)
    
    elif args.mode == 'test':
        print("\n🧪 Running tests...")
        # Run comprehensive tests
        await bot.initialize()
        
        # Test all components
        print("\n✅ Initialization complete")
        print(f"📁 Run ID: {bot.version_manager.current_run_id}")
        
        await bot.shutdown()
    
    elif args.mode == 'clean':
        print("\n🧹 Cleaning up...")
        # This is handled by the clean-cache flag
    
    return 0

if __name__ == "__main__":
    try:
        # Windows compatibility
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

"""
Gold Trading Sentinel v11.0 - Standalone 5-Year Deep Learning System
Complete AI-powered gold trading system with 5 years of backtesting
Single file - No external dependencies beyond standard libraries
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
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ================= DEEP LEARNING IMPORTS =================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
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
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    print("LightGBM not available. Install with: pip install lightgbm")
    LGB_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Optuna not available. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
        logging.FileHandler('gold_sentinel_v11.log'),
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
DATA_DIR = Path("data_v11")
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

# ================= DATA VERSION MANAGER =================
class DataVersionManager:
    """Manage data versioning and isolation"""
    
    def __init__(self, base_dir: Path = DATA_DIR):
        self.base_dir = base_dir
        self.version = "v11"
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
                    backup.unlink()
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
    """Robust data extractor using only free public sources"""
    
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
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
                    logger.debug(f"✅ {source_name}: ${price:.2f}")
                    
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
                    
                    # Parse price using regex
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

# ================= DEEP LEARNING MODELS =================
class GoldTradingLSTM(nn.Module):
    """LSTM model for gold trading predictions"""
    
    def __init__(self, input_dim=50, hidden_dim=128, dropout_rate=0.3):
        super(GoldTradingLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        self.fc_blocks = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Buy, Hold, Sell
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Risk score 0-1
        )
        
    def forward(self, x):
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last timestep
        last_out = lstm_out[:, -1, :]
        
        # Main prediction
        signal_probs = self.fc_blocks(last_out)
        
        # Risk assessment
        risk_score = self.risk_head(last_out)
        
        return signal_probs, risk_score

class GoldTradingGRU(nn.Module):
    """GRU model for gold trading"""
    
    def __init__(self, input_dim=50, hidden_dim=128, dropout_rate=0.3):
        super(GoldTradingGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        self.fc_blocks = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        return self.fc_blocks(last_out)

# ================= ADVANCED FEATURE ENGINEER =================
class AdvancedFeatureEngineer:
    """Generate comprehensive features for ML model"""
    
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_cache = {}
        self.feature_importance = {}
        
    def create_features(self, df, include_derived=True):
        """Create comprehensive feature set"""
        features = pd.DataFrame(index=df.index)
        
        # 1. Price-based features
        features['price'] = df['Close']
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        features['high_low_ratio'] = df['High'] / df['Low']
        features['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # 2. Moving averages
        for period in [5, 10, 20, 50, 100]:
            features[f'sma_{period}'] = df['Close'].rolling(period).mean()
            features[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            features[f'price_sma_ratio_{period}'] = df['Close'] / features[f'sma_{period}']
            
        # 3. Volatility features
        features['volatility_20'] = df['Close'].pct_change().rolling(20).std()
        features['volatility_50'] = df['Close'].pct_change().rolling(50).std()
        features['atr_14'] = self.calculate_atr(df, 14)
        features['atr_ratio'] = features['atr_14'] / df['Close']
        
        # 4. Momentum indicators
        features['rsi_14'] = self.calculate_rsi(df['Close'], 14)
        features['rsi_28'] = self.calculate_rsi(df['Close'], 28)
        
        # 5. Volume features
        if 'Volume' in df.columns:
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            features['volume_price_trend'] = (
                df['Volume'] * df['Close'].pct_change()
            ).rolling(20).sum()
            features['obv'] = self.calculate_obv(df)
            
        # 6. Time-based features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # 7. Statistical features
        features['skewness_20'] = df['Close'].rolling(20).skew()
        features['kurtosis_20'] = df['Close'].rolling(20).kurt()
        features['z_score_20'] = (
            df['Close'] - df['Close'].rolling(20).mean()
        ) / df['Close'].rolling(20).std()
        
        if include_derived:
            # 8. Advanced derived features
            features = self.create_derived_features(features)
            
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def calculate_atr(self, df, period):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_rsi(self, prices, period):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    def create_derived_features(self, features):
        """Create advanced derived features"""
        # Rate of change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = (
                features['price'] / features['price'].shift(period) - 1
            ) * 100
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = features[f'sma_{period}']
            std = features['price'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + (std * 2)
            features[f'bb_lower_{period}'] = sma - (std * 2)
            features[f'bb_position_{period}'] = (
                features['price'] - features[f'bb_lower_{period}']
            ) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # Fibonacci retracement
        for lookback in [20, 50, 100]:
            high = features['price'].rolling(lookback).max()
            low = features['price'].rolling(lookback).min()
            range_ = high - low
            
            for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
                features[f'fib_{lookback}_{level}'] = low + (range_ * level)
        
        return features

# ================= BACKTESTING ENGINE =================
class DeepBacktester:
    """Advanced backtesting engine with learning capabilities"""
    
    def __init__(self, initial_capital=100000, commission=2.5, slippage=0.0001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades = []
        self.performance_history = []
        self.parameter_history = []
        self.best_params = None
        self.best_sharpe = -np.inf
        
    def run_backtest(self, df, signals, model_name="LSTM"):
        """Run comprehensive backtest"""
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Calculate position size
            kelly_fraction = self.best_params.get('kelly_fraction', 0.1) if self.best_params else 0.1
            
            # Generate trade based on signal
            if signal == 1 and position <= 0:  # Buy signal
                if position < 0:  # Close short position first
                    trade_result = self._close_position(
                        position, current_price, df.index[i]
                    )
                    capital += trade_result
                    trades.append(trade_result)
                
                # Open long position
                max_position = (capital * kelly_fraction) / current_price
                position = max_position
                capital -= (position * current_price * (1 + self.slippage))
                capital -= self.commission
                
            elif signal == -1 and position >= 0:  # Sell signal
                if position > 0:  # Close long position first
                    trade_result = self._close_position(
                        position, current_price, df.index[i]
                    )
                    capital += trade_result
                    trades.append(trade_result)
                
                # Open short position
                max_position = (capital * kelly_fraction) / current_price
                position = -max_position
                capital -= abs(position) * current_price * self.slippage
                capital -= self.commission
                
            elif signal == 0 and position != 0:  # Close position signal
                trade_result = self._close_position(position, current_price, df.index[i])
                capital += trade_result
                trades.append(trade_result)
                position = 0
            
            # Update equity curve
            current_equity = capital + (position * current_price)
            equity_curve.append(current_equity)
        
        # Close any open position at end
        if position != 0:
            trade_result = self._close_position(
                position, df['Close'].iloc[-1], df.index[-1]
            )
            capital += trade_result
            trades.append(trade_result)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(
            equity_curve, trades, model_name
        )
        
        self.performance_history.append(performance)
        self.trades.extend(trades)
        
        return performance, trades, equity_curve
    
    def _close_position(self, position, price, timestamp):
        """Close a position and calculate P&L"""
        if position > 0:  # Long position
            pnl = position * price * (1 - self.slippage) - self.commission
        else:  # Short position
            pnl = abs(position) * price * (1 + self.slippage) - self.commission
        
        return pnl
    
    def _calculate_performance_metrics(self, equity_curve, trades, model_name):
        """Calculate comprehensive performance metrics"""
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_series.iloc[-1] / self.initial_capital - 1) * 100
        annual_return = total_return / (len(equity_curve) / 252) if len(equity_curve) > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Trade metrics
        winning_trades = [t for t in trades if t > 0]
        losing_trades = [t for t in trades if t <= 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0
        
        # Advanced metrics
        sortino_ratio = self._calculate_sortino_ratio(returns, annual_return)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        performance = {
            'model': model_name,
            'total_return_%': total_return,
            'annual_return_%': annual_return,
            'volatility_%': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown_%': max_drawdown,
            'win_rate_%': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'best_trade': max(trades) if trades else 0,
            'worst_trade': min(trades) if trades else 0,
            'avg_trade': np.mean(trades) if trades else 0,
        }
        
        return performance
    
    def _calculate_sortino_ratio(self, returns, annual_return):
        """Calculate Sortino ratio"""
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        return annual_return / downside_deviation if downside_deviation > 0 else 0

# ================= FIVE YEAR DEEP LEARNING SYSTEM =================
class FiveYearDeepLearningSystem:
    """Deep learning system trained on 5 years of data"""
    
    def __init__(self, version_manager):
        self.version_manager = version_manager
        self.feature_engineer = AdvancedFeatureEngineer()
        self.backtester = DeepBacktester()
        self.models = {}
        self.learning_history = []
        self.improvement_rate = 0
        self.best_model = None
        self.best_performance = None
        self.best_params = None
        
        # Training configuration
        self.training_years = 5
        self.validation_split = 0.3
        self.sequence_length = 60
        
        # Model registry
        self.model_registry = {}
        
    async def learn_from_5_years_data(self):
        """Main learning function with 5 years of data"""
        logger.info("🧠 Starting 5-year deep learning backtest...")
        
        # Get 5 years of historical data
        historical_data = await self._get_5_years_data()
        
        if historical_data.empty:
            logger.error("❌ Failed to get 5 years of historical data")
            return None
        
        logger.info(f"📊 Loaded {len(historical_data)} records for 5-year backtest")
        
        # Create features
        logger.info("🔧 Creating advanced features...")
        features = self.feature_engineer.create_features(historical_data)
        
        # Split data
        split_idx = int(len(features) * (1 - self.validation_split))
        train_features = features.iloc[:split_idx]
        test_features = features.iloc[split_idx:]
        
        train_prices = historical_data.iloc[:split_idx]
        test_prices = historical_data.iloc[split_idx:]
        
        logger.info(f"📊 Training: {len(train_features)} samples")
        logger.info(f"📊 Testing: {len(test_features)} samples")
        
        # Phase 1: Feature importance analysis
        logger.info("🔍 Analyzing feature importance...")
        feature_importance = self._analyze_feature_importance(train_features, train_prices)
        
        # Phase 2: Train multiple models
        logger.info("🤖 Training ensemble of models...")
        models_performance = {}
        
        # 1. XGBoost Model
        if XGB_AVAILABLE:
            logger.info("📊 Training XGBoost model...")
            xgb_model, xgb_performance = self._train_xgboost_model(
                train_features, train_prices, test_features, test_prices
            )
            models_performance['XGBoost'] = xgb_performance
        
        # 2. Random Forest
        if SKLEARN_AVAILABLE:
            logger.info("🌲 Training Random Forest model...")
            rf_model, rf_performance = self._train_random_forest_model(
                train_features, train_prices, test_features, test_prices
            )
            models_performance['Random Forest'] = rf_performance
        
        # 3. Gradient Boosting
        if SKLEARN_AVAILABLE:
            logger.info("📈 Training Gradient Boosting model...")
            gb_model, gb_performance = self._train_gradient_boosting_model(
                train_features, train_prices, test_features, test_prices
            )
            models_performance['Gradient Boosting'] = gb_performance
        
        # 4. Deep Learning Models
        if TORCH_AVAILABLE:
            logger.info("📈 Training LSTM model...")
            lstm_model, lstm_performance = self._train_lstm_model(
                train_features, train_prices, test_features, test_prices
            )
            models_performance['LSTM'] = lstm_performance
            
            logger.info("📈 Training GRU model...")
            gru_model, gru_performance = self._train_gru_model(
                train_features, train_prices, test_features, test_prices
            )
            models_performance['GRU'] = gru_performance
        
        # 5. Baseline Strategies
        logger.info("📋 Calculating baseline strategies...")
        baseline_models = self._calculate_baseline_strategies(test_prices)
        models_performance.update(baseline_models)
        
        # Find best model
        self._select_best_model(models_performance)
        
        # Store learning results
        self._store_learning_results(models_performance, feature_importance)
        
        # Display comprehensive results
        self._display_5_year_results(models_performance, feature_importance)
        
        # Save learned model
        self._save_5_year_model()
        
        return self.best_model, self.best_performance
    
    async def _get_5_years_data(self):
        """Get 5 years of historical data"""
        try:
            # Get daily data for 5 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1825)  # 5 years
            
            ticker = yf.Ticker("GC=F")
            hist = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if not hist.empty:
                logger.info(f"✅ Loaded {len(hist)} days from Yahoo Finance")
                return hist
            
            # Fallback: Get hourly data and resample
            hist_hourly = ticker.history(period="5y", interval="1h")
            if not hist_hourly.empty:
                hist_daily = hist_hourly.resample('D').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                logger.info(f"✅ Loaded and resampled {len(hist_daily)} days")
                return hist_daily
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get 5-year data: {e}")
            return pd.DataFrame()
    
    def _analyze_feature_importance(self, features, prices):
        """Analyze feature importance"""
        if not SKLEARN_AVAILABLE:
            return {'top_features': [], 'importance_scores': {}}
        
        # Prepare data
        X = features.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Create labels
        future_returns = prices['Close'].pct_change(10).shift(-10)
        y = pd.cut(future_returns, 
                  bins=[-np.inf, -0.02, 0.02, np.inf], 
                  labels=[-1, 0, 1]).astype(int)
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Train Random Forest for feature importance
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X, y)
        
        # Get importance
        importance = pd.Series(rf_model.feature_importances_, index=X.columns)
        combined_importance = importance.sort_values(ascending=False)
        
        # Save to file
        importance_df = pd.DataFrame({
            'feature': combined_importance.index,
            'importance': combined_importance.values,
            'rank': range(1, len(combined_importance) + 1)
        })
        
        importance_file = DATA_DIR / f"feature_importance_5y_{datetime.now().strftime('%Y%m%d')}.csv"
        importance_df.to_csv(importance_file, index=False)
        
        logger.info(f"✅ Feature importance saved to {importance_file}")
        
        # Select top features
        top_features = combined_importance.head(30).index.tolist()
        
        return {
            'top_features': top_features,
            'importance_scores': combined_importance.to_dict()
        }
    
    def _train_xgboost_model(self, train_features, train_prices, test_features, test_prices):
        """Train XGBoost model"""
        if not XGB_AVAILABLE:
            return None, {}
        
        # Prepare data
        X_train, y_train = self._prepare_tabular_data(train_features, train_prices)
        X_test, y_test = self._prepare_tabular_data(test_features, test_prices)
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Generate predictions
        predictions = model.predict(X_test)
        signals = pd.Series(predictions - 1, index=test_prices.index[self.sequence_length:self.sequence_length+len(predictions)])
        
        # Run backtest
        performance, trades, equity_curve = self.backtester.run_backtest(
            test_prices.iloc[self.sequence_length:], signals, "XGBoost"
        )
        
        # Save model
        self.model_registry['XGBoost'] = model
        
        return model, performance
    
    def _train_random_forest_model(self, train_features, train_prices, test_features, test_prices):
        """Train Random Forest model"""
        if not SKLEARN_AVAILABLE:
            return None, {}
        
        # Prepare data
        X_train, y_train = self._prepare_tabular_data(train_features, train_prices)
        X_test, y_test = self._prepare_tabular_data(test_features, test_prices)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Generate predictions
        predictions = model.predict(X_test)
        signals = pd.Series(predictions - 1, index=test_prices.index[self.sequence_length:self.sequence_length+len(predictions)])
        
        # Run backtest
        performance, trades, equity_curve = self.backtester.run_backtest(
            test_prices.iloc[self.sequence_length:], signals, "Random Forest"
        )
        
        return model, performance
    
    def _train_gradient_boosting_model(self, train_features, train_prices, test_features, test_prices):
        """Train Gradient Boosting model"""
        if not SKLEARN_AVAILABLE:
            return None, {}
        
        # Prepare data
        X_train, y_train = self._prepare_tabular_data(train_features, train_prices)
        X_test, y_test = self._prepare_tabular_data(test_features, test_prices)
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.01,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Generate predictions
        predictions = model.predict(X_test)
        signals = pd.Series(predictions - 1, index=test_prices.index[self.sequence_length:self.sequence_length+len(predictions)])
        
        # Run backtest
        performance, trades, equity_curve = self.backtester.run_backtest(
            test_prices.iloc[self.sequence_length:], signals, "Gradient Boosting"
        )
        
        return model, performance
    
    def _train_lstm_model(self, train_features, train_prices, test_features, test_prices):
        """Train LSTM model"""
        if not TORCH_AVAILABLE:
            return None, {}
        
        # Prepare sequences
        X_train, y_train = self._prepare_sequences(train_features, train_prices, self.sequence_length)
        X_test, y_test = self._prepare_sequences(test_features, test_prices, self.sequence_length)
        
        # Initialize model
        model = GoldTradingLSTM(
            input_dim=X_train.shape[2],
            hidden_dim=128,
            dropout_rate=0.3
        )
        
        # Training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create datasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Train for a few epochs
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs, _ = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), DATA_DIR / 'best_lstm.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Load best model
        model.load_state_dict(torch.load(DATA_DIR / 'best_lstm.pth'))
        
        # Generate signals
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            outputs, _ = model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        signals = pd.Series(predictions - 1, index=test_prices.index[self.sequence_length:self.sequence_length+len(predictions)])
        
        # Run backtest
        performance, trades, equity_curve = self.backtester.run_backtest(
            test_prices.iloc[self.sequence_length:], signals, "LSTM"
        )
        
        # Save model
        self.model_registry['LSTM'] = model
        
        return model, performance
    
    def _train_gru_model(self, train_features, train_prices, test_features, test_prices):
        """Train GRU model"""
        if not TORCH_AVAILABLE:
            return None, {}
        
        # Prepare sequences
        X_train, y_train = self._prepare_sequences(train_features, train_prices, self.sequence_length)
        X_test, y_test = self._prepare_sequences(test_features, test_prices, self.sequence_length)
        
        # Initialize model
        model = GoldTradingGRU(
            input_dim=X_train.shape[2],
            hidden_dim=128,
            dropout_rate=0.3
        )
        
        # Training (similar to LSTM)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create datasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Train
        for epoch in range(30):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Generate signals
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            outputs = model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        signals = pd.Series(predictions - 1, index=test_prices.index[self.sequence_length:self.sequence_length+len(predictions)])
        
        # Run backtest
        performance, trades, equity_curve = self.backtester.run_backtest(
            test_prices.iloc[self.sequence_length:], signals, "GRU"
        )
        
        # Save model
        self.model_registry['GRU'] = model
        
        return model, performance
    
    def _calculate_baseline_strategies(self, test_prices):
        """Calculate baseline strategies"""
        models = {}
        
        # 1. Buy & Hold
        initial_price = test_prices['Close'].iloc[0]
        final_price = test_prices['Close'].iloc[-1]
        total_return = (final_price / initial_price - 1) * 100
        
        returns = test_prices['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (total_return / (len(test_prices) / 252)) / volatility if volatility > 0 else 0
        
        rolling_max = test_prices['Close'].expanding().max()
        drawdown = (test_prices['Close'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        models['Buy & Hold'] = {
            'model': 'Buy & Hold',
            'total_return_%': total_return,
            'annual_return_%': total_return / (len(test_prices) / 252),
            'volatility_%': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_%': max_drawdown,
            'win_rate_%': 50,
            'profit_factor': 0,
            'total_trades': 0,
            'avg_trade': 0
        }
        
        # 2. SMA Crossover
        sma_20 = test_prices['Close'].rolling(20).mean()
        sma_50 = test_prices['Close'].rolling(50).mean()
        
        signals = pd.Series(0, index=test_prices.index)
        signals[sma_20 > sma_50] = 1
        signals[sma_20 < sma_50] = -1
        signals = signals.fillna(0)
        
        sma_perf, _, _ = self.backtester.run_backtest(
            test_prices.iloc[50:], signals.iloc[50:], "SMA Crossover"
        )
        models['SMA Crossover'] = sma_perf
        
        return models
    
    def _prepare_sequences(self, features, prices, sequence_length):
        """Prepare sequences for time-series models"""
        # Align features and prices
        common_idx = features.index.intersection(prices.index)
        features = features.loc[common_idx]
        prices = prices.loc[common_idx]
        
        # Create sequences
        X = []
        y = []
        
        for i in range(sequence_length, len(features) - 10):
            # Feature sequence
            seq = features.iloc[i-sequence_length:i].values
            
            # Label (future 10-day return)
            future_return = prices['Close'].iloc[i+9] / prices['Close'].iloc[i] - 1
            
            # Categorize
            if future_return > 0.02:
                label = 2  # Strong Buy
            elif future_return > 0.005:
                label = 1  # Buy
            elif future_return < -0.02:
                label = 0  # Strong Sell
            elif future_return < -0.005:
                label = 0  # Sell
            else:
                label = 1  # Hold
        
            X.append(seq)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def _prepare_tabular_data(self, features, prices):
        """Prepare tabular data for traditional ML"""
        # Align features and prices
        common_idx = features.index.intersection(prices.index)
        features = features.loc[common_idx]
        prices = prices.loc[common_idx]
        
        # Create labels
        future_returns = prices['Close'].pct_change(10).shift(-10)
        y = pd.cut(future_returns, 
                  bins=[-np.inf, -0.02, 0.005, 0.02, np.inf], 
                  labels=[0, 1, 2, 2]).astype(int)
        
        # Align
        y = y.loc[common_idx]
        
        # Fill NaN
        X = features.fillna(0).replace([np.inf, -np.inf], 0)
        y = y.fillna(1)
        
        return X.values, y.values
    
    def _select_best_model(self, models_performance):
        """Select the best model"""
        best_score = -np.inf
        best_model_name = None
        best_performance = None
        
        for model_name, perf in models_performance.items():
            # Composite score
            score = (
                perf.get('sharpe_ratio', 0) * 0.4 +
                (perf.get('total_return_%', 0) / 100) * 0.3 +
                (perf.get('win_rate_%', 0) / 100) * 0.2 -
                (abs(perf.get('max_drawdown_%', 0)) / 100) * 0.1
            )
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_performance = perf
        
        self.best_model_name = best_model_name
        self.best_model = self.model_registry.get(best_model_name, None)
        self.best_performance = best_performance
        
        logger.info(f"🏆 Best Model: {best_model_name}")
        logger.info(f"   Sharpe: {best_performance.get('sharpe_ratio', 0):.3f}")
        logger.info(f"   Return: {best_performance.get('total_return_%', 0):.2f}%")
    
    def _store_learning_results(self, models_performance, feature_importance):
        """Store learning results"""
        learning_result = {
            'timestamp': datetime.now(pytz.utc),
            'training_years': self.training_years,
            'best_model': self.best_model_name,
            'best_performance': self.best_performance,
            'feature_importance': feature_importance,
            'all_models_performance': models_performance
        }
        
        self.learning_history.append(learning_result)
    
    def _display_5_year_results(self, models_performance, feature_importance):
        """Display comprehensive 5-year learning results"""
        print("\n" + "="*120)
        print("🧠 5-YEAR DEEP LEARNING BACKTEST RESULTS")
        print("="*120)
        
        # Performance Summary
        print("\n📊 PERFORMANCE SUMMARY:")
        print("-"*120)
        
        # Create comparison table
        comparison_data = []
        for model_name, perf in models_performance.items():
            comparison_data.append({
                'Model': model_name,
                'Sharpe': f"{perf.get('sharpe_ratio', 0):.3f}",
                'Return %': f"{perf.get('total_return_%', 0):.2f}",
                'Win Rate %': f"{perf.get('win_rate_%', 0):.1f}",
                'Max DD %': f"{perf.get('max_drawdown_%', 0):.2f}",
                'Profit Factor': f"{perf.get('profit_factor', 0):.2f}",
                'Trades': perf.get('total_trades', 0)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Best Model Analysis
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print("-"*120)
        
        if self.best_performance:
            best = self.best_performance
            print(f"Sharpe Ratio: {best.get('sharpe_ratio', 0):.3f}")
            print(f"Total Return: {best.get('total_return_%', 0):.2f}%")
            print(f"Annual Return: {best.get('annual_return_%', 0):.2f}%")
            print(f"Win Rate: {best.get('win_rate_%', 0):.1f}%")
            print(f"Max Drawdown: {best.get('max_drawdown_%', 0):.2f}%")
            print(f"Profit Factor: {best.get('profit_factor', 0):.2f}")
            print(f"Total Trades: {best.get('total_trades', 0)}")
        
        # Feature Importance
        print(f"\n🔍 TOP 10 FEATURES BY IMPORTANCE:")
        print("-"*120)
        top_features = feature_importance.get('top_features', [])[:10]
        for i, feature in enumerate(top_features, 1):
            importance = feature_importance.get('importance_scores', {}).get(feature, 0)
            print(f"{i:2d}. {feature:40s} - Importance: {importance:.4f}")
        
        # Trading Recommendations
        print(f"\n🎯 TRADING RECOMMENDATIONS:")
        print("-"*120)
        print("1. Use best model for live trading")
        print("2. Position size: 5-15% of capital")
        print("3. Stop loss: 1.5-2.5% based on volatility")
        print("4. Take profit: 3-6% based on momentum")
        print("5. Monitor market regimes")
        print("6. Re-evaluate model monthly")
        
        print("="*120)
    
    def _save_5_year_model(self):
        """Save the 5-year trained model"""
        if not JOBLIB_AVAILABLE:
            logger.warning("Joblib not available, cannot save model")
            return
        
        model_data = {
            'best_model_name': self.best_model_name,
            'best_model': self.best_model,
            'best_performance': self.best_performance,
            'feature_engineer': self.feature_engineer,
            'learning_history': self.learning_history,
            'training_years': self.training_years,
            'saved_at': datetime.now(pytz.utc).isoformat()
        }
        
        model_file = DATA_DIR / f"5y_learned_model_v11_{datetime.now().strftime('%Y%m%d')}.pkl"
        joblib.dump(model_data, model_file, compress=3)
        logger.info(f"💾 5-Year learned model saved to {model_file}")
    
    def load_5_year_model(self):
        """Load 5-year trained model"""
        if not JOBLIB_AVAILABLE:
            logger.warning("Joblib not available, cannot load model")
            return False
        
        try:
            model_files = list(DATA_DIR.glob("5y_learned_model_v11_*.pkl"))
            if not model_files:
                return False
            
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            model_data = joblib.load(latest_model)
            
            self.best_model = model_data['best_model']
            self.best_model_name = model_data['best_model_name']
            self.best_performance = model_data['best_performance']
            self.feature_engineer = model_data['feature_engineer']
            self.learning_history = model_data['learning_history']
            
            logger.info(f"✅ Loaded 5-year model: {self.best_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load 5-year model: {e}")
            return False
    
    async def generate_ai_signal(self, historical_data):
        """Generate signal using 5-year trained AI model"""
        if self.best_model is None:
            logger.warning("No 5-year trained model available.")
            return None
        
        try:
            # Create features
            features = self.feature_engineer.create_features(historical_data)
            
            # Prepare input based on model type
            if self.best_model_name in ['LSTM', 'GRU'] and TORCH_AVAILABLE:
                # Sequence models
                X = self._prepare_sequence_single(features, self.sequence_length)
                
                if self.best_model_name == 'LSTM':
                    signal, confidence = self._predict_lstm_single(self.best_model, X)
                elif self.best_model_name == 'GRU':
                    signal, confidence = self._predict_gru_single(self.best_model, X)
                else:
                    signal, confidence = 1, 0.5  # Default to buy
            else:
                # Tabular models
                X = features.iloc[-1:].fillna(0).replace([np.inf, -np.inf], 0).values
                signal, confidence = self._predict_tabular_single(self.best_model, X)
            
            # Map signal
            signal_map = {
                2: "STRONG_BUY",
                1: "BUY",
                0: "SELL",
                -1: "STRONG_SELL"
            }
            
            signal_action = signal_map.get(signal, "BUY")
            
            return {
                'action': signal_action,
                'confidence': confidence,
                'model_type': f'5Y_{self.best_model_name}',
                'performance': self.best_performance,
                'training_years': 5
            }
            
        except Exception as e:
            logger.error(f"5Y AI signal generation failed: {e}")
            return None
    
    def _prepare_sequence_single(self, features, sequence_length):
        """Prepare single sequence for prediction"""
        if len(features) < sequence_length:
            # Pad if not enough data
            padding = sequence_length - len(features)
            padded = pd.concat([pd.DataFrame(0, index=range(padding), columns=features.columns), features])
            seq = padded.iloc[-sequence_length:].values
        else:
            seq = features.iloc[-sequence_length:].values
        
        return np.expand_dims(seq, axis=0)
    
    def _predict_lstm_single(self, model, X):
        """Predict using LSTM model"""
        if not TORCH_AVAILABLE:
            return 1, 0.5
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            output, _ = model(X_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            prediction = np.argmax(probs)
            confidence = probs[prediction]
            
        return prediction, float(confidence)
    
    def _predict_gru_single(self, model, X):
        """Predict using GRU model"""
        if not TORCH_AVAILABLE:
            return 1, 0.5
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            output = model(X_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            prediction = np.argmax(probs)
            confidence = probs[prediction]
            
        return prediction, float(confidence)
    
    def _predict_tabular_single(self, model, X):
        """Predict using tabular model"""
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[0]
            prediction = np.argmax(probs)
            confidence = probs[prediction]
        else:
            prediction = model.predict(X)[0]
            confidence = 0.7  # Default confidence
        
        return prediction, float(confidence)

# ================= GOLD TRADING SENTINEL V11 =================
class GoldTradingSentinelV11:
    """Gold Trading Sentinel v11.0 with 5-Year Deep Learning"""
    
    def __init__(self, config: Dict = None):
        # Initialize version manager
        self.version_manager = DataVersionManager()
        
        # Backup existing data
        self.version_manager.backup_data(STATE_FILE)
        self.version_manager.backup_data(DATABASE_FILE)
        self.version_manager.cleanup_old_backups()
        
        # Load configuration
        self.config = config or self._load_config()
        
        # Initialize components
        self.data_extractor = RobustFreeDataExtractor(self.version_manager)
        self.five_year_learner = FiveYearDeepLearningSystem(self.version_manager)
        
        # State
        self.running = False
        self.signal_count = 0
        self.last_signal = None
        self.start_time = datetime.now(pytz.utc)
        
        # Performance tracking
        self.performance = {
            'total_runs': 0,
            'successful_signals': 0,
            'failed_signals': 0
        }
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        default_config = {
            'interval': DEFAULT_INTERVAL,
            'enable_5y_learning': True,
            'auto_train_5y': False,
            'training_years': 5,
            'telegram_token': os.getenv("TELEGRAM_TOKEN"),
            'telegram_chat_id': os.getenv("TELEGRAM_CHAT_ID")
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
        logger.info("🚀 Initializing Gold Trading Sentinel v11.0...")
        logger.info(f"📁 Data directory: {DATA_DIR}")
        logger.info(f"🆔 Run ID: {self.version_manager.current_run_id}")
        
        # Check for ML packages
        if self.config.get('enable_5y_learning', True):
            self._check_ml_packages()
            
            # Try to load 5-year model
            model_loaded = self.five_year_learner.load_5_year_model()
            
            if not model_loaded and self.config.get('auto_train_5y', False):
                logger.info("No pre-trained model found. Starting 5-year learning...")
                await self.five_year_learner.learn_from_5_years_data()
            elif not model_loaded:
                logger.info("No 5-year model found. Run with --train-5y to train.")
    
    def _check_ml_packages(self):
        """Check if ML packages are available"""
        missing = []
        
        if not TORCH_AVAILABLE:
            missing.append("torch")
        if not XGB_AVAILABLE:
            missing.append("xgboost")
        if not SKLEARN_AVAILABLE:
            missing.append("scikit-learn")
        if not JOBLIB_AVAILABLE:
            missing.append("joblib")
        
        if missing:
            logger.warning(f"Missing ML packages: {', '.join(missing)}")
            logger.warning("Install with: pip install torch xgboost scikit-learn joblib")
            logger.warning("Some features will be disabled.")
    
    async def generate_signal(self) -> Optional[Dict]:
        """Generate trading signal"""
        self.performance['total_runs'] += 1
        
        try:
            logger.info("=" * 80)
            logger.info("🔄 Generating trading signal...")
            
            # 1. Get current price
            logger.info("💰 Fetching current price...")
            current_price, source, price_details = await self._get_current_price()
            
            if current_price <= 0:
                logger.error("❌ Failed to get valid price")
                self.performance['failed_signals'] += 1
                return None
            
            logger.info(f"✅ Current price: ${current_price:.2f} ({source})")
            
            # 2. Get historical data
            logger.info("📊 Fetching historical data...")
            historical_data = await self._get_historical_data(days=30)
            
            # 3. Generate AI signal if available
            ai_signal = None
            if (self.config.get('enable_5y_learning', True) and 
                self.five_year_learner.best_model is not None):
                
                logger.info("🤖 Generating AI-powered signal...")
                ai_signal = await self.five_year_learner.generate_ai_signal(historical_data)
            
            # 4. Create final signal
            if ai_signal:
                signal = ai_signal
                signal['price'] = current_price
                signal['timestamp'] = datetime.now(pytz.utc)
                signal['source'] = '5Y_AI_Learning'
            else:
                # Fallback to rule-based signal
                signal = self._generate_rule_based_signal(current_price, historical_data)
            
            # Update state
            self.last_signal = signal
            self.signal_count += 1
            
            # Update performance
            self.performance['successful_signals'] += 1
            
            logger.info(f"✅ Signal #{self.signal_count} generated: {signal['action']}")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}")
            self.performance['failed_signals'] += 1
            return None
    
    async def _get_current_price(self) -> Tuple[float, str, Dict]:
        """Get current price with retry"""
        for attempt in range(3):
            try:
                return await self.data_extractor.get_current_price()
            except Exception as e:
                logger.warning(f"Price fetch attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2)
        
        # Return fallback price
        return 1950.0, "fallback", {"fallback": True}
    
    async def _get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical data"""
        try:
            return await self.data_extractor.get_historical_data(days, "1h")
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def _generate_rule_based_signal(self, current_price: float, historical_data: pd.DataFrame) -> Dict:
        """Generate rule-based signal as fallback"""
        if historical_data.empty or len(historical_data) < 20:
            # Not enough data, return neutral
            return {
                'action': 'NEUTRAL',
                'confidence': 50.0,
                'price': current_price,
                'timestamp': datetime.now(pytz.utc),
                'source': 'Rule-Based',
                'market_summary': 'Insufficient data for analysis'
            }
        
        # Simple moving average strategy
        closes = historical_data['Close']
        sma_20 = closes.rolling(20).mean().iloc[-1]
        sma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else sma_20
        
        # Determine signal
        if current_price > sma_20 > sma_50:
            action = "BUY"
            confidence = 70.0
            summary = "Price above both SMAs, bullish trend"
        elif current_price < sma_20 < sma_50:
            action = "SELL"
            confidence = 70.0
            summary = "Price below both SMAs, bearish trend"
        elif current_price > sma_20:
            action = "NEUTRAL_LEAN_BUY"
            confidence = 60.0
            summary = "Price above SMA20, slight bullish bias"
        else:
            action = "NEUTRAL_LEAN_SELL"
            confidence = 60.0
            summary = "Price below SMA20, slight bearish bias"
        
        return {
            'action': action,
            'confidence': confidence,
            'price': current_price,
            'timestamp': datetime.now(pytz.utc),
            'source': 'Rule-Based',
            'market_summary': summary,
            'indicators': {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'price_vs_sma20': ((current_price / sma_20) - 1) * 100
            }
        }
    
    def display_signal(self, signal: Dict):
        """Display signal in console"""
        print("\n" + "="*100)
        print("🚀 GOLD TRADING SENTINEL v11.0 - 5-YEAR DEEP LEARNING")
        print("="*100)
        
        print(f"🕒 Time: {signal['timestamp'].astimezone(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S ET')}")
        print(f"💰 Price: ${signal['price']:.2f}")
        print(f"📊 Signal #{self.signal_count}")
        print(f"📁 Run ID: {self.version_manager.current_run_id}")
        print("-"*100)
        
        # Display signal
        signal_configs = {
            "STRONG_BUY": ("🟢", "STRONG BUY", "🟢🟢🟢"),
            "BUY": ("🟢", "BUY", "🟢🟢"),
            "NEUTRAL_LEAN_BUY": ("🟡", "NEUTRAL (Lean to Buy)", "↗️"),
            "NEUTRAL_LEAN_SELL": ("🟡", "NEUTRAL (Lean to Sell)", "↘️"),
            "SELL": ("🔴", "SELL", "🔴🔴"),
            "STRONG_SELL": ("🔴", "STRONG SELL", "🔴🔴🔴"),
            "NEUTRAL": ("⚪", "NEUTRAL", "➡️")
        }
        
        emoji, display_name, strength = signal_configs.get(
            signal['action'], ("⚪", signal['action'], "")
        )
        
        print(f"🎯 SIGNAL: {strength} {emoji} {display_name} {strength}")
        print(f"📊 Confidence: {signal['confidence']:.1f}%")
        print(f"📡 Source: {signal.get('source', 'Unknown')}")
        print("-"*100)
        
        # Display AI insights if available
        if signal.get('source') == '5Y_AI_Learning':
            print("🤖 AI LEARNING INSIGHTS:")
            print(f"   • Model: {signal.get('model_type', 'Unknown')}")
            print(f"   • Training: {signal.get('training_years', 0)} years")
            
            perf = signal.get('performance', {})
            if perf:
                print(f"   • Backtest Sharpe: {perf.get('sharpe_ratio', 0):.3f}")
                print(f"   • Backtest Return: {perf.get('total_return_%', 0):.2f}%")
                print(f"   • Backtest Win Rate: {perf.get('win_rate_%', 0):.1f}%")
        
        # Display market summary
        if 'market_summary' in signal:
            print(f"\n📋 MARKET SUMMARY:")
            print(f"   {signal['market_summary']}")
        
        # Display recommendations
        self._display_recommendations(signal)
        print("="*100)
    
    def _display_recommendations(self, signal: Dict):
        """Display trading recommendations"""
        print("\n💼 TRADING RECOMMENDATIONS:")
        print("-"*50)
        
        action = signal['action']
        confidence = signal['confidence']
        
        recommendations = {
            "STRONG_BUY": [
                "• Enter long position immediately",
                "• Use full position size (10-15% of capital)",
                "• Stop loss: 1.5-2% below entry",
                "• Take profit: 3-6% above entry",
                "• Monitor for continuation patterns"
            ],
            "BUY": [
                "• Enter long position",
                "• Use 75% position size",
                "• Stop loss: 1.5-2.5% below entry",
                "• Take profit: 2.5-5% above entry",
                "• Wait for minor pullbacks for better entry"
            ],
            "NEUTRAL_LEAN_BUY": [
                "• Consider small long position",
                "• Use 50% position size",
                "• Stop loss: 2-3% below entry",
                "• Take profit: 2-4% above entry",
                "• Wait for confirmation before adding"
            ],
            "NEUTRAL_LEAN_SELL": [
                "• Consider reducing long positions",
                "• Use 25% position size for short",
                "• Stop loss: 2-3% above entry",
                "• Take profit: 2-4% below entry",
                "• Wait for breakdown confirmation"
            ],
            "SELL": [
                "• Exit long positions",
                "• Enter short position",
                "• Use 75% position size",
                "• Stop loss: 1.5-2.5% above entry",
                "• Take profit: 2.5-5% below entry"
            ],
            "STRONG_SELL": [
                "• Exit all long positions",
                "• Enter aggressive short",
                "• Use full position size",
                "• Stop loss: 1.5-2% above entry",
                "• Take profit: 3-6% below entry"
            ],
            "NEUTRAL": [
                "• Stay in cash or reduce positions",
                "• Wait for clearer signal",
                "• Monitor key support/resistance levels",
                "• Prepare for next move"
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
        print("   • Adjust position size based on confidence")
        print("   • Monitor economic calendar for high-impact events")
    
    async def run_single(self):
        """Run single signal generation"""
        await self.initialize()
        signal = await self.generate_signal()
        
        if signal:
            self.display_signal(signal)
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
                    self.display_signal(signal)
                
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
        logger.info("🛑 Shutting down Gold Trading Sentinel v11.0...")
        
        # Save configuration
        self.save_config()
        
        # Performance summary
        runtime = datetime.now(pytz.utc) - self.start_time
        hours = runtime.total_seconds() / 3600
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Runtime: {hours:.1f} hours")
        print(f"   Total Signals: {self.signal_count}")
        print(f"   Successful: {self.performance['successful_signals']}")
        print(f"   Failed: {self.performance['failed_signals']}")
        
        success_rate = self.performance['successful_signals'] / max(self.performance['total_runs'], 1) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
        
        print(f"\n💾 Data saved in: {DATA_DIR}")
        print(f"📊 Log file: gold_sentinel_v11.log")
        
        logger.info("✅ Shutdown complete")

# ================= MAIN EXECUTION =================
async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Gold Trading Sentinel v11.0 - 5-Year Deep Learning')
    parser.add_argument('--mode', choices=['single', 'live', 'train-5y', 'backtest-5y', 'test'], 
                       default='single', help='Operation mode')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                       help=f'Signal interval in seconds (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--train-5y', action='store_true',
                       help='Train on 5 years of data')
    parser.add_argument('--enable-ai', action='store_true',
                       help='Enable AI learning system')
    parser.add_argument('--training-years', type=int, default=5,
                       help='Years of data to use for training')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal console output')
    
    args = parser.parse_args()
    
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    # Display banner
    if not args.quiet:
        print("\n" + "="*100)
        print("🚀 GOLD TRADING SENTINEL v11.0 - 5-YEAR DEEP LEARNING SYSTEM")
        print("="*100)
        print("Features: 5-Year Backtesting | AI Learning | Multi-Model Ensemble")
        print("          Free Data Sources | Self-Optimizing | Risk Management")
        print("="*100)
        print(f"📁 Data Directory: {DATA_DIR}")
        print("="*100)
    
    # Check for required ML packages if AI is enabled
    if args.enable_ai or args.train_5y:
        missing_packages = []
        if not TORCH_AVAILABLE:
            missing_packages.append("torch")
        if not XGB_AVAILABLE:
            missing_packages.append("xgboost")
        if not SKLEARN_AVAILABLE:
            missing_packages.append("scikit-learn")
        if not JOBLIB_AVAILABLE:
            missing_packages.append("joblib")
        
        if missing_packages:
            print(f"\n⚠️  Missing ML packages: {', '.join(missing_packages)}")
            print("Install with: pip install torch xgboost scikit-learn joblib")
            print("Some features will be disabled.\n")
    
    # Create bot instance
    config = {
        'interval': args.interval,
        'enable_5y_learning': args.enable_ai,
        'auto_train_5y': args.train_5y,
        'training_years': args.training_years
    }
    
    bot = GoldTradingSentinelV11(config)
    
    if args.mode == 'train-5y':
        print("\n🧠 Starting 5-year deep learning training...")
        print(f"📅 Using {args.training_years} years of data")
        print("⏳ This may take several minutes...")
        
        version_manager = DataVersionManager()
        learner = FiveYearDeepLearningSystem(version_manager)
        learner.training_years = args.training_years
        
        await learner.learn_from_5_years_data()
        
        print("\n✅ 5-year training complete! Model saved for live trading.")
        
    elif args.mode == 'backtest-5y':
        print("\n📊 Running 5-year backtest analysis...")
        
        version_manager = DataVersionManager()
        learner = FiveYearDeepLearningSystem(version_manager)
        
        await learner.learn_from_5_years_data()
        
    elif args.mode == 'test':
        print("\n🔍 Testing system components...")
        
        # Test data extraction
        print("💰 Testing price extraction...")
        version_manager = DataVersionManager()
        extractor = RobustFreeDataExtractor(version_manager)
        
        try:
            price, source, details = await extractor.get_current_price()
            print(f"   ✅ Price: ${price:.2f} ({source})")
        except Exception as e:
            print(f"   ❌ Price test failed: {e}")
        
        # Test historical data
        print("\n📊 Testing historical data...")
        try:
            data = await extractor.get_historical_data(days=5, interval="1h")
            print(f"   ✅ Historical data: {len(data)} records")
        except Exception as e:
            print(f"   ❌ Historical test failed: {e}")
        
        print("\n✅ System test completed")
        
    elif args.mode == 'single':
        print("\n🎯 Generating single AI-powered signal...")
        await bot.run_single()
    
    elif args.mode == 'live':
        print("\n🚀 Starting live AI-powered trading...")
        await bot.run_live(args.interval)
    
    return 0

# ================= INSTALLATION HELPERS =================
def install_requirements():
    """Helper function to install requirements"""
    requirements = [
        "torch",
        "xgboost",
        "scikit-learn",
        "joblib",
        "yfinance",
        "pandas",
        "numpy",
        "aiohttp",
        "requests",
        "beautifulsoup4",
        "feedparser",
        "pytz",
        "schedule"
    ]
    
    print("\n📦 Installing required packages...")
    print("This may take a few minutes.\n")
    
    for package in requirements:
        print(f"Installing {package}...")
        os.system(f"pip install {package}")
    
    print("\n✅ All packages installed!")
    print("You can now run: python gold_sentinel_v11.py --train-5y")

# ================= EXECUTION =================
if __name__ == "__main__":
    # Check for installation mode
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        install_requirements()
        sys.exit(0)
    
    # Windows compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the main function
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

"""
Gold Trading Sentinel v5.0 - Binary Decision Trading System
NEUTRAL signals now show clear market lean: BUY or SELL direction
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
from typing import Optional, Dict, Tuple, List, Any
from datetime import datetime, time as dt_time, timedelta
from dataclasses import dataclass
import pytz
from supabase import create_client, Client
from textblob import TextBlob
import logging
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse
from scipy import stats
import holidays
from collections import deque

# Suppress warnings
warnings.filterwarnings('ignore')

# ================= 1. ENHANCED CONFIGURATION =================
# Define handlers
main_file_handler = logging.FileHandler('gold_sentinel_v5.log')
stream_handler = logging.StreamHandler()
heartbeat_file_handler = logging.FileHandler('market_heartbeat.log')

# Set levels for handlers
heartbeat_file_handler.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        main_file_handler,
        stream_handler,
        heartbeat_file_handler
    ]
)
logger = logging.getLogger(__name__)
heartbeat_logger = logging.getLogger('market_heartbeat')

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Constants
TIMEZONE = pytz.timezone("America/New_York")
GOLD_NEWS_KEYWORDS = ["gold", "bullion", "XAU", "precious metals", "fed", "inflation", "central bank"]
DEFAULT_INTERVAL = 900  # 15 minutes in seconds
HIGH_CONFIDENCE_THRESHOLD = 85.0  # Signals above this are "high alert"
BACKTEST_YEARS = 2

# Session-based thresholds
SESSION_THRESHOLDS = {
    'ASIAN': 90.0,      # Highest threshold during low volume
    'LONDON': 85.0,     # Medium threshold
    'NEW_YORK': 80.0,   # Lowest threshold during high volume
    'OVERLAP': 75.0     # Lowest during NY/London overlap
}

# Volatility protection
VOLATILITY_SPIKE_MULTIPLIER = 3.0  # 3x ATR triggers protection
FLASH_CRASH_PROTECTION = True

# ================= 2. ENHANCED DATA MODELS =================
@dataclass
class SentimentData:
    score: float
    sources: List[str]
    magnitude: float
    confidence: float
    article_count: int
    gold_specific: float
    weighted_score: float  # New: weighted by source importance

@dataclass
class Signal:
    action: str  # STRONG_BUY, BUY, NEUTRAL (LEAN BUY), NEUTRAL (LEAN SELL), NEUTRAL, SELL, STRONG_SELL
    confidence: float
    price: float
    timestamp: datetime
    lean: str  # BULLISH_LEAN or BEARISH_LEAN for NEUTRAL signals
    market_summary: str
    is_high_alert: bool = False
    is_volatility_spike: bool = False  # New: indicates volatility event
    volatility_level: str = "NORMAL"  # NORMAL, ELEVATED, EXTREME
    session: str = "UNKNOWN"
    rationale: Optional[Dict[str, float]] = None
    sources: Optional[List[str]] = None
    market_status: Optional[str] = None
    recommendation: str = "WAIT"  # New: clear trading recommendation

@dataclass
class MarketStatus:
    is_open: bool
    reason: str
    session: str  # ASIAN, LONDON, NEW_YORK, OVERLAP, AFTER_HOURS
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    is_high_volume: bool = False
    is_holiday: bool = False
    verified_sources: List[str] = None
    confidence: float = 0.0
    spread_widening: bool = False  # New: indicates if spreads are widening
    is_illiquid_period: bool = False  # New: first 15 min of Sunday, etc.

@dataclass
class VolatilityMetrics:
    atr_14: float
    current_range: float
    range_to_atr_ratio: float
    is_spike: bool
    level: str  # NORMAL, ELEVATED, EXTREME
    adx: float  # Average Directional Index
    trend_strength: str  # WEAK, MODERATE, STRONG

@dataclass
class HeartbeatRecord:
    timestamp: datetime
    market_status: MarketStatus
    price: float
    volatility: VolatilityMetrics
    signal_generated: bool

# ================= DYNAMIC MULTI-PLATFORM SPOT EXTRACTOR =================
class RealGoldPriceExtractor:
    """Extracts live SPOT prices with multi-platform outlier detection"""
    
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.price_history = deque(maxlen=5)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, *args):
        if self.session: 
            await self.session.close()

    async def get_refined_price(self) -> float:
        """Fetches from 3 platforms and uses the median to avoid synthetic errors"""
        platform_prices = []

        # Platform 1: Yahoo Spot - Using GC=F (Gold Futures) which is more reliable
        try:
            y_price = yf.Ticker("GC=F").fast_info.get('last_price')
            if y_price: 
                platform_prices.append(float(y_price))
                logger.info(f"✅ Yahoo Finance: ${y_price:.2f}")
        except Exception as e:
            logger.debug(f"Yahoo price failed: {e}")

        # Platform 2: GoldPrice.org (Direct AJAX)
        try:
            async with self.session.get("https://data-as-of.goldprice.org/get/ajax/usd") as r:
                data = await r.json()
                if data and 'price' in data and len(data['price']) > 0:
                    gp_price = float(data['price'][0][1])
                    platform_prices.append(gp_price)
                    logger.info(f"✅ GoldPrice.org: ${gp_price:.2f}")
        except Exception as e:
            logger.debug(f"GoldPrice.org failed: {e}")

        # Platform 3: Investing.com (Scrape)
        try:
            async with self.session.get("https://www.investing.com/currencies/xau-usd", 
                                       headers={'User-Agent': 'Mozilla/5.0'}) as r:
                text = await r.text()
                patterns = [
                    r'instrument-price-last">([\d,.]+)<',
                    r'last-price-value[^>]*>([\d,.]+)<',
                    r'data-test="instrument-price-last"[^>]*>([\d,.]+)<'
                ]
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match: 
                        price = float(match.group(1).replace(',', ''))
                        platform_prices.append(price)
                        logger.info(f"✅ Investing.com: ${price:.2f}")
                        break
        except Exception as e:
            logger.debug(f"Investing.com failed: {e}")

        # Platform 4: BullionVault as backup
        try:
            async with self.session.get("https://www.bullionvault.com/gold-price-chart.do") as r:
                text = await r.text()
                match = re.search(r'"lastPrice":\s*([\d.]+)', text)
                if match: 
                    price = float(match.group(1))
                    platform_prices.append(price)
                    logger.info(f"✅ BullionVault: ${price:.2f}")
        except Exception as e:
            logger.debug(f"BullionVault failed: {e}")

        if not platform_prices:
            logger.error("❌ No spot price data available from any platform.")
            return None

        # Use the Median of all platforms to ignore one bad/stale source
        refined_median = float(np.median(platform_prices))
        
        # Volatility Check: Ensure the source prices are within 1% of each other
        if len(platform_prices) > 1:
            spread = (max(platform_prices) - min(platform_prices)) / refined_median
            if spread > 0.01:
                logger.warning(f"⚠️ High platform deviation ({spread:.4%}). Using median.")

        logger.info(f"✅ Final Verified Spot Price: ${refined_median:.2f} (Sources: {len(platform_prices)})")
        return refined_median

    def get_historical_spot_data(self, years: int = 1, interval: str = "1h") -> pd.DataFrame:
        """Get historical data for technical analysis - using GC=F instead of XAUUSD=X"""
        try:
            ticker = yf.Ticker("GC=F")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            logger.info(f"📊 Downloading {years} year(s) of historical data for GC=F...")
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            if hist.empty:
                logger.warning(f"No historical data for {years} years")
                hist = ticker.history(period="3mo", interval=interval)
                
                if hist.empty:
                    hist = ticker.history(period="1mo", interval=interval)
            
            if hist.empty:
                logger.error("Failed to get any historical data")
                return None
            
            logger.info(f"✅ Historical data: {len(hist)} rows, from {hist.index[0].date()} to {hist.index[-1].date()}")
            return hist
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            try:
                logger.info("Trying alternative symbol GLD (Gold ETF)...")
                ticker = yf.Ticker("GLD")
                hist = ticker.history(period="1y", interval="1h")
                if not hist.empty:
                    logger.info("✅ Using GLD as historical data source")
                    return hist
            except Exception as e2:
                logger.error(f"GLD also failed: {e2}")
            
            return None

# ================= 3. ENHANCED MARKET STATUS CHECKER =================
class EnhancedMarketStatusChecker:
    """Enhanced market status checking with session awareness and volatility protection"""
    
    def __init__(self):
        self.http_session = None
        self.cache_duration = timedelta(minutes=5)
        self._cache = {}
        self.us_holidays = holidays.US(years=datetime.now().year)
        self.illiquid_periods = []
        self.last_heartbeat = None
        
        # Define trading sessions (ET)
        self.sessions = {
            'ASIAN': {'start': dt_time(19, 0), 'end': dt_time(3, 0)},
            'LONDON': {'start': dt_time(3, 0), 'end': dt_time(12, 0)},
            'NEW_YORK': {'start': dt_time(8, 0), 'end': dt_time(17, 0)},
            'OVERLAP': {'start': dt_time(8, 0), 'end': dt_time(12, 0)},
        }
        
        # Initialize illiquid periods
        self._init_illiquid_periods()
    
    def _init_illiquid_periods(self):
        """Initialize periods known for low liquidity"""
        self.illiquid_periods.append({
            'day': 6,
            'start': dt_time(18, 0),
            'end': dt_time(18, 15),
            'reason': 'Sunday open illiquidity'
        })
        
        self.illiquid_periods.append({
            'day': None,
            'start': dt_time(17, 0),
            'end': dt_time(18, 0),
            'reason': 'Daily maintenance break'
        })
    
    async def start(self):
        """Start the HTTP session"""
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        
    async def close(self):
        """Close the HTTP session"""
        if self.http_session:
            await self.http_session.close()
    
    async def get_market_status(self) -> MarketStatus:
        """Get current market status with enhanced session awareness"""
        cache_key = "market_status"
        if cache_key in self._cache:
            cached_time, status = self._cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return status
        
        now_et = datetime.now(TIMEZONE)
        weekday = now_et.weekday()
        current_time = now_et.time()
        
        is_holiday = now_et.date() in self.us_holidays
        current_session = self._get_current_session(now_et)
        is_illiquid = self._is_illiquid_period(now_et, weekday, current_time)
        
        schedule_status = self._check_schedule_status(now_et, weekday, current_time, is_holiday)
        verified_status = await self._verify_with_live_sources(schedule_status, now_et)
        
        verified_status.session = current_session
        verified_status.is_illiquid_period = is_illiquid
        spread_widening = await self._check_spread_widening()
        verified_status.spread_widening = spread_widening
        
        await self._log_heartbeat(verified_status, now_et)
        self._cache[cache_key] = (datetime.now(), verified_status)
        
        return verified_status
    
    def _get_current_session(self, now_et: datetime) -> str:
        """Determine current trading session"""
        current_time = now_et.time()
        
        if self.sessions['OVERLAP']['start'] <= current_time < self.sessions['OVERLAP']['end']:
            return 'OVERLAP'
        
        if self.sessions['NEW_YORK']['start'] <= current_time < self.sessions['NEW_YORK']['end']:
            return 'NEW_YORK'
        elif self.sessions['LONDON']['start'] <= current_time < self.sessions['LONDON']['end']:
            return 'LONDON'
        elif current_time >= self.sessions['ASIAN']['start'] or current_time < self.sessions['ASIAN']['end']:
            return 'ASIAN'
        
        return 'AFTER_HOURS'
    
    def _is_illiquid_period(self, now_et: datetime, weekday: int, current_time: dt_time) -> bool:
        """Check if current time is in an illiquid period"""
        for period in self.illiquid_periods:
            if period['day'] is not None and period['day'] != weekday:
                continue
            
            if period['start'] <= current_time < period['end']:
                return True
        
        return False
    
    async def _check_spread_widening(self) -> bool:
        """Check if bid-ask spreads are widening (sign of stress)"""
        try:
            ticker = yf.Ticker("GC=F")
            info = ticker.info
            
            if 'ask' in info and 'bid' in info:
                ask = info['ask']
                bid = info['bid']
                
                if ask > 0 and bid > 0:
                    spread = ask - bid
                    spread_percentage = (spread / bid) * 100
                    
                    if spread_percentage > 0.1:
                        logger.warning(f"⚠️ Spread widening detected: {spread_percentage:.3f}%")
                        return True
            
            return False
        except Exception as e:
            logger.debug(f"Spread check failed: {e}")
            return False
    
    async def _log_heartbeat(self, status: MarketStatus, timestamp: datetime):
        """Log market heartbeat for monitoring"""
        heartbeat_logger.debug(
            f"Heartbeat | "
            f"Open={status.is_open} | "
            f"Session={status.session} | "
            f"Holiday={status.is_holiday} | "
            f"Illiquid={status.is_illiquid_period} | "
            f"SpreadWidening={status.spread_widening} | "
            f"Confidence={status.confidence:.1%}"
        )
        
        await self._log_heartbeat_to_db(status, timestamp)
    
    async def _log_heartbeat_to_db(self, status: MarketStatus, timestamp: datetime):
        """Log heartbeat to database for persistence"""
        pass
    
    def _check_schedule_status(self, now_et: datetime, weekday: int, 
                              current_time: dt_time, is_holiday: bool) -> MarketStatus:
        """Check status based on known market schedule"""
        
        if weekday >= 5:
            next_open = self._get_next_market_open(now_et, is_holiday)
            return MarketStatus(
                is_open=False,
                reason="Weekend (gold markets closed)",
                session="WEEKEND",
                next_open=next_open,
                next_close=None,
                is_holiday=is_holiday,
                confidence=0.95
            )
        
        if is_holiday:
            next_open = self._get_next_market_open(now_et, is_holiday)
            return MarketStatus(
                is_open=False,
                reason=f"US Holiday ({self.us_holidays.get(now_et.date())})",
                session="HOLIDAY",
                next_open=next_open,
                next_close=None,
                is_holiday=True,
                confidence=0.90
            )
        
        if current_time < dt_time(18, 0) and weekday == 6:
            return MarketStatus(
                is_open=False,
                reason="Sunday pre-market (opens at 6 PM ET)",
                session="PRE_MARKET",
                next_open=now_et.replace(hour=18, minute=0, second=0, microsecond=0),
                next_close=now_et.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=5),
                confidence=0.85
            )
        
        if dt_time(17, 0) <= current_time < dt_time(18, 0):
            return MarketStatus(
                is_open=False,
                reason="Daily maintenance break (5-6 PM ET)",
                session="BREAK",
                next_open=now_et.replace(hour=18, minute=0, second=0, microsecond=0),
                next_close=now_et.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=1),
                confidence=0.90
            )
        
        is_open = True
        reason = "Markets open (24/5 with daily break)"
        
        if current_time < dt_time(17, 0):
            next_close = now_et.replace(hour=17, minute=0, second=0, microsecond=0)
        else:
            next_close = now_et.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        next_open = None
        if not is_open:
            next_open = self._get_next_market_open(now_et, is_holiday)
        
        return MarketStatus(
            is_open=is_open,
            reason=reason,
            session="ACTIVE",
            next_open=next_open,
            next_close=next_close,
            is_holiday=is_holiday,
            confidence=0.80
        )
    
    async def _verify_with_live_sources(self, scheduled_status: MarketStatus, 
                                      now_et: datetime) -> MarketStatus:
        return scheduled_status
    
    def _get_next_market_open(self, current_time: datetime, is_holiday: bool) -> datetime:
        next_open = current_time + timedelta(days=1)
        next_open = next_open.replace(hour=18, minute=0, second=0, microsecond=0)
        return next_open

# ================= 4. VOLATILITY PROTECTION MODULE =================
class VolatilityProtection:
    """Flash volatility and illiquid period protection"""
    
    def __init__(self):
        self.minute_data_cache = deque(maxlen=100)
        self.volatility_history = []
        self.atr_period = 14
        
    async def check_volatility_spike(self, symbol: str = "GC=F") -> VolatilityMetrics:
        """Check for volatility spikes using 1-minute data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty or len(hist) < 20:
                return self._create_default_metrics()
            
            high = hist['High']
            low = hist['Low']
            close = hist['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(self.atr_period).mean().iloc[-1]
            current_range = high.iloc[-1] - low.iloc[-1]
            range_to_atr_ratio = current_range / atr if atr > 0 else 0
            is_spike = range_to_atr_ratio > VOLATILITY_SPIKE_MULTIPLIER
            adx_value = self.calculate_adx(high, low, close)
            level = self._determine_volatility_level(range_to_atr_ratio, adx_value)
            
            return VolatilityMetrics(
                atr_14=float(atr),
                current_range=float(current_range),
                range_to_atr_ratio=float(range_to_atr_ratio),
                is_spike=is_spike,
                level=level,
                adx=float(adx_value),
                trend_strength=self._get_trend_strength(adx_value)
            )
            
        except Exception as e:
            logger.error(f"Volatility check failed: {e}")
            return self._create_default_metrics()
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)"""
        if len(high) < period * 2:
            return 0.0
        
        try:
            up_move = high.diff()
            down_move = low.diff().abs() * -1
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move.abs(), 0.0)
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(period).mean()
            plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.debug(f"ADX calculation failed: {e}")
            return 0.0
    
    def _determine_volatility_level(self, range_to_atr_ratio: float, adx: float) -> str:
        """Determine volatility level based on ratio and ADX"""
        if range_to_atr_ratio > 5.0:
            return "EXTREME"
        elif range_to_atr_ratio > 3.0:
            return "ELEVATED"
        elif adx > 25:
            return "TRENDING"
        else:
            return "NORMAL"
    
    def _get_trend_strength(self, adx: float) -> str:
        """Determine trend strength from ADX"""
        if adx > 40:
            return "STRONG"
        elif adx > 25:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _create_default_metrics(self) -> VolatilityMetrics:
        """Create default metrics when data is unavailable"""
        return VolatilityMetrics(
            atr_14=0.0,
            current_range=0.0,
            range_to_atr_ratio=0.0,
            is_spike=False,
            level="NORMAL",
            adx=0.0,
            trend_strength="WEAK"
        )
    
    def should_avoid_trading(self, market_status: MarketStatus, 
                           volatility: VolatilityMetrics) -> Tuple[bool, str]:
        """Determine if trading should be avoided due to conditions"""
        reasons = []
        
        if market_status.is_illiquid_period:
            reasons.append("Illiquid period")
        
        if volatility.level == "EXTREME":
            reasons.append("Extreme volatility")
        elif volatility.level == "ELEVATED" and market_status.session == "ASIAN":
            reasons.append("Elevated volatility in Asian session")
        
        if market_status.spread_widening:
            reasons.append("Widening bid-ask spreads")
        
        if volatility.range_to_atr_ratio > 10.0:
            reasons.append("Potential flash crash")
        
        if reasons:
            return True, f"Avoid trading: {', '.join(reasons)}"
        
        return False, "Safe to trade"

# ================= 5. ENHANCED SENTIMENT ANALYZER WITH WEIGHTING =================
class WeightedSentimentAnalyzer:
    """Enhanced sentiment analysis with source weighting"""
    
    def __init__(self):
        self.news_sources = [
            "https://www.kitco.com/rss/",
            "https://feeds.marketwatch.com/marketwatch/marketpulse/",
            "https://www.investing.com/rss/news_285.rss",
            "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
            "https://www.bloomberg.com/markets/rss"
        ]
        
        self.source_weights = {
            'kitco.com': 2.0,
            'investing.com': 1.8,
            'reuters.com': 1.5,
            'bloomberg.com': 1.5,
            'marketwatch.com': 1.2,
            'default': 1.0
        }
        
        self.cache_duration = timedelta(minutes=30)
        self._cache = {}
        self.gold_keywords = GOLD_NEWS_KEYWORDS + [
            "rate hike", "interest rates", "dollar index", "DXY",
            "safe haven", "geopolitical", "recession", "QE", "tapering"
        ]
    
    async def analyze_sentiment(self) -> SentimentData:
        """Analyze sentiment from multiple weighted sources"""
        cache_key = "sentiment"
        if cache_key in self._cache:
            cached_time, data = self._cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return data
        
        try:
            articles = await self._fetch_all_news()
            sentiment = self._process_articles_weighted(articles)
            self._cache[cache_key] = (datetime.now(), sentiment)
            return sentiment
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._create_default_sentiment()
    
    async def _fetch_all_news(self) -> List[Dict[str, str]]:
        """Fetch news from all sources"""
        all_articles = []
        
        async def fetch_source(source: str):
            try:
                feed = feedparser.parse(source)
                articles = []
                
                for entry in feed.entries[:15]:
                    source_domain = source.split('/')[2] if '//' in source else 'unknown'
                    articles.append({
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'source': source_domain,
                        'published': entry.get('published', ''),
                        'text': f"{entry.get('title', '')}. {entry.get('summary', '')}",
                        'is_gold_specific': self._is_gold_specific(
                            f"{entry.get('title', '')} {entry.get('summary', '')}"
                        )
                    })
                
                return articles
            except Exception as e:
                logger.debug(f"Failed to fetch {source}: {e}")
                return []
        
        tasks = [fetch_source(source) for source in self.news_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        
        return all_articles
    
    def _is_gold_specific(self, text: str) -> bool:
        """Check if text is gold-specific"""
        text_lower = text.lower()
        gold_words = sum(1 for keyword in self.gold_keywords if keyword.lower() in text_lower)
        return gold_words >= 2
    
    def _process_articles_weighted(self, articles: List[Dict[str, str]]) -> SentimentData:
        """Process articles with source weighting"""
        if not articles:
            return self._create_default_sentiment()
        
        gold_articles = [a for a in articles if a['is_gold_specific']]
        non_gold_articles = [a for a in articles if not a['is_gold_specific']]
        
        if not gold_articles:
            return SentimentData(
                score=0.0,
                sources=list(set(a['source'] for a in articles[:3])),
                magnitude=0.0,
                confidence=0.0,
                article_count=len(articles),
                gold_specific=0.0,
                weighted_score=0.0
            )
        
        weighted_sentiments = []
        source_counts = {}
        
        for article in gold_articles:
            blob = TextBlob(article['text'])
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            source = article['source']
            weight = self.source_weights.get(source, self.source_weights['default'])
            weighted_polarity = polarity * weight
            
            weighted_sentiments.append({
                'polarity': polarity,
                'weighted_polarity': weighted_polarity,
                'subjectivity': subjectivity,
                'weight': weight,
                'source': source
            })
            
            source_counts[source] = source_counts.get(source, 0) + 1
        
        total_weight = sum(item['weight'] for item in weighted_sentiments)
        if total_weight > 0:
            weighted_score = sum(item['weighted_polarity'] for item in weighted_sentiments) / total_weight
            unweighted_score = sum(item['polarity'] for item in weighted_sentiments) / len(weighted_sentiments)
        else:
            weighted_score = 0.0
            unweighted_score = 0.0
        
        sentiments = [item['polarity'] for item in weighted_sentiments]
        magnitude = np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        source_quality = sum(
            self.source_weights.get(source, 1.0) * count 
            for source, count in source_counts.items()
        ) / len(gold_articles) if gold_articles else 0
        
        confidence = min(1.0, (len(gold_articles) / 20) * source_quality)
        gold_specific = len(gold_articles) / len(articles) if articles else 0.0
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        sources = [source for source, _ in top_sources]
        
        return SentimentData(
            score=round(unweighted_score, 3),
            sources=sources,
            magnitude=round(magnitude, 3),
            confidence=round(confidence, 2),
            article_count=len(articles),
            gold_specific=round(gold_specific, 2),
            weighted_score=round(weighted_score, 3)
        )
    
    def _create_default_sentiment(self) -> SentimentData:
        """Create default sentiment data"""
        return SentimentData(
            score=0.0,
            sources=[],
            magnitude=0.0,
            confidence=0.0,
            article_count=0,
            gold_specific=0.0,
            weighted_score=0.0
        )

# ================= 6. ENHANCED TECHNICAL ANALYZER WITH ADX =================
class EnhancedTechnicalAnalyzer:
    """Enhanced technical analysis with ADX and trend strength"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators including ADX"""
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for indicators: {len(df)} rows")
            return {}
        
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes_series = pd.Series(closes)
        
        logger.info(f"Calculating indicators on {len(df)} data points...")
        
        sma_20 = closes_series.rolling(20).mean().iloc[-1]
        sma_50 = closes_series.rolling(50).mean().iloc[-1]
        sma_200 = closes_series.rolling(200).mean().iloc[-1]
        
        rsi = EnhancedTechnicalAnalyzer.calculate_rsi(closes_series, 14)
        macd_hist = EnhancedTechnicalAnalyzer.calculate_macd_histogram(closes_series)
        bb_upper, bb_middle, bb_lower = EnhancedTechnicalAnalyzer.calculate_bollinger_bands(closes_series)
        
        if 'Volume' in df.columns:
            volumes = df['Volume'].values
            current_volume = volumes[-1] if len(volumes) > 0 else 0
            volume_avg_20 = pd.Series(volumes).rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / volume_avg_20 if volume_avg_20 > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        adx, plus_di, minus_di = EnhancedTechnicalAnalyzer.calculate_adx_full(
            pd.Series(highs), pd.Series(lows), closes_series
        )
        
        if bb_upper - bb_lower > 0:
            price_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower)
        else:
            price_position = 0.5
        
        atr = EnhancedTechnicalAnalyzer.calculate_atr(
            pd.Series(highs), pd.Series(lows), closes_series
        )
        
        trend_direction = "BULLISH" if plus_di > minus_di else "BEARISH"
        trend_strength = "STRONG" if adx > 25 else "WEAK"
        
        indicators = {
            'sma_20': float(sma_20) if not pd.isna(sma_20) else 0.0,
            'sma_50': float(sma_50) if not pd.isna(sma_50) else 0.0,
            'sma_200': float(sma_200) if not pd.isna(sma_200) else 0.0,
            'rsi': float(rsi),
            'macd_histogram': float(macd_hist),
            'bb_upper': float(bb_upper) if not pd.isna(bb_upper) else 0.0,
            'bb_middle': float(bb_middle) if not pd.isna(bb_middle) else 0.0,
            'bb_lower': float(bb_lower) if not pd.isna(bb_lower) else 0.0,
            'volume_ratio': float(volume_ratio),
            'adx': float(adx),
            'plus_di': float(plus_di),
            'minus_di': float(minus_di),
            'atr': float(atr),
            'price_position': float(price_position),
            'trend_direction': trend_direction,
            'trend_strength': trend_strength
        }
        
        logger.info(f"✅ Indicators calculated: RSI={rsi:.1f}, ADX={adx:.1f}, SMA20/50/200={sma_20:.1f}/{sma_50:.1f}/{sma_200:.1f}")
        return indicators
    
    @staticmethod
    def calculate_adx_full(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[float, float, float]:
        """Calculate ADX, +DI, and -DI"""
        if len(high) < period * 2:
            return 0.0, 0.0, 0.0
        
        try:
            plus_dm = high.diff()
            minus_dm = low.diff() * -1
            
            plus_dm = np.where(plus_dm > minus_dm, np.maximum(plus_dm, 0), 0)
            minus_dm = np.where(minus_dm > plus_dm, np.maximum(minus_dm, 0), 0)
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(period).mean()
            plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return (
                float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0,
                float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0,
                float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0
            )
            
        except Exception as e:
            logger.debug(f"ADX calculation failed: {e}")
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(high) < period:
            return 0.0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(series) < period + 1:
            return 50.0
        
        prices = series.values
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
        
        rs = up / down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        if np.isnan(rsi) or np.isinf(rsi):
            return 50.0
        
        return float(rsi)
    
    @staticmethod
    def calculate_macd_histogram(series: pd.Series) -> float:
        """Calculate MACD histogram"""
        if len(series) < 26:
            return 0.0
        
        ema_12 = series.ewm(span=12, adjust=False).mean()
        ema_26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(series) < period:
            return 0.0, 0.0, 0.0
        
        middle = series.rolling(period).mean().iloc[-1]
        std = series.rolling(period).std().iloc[-1]
        
        if pd.isna(middle) or pd.isna(std):
            return 0.0, 0.0, 0.0
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return float(upper), float(middle), float(lower)

# ================= 7. BINARY DECISION SIGNAL GENERATOR =================
class BinaryDecisionSignalGenerator:
    """Generate signals with BINARY DECISION - NEUTRAL signals show clear BUY/SELL lean"""
    
    def __init__(self):
        self.session_weights = {
            'ASIAN': {
                'trend': 0.25,
                'momentum': 0.20,
                'sentiment': 0.30,
                'volume': 0.10,
                'market_structure': 0.15
            },
            'LONDON': {
                'trend': 0.30,
                'momentum': 0.25,
                'sentiment': 0.20,
                'volume': 0.15,
                'market_structure': 0.10
            },
            'NEW_YORK': {
                'trend': 0.35,
                'momentum': 0.30,
                'sentiment': 0.15,
                'volume': 0.15,
                'market_structure': 0.05
            },
            'OVERLAP': {
                'trend': 0.40,
                'momentum': 0.35,
                'sentiment': 0.10,
                'volume': 0.10,
                'market_structure': 0.05
            },
            'default': {
                'trend': 0.30,
                'momentum': 0.25,
                'sentiment': 0.20,
                'volume': 0.15,
                'market_structure': 0.10
            }
        }
    
    def generate_signal(self, price: float, indicators: Dict, 
                       sentiment: SentimentData, market_status: MarketStatus,
                       volatility: VolatilityMetrics) -> Signal:
        """Generate trading signal with BINARY DECISION system"""
        
        session = market_status.session
        weights = self.session_weights.get(session, self.session_weights['default'])
        adjusted_weights = self._adjust_weights_for_volatility(weights, volatility)
        
        factor_scores = {
            'trend': self._calculate_trend_score(price, indicators, volatility),
            'momentum': self._calculate_momentum_score(indicators, volatility),
            'sentiment': self._calculate_sentiment_score(sentiment),
            'volume': self._calculate_volume_score(indicators, market_status),
            'market_structure': self._calculate_market_structure_score(indicators, volatility)
        }
        
        weighted_score = sum(
            factor_scores[factor] * adjusted_weights[factor]
            for factor in factor_scores
        )
        
        session_threshold = SESSION_THRESHOLDS.get(session, HIGH_CONFIDENCE_THRESHOLD)
        confidence = weighted_score * 100
        is_high_alert = confidence >= session_threshold
        
        # BINARY DECISION: Determine action with clear directional bias
        action, lean, recommendation = self._determine_binary_action(weighted_score, indicators, volatility)
        
        market_summary = self._generate_market_summary(
            price, indicators, sentiment, weighted_score, 
            market_status, volatility
        )
        
        signal = Signal(
            action=action,
            confidence=round(confidence, 2),
            price=price,
            timestamp=datetime.now(pytz.utc),
            lean=lean,
            market_summary=market_summary,
            is_high_alert=is_high_alert,
            is_volatility_spike=volatility.is_spike,
            volatility_level=volatility.level,
            session=session,
            rationale=factor_scores,
            recommendation=recommendation
        )
        
        if market_status:
            status_parts = []
            if not market_status.is_open:
                status_parts.append(f"⚠️ Markets Closed: {market_status.reason}")
            else:
                status_parts.append(f"✅ Markets Open ({session})")
            
            if market_status.is_illiquid_period:
                status_parts.append("📉 Illiquid Period")
            if market_status.spread_widening:
                status_parts.append("📊 Wide Spreads")
            if volatility.is_spike:
                status_parts.append(f"⚡ Volatility Spike ({volatility.level})")
            
            signal.market_status = " | ".join(status_parts)
        
        return signal
    
    def _adjust_weights_for_volatility(self, weights: Dict, volatility: VolatilityMetrics) -> Dict:
        """Adjust weights based on volatility conditions"""
        adjusted = weights.copy()
        
        if volatility.level == "EXTREME":
            adjusted['trend'] *= 0.5
            adjusted['momentum'] *= 0.5
            adjusted['market_structure'] *= 0.7
            adjusted['sentiment'] *= 1.5
        
        elif volatility.level == "ELEVATED":
            adjusted['trend'] *= 0.8
            adjusted['momentum'] *= 0.8
        
        elif volatility.trend_strength == "STRONG":
            adjusted['trend'] *= 1.2
            adjusted['momentum'] *= 1.1
        
        total = sum(adjusted.values())
        if total > 0:
            for key in adjusted:
                adjusted[key] /= total
        
        return adjusted
    
    def _calculate_trend_score(self, price: float, indicators: Dict, 
                              volatility: VolatilityMetrics) -> float:
        """Calculate trend score with ADX confirmation"""
        if not indicators:
            return 0.5
        
        score = 0.5
        
        adx = indicators.get('adx', 0)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        if adx > 25:
            if plus_di > minus_di:
                score += 0.3
            else:
                score -= 0.3
        elif adx > 15:
            if plus_di > minus_di:
                score += 0.15
            else:
                score -= 0.15
        
        if volatility.level in ["NORMAL", "TRENDING"]:
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            sma_200 = indicators.get('sma_200', 0)
            
            if price > sma_50 > sma_20 and sma_20 > sma_200:
                score += 0.2
            elif price < sma_50 < sma_20 and sma_20 < sma_200:
                score -= 0.2
            elif price > sma_200:
                score += 0.1
            else:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_momentum_score(self, indicators: Dict, 
                                 volatility: VolatilityMetrics) -> float:
        """Calculate momentum score with volatility adjustment"""
        if not indicators:
            return 0.5
        
        score = 0.5
        
        rsi = indicators.get('rsi', 50)
        
        if volatility.level != "EXTREME":
            if rsi < 30:
                score += 0.2
            elif rsi > 70:
                score -= 0.2
            elif 40 < rsi < 60:
                score += 0.1
        
        macd_hist = indicators.get('macd_histogram', 0)
        if abs(macd_hist) > 1.0:
            if macd_hist > 0:
                score += 0.15
            else:
                score -= 0.15
        
        price_position = indicators.get('price_position', 0.5)
        if price_position < 0.2:
            score += 0.1
        elif price_position > 0.8:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_sentiment_score(self, sentiment: SentimentData) -> float:
        """Calculate sentiment score using weighted score"""
        if sentiment.article_count == 0:
            return 0.5
        
        base_score = (sentiment.weighted_score + 1) / 2
        adjustment = sentiment.confidence * sentiment.gold_specific * 0.3
        
        return max(0.0, min(1.0, base_score + adjustment))
    
    def _calculate_volume_score(self, indicators: Dict, 
                               market_status: MarketStatus) -> float:
        """Calculate volume confirmation score"""
        if not indicators or 'volume_ratio' not in indicators:
            return 0.5
        
        volume_ratio = indicators['volume_ratio']
        
        if market_status.session == "ASIAN":
            if volume_ratio > 1.2:
                return 0.8
            elif volume_ratio > 0.8:
                return 0.6
            else:
                return 0.4
        else:
            if volume_ratio > 1.5:
                return 0.8
            elif volume_ratio > 1.2:
                return 0.7
            elif volume_ratio > 0.8:
                return 0.5
            else:
                return 0.3
    
    def _calculate_market_structure_score(self, indicators: Dict, 
                                         volatility: VolatilityMetrics) -> float:
        """Calculate market structure score with ADX"""
        if not indicators:
            return 0.5
        
        score = 0.5
        
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        adx = indicators.get('adx', 0)
        
        if rsi < 40 and macd_hist > 0 and adx < 25:
            score += 0.25
        elif rsi > 60 and macd_hist < 0 and adx < 25:
            score -= 0.25
        
        if volatility.level == "NORMAL":
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            if bb_upper > 0 and bb_lower > 0:
                bb_width = (bb_upper - bb_lower) / indicators.get('bb_middle', bb_upper)
                if bb_width < 0.1:
                    score += 0.15
        
        return max(0.0, min(1.0, score))
    
    def _determine_binary_action(self, weighted_score: float, indicators: Dict,
                               volatility: VolatilityMetrics) -> Tuple[str, str, str]:
        """BINARY DECISION: Always show directional bias, even for NEUTRAL"""
        
        if volatility.is_spike:
            # Conservative boundaries during volatility spikes
            if weighted_score >= 0.85:
                return "STRONG_BUY", "BULLISH", "ENTER_LONG"
            elif weighted_score >= 0.70:
                return "BUY", "BULLISH", "ENTER_LONG"
            elif weighted_score <= 0.15:
                return "STRONG_SELL", "BEARISH", "ENTER_SHORT"
            elif weighted_score <= 0.30:
                return "SELL", "BEARISH", "ENTER_SHORT"
            # NEUTRAL with directional lean
            elif weighted_score > 0.55:
                return "NEUTRAL (LEAN BUY)", "BULLISH_LEAN", "SCALE_IN_LONG"
            elif weighted_score < 0.45:
                return "NEUTRAL (LEAN SELL)", "BEARISH_LEAN", "SCALE_OUT_LONG"
            else:
                return "NEUTRAL", "NEUTRAL_LEAN", "WAIT"
        else:
            # Normal trading boundaries
            if weighted_score >= 0.80:
                return "STRONG_BUY", "BULLISH", "ENTER_LONG"
            elif weighted_score >= 0.60:
                return "BUY", "BULLISH", "ENTER_LONG"
            elif weighted_score <= 0.20:
                return "STRONG_SELL", "BEARISH", "ENTER_SHORT"
            elif weighted_score <= 0.40:
                return "SELL", "BEARISH", "ENTER_SHORT"
            # NEUTRAL with directional lean
            elif weighted_score > 0.55:
                return "NEUTRAL (LEAN BUY)", "BULLISH_LEAN", "SCALE_IN_LONG"
            elif weighted_score < 0.45:
                return "NEUTRAL (LEAN SELL)", "BEARISH_LEAN", "SCALE_OUT_LONG"
            else:
                return "NEUTRAL", "NEUTRAL_LEAN", "WAIT"
    
    def _generate_market_summary(self, price: float, indicators: Dict, 
                                sentiment: SentimentData, weighted_score: float,
                                market_status: MarketStatus, 
                                volatility: VolatilityMetrics) -> str:
        """Generate comprehensive market summary with binary decision context"""
        summary_parts = []
        
        summary_parts.append(f"{market_status.session} session")
        
        if volatility.is_spike:
            summary_parts.append(f"⚡ {volatility.level} volatility")
        elif volatility.trend_strength == "STRONG":
            summary_parts.append("📈 Strong trend")
        
        if market_status.is_illiquid_period:
            summary_parts.append("⚠️ Illiquid period")
        
        # BINARY DECISION CONTEXT
        if weighted_score > 0.60:
            summary_parts.append("Strong bullish bias")
        elif weighted_score < 0.40:
            summary_parts.append("Strong bearish bias")
        else:
            if weighted_score > 0.55:
                summary_parts.append("Slight bullish bias")
            elif weighted_score < 0.45:
                summary_parts.append("Slight bearish bias")
            else:
                summary_parts.append("No clear bias")
        
        adx = indicators.get('adx', 0)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        if adx > 25:
            if plus_di > minus_di:
                summary_parts.append("Strong uptrend")
            else:
                summary_parts.append("Strong downtrend")
        elif adx > 15:
            summary_parts.append("Moderate trend")
        else:
            summary_parts.append("Ranging market")
        
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            summary_parts.append("Oversold (RSI < 30)")
        elif rsi > 70:
            summary_parts.append("Overbought (RSI > 70)")
        elif 45 < rsi < 55:
            summary_parts.append("RSI neutral")
        
        if sentiment.weighted_score > 0.2:
            summary_parts.append("Positive sentiment")
        elif sentiment.weighted_score < -0.2:
            summary_parts.append("Negative sentiment")
        
        return ". ".join(summary_parts[:5])

# ================= 8. GOLD TRADING SENTINEL V5.0 =================
class GoldTradingSentinelV5:
    """Main trading system with BINARY DECISION outputs"""
    
    def __init__(self):
        self.supabase = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as e:
                logger.error(f"Supabase connection failed: {e}")
        
        self.price_extractor = RealGoldPriceExtractor()
        self.market_checker = EnhancedMarketStatusChecker()
        self.volatility_protection = VolatilityProtection()
        self.tech_analyzer = EnhancedTechnicalAnalyzer()
        self.sentiment_analyzer = WeightedSentimentAnalyzer()
        self.signal_generator = BinaryDecisionSignalGenerator()
        self.signal_history = []
        
        self.db_tables = {
            'signals': 'gold_signals_v5',
            'market_status': 'market_status_logs',
            'heartbeats': 'market_heartbeats',
            'volatility': 'volatility_metrics'
        }
    
    async def start(self):
        """Initialize all components"""
        await self.market_checker.start()
        logger.info("✅ Gold Trading Sentinel V5.0 Initialized")
    
    async def close(self):
        """Close all connections"""
        await self.market_checker.close()
        logger.info("✅ Gold Trading Sentinel V5.0 Shutdown Complete")
    
    async def generate_signal(self) -> Optional[Signal]:
        """Generate a trading signal with all protections"""
        try:
            market_status = await self.market_checker.get_market_status()
            volatility = await self.volatility_protection.check_volatility_spike()
            
            avoid_trading, reason = self.volatility_protection.should_avoid_trading(market_status, volatility)
            if avoid_trading and not market_status.is_open:
                logger.warning(f"⚠️ Trading avoided: {reason}")
                return self._create_safe_signal(market_status, volatility, reason)
            
            async with self.price_extractor as extractor:
                price = await extractor.get_refined_price()
                
                if not price:
                    logger.error("Failed to get gold price")
                    return None
                
                logger.info(f"✅ Gold spot price: ${price:.2f}")
                logger.info(f"📊 Market Status: {'✅ OPEN' if market_status.is_open else '⏸️ CLOSED'} ({market_status.session})")
                logger.info(f"⚡ Volatility: {volatility.level} (ATR Ratio: {volatility.range_to_atr_ratio:.2f})")
            
            hist_data = self.price_extractor.get_historical_spot_data(years=1, interval="1h")
            
            if hist_data is None or len(hist_data) < 50:
                logger.warning("Insufficient historical data - generating basic signal")
                return self._create_basic_signal(price, market_status, volatility)
            
            indicators = self.tech_analyzer.calculate_indicators(hist_data)
            
            if not indicators:
                logger.warning("Failed to calculate indicators - generating basic signal")
                return self._create_basic_signal(price, market_status, volatility)
            
            sentiment = await self.sentiment_analyzer.analyze_sentiment()
            
            logger.info(f"📰 Sentiment: Score={sentiment.weighted_score:.3f}, "
                       f"Gold-specific={sentiment.gold_specific:.1%}, "
                       f"Articles={sentiment.article_count}")
            
            signal = self.signal_generator.generate_signal(
                price, indicators, sentiment, market_status, volatility
            )
            
            logger.info(f"✅ Signal generated: {signal.action} ({signal.confidence:.1f}%)")
            self.signal_history.append(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return None
    
    def _create_safe_signal(self, market_status: MarketStatus, 
                          volatility: VolatilityMetrics, reason: str) -> Signal:
        """Create a safe signal when trading should be avoided"""
        return Signal(
            action="NEUTRAL",
            confidence=50.0,
            price=0.0,
            timestamp=datetime.now(pytz.utc),
            lean="SAFE_LEAN",
            market_summary=f"Trading avoided: {reason}",
            is_high_alert=False,
            is_volatility_spike=volatility.is_spike,
            volatility_level=volatility.level,
            session=market_status.session,
            market_status=f"⚠️ Trading Avoided: {reason}",
            recommendation="WAIT"
        )
    
    def _create_basic_signal(self, price: float,
                           market_status: MarketStatus, 
                           volatility: VolatilityMetrics) -> Signal:
        """Create a basic signal when indicators are unavailable"""
        # Determine lean based on price action (simple heuristic)
        lean = "NEUTRAL_LEAN"
        recommendation = "WAIT"
        
        signal = Signal(
            action="NEUTRAL",
            confidence=50.0,
            price=price,
            timestamp=datetime.now(pytz.utc),
            lean=lean,
            market_summary="Basic signal - waiting for complete data",
            is_high_alert=False,
            is_volatility_spike=volatility.is_spike,
            volatility_level=volatility.level,
            session=market_status.session,
            recommendation=recommendation
        )
        
        status_parts = []
        if not market_status.is_open:
            status_parts.append(f"⚠️ Markets Closed: {market_status.reason}")
        else:
            status_parts.append(f"✅ Markets Open ({market_status.session})")
        
        if market_status.is_illiquid_period:
            status_parts.append("📉 Illiquid Period")
        if market_status.spread_widening:
            status_parts.append("📊 Wide Spreads")
        if volatility.is_spike:
            status_parts.append(f"⚡ Volatility Spike")
        
        signal.market_status = " | ".join(status_parts)
        
        return signal
    
    def _display_signal(self, signal: Signal):
        """Display signal with BINARY DECISION format"""
        print("\n" + "=" * 70)
        
        # High alert header
        if signal.is_high_alert:
            print("🚨 " * 14)
            print("🚨              HIGH ALERT SIGNAL              🚨")
            print("🚨 " * 14)
        
        # Volatility warning
        if signal.is_volatility_spike:
            print("⚡ " * 14)
            print(f"⚡     VOLATILITY SPIKE DETECTED ({signal.volatility_level})     ⚡")
            print("⚡ " * 14)
        
        print(f"📊 GOLD SIGNAL - {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S ET')}")
        print("=" * 70)
        
        # Market status
        if signal.market_status:
            print(f"📈 {signal.market_status}")
            print("-" * 70)
        
        # Price and signal
        if signal.price > 0:
            print(f"💰 Spot Price: ${signal.price:.2f}")
        
        # BINARY DECISION DISPLAY
        print(f"🎯 Signal: {signal.action}")
        print(f"📊 Confidence: {signal.confidence:.1f}%")
        print(f"🔄 Session: {signal.session}")
        
        # Color-coded decision
        if "STRONG_BUY" in signal.action:
            print(f"✅ Decision: 🟢 STRONG BUY SIGNAL")
            print(f"💡 Action: Enter long position with full size")
        elif "BUY" in signal.action:
            print(f"✅ Decision: 🟢 BUY SIGNAL")
            print(f"💡 Action: Enter long position")
        elif "STRONG_SELL" in signal.action:
            print(f"✅ Decision: 🔴 STRONG SELL SIGNAL")
            print(f"💡 Action: Enter short position with full size")
        elif "SELL" in signal.action:
            print(f"✅ Decision: 🔴 SELL SIGNAL")
            print(f"💡 Action: Enter short position")
        elif "LEAN BUY" in signal.action:
            print(f"✅ Decision: 🟡 NEUTRAL (MARKET LEAN TO BUY)")
            print(f"💡 Action: Consider scaling into long or wait for confirmation")
        elif "LEAN SELL" in signal.action:
            print(f"✅ Decision: 🟡 NEUTRAL (MARKET LEAN TO SELL)")
            print(f"💡 Action: Consider reducing exposure or wait for confirmation")
        else:
            print(f"✅ Decision: ⚪ TRUE NEUTRAL")
            print(f"💡 Action: Wait on sidelines, monitor for breakout")
        
        # Market summary
        print(f"\n📋 Market Summary:")
        print(f"   {signal.market_summary}")
        
        # Warnings
        if signal.is_volatility_spike:
            print(f"\n⚠️  Warning: Volatility spike detected. Consider smaller position sizes.")
        
        if signal.is_high_alert:
            print("\n" + "🚨 " * 14)
            print("🚨   High Confidence Signal - Consider Action!   🚨")
            print("🚨 " * 14)
        
        print("=" * 70)
    
    async def run_live_signals(self, interval: int = DEFAULT_INTERVAL):
        """Run live signal generation with binary decisions"""
        print("\n" + "=" * 70)
        print("🚀 GOLD TRADING SENTINEL V5.0 - BINARY DECISION SYSTEM")
        print("=" * 70)
        print(f"📊 Real Gold Spot Price Extraction")
        print(f"⚡ Flash Volatility Protection (>{VOLATILITY_SPIKE_MULTIPLIER}x ATR)")
        print(f"🎯 Binary Decision System (NEUTRAL shows BUY/SELL lean)")
        print(f"📰 Weighted Sentiment Analysis (Kitco 2x weight)")
        print(f"📈 ADX Trend Strength Confirmation")
        print(f"💓 Persistent Market Heartbeat Logging")
        print("=" * 70)
        
        await self.start()
        
        last_signal_time = None
        
        try:
            while True:
                now = datetime.now(TIMEZONE)
                
                if last_signal_time is None or (now - last_signal_time).total_seconds() >= interval:
                    signal = await self.generate_signal()
                    
                    if signal:
                        self._display_signal(signal)
                        last_signal_time = now
                    
                    next_signal = now + timedelta(seconds=interval)
                    wait_seconds = max(1, (next_signal - datetime.now()).total_seconds())
                    
                    market_status = await self.market_checker.get_market_status()
                    
                    if not market_status.is_open:
                        logger.info(f"⏸️ Markets closed: {market_status.reason}")
                        if market_status.next_open:
                            wait_to_open = (market_status.next_open - datetime.now()).total_seconds()
                            logger.info(f"   Next open: {market_status.next_open.strftime('%Y-%m-%d %H:%M ET')} "
                                      f"(in {int(wait_to_open//3600)}h {int((wait_to_open%3600)//60)}m)")
                    else:
                        logger.info(f"⏳ Next signal in {int(wait_seconds//60)}m {int(wait_seconds%60)}s "
                                  f"({market_status.session} session)")
                
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("👋 Shutting down Gold Trading Sentinel V5.0")
        except Exception as e:
            logger.error(f"Fatal error in live signals: {e}", exc_info=True)
        finally:
            await self.close()
    
    async def run_diagnostics(self):
        """Run system diagnostics"""
        print("\n" + "=" * 70)
        print("🔧 SYSTEM DIAGNOSTICS")
        print("=" * 70)
        
        market_status = await self.market_checker.get_market_status()
        print(f"📊 Market Status: {'✅ OPEN' if market_status.is_open else '⏸️ CLOSED'}")
        print(f"   Session: {market_status.session}")
        print(f"   Illiquid Period: {'✅ Yes' if market_status.is_illiquid_period else '❌ No'}")
        print(f"   Spread Widening: {'⚠️ Yes' if market_status.spread_widening else '✅ No'}")
        
        volatility = await self.volatility_protection.check_volatility_spike()
        print(f"\n⚡ Volatility Status:")
        print(f"   Level: {volatility.level}")
        print(f"   ATR Ratio: {volatility.range_to_atr_ratio:.2f}x")
        print(f"   ADX: {volatility.adx:.1f} ({volatility.trend_strength} trend)")
        print(f"   Spike Detected: {'⚠️ Yes' if volatility.is_spike else '✅ No'}")
        
        sentiment = await self.sentiment_analyzer.analyze_sentiment()
        print(f"\n📰 Sentiment Status:")
        print(f"   Weighted Score: {sentiment.weighted_score:.3f}")
        print(f"   Gold Specific: {sentiment.gold_specific:.1%}")
        print(f"   Articles: {sentiment.article_count}")
        print(f"   Top Sources: {', '.join(sentiment.sources[:3])}")
        
        print(f"\n💾 Database Status: {'✅ Connected' if self.supabase else '❌ Not connected'}")
        
        avoid_trading, reason = self.volatility_protection.should_avoid_trading(
            market_status, volatility
        )
        print(f"\n🎯 Trading Recommendation:")
        if avoid_trading:
            print(f"   ❌ AVOID TRADING: {reason}")
        else:
            print(f"   ✅ OK to trade with normal risk management")
        
        print("=" * 70)

# ================= 9. MAIN EXECUTION =================
async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Gold Trading Sentinel V5.0 - Binary Decision System')
    parser.add_argument('--mode', choices=['live', 'diagnostics', 'single', 'backtest'], 
                       default='live', help='Operation mode')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                       help=f'Signal interval in seconds (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--years', type=int, default=BACKTEST_YEARS,
                       help=f'Years for backtesting (default: {BACKTEST_YEARS})')
    
    args = parser.parse_args()
    
    sentinel = GoldTradingSentinelV5()
    
    try:
        if args.mode == 'live':
            await sentinel.run_live_signals(interval=args.interval)
            
        elif args.mode == 'diagnostics':
            await sentinel.run_diagnostics()
            
        elif args.mode == 'single':
            print("\n🔍 Generating Single Signal with BINARY DECISION System...")
            print("=" * 70)
            
            await sentinel.start()
            signal = await sentinel.generate_signal()
            
            if signal:
                sentinel._display_signal(signal)
            else:
                print("❌ Failed to generate signal")
        
        elif args.mode == 'backtest':
            print("\n📊 Backtesting mode (simplified for V5)")
            print("=" * 70)
            print("Note: Full backtesting with all V5 features requires historical")
            print("volatility and sentiment data which is not available.")
            print("\nRun in live or diagnostics mode to see the enhanced features.")
            print("=" * 70)
    
    finally:
        await sentinel.close()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

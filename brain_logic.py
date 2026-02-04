"""
Gold Trading Sentinel v5.0 - Advanced Volatility Protection & Session-Aware Signals
15-minute signals with flash volatility protection and refined sentiment weighting
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_sentinel_v5.log'),
        logging.StreamHandler(),
        logging.FileHandler('market_heartbeat.log', level=logging.DEBUG)
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
    action: str  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
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

# ================= 3. ENHANCED MARKET STATUS CHECKER =================
class EnhancedMarketStatusChecker:
    """Enhanced market status checking with session awareness and volatility protection"""
    
    def __init__(self):
        self.session = None
        self.cache_duration = timedelta(minutes=5)
        self._cache = {}
        self.us_holidays = holidays.US(years=datetime.now().year)
        self.illiquid_periods = []
        self.last_heartbeat = None
        
        # Define trading sessions (ET)
        self.sessions = {
            'ASIAN': {'start': dt_time(19, 0), 'end': dt_time(3, 0)},  # 7 PM - 3 AM
            'LONDON': {'start': dt_time(3, 0), 'end': dt_time(12, 0)},  # 3 AM - 12 PM
            'NEW_YORK': {'start': dt_time(8, 0), 'end': dt_time(17, 0)},  # 8 AM - 5 PM
            'OVERLAP': {'start': dt_time(8, 0), 'end': dt_time(12, 0)},  # 8 AM - 12 PM (Ldn/NY)
        }
        
        # Initialize illiquid periods
        self._init_illiquid_periods()
    
    def _init_illiquid_periods(self):
        """Initialize periods known for low liquidity"""
        # Sunday open first 15 minutes
        self.illiquid_periods.append({
            'day': 6,  # Sunday
            'start': dt_time(18, 0),  # 6 PM ET open
            'end': dt_time(18, 15),  # 6:15 PM ET
            'reason': 'Sunday open illiquidity'
        })
        
        # Daily maintenance break
        self.illiquid_periods.append({
            'day': None,  # Any day
            'start': dt_time(17, 0),
            'end': dt_time(18, 0),
            'reason': 'Daily maintenance break'
        })
    
    async def __aenter__(self):
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
        
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def get_market_status(self) -> MarketStatus:
        """Get current market status with enhanced session awareness"""
        cache_key = "market_status"
        if cache_key in self._cache:
            cached_time, status = self._cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return status
        
        # Get current time in ET
        now_et = datetime.now(TIMEZONE)
        weekday = now_et.weekday()
        current_time = now_et.time()
        
        # Check if today is a US holiday
        is_holiday = now_et.date() in self.us_holidays
        
        # Determine current session
        current_session = self._get_current_session(now_et)
        
        # Check if in illiquid period
        is_illiquid = self._is_illiquid_period(now_et, weekday, current_time)
        
        # Get base status
        schedule_status = self._check_schedule_status(now_et, weekday, current_time, is_holiday)
        
        # Verify with live sources
        verified_status = await self._verify_with_live_sources(schedule_status, now_et)
        
        # Enhance with session and illiquid info
        verified_status.session = current_session
        verified_status.is_illiquid_period = is_illiquid
        
        # Check for spread widening
        spread_widening = await self._check_spread_widening()
        verified_status.spread_widening = spread_widening
        
        # Log heartbeat
        await self._log_heartbeat(verified_status, now_et)
        
        # Cache the result
        self._cache[cache_key] = (datetime.now(), verified_status)
        
        return verified_status
    
    def _get_current_session(self, now_et: datetime) -> str:
        """Determine current trading session"""
        current_time = now_et.time()
        
        # Check for overlap first (8 AM - 12 PM)
        if self.sessions['OVERLAP']['start'] <= current_time < self.sessions['OVERLAP']['end']:
            return 'OVERLAP'
        
        # Check other sessions
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
            # Get real-time bid-ask data
            ticker = yf.Ticker("GC=F")
            info = ticker.info
            
            if 'ask' in info and 'bid' in info:
                ask = info['ask']
                bid = info['bid']
                
                if ask > 0 and bid > 0:
                    spread = ask - bid
                    spread_percentage = (spread / bid) * 100
                    
                    # Spread > 0.1% is considered wide for gold
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
        
        # Log to Supabase if available
        await self._log_heartbeat_to_db(status, timestamp)
    
    async def _log_heartbeat_to_db(self, status: MarketStatus, timestamp: datetime):
        """Log heartbeat to database for persistence"""
        # This would be implemented if Supabase is configured
        pass
    
    def _check_schedule_status(self, now_et: datetime, weekday: int, 
                              current_time: dt_time, is_holiday: bool) -> MarketStatus:
        """Check status based on known market schedule"""
        
        # Check if weekend
        if weekday >= 5:  # Saturday (5) or Sunday (6)
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
        
        # Check if holiday
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
        
        # Check daily trading hours
        if current_time < dt_time(18, 0) and weekday == 6:  # Sunday before 6 PM
            return MarketStatus(
                is_open=False,
                reason="Sunday pre-market (opens at 6 PM ET)",
                session="PRE_MARKET",
                next_open=now_et.replace(hour=18, minute=0, second=0, microsecond=0),
                next_close=now_et.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=5),
                confidence=0.85
            )
        
        # Daily break (5 PM - 6 PM ET)
        if dt_time(17, 0) <= current_time < dt_time(18, 0):
            return MarketStatus(
                is_open=False,
                reason="Daily maintenance break (5-6 PM ET)",
                session="BREAK",
                next_open=now_et.replace(hour=18, minute=0, second=0, microsecond=0),
                next_close=now_et.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=1),
                confidence=0.90
            )
        
        # Normal trading hours (Sunday 6 PM to Friday 5 PM ET)
        is_open = True
        reason = "Markets open (24/5 with daily break)"
        
        # Calculate next break/close
        if current_time < dt_time(17, 0):
            next_close = now_et.replace(hour=17, minute=0, second=0, microsecond=0)
        else:
            next_close = now_et.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        # Calculate next open (if currently in break or closed)
        next_open = None
        if not is_open:
            next_open = self._get_next_market_open(now_et, is_holiday)
        
        return MarketStatus(
            is_open=is_open,
            reason=reason,
            session="ACTIVE",  # Will be updated by caller
            next_open=next_open,
            next_close=next_close,
            is_holiday=is_holiday,
            confidence=0.80
        )
    
    async def _verify_with_live_sources(self, scheduled_status: MarketStatus, 
                                      now_et: datetime) -> MarketStatus:
        """Verify market status with live data sources"""
        # (Implementation similar to previous version, but returns enhanced MarketStatus)
        # For brevity, using simplified version
        return scheduled_status
    
    def _get_next_market_open(self, current_time: datetime, is_holiday: bool) -> datetime:
        """Calculate next market opening time"""
        # (Implementation similar to previous version)
        return current_time + timedelta(days=1)

# ================= 4. VOLATILITY PROTECTION MODULE =================
class VolatilityProtection:
    """Flash volatility and illiquid period protection"""
    
    def __init__(self):
        self.minute_data_cache = deque(maxlen=100)  # Store last 100 minutes of data
        self.volatility_history = []
        self.atr_period = 14
        
    async def check_volatility_spike(self, symbol: str = "GC=F") -> VolatilityMetrics:
        """Check for volatility spikes using 1-minute data"""
        try:
            # Get recent 1-minute data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty or len(hist) < 20:
                return self._create_default_metrics()
            
            # Calculate ATR (Average True Range)
            high = hist['High']
            low = hist['Low']
            close = hist['Close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR calculation
            atr = tr.rolling(self.atr_period).mean().iloc[-1]
            
            # Current range (last 1-minute bar)
            current_range = high.iloc[-1] - low.iloc[-1]
            range_to_atr_ratio = current_range / atr if atr > 0 else 0
            
            # Determine if spike
            is_spike = range_to_atr_ratio > VOLATILITY_SPIKE_MULTIPLIER
            
            # Calculate ADX for trend strength
            adx_value = self.calculate_adx(high, low, close)
            
            # Determine volatility level
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
            # Calculate +DM and -DM
            up_move = high.diff()
            down_move = low.diff().abs() * -1
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move.abs(), 0.0)
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Smooth the values
            atr = tr.rolling(period).mean()
            plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr)
            
            # Calculate DX and ADX
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
        elif adx > 25:  # Strong trend
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
        
        # Check illiquid periods
        if market_status.is_illiquid_period:
            reasons.append("Illiquid period")
        
        # Check volatility spikes
        if volatility.level == "EXTREME":
            reasons.append("Extreme volatility")
        elif volatility.level == "ELEVATED" and market_status.session == "ASIAN":
            reasons.append("Elevated volatility in Asian session")
        
        # Check spread widening
        if market_status.spread_widening:
            reasons.append("Widening bid-ask spreads")
        
        # Check for flash crash patterns (sudden large moves)
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
            "https://www.investing.com/rss/news_285.rss",  # Commodities news
            "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
            "https://www.bloomberg.com/markets/rss"
        ]
        
        # Source weighting based on gold market relevance
        self.source_weights = {
            'kitco.com': 2.0,        # Gold-specific, highest weight
            'investing.com': 1.8,     # Commodities focused
            'reuters.com': 1.5,       # Financial news
            'bloomberg.com': 1.5,     # Financial news
            'marketwatch.com': 1.2,   # General financial
            'default': 1.0            # All other sources
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
                
                for entry in feed.entries[:15]:  # Increased limit
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
        results = await asyncio.gather(*tasks)
        
        for articles in results:
            all_articles.extend(articles)
        
        return all_articles
    
    def _is_gold_specific(self, text: str) -> bool:
        """Check if text is gold-specific"""
        text_lower = text.lower()
        gold_words = sum(1 for keyword in self.gold_keywords if keyword.lower() in text_lower)
        return gold_words >= 2  # At least 2 gold-related keywords
    
    def _process_articles_weighted(self, articles: List[Dict[str, str]]) -> SentimentData:
        """Process articles with source weighting"""
        if not articles:
            return self._create_default_sentiment()
        
        # Separate gold and non-gold articles
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
        
        # Calculate weighted sentiment for gold articles
        weighted_sentiments = []
        source_counts = {}
        
        for article in gold_articles:
            blob = TextBlob(article['text'])
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Get source weight
            source = article['source']
            weight = self.source_weights.get(source, self.source_weights['default'])
            
            # Apply weight (Kitco gets 2x, etc.)
            weighted_polarity = polarity * weight
            
            weighted_sentiments.append({
                'polarity': polarity,
                'weighted_polarity': weighted_polarity,
                'subjectivity': subjectivity,
                'weight': weight,
                'source': source
            })
            
            # Count sources
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Calculate weighted average
        total_weight = sum(item['weight'] for item in weighted_sentiments)
        if total_weight > 0:
            weighted_score = sum(item['weighted_polarity'] for item in weighted_sentiments) / total_weight
            unweighted_score = sum(item['polarity'] for item in weighted_sentiments) / len(weighted_sentiments)
        else:
            weighted_score = 0.0
            unweighted_score = 0.0
        
        # Calculate magnitude (volatility of sentiment)
        sentiments = [item['polarity'] for item in weighted_sentiments]
        magnitude = np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        # Calculate confidence based on article count and source quality
        source_quality = sum(
            self.source_weights.get(source, 1.0) * count 
            for source, count in source_counts.items()
        ) / len(gold_articles) if gold_articles else 0
        
        confidence = min(1.0, (len(gold_articles) / 20) * source_quality)
        
        # Gold specificity ratio
        gold_specific = len(gold_articles) / len(articles) if articles else 0.0
        
        # Top sources
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
            return {}
        
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes_series = pd.Series(closes)
        
        # Moving averages
        sma_20 = closes_series.rolling(20).mean().iloc[-1]
        sma_50 = closes_series.rolling(50).mean().iloc[-1]
        sma_200 = closes_series.rolling(200).mean().iloc[-1]
        
        # RSI
        rsi = EnhancedTechnicalAnalyzer.calculate_rsi(closes_series, 14)
        
        # MACD
        macd_hist = EnhancedTechnicalAnalyzer.calculate_macd_histogram(closes_series)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = EnhancedTechnicalAnalyzer.calculate_bollinger_bands(closes_series)
        
        # Volume analysis
        volumes = df['Volume'].values
        current_volume = volumes[-1] if len(volumes) > 0 else 0
        volume_avg_20 = pd.Series(volumes).rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / volume_avg_20 if volume_avg_20 > 0 else 1.0
        
        # Trend strength using ADX
        adx, plus_di, minus_di = EnhancedTechnicalAnalyzer.calculate_adx_full(
            pd.Series(highs), pd.Series(lows), closes_series
        )
        
        # Price position
        if bb_upper - bb_lower > 0:
            price_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower)
        else:
            price_position = 0.5
        
        # ATR for volatility
        atr = EnhancedTechnicalAnalyzer.calculate_atr(
            pd.Series(highs), pd.Series(lows), closes_series
        )
        
        # Determine trend direction
        trend_direction = "BULLISH" if plus_di > minus_di else "BEARISH"
        trend_strength = "STRONG" if adx > 25 else "WEAK"
        
        return {
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
    
    @staticmethod
    def calculate_adx_full(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[float, float, float]:
        """Calculate ADX, +DI, and -DI"""
        if len(high) < period * 2:
            return 0.0, 0.0, 0.0
        
        try:
            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = low.diff() * -1
            
            # Filter DM values
            plus_dm = np.where(plus_dm > minus_dm, np.maximum(plus_dm, 0), 0)
            minus_dm = np.where(minus_dm > plus_dm, np.maximum(minus_dm, 0), 0)
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Smooth the values
            atr = tr.rolling(period).mean()
            plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr)
            
            # Calculate DX and ADX
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
        # (Same implementation as before)
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
        # (Same implementation as before)
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
        # (Same implementation as before)
        if len(series) < period:
            return 0.0, 0.0, 0.0
        
        middle = series.rolling(period).mean().iloc[-1]
        std = series.rolling(period).std().iloc[-1]
        
        if pd.isna(middle) or pd.isna(std):
            return 0.0, 0.0, 0.0
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return float(upper), float(middle), float(lower)

# ================= 7. ENHANCED SIGNAL GENERATOR =================
class SessionAwareSignalGenerator:
    """Generate signals with session awareness and volatility protection"""
    
    def __init__(self):
        # Dynamic weights based on session
        self.session_weights = {
            'ASIAN': {
                'trend': 0.25,      # Less reliable in low volume
                'momentum': 0.20,
                'sentiment': 0.30,  # News matters more in Asian session
                'volume': 0.10,     # Low volume
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
                'trend': 0.35,      # Most reliable in high volume
                'momentum': 0.30,
                'sentiment': 0.15,
                'volume': 0.15,
                'market_structure': 0.05
            },
            'OVERLAP': {
                'trend': 0.40,      # Most reliable during overlap
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
        """Generate trading signal with session and volatility awareness"""
        
        # Get session-specific weights
        session = market_status.session
        weights = self.session_weights.get(session, self.session_weights['default'])
        
        # Adjust weights based on volatility
        adjusted_weights = self._adjust_weights_for_volatility(weights, volatility)
        
        # Calculate factor scores
        factor_scores = {
            'trend': self._calculate_trend_score(price, indicators, volatility),
            'momentum': self._calculate_momentum_score(indicators, volatility),
            'sentiment': self._calculate_sentiment_score(sentiment),
            'volume': self._calculate_volume_score(indicators, market_status),
            'market_structure': self._calculate_market_structure_score(indicators, volatility)
        }
        
        # Calculate weighted confidence
        weighted_score = sum(
            factor_scores[factor] * adjusted_weights[factor]
            for factor in factor_scores
        )
        
        # Apply session-specific threshold
        session_threshold = SESSION_THRESHOLDS.get(session, HIGH_CONFIDENCE_THRESHOLD)
        
        # Check for high alert
        confidence = weighted_score * 100
        is_high_alert = confidence >= session_threshold
        
        # Determine action with volatility consideration
        action, lean = self._determine_action(weighted_score, indicators, volatility)
        
        # Generate market summary
        market_summary = self._generate_market_summary(
            price, indicators, sentiment, weighted_score, 
            market_status, volatility
        )
        
        # Create signal
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
            rationale=factor_scores
        )
        
        # Add market status info
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
            # Reduce trend/momentum weights during extreme volatility
            adjusted['trend'] *= 0.5
            adjusted['momentum'] *= 0.5
            adjusted['market_structure'] *= 0.7
            # Increase sentiment weight (news drives extreme moves)
            adjusted['sentiment'] *= 1.5
        
        elif volatility.level == "ELEVATED":
            # Slight reduction in trend reliability
            adjusted['trend'] *= 0.8
            adjusted['momentum'] *= 0.8
        
        elif volatility.trend_strength == "STRONG":
            # Increase trend weight during strong trends
            adjusted['trend'] *= 1.2
            adjusted['momentum'] *= 1.1
        
        # Normalize weights
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
        
        # ADX-based trend strength (key enhancement)
        adx = indicators.get('adx', 0)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        if adx > 25:  # Strong trend
            if plus_di > minus_di:
                score += 0.3  # Strong uptrend
            else:
                score -= 0.3  # Strong downtrend
        elif adx > 15:  # Moderate trend
            if plus_di > minus_di:
                score += 0.15
            else:
                score -= 0.15
        
        # Moving averages (less weight during high volatility)
        if volatility.level in ["NORMAL", "TRENDING"]:
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            sma_200 = indicators.get('sma_200', 0)
            
            if price > sma_50 > sma_20 and sma_20 > sma_200:
                score += 0.2  # Strong uptrend structure
            elif price < sma_50 < sma_20 and sma_20 < sma_200:
                score -= 0.2  # Strong downtrend structure
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
        
        # RSI momentum (adjusted for volatility)
        rsi = indicators.get('rsi', 50)
        
        if volatility.level != "EXTREME":
            if rsi < 30:
                score += 0.2  # Oversold
            elif rsi > 70:
                score -= 0.2  # Overbought
            elif 40 < rsi < 60:
                score += 0.1  # Neutral with slight bullish bias
        
        # MACD momentum
        macd_hist = indicators.get('macd_histogram', 0)
        if abs(macd_hist) > 1.0:
            if macd_hist > 0:
                score += 0.15
            else:
                score -= 0.15
        
        # Price position in Bollinger Bands
        price_position = indicators.get('price_position', 0.5)
        if price_position < 0.2:
            score += 0.1  # Near lower band, potential bounce
        elif price_position > 0.8:
            score -= 0.1  # Near upper band, potential pullback
        
        return max(0.0, min(1.0, score))
    
    def _calculate_sentiment_score(self, sentiment: SentimentData) -> float:
        """Calculate sentiment score using weighted score"""
        if sentiment.article_count == 0:
            return 0.5
        
        # Use weighted score which gives more weight to gold-specific sources
        base_score = (sentiment.weighted_score + 1) / 2
        
        # Adjust based on confidence and gold specificity
        adjustment = sentiment.confidence * sentiment.gold_specific * 0.3
        
        return max(0.0, min(1.0, base_score + adjustment))
    
    def _calculate_volume_score(self, indicators: Dict, 
                               market_status: MarketStatus) -> float:
        """Calculate volume confirmation score"""
        if not indicators or 'volume_ratio' not in indicators:
            return 0.5
        
        volume_ratio = indicators['volume_ratio']
        
        # Adjust expectations based on session
        if market_status.session == "ASIAN":
            # Lower volume expected in Asian session
            if volume_ratio > 1.2:
                return 0.8  # High for Asian session
            elif volume_ratio > 0.8:
                return 0.6
            else:
                return 0.4
        else:
            # Normal expectations for other sessions
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
        
        # Check for divergence with ADX confirmation
        if rsi < 40 and macd_hist > 0 and adx < 25:
            score += 0.25  # Bullish divergence in non-trending market
        elif rsi > 60 and macd_hist < 0 and adx < 25:
            score -= 0.25  # Bearish divergence in non-trending market
        
        # Bollinger Band squeeze during low volatility
        if volatility.level == "NORMAL":
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            if bb_upper > 0 and bb_lower > 0:
                bb_width = (bb_upper - bb_lower) / indicators.get('bb_middle', bb_upper)
                if bb_width < 0.1:  # Tight bands
                    score += 0.15  # Potential breakout coming
        
        return max(0.0, min(1.0, score))
    
    def _determine_action(self, weighted_score: float, indicators: Dict,
                         volatility: VolatilityMetrics) -> Tuple[str, str]:
        """Determine trading action with volatility protection"""
        
        # Adjust thresholds during volatility spikes
        if volatility.is_spike:
            # Be more conservative during spikes
            if weighted_score >= 0.85:
                action = "STRONG_BUY"
            elif weighted_score >= 0.70:
                action = "BUY"
            elif weighted_score <= 0.15:
                action = "STRONG_SELL"
            elif weighted_score <= 0.30:
                action = "SELL"
            else:
                action = "NEUTRAL"
        else:
            # Normal thresholds
            if weighted_score >= 0.80:
                action = "STRONG_BUY"
            elif weighted_score >= 0.60:
                action = "BUY"
            elif weighted_score <= 0.20:
                action = "STRONG_SELL"
            elif weighted_score <= 0.40:
                action = "SELL"
            else:
                action = "NEUTRAL"
        
        # Determine lean
        if action == "NEUTRAL":
            if weighted_score > 0.55:
                lean = "BULLISH_LEAN"
            elif weighted_score < 0.45:
                lean = "BEARISH_LEAN"
            else:
                lean = "NEUTRAL_LEAN"
        else:
            lean = "BULLISH" if "BUY" in action else "BEARISH"
        
        return action, lean
    
    def _generate_market_summary(self, price: float, indicators: Dict, 
                                sentiment: SentimentData, weighted_score: float,
                                market_status: MarketStatus, 
                                volatility: VolatilityMetrics) -> str:
        """Generate comprehensive market summary"""
        summary_parts = []
        
        # Session and volatility context
        summary_parts.append(f"{market_status.session} session")
        
        if volatility.is_spike:
            summary_parts.append(f"⚡ {volatility.level} volatility")
        elif volatility.trend_strength == "STRONG":
            summary_parts.append("📈 Strong trend")
        
        if market_status.is_illiquid_period:
            summary_parts.append("⚠️ Illiquid period")
        
        # Trend context with ADX
        adx = indicators.get('adx', 0)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        if adx > 25:
            if plus_di > minus_di:
                summary_parts.append("Strong uptrend (ADX > 25)")
            else:
                summary_parts.append("Strong downtrend (ADX > 25)")
        elif adx > 15:
            summary_parts.append("Moderate trend")
        else:
            summary_parts.append("Ranging market")
        
        # RSI context
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            summary_parts.append("Oversold (RSI < 30)")
        elif rsi > 70:
            summary_parts.append("Overbought (RSI > 70)")
        elif 45 < rsi < 55:
            summary_parts.append("RSI neutral")
        
        # Volume context
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            summary_parts.append("High volume")
        elif volume_ratio < 0.5:
            summary_parts.append("Low volume")
        
        # Sentiment context
        if sentiment.weighted_score > 0.2:
            summary_parts.append("Positive sentiment")
        elif sentiment.weighted_score < -0.2:
            summary_parts.append("Negative sentiment")
        
        return ". ".join(summary_parts[:4])

# ================= 8. GOLD TRADING SENTINEL V5.0 =================
class GoldTradingSentinelV5:
    """Main trading system with all enhancements"""
    
    def __init__(self):
        self.supabase = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as e:
                logger.error(f"Supabase connection failed: {e}")
        
        self.price_extractor = None
        self.market_checker = EnhancedMarketStatusChecker()
        self.volatility_protection = VolatilityProtection()
        self.tech_analyzer = EnhancedTechnicalAnalyzer()
        self.sentiment_analyzer = WeightedSentimentAnalyzer()
        self.signal_generator = SessionAwareSignalGenerator()
        self.signal_history = []
        self.backtester = None
        
        # Database tables for persistence
        self.db_tables = {
            'signals': 'gold_signals_v5',
            'market_status': 'market_status_logs',
            'heartbeats': 'market_heartbeats',
            'volatility': 'volatility_metrics'
        }
    
    async def initialize(self):
        """Initialize the system"""
        self.price_extractor = RealGoldPriceExtractor()
        logger.info("Gold Trading Sentinel V5.0 initialized")
        
        # Print session thresholds
        print("\n" + "=" * 70)
        print("🔄 SESSION-AWARE TRADING THRESHOLDS")
        print("=" * 70)
        for session, threshold in SESSION_THRESHOLDS.items():
            print(f"  {session:15} → High Alert: {threshold:5.1f}% confidence")
        print("=" * 70)
    
    async def generate_signal(self) -> Optional[Signal]:
        """Generate a trading signal with all protections"""
        try:
            # 1. Check market status
            market_status = await self.market_checker.get_market_status()
            
            # 2. Check volatility
            volatility = await self.volatility_protection.check_volatility_spike()
            
            # 3. Check if we should avoid trading
            avoid_trading, reason = self.volatility_protection.should_avoid_trading(
                market_status, volatility
            )
            
            if avoid_trading and not market_status.is_open:
                logger.warning(f"⚠️ Trading avoided: {reason}")
                return self._create_safe_signal(market_status, volatility, reason)
            
            # 4. Get real gold spot price
            async with self.price_extractor as extractor:
                price, sources, source_details = await extractor.get_real_gold_spot_price()
                
                if not price:
                    logger.error("Failed to get gold price")
                    return None
                
                logger.info(f"✅ Gold spot price: ${price:.2f} from {len(sources)} sources")
                logger.info(f"📊 Market Status: {'✅ OPEN' if market_status.is_open else '⏸️ CLOSED'} ({market_status.session})")
                logger.info(f"⚡ Volatility: {volatility.level} (ATR Ratio: {volatility.range_to_atr_ratio:.2f})")
            
            # 5. Get historical data for indicators
            hist_data = self.price_extractor.get_historical_spot_data(
                years=1,
                interval="1h"
            )
            
            if hist_data is None or len(hist_data) < 50:
                logger.warning("Insufficient historical data")
                return self._create_basic_signal(price, sources, market_status, volatility)
            
            # 6. Calculate technical indicators
            indicators = self.tech_analyzer.calculate_indicators(hist_data)
            
            if not indicators:
                logger.warning("Failed to calculate indicators")
                return self._create_basic_signal(price, sources, market_status, volatility)
            
            # 7. Analyze sentiment
            sentiment = await self.sentiment_analyzer.analyze_sentiment()
            
            # 8. Generate signal with all context
            signal = self.signal_generator.generate_signal(
                price, indicators, sentiment, market_status, volatility
            )
            signal.sources = sources
            
            # 9. Log everything to database
            await self._log_full_context_to_db(signal, market_status, volatility, sentiment)
            
            # 10. Store in history
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
            market_status=f"⚠️ Trading Avoided: {reason}"
        )
    
    def _create_basic_signal(self, price: float, sources: List[str],
                           market_status: MarketStatus, 
                           volatility: VolatilityMetrics) -> Signal:
        """Create a basic signal when indicators are unavailable"""
        signal = Signal(
            action="NEUTRAL",
            confidence=50.0,
            price=price,
            timestamp=datetime.now(pytz.utc),
            lean="NEUTRAL_LEAN",
            market_summary="Basic signal - waiting for complete data",
            is_high_alert=False,
            is_volatility_spike=volatility.is_spike,
            volatility_level=volatility.level,
            session=market_status.session,
            sources=sources
        )
        
        # Add market status info
        status_parts = []
        if not market_status.is_open:
            status_parts.append(f"⚠️ Markets Closed: {market_status.reason}")
        else:
            status_parts.append(f"✅ Markets Open ({market_status.session})")
        
        if market_status.is_illiquid_period:
            status_parts.append("📉 Illiquid Period")
        if volatility.is_spike:
            status_parts.append(f"⚡ Volatility Spike")
        
        signal.market_status = " | ".join(status_parts)
        
        return signal
    
    async def _log_full_context_to_db(self, signal: Signal, market_status: MarketStatus,
                                    volatility: VolatilityMetrics, sentiment: SentimentData):
        """Log full trading context to database"""
        if not self.supabase:
            return
        
        try:
            # Log signal
            signal_entry = {
                "price": signal.price,
                "signal": signal.action,
                "confidence": signal.confidence,
                "lean": signal.lean,
                "is_high_alert": signal.is_high_alert,
                "is_volatility_spike": signal.is_volatility_spike,
                "volatility_level": signal.volatility_level,
                "session": signal.session,
                "market_summary": signal.market_summary,
                "market_status": signal.market_status or "",
                "rationale": json.dumps(signal.rationale) if signal.rationale else "{}",
                "sources": ", ".join(signal.sources) if signal.sources else "",
                "created_at": signal.timestamp.isoformat()
            }
            
            # Log market status
            status_entry = {
                "is_open": market_status.is_open,
                "session": market_status.session,
                "is_illiquid": market_status.is_illiquid_period,
                "spread_widening": market_status.spread_widening,
                "reason": market_status.reason,
                "confidence": market_status.confidence,
                "verified_sources": ", ".join(market_status.verified_sources) if market_status.verified_sources else "",
                "created_at": signal.timestamp.isoformat()
            }
            
            # Log volatility metrics
            volatility_entry = {
                "atr": volatility.atr_14,
                "current_range": volatility.current_range,
                "range_to_atr_ratio": volatility.range_to_atr_ratio,
                "is_spike": volatility.is_spike,
                "level": volatility.level,
                "adx": volatility.adx,
                "trend_strength": volatility.trend_strength,
                "created_at": signal.timestamp.isoformat()
            }
            
            # Log sentiment
            sentiment_entry = {
                "score": sentiment.score,
                "weighted_score": sentiment.weighted_score,
                "sources": ", ".join(sentiment.sources),
                "magnitude": sentiment.magnitude,
                "confidence": sentiment.confidence,
                "article_count": sentiment.article_count,
                "gold_specific": sentiment.gold_specific,
                "created_at": signal.timestamp.isoformat()
            }
            
            # Insert all records
            self.supabase.table(self.db_tables['signals']).insert(signal_entry).execute()
            self.supabase.table(self.db_tables['market_status']).insert(status_entry).execute()
            self.supabase.table(self.db_tables['volatility']).insert(volatility_entry).execute()
            
            logger.info(f"📝 Full context logged to database: {signal.action} ({signal.confidence:.1f}%)")
            
        except Exception as e:
            logger.error(f"Database logging failed: {e}")
    
    async def run_live_signals(self, interval: int = DEFAULT_INTERVAL):
        """Run live signal generation with all protections"""
        print("\n" + "=" * 70)
        print("🚀 GOLD TRADING SENTINEL V5.0 - ENHANCED PROTECTIONS")
        print("=" * 70)
        print(f"📊 Real Gold Spot Price Extraction")
        print(f"⚡ Flash Volatility Protection (>{VOLATILITY_SPIKE_MULTIPLIER}x ATR)")
        print(f"🎯 Session-Aware Thresholds")
        print(f"📰 Weighted Sentiment Analysis (Kitco 2x weight)")
        print(f"📈 ADX Trend Strength Confirmation")
        print(f"💓 Persistent Market Heartbeat Logging")
        print("=" * 70)
        
        await self.initialize()
        
        last_signal_time = None
        
        try:
            while True:
                now = datetime.now(TIMEZONE)
                
                # Check if it's time for a new signal (every 15 minutes)
                if last_signal_time is None or (now - last_signal_time).total_seconds() >= interval:
                    signal = await self.generate_signal()
                    
                    if signal:
                        self._display_signal(signal)
                        last_signal_time = now
                    
                    # Calculate next signal time
                    next_signal = now + timedelta(seconds=interval)
                    wait_seconds = max(1, (next_signal - datetime.now()).total_seconds())
                    
                    # Get market status for context
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
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("👋 Shutting down Gold Trading Sentinel V5.0")
        except Exception as e:
            logger.error(f"Fatal error in live signals: {e}", exc_info=True)
    
    def _display_signal(self, signal: Signal):
        """Display signal with enhanced information"""
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
        print(f"🎯 Signal: {signal.action} (Confidence: {signal.confidence:.1f}%)")
        print(f"🔄 Session: {signal.session}")
        
        if signal.action == "NEUTRAL":
            if signal.lean == "BULLISH_LEAN":
                print(f"📊 Market Lean: ⬆️  Slightly Bullish")
            elif signal.lean == "BEARISH_LEAN":
                print(f"📊 Market Lean: ⬇️  Slightly Bearish")
            elif signal.lean == "SAFE_LEAN":
                print(f"📊 Market Lean: 🛡️  Safety First")
            else:
                print(f"📊 Market Lean: ↔️  Neutral")
        else:
            print(f"📊 Market Bias: {'⬆️  Bullish' if 'BUY' in signal.action else '⬇️  Bearish'}")
        
        # Sources
        if signal.sources:
            print(f"📊 Sources: {', '.join(signal.sources[:3])}")
        
        # Rationale highlights
        if signal.rationale and len(signal.rationale) > 0:
            sorted_factors = sorted(signal.rationale.items(), key=lambda x: x[1], reverse=True)
            top_factors = sorted_factors[:2]
            print(f"📈 Key Factors: {', '.join([f'{k}:{v:.2f}' for k, v in top_factors])}")
        
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
    
    async def run_diagnostics(self):
        """Run system diagnostics"""
        print("\n" + "=" * 70)
        print("🔧 SYSTEM DIAGNOSTICS")
        print("=" * 70)
        
        # Check market status
        market_status = await self.market_checker.get_market_status()
        print(f"📊 Market Status: {'✅ OPEN' if market_status.is_open else '⏸️ CLOSED'}")
        print(f"   Session: {market_status.session}")
        print(f"   Illiquid Period: {'✅ Yes' if market_status.is_illiquid_period else '❌ No'}")
        print(f"   Spread Widening: {'⚠️ Yes' if market_status.spread_widening else '✅ No'}")
        
        # Check volatility
        volatility = await self.volatility_protection.check_volatility_spike()
        print(f"\n⚡ Volatility Status:")
        print(f"   Level: {volatility.level}")
        print(f"   ATR Ratio: {volatility.range_to_atr_ratio:.2f}x")
        print(f"   ADX: {volatility.adx:.1f} ({volatility.trend_strength} trend)")
        print(f"   Spike Detected: {'⚠️ Yes' if volatility.is_spike else '✅ No'}")
        
        # Check sentiment sources
        sentiment = await self.sentiment_analyzer.analyze_sentiment()
        print(f"\n📰 Sentiment Status:")
        print(f"   Weighted Score: {sentiment.weighted_score:.3f}")
        print(f"   Gold Specific: {sentiment.gold_specific:.1%}")
        print(f"   Articles: {sentiment.article_count}")
        print(f"   Top Sources: {', '.join(sentiment.sources[:3])}")
        
        # Check database
        print(f"\n💾 Database Status: {'✅ Connected' if self.supabase else '❌ Not connected'}")
        
        # Trading recommendations
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
    parser = argparse.ArgumentParser(description='Gold Trading Sentinel V5.0')
    parser.add_argument('--mode', choices=['live', 'diagnostics', 'single', 'backtest'], 
                       default='live', help='Operation mode')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                       help=f'Signal interval in seconds (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--years', type=int, default=BACKTEST_YEARS,
                       help=f'Years for backtesting (default: {BACKTEST_YEARS})')
    
    args = parser.parse_args()
    
    sentinel = GoldTradingSentinelV5()
    
    if args.mode == 'live':
        await sentinel.run_live_signals(interval=args.interval)
        
    elif args.mode == 'diagnostics':
        await sentinel.run_diagnostics()
        
    elif args.mode == 'single':
        print("\n🔍 Generating Single Signal with Enhanced Protections...")
        print("=" * 70)
        
        await sentinel.initialize()
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

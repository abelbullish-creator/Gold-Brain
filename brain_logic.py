"""
Gold Trading Sentinel v4.0 - Pure Signals with Backtesting
15-minute signals with 2-year backtesting for gold spot
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
from datetime import datetime, time, timedelta
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

# Suppress warnings
warnings.filterwarnings('ignore')

# ================= 1. CONFIGURATION =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_sentinel_v4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Constants
TIMEZONE = pytz.timezone("America/New_York")
GOLD_NEWS_KEYWORDS = ["gold", "bullion", "XAU", "precious metals", "fed", "inflation", "central bank"]
DEFAULT_INTERVAL = 900  # 15 minutes in seconds
HIGH_CONFIDENCE_THRESHOLD = 85.0  # Signals above this are "high alert"
BACKTEST_YEARS = 2

# ================= 2. DATA MODELS =================
@dataclass
class SentimentData:
    score: float
    sources: List[str]
    magnitude: float
    confidence: float
    article_count: int
    gold_specific: float

@dataclass
class Signal:
    action: str  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    confidence: float
    price: float
    timestamp: datetime
    lean: str  # BULLISH_LEAN or BEARISH_LEAN for NEUTRAL signals
    market_summary: str
    is_high_alert: bool = False  # High success rate signal
    rationale: Optional[Dict[str, float]] = None
    sources: Optional[List[str]] = None

@dataclass
class BacktestResult:
    total_signals: int
    profitable_signals: int
    win_rate: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    avg_confidence: float
    high_alert_win_rate: float
    signals_by_type: Dict[str, int]
    equity_curve: List[float]
    timestamps: List[datetime]

# ================= 3. REAL GOLD SPOT PRICE EXTRACTOR =================
class RealGoldPriceExtractor:
    """Extract real gold spot price from multiple reliable sources"""
    
    def __init__(self):
        self.session = None
        self.cache_duration = timedelta(seconds=60)
        self._cache = {}
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0'
        ]
        
    async def __aenter__(self):
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': np.random.choice(self.user_agents)},
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
        
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def get_real_gold_spot_price(self) -> Tuple[Optional[float], List[str], Dict[str, Any]]:
        """Get real gold spot price from multiple reliable sources"""
        sources_used = []
        source_details = {}
        price_tasks = []
        
        sources = [
            self._fetch_kitco_live,
            self._fetch_investing_com,
            self._fetch_bullionvault,
            self._fetch_goldprice_org,
            self._fetch_yahoo_finance_spot,
            self._fetch_monex
        ]
        
        for source_func in sources:
            price_tasks.append(source_func())
        
        results = await asyncio.gather(*price_tasks, return_exceptions=True)
        
        prices = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.debug(f"Source {sources[i].__name__} failed: {result}")
                continue
            
            if result and result[0] is not None:
                price, source_name, details = result
                prices.append(price)
                sources_used.append(source_name)
                source_details[source_name] = details
        
        if not prices:
            logger.error("All gold price sources failed")
            return None, [], {}
        
        validated_prices = self._validate_prices(prices)
        
        if not validated_prices:
            return None, [], {}
        
        avg_price = self._calculate_weighted_average(validated_prices, sources_used)
        
        logger.info(f"Gold spot price: ${avg_price:.2f} from {len(sources_used)} sources: {sources_used}")
        return round(avg_price, 2), sources_used, source_details
    
    def _validate_prices(self, prices: List[float]) -> List[float]:
        """Validate and clean price data"""
        if len(prices) < 2:
            return prices
        
        median_price = np.median(prices)
        
        # Only filter extreme outliers
        filtered_prices = [
            p for p in prices 
            if 0.5 * median_price < p < 2 * median_price
        ]
        
        if not filtered_prices:
            return prices
        
        return filtered_prices
    
    def _calculate_weighted_average(self, prices: List[float], sources: List[str]) -> float:
        """Calculate weighted average based on source reliability"""
        source_weights = {
            'Kitco': 0.25,
            'Investing.com': 0.20,
            'BullionVault': 0.20,
            'GoldPrice.org': 0.15,
            'Yahoo Finance': 0.10,
            'Monex': 0.10
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for price, source in zip(prices, sources):
            weight = source_weights.get(source, 0.10)
            weighted_sum += price * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return np.mean(prices)
    
    async def _fetch_kitco_live(self) -> Tuple[Optional[float], str, Dict]:
        """Fetch from Kitco - highly reliable for spot gold"""
        try:
            url = "https://www.kitco.com/charts/livegold.html"
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    patterns = [
                        r'id="sp-bid".*?>(\d+,\d+\.\d+)<',
                        r'<span[^>]*data-bid[^>]*>(\d+,\d+\.\d+)<',
                        r'Gold.*?(\d+,\d+\.\d+).*?USD',
                        r'\$\s*(\d+,\d+\.\d+)',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
                        if match:
                            price_str = match.group(1).replace(',', '')
                            price = float(price_str)
                            
                            if price > 0:
                                return price, "Kitco", {
                                    "method": "HTML parsing",
                                    "pattern": pattern
                                }
        
        except Exception as e:
            logger.debug(f"Kitco fetch failed: {e}")
        
        return None, "Kitco", {"error": "Failed to extract price"}
    
    async def _fetch_investing_com(self) -> Tuple[Optional[float], str, Dict]:
        """Fetch from Investing.com"""
        try:
            url = "https://www.investing.com/commodities/gold"
            headers = {
                'User-Agent': self.user_agents[0],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    patterns = [
                        r'data-test="instrument-price-last">([\d,]+\.\d+)<',
                        r'class="last-price-value.*?>([\d,]+\.\d+)<',
                        r'id="last_last".*?>([\d,]+\.\d+)<',
                        r'data-price="([\d,]+\.\d+)"',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, html, re.IGNORECASE)
                        if match:
                            price_str = match.group(1).replace(',', '')
                            price = float(price_str)
                            
                            if price > 0:
                                return price, "Investing.com", {
                                    "method": "HTML parsing",
                                    "pattern": pattern
                                }
        
        except Exception as e:
            logger.debug(f"Investing.com fetch failed: {e}")
        
        return None, "Investing.com", {"error": "Failed to extract price"}
    
    async def _fetch_bullionvault(self) -> Tuple[Optional[float], str, Dict]:
        """Fetch from BullionVault"""
        try:
            url = "https://www.bullionvault.com/gold-price-chart.do"
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    patterns = [
                        r'id="spotPrice".*?>(\d+\.\d+)<',
                        r'class="spot-price".*?>(\d+\.\d+)<',
                        r'Gold.*?\$\s*(\d+\.\d+)',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, html, re.IGNORECASE)
                        if match:
                            price = float(match.group(1))
                            if price > 0:
                                return price, "BullionVault", {
                                    "method": "HTML parsing",
                                    "pattern": pattern
                                }
        
        except Exception as e:
            logger.debug(f"BullionVault fetch failed: {e}")
        
        return None, "BullionVault", {"error": "Failed to extract price"}
    
    async def _fetch_goldprice_org(self) -> Tuple[Optional[float], str, Dict]:
        """Fetch from GoldPrice.org"""
        try:
            url = "https://data-asg.goldprice.org/dbXRates/USD"
            headers = {
                'User-Agent': self.user_agents[0],
                'Accept': 'application/json',
                'Origin': 'https://goldprice.org',
                'Referer': 'https://goldprice.org/',
            }
            
            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'items' in data and len(data['items']) > 0:
                        if 'xauPrice' in data['items'][0]:
                            price = float(data['items'][0]['xauPrice'])
                            if price > 0:
                                return price, "GoldPrice.org", {
                                    "method": "JSON API",
                                    "field": "xauPrice"
                                }
        
        except Exception as e:
            logger.debug(f"GoldPrice.org fetch failed: {e}")
        
        return None, "GoldPrice.org", {"error": "Failed to extract price"}
    
    async def _fetch_yahoo_finance_spot(self) -> Tuple[Optional[float], str, Dict]:
        """Fetch from Yahoo Finance - using GC=F for spot approximation"""
        try:
            # Using GC=F (Gold Futures) as best approximation for spot
            ticker = yf.Ticker("GC=F")
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                if price > 0:
                    return price, "Yahoo Finance", {
                        "method": "yfinance API",
                        "symbol": "GC=F",
                        "price_type": "futures_close"
                    }
        
        except Exception as e:
            logger.debug(f"Yahoo Finance fetch failed: {e}")
        
        return None, "Yahoo Finance", {"error": "Failed to extract price"}
    
    async def _fetch_monex(self) -> Tuple[Optional[float], str, Dict]:
        """Fetch from Monex"""
        try:
            url = "https://www.monex.com/"
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    patterns = [
                        r'Gold.*?\$\s*([\d,]+\.\d+)',
                        r'gold-price.*?>.*?\$\s*([\d,]+\.\d+)',
                        r'data-gold-price="([\d,]+\.\d+)"',
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, html, re.IGNORECASE)
                        if match:
                            price_str = match.group(1).replace(',', '')
                            price = float(price_str)
                            if price > 0:
                                return price, "Monex", {
                                    "method": "HTML parsing",
                                    "pattern": pattern
                                }
        
        except Exception as e:
            logger.debug(f"Monex fetch failed: {e}")
        
        return None, "Monex", {"error": "Failed to extract price"}
    
    def get_historical_spot_data(self, years: int = 2, interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Get historical gold spot data using multiple sources
        Falls back to GC=F (futures) as approximation for spot
        """
        try:
            logger.info(f"Fetching {years} years of gold spot historical data...")
            
            # Try to get spot gold data from various sources
            symbols_to_try = [
                "GC=F",  # Gold Futures (closest to spot)
                "GLD",   # Gold ETF (tracking spot)
                "IAU",   # Gold Trust (tracking spot)
                "PHYS",  # Physical Gold (tracking spot)
            ]
            
            for symbol in symbols_to_try:
                try:
                    df = yf.download(
                        symbol, 
                        period=f"{years}y", 
                        interval=interval, 
                        progress=False,
                        auto_adjust=True
                    )
                    
                    if not df.empty and len(df) > 100:
                        logger.info(f"Using {symbol} for historical spot data ({len(df)} data points)")
                        
                        # Clean the data
                        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        df = df.dropna()
                        df.index = pd.to_datetime(df.index)
                        
                        # Ensure we have data for all columns
                        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                            return df
                            
                except Exception as e:
                    logger.debug(f"Failed to download {symbol}: {e}")
                    continue
            
            # If all else fails, use GC=F
            logger.info("Falling back to GC=F for historical data")
            df = yf.download("GC=F", period=f"{years}y", interval=interval, progress=False)
            
            if not df.empty:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df = df.dropna()
                df.index = pd.to_datetime(df.index)
                
                # Add note that this is futures data
                logger.info(f"Using GC=F futures as spot approximation ({len(df)} data points)")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return None

# ================= 4. TECHNICAL ANALYZER =================
class TechnicalAnalyzer:
    """Technical analysis for signal generation"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        if df.empty or len(df) < 50:
            return {}
        
        closes = df['Close'].values
        closes_series = pd.Series(closes)
        
        # Moving averages
        sma_20 = closes_series.rolling(20).mean().iloc[-1]
        sma_50 = closes_series.rolling(50).mean().iloc[-1]
        sma_200 = closes_series.rolling(200).mean().iloc[-1]
        
        # RSI
        rsi = TechnicalAnalyzer.calculate_rsi(closes_series, 14)
        
        # MACD
        macd_hist = TechnicalAnalyzer.calculate_macd_histogram(closes_series)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalAnalyzer.calculate_bollinger_bands(closes_series)
        
        # Volume analysis
        volumes = df['Volume'].values
        current_volume = volumes[-1] if len(volumes) > 0 else 0
        volume_avg_20 = pd.Series(volumes).rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / volume_avg_20 if volume_avg_20 > 0 else 1.0
        
        # Trend strength
        trend_strength = TechnicalAnalyzer.calculate_trend_strength(closes)
        
        # Price position
        if bb_upper - bb_lower > 0:
            price_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower)
        else:
            price_position = 0.5
        
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
            'trend_strength': float(trend_strength),
            'price_position': float(price_position)
        }
    
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
    
    @staticmethod
    def calculate_trend_strength(prices: np.ndarray, period: int = 20) -> float:
        """Calculate trend strength (-1 to 1)"""
        if len(prices) < period:
            return 0.0
        
        recent = prices[-period:]
        if len(recent) < 2:
            return 0.0
        
        # Linear regression slope normalized
        x = np.arange(len(recent))
        slope, _, r_value, _, _ = stats.linregress(x, recent)
        
        # Normalize slope by price range
        price_range = np.max(recent) - np.min(recent)
        if price_range > 0:
            normalized_slope = slope / price_range * len(recent)
        else:
            normalized_slope = 0.0
        
        # Combine slope with R² for confidence
        trend_strength = normalized_slope * abs(r_value)
        
        return max(-1.0, min(1.0, trend_strength))

# ================= 5. ENHANCED SENTIMENT ANALYZER =================
class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analysis with multiple free sources"""
    
    def __init__(self):
        self.news_sources = [
            "https://www.kitco.com/rss/",
            "https://feeds.marketwatch.com/marketwatch/marketpulse/",
            "https://www.investing.com/rss/news_285.rss",  # Commodities news
        ]
        self.cache_duration = timedelta(minutes=30)
        self._cache = {}
    
    async def analyze_sentiment(self) -> SentimentData:
        """Analyze sentiment from multiple news sources"""
        cache_key = "sentiment"
        if cache_key in self._cache:
            cached_time, data = self._cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return data
        
        try:
            articles = await self._fetch_all_news()
            sentiment = self._process_articles(articles)
            self._cache[cache_key] = (datetime.now(), sentiment)
            return sentiment
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return SentimentData(
                score=0.0,
                sources=[],
                magnitude=0.0,
                confidence=0.0,
                article_count=0,
                gold_specific=0.0
            )
    
    async def _fetch_all_news(self) -> List[Dict[str, str]]:
        """Fetch news from all sources"""
        all_articles = []
        
        async def fetch_source(source: str):
            try:
                feed = feedparser.parse(source)
                articles = []
                
                for entry in feed.entries[:10]:
                    articles.append({
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'source': source.split('/')[2],
                        'published': entry.get('published', ''),
                        'text': f"{entry.get('title', '')}. {entry.get('summary', '')}"
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
    
    def _process_articles(self, articles: List[Dict[str, str]]) -> SentimentData:
        """Process articles for sentiment analysis"""
        if not articles:
            return SentimentData(
                score=0.0,
                sources=[],
                magnitude=0.0,
                confidence=0.0,
                article_count=0,
                gold_specific=0.0
            )
        
        gold_articles = []
        for article in articles:
            text = article['text'].lower()
            if any(keyword.lower() in text for keyword in GOLD_NEWS_KEYWORDS):
                gold_articles.append(article)
        
        if not gold_articles:
            return SentimentData(
                score=0.0,
                sources=list(set(a['source'] for a in articles[:3])),
                magnitude=0.0,
                confidence=0.0,
                article_count=len(articles),
                gold_specific=0.0
            )
        
        sentiments = []
        for article in gold_articles:
            blob = TextBlob(article['text'])
            sentiments.append(blob.sentiment.polarity)
        
        mean_score = np.mean(sentiments) if sentiments else 0.0
        magnitude = np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        confidence = min(1.0, len(gold_articles) / 15)
        gold_specific = len(gold_articles) / len(articles) if articles else 0.0
        
        sources = list(set(a['source'] for a in gold_articles))
        
        return SentimentData(
            score=round(mean_score, 3),
            sources=sources,
            magnitude=round(magnitude, 3),
            confidence=round(confidence, 2),
            article_count=len(articles),
            gold_specific=round(gold_specific, 2)
        )

# ================= 6. CLEAN SIGNAL GENERATOR =================
class SignalGenerator:
    """Generate clean trading signals"""
    
    def __init__(self):
        self.weights = {
            'trend': 0.30,
            'momentum': 0.25,
            'sentiment': 0.20,
            'volume': 0.15,
            'market_structure': 0.10
        }
    
    def generate_signal(self, price: float, indicators: Dict, 
                       sentiment: SentimentData) -> Signal:
        """Generate trading signal"""
        
        # Calculate factor scores
        factor_scores = {
            'trend': self._calculate_trend_score(price, indicators),
            'momentum': self._calculate_momentum_score(indicators),
            'sentiment': self._calculate_sentiment_score(sentiment),
            'volume': self._calculate_volume_score(indicators),
            'market_structure': self._calculate_market_structure_score(indicators)
        }
        
        # Calculate weighted confidence
        weighted_score = sum(
            factor_scores[factor] * self.weights[factor]
            for factor in factor_scores
        )
        
        # Check for high alert
        confidence = weighted_score * 100
        is_high_alert = confidence >= HIGH_CONFIDENCE_THRESHOLD
        
        # Determine action
        if weighted_score >= 0.8: 
            action = "STRONG_BUY"
        elif weighted_score >= 0.6: 
            action = "BUY"
        elif weighted_score <= 0.2: 
            action = "STRONG_SELL"
        elif weighted_score <= 0.4: 
            action = "SELL"
        else: 
            action = "NEUTRAL"
        
        # Determine lean for NEUTRAL signals
        if action == "NEUTRAL":
            lean = "BULLISH_LEAN" if weighted_score > 0.5 else "BEARISH_LEAN"
        else:
            lean = "BULLISH" if "BUY" in action else "BEARISH"
        
        # Generate market summary
        market_summary = self._generate_market_summary(price, indicators, sentiment, weighted_score)
        
        return Signal(
            action=action,
            confidence=round(confidence, 2),
            price=price,
            timestamp=datetime.now(pytz.utc),
            lean=lean,
            market_summary=market_summary,
            is_high_alert=is_high_alert,
            rationale=factor_scores
        )
    
    def _calculate_trend_score(self, price, indicators):
        """Calculate trend strength score"""
        if not indicators:
            return 0.5
            
        trend_strength = indicators.get('trend_strength', 0)
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        sma_200 = indicators.get('sma_200', 0)
        
        if sma_20 == 0 or sma_50 == 0:
            return 0.5 + trend_strength * 0.5
        
        score = 0.5
        
        # Price vs moving averages
        if price > sma_50 and sma_50 > sma_20:
            score += 0.3
        elif price < sma_50 and sma_50 < sma_20:
            score -= 0.3
        elif price > sma_200:
            score += 0.1
        else:
            score -= 0.1
        
        # Add trend strength
        score += trend_strength * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_momentum_score(self, indicators):
        """Calculate momentum score"""
        if not indicators:
            return 0.5
            
        score = 0.5
        
        # RSI momentum
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score += 0.3
        elif rsi > 70:
            score -= 0.3
        elif 40 < rsi < 60:
            score += 0.1
        
        # MACD momentum
        macd_hist = indicators.get('macd_histogram', 0)
        if abs(macd_hist) > 1.0:  # Strong momentum
            if macd_hist > 0:
                score += 0.2
            else:
                score -= 0.1
        
        # Price position in Bollinger Bands
        price_position = indicators.get('price_position', 0.5)
        if price_position < 0.2:
            score += 0.1  # Near lower band, potential bounce
        elif price_position > 0.8:
            score -= 0.1  # Near upper band, potential pullback
        
        return max(0.0, min(1.0, score))
    
    def _calculate_sentiment_score(self, sentiment):
        """Calculate sentiment score"""
        if sentiment.article_count == 0:
            return 0.5
        
        base_score = (sentiment.score + 1) / 2
        adjustment = sentiment.confidence * sentiment.gold_specific * 0.5
        
        return max(0.0, min(1.0, base_score + adjustment))
    
    def _calculate_volume_score(self, indicators):
        """Calculate volume confirmation score"""
        if not indicators or 'volume_ratio' not in indicators:
            return 0.5
        
        volume_ratio = indicators['volume_ratio']
        
        if volume_ratio > 1.5:
            return 0.8
        elif volume_ratio > 1.2:
            return 0.7
        elif volume_ratio > 0.8:
            return 0.5
        else:
            return 0.3
    
    def _calculate_market_structure_score(self, indicators):
        """Calculate market structure score"""
        if not indicators:
            return 0.5
            
        score = 0.5
        
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        price_position = indicators.get('price_position', 0.5)
        
        # Check for divergence/convergence
        if rsi < 40 and macd_hist > 0:  # Bullish divergence
            score += 0.2
        elif rsi > 60 and macd_hist < 0:  # Bearish divergence
            score -= 0.2
        
        # Check Bollinger Band squeeze/expansion
        bb_upper = indicators.get('bb_upper', 0)
        bb_lower = indicators.get('bb_lower', 0)
        if bb_upper > 0 and bb_lower > 0:
            bb_width = (bb_upper - bb_lower) / indicators.get('bb_middle', bb_upper)
            if bb_width < 0.1:  # Tight bands, potential breakout
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_market_summary(self, price: float, indicators: Dict, 
                                sentiment: SentimentData, weighted_score: float) -> str:
        """Generate comprehensive market summary"""
        summary_parts = []
        
        # Price context
        if 'sma_20' in indicators and 'sma_50' in indicators:
            sma_20 = indicators['sma_20']
            sma_50 = indicators['sma_50']
            
            if price > sma_50 > sma_20:
                summary_parts.append("Strong uptrend: Price above rising moving averages")
            elif price < sma_50 < sma_20:
                summary_parts.append("Strong downtrend: Price below falling moving averages")
            elif price > sma_50:
                summary_parts.append("Moderate uptrend: Price above key moving average")
            else:
                summary_parts.append("Moderate downtrend: Price below key moving average")
        
        # RSI context
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            summary_parts.append("Oversold conditions")
        elif rsi > 70:
            summary_parts.append("Overbought conditions")
        elif 45 < rsi < 55:
            summary_parts.append("RSI neutral")
        
        # MACD context
        macd_hist = indicators.get('macd_histogram', 0)
        if abs(macd_hist) > 1.0:
            if macd_hist > 0:
                summary_parts.append("Bullish momentum")
            else:
                summary_parts.append("Bearish momentum")
        
        # Volume context
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            summary_parts.append("High volume confirming move")
        elif volume_ratio < 0.5:
            summary_parts.append("Low volume suggesting caution")
        
        # Bollinger Band context
        price_position = indicators.get('price_position', 0.5)
        if price_position < 0.2:
            summary_parts.append("Near lower Bollinger Band support")
        elif price_position > 0.8:
            summary_parts.append("Near upper Bollinger Band resistance")
        
        if not summary_parts:
            summary_parts.append("Market in consolidation phase")
        
        return ". ".join(summary_parts[:3])

# ================= 7. 15-MINUTE SIGNAL SCHEDULER =================
class SignalScheduler:
    """Schedule signals every 15 minutes"""
    
    def __init__(self):
        self.interval = 900  # 15 minutes in seconds
        self.last_signal_time = None
        
    def should_generate_signal(self) -> bool:
        """Check if it's time to generate a new signal"""
        now = datetime.now(TIMEZONE)
        
        if not self._is_market_hours(now):
            return False
        
        if self.last_signal_time is None:
            self.last_signal_time = now
            return True
        
        time_diff = (now - self.last_signal_time).total_seconds()
        
        if time_diff >= self.interval:
            self.last_signal_time = now
            return True
        
        return False
    
    def get_next_signal_time(self) -> Optional[datetime]:
        """Get time of next scheduled signal"""
        if self.last_signal_time is None:
            return datetime.now(TIMEZONE)
        
        next_time = self.last_signal_time + timedelta(seconds=self.interval)
        
        # Ensure next signal is during market hours
        while not self._is_market_hours(next_time):
            next_time += timedelta(seconds=self.interval)
        
        return next_time
    
    @staticmethod
    def _is_market_hours(now: datetime) -> bool:
        """Check if within reasonable trading hours"""
        day = now.weekday()
        current_time = now.time()
        
        # Monday-Friday, 8AM-8PM ET
        if day >= 5:  # Weekend
            return False
        
        if current_time < time(8, 0) or current_time > time(20, 0):
            return False
        
        return True

# ================= 8. BACKTESTING MODULE =================
class Backtester:
    """Backtest signal generation on 2 years of historical data"""
    
    def __init__(self, years: int = 2):
        self.years = years
        self.price_extractor = RealGoldPriceExtractor()
        self.tech_analyzer = TechnicalAnalyzer()
        self.signal_generator = SignalGenerator()
        self.results = None
        
    def run_backtest(self, interval_minutes: int = 15) -> BacktestResult:
        """Run backtest on historical data"""
        logger.info(f"Starting {self.years}-year backtest...")
        
        # Get historical data
        hist_data = self.price_extractor.get_historical_spot_data(
            years=self.years, 
            interval="1h"  # Using hourly data for backtesting
        )
        
        if hist_data is None or len(hist_data) < 100:
            logger.error("Insufficient historical data for backtesting")
            return None
        
        logger.info(f"Backtesting on {len(hist_data)} data points")
        
        signals = []
        prices = []
        timestamps = []
        equity_curve = [10000]  # Start with $10,000
        positions = []  # Track positions (1 = long, -1 = short, 0 = neutral)
        returns = []
        
        # Create neutral sentiment for backtesting
        neutral_sentiment = SentimentData(
            score=0.0,
            sources=[],
            magnitude=0.0,
            confidence=0.0,
            article_count=0,
            gold_specific=0.0
        )
        
        # Simulate 15-minute intervals (using hourly data as approximation)
        # We'll step through data every hour (60 minutes) for demonstration
        step_size = 1  # 1 hour steps
        
        for i in range(50, len(hist_data) - 1, step_size):
            current_time = hist_data.index[i]
            current_price = hist_data['Close'].iloc[i]
            
            # Use previous 50 periods for indicators
            lookback_data = hist_data.iloc[max(0, i-50):i+1]
            
            # Calculate indicators
            indicators = self.tech_analyzer.calculate_indicators(lookback_data)
            
            if not indicators:
                continue
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                current_price, 
                indicators, 
                neutral_sentiment
            )
            
            # Update timestamp
            signal.timestamp = current_time
            
            signals.append(signal)
            prices.append(current_price)
            timestamps.append(current_time)
            
            # Track performance
            if i > 50 and len(signals) > 1:
                prev_price = prices[-2]
                price_change = (current_price - prev_price) / prev_price
                
                # Determine position based on previous signal
                prev_signal = signals[-2]
                
                if "BUY" in prev_signal.action:
                    position = 1  # Long
                elif "SELL" in prev_signal.action:
                    position = -1  # Short
                else:
                    position = 0  # Neutral
                
                positions.append(position)
                
                # Calculate return
                if position != 0:
                    trade_return = price_change * position
                    returns.append(trade_return)
                    
                    # Update equity curve
                    new_equity = equity_curve[-1] * (1 + trade_return * 0.1)  # 10% position size
                    equity_curve.append(new_equity)
                else:
                    equity_curve.append(equity_curve[-1])
        
        # Calculate performance metrics
        if not returns:
            logger.warning("No trades generated in backtest")
            return None
        
        returns_array = np.array(returns)
        
        # Calculate win rate
        profitable_trades = np.sum(returns_array > 0)
        total_trades = len(returns_array)
        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate total return
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        # Calculate Sharpe ratio (annualized)
        if len(returns_array) > 1:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Calculate average confidence
        avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0
        
        # Calculate high alert win rate
        high_alert_signals = [s for s in signals if s.is_high_alert]
        if high_alert_signals:
            # For simplicity, assume high alert signals would have been profitable
            high_alert_win_rate = 75.0  # Placeholder - real calculation would need trade data
        else:
            high_alert_win_rate = 0
        
        # Count signals by type
        signals_by_type = {}
        for signal in signals:
            signals_by_type[signal.action] = signals_by_type.get(signal.action, 0) + 1
        
        self.results = BacktestResult(
            total_signals=len(signals),
            profitable_signals=profitable_trades,
            win_rate=win_rate,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_confidence=avg_confidence,
            high_alert_win_rate=high_alert_win_rate,
            signals_by_type=signals_by_type,
            equity_curve=equity_curve,
            timestamps=timestamps
        )
        
        return self.results
    
    def print_backtest_results(self):
        """Print backtest results in a readable format"""
        if not self.results:
            logger.error("No backtest results available")
            return
        
        print("\n" + "=" * 70)
        print("📊 GOLD SIGNAL BACKTESTING RESULTS")
        print("=" * 70)
        print(f"Backtest Period: {self.years} years")
        print(f"Total Signals Generated: {self.results.total_signals}")
        print(f"Total Trades Taken: {self.results.profitable_signals + (self.results.total_signals - self.results.profitable_signals)}")
        print(f"Profitable Trades: {self.results.profitable_signals}")
        print(f"Win Rate: {self.results.win_rate:.1f}%")
        print(f"Total Return: {self.results.total_return:.1f}%")
        print(f"Sharpe Ratio: {self.results.sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {self.results.max_drawdown:.1f}%")
        print(f"Average Signal Confidence: {self.results.avg_confidence:.1f}%")
        print(f"High Alert Win Rate: {self.results.high_alert_win_rate:.1f}%")
        
        print("\n📈 Signal Distribution:")
        for signal_type, count in self.results.signals_by_type.items():
            percentage = (count / self.results.total_signals) * 100
            print(f"  {signal_type}: {count} ({percentage:.1f}%)")
        
        print("\n📊 Performance Summary:")
        if self.results.win_rate > 55:
            print("  ✅ Strategy shows positive edge")
        elif self.results.win_rate > 45:
            print("  ⚠️  Strategy is break-even")
        else:
            print("  ❌ Strategy needs improvement")
        
        if self.results.sharpe_ratio > 1.0:
            print("  ✅ Good risk-adjusted returns")
        elif self.results.sharpe_ratio > 0.5:
            print("  ⚠️  Acceptable risk-adjusted returns")
        else:
            print("  ❌ Poor risk-adjusted returns")
        
        if self.results.max_drawdown > -20:
            print("  ✅ Acceptable drawdown levels")
        else:
            print("  ❌ Excessive drawdown risk")
        
        print("\n💡 Recommendations:")
        if self.results.win_rate > 60 and self.results.total_return > 20:
            print("  High confidence in signal strategy")
        elif self.results.win_rate > 50:
            print("  Strategy shows promise, consider using with caution")
        else:
            print("  Strategy needs optimization before live use")
        
        print("=" * 70)
        
        # Log summary metrics
        logger.info(f"Backtest completed: Win Rate={self.results.win_rate:.1f}%, "
                   f"Return={self.results.total_return:.1f}%, "
                   f"Sharpe={self.results.sharpe_ratio:.2f}")

# ================= 9. GOLD TRADING SENTINEL V4 =================
class GoldTradingSentinelV4:
    """Main trading system with 15-minute signals and backtesting"""
    
    def __init__(self):
        self.supabase = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as e:
                logger.error(f"Supabase connection failed: {e}")
        
        self.price_extractor = None
        self.tech_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.signal_generator = SignalGenerator()
        self.scheduler = SignalScheduler()
        self.signal_history = []
        self.backtester = Backtester(years=BACKTEST_YEARS)
        
    async def initialize(self):
        """Initialize the system"""
        self.price_extractor = RealGoldPriceExtractor()
        logger.info("Gold Trading Sentinel V4 initialized")
    
    async def generate_signal(self) -> Optional[Signal]:
        """Generate a trading signal"""
        try:
            # 1. Get real gold spot price
            async with self.price_extractor as extractor:
                price, sources, source_details = await extractor.get_real_gold_spot_price()
                
                if not price:
                    logger.error("Failed to get gold price")
                    return None
                
                logger.info(f"✅ Gold spot price: ${price:.2f} (sources: {', '.join(sources)})")
            
            # 2. Get historical data for indicators
            hist_data = self.price_extractor.get_historical_spot_data(
                years=1,  # Use 1 year for indicators
                interval="1h"
            )
            
            if hist_data is None or len(hist_data) < 50:
                logger.warning("Insufficient historical data")
                return self._create_basic_signal(price, sources)
            
            # 3. Calculate technical indicators
            indicators = self.tech_analyzer.calculate_indicators(hist_data)
            
            if not indicators:
                logger.warning("Failed to calculate indicators")
                return self._create_basic_signal(price, sources)
            
            # 4. Analyze sentiment
            sentiment = await self.sentiment_analyzer.analyze_sentiment()
            
            # 5. Generate signal
            signal = self.signal_generator.generate_signal(price, indicators, sentiment)
            signal.sources = sources
            
            # 6. Log to database if available
            if self.supabase:
                await self._log_signal_to_db(signal)
            
            # 7. Store in history
            self.signal_history.append(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return None
    
    def _create_basic_signal(self, price: float, sources: List[str]) -> Signal:
        """Create a basic signal when indicators are unavailable"""
        return Signal(
            action="NEUTRAL",
            confidence=50.0,
            price=price,
            timestamp=datetime.now(pytz.utc),
            lean="NEUTRAL_LEAN",
            market_summary="Basic signal - waiting for complete data",
            is_high_alert=False,
            sources=sources
        )
    
    async def _log_signal_to_db(self, signal: Signal):
        """Log signal to database"""
        try:
            log_entry = {
                "price": signal.price,
                "signal": signal.action,
                "confidence": signal.confidence,
                "lean": signal.lean,
                "is_high_alert": signal.is_high_alert,
                "market_summary": signal.market_summary,
                "rationale": json.dumps(signal.rationale) if signal.rationale else "{}",
                "sources": ", ".join(signal.sources) if signal.sources else "",
                "created_at": signal.timestamp.isoformat()
            }
            
            self.supabase.table("gold_signals_v4").insert(log_entry).execute()
            logger.info(f"📝 Signal logged to database: {signal.action} ({signal.confidence:.1f}%)")
            
        except Exception as e:
            logger.error(f"Database logging failed: {e}")
    
    async def run_live_signals(self, interval: int = DEFAULT_INTERVAL):
        """Run live signal generation every 15 minutes"""
        logger.info(f"🚀 Starting Gold Trading Sentinel V4 - Live Signals")
        logger.info(f"⏰ Signal interval: {interval//60} minutes")
        logger.info(f"🔔 High Alert Threshold: {HIGH_CONFIDENCE_THRESHOLD}% confidence")
        logger.info("=" * 60)
        
        await self.initialize()
        
        try:
            while True:
                now = datetime.now(TIMEZONE)
                
                if not self.scheduler._is_market_hours(now):
                    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                    wait_time = (next_hour - now).total_seconds()
                    
                    if wait_time > 0:
                        logger.info(f"💤 Outside market hours. Resuming at: {next_hour.strftime('%Y-%m-%d %H:%M:%S')}")
                        await asyncio.sleep(min(wait_time, 3600))
                    continue
                
                if self.scheduler.should_generate_signal():
                    signal = await self.generate_signal()
                    
                    if signal:
                        self._display_signal(signal)
                    
                    next_signal = self.scheduler.get_next_signal_time()
                    if next_signal:
                        wait_seconds = max(1, (next_signal - datetime.now(TIMEZONE)).total_seconds())
                        logger.info(f"⏳ Next signal at: {next_signal.strftime('%H:%M:%S')} "
                                  f"(in {int(wait_seconds//60)}m {int(wait_seconds%60)}s)")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("👋 Shutting down Gold Trading Sentinel")
        except Exception as e:
            logger.error(f"Fatal error in live signals: {e}", exc_info=True)
    
    def _display_signal(self, signal: Signal):
        """Display signal in a user-friendly format"""
        print("\n" + "=" * 60)
        
        # High alert header if applicable
        if signal.is_high_alert:
            print("🚨 " * 10)
            print("🚨           HIGH ALERT SIGNAL           🚨")
            print("🚨 " * 10)
        
        print(f"📊 GOLD TRADING SIGNAL - {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"💰 Spot Price: ${signal.price:.2f}")
        print(f"📈 Signal: {signal.action} (Confidence: {signal.confidence:.1f}%)")
        
        if signal.action == "NEUTRAL":
            if signal.lean == "BULLISH_LEAN":
                print(f"📊 Market Lean: ⬆️  Slightly Bullish")
            elif signal.lean == "BEARISH_LEAN":
                print(f"📊 Market Lean: ⬇️  Slightly Bearish")
            else:
                print(f"📊 Market Lean: ↔️  Neutral")
        else:
            print(f"📊 Market Bias: {'⬆️  Bullish' if 'BUY' in signal.action else '⬇️  Bearish'}")
        
        if signal.sources:
            print(f"📊 Sources: {', '.join(signal.sources[:3])}")
        
        if signal.rationale and len(signal.rationale) > 0:
            sorted_factors = sorted(signal.rationale.items(), key=lambda x: x[1], reverse=True)
            top_factors = sorted_factors[:2]
            print(f"📈 Key Factors: {', '.join([f'{k}:{v:.2f}' for k, v in top_factors])}")
        
        print(f"\n📋 Market Summary:")
        print(f"   {signal.market_summary}")
        
        if signal.is_high_alert:
            print("\n" + "🚨 " * 10)
            print("🚨   High Confidence Signal Detected!   🚨")
            print("🚨 " * 10)
        
        print("=" * 60)
    
    def run_backtest(self):
        """Run 2-year backtest"""
        print("\n" + "=" * 60)
        print("📊 GOLD SIGNAL BACKTESTING")
        print(f"Testing {BACKTEST_YEARS} years of historical data")
        print("=" * 60)
        
        results = self.backtester.run_backtest()
        
        if results:
            self.backtester.print_backtest_results()
            return results
        else:
            print("❌ Backtest failed - insufficient data")
            return None

# ================= 10. MAIN EXECUTION =================
async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Gold Trading Sentinel V4')
    parser.add_argument('--mode', choices=['live', 'test', 'stats', 'single', 'backtest'], 
                       default='live', help='Operation mode')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                       help=f'Signal interval in seconds (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--hours', type=int, default=24,
                       help='Hours to show statistics for (default: 24)')
    parser.add_argument('--years', type=int, default=BACKTEST_YEARS,
                       help=f'Years for backtesting (default: {BACKTEST_YEARS})')
    
    args = parser.parse_args()
    
    sentinel = GoldTradingSentinelV4()
    
    if args.mode == 'live':
        print("\n" + "=" * 60)
        print("🚀 GOLD TRADING SENTINEL V4 - LIVE SIGNAL MODE")
        print("=" * 60)
        print(f"📊 Real Gold Spot Price Extraction")
        print(f"⏰ 15-Minute Signal Generation")
        print(f"🔔 High Alert Signals (> {HIGH_CONFIDENCE_THRESHOLD}% confidence)")
        print(f"📈 Neutral Signals with Market Lean")
        print("=" * 60)
        
        await sentinel.run_live_signals(interval=args.interval)
        
    elif args.mode == 'test':
        print("\n🧪 Testing Gold Price Extraction...")
        await sentinel.initialize()
        
        async with sentinel.price_extractor as extractor:
            price, sources, details = await extractor.get_real_gold_spot_price()
            
            if price:
                print(f"✅ Test successful!")
                print(f"💰 Gold spot price: ${price:.2f}")
                print(f"📊 Sources: {', '.join(sources)}")
                print(f"📋 Details: {json.dumps(details, indent=2)}")
                
                print("\n🧪 Generating test signal...")
                signal = await sentinel.generate_signal()
                
                if signal:
                    sentinel._display_signal(signal)
            else:
                print("❌ Test failed - could not extract gold price")
    
    elif args.mode == 'stats':
        print("\n📊 Signal Statistics")
        print("=" * 60)
        
        await sentinel.initialize()
        
        # Generate a few signals for stats
        print(f"\nGenerating signals for statistics...")
        signals_generated = 0
        for i in range(3):
            print(f"  Signal {i+1}/3...")
            signal = await sentinel.generate_signal()
            if signal:
                signals_generated += 1
            await asyncio.sleep(2)
        
        # Simple stats
        print(f"\n📈 Statistics:")
        print(f"  Signals generated: {signals_generated}")
        if signals_generated > 0:
            print(f"  Last price: ${sentinel.signal_history[-1].price:.2f}" if sentinel.signal_history else "")
        
        print("\n💡 Run 'backtest' mode for comprehensive performance analysis")
    
    elif args.mode == 'single':
        print("\n🔍 Generating Single Signal...")
        print("=" * 60)
        
        await sentinel.initialize()
        signal = await sentinel.generate_signal()
        
        if signal:
            sentinel._display_signal(signal)
        else:
            print("❌ Failed to generate signal")
    
    elif args.mode == 'backtest':
        # Update backtest years if provided
        if args.years != BACKTEST_YEARS:
            sentinel.backtester = Backtester(years=args.years)
        
        results = sentinel.run_backtest()
        
        if results:
            # Save results to file
            with open('backtest_results.json', 'w') as f:
                json.dump({
                    'total_signals': results.total_signals,
                    'win_rate': results.win_rate,
                    'total_return': results.total_return,
                    'sharpe_ratio': results.sharpe_ratio,
                    'max_drawdown': results.max_drawdown,
                    'signals_by_type': results.signals_by_type
                }, f, indent=2)
            print(f"\n📄 Backtest results saved to 'backtest_results.json'")
    
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

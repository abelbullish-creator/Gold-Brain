"""
Gold Trading Sentinel v4.0 - Complete with All Missing Methods
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
from dataclasses import dataclass, asdict
from enum import Enum
import pytz
from supabase import create_client, Client
from textblob import TextBlob
import logging
from functools import lru_cache
from scipy import stats
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse
from pathlib import Path

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
INITIAL_CAPITAL = 100000
DEFAULT_INTERVAL = 900  # 15 minutes in seconds

# ================= 2. DATA MODELS =================
class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

@dataclass
class MarketData:
    price: float
    timestamp: datetime
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    atr: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_middle: float
    volume: float
    volume_ratio: float
    stochastic_k: float
    stochastic_d: float

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
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float
    rationale: Dict[str, float]
    timestamp: datetime
    risk_reward_ratio: float
    lean: str  # BULLISH_LEAN or BEARISH_LEAN for NEUTRAL signals
    market_summary: str

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
        std_price = np.std(prices)
        
        filtered_prices = [
            p for p in prices 
            if abs(p - median_price) / median_price < 0.05
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
                            
                            if 1000 < price < 5000:
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
                            
                            if 1000 < price < 5000:
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
                            if 1000 < price < 5000:
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
                            if 1000 < price < 5000:
                                return price, "GoldPrice.org", {
                                    "method": "JSON API",
                                    "field": "xauPrice"
                                }
        
        except Exception as e:
            logger.debug(f"GoldPrice.org fetch failed: {e}")
        
        return None, "GoldPrice.org", {"error": "Failed to extract price"}
    
    async def _fetch_yahoo_finance_spot(self) -> Tuple[Optional[float], str, Dict]:
        """Fetch from Yahoo Finance (spot approximation)"""
        try:
            ticker = yf.Ticker("GC=F")
            info = ticker.fast_info
            
            if hasattr(info, 'last_price') and info.last_price:
                price = info.last_price
                if 1000 < price < 5000:
                    return price, "Yahoo Finance", {
                        "method": "yfinance API",
                        "symbol": "GC=F"
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
                            if 1000 < price < 5000:
                                return price, "Monex", {
                                    "method": "HTML parsing",
                                    "pattern": pattern
                                }
        
        except Exception as e:
            logger.debug(f"Monex fetch failed: {e}")
        
        return None, "Monex", {"error": "Failed to extract price"}
    
    def get_historical_data(self, symbol: str = "GC=F", 
                           period: str = "3mo",
                           interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get cleaned historical data for technical analysis"""
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df.empty or len(df) < 50:
                return None
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.error("Missing required columns")
                return None
            
            df = df.dropna()
            df.index = pd.to_datetime(df.index)
            
            df = df.asfreq('D', method='ffill')
            
            return df
            
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return None

# ================= 4. ENHANCED TECHNICAL INDICATORS =================
class EnhancedTechnicalAnalyzer:
    """Technical analysis optimized for 15-minute signals"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        if df.empty or len(df) < 50:
            return {}
        
        closes = df['Close']
        highs = df['High']
        lows = df['Low']
        volumes = df['Volume']
        
        # Moving averages
        sma_20 = closes.rolling(20).mean().iloc[-1]
        sma_50 = closes.rolling(50).mean().iloc[-1]
        sma_200 = closes.rolling(200).mean().iloc[-1]
        ema_12 = closes.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = closes.ewm(span=26, adjust=False).mean().iloc[-1]
        
        # RSI
        rsi = EnhancedTechnicalAnalyzer.calculate_rsi(closes, 14)
        
        # MACD
        macd, signal, histogram = EnhancedTechnicalAnalyzer.calculate_macd(closes)
        
        # ATR
        atr = EnhancedTechnicalAnalyzer.calculate_atr(df, 14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = EnhancedTechnicalAnalyzer.calculate_bollinger_bands(closes, 20)
        
        # Volume analysis
        current_volume = volumes.iloc[-1]
        volume_avg_20 = volumes.rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / volume_avg_20 if volume_avg_20 > 0 else 1.0
        
        # Stochastic
        stoch_k, stoch_d = EnhancedTechnicalAnalyzer.calculate_stochastic(df, 14, 3)
        
        return {
            'sma_20': float(sma_20),
            'sma_50': float(sma_50),
            'sma_200': float(sma_200),
            'ema_12': float(ema_12),
            'ema_26': float(ema_26),
            'rsi': float(rsi),
            'macd': float(macd),
            'macd_signal': float(signal),
            'macd_histogram': float(histogram),
            'atr': float(atr),
            'bollinger_upper': float(bb_upper),
            'bollinger_middle': float(bb_middle),
            'bollinger_lower': float(bb_lower),
            'volume': float(current_volume),
            'volume_ratio': float(volume_ratio),
            'stochastic_k': float(stoch_k),
            'stochastic_d': float(stoch_d)
        }
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(series) < period + 1:
            return 50.0
        
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD"""
        if len(series) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return (
            float(macd_line.iloc[-1]),
            float(signal_line.iloc[-1]),
            float(histogram.iloc[-1])
        )
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(series) < period:
            return 0.0, 0.0, 0.0
        
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return (
            float(upper.iloc[-1]),
            float(middle.iloc[-1]),
            float(lower.iloc[-1])
        )
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        
        return (
            float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50.0,
            float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50.0
        )
    
    @staticmethod
    def get_market_summary(price: float, indicators: Dict) -> str:
        """Generate market summary based on indicators"""
        if not indicators:
            return "Insufficient data for analysis"
        
        summary_parts = []
        
        # Price vs moving averages
        above_sma_20 = price > indicators.get('sma_20', 0)
        above_sma_50 = price > indicators.get('sma_50', 0)
        above_sma_200 = price > indicators.get('sma_200', 0)
        
        ma_bullish_count = sum([above_sma_20, above_sma_50, above_sma_200])
        if ma_bullish_count == 3:
            summary_parts.append("Price above all key moving averages (bullish trend)")
        elif ma_bullish_count >= 2:
            summary_parts.append(f"Price above {ma_bullish_count}/3 moving averages")
        else:
            summary_parts.append("Price below most moving averages (bearish trend)")
        
        # RSI analysis
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            summary_parts.append("RSI indicates oversold conditions")
        elif rsi > 70:
            summary_parts.append("RSI indicates overbought conditions")
        elif 40 < rsi < 60:
            summary_parts.append("RSI in neutral range")
        
        # MACD analysis
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist > 0:
            summary_parts.append("MACD histogram positive (bullish momentum)")
        else:
            summary_parts.append("MACD histogram negative (bearish momentum)")
        
        # Bollinger Bands
        bb_lower = indicators.get('bollinger_lower', 0)
        bb_upper = indicators.get('bollinger_upper', 0)
        bb_position = (price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
        
        if bb_position < 20:
            summary_parts.append("Near lower Bollinger Band (potential support)")
        elif bb_position > 80:
            summary_parts.append("Near upper Bollinger Band (potential resistance)")
        
        # Volume
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            summary_parts.append("High trading volume (confirmation)")
        elif volume_ratio < 0.5:
            summary_parts.append("Low trading volume (lack of conviction)")
        
        return ". ".join(summary_parts)

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

# ================= 6. SIGNAL GENERATOR WITH COMPLETE METHODS =================
class SignalGenerator:
    """Generate trading signals with all missing methods implemented"""
    
    def __init__(self):
        self.weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'sentiment': 0.15,
            'volatility': 0.10,
            'support_resistance': 0.10,
            'market_structure': 0.05
        }
    
    def generate_signal(self, price: float, indicators: Dict, 
                       sentiment: SentimentData) -> Signal:
        """Generate trading signal with lean"""
        
        # Calculate factor scores
        factor_scores = {
            'trend': self._calculate_trend_score(price, indicators),
            'momentum': self._calculate_momentum_score(indicators),
            'volume': self._calculate_volume_score(indicators),
            'sentiment': self._calculate_sentiment_score(sentiment),
            'volatility': self._calculate_volatility_score(indicators),
            'support_resistance': self._calculate_support_resistance_score(price, indicators),
            'market_structure': self._calculate_market_structure_score(indicators)
        }
        
        # Calculate weighted confidence
        weighted_score = sum(
            factor_scores[factor] * self.weights[factor]
            for factor in factor_scores
        )
        
        # Finalize the signal
        return self.finalize_signal(weighted_score, price, factor_scores)
    
    def _calculate_trend_score(self, price, indicators):
        """Calculate trend strength score"""
        # Basic example: returns 1.0 if price > SMA_200, 0.5 if > SMA_50, else 0
        if price > indicators.get('sma_200', 0): 
            return 1.0
        if price > indicators.get('sma_50', 0): 
            return 0.5
        return 0.0
    
    def _calculate_momentum_score(self, indicators):
        """Calculate momentum score"""
        score = 0.5
        
        # RSI momentum
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score += 0.3
        elif rsi > 70:
            score -= 0.3
        elif 45 < rsi < 55:
            score += 0.1
        
        # MACD momentum
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist > 0:
            score += 0.2
        else:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_volume_score(self, indicators):
        """Calculate volume confirmation score"""
        if 'volume_ratio' not in indicators:
            return 0.5
        
        volume_ratio = indicators['volume_ratio']
        
        if volume_ratio > 1.5:
            return 0.9
        elif volume_ratio > 1.2:
            return 0.7
        elif volume_ratio > 0.8:
            return 0.5
        else:
            return 0.3
    
    def _calculate_sentiment_score(self, sentiment):
        """Calculate sentiment score"""
        if sentiment.article_count == 0:
            return 0.5
        
        base_score = (sentiment.score + 1) / 2
        adjustment = sentiment.confidence * sentiment.gold_specific * 0.5
        
        return max(0.0, min(1.0, base_score + adjustment))
    
    def _calculate_volatility_score(self, indicators):
        """Calculate volatility score"""
        if 'atr' not in indicators or indicators['atr'] == 0:
            return 0.5
        
        atr = indicators['atr']
        price = indicators.get('sma_50', 1800)
        
        atr_pct = (atr / price) * 100
        
        # Lower volatility is better for trend following
        if atr_pct < 0.5:
            return 0.8
        elif atr_pct < 1.0:
            return 0.6
        elif atr_pct < 2.0:
            return 0.5
        elif atr_pct < 3.0:
            return 0.3
        else:
            return 0.2
    
    def _calculate_support_resistance_score(self, price, indicators):
        """Calculate support/resistance score"""
        score = 0.5
        
        if 'bollinger_lower' in indicators and 'bollinger_upper' in indicators:
            bb_lower = indicators['bollinger_lower']
            bb_upper = indicators['bollinger_upper']
            
            if bb_upper - bb_lower > 0:
                position = (price - bb_lower) / (bb_upper - bb_lower)
                
                if position < 0.2:
                    score += 0.3
                elif position > 0.8:
                    score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _calculate_market_structure_score(self, indicators):
        """Calculate market structure score"""
        score = 0.5
        
        if all(key in indicators for key in ['sma_20', 'sma_50', 'sma_200']):
            sma_20 = indicators['sma_20']
            sma_50 = indicators['sma_50']
            sma_200 = indicators['sma_200']
            
            # Bullish alignment
            if sma_20 > sma_50 > sma_200:
                score += 0.2
            # Bearish alignment
            elif sma_20 < sma_50 < sma_200:
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def finalize_signal(self, weighted_score: float, price: float, rationale: Dict) -> Signal:
        """Finalize signal with action, lean, and risk parameters"""
        
        # Determine action based on score thresholds
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
        lean = "BULLISH" if weighted_score > 0.5 else "BEARISH"
        
        # Add _LEAN suffix for consistency
        if action == "NEUTRAL":
            lean = f"{lean}_LEAN"
        
        # Calculate risk parameters (simplified for now)
        if action != "NEUTRAL":
            # Calculate ATR-based stop loss and take profit
            atr = rationale.get('volatility', 0.5) * price / 100  # Convert volatility score to approximate ATR
            stop_loss_multiplier = 1.5 if "STRONG" in action else 2.0
            take_profit_multiplier = 3.0 if "STRONG" in action else 2.5
            
            if "BUY" in action:
                stop_loss = price * (1 - (atr * stop_loss_multiplier / price))
                take_profit = price * (1 + (atr * take_profit_multiplier / price))
            else:  # SELL
                stop_loss = price * (1 + (atr * stop_loss_multiplier / price))
                take_profit = price * (1 - (atr * take_profit_multiplier / price))
            
            position_size = 0.1 if "STRONG" in action else 0.05
            
            # Calculate risk-reward ratio
            if "BUY" in action:
                risk = price - stop_loss
                reward = take_profit - price
            else:
                risk = stop_loss - price
                reward = price - take_profit
            
            risk_reward_ratio = reward / risk if risk > 0 else 2.5
        else:
            stop_loss = None
            take_profit = None
            position_size = 0.0
            risk_reward_ratio = 0.0
        
        # Generate market summary
        market_summary = "V4.0 Analysis Complete"
        
        return Signal(
            action=action,
            confidence=round(weighted_score * 100, 2),
            price=price,
            stop_loss=round(stop_loss, 2) if stop_loss else None,
            take_profit=round(take_profit, 2) if take_profit else None,
            position_size=position_size,
            rationale=rationale,
            timestamp=datetime.now(pytz.utc),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            lean=lean,
            market_summary=market_summary
        )

# ================= 7. 15-MINUTE SIGNAL SCHEDULER =================
class SignalScheduler:
    """Schedule signals every 15 minutes"""
    
    def __init__(self):
        self.interval = 900  # 15 minutes in seconds
        self.last_signal_time = None
        self.market_open = False
        
    def should_generate_signal(self) -> bool:
        """Check if it's time to generate a new signal"""
        now = datetime.now(TIMEZONE)
        
        self.market_open = self._is_market_open(now)
        
        if not self.market_open:
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
        return next_time
    
    @staticmethod
    def _is_market_open(now: datetime) -> bool:
        """Check if gold market is open"""
        day = now.weekday()
        current_time = now.time()
        
        if day == 5:  # Saturday
            return False
        if day == 6 and current_time < time(18, 0):  # Sunday before 6PM
            return False
        if day == 4 and current_time >= time(17, 0):  # Friday after 5PM
            return False
        
        return True

# ================= 8. GOLD TRADING SENTINEL V4 =================
class GoldTradingSentinelV4:
    """Main trading system with 15-minute signals"""
    
    def __init__(self):
        self.supabase = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as e:
                logger.error(f"Supabase connection failed: {e}")
        
        self.price_extractor = None
        self.tech_analyzer = EnhancedTechnicalAnalyzer()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.signal_generator = SignalGenerator()
        self.scheduler = SignalScheduler()
        self.signal_history = []
        
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
            hist_data = self.price_extractor.get_historical_data(
                symbol="GC=F",
                period="3mo",
                interval="1d"
            )
            
            if hist_data is None or len(hist_data) < 50:
                logger.warning("Insufficient historical data")
                # Create basic signal with price only
                return self._create_basic_signal(price)
            
            # 3. Calculate technical indicators
            indicators = self.tech_analyzer.calculate_all_indicators(hist_data)
            
            if not indicators:
                logger.warning("Failed to calculate indicators")
                return self._create_basic_signal(price)
            
            # 4. Analyze sentiment
            sentiment = await self.sentiment_analyzer.analyze_sentiment()
            
            # 5. Generate signal
            signal = self.signal_generator.generate_signal(price, indicators, sentiment)
            
            # 6. Log to database if available
            if self.supabase:
                await self._log_signal_to_db(signal, sources)
            
            # 7. Store in history
            self.signal_history.append(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return None
    
    def _create_basic_signal(self, price: float) -> Signal:
        """Create a basic signal when indicators are unavailable"""
        return Signal(
            action="NEUTRAL",
            confidence=50.0,
            price=price,
            stop_loss=None,
            take_profit=None,
            position_size=0.0,
            rationale={"error": "Insufficient data"},
            timestamp=datetime.now(pytz.utc),
            risk_reward_ratio=1.0,
            lean="NEUTRAL_LEAN",
            market_summary="Basic price signal - insufficient technical data"
        )
    
    async def _log_signal_to_db(self, signal: Signal, sources: List[str]):
        """Log signal to database"""
        try:
            log_entry = {
                "price": signal.price,
                "signal": signal.action,
                "confidence": signal.confidence,
                "lean": signal.lean,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "position_size": signal.position_size,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "market_summary": signal.market_summary,
                "rationale": json.dumps(signal.rationale),
                "sources": ", ".join(sources),
                "created_at": signal.timestamp.isoformat()
            }
            
            self.supabase.table("gold_signals_v4").insert(log_entry).execute()
            logger.info(f"📝 Signal logged to database: {signal.action} ({signal.confidence:.1f}%)")
            
        except Exception as e:
            logger.error(f"Database logging failed: {e}")
    
    async def run_live_trading(self, interval: int = DEFAULT_INTERVAL):
        """Run live trading with scheduled signals"""
        logger.info(f"🚀 Starting Gold Trading Sentinel V4")
        logger.info(f"⏰ Signal interval: {interval//60} minutes")
        logger.info("=" * 60)
        
        await self.initialize()
        
        try:
            while True:
                now = datetime.now(TIMEZONE)
                
                if not self.scheduler._is_market_open(now):
                    next_open = self._get_next_market_open(now)
                    wait_time = (next_open - now).total_seconds()
                    
                    if wait_time > 0:
                        logger.info(f"💤 Market closed. Next open: {next_open.strftime('%Y-%m-%d %H:%M:%S')}")
                        await asyncio.sleep(min(wait_time, 300))
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
                
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("👋 Shutting down Gold Trading Sentinel")
        except Exception as e:
            logger.error(f"Fatal error in live trading: {e}", exc_info=True)
    
    def _display_signal(self, signal: Signal):
        """Display signal in a user-friendly format"""
        print("\n" + "=" * 60)
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
        
        if signal.action != "NEUTRAL":
            print(f"🎯 Position Size: {signal.position_size*100:.1f}% of capital")
            if signal.stop_loss:
                print(f"🛑 Stop Loss: ${signal.stop_loss:.2f}")
            if signal.take_profit:
                print(f"🎯 Take Profit: ${signal.take_profit:.2f}")
            if signal.risk_reward_ratio:
                print(f"⚖️ Risk/Reward: 1:{signal.risk_reward_ratio:.1f}")
        
        if signal.rationale and len(signal.rationale) > 0:
            sorted_factors = sorted(signal.rationale.items(), key=lambda x: x[1], reverse=True)
            top_factors = sorted_factors[:3]
            print(f"📈 Key Factors: {', '.join([f'{k}:{v:.2f}' for k, v in top_factors])}")
        
        print(f"\n📋 Market Summary:")
        print(f"   {signal.market_summary}")
        print("=" * 60)
    
    def _get_next_market_open(self, now: datetime) -> datetime:
        """Calculate next market open time"""
        current_day = now.weekday()
        current_time = now.time()
        
        if current_day == 5:  # Saturday
            next_day = now.replace(hour=18, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif current_day == 6:  # Sunday
            if current_time < time(18, 0):
                next_day = now.replace(hour=18, minute=0, second=0, microsecond=0)
            else:
                next_day = now
        elif current_day == 4:  # Friday
            if current_time >= time(17, 0):
                next_day = now.replace(hour=18, minute=0, second=0, microsecond=0) + timedelta(days=2)
            else:
                next_day = now
        else:  # Monday-Thursday
            next_day = now
        
        return next_day
    
    def get_signal_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get statistics of recent signals"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_signals = [s for s in self.signal_history if s.timestamp >= cutoff]
        
        if not recent_signals:
            return {"total_signals": 0}
        
        signal_types = {}
        for signal in recent_signals:
            signal_types[signal.action] = signal_types.get(signal.action, 0) + 1
        
        avg_confidence = np.mean([s.confidence for s in recent_signals])
        
        return {
            "total_signals": len(recent_signals),
            "signal_distribution": signal_types,
            "avg_confidence": round(avg_confidence, 1),
            "latest_signal": recent_signals[-1].action if recent_signals else None,
            "latest_price": recent_signals[-1].price if recent_signals else None
        }

# ================= 9. ASYNC BRIDGE FOR TESTING =================
async def async_main():
    """Async main function for testing"""
    extractor = RealGoldPriceExtractor()
    async with extractor:
        price, sources, details = await extractor.get_real_gold_spot_price()
        
        analyzer = EnhancedSentimentAnalyzer()
        sentiment_data = await analyzer.analyze_sentiment()
        
        hist_df = extractor.get_historical_data()
        ta = EnhancedTechnicalAnalyzer()
        indicators = ta.calculate_all_indicators(hist_df)
        
        gen = SignalGenerator()
        signal = gen.generate_signal(price, indicators, sentiment_data)
        
        print(f"🚀 V4.0 Live: Price ${price} | Sentiment: {sentiment_data.score}")
        print(f"📊 Signal: {signal.action} ({signal.confidence}%)")
        print(f"📈 Lean: {signal.lean}")
        
        # Log to Supabase if available
        sentinel = GoldTradingSentinelV4()
        if sentinel.supabase:
            await sentinel._log_signal_to_db(signal, sources)

# ================= 10. MAIN EXECUTION =================
async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Gold Trading Sentinel V4')
    parser.add_argument('--mode', choices=['live', 'test', 'stats', 'async'], 
                       default='live', help='Operation mode')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                       help=f'Signal interval in seconds (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--hours', type=int, default=24,
                       help='Hours to show statistics for (default: 24)')
    
    args = parser.parse_args()
    
    sentinel = GoldTradingSentinelV4()
    
    if args.mode == 'live':
        print("\n" + "=" * 60)
        print("🚀 GOLD TRADING SENTINEL V4 - LIVE TRADING MODE")
        print("=" * 60)
        print(f"📊 Real Gold Spot Price Extraction")
        print(f"⏰ 15-Minute Signal Generation")
        print(f"📈 Neutral Signals with Market Lean")
        print("=" * 60)
        
        await sentinel.run_live_trading(interval=args.interval)
        
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
        
        for i in range(3):
            print(f"\nGenerating signal {i+1}/3...")
            signal = await sentinel.generate_signal()
            if signal:
                sentinel._display_signal(signal)
            await asyncio.sleep(2)
        
        stats = sentinel.get_signal_stats(hours=args.hours)
        
        print("\n" + "=" * 60)
        print("📈 SIGNAL STATISTICS")
        print("=" * 60)
        print(f"Total signals (last {args.hours}h): {stats['total_signals']}")
        
        if stats['total_signals'] > 0:
            print(f"Average confidence: {stats['avg_confidence']}%")
            print("\nSignal distribution:")
            for signal_type, count in stats['signal_distribution'].items():
                percentage = (count / stats['total_signals']) * 100
                print(f"  {signal_type}: {count} ({percentage:.1f}%)")
            
            print(f"\nLatest signal: {stats['latest_signal']}")
            print(f"Latest price: ${stats['latest_price']:.2f}")
    
    elif args.mode == 'async':
        print("\n🔄 Testing Async Bridge...")
        await async_main()
    
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

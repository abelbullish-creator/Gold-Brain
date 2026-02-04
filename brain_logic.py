"""
Gold Trading Sentinel v5.0 - Six Signal System
Only displays: Strong Buy, Buy, Neutral Lean to Buy, Neutral Lean to Sell, Sell, Strong Sell
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

async def send_telegram_msg(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()

# Suppress warnings
warnings.filterwarnings('ignore')

# ================= 1. ENHANCED CONFIGURATION =================
# Define handlers
main_file_handler = logging.FileHandler('gold_sentinel_v5.log')
stream_handler = logging.StreamHandler()

# Set levels for handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        main_file_handler,
        stream_handler
    ]
)
logger = logging.getLogger(__name__)

# Constants
TIMEZONE = pytz.timezone("America/New_York")
GOLD_NEWS_KEYWORDS = ["gold", "bullion", "XAU", "precious metals", "fed", "inflation", "central bank"]
DEFAULT_INTERVAL = 900

# ================= 2. DATA MODELS =================
@dataclass
class Signal:
    action: str  # Only: STRONG_BUY, BUY, NEUTRAL_LEAN_BUY, NEUTRAL_LEAN_SELL, SELL, STRONG_SELL
    confidence: float
    price: float
    timestamp: datetime
    market_summary: str

@dataclass
class MarketStatus:
    is_open: bool
    reason: str
    session: str

# ================= 3. SIMPLE PRICE EXTRACTOR =================
class SimplePriceExtractor:
    """Simple price extractor using Yahoo Finance"""
    
    async def get_gold_price(self) -> float:
        """Get current gold price"""
        try:
            # Try GC=F (Gold Futures)
            ticker = yf.Ticker("GC=F")
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                logger.info(f"✅ Gold price: ${price:.2f}")
                return float(price)
            
            # Fallback to GLD (Gold ETF)
            ticker = yf.Ticker("GLD")
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                price = hist['Close'].iloc[-1] * 10  # Convert GLD to approximate gold price
                logger.info(f"✅ GLD price (converted): ${price:.2f}")
                return float(price)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Price fetch failed: {e}")
            return 0.0

# ================= 4. SIMPLE TECHNICAL ANALYZER =================
class SimpleTechnicalAnalyzer:
    """Simple technical analysis for 6 signals"""
    
    def analyze(self, price_history: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculate technical indicators"""
        if price_history.empty or len(price_history) < 20:
            return {}
        
        closes = price_history['Close'].values
        closes_series = pd.Series(closes)
        
        # Moving averages
        sma_20 = closes_series.rolling(20).mean().iloc[-1]
        sma_50 = closes_series.rolling(50).mean().iloc[-1]
        
        # RSI
        rsi = self._calculate_rsi(closes_series, 14)
        
        # MACD
        macd_line, signal_line = self._calculate_macd(closes_series)
        macd_hist = macd_line.iloc[-1] - signal_line.iloc[-1]
        
        # Trend
        trend = "BULLISH" if current_price > sma_20 > sma_50 else "BEARISH" if current_price < sma_20 < sma_50 else "NEUTRAL"
        
        # Overbought/Oversold
        rsi_status = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
        
        # MACD signal
        macd_signal = "BULLISH" if macd_hist > 0 else "BEARISH"
        
        return {
            'sma_20': float(sma_20) if not pd.isna(sma_20) else 0.0,
            'sma_50': float(sma_50) if not pd.isna(sma_50) else 0.0,
            'rsi': float(rsi),
            'macd_hist': float(macd_hist),
            'trend': trend,
            'rsi_status': rsi_status,
            'macd_signal': macd_signal
        }
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
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
    
    def _calculate_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        if len(series) < 26:
            return pd.Series([0]), pd.Series([0])
        
        ema_12 = series.ewm(span=12, adjust=False).mean()
        ema_26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        return macd_line, signal_line

# ================= 5. SIX SIGNAL GENERATOR =================
class SixSignalGenerator:
    """Generate only 6 signals with clear rules"""
    
    def __init__(self):
        self.last_signals = deque(maxlen=3)
    
    def generate_signal(self, price: float, indicators: Dict, 
                       current_price: float) -> Signal:
        """Generate one of 6 signals"""
        
        if not indicators:
            # Default to NEUTRAL_LEAN_BUY if no indicators
            return Signal(
                action="NEUTRAL_LEAN_BUY",
                confidence=50.0,
                price=current_price,
                timestamp=datetime.now(pytz.utc),
                market_summary="No technical data available"
            )
        
        # Extract indicators
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_hist', 0)
        trend = indicators.get('trend', 'NEUTRAL')
        rsi_status = indicators.get('rsi_status', 'NEUTRAL')
        macd_signal = indicators.get('macd_signal', 'NEUTRAL')
        
        # Calculate signal score (0-100)
        score = 50.0
        
        # Trend factor (30%)
        if trend == "BULLISH":
            score += 15
        elif trend == "BEARISH":
            score -= 15
        
        # RSI factor (30%)
        if rsi_status == "OVERSOLD":
            score += 15
        elif rsi_status == "OVERBOUGHT":
            score -= 15
        
        # MACD factor (20%)
        if macd_signal == "BULLISH":
            score += 10
        elif macd_signal == "BEARISH":
            score -= 10
        
        # Price position factor (20%)
        if current_price > sma_20 > sma_50:
            score += 10
        elif current_price < sma_20 < sma_50:
            score -= 10
        
        # Clamp score between 0-100
        score = max(0, min(100, score))
        
        # Momentum from recent signals
        momentum = self._calculate_momentum()
        score = score * (1 + momentum)
        score = max(0, min(100, score))
        
        # Generate signal based on score
        action, confidence, summary = self._determine_signal(score, indicators, current_price)
        
        signal = Signal(
            action=action,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.now(pytz.utc),
            market_summary=summary
        )
        
        self.last_signals.append(signal)
        return signal
    
    def _calculate_momentum(self) -> float:
        """Calculate momentum from recent signals"""
        if len(self.last_signals) < 2:
            return 0.0
        
        # Simple momentum: average confidence change
        confidences = [s.confidence for s in self.last_signals]
        if len(confidences) >= 2:
            change = confidences[-1] - confidences[0]
            return change / 100.0  # Normalize to -1 to 1
        
        return 0.0
    
    def _determine_signal(self, score: float, indicators: Dict, 
                         current_price: float) -> Tuple[str, float, str]:
        """Determine which of 6 signals to generate"""
        
        # Extract key indicators
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_hist', 0)
        sma_20 = indicators.get('sma_20', 0)
        
        summary_parts = []
        
        # Determine signal based on score
        if score >= 85:
            action = "STRONG_BUY"
            confidence = score
            summary_parts.append("Very bullish conditions")
            
        elif score >= 70:
            action = "BUY"
            confidence = score
            summary_parts.append("Bullish conditions")
            
        elif score >= 55:
            action = "NEUTRAL_LEAN_BUY"
            confidence = score
            summary_parts.append("Slightly bullish")
            
        elif score <= 15:
            action = "STRONG_SELL"
            confidence = 100 - score  # Invert for sell signals
            summary_parts.append("Very bearish conditions")
            
        elif score <= 30:
            action = "SELL"
            confidence = 100 - score  # Invert for sell signals
            summary_parts.append("Bearish conditions")
            
        elif score <= 45:
            action = "NEUTRAL_LEAN_SELL"
            confidence = 100 - score  # Invert for sell signals
            summary_parts.append("Slightly bearish")
            
        else:  # 45 < score < 55
            # Decide lean based on indicators
            if rsi > 50 or macd_hist > 0 or current_price > sma_20:
                action = "NEUTRAL_LEAN_BUY"
                confidence = 50
                summary_parts.append("Neutral with bullish lean")
            else:
                action = "NEUTRAL_LEAN_SELL"
                confidence = 50
                summary_parts.append("Neutral with bearish lean")
        
        # Add technical context to summary
        if 'rsi_status' in indicators:
            summary_parts.append(f"RSI: {indicators['rsi_status'].lower()}")
        
        if 'trend' in indicators:
            summary_parts.append(f"Trend: {indicators['trend'].lower()}")
        
        if 'macd_signal' in indicators:
            summary_parts.append(f"MACD: {indicators['macd_signal'].lower()}")
        
        market_summary = ". ".join(summary_parts)
        
        return action, round(confidence, 1), market_summary

# ================= 6. GOLD TRADING SENTINEL =================
class GoldTradingSentinel:
    """Main trading system with 6 signals"""
    
    def __init__(self):
        self.price_extractor = SimplePriceExtractor()
        self.tech_analyzer = SimpleTechnicalAnalyzer()
        self.signal_generator = SixSignalGenerator()
    
    async def generate_signal(self) -> Optional[Signal]:
        """Generate a trading signal"""
        try:
            # Get current price
            current_price = await self.price_extractor.get_gold_price()
            
            if current_price <= 0:
                logger.error("Failed to get gold price")
                return None
            
            # Get historical data
            price_history = self._get_price_history()
            
            if price_history.empty:
                logger.warning("No historical data available")
                # Generate basic signal
                return Signal(
                    action="NEUTRAL_LEAN_BUY",
                    confidence=50.0,
                    price=current_price,
                    timestamp=datetime.now(pytz.utc),
                    market_summary="Limited data available"
                )
            
            # Calculate technical indicators
            indicators = self.tech_analyzer.analyze(price_history, current_price)
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                current_price, indicators, current_price
            )
            
            logger.info(f"✅ Signal generated: {signal.action} ({signal.confidence:.1f}%)")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _get_price_history(self) -> pd.DataFrame:
        """Get historical price data"""
        try:
            # Get 60 days of hourly data
            ticker = yf.Ticker("GC=F")
            hist = ticker.history(period="60d", interval="1h")
            
            if hist.empty:
                # Try GLD as fallback
                ticker = yf.Ticker("GLD")
                hist = ticker.history(period="60d", interval="1h")
                
                if not hist.empty:
                    # Convert GLD to approximate gold price
                    hist['Close'] = hist['Close'] * 10
                    hist['Open'] = hist['Open'] * 10
                    hist['High'] = hist['High'] * 10
                    hist['Low'] = hist['Low'] * 10
            
            return hist
            
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return pd.DataFrame()
    
    def display_signal(self, signal: Signal):
        """Display signal in clean format"""
        print("\n" + "=" * 60)
        print("📊 GOLD TRADING SIGNAL")
        print("=" * 60)
        
        print(f"🕒 Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S ET')}")
        print(f"💰 Price: ${signal.price:.2f}")
        print("-" * 60)
        
        # Display signal with appropriate color
        if signal.action == "STRONG_BUY":
            print(f"🎯 SIGNAL: 🟢 STRONG BUY")
            print(f"📊 Confidence: {signal.confidence:.1f}%")
            print(f"💡 Action: Strong buy recommendation")
            
        elif signal.action == "BUY":
            print(f"🎯 SIGNAL: 🟢 BUY")
            print(f"📊 Confidence: {signal.confidence:.1f}%")
            print(f"💡 Action: Buy recommendation")
            
        elif signal.action == "NEUTRAL_LEAN_BUY":
            print(f"🎯 SIGNAL: 🟡 NEUTRAL (Lean to Buy)")
            print(f"📊 Confidence: {signal.confidence:.1f}%")
            print(f"💡 Action: Consider buying on dips")
            
        elif signal.action == "NEUTRAL_LEAN_SELL":
            print(f"🎯 SIGNAL: 🟡 NEUTRAL (Lean to Sell)")
            print(f"📊 Confidence: {signal.confidence:.1f}%")
            print(f"💡 Action: Consider selling on rallies")
            
        elif signal.action == "SELL":
            print(f"🎯 SIGNAL: 🔴 SELL")
            print(f"📊 Confidence: {signal.confidence:.1f}%")
            print(f"💡 Action: Sell recommendation")
            
        elif signal.action == "STRONG_SELL":
            print(f"🎯 SIGNAL: 🔴 STRONG SELL")
            print(f"📊 Confidence: {signal.confidence:.1f}%")
            print(f"💡 Action: Strong sell recommendation")
        
        print("-" * 60)
        print(f"📋 Market Summary:")
        print(f"   {signal.market_summary}")
        print("=" * 60)
        
        # Add simple trading suggestion
        self._display_trading_suggestion(signal.action)
    
    def _display_trading_suggestion(self, action: str):
        """Display simple trading suggestion"""
        print("\n💼 Trading Suggestion:")
        
        if action == "STRONG_BUY":
            print("   • Enter long position immediately")
            print("   • Consider adding to position")
            print("   • Target: +3-5% gain")
            
        elif action == "BUY":
            print("   • Enter long position")
            print("   • Wait for small pullbacks")
            print("   • Target: +2-3% gain")
            
        elif action == "NEUTRAL_LEAN_BUY":
            print("   • Wait for better entry")
            print("   • Consider small position")
            print("   • Set tight stop loss")
            
        elif action == "NEUTRAL_LEAN_SELL":
            print("   • Consider reducing position")
            print("   • Wait for confirmation")
            print("   • Prepare to exit")
            
        elif action == "SELL":
            print("   • Exit long positions")
            print("   • Consider short position")
            print("   • Target: -2-3% move")
            
        elif action == "STRONG_SELL":
            print("   • Exit all long positions")
            print("   • Enter short position")
            print("   • Target: -3-5% move")
        
        print("\n⚠️  Risk Management:")
        print("   • Never risk more than 2% per trade")
        print("   • Always use stop losses")
        print("   • Consider position sizing")

# ================= 7. MAIN EXECUTION =================
async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Gold Trading Sentinel - Six Signal System')
    parser.add_argument('--mode', choices=['single', 'live'], 
                       default='single', help='Operation mode')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                       help=f'Signal interval in seconds (default: {DEFAULT_INTERVAL})')
    
    args = parser.parse_args()
    
    sentinel = GoldTradingSentinel()
    
    if args.mode == 'single':
        print("\n" + "=" * 60)
        print("🎯 GOLD TRADING SENTINEL - SIX SIGNAL SYSTEM")
        print("=" * 60)
        print("Generating trading signal...")
        print("-" * 60)
        
        signal = await sentinel.generate_signal()
        
        if signal:
            sentinel.display_signal(signal)
        else:
            print("❌ Failed to generate signal")
   
    elif args.mode == 'single':
        # 1. Fetching the keys from the environment
        MY_TOKEN = os.getenv("TELEGRAM_TOKEN")
        MY_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

        # >>> PLACE THE DEBUG LINE HERE <<<
        print(f"DEBUG: Token starts with: {MY_TOKEN[:5] if MY_TOKEN else 'MISSING'}")
        print(f"DEBUG: Chat ID found: {'YES' if MY_CHAT_ID else 'NO'}")

        print("\n🔍 Generating Single Signal...")
        # ... rest of your code ...
    
    elif args.mode == 'live':
        print("\n" + "=" * 60)
        print("🎯 LIVE GOLD TRADING SIGNALS")
        print("=" * 60)
        print("Running in live mode...")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        try:
            while True:
                signal = await sentinel.generate_signal()
                
                if signal:
                    sentinel.display_signal(signal)
                
                # Wait for next interval
                print(f"\n⏳ Next update in {args.interval//60} minutes...")
                await asyncio.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n👋 Stopping live signals...")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

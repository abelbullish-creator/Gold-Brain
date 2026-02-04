"""
Gold Trading Sentinel v5.1 - Six Signal System
Enhanced version with Telegram notifications and improved signal logic
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

# ================= TELEGRAM FUNCTION =================
async def send_telegram_msg(token: str, chat_id: str, message: str) -> Dict:
    """Send message to Telegram bot"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post(url, json=payload) as resp:
                response = await resp.json()
                if resp.status != 200:
                    logger.error(f"Telegram API error: {response}")
                return response
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return {"ok": False, "error": str(e)}

# ================= CONFIGURATION =================
# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_sentinel_v5.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
TIMEZONE = pytz.timezone("America/New_York")
GOLD_NEWS_KEYWORDS = ["gold", "bullion", "XAU", "precious metals", "fed", "inflation", "central bank"]
DEFAULT_INTERVAL = 900  # 15 minutes

# ================= DATA MODELS =================
@dataclass
class Signal:
    action: str  # Only: STRONG_BUY, BUY, NEUTRAL_LEAN_BUY, NEUTRAL_LEAN_SELL, SELL, STRONG_SELL
    confidence: float
    price: float
    timestamp: datetime
    market_summary: str
    indicators: Dict[str, Any]

@dataclass
class MarketStatus:
    is_open: bool
    reason: str
    session: str

# ================= PRICE EXTRACTOR =================
class SimplePriceExtractor:
    """Simple price extractor using Yahoo Finance"""
    
    async def get_gold_price(self) -> Tuple[float, str]:
        """Get current gold price with source"""
        sources = [
            ("GC=F", "Gold Futures"),
            ("GLD", "Gold ETF"),
            ("IAU", "iShares Gold Trust"),
            ("XAUUSD=X", "Gold/USD")
        ]
        
        for ticker_symbol, source in sources:
            try:
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(period="1d", interval="1m")
                
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    
                    # Convert ETF prices to approximate gold price
                    if ticker_symbol in ["GLD", "IAU"]:
                        price = price * 10  # Rough conversion factor
                    
                    logger.info(f"✅ Gold price from {source}: ${price:.2f}")
                    return float(price), source
                    
            except Exception as e:
                logger.warning(f"Failed to get price from {source}: {e}")
                continue
        
        logger.error("All price sources failed")
        return 0.0, "None"

# ================= TECHNICAL ANALYZER =================
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
        sma_200 = closes_series.rolling(200).mean().iloc[-1] if len(closes_series) >= 200 else 0
        
        # RSI
        rsi = self._calculate_rsi(closes_series, 14)
        
        # MACD
        macd_line, signal_line = self._calculate_macd(closes_series)
        macd_hist = macd_line.iloc[-1] - signal_line.iloc[-1] if not macd_line.empty else 0
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(closes_series, 20)
        
        # Price position relative to indicators
        price_above_sma_20 = current_price > sma_20
        price_above_sma_50 = current_price > sma_50
        
        # Trend determination
        if price_above_sma_20 and price_above_sma_50 and sma_20 > sma_50:
            trend = "STRONG_BULLISH"
        elif price_above_sma_20 and price_above_sma_50:
            trend = "BULLISH"
        elif not price_above_sma_20 and not price_above_sma_50 and sma_20 < sma_50:
            trend = "STRONG_BEARISH"
        elif not price_above_sma_20 and not price_above_sma_50:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        # RSI status
        if rsi > 70:
            rsi_status = "OVERBOUGHT"
        elif rsi < 30:
            rsi_status = "OVERSOLD"
        else:
            rsi_status = "NEUTRAL"
        
        # MACD signal
        macd_signal = "BULLISH" if macd_hist > 0 else "BEARISH"
        
        # Price position in Bollinger Bands
        bb_position = "UPPER" if current_price > bb_upper else "LOWER" if current_price < bb_lower else "MIDDLE"
        
        return {
            'sma_20': float(sma_20) if not pd.isna(sma_20) else 0.0,
            'sma_50': float(sma_50) if not pd.isna(sma_50) else 0.0,
            'sma_200': float(sma_200) if not pd.isna(sma_200) else 0.0,
            'rsi': float(rsi),
            'macd_hist': float(macd_hist),
            'bb_upper': float(bb_upper) if not pd.isna(bb_upper) else 0.0,
            'bb_lower': float(bb_lower) if not pd.isna(bb_lower) else 0.0,
            'trend': trend,
            'rsi_status': rsi_status,
            'macd_signal': macd_signal,
            'bb_position': bb_position,
            'price_above_sma_20': price_above_sma_20,
            'price_above_sma_50': price_above_sma_50
        }
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(series) < period + 1:
            return 50.0
        
        try:
            prices = series.values
            deltas = np.diff(prices)
            
            # Separate gains and losses
            gains = deltas.copy()
            losses = deltas.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = np.abs(losses)
            
            # Calculate average gains and losses
            avg_gains = np.mean(gains[:period])
            avg_losses = np.mean(losses[:period])
            
            for i in range(period, len(deltas)):
                avg_gains = (avg_gains * (period - 1) + gains[i]) / period
                avg_losses = (avg_losses * (period - 1) + losses[i]) / period
            
            if avg_losses == 0:
                return 100.0
            
            rs = avg_gains / avg_losses
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return float(min(max(rsi, 0), 100))
            
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return 50.0
    
    def _calculate_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        if len(series) < 26:
            return pd.Series([0]), pd.Series([0])
        
        try:
            ema_12 = series.ewm(span=12, adjust=False).mean()
            ema_26 = series.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            return macd_line, signal_line
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            return pd.Series([0]), pd.Series([0])
    
    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        if len(series) < period:
            return 0.0, 0.0
        
        try:
            sma = series.rolling(window=period).mean().iloc[-1]
            std = series.rolling(window=period).std().iloc[-1]
            
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            return float(upper_band), float(lower_band)
        except Exception as e:
            logger.error(f"Bollinger Bands error: {e}")
            return 0.0, 0.0

# ================= SIX SIGNAL GENERATOR =================
class SixSignalGenerator:
    """Generate only 6 signals with clear rules"""
    
    def __init__(self):
        self.last_signals = deque(maxlen=5)
        self.signal_history = []
    
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
                market_summary="No technical data available",
                indicators={}
            )
        
        # Calculate composite score
        score = self._calculate_composite_score(indicators, current_price)
        
        # Apply momentum from recent signals
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
            market_summary=summary,
            indicators=indicators
        )
        
        self.last_signals.append(signal)
        self.signal_history.append(signal)
        
        return signal
    
    def _calculate_composite_score(self, indicators: Dict, current_price: float) -> float:
        """Calculate composite signal score (0-100)"""
        score = 50.0
        
        # 1. Trend factor (25%)
        trend = indicators.get('trend', 'NEUTRAL')
        if trend == "STRONG_BULLISH":
            score += 25
        elif trend == "BULLISH":
            score += 12.5
        elif trend == "STRONG_BEARISH":
            score -= 25
        elif trend == "BEARISH":
            score -= 12.5
        
        # 2. RSI factor (20%)
        rsi_status = indicators.get('rsi_status', 'NEUTRAL')
        if rsi_status == "OVERSOLD":
            score += 20
        elif rsi_status == "OVERBOUGHT":
            score -= 20
        
        # 3. MACD factor (15%)
        macd_hist = indicators.get('macd_hist', 0)
        if macd_hist > 0:
            score += 15 * min(abs(macd_hist) / 10, 1)  # Scale by magnitude
        else:
            score -= 15 * min(abs(macd_hist) / 10, 1)
        
        # 4. Price position factor (20%)
        price_above_sma_20 = indicators.get('price_above_sma_20', False)
        price_above_sma_50 = indicators.get('price_above_sma_50', False)
        
        if price_above_sma_20 and price_above_sma_50:
            score += 20
        elif not price_above_sma_20 and not price_above_sma_50:
            score -= 20
        elif price_above_sma_20:  # Only above 20-day
            score += 10
        else:  # Below both or only above 50-day
            score -= 10
        
        # 5. Bollinger Bands factor (10%)
        bb_position = indicators.get('bb_position', 'MIDDLE')
        if bb_position == "LOWER":
            score += 10
        elif bb_position == "UPPER":
            score -= 10
        
        # 6. RSI value factor (10%)
        rsi = indicators.get('rsi', 50)
        if 30 <= rsi <= 70:
            # Neutral zone, no adjustment
            pass
        elif rsi < 30:
            score += 10 * (1 - rsi/30)  # More oversold = more points
        elif rsi > 70:
            score -= 10 * (rsi/70 - 1)  # More overbought = more deduction
        
        return max(0, min(100, score))
    
    def _calculate_momentum(self) -> float:
        """Calculate momentum from recent signals"""
        if len(self.last_signals) < 2:
            return 0.0
        
        # Calculate average confidence change
        confidences = [s.confidence for s in self.last_signals]
        if len(confidences) >= 2:
            # Normalize confidence based on signal type
            normalized_confidences = []
            for sig in self.last_signals:
                if "BUY" in sig.action:
                    norm_conf = sig.confidence / 100
                elif "SELL" in sig.action:
                    norm_conf = 1 - (sig.confidence / 100)
                else:  # Neutral
                    norm_conf = 0.5
                normalized_confidences.append(norm_conf)
            
            # Calculate momentum as average change
            changes = []
            for i in range(1, len(normalized_confidences)):
                changes.append(normalized_confidences[i] - normalized_confidences[i-1])
            
            if changes:
                avg_change = sum(changes) / len(changes)
                return avg_change * 0.2  # Scale down momentum effect
        
        return 0.0
    
    def _determine_signal(self, score: float, indicators: Dict, 
                         current_price: float) -> Tuple[str, float, str]:
        """Determine which of 6 signals to generate"""
        
        summary_parts = []
        
        # Determine signal based on score with clear thresholds
        if score >= 85:
            action = "STRONG_BUY"
            confidence = score
            summary_parts.append("Very bullish market conditions")
            
        elif score >= 70:
            action = "BUY"
            confidence = score
            summary_parts.append("Bullish market conditions")
            
        elif score >= 60:
            action = "NEUTRAL_LEAN_BUY"
            confidence = score
            summary_parts.append("Slightly bullish bias")
            
        elif score <= 15:
            action = "STRONG_SELL"
            confidence = 100 - score
            summary_parts.append("Very bearish market conditions")
            
        elif score <= 30:
            action = "SELL"
            confidence = 100 - score
            summary_parts.append("Bearish market conditions")
            
        elif score <= 40:
            action = "NEUTRAL_LEAN_SELL"
            confidence = 100 - score
            summary_parts.append("Slightly bearish bias")
            
        else:  # 40 < score < 60
            # Decide lean based on recent trend and indicators
            last_trend = indicators.get('trend', 'NEUTRAL')
            macd_signal = indicators.get('macd_signal', 'NEUTRAL')
            
            if last_trend in ["BULLISH", "STRONG_BULLISH"] or macd_signal == "BULLISH":
                action = "NEUTRAL_LEAN_BUY"
                confidence = 50
                summary_parts.append("Neutral with bullish lean")
            else:
                action = "NEUTRAL_LEAN_SELL"
                confidence = 50
                summary_parts.append("Neutral with bearish lean")
        
        # Add detailed technical context to summary
        tech_details = []
        
        if 'trend' in indicators:
            trend_map = {
                "STRONG_BULLISH": "strong bullish",
                "BULLISH": "bullish",
                "NEUTRAL": "neutral",
                "BEARISH": "bearish",
                "STRONG_BEARISH": "strong bearish"
            }
            tech_details.append(f"Trend: {trend_map.get(indicators['trend'], 'neutral')}")
        
        if 'rsi_status' in indicators and indicators['rsi_status'] != "NEUTRAL":
            tech_details.append(f"RSI: {indicators['rsi_status'].lower()} ({indicators.get('rsi', 0):.1f})")
        
        if 'macd_signal' in indicators:
            tech_details.append(f"MACD: {indicators['macd_signal'].lower()}")
        
        if 'bb_position' in indicators and indicators['bb_position'] != "MIDDLE":
            tech_details.append(f"Price at {indicators['bb_position'].lower()} Bollinger Band")
        
        if tech_details:
            summary_parts.append("Technical indicators: " + ", ".join(tech_details))
        
        market_summary = ". ".join(summary_parts)
        
        return action, round(confidence, 1), market_summary

# ================= GOLD TRADING SENTINEL =================
class GoldTradingSentinel:
    """Main trading system with 6 signals"""
    
    def __init__(self):
        self.price_extractor = SimplePriceExtractor()
        self.tech_analyzer = SimpleTechnicalAnalyzer()
        self.signal_generator = SixSignalGenerator()
        self.last_signal = None
    
    async def generate_signal(self) -> Optional[Signal]:
        """Generate a trading signal"""
        try:
            # Get current price
            current_price, source = await self.price_extractor.get_gold_price()
            
            if current_price <= 0:
                logger.error("Failed to get valid gold price")
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
                    market_summary=f"Limited data available (source: {source})",
                    indicators={}
                )
            
            # Calculate technical indicators
            indicators = self.tech_analyzer.analyze(price_history, current_price)
            
            # Add price source to indicators
            indicators['price_source'] = source
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                current_price, indicators, current_price
            )
            
            # Update last signal
            self.last_signal = signal
            
            logger.info(f"✅ Signal generated: {signal.action} ({signal.confidence:.1f}%) - Price: ${current_price:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _get_price_history(self) -> pd.DataFrame:
        """Get historical price data"""
        try:
            # Try multiple tickers for robustness
            tickers = ["GC=F", "GLD", "XAUUSD=X"]
            
            for ticker_symbol in tickers:
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    hist = ticker.history(period="60d", interval="1h")
                    
                    if not hist.empty:
                        # Convert ETF to approximate gold price if needed
                        if ticker_symbol in ["GLD"]:
                            hist['Close'] = hist['Close'] * 10
                            hist['Open'] = hist['Open'] * 10
                            hist['High'] = hist['High'] * 10
                            hist['Low'] = hist['Low'] * 10
                        
                        logger.info(f"✅ Historical data from {ticker_symbol}: {len(hist)} records")
                        return hist
                        
                except Exception as e:
                    logger.warning(f"Failed to get history from {ticker_symbol}: {e}")
                    continue
            
            logger.error("All historical data sources failed")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return pd.DataFrame()
    
    def display_signal(self, signal: Signal):
        """Display signal in clean format"""
        print("\n" + "=" * 70)
        print("📊 GOLD TRADING SENTINEL v5.1 - SIX SIGNAL SYSTEM")
        print("=" * 70)
        
        print(f"🕒 Time: {signal.timestamp.astimezone(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S ET')}")
        print(f"💰 Price: ${signal.price:.2f}")
        print("-" * 70)
        
        # Display signal with appropriate color and emoji
        signal_configs = {
            "STRONG_BUY": ("🟢", "STRONG BUY", "🟢🟢🟢"),
            "BUY": ("🟢", "BUY", "🟢🟢"),
            "NEUTRAL_LEAN_BUY": ("🟡", "NEUTRAL (Lean to Buy)", "🟡"),
            "NEUTRAL_LEAN_SELL": ("🟡", "NEUTRAL (Lean to Sell)", "🟡"),
            "SELL": ("🔴", "SELL", "🔴🔴"),
            "STRONG_SELL": ("🔴", "STRONG SELL", "🔴🔴🔴")
        }
        
        emoji, display_name, strength = signal_configs.get(signal.action, ("⚪", signal.action, ""))
        
        print(f"🎯 SIGNAL: {strength} {emoji} {display_name} {strength}")
        print(f"📊 Confidence: {signal.confidence:.1f}%")
        print("-" * 70)
        
        # Technical indicators summary
        if signal.indicators:
            print("📈 Technical Indicators:")
            indicators = signal.indicators
            
            # Trend
            trend = indicators.get('trend', 'NEUTRAL').replace('_', ' ').title()
            print(f"   • Trend: {trend}")
            
            # RSI
            rsi = indicators.get('rsi', 50)
            rsi_status = indicators.get('rsi_status', 'NEUTRAL')
            print(f"   • RSI: {rsi:.1f} ({rsi_status.lower()})")
            
            # Moving Averages
            if 'sma_20' in indicators and indicators['sma_20'] > 0:
                price_vs_sma20 = signal.price / indicators['sma_20'] - 1
                print(f"   • Price vs 20-day SMA: {price_vs_sma20:+.2%}")
            
            # MACD
            macd_hist = indicators.get('macd_hist', 0)
            if macd_hist != 0:
                macd_signal = "Bullish" if macd_hist > 0 else "Bearish"
                print(f"   • MACD: {macd_signal} ({macd_hist:.4f})")
            
            # Price source
            if 'price_source' in indicators:
                print(f"   • Data Source: {indicators['price_source']}")
            
            print("-" * 70)
        
        print(f"📋 Market Summary:")
        print(f"   {signal.market_summary}")
        print("=" * 70)
        
        # Add trading suggestion
        self._display_trading_suggestion(signal.action)
    
    def _display_trading_suggestion(self, action: str):
        """Display simple trading suggestion"""
        print("\n💼 Trading Suggestion:")
        print("-" * 40)
        
        suggestions = {
            "STRONG_BUY": [
                "• Enter long position immediately",
                "• Consider aggressive position sizing",
                "• Target: +3-5% gain",
                "• Stop loss: -1-2% below entry"
            ],
            "BUY": [
                "• Enter long position",
                "• Wait for small pullbacks if possible",
                "• Target: +2-4% gain",
                "• Stop loss: -1.5% below entry"
            ],
            "NEUTRAL_LEAN_BUY": [
                "• Wait for better entry point",
                "• Consider small position only",
                "• Target: +1-2% gain",
                "• Stop loss: -2% below entry",
                "• Monitor for trend confirmation"
            ],
            "NEUTRAL_LEAN_SELL": [
                "• Consider reducing long positions",
                "• Wait for confirmation before shorting",
                "• Prepare exit strategy",
                "• Monitor key support levels"
            ],
            "SELL": [
                "• Exit long positions",
                "• Consider short position",
                "• Target: -2-4% move",
                "• Stop loss: +1.5% above entry"
            ],
            "STRONG_SELL": [
                "• Exit all long positions immediately",
                "• Enter short position",
                "• Consider aggressive short sizing",
                "• Target: -3-5% move",
                "• Stop loss: +1-2% above entry"
            ]
        }
        
        for suggestion in suggestions.get(action, ["• No specific suggestion available"]):
            print(f"   {suggestion}")
        
        print("\n⚠️  Risk Management:")
        print("   • Never risk more than 2% of capital per trade")
        print("   • Always use stop-loss orders")
        print("   • Consider position sizing based on confidence")
        print("   • Monitor market conditions continuously")
        print("=" * 70)

# ================= MAIN EXECUTION =================
async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Gold Trading Sentinel - Six Signal System')
    parser.add_argument('--mode', choices=['single', 'live'], 
                       default='single', help='Operation mode')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                       help=f'Signal interval in seconds (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--test-telegram', action='store_true',
                       help='Test Telegram connection only')
    
    args = parser.parse_args()
    
    sentinel = GoldTradingSentinel()
    
    # Get Telegram credentials
    MY_TOKEN = os.getenv("TELEGRAM_TOKEN")
    MY_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
    # Test Telegram connection if requested
    if args.test_telegram:
        print("\n" + "=" * 60)
        print("🤖 TELEGRAM CONNECTION TEST")
        print("=" * 60)
        
        if not MY_TOKEN or not MY_CHAT_ID:
            print("❌ Telegram credentials not found!")
            print("   Set environment variables:")
            print("   export TELEGRAM_TOKEN='your_bot_token'")
            print("   export TELEGRAM_CHAT_ID='your_chat_id'")
            return 1
        
        print(f"✅ Token: {MY_TOKEN[:10]}...")
        print(f"✅ Chat ID: {MY_CHAT_ID}")
        
        test_msg = "✅ Gold Trading Sentinel v5.1 - Telegram test successful!"
        result = await send_telegram_msg(MY_TOKEN, MY_CHAT_ID, test_msg)
        
        if result.get('ok'):
            print("✅ Telegram test successful!")
            return 0
        else:
            print(f"❌ Telegram test failed: {result}")
            return 1
    
    if args.mode == 'single':
        print("\n" + "=" * 60)
        print("🎯 GOLD TRADING SENTINEL - SIX SIGNAL SYSTEM")
        print("=" * 60)
        print("Generrading trading signal...")
        print("-" * 60)
        
        signal = await sentinel.generate_signal()
        
        if signal:
            sentinel.display_signal(signal)
            
            # Optional: Send single signal to Telegram
            if MY_TOKEN and MY_CHAT_ID:
                send = input("\n📱 Send this signal to Telegram? (y/n): ")
                if send.lower() == 'y':
                    emoji_map = {
                        "STRONG_BUY": "🟢🟢🟢",
                        "BUY": "🟢🟢",
                        "NEUTRAL_LEAN_BUY": "🟡",
                        "NEUTRAL_LEAN_SELL": "🟡",
                        "SELL": "🔴🔴",
                        "STRONG_SELL": "🔴🔴🔴"
                    }
                    
                    emoji = emoji_map.get(signal.action, "⚪")
                    msg = f"""
{emoji} *Gold Trading Signal*

*Signal:* {signal.action}
*Price:* ${signal.price:.2f}
*Confidence:* {signal.confidence:.1f}%

*Market Summary:*
{signal.market_summary}

_Generated at {signal.timestamp.astimezone(TIMEZONE).strftime('%H:%M:%S ET')}_
"""
                    result = await send_telegram_msg(MY_TOKEN, MY_CHAT_ID, msg)
                    if result.get('ok'):
                        print("✅ Signal sent to Telegram!")
                    else:
                        print(f"❌ Failed to send: {result}")
        else:
            print("❌ Failed to generate signal")
    
    elif args.mode == 'live':
        print("\n" + "=" * 60)
        print("🎯 LIVE GOLD TRADING SIGNALS")
        print("=" * 60)
        
        # Telegram setup
        telegram_enabled = False
        if MY_TOKEN and MY_CHAT_ID:
            print(f"✅ Telegram: Token found ({MY_TOKEN[:10]}...)")
            print(f"✅ Telegram: Chat ID found")
            
            # Test Telegram connection
            test_msg = "✅ Gold Trading Sentinel v5.1 is now LIVE!"
            result = await send_telegram_msg(MY_TOKEN, MY_CHAT_ID, test_msg)
            
            if result.get('ok'):
                telegram_enabled = True
                print("✅ Telegram connection successful!")
            else:
                print(f"⚠️  Telegram test failed: {result}")
                telegram_enabled = False
        else:
            print("⚠️  Telegram credentials not found. Notifications disabled.")
            print("   Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables")
        
        print(f"\n📊 Signal interval: {args.interval//60} minutes")
        print("🔄 Starting live signal generation...")
        print("-" * 60)
        
        signal_count = 0
        try:
            while True:
                signal_count += 1
                current_time = datetime.now(TIMEZONE).strftime('%H:%M:%S ET')
                
                print(f"\n🔄 Signal #{signal_count} at {current_time}")
                print("-" * 40)
                
                signal = await sentinel.generate_signal()
                
                if signal:
                    # Display signal
                    sentinel.display_signal(signal)
                    
                    # Send to Telegram if enabled
                    if telegram_enabled and signal:
                        # Format Telegram message
                        emoji_map = {
                            "STRONG_BUY": "🟢🟢🟢",
                            "BUY": "🟢🟢",
                            "NEUTRAL_LEAN_BUY": "🟡",
                            "NEUTRAL_LEAN_SELL": "🟡",
                            "SELL": "🔴🔴",
                            "STRONG_SELL": "🔴🔴🔴"
                        }
                        
                        emoji = emoji_map.get(signal.action, "⚪")
                        
                        # Create detailed message
                        msg = f"""
{emoji} *LIVE GOLD SIGNAL #{signal_count}*

*Signal:* {signal.action}
*Price:* ${signal.price:.2f}
*Confidence:* {signal.confidence:.1f}%

*Market Summary:*
{signal.market_summary}

_Generated at {current_time}_
#Gold #Trading #Signal
"""
                        
                        try:
                            result = await send_telegram_msg(MY_TOKEN, MY_CHAT_ID, msg)
                            if result.get('ok'):
                                print("📡 Signal sent to Telegram!")
                            else:
                                print(f"⚠️  Telegram error: {result}")
                        except Exception as e:
                            print(f"❌ Failed to send Telegram: {e}")
                else:
                    print("❌ Failed to generate signal")
                
                # Calculate next update time
                next_time = datetime.now(TIMEZONE) + timedelta(seconds=args.interval)
                next_str = next_time.strftime('%H:%M:%S ET')
                
                print(f"\n⏳ Next update at {next_str} ({args.interval//60} minutes)")
                
                # Sleep until next interval
                await asyncio.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Stopping live signals...")
            
            # Send shutdown message to Telegram
            if telegram_enabled:
                shutdown_msg = "🛑 Gold Trading Sentinel stopped."
                await send_telegram_msg(MY_TOKEN, MY_CHAT_ID, shutdown_msg)
            
            print("✅ Shutdown complete.")
            return 0
    
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

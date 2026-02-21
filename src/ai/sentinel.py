"""
NewsSentinel - AI-powered news sentiment analysis for trading signals.

This module acts as the final "gatekeeper" for entry candidates, scanning
recent news (48-72 hours) for fundamental red flags that technicals haven't
priced in yet (e.g., secondary offerings, lawsuits, FDA rejections).
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import yfinance as yf

# Optional imports - gracefully handle if not installed
try:
    from gnews import GNews
    GNEWS_AVAILABLE = True
except ImportError:
    GNEWS_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class SentinelResult:
    """Result from AI sentiment analysis."""
    ticker: str
    risk_score: int  # 0 (safe) to 10 (do not trade)
    sentiment: str   # POSITIVE, NEUTRAL, NEGATIVE
    reason: str      # Summary of key risk driver
    headlines_count: int
    

class NewsSentinel:
    """
    AI-powered news sentiment analyzer for trading signals.
    
    Uses yfinance and GNews for news fetching, and Gemini for analysis.
    Falls back gracefully if API is unavailable.
    """
    
    def __init__(self, api_key: Optional[str] = None, risk_threshold: int = 7):
        """
        Initialize the NewsSentinel.
        
        Args:
            api_key: Google Gemini API key (or reads from GEMINI_API_KEY env var)
            risk_threshold: Risk score >= this value triggers rejection (default: 7)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.risk_threshold = risk_threshold
        self.client = None
        self.google_news = None
        
        # Initialize Gemini client if available
        if GEMINI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"⚠️ Sentinel: Failed to initialize Gemini: {e}")
                self.client = None
        
        # Initialize GNews if available
        if GNEWS_AVAILABLE:
            try:
                self.google_news = GNews(language='en', period='3d', max_results=5)
            except Exception:
                self.google_news = None
    
    def is_available(self) -> bool:
        """Check if the sentinel is properly configured and available."""
        return self.client is not None
    
    def fetch_news(self, ticker: str) -> List[str]:
        """
        Fetches news headlines from yfinance (primary) and GNews (fallback).
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            List of headline strings: "Title (Source)"
        """
        headlines = []
        
        # 1. Try yfinance first (specific to stock)
        try:
            yf_ticker = yf.Ticker(ticker)
            news_items = yf_ticker.news
            if news_items:
                for n in news_items[:5]:
                    title = n.get('title', '')
                    publisher = n.get('publisher', 'Unknown')
                    if title:
                        headlines.append(f"{title} ({publisher})")
        except Exception:
            pass

        # 2. Augment with Google News (broad coverage) if available
        if self.google_news:
            try:
                g_news = self.google_news.get_news(ticker)
                for n in g_news[:5]:
                    title = n.get('title', '')
                    publisher = n.get('publisher', {})
                    if isinstance(publisher, dict):
                        publisher = publisher.get('title', 'Unknown')
                    if title:
                        headlines.append(f"{title} ({publisher})")
            except Exception:
                pass
            
        # Deduplicate while preserving order
        seen = set()
        unique_headlines = []
        for h in headlines:
            if h not in seen:
                seen.add(h)
                unique_headlines.append(h)
        
        return unique_headlines[:10]  # Cap at 10 headlines

    def analyze_ticker(self, ticker: str) -> SentinelResult:
        """
        Analyzes the ticker's news and returns a risk assessment.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            SentinelResult with risk_score, sentiment, and reason
        """
        # If client not available, return neutral result
        if not self.client:
            return SentinelResult(
                ticker=ticker,
                risk_score=0,
                sentiment="NEUTRAL",
                reason="AI Sentinel not available (missing API key)",
                headlines_count=0
            )
        
        headlines = self.fetch_news(ticker)
        
        if not headlines:
            return SentinelResult(
                ticker=ticker,
                risk_score=0,
                sentiment="NEUTRAL",
                reason="No recent news found",
                headlines_count=0
            )

        prompt = f"""You are a financial risk analyst. Analyze the following news headlines for stock ticker {ticker} from the last 72 hours:

{json.dumps(headlines, indent=2)}

Task:
1. Identify any IMMEDIATE fundamental risks:
   - Secondary offerings, share dilution, stock sales by insiders
   - Lawsuits, regulatory actions, SEC investigations
   - Earnings miss, revenue miss, guidance cuts
   - FDA rejection, clinical trial failure (for biotech)
   - Executive scandal, fraud allegations
   - Bankruptcy, debt issues, credit downgrades
   - Major contract loss, key customer departure

2. Assign a Risk Score from 0 (Safe) to 10 (Do Not Trade):
   - 0-2: No significant risks, normal market noise
   - 3-4: Minor concerns, worth monitoring
   - 5-6: Moderate risk, proceed with caution
   - 7-8: High risk, consider avoiding
   - 9-10: Critical risk, do not trade

3. Determine overall Sentiment: POSITIVE, NEUTRAL, or NEGATIVE

Important Constraints:
- If news is just generic market noise or routine analyst price target adjustments, Risk Score should be LOW (0-2)
- If there is a secondary offering, dilution event, or insider selling, Risk Score MUST be >= 8
- If there is an active lawsuit, SEC investigation, or fraud allegation, Risk Score MUST be >= 7
- Positive earnings beats or upgrades do NOT reduce risk from negative events

Output strictly in JSON format:
{{
    "risk_score": <integer 0-10>,
    "sentiment": "<POSITIVE|NEUTRAL|NEGATIVE>",
    "reason": "<one sentence summary of the key risk or lack thereof>"
}}"""

        # Retry with exponential backoff for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json'
                    )
                )
                result = json.loads(response.text)
                
                return SentinelResult(
                    ticker=ticker,
                    risk_score=int(result.get("risk_score", 0)),
                    sentiment=result.get("sentiment", "NEUTRAL"),
                    reason=result.get("reason", "Analysis completed"),
                    headlines_count=len(headlines)
                )
            except json.JSONDecodeError as e:
                print(f"⚠️ Sentinel: JSON parse error for {ticker}: {e}")
                return SentinelResult(
                    ticker=ticker,
                    risk_score=0,
                    sentiment="NEUTRAL",
                    reason="Error parsing AI response",
                    headlines_count=len(headlines)
                )
            except Exception as e:
                error_str = str(e)
                # Check for rate limiting (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 10  # 10s, 20s, 40s
                        print(f"⚠️ Sentinel: Rate limited for {ticker}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"⚠️ Sentinel: Rate limit exceeded for {ticker} after {max_retries} attempts")
                        return SentinelResult(
                            ticker=ticker,
                            risk_score=0,
                            sentiment="NEUTRAL",
                            reason="Rate limit exceeded - skipped",
                            headlines_count=len(headlines)
                        )
                else:
                    print(f"⚠️ Sentinel: Error analyzing {ticker}: {error_str[:100]}")
                    return SentinelResult(
                        ticker=ticker,
                        risk_score=0,
                        sentiment="NEUTRAL",
                        reason=f"Error in AI analysis: {error_str[:50]}",
                        headlines_count=len(headlines)
                    )
        
        # Should not reach here, but just in case
        return SentinelResult(
            ticker=ticker,
            risk_score=0,
            sentiment="NEUTRAL",
            reason="Unknown error",
            headlines_count=len(headlines)
        )
    
    def analyze_batch(
        self, 
        tickers: List[str], 
        delay_seconds: float = 0.5
    ) -> Dict[str, SentinelResult]:
        """
        Analyze multiple tickers with rate limiting.
        
        Args:
            tickers: List of ticker symbols
            delay_seconds: Delay between API calls to avoid rate limits
            
        Returns:
            Dict mapping ticker to SentinelResult
        """
        results = {}
        
        for i, ticker in enumerate(tickers):
            results[ticker] = self.analyze_ticker(ticker)
            
            # Rate limiting (skip delay on last item)
            if i < len(tickers) - 1 and delay_seconds > 0:
                time.sleep(delay_seconds)
        
        return results
    
    def filter_candidates(
        self, 
        tickers: List[str],
        delay_seconds: float = 0.5
    ) -> tuple[List[str], List[tuple[str, SentinelResult]], Dict[str, SentinelResult]]:
        """
        Filter candidates based on AI risk analysis.
        
        Args:
            tickers: List of candidate ticker symbols
            delay_seconds: Delay between API calls
            
        Returns:
            Tuple of (accepted_tickers, rejected_list, all_results)
            where rejected_list is [(ticker, SentinelResult), ...]
            and all_results is {ticker: SentinelResult, ...} for all analyzed tickers
        """
        if not self.is_available():
            print("⚠️ Sentinel: Not available, passing all candidates through")
            return tickers, [], {}
        
        print(f"🤖 Sentinel: Scanning {len(tickers)} candidates for news risks...")
        
        accepted = []
        rejected = []
        all_results = {}
        
        for i, ticker in enumerate(tickers):
            result = self.analyze_ticker(ticker)
            all_results[ticker] = result
            
            if result.risk_score >= self.risk_threshold:
                print(f"  ❌ REJECTED {ticker}: {result.reason} (Risk: {result.risk_score})")
                rejected.append((ticker, result))
            else:
                status = "✅" if result.risk_score < 3 else "⚠️"
                print(f"  {status} ACCEPTED {ticker}: {result.reason} (Risk: {result.risk_score})")
                accepted.append(ticker)
            
            # Rate limiting
            if i < len(tickers) - 1 and delay_seconds > 0:
                time.sleep(delay_seconds)
        
        print(f"🤖 Sentinel: {len(accepted)} accepted, {len(rejected)} rejected")
        
        return accepted, rejected, all_results


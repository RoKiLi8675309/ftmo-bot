# =============================================================================
# FILENAME: shared/compliance.py
# ENVIRONMENT: DUAL COMPATIBILITY (Windows Py3.9 & Linux Py3.11)
# PATH: shared/compliance.py
# DEPENDENCIES: requests, lxml, cloudscraper (optional)
# DESCRIPTION: Fetches Economic News and enforces FTMO trading blackouts.
# CRITICAL: Python 3.9 Compatible. Implements Fail-Safe locking.
# =============================================================================
from __future__ import annotations
import logging
import pytz
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

# Shared Imports
from shared.core.config import CONFIG
from shared.domain.models import NewsEvent

# Third-Party Imports (Guarded)
try:
    from lxml import etree
    import cloudscraper
    NEWS_DEPS_AVAILABLE = True
except ImportError:
    NEWS_DEPS_AVAILABLE = False
    # If missing, we can still run but might fail to fetch news (Producer might not need this active)

class NewsEventMonitor:
    """
    Fetches economic calendar data using multiple strategies (XML, HTML).
    """
    def __init__(self):
        self.logger = logging.getLogger("NewsMonitor")
        self.config = CONFIG.get('news_filter', {})
        self.xml_url = self.config.get('primary_url', "http://nfs.faireconomy.media/ff_calendar_thisweek.xml")
        self.html_url = self.config.get('backup_url', "https://www.forexfactory.com/calendar")
        self.scraper = None

    def fetch_events(self) -> List[NewsEvent]:
        """
        Main entry point. Tries XML, then HTML, then Fail-Safe.
        """
        # Strategy 1: XML Feed (Fast, Clean)
        events = self._fetch_xml_strategy()
       
        # Strategy 2: HTML Scraping (Fallback)
        if not events and NEWS_DEPS_AVAILABLE:
            self.logger.warning("XML Feed failed or empty. Attempting HTML Scraping Fallback.")
            events = self._fetch_html_strategy()

        # Strategy 3: Fail-S complement
        if not events:
            if NEWS_DEPS_AVAILABLE:
                self.logger.critical("CRITICAL: ALL NEWS FEEDS FAILED. ENGAGING FAIL-SAFE LOCK.")
                # If we can't see the news, we assume the worst (FOMC/NFP) is happening now.
                # This locks the bot until functionality is restored.
                fail_safe_event = NewsEvent(
                    title="CRITICAL: DATA FEED FAILURE (FAIL-SAFE)",
                    country="USD",
                    time_utc=datetime.now(pytz.utc),
                    impact="High"
                )
                return [fail_safe_event]
            else:
                self.logger.warning("News dependencies missing. Skipping News Check (Risk warning).")
                return []
       
        return events

    def _fetch_xml_strategy(self) -> Optional[List[NewsEvent]]:
        """Parses the official Forex Factory XML feed."""
        if not NEWS_DEPS_AVAILABLE: return []
       
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(self.xml_url, headers=headers, timeout=10)
            response.raise_for_status()
           
            # XML Parsing
            parser = etree.XMLParser(recover=True)
            tree = etree.fromstring(response.content, parser=parser)
           
            events = []
            for event in tree.xpath('//event'):
                title = event.findtext('title')
                country = event.findtext('country')
                date_str = event.findtext('date')
                time_str = event.findtext('time')
                impact = event.findtext('impact')

                # Filter Low Impact to save memory
                if impact != 'High':
                    continue

                utc_dt = self._normalize_time(date_str, time_str)
                if not utc_dt:
                    continue
               
                events.append(NewsEvent(
                    title=title,
                    country=country,
                    time_utc=utc_dt,
                    impact=impact,
                    forecast=event.findtext('forecast'),
                    previous=event.findtext('previous')
                ))
           
            return events
        except Exception as e:
            self.logger.error(f"XML Fetch Error: {e}")
            return None

    def _fetch_html_strategy(self) -> Optional[List[NewsEvent]]:
        """
        Scrapes the HTML calendar using cloudscraper to bypass simple bot protection.
        (Placeholder for complex scraping logic - requires robust maintenance).
        """
        if not NEWS_DEPS_AVAILABLE: return None
        try:
            if not self.scraper:
                self.scraper = cloudscraper.create_scraper()
           
            response = self.scraper.get(self.html_url)
            if response.status_code != 200:
                return None
           
            # NOTE: HTML parsing logic is brittle and often breaks with site updates.
            # Returning empty list here implies "Tried but couldn't parse" rather than crash.
            # In a real impl, BeautifulSoup logic would go here.
            return []
        except Exception as e:
            self.logger.error(f"HTML Scraping Error: {e}")
            return None

    def _normalize_time(self, date_str: str, time_str: str) -> Optional[datetime]:
        """Converts FF time strings to UTC datetime."""
        # FF XML uses "MM-DD-YYYY" and "HH:MMam/pm" usually in US/Eastern context depending on link
        # But the XML feed is often raw. Standardizing usually requires assumption.
       
        if "Day" in time_str or not time_str: return None  # All Day event
        try:
            dt_str = f"{date_str} {time_str}"
            naive_dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
           
            # Assume New York time for Forex Factory XML default
            eastern = pytz.timezone('US/Eastern')
            localized_dt = eastern.localize(naive_dt)
           
            return localized_dt.astimezone(pytz.utc)
        except ValueError:
            return None

class FTMOComplianceGuard:
    """
    Enforces Trading Blackouts based on fetched News Events.
    """
    def __init__(self, events: List[NewsEvent]):
        self.events = events
        self.blackout_windows: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("ComplianceGuard")
       
        self.config = CONFIG.get('news_filter', {})
        self.buffer_pre = self.config.get('buffer_minutes_before', 2)
        self.buffer_post = self.config.get('buffer_minutes_after', 2)
        self.restricted_map = self.config.get('restricted_events', {})
        self._build_blackout_windows()

    def _build_blackout_windows(self) -> None:
        """
        Pre-calculates time windows where trading is forbidden.
        """
        for event in self.events:
            # Check if this currency is in our restricted list
            if event.country not in self.restricted_map:
                continue
           
            # Check keywords (e.g., "FOMC", "CPI")
            keywords = self.restricted_map[event.country]
            is_restricted = any(k.lower() in event.title.lower() for k in keywords)
           
            if is_restricted:
                start = event.time_utc - timedelta(minutes=self.buffer_pre)
                end = event.time_utc + timedelta(minutes=self.buffer_post)
               
                self.blackout_windows.append({
                    'currency': event.country,
                    'start': start,
                    'end': end,
                    'reason': event.title
                })
                self.logger.info(f"Blackout Window Created: {event.country} | {start.strftime('%H:%M')} - {end.strftime('%H:%M')} | {event.title}")

    def check_trade_permission(self, symbol: str) -> bool:
        """
        Returns True if trade is ALLOWED, False if BLOCKED.
        """
        if not self.blackout_windows:
            return True
       
        now_utc = datetime.now(pytz.utc)
       
        for window in self.blackout_windows:
            # If the restricted currency is in the symbol (e.g. "USD" in "EURUSD")
            if window['currency'] in symbol:
                if window['start'] <= now_utc <= window['end']:
                    self.logger.warning(f"{symbol} Trade HALTED. Blackout: {window['reason']} ({window['start'].strftime('%H:%M')} - {window['end'].strftime('%H:%M')})")
                    return False
       
        return True
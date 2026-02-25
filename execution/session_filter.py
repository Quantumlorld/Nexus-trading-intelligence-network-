"""
Nexus Trading System - Session Filter
Manages trading session filters and time-based restrictions
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, time
import logging
from dataclasses import dataclass
from pathlib import Path
import yaml
import json
from enum import Enum

from core.logger import get_logger


class SessionType(Enum):
    """Trading session types"""
    LONDON = "london"
    NEW_YORK = "new_york"
    ASIAN = "asian"
    OVERLAP = "overlap"
    OFF_SESSION = "off_session"


@dataclass
class TradingSession:
    """Trading session definition"""
    name: str
    start_time: time
    end_time: time
    timezone: str
    days_active: List[int]  # 0=Monday, 6=Sunday
    description: str
    priority: int = 1  # Higher priority takes precedence


@dataclass
class SessionFilterResult:
    """Result of session filter check"""
    allowed: bool
    current_session: Optional[SessionType] = None
    reason: str = ""
    next_allowed_time: Optional[datetime] = None
    current_time: Optional[datetime] = None
    active_sessions: List[SessionType] = None


class SessionFilter:
    """
    Advanced session filtering system for trading time management
    Enforces session-specific trading rules and time-based restrictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger_instance = get_logger()
        self.logger = logger_instance.system_logger
        
        # Session definitions
        self.sessions: Dict[str, TradingSession] = {}
        self.asset_sessions: Dict[str, List[str]] = {}  # asset -> session names
        
        # Session priority
        self.session_priorities = {
            SessionType.LONDON: 3,
            SessionType.NEW_YORK: 3,
            SessionType.OVERLAP: 4,
            SessionType.ASIAN: 2,
            SessionType.OFF_SESSION: 1
        }
        
        # Enable/disable flag
        self.is_enabled = self.config.get('session_filters', {}).get('enabled', True)
        
        # Load session configurations
        self._load_session_configs()
        
        # Cache for performance
        self._filter_cache: Dict[str, SessionFilterResult] = {}
        self._cache_expiry = timedelta(minutes=5)
        
        self.logger.info("Session filter initialized")
    
    def _load_session_configs(self):
        """Load session configurations from config"""
        
        # Define standard trading sessions
        self.sessions = {
            'london_morning': TradingSession(
                name='london_morning',
                start_time=time(8, 0),  # 08:00 UTC
                end_time=time(13, 0),    # 13:00 UTC
                timezone='UTC',
                days_active=[0, 1, 2, 3, 4],  # Monday-Friday
                description='London morning session',
                priority=3
            ),
            'london_afternoon': TradingSession(
                name='london_afternoon',
                start_time=time(13, 0),  # 13:00 UTC
                end_time=time(17, 0),    # 17:00 UTC
                timezone='UTC',
                days_active=[0, 1, 2, 3, 4],  # Monday-Friday
                description='London afternoon session',
                priority=3
            ),
            'new_york': TradingSession(
                name='new_york',
                start_time=time(13, 0),  # 13:00 UTC
                end_time=time(22, 0),    # 22:00 UTC
                timezone='UTC',
                days_active=[0, 1, 2, 3, 4],  # Monday-Friday
                description='New York session',
                priority=3
            ),
            'london_ny_overlap': TradingSession(
                name='london_ny_overlap',
                start_time=time(13, 0),  # 13:00 UTC
                end_time=time(17, 0),    # 17:00 UTC
                timezone='UTC',
                days_active=[0, 1, 2, 3, 4],  # Monday-Friday
                description='London/New York overlap',
                priority=4
            ),
            'asian': TradingSession(
                name='asian',
                start_time=time(23, 0),  # 23:00 UTC
                end_time=time(8, 0),     # 08:00 UTC next day
                timezone='UTC',
                days_active=[0, 1, 2, 3, 4, 5, 6],  # All days
                description='Asian session',
                priority=2
            ),
            'weekend': TradingSession(
                name='weekend',
                start_time=time(0, 0),    # 00:00 UTC
                end_time=time(23, 59),   # 23:59 UTC
                timezone='UTC',
                days_active=[5, 6],      # Saturday, Sunday
                description='Weekend session',
                priority=1
            )
        }
        
        # Asset-specific session mappings
        self.asset_sessions = {
            'XAUUSD': ['london_morning', 'london_afternoon', 'new_york', 'london_ny_overlap'],
            'EURUSD': ['london_morning', 'london_afternoon', 'new_york', 'london_ny_overlap'],
            'USDX': ['london_morning', 'london_afternoon', 'new_york', 'london_ny_overlap'],
            'BTCUSD': ['asian', 'london_morning', 'london_afternoon', 'new_york'],  # 24/7 with all sessions
        }
        
        # Load custom sessions from config if available
        self._load_custom_sessions()
    
    def _load_custom_sessions(self):
        """Load custom session configurations"""
        
        custom_sessions = self.config.get('custom_sessions', {})
        
        for session_name, session_config in custom_sessions.items():
            try:
                start_time = time.fromisoformat(session_config['start_time'])
                end_time = time.fromisoformat(session_config['end_time'])
                days_active = session_config.get('days_active', [0, 1, 2, 3, 4])
                
                session = TradingSession(
                    name=session_name,
                    start_time=start_time,
                    end_time=end_time,
                    timezone=session_config.get('timezone', 'UTC'),
                    days_active=days_active,
                    description=session_config.get('description', ''),
                    priority=session_config.get('priority', 1)
                )
                
                self.sessions[session_name] = session
                
                # Add to asset mappings if specified
                if 'assets' in session_config:
                    for asset in session_config['assets']:
                        if asset not in self.asset_sessions:
                            self.asset_sessions[asset] = []
                        self.asset_sessions[asset].append(session_name)
                
                self.logger.info(f"Loaded custom session: {session_name}")
                
            except Exception as e:
                self.logger.error(f"Error loading custom session {session_name}: {e}")
    
    def is_trading_allowed(self, symbol: str, current_time: Optional[datetime] = None) -> SessionFilterResult:
        """
        Check if trading is allowed for a symbol at the current time
        
        Args:
            symbol: Trading symbol
            current_time: Current time (defaults to now)
            
        Returns:
            SessionFilterResult with detailed information
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        # Check cache
        cache_key = f"{symbol}_{current_time.strftime('%Y%m%d%H%M')}"
        if cache_key in self._filter_cache:
            cached_result = self._filter_cache[cache_key]
            # Check if cache is still valid
            if datetime.now() - cached_result.current_time < self._cache_expiry:
                return cached_result
        
        # Get active sessions for the symbol
        active_sessions = self._get_active_sessions_for_symbol(symbol, current_time)
        
        if not active_sessions:
            result = SessionFilterResult(
                allowed=False,
                current_session=SessionType.OFF_SESSION,
                reason="No active trading sessions",
                current_time=current_time,
                active_sessions=[]
            )
        else:
            # Determine highest priority session
            highest_priority_session = max(active_sessions, key=lambda s: s.priority)
            session_type = self._get_session_type(highest_priority_session.name)
            
            # Check for avoid periods
            avoid_result = self._check_avoid_periods(symbol, current_time)
            
            if avoid_result['avoid']:
                result = SessionFilterResult(
                    allowed=False,
                    current_session=session_type,
                    reason=avoid_result['reason'],
                    next_allowed_time=avoid_result['next_allowed_time'],
                    current_time=current_time,
                    active_sessions=[self._get_session_type(s.name) for s in active_sessions]
                )
            else:
                result = SessionFilterResult(
                    allowed=True,
                    current_session=session_type,
                    reason=f"Trading allowed in {highest_priority_session.name}",
                    current_time=current_time,
                    active_sessions=[self._get_session_type(s.name) for s in active_sessions]
                )
        
        # Cache the result
        self._filter_cache[cache_key] = result
        
        return result
    
    def _get_active_sessions_for_symbol(self, symbol: str, current_time: datetime) -> List[TradingSession]:
        """Get active trading sessions for a symbol"""
        
        # Get sessions for this asset
        asset_session_names = self.asset_sessions.get(symbol, [])
        
        if not asset_session_names:
            # Default to all sessions if no specific mapping
            asset_session_names = list(self.sessions.keys())
        
        active_sessions = []
        current_weekday = current_time.weekday()
        current_time_only = current_time.time()
        
        for session_name in asset_session_names:
            if session_name not in self.sessions:
                continue
            
            session = self.sessions[session_name]
            
            # Check if session is active today
            if current_weekday not in session.days_active:
                continue
            
            # Check if current time is within session hours
            if session.start_time <= session.end_time:
                # Same day session
                if session.start_time <= current_time_only <= session.end_time:
                    active_sessions.append(session)
            else:
                # Overnight session (e.g., Asian session)
                if current_time_only >= session.start_time or current_time_only <= session.end_time:
                    active_sessions.append(session)
        
        return active_sessions
    
    def _get_session_type(self, session_name: str) -> SessionType:
        """Convert session name to SessionType enum"""
        
        session_mapping = {
            'london_morning': SessionType.LONDON,
            'london_afternoon': SessionType.LONDON,
            'new_york': SessionType.NEW_YORK,
            'london_ny_overlap': SessionType.OVERLAP,
            'asian': SessionType.ASIAN,
            'weekend': SessionType.OFF_SESSION
        }
        
        return session_mapping.get(session_name, SessionType.OFF_SESSION)
    
    def _check_avoid_periods(self, symbol: str, current_time: datetime) -> Dict[str, Any]:
        """Check if current time falls within avoid periods"""
        
        # Get asset-specific avoid periods
        asset_config = self._get_asset_config(symbol)
        
        if not asset_config or 'avoid_periods' not in asset_config:
            return {'avoid': False}
        
        current_time_only = current_time.time()
        current_weekday = current_time.weekday()
        
        for avoid_period in asset_config['avoid_periods']:
            # Parse avoid period times
            try:
                start_time = time.fromisoformat(avoid_period['start'])
                end_time = time.fromisoformat(avoid_period['end'])
                reason = avoid_period.get('reason', 'Avoid period')
                
                # Check if current time is within avoid period
                if start_time <= end_time:
                    # Same day avoid period
                    if start_time <= current_time_only <= end_time:
                        return {
                            'avoid': True,
                            'reason': reason,
                            'next_allowed_time': self._calculate_next_allowed_time(current_time, end_time)
                        }
                else:
                    # Overnight avoid period
                    if current_time_only >= start_time or current_time_only <= end_time:
                        return {
                            'avoid': True,
                            'reason': reason,
                            'next_allowed_time': self._calculate_next_allowed_time(current_time, end_time)
                        }
            
            except Exception as e:
                self.logger.error(f"Error parsing avoid period: {e}")
                continue
        
        return {'avoid': False}
    
    def _calculate_next_allowed_time(self, current_time: datetime, end_time: time) -> datetime:
        """Calculate next allowed trading time"""
        
        # Create datetime for end of avoid period today
        today_end = current_time.replace(hour=end_time.hour, minute=end_time.minute, 
                                         second=end_time.second, microsecond=0)
        
        # If end time is in the future, return it
        if today_end > current_time:
            return today_end
        
        # Otherwise, return same time tomorrow (assuming trading is allowed then)
        tomorrow = current_time + timedelta(days=1)
        return tomorrow.replace(hour=end_time.hour, minute=end_time.minute,
                              second=end_time.second, microsecond=0)
    
    def _get_asset_config(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get asset configuration"""
        
        # This would load from asset config file in production
        asset_configs = {
            'XAUUSD': {
                'avoid_periods': [
                    {'start': '22:00', 'end': '01:00', 'reason': 'Low liquidity'},
                    {'start': '17:00', 'end': '18:00', 'reason': 'Session overlap end'}
                ]
            },
            'EURUSD': {
                'avoid_periods': [
                    {'start': '22:00', 'end': '01:00', 'reason': 'Low liquidity'}
                ]
            },
            'USDX': {
                'avoid_periods': [
                    {'start': '22:00', 'end': '01:00', 'reason': 'Low liquidity'}
                ]
            },
            'BTCUSD': {
                'avoid_periods': []  # BTC trades 24/7
            }
        }
        
        return asset_configs.get(symbol)
    
    def get_session_schedule(self, symbol: str, date: datetime) -> List[Dict[str, Any]]:
        """Get session schedule for a specific date"""
        
        schedule = []
        asset_session_names = self.asset_sessions.get(symbol, list(self.sessions.keys()))
        
        for session_name in asset_session_names:
            if session_name not in self.sessions:
                continue
            
            session = self.sessions[session_name]
            
            if date.weekday() in session.days_active:
                # Create datetime objects for session start and end
                session_start = date.replace(hour=session.start_time.hour, 
                                           minute=session.start_time.minute,
                                           second=session.start_time.second,
                                           microsecond=0)
                
                if session.start_time <= session.end_time:
                    # Same day session
                    session_end = date.replace(hour=session.end_time.hour,
                                             minute=session.end_time.minute,
                                             second=session.end_time.second,
                                             microsecond=0)
                else:
                    # Overnight session
                    session_end = (date + timedelta(days=1)).replace(hour=session.end_time.hour,
                                                                  minute=session.end_time.minute,
                                                                  second=session.end_time.second,
                                                                  microsecond=0)
                
                schedule.append({
                    'session_name': session.name,
                    'session_type': self._get_session_type(session.name).value,
                    'start_time': session_start,
                    'end_time': session_end,
                    'description': session.description,
                    'priority': session.priority
                })
        
        # Sort by start time
        schedule.sort(key=lambda x: x['start_time'])
        
        return schedule
    
    def get_next_trading_session(self, symbol: str, current_time: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Get the next trading session for a symbol"""
        
        if current_time is None:
            current_time = datetime.now()
        
        # Get today's schedule
        today_schedule = self.get_session_schedule(symbol, current_time.date())
        
        # Find next session today
        for session in today_schedule:
            if session['start_time'] > current_time:
                return session
        
        # If no session today, get first session tomorrow
        tomorrow = current_time.date() + timedelta(days=1)
        tomorrow_schedule = self.get_session_schedule(symbol, tomorrow)
        
        if tomorrow_schedule:
            return tomorrow_schedule[0]
        
        return None
    
    def is_session_overlap(self, current_time: Optional[datetime] = None) -> bool:
        """Check if current time is in a session overlap period"""
        
        if current_time is None:
            current_time = datetime.now()
        
        # Check for London/NY overlap (13:00-17:00 UTC)
        overlap_start = time(13, 0)
        overlap_end = time(17, 0)
        current_time_only = current_time.time()
        
        return overlap_start <= current_time_only <= overlap_end
    
    def get_session_statistics(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get session statistics for analysis"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        session_stats = {
            'total_days': days,
            'sessions': {},
            'overlap_percentage': 0.0,
            'most_active_session': None,
            'least_active_session': None
        }
        
        session_counts = {session_type.value: 0 for session_type in SessionType}
        overlap_minutes = 0
        total_minutes = days * 24 * 60  # Total minutes in period
        
        current_date = start_date
        while current_date <= end_date:
            schedule = self.get_session_schedule(symbol, current_date)
            
            for session in schedule:
                session_type = session['session_type']
                session_counts[session_type] += 1
                
                # Calculate session duration in minutes
                duration = (session['end_time'] - session['start_time']).total_seconds() / 60
                
                if session_type == 'overlap':
                    overlap_minutes += duration
            
            current_date += timedelta(days=1)
        
        # Calculate percentages
        for session_type, count in session_counts.items():
            session_stats['sessions'][session_type] = {
                'days': count,
                'percentage': (count / days) * 100
            }
        
        session_stats['overlap_percentage'] = (overlap_minutes / total_minutes) * 100
        
        # Find most/least active sessions
        if session_counts:
            most_active = max(session_counts, key=session_counts.get)
            least_active = min(session_counts, key=session_counts.get)
            
            session_stats['most_active_session'] = most_active
            session_stats['least_active_session'] = least_active
        
        return session_stats
    
    def add_custom_session(self, session_name: str, start_time: str, end_time: str,
                          days_active: List[int], description: str = "",
                          priority: int = 1, assets: List[str] = None):
        """Add a custom trading session"""
        
        try:
            start = time.fromisoformat(start_time)
            end = time.fromisoformat(end_time)
            
            session = TradingSession(
                name=session_name,
                start_time=start,
                end_time=end,
                timezone='UTC',
                days_active=days_active,
                description=description,
                priority=priority
            )
            
            self.sessions[session_name] = session
            
            # Add to asset mappings
            if assets:
                for asset in assets:
                    if asset not in self.asset_sessions:
                        self.asset_sessions[asset] = []
                    self.asset_sessions[asset].append(session_name)
            
            self.logger.info(f"Added custom session: {session_name}")
            
        except Exception as e:
            self.logger.error(f"Error adding custom session {session_name}: {e}")
    
    def remove_session(self, session_name: str):
        """Remove a trading session"""
        
        if session_name in self.sessions:
            del self.sessions[session_name]
            
            # Remove from asset mappings
            for asset, sessions in self.asset_sessions.items():
                if session_name in sessions:
                    sessions.remove(session_name)
            
            self.logger.info(f"Removed session: {session_name}")
    
    def export_session_config(self, filepath: str):
        """Export session configuration to file"""
        
        config_data = {
            'sessions': {},
            'asset_sessions': self.asset_sessions
        }
        
        for session_name, session in self.sessions.items():
            config_data['sessions'][session_name] = {
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat(),
                'timezone': session.timezone,
                'days_active': session.days_active,
                'description': session.description,
                'priority': session.priority
            }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Session configuration exported to {filepath}")
    
    def clear_cache(self):
        """Clear the filter cache"""
        self._filter_cache.clear()
        self.logger.info("Session filter cache cleared")
    
    def get_filter_status(self) -> Dict[str, Any]:
        """Get current filter status"""
        
        return {
            'total_sessions': len(self.sessions),
            'asset_mappings': len(self.asset_sessions),
            'cache_size': len(self._filter_cache),
            'session_priorities': {k.value: v for k, v in self.session_priorities.items()},
            'enabled': self.config.get('session_filters', {}).get('enabled', True)
        }

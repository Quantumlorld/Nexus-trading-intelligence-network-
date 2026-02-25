"""
Nexus Trading System - Trade Manager
Manages position tracking, trade lifecycle, and trade history
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class Position:
    """Data class representing a trading position"""
    symbol: str
    direction: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    sl_price: float
    tp_price: float
    strategy: str
    timestamp: datetime
    ticket_id: Optional[int] = None
    current_sl: Optional[float] = None
    current_tp: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: str = "open"  # 'open', 'closed', 'cancelled'
    close_timestamp: Optional[datetime] = None
    close_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)


class TradeManager:
    """Manages all trading positions and trade history"""
    
    def __init__(self, risk_config: Dict[str, Any]):
        self.risk_config = risk_config
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.trade_history: List[Position] = []
        self.daily_trades: Dict[str, int] = {}  # timeframe -> trade count
        self.logger = logging.getLogger(__name__)
        
        # Track daily statistics
        self.daily_stats = {
            'trades_opened': 0,
            'trades_closed': 0,
            'total_pnl': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'consecutive_losses': 0,
            'max_drawdown': 0.0,
            'last_reset': datetime.now().date()
        }
        
        # Load existing trade history if available
        self._load_trade_history()
    
    def _load_trade_history(self):
        """Load trade history from file"""
        try:
            history_file = Path("data/processed/trade_history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # Convert dict data back to Position objects
                    for pos_data in data:
                        pos_data['timestamp'] = datetime.fromisoformat(pos_data['timestamp'])
                        if pos_data['close_timestamp']:
                            pos_data['close_timestamp'] = datetime.fromisoformat(pos_data['close_timestamp'])
                        self.trade_history.append(Position(**pos_data))
                self.logger.info(f"Loaded {len(self.trade_history)} historical trades")
        except Exception as e:
            self.logger.error(f"Failed to load trade history: {e}")
    
    def _save_trade_history(self):
        """Save trade history to file"""
        try:
            history_file = Path("data/processed/trade_history.json")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert Position objects to dict for JSON serialization
            data = []
            for pos in self.trade_history:
                pos_dict = {
                    'symbol': pos.symbol,
                    'direction': pos.direction,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'sl_price': pos.sl_price,
                    'tp_price': pos.tp_price,
                    'strategy': pos.strategy,
                    'timestamp': pos.timestamp.isoformat(),
                    'ticket_id': pos.ticket_id,
                    'current_sl': pos.current_sl,
                    'current_tp': pos.current_tp,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl,
                    'status': pos.status,
                    'close_timestamp': pos.close_timestamp.isoformat() if pos.close_timestamp else None,
                    'close_reason': pos.close_reason,
                    'notes': pos.notes
                }
                data.append(pos_dict)
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save trade history: {e}")
    
    def can_open_position(self, symbol: str, timeframe: str = "1H") -> tuple[bool, str]:
        """Check if a new position can be opened for the given symbol"""
        
        # Check if we already have an active position for this asset
        if symbol in self.positions and self.positions[symbol].status == "open":
            return False, f"Already have open position for {symbol}"
        
        # Check daily trade limits
        max_trades_per_day = self.risk_config['daily_limits']['max_daily_trades']
        if self.daily_stats['trades_opened'] >= max_trades_per_day:
            return False, f"Max daily trades ({max_trades_per_day}) reached"
        
        # Check timeframe-specific limits
        timeframe_limits = {
            "9H": 2,
            "6H": 2, 
            "3H": 1
        }
        
        if timeframe in timeframe_limits:
            daily_tf_trades = self.daily_trades.get(timeframe, 0)
            max_tf_trades = timeframe_limits[timeframe]
            if daily_tf_trades >= max_tf_trades:
                return False, f"Max {timeframe} trades ({max_tf_trades}) reached"
        
        # Check daily loss limit
        max_daily_loss = self.risk_config['daily_limits']['max_daily_loss']
        if self.daily_stats['total_pnl'] <= -max_daily_loss:
            return False, f"Max daily loss (${max_daily_loss}) reached"
        
        # Check kill switch conditions
        if self.daily_stats['consecutive_losses'] >= self.risk_config['kill_switch']['max_consecutive_losses']:
            return False, "Kill switch activated - too many consecutive losses"
        
        # Check max drawdown
        if self.daily_stats['max_drawdown'] >= self.risk_config['kill_switch']['max_drawdown_percent']:
            return False, "Max drawdown reached - kill switch activated"
        
        return True, "Position can be opened"
    
    def open_position(self, symbol: str, direction: str, size: float, 
                     entry_price: float, sl_price: float, tp_price: float,
                     strategy: str, ticket_id: Optional[int] = None) -> bool:
        """Open a new trading position"""
        
        can_open, reason = self.can_open_position(symbol)
        if not can_open:
            self.logger.warning(f"Cannot open position for {symbol}: {reason}")
            return False
        
        # Create new position
        position = Position(
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            strategy=strategy,
            timestamp=datetime.now(),
            ticket_id=ticket_id,
            current_sl=sl_price,
            current_tp=tp_price
        )
        
        # Add to active positions
        self.positions[symbol] = position
        
        # Update daily statistics
        self.daily_stats['trades_opened'] += 1
        self.daily_trades['1H'] = self.daily_trades.get('1H', 0) + 1  # Default to 1H tracking
        
        # Log the position opening
        self.logger.info(f"Position opened: {symbol} {direction} {size} @ {entry_price}, SL: {sl_price}, TP: {tp_price}")
        
        # Add note
        position.notes.append(f"Position opened via {strategy} strategy")
        
        return True
    
    def close_position(self, symbol: str, pnl: float, reason: str, 
                      close_price: Optional[float] = None) -> bool:
        """Close an existing position"""
        
        if symbol not in self.positions:
            self.logger.warning(f"No open position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        # Update position details
        position.realized_pnl = pnl
        position.status = "closed"
        position.close_timestamp = datetime.now()
        position.close_reason = reason
        
        # Move to trade history
        self.trade_history.append(position)
        del self.positions[symbol]
        
        # Update daily statistics
        self.daily_stats['trades_closed'] += 1
        self.daily_stats['total_pnl'] += pnl
        
        if pnl > 0:
            self.daily_stats['winning_trades'] += 1
            self.daily_stats['consecutive_losses'] = 0
        else:
            self.daily_stats['losing_trades'] += 1
            self.daily_stats['consecutive_losses'] += 1
        
        # Log the position closing
        self.logger.info(f"Position closed: {symbol} P&L: {pnl:.2f} Reason: {reason}")
        
        # Add note
        position.notes.append(f"Position closed: {reason}")
        
        # Save updated trade history
        self._save_trade_history()
        
        return True
    
    def modify_position(self, symbol: str, sl_price: Optional[float] = None, 
                       tp_price: Optional[float] = None) -> bool:
        """Modify stop loss or take profit for an existing position"""
        
        if symbol not in self.positions:
            self.logger.warning(f"No open position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        if sl_price is not None:
            old_sl = position.current_sl
            position.current_sl = sl_price
            position.notes.append(f"SL modified: {old_sl} -> {sl_price}")
            self.logger.info(f"SL modified for {symbol}: {old_sl} -> {sl_price}")
        
        if tp_price is not None:
            old_tp = position.current_tp
            position.current_tp = tp_price
            position.notes.append(f"TP modified: {old_tp} -> {tp_price}")
            self.logger.info(f"TP modified for {symbol}: {old_tp} -> {tp_price}")
        
        return True
    
    def update_position_pnl(self, symbol: str, current_price: float) -> float:
        """Update unrealized P&L for a position"""
        
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        if position.direction == 'buy':
            pnl = (current_price - position.entry_price) * position.size
        else:  # sell
            pnl = (position.entry_price - current_price) * position.size
        
        position.unrealized_pnl = pnl
        return pnl
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all active positions"""
        return self.positions.copy()
    
    def has_active_position(self, symbol: str) -> bool:
        """Check if there's an active position for the symbol"""
        return symbol in self.positions and self.positions[symbol].status == "open"
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily trading statistics"""
        # Calculate win rate
        total_closed = self.daily_stats['trades_closed']
        if total_closed > 0:
            win_rate = (self.daily_stats['winning_trades'] / total_closed) * 100
        else:
            win_rate = 0.0
        
        # Calculate average win/loss
        winning_pnl = sum(pos.realized_pnl for pos in self.trade_history if pos.realized_pnl > 0)
        losing_pnl = sum(pos.realized_pnl for pos in self.trade_history if pos.realized_pnl < 0)
        
        avg_win = winning_pnl / self.daily_stats['winning_trades'] if self.daily_stats['winning_trades'] > 0 else 0
        avg_loss = losing_pnl / self.daily_stats['losing_trades'] if self.daily_stats['losing_trades'] > 0 else 0
        
        return {
            **self.daily_stats,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'active_positions': len(self.positions),
            'total_trades_today': self.daily_stats['trades_opened']
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics (typically called at start of new day)"""
        self.daily_stats = {
            'trades_opened': 0,
            'trades_closed': 0,
            'total_pnl': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'consecutive_losses': 0,
            'max_drawdown': 0.0,
            'last_reset': datetime.now().date()
        }
        self.daily_trades.clear()
        
        self.logger.info("Daily statistics reset")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate and return risk metrics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_consecutive_losses': 0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # Basic metrics
        winning_trades = [pos for pos in self.trade_history if pos.realized_pnl > 0]
        losing_trades = [pos for pos in self.trade_history if pos.realized_pnl < 0]
        
        total_trades = len(self.trade_history)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = sum(pos.realized_pnl for pos in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(pos.realized_pnl for pos in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(pos.realized_pnl for pos in winning_trades)
        total_losses = abs(sum(pos.realized_pnl for pos in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Largest win/loss
        largest_win = max(pos.realized_pnl for pos in self.trade_history) if self.trade_history else 0
        largest_loss = min(pos.realized_pnl for pos in self.trade_history) if self.trade_history else 0
        
        # Max consecutive losses
        max_consecutive_losses = 0
        current_consecutive = 0
        for pos in sorted(self.trade_history, key=lambda x: x.timestamp):
            if pos.realized_pnl < 0:
                current_consecutive += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
            else:
                current_consecutive = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_losses': max_consecutive_losses,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def export_trade_log(self, filepath: str):
        """Export trade history to CSV file"""
        try:
            import pandas as pd
            
            data = []
            for pos in self.trade_history:
                data.append({
                    'timestamp': pos.timestamp,
                    'symbol': pos.symbol,
                    'direction': pos.direction,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'sl_price': pos.sl_price,
                    'tp_price': pos.tp_price,
                    'strategy': pos.strategy,
                    'realized_pnl': pos.realized_pnl,
                    'close_timestamp': pos.close_timestamp,
                    'close_reason': pos.close_reason,
                    'status': pos.status
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Trade log exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export trade log: {e}")

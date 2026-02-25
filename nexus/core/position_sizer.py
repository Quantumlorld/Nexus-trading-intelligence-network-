"""
Nexus Trading System - Position Sizer
Calculates optimal position sizes based on risk management rules
"""

from typing import Dict, Any, Optional
import logging
import math


class PositionSizer:
    """Calculates position sizes based on risk management parameters"""
    
    def __init__(self, risk_config: Dict[str, Any]):
        self.risk_config = risk_config
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, signal: Dict[str, Any], market_data: Dict[str, Any],
                               daily_stats: Dict[str, Any], account_balance: float = 10000.0) -> float:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            signal: Trading signal with direction, confidence, etc.
            market_data: Current market data including price, volatility, etc.
            daily_stats: Daily trading statistics
            account_balance: Current account balance
            
        Returns:
            Position size in lots/units
        """
        
        # Get risk parameters
        position_sizing = self.risk_config['position_sizing']
        
        # Determine risk amount
        if position_sizing['dollar_override_enabled']:
            risk_amount = min(
                position_sizing['default_dollar_risk'],
                position_sizing['max_dollar_risk']
            )
        else:
            risk_percent = min(
                position_sizing['default_risk_percent'],
                position_sizing['max_risk_percent']
            )
            risk_amount = account_balance * (risk_percent / 100.0)
        
        # Adjust risk based on daily performance
        risk_amount = self._adjust_risk_for_daily_performance(risk_amount, daily_stats)
        
        # Calculate position size based on stop loss distance
        sl_distance = self._calculate_sl_distance(signal, market_data)
        
        if sl_distance <= 0:
            self.logger.warning("Invalid SL distance, using minimum position size")
            return 0.01  # Minimum position size
        
        # Calculate position size
        position_size = risk_amount / sl_distance
        
        # Apply position size limits
        position_size = self._apply_position_limits(position_size, market_data)
        
        # Round to appropriate precision
        position_size = self._round_position_size(position_size)
        
        self.logger.info(f"Calculated position size: {position_size} (risk: ${risk_amount:.2f}, SL distance: {sl_distance:.2f})")
        
        return position_size
    
    def _adjust_risk_for_daily_performance(self, base_risk: float, daily_stats: Dict[str, Any]) -> float:
        """Adjust risk amount based on daily performance"""
        
        # Reduce risk if we're having a bad day
        if daily_stats.get('total_pnl', 0) < 0:
            # Reduce risk by 50% if we're down for the day
            adjusted_risk = base_risk * 0.5
            self.logger.debug(f"Reduced risk due to daily loss: ${base_risk:.2f} -> ${adjusted_risk:.2f}")
            return adjusted_risk
        
        # Reduce risk after consecutive losses
        consecutive_losses = daily_stats.get('consecutive_losses', 0)
        if consecutive_losses >= 2:
            risk_reduction = 0.3 * consecutive_losses  # 30% reduction per loss
            adjusted_risk = base_risk * (1 - min(risk_reduction, 0.8))  # Max 80% reduction
            self.logger.debug(f"Reduced risk due to consecutive losses: ${base_risk:.2f} -> ${adjusted_risk:.2f}")
            return adjusted_risk
        
        # Can increase risk slightly if we're doing well (optional)
        if daily_stats.get('win_rate', 0) > 70 and daily_stats.get('trades_closed', 0) >= 3:
            # Increase risk by 20% if we're performing well
            adjusted_risk = base_risk * 1.2
            self.logger.debug(f"Increased risk due to good performance: ${base_risk:.2f} -> ${adjusted_risk:.2f}")
            return adjusted_risk
        
        return base_risk
    
    def _calculate_sl_distance(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate stop loss distance in price units"""
        
        # Get default SL distance from config
        default_sl_points = self.risk_config['stop_loss']['default_sl_points']
        
        # Get asset-specific point value
        symbol = market_data.get('symbol', '')
        point_value = self._get_point_value(symbol)
        
        # Convert points to price distance
        sl_distance = default_sl_points / point_value
        
        # Adjust for volatility if enabled
        if self.risk_config['volatility_filter']['enabled']:
            volatility_adjustment = self._calculate_volatility_adjustment(market_data)
            sl_distance *= volatility_adjustment
        
        # Apply buffer
        sl_buffer = self.risk_config['stop_loss']['sl_buffer_percent']
        sl_distance *= (1 + sl_buffer)
        
        return sl_distance
    
    def _get_point_value(self, symbol: str) -> float:
        """Get point value for a symbol"""
        # Default point values (would be loaded from config in production)
        point_values = {
            'XAUUSD': 100,
            'EURUSD': 100000,
            'USDX': 1000,
            'BTCUSD': 1
        }
        return point_values.get(symbol, 100)
    
    def _calculate_volatility_adjustment(self, market_data: Dict[str, Any]) -> float:
        """Calculate position size adjustment based on volatility"""
        
        volatility = market_data.get('volatility', 1.0)
        min_vol = self.risk_config['volatility_filter']['min_volatility_threshold']
        max_vol = self.risk_config['volatility_filter']['max_volatility_threshold']
        
        # If volatility is outside range, return 1 (no adjustment)
        if volatility < min_vol or volatility > max_vol:
            return 1.0
        
        # Adjust position size inversely with volatility
        # Higher volatility = smaller position size
        volatility_range = max_vol - min_vol
        volatility_ratio = (volatility - min_vol) / volatility_range if volatility_range > 0 else 0
        
        # Reduce position size by up to 50% for high volatility
        adjustment = 1.0 - (volatility_ratio * 0.5)
        
        return max(adjustment, 0.5)  # Never reduce by more than 50%
    
    def _apply_position_limits(self, position_size: float, market_data: Dict[str, Any]) -> float:
        """Apply position size limits and constraints"""
        
        # Get maximum position size from config
        max_position_size = self.risk_config['portfolio_risk'].get('max_position_size', 1.0)
        
        # Apply maximum limit
        position_size = min(position_size, max_position_size)
        
        # Get minimum position size (asset-specific)
        min_position_size = self._get_minimum_position_size(market_data.get('symbol', ''))
        position_size = max(position_size, min_position_size)
        
        # Check portfolio heat (total exposure)
        current_exposure = self._get_current_portfolio_exposure()
        max_portfolio_heat = self.risk_config['portfolio_risk']['max_portfolio_heat']
        
        if current_exposure >= max_portfolio_heat:
            self.logger.warning(f"Portfolio heat limit reached: {current_exposure:.1f}% >= {max_portfolio_heat:.1f}%")
            return 0.0  # No new positions
        
        # Reduce position size if portfolio is getting hot
        heat_ratio = current_exposure / max_portfolio_heat
        if heat_ratio > 0.7:  # If we're at 70% of max heat
            reduction_factor = 1.0 - ((heat_ratio - 0.7) / 0.3) * 0.5  # Reduce by up to 50%
            position_size *= reduction_factor
            self.logger.debug(f"Reduced position size due to portfolio heat: {reduction_factor:.2f}")
        
        return position_size
    
    def _get_minimum_position_size(self, symbol: str) -> float:
        """Get minimum position size for a symbol"""
        # Asset-specific minimum sizes
        min_sizes = {
            'XAUUSD': 0.01,
            'EURUSD': 0.01,
            'USDX': 0.01,
            'BTCUSD': 0.001
        }
        return min_sizes.get(symbol, 0.01)
    
    def _get_current_portfolio_exposure(self) -> float:
        """Get current portfolio exposure as percentage"""
        # This would integrate with trade manager to get current positions
        # For now, return 0 (no exposure)
        return 0.0
    
    def _round_position_size(self, position_size: float) -> float:
        """Round position size to appropriate precision"""
        
        # Round to 2 decimal places for most instruments
        # Some instruments (like crypto) might need more precision
        rounded_size = round(position_size, 2)
        
        # Ensure minimum size
        if rounded_size < 0.01:
            rounded_size = 0.01
        
        return rounded_size
    
    def calculate_risk_per_trade(self, account_balance: float, position_size: float,
                                entry_price: float, sl_price: float, symbol: str) -> float:
        """Calculate actual risk amount for a trade"""
        
        point_value = self._get_point_value(symbol)
        sl_distance_points = abs(entry_price - sl_price) * point_value
        
        risk_amount = position_size * sl_distance_points
        
        return risk_amount
    
    def validate_position_size(self, position_size: float, account_balance: float,
                             symbol: str) -> tuple[bool, str]:
        """Validate if position size meets all requirements"""
        
        if position_size <= 0:
            return False, "Position size must be positive"
        
        # Check minimum size
        min_size = self._get_minimum_position_size(symbol)
        if position_size < min_size:
            return False, f"Position size {position_size} below minimum {min_size}"
        
        # Check maximum size
        max_size = self.risk_config['portfolio_risk'].get('max_position_size', 1.0)
        if position_size > max_size:
            return False, f"Position size {position_size} exceeds maximum {max_size}"
        
        # Check if position size is reasonable for account balance
        max_risk_percent = self.risk_config['position_sizing']['max_risk_percent']
        max_account_risk = account_balance * (max_risk_percent / 100.0)
        
        # Estimate risk (this is approximate since we don't have SL price)
        estimated_risk = position_size * 100  # Rough estimate
        if estimated_risk > max_account_risk:
            return False, f"Position size risk {estimated_risk} exceeds account risk limit {max_account_risk}"
        
        return True, "Position size valid"
    
    def get_position_size_info(self, signal: Dict[str, Any], market_data: Dict[str, Any],
                            daily_stats: Dict[str, Any], account_balance: float = 10000.0) -> Dict[str, Any]:
        """Get detailed information about position size calculation"""
        
        position_size = self.calculate_position_size(signal, market_data, daily_stats, account_balance)
        
        # Calculate risk metrics
        sl_distance = self._calculate_sl_distance(signal, market_data)
        risk_amount = position_size * sl_distance
        risk_percent = (risk_amount / account_balance) * 100 if account_balance > 0 else 0
        
        return {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'sl_distance': sl_distance,
            'account_balance': account_balance,
            'symbol': market_data.get('symbol', ''),
            'direction': signal.get('direction', ''),
            'strategy': signal.get('strategy', '')
        }

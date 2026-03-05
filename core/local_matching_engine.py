"""
Nexus Trading System - Local Matching Engine
Deterministic order matching using OHLC data
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
import random
import time

logger = logging.getLogger(__name__)

@dataclass
class OrderBookEntry:
    """Order book entry"""
    order_id: str
    price: Decimal
    quantity: Decimal
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    user_id: int

@dataclass
class TradeExecution:
    """Trade execution result"""
    success: bool
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    execution_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    slippage: Optional[float] = None
    partially_filled: bool = False
    error: Optional[str] = None
    commission: Optional[float] = None

class LocalMatchingEngine:
    """Deterministic local matching engine"""
    
    def __init__(self):
        self.order_books: Dict[str, Dict[str, List[OrderBookEntry]]] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.commission_rate = 0.001  # 0.1% commission
        self.spread_bps = 5  # 5 basis points spread
        
        # Initialize order books for common symbols
        self._initialize_order_books()
    
    def _initialize_order_books(self):
        """Initialize order books for common symbols"""
        symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'BTC/USD', 'ETH/USD']
        for symbol in symbols:
            self.order_books[symbol] = {
                'buy': [],  # Bids (sorted by price descending)
                'sell': []  # Asks (sorted by price ascending)
            }
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information (simulated)"""
        return {
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0,
            'free_margin': 10000.0,
            'margin_level': 0.0,
            'profit': 0.0,
            'leverage': 100
        }
    
    def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open positions"""
        if symbol:
            return [pos for pos in self.trade_history if pos['symbol'] == symbol and pos.get('status') == 'open']
        return [pos for pos in self.trade_history if pos.get('status') == 'open']
    
    def place_order(self, **kwargs) -> TradeExecution:
        """Place an order and attempt to match it"""
        try:
            symbol = kwargs.get('symbol')
            action = kwargs.get('action')  # 'buy' or 'sell'
            volume = Decimal(str(kwargs.get('volume', 0)))
            price = Decimal(str(kwargs.get('price', 0)))
            comment = kwargs.get('comment', '')
            
            # Validate order
            if not symbol or symbol not in self.order_books:
                return TradeExecution(success=False, error=f"Invalid symbol: {symbol}")
            
            if volume <= 0:
                return TradeExecution(success=False, error="Invalid volume")
            
            if price <= 0:
                return TradeExecution(success=False, error="Invalid price")
            
            # Generate order ID
            order_id = f"ORDER_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            
            # Create order book entry
            entry = OrderBookEntry(
                order_id=order_id,
                price=price,
                quantity=volume,
                timestamp=datetime.utcnow(),
                side=action,
                user_id=kwargs.get('user_id', 1)
            )
            
            # Add to order book
            self.order_books[symbol][action].append(entry)
            
            # Sort order book
            if action == 'buy':
                self.order_books[symbol]['buy'].sort(key=lambda x: x.price, reverse=True)
            else:
                self.order_books[symbol]['sell'].sort(key=lambda x: x.price)
            
            # Attempt to match the order
            execution = self._match_order(symbol, entry)
            
            return execution
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return TradeExecution(success=False, error=str(e))
    
    def _match_order(self, symbol: str, order: OrderBookEntry) -> TradeExecution:
        """Match order against order book"""
        try:
            opposite_side = 'sell' if order.side == 'buy' else 'buy'
            order_book = self.order_books[symbol][opposite_side]
            
            if not order_book:
                # No matching orders available
                return TradeExecution(
                    success=False,
                    error="No liquidity available",
                    order_id=order.order_id
                )
            
            # Find best matching price
            if order.side == 'buy':
                # Buy order matches with lowest ask
                matching_orders = [o for o in order_book if o.price <= order.price]
            else:
                # Sell order matches with highest bid
                matching_orders = [o for o in order_book if o.price >= order.price]
            
            if not matching_orders:
                # No orders at acceptable price
                return TradeExecution(
                    success=False,
                    error="No orders at acceptable price",
                    order_id=order.order_id
                )
            
            # Execute trade with best match
            best_match = matching_orders[0]
            
            # Calculate execution price (mid-price with small random slippage)
            mid_price = (order.price + best_match.price) / 2
            slippage_factor = 1 + random.uniform(-0.0001, 0.0001)  # ±0.01% slippage
            execution_price = float(mid_price * slippage_factor)
            
            # Calculate commission
            commission = execution_price * float(order.quantity) * self.commission_rate
            
            # Determine fill quantity (can be partial)
            fill_quantity = min(order.quantity, best_match.quantity)
            partially_filled = fill_quantity < order.quantity
            
            # Update order book
            if best_match.quantity <= fill_quantity:
                # Fully consume the matching order
                self.order_books[symbol][opposite_side].remove(best_match)
            else:
                # Partially consume the matching order
                best_match.quantity -= fill_quantity
            
            # Remove our order if fully filled
            if not partially_filled:
                self.order_books[symbol][order.side].remove(order)
            else:
                # Update remaining quantity
                order.quantity -= fill_quantity
            
            # Record trade
            trade_record = {
                'trade_id': f"TRADE_{int(time.time() * 1000)}",
                'symbol': symbol,
                'order_id': order.order_id,
                'matched_order_id': best_match.order_id,
                'side': order.side,
                'quantity': float(fill_quantity),
                'price': execution_price,
                'commission': commission,
                'timestamp': datetime.utcnow(),
                'status': 'closed'
            }
            
            self.trade_history.append(trade_record)
            
            # Calculate slippage
            slippage = ((execution_price - float(order.price)) / float(order.price)) * 100
            
            logger.info(f"Trade executed: {symbol} {order.side} {fill_quantity} @ {execution_price}")
            
            return TradeExecution(
                success=True,
                order_id=order.order_id,
                position_id=trade_record['trade_id'],
                execution_price=execution_price,
                filled_quantity=float(fill_quantity),
                slippage=slippage,
                partially_filled=partially_filled,
                commission=commission
            )
            
        except Exception as e:
            logger.error(f"Order matching failed: {e}")
            return TradeExecution(success=False, error=str(e))
    
    def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price (mid of best bid/ask)"""
        try:
            if symbol not in self.order_books:
                return None
            
            bids = self.order_books[symbol]['buy']
            asks = self.order_books[symbol]['sell']
            
            if not bids or not asks:
                # Return a reasonable default price based on symbol
                default_prices = {
                    'EUR/USD': 1.1000,
                    'GBP/USD': 1.2500,
                    'USD/JPY': 110.00,
                    'BTC/USD': 45000.0,
                    'ETH/USD': 3000.0
                }
                return default_prices.get(symbol, 1.0)
            
            best_bid = bids[0].price
            best_ask = asks[0].price
            mid_price = (best_bid + best_ask) / 2
            
            return float(mid_price)
            
        except Exception as e:
            logger.error(f"Failed to get market price: {e}")
            return None
    
    def add_liquidity(self, symbol: str, side: str, price: float, quantity: float):
        """Add liquidity to the order book (for testing)"""
        try:
            if symbol not in self.order_books:
                self.order_books[symbol] = {'buy': [], 'sell': []}
            
            entry = OrderBookEntry(
                order_id=f"LIQ_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                price=Decimal(str(price)),
                quantity=Decimal(str(quantity)),
                timestamp=datetime.utcnow(),
                side=side,
                user_id=0  # System liquidity
            )
            
            self.order_books[symbol][side].append(entry)
            
            # Sort order book
            if side == 'buy':
                self.order_books[symbol]['buy'].sort(key=lambda x: x.price, reverse=True)
            else:
                self.order_books[symbol]['sell'].sort(key=lambda x: x.price)
            
            logger.info(f"Added liquidity: {symbol} {side} {quantity} @ {price}")
            
        except Exception as e:
            logger.error(f"Failed to add liquidity: {e}")
    
    def initialize_market(self):
        """Initialize market with some liquidity"""
        # Add liquidity for EUR/USD
        base_price = 1.1000
        for i in range(10):
            bid_price = base_price - (i * 0.0001)
            ask_price = base_price + (i * 0.0001)
            self.add_liquidity('EUR/USD', 'buy', bid_price, 1.0)
            self.add_liquidity('EUR/USD', 'sell', ask_price, 1.0)
        
        # Add liquidity for BTC/USD
        base_price = 45000.0
        for i in range(5):
            bid_price = base_price - (i * 10)
            ask_price = base_price + (i * 10)
            self.add_liquidity('BTC/USD', 'buy', bid_price, 0.1)
            self.add_liquidity('BTC/USD', 'sell', ask_price, 0.1)
        
        logger.info("Market initialized with liquidity")

# Global matching engine instance
local_matching_engine = LocalMatchingEngine()

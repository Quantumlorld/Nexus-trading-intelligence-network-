"""
NEXUS MT5 Bridge Client
MQL5 WebRequest Bridge Implementation
Bypasses Python MT5 IPC issues by using MQL5 service as bridge
"""

import requests
import json
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BridgeCommand:
    command: str
    params: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MT5BridgeClient:
    """Client for MQL5 WebRequest bridge"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.bridge_url = f"{backend_url}/mt5/bridge"
        self.pending_commands = []
        self.command_responses = {}
        self.bridge_active = False
        
    def send_command(self, command: str, params: Dict[str, Any] = None, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Send command to MQL5 bridge and wait for response"""
        cmd = BridgeCommand(command=command, params=params or {})
        
        # Add to pending commands queue
        self.pending_commands.append(cmd)
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if response received
            response_key = f"{command}_{int(cmd.timestamp)}"
            if response_key in self.command_responses:
                response = self.command_responses.pop(response_key)
                return response
            
            # Poll backend for bridge data
            self._poll_bridge_data()
            time.sleep(0.5)
        
        logger.error(f"Command {command} timed out")
        return None
    
    def _poll_bridge_data(self):
        """Poll backend for bridge responses"""
        try:
            response = requests.get(f"{self.bridge_url}/responses", timeout=5)
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    cmd_key = f"{item.get('command')}_{int(item.get('timestamp', 0))}"
                    self.command_responses[cmd_key] = item
        except Exception as e:
            logger.error(f"Error polling bridge data: {e}")
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information via bridge"""
        response = self.send_command("get_account_info")
        if response and response.get("command") == "account_info":
            return {
                "login": response.get("login"),
                "server": response.get("server"),
                "balance": response.get("balance"),
                "equity": response.get("equity"),
                "margin": response.get("margin"),
                "free_margin": response.get("free_margin"),
                "profit": response.get("profit"),
                "leverage": response.get("leverage"),
                "currency": response.get("currency")
            }
        return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information via bridge"""
        response = self.send_command("get_symbol_info", {"symbol": symbol})
        if response and response.get("command") == "symbol_info":
            return {
                "symbol": response.get("symbol"),
                "bid": response.get("bid"),
                "ask": response.get("ask"),
                "last": response.get("last"),
                "volume": response.get("volume"),
                "time": response.get("time"),
                "digits": response.get("digits"),
                "point": response.get("point"),
                "trade_contract_size": response.get("trade_contract_size"),
                "volume_min": response.get("volume_min"),
                "volume_max": response.get("volume_max"),
                "volume_step": response.get("volume_step"),
                "spread": response.get("spread")
            }
        return None
    
    def place_order(self, symbol: str, volume: float, order_type: int, 
                   price: float = 0, sl: float = 0, tp: float = 0, 
                   comment: str = "NEXUS") -> Optional[Dict[str, Any]]:
        """Place order via bridge"""
        params = {
            "symbol": symbol,
            "volume": volume,
            "order_type": order_type,  # 0=buy, 1=sell
            "price": price,
            "sl": sl,
            "tp": tp,
            "comment": comment
        }
        
        response = self.send_command("place_order", params)
        if response and response.get("command") == "order_result":
            return {
                "result": response.get("result"),
                "ticket": response.get("ticket"),
                "error": response.get("error"),
                "symbol": response.get("symbol"),
                "volume": response.get("volume"),
                "order_type": response.get("order_type")
            }
        return None
    
    def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        """Get open positions via bridge"""
        response = self.send_command("get_positions")
        if response and response.get("command") == "positions":
            return response.get("positions", [])
        return None
    
    def test_connection(self) -> bool:
        """Test bridge connection"""
        account_info = self.get_account_info()
        return account_info is not None

# Bridge integration with existing MT5 connector
class BridgeMT5Connector:
    """MT5 Connector using MQL5 WebRequest bridge"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.bridge = MT5BridgeClient(backend_url)
        self.connected = False
        self.account_info = None
        
    def initialize(self) -> bool:
        """Initialize bridge connection"""
        try:
            # Test bridge connection
            self.connected = self.bridge.test_connection()
            if self.connected:
                self.account_info = self.bridge.get_account_info()
                logger.info("Bridge MT5 connector initialized successfully")
            return self.connected
        except Exception as e:
            logger.error(f"Failed to initialize bridge connector: {e}")
            return False
    
    def login(self, account: int, password: str, server: str) -> bool:
        """Login via bridge (handled by MQL5 service)"""
        # Bridge handles login automatically through MQL5 service
        return self.connected
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account info via bridge"""
        if self.connected:
            return self.bridge.get_account_info()
        return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol info via bridge"""
        if self.connected:
            return self.bridge.get_symbol_info(symbol)
        return None
    
    def place_buy_order(self, symbol: str, volume: float, price: float = 0, 
                       sl: float = 0, tp: float = 0, comment: str = "NEXUS") -> Optional[Dict[str, Any]]:
        """Place buy order via bridge"""
        if self.connected:
            return self.bridge.place_order(symbol, volume, 0, price, sl, tp, comment)
        return None
    
    def place_sell_order(self, symbol: str, volume: float, price: float = 0, 
                        sl: float = 0, tp: float = 0, comment: str = "NEXUS") -> Optional[Dict[str, Any]]:
        """Place sell order via bridge"""
        if self.connected:
            return self.bridge.place_order(symbol, volume, 1, price, sl, tp, comment)
        return None
    
    def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        """Get positions via bridge"""
        if self.connected:
            return self.bridge.get_positions()
        return None
    
    def shutdown(self):
        """Shutdown bridge connection"""
        self.connected = False
        self.account_info = None

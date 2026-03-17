"""
Nexus Trading System - IMMEDIATE DEMO TRADING
Fast solution to get demo trading working NOW
"""

import asyncio
import logging
import requests
import time
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImmediateDemoTrader:
    """Immediate demo trading solution"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.trading_enabled = False
        self.demo_balance = 10000.0
        self.trades_executed = 0
        self.win_rate = 0.0
        
    def check_backend_health(self) -> bool:
        """Check if backend is running"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_demo_trading(self) -> Dict[str, Any]:
        """Start immediate demo trading"""
        logger.info("🚀 STARTING IMMEDIATE DEMO TRADING")
        
        if not self.check_backend_health():
            return {"success": False, "error": "Backend not running"}
        
        # Enable trading first
        try:
            response = requests.post(f"{self.backend_url}/admin/enable-trading", timeout=5)
            if response.status_code != 200:
                return {"success": False, "error": "Failed to enable trading"}
        except Exception as e:
            return {"success": False, "error": f"Trading enable error: {e}"}
        
        # Enable broker connection
        try:
            response = requests.post(f"{self.backend_url}/admin/simulate-broker-recovery", timeout=5)
            if response.status_code != 200:
                return {"success": False, "error": "Failed to connect broker"}
        except Exception as e:
            return {"success": False, "error": f"Broker connection error: {e}"}
        
        # Initialize demo trading via backend
        try:
            response = requests.post(
                f"{self.backend_url}/admin/start-demo",
                json={"trades_target": 500, "risk_per_trade": 1.0},
                timeout=10
            )
            
            if response.status_code == 200:
                self.trading_enabled = True
                logger.info("✅ Demo trading STARTED!")
                return {"success": True, "message": "Demo trading started"}
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        try:
            response = requests.get(f"{self.backend_url}/admin/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        return {
            "demo_trading": self.trading_enabled,
            "balance": self.demo_balance,
            "trades_executed": self.trades_executed,
            "win_rate": self.win_rate
        }
    
    def execute_demo_trade(self, symbol: str, action: str, volume: float) -> Dict[str, Any]:
        """Execute a demo trade"""
        if not self.trading_enabled:
            return {"success": False, "error": "Demo trading not enabled"}
        
        try:
            response = requests.post(
                f"{self.backend_url}/trade",
                json={
                    "symbol": symbol,
                    "action": action,
                    "quantity": volume,
                    "order_type": "MARKET"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self.trades_executed += 1
                    logger.info(f"✅ Trade executed: {action} {volume} {symbol}")
                    return result
            
            return {"success": False, "error": response.text}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    """Main execution function"""
    trader = ImmediateDemoTrader()
    
    print("🎯 NEXUS IMMEDIATE DEMO TRADING")
    print("=" * 50)
    
    # Check backend
    if not trader.check_backend_health():
        print("❌ Backend not running! Start with: python simple_app.py")
        return
    
    print("✅ Backend is running")
    
    # Start demo trading
    result = trader.start_demo_trading()
    if result["success"]:
        print("✅ Demo trading started successfully!")
        
        # Show status
        status = trader.get_trading_status()
        print(f"📊 Status: {status}")
        
        # Execute sample trades
        print("\n🚀 Executing sample demo trades...")
        
        trades = [
            ("EUR/USD", "BUY", 0.01),
            ("XAU/USD", "BUY", 0.01),
            ("BTC/USD", "SELL", 0.001)
        ]
        
        for symbol, action, volume in trades:
            result = trader.execute_demo_trade(symbol, action, volume)
            if result["success"]:
                print(f"✅ {action} {volume} {symbol}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Final status
        final_status = trader.get_trading_status()
        print(f"\n📈 Final Status: {final_status}")
        
    else:
        print(f"❌ Failed to start demo trading: {result.get('error')}")
    
    print("\n🎉 IMMEDIATE DEMO TRADING COMPLETE!")
    print("🌐 Frontend: http://localhost:5173")
    print("📊 Backend API: http://localhost:8000/docs")

if __name__ == "__main__":
    main()

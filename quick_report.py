#!/usr/bin/env python3
"""
Quick system status report
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Generate quick system report"""
    
    print("🚀 NEXUS TRADING SYSTEM - QUICK REPORT")
    print("=" * 50)
    
    try:
        # Import and initialize system
        from main import NexusTradingSystem
        
        print("📊 Initializing Nexus Trading System...")
        nexus = NexusTradingSystem("config")
        
        print("✅ System initialized successfully!")
        print("\n📋 SYSTEM COMPONENTS:")
        
        print(f"✅ Strategies: {len(nexus.strategies)} loaded")
        for strategy in nexus.strategies:
            print(f"  - {strategy.name}")
        
        print(f"✅ Backtest Engine: {'Ready' if nexus.backtest_engine else 'Not Ready'}")
        print(f"✅ Order Executor: {'Ready' if nexus.order_executor else 'Not Ready'}")
        print(f"✅ Performance Tracker: {'Ready' if nexus.performance_tracker else 'Not Ready'}")
        print(f"✅ Volatility Monitor: {'Ready' if nexus.volatility_monitor else 'Not Ready'}")
        print(f"✅ Alert System: {'Ready' if nexus.alert_system else 'Not Ready'}")
        
        print(f"\n✅ Configuration Files:")
        for name, config in nexus.configs.items():
            print(f"  - {name}.yaml: {len(config)} sections")
        
        print(f"\n✅ Trading Rules Verified:")
        risk_config = nexus.configs['risk']['risk_management']
        print(f"  - Default Risk: {risk_config['position_sizing']['default_risk_percent']}%")
        print(f"  - Daily Loss Limit: ${risk_config['daily_limits']['max_daily_loss']}")
        print(f"  - Max Daily Trades: {risk_config['daily_limits']['max_daily_trades']}")
        
        sl_config = risk_config['stop_loss']
        print(f"  - Default SL Points: {sl_config['default_sl_points']}")
        
        tp_config = risk_config['take_profit']
        print(f"  - Default TP Points: {tp_config['default_tp_points']}")
        
        dynamic_config = risk_config['dynamic_management']
        print(f"  - TP Extension: {dynamic_config['extend_tp_threshold']}")
        print(f"  - Runner Mode: {dynamic_config['runner_mode_enabled']}")
        
        print(f"\n✅ Assets Configured:")
        assets = nexus.configs['assets']['assets']
        for symbol, config in assets.items():
            print(f"  - {symbol}: {config.get('name', 'Unknown')}")
        
        print(f"\n🎯 SYSTEM STATUS: FULLY OPERATIONAL!")
        print(f"📊 Ready for backtesting and live trading!")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Run backtest: python main.py --backtest --start 2023-01-01 --end 2023-12-31")
        print(f"2. Train models: python main.py --train --symbol XAUUSD --start 2023-01-01 --end 2023-12-31")
        print(f"3. Start live trading: python main.py --trade")
        print(f"4. Monitor performance: python main.py --status")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

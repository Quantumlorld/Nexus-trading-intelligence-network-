#!/usr/bin/env python3
"""
Simple test script to verify Nexus Trading System functionality
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    
    try:
        # Test core imports
        from core.trade_manager import TradeManager
        from core.risk_engine import RiskEngine
        from core.position_sizer import PositionSizer
        from core.logger import setup_logging
        print("✅ Core components imported successfully")
        
        # Test strategy imports
        from strategy.base_strategy import BaseStrategy
        from strategy.ema_crossover import EMACrossoverStrategy
        print("✅ Strategy components imported successfully")
        
        # Test data imports
        from data.loaders import DataManager
        print("✅ Data components imported successfully")
        
        # Test backtest imports
        from backtest.backtest_engine import BacktestEngine
        print("✅ Backtest components imported successfully")
        
        # Test execution imports
        from execution.order_executor import OrderExecutor
        from execution.session_filter import SessionFilter
        print("✅ Execution components imported successfully")
        
        # Test monitoring imports
        from monitoring.performance_tracker import PerformanceTracker
        from monitoring.volatility_monitor import VolatilityMonitor
        from monitoring.alert_system import AlertSystem
        print("✅ Monitoring components imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n🔍 Testing configuration loading...")
    
    try:
        import yaml
        from pathlib import Path
        
        config_dir = Path("config")
        if not config_dir.exists():
            print("❌ Config directory not found")
            return False
        
        config_files = list(config_dir.glob("*.yaml"))
        print(f"✅ Found {len(config_files)} config files:")
        
        for config_file in config_files:
            print(f"  - {config_file.name}")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                print(f"    Keys: {list(config.keys()) if config else 'Empty'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Test trade manager
        from core.trade_manager import TradeManager
        trade_manager = TradeManager({})
        print("✅ TradeManager created successfully")
        
        # Test risk engine
        from core.risk_engine import RiskEngine
        risk_engine = RiskEngine({})
        print("✅ RiskEngine created successfully")
        
        # Test position sizer
        from core.position_sizer import PositionSizer
        position_sizer = PositionSizer({})
        print("✅ PositionSizer created successfully")
        
        # Test logger
        from core.logger import setup_logging
        logger = setup_logging("INFO")
        print("✅ Logger setup successfully")
        
        # Test strategy
        from strategy.ema_crossover import EMACrossoverStrategy, EMACrossoverConfig
        ema_config = EMACrossoverConfig(
            fast_ema_period=12,
            slow_ema_period=26,
            signal_ema_period=9,
            min_confidence=0.6,
            assets=['XAUUSD']
        )
        ema_strategy = EMACrossoverStrategy(ema_config)
        print("✅ EMACrossoverStrategy created successfully")
        
        # Test session filter
        from execution.session_filter import SessionFilter
        session_filter = SessionFilter({})
        print("✅ SessionFilter created successfully")
        
        # Test performance tracker
        from monitoring.performance_tracker import PerformanceTracker
        perf_tracker = PerformanceTracker("test_performance.db")
        print("✅ PerformanceTracker created successfully")
        
        # Test volatility monitor
        from monitoring.volatility_monitor import VolatilityMonitor
        vol_monitor = VolatilityMonitor("test_volatility.db")
        print("✅ VolatilityMonitor created successfully")
        
        # Test alert system
        from monitoring.alert_system import AlertSystem, AlertCategory, AlertSeverity
        alert_system = AlertSystem({})
        print("✅ AlertSystem created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trading_rules():
    """Test trading rules implementation"""
    print("\n🔍 Testing trading rules...")
    
    try:
        # Test risk limits
        risk_config = {
            'default_risk_percent': 1.0,
            'daily_loss_limit': 9.99,
            'max_consecutive_losses': 3,
            'position_sizing': {
                'default_risk_percent': 1.0,
                'max_position_size': 1.0
            }
        }
        
        from core.risk_engine import RiskEngine
        risk_engine = RiskEngine(risk_config)
        
        # Test basic risk check
        trade_request = {
            'symbol': 'XAUUSD',
            'direction': 'BUY',
            'size': 0.01,
            'entry_price': 2000.0,
            'stop_loss': 1997.0,
            'take_profit': 2009.9
        }
        
        risk_result = risk_engine.assess_trade_risk(trade_request)
        print(f"✅ Risk assessment completed: {risk_result['risk_score']:.2f}")
        print(f"   Recommendation: {risk_result['recommendation']}")
        
        # Test position sizing
        from core.position_sizer import PositionSizer
        position_sizer = PositionSizer(risk_config)
        
        position_size = position_sizer.calculate_position_size(
            symbol='XAUUSD',
            account_balance=10000.0,
            risk_percent=1.0,
            stop_loss_distance=3.0
        )
        print(f"✅ Position sizing: {position_size:.4f} lots")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading rules test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 NEXUS TRADING SYSTEM - SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_config_loading,
        test_basic_functionality,
        test_trading_rules
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test failed: {test.__name__}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
        print("\n📋 NEXT STEPS:")
        print("1. Configure your trading parameters in config/ directory")
        print("2. Load market data (CSV or MT5)")
        print("3. Run: python main.py --backtest --start 2023-01-01 --end 2023-12-31")
        print("4. Start live trading: python main.py --trade")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

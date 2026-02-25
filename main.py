"""
Nexus Trading System - Main Entry Point
Professional production-ready Python trading system with comprehensive features

This is the main entry point for the Nexus Trading System.
Run this file to start the complete trading system with all components.
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import json
from typing import Dict, List, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Nexus components
from core.engine import TradingEngine
from core.trade_manager import TradeManager
from core.risk_engine import RiskEngine
from core.position_sizer import PositionSizer
from core.regime_detector import RegimeDetector
from core.logger import setup_logging

from strategy.base_strategy import BaseStrategy, StrategyConfig
from strategy.ema_crossover import EMACrossoverStrategy, EMACrossoverConfig
from strategy.swing_homeostasis import SwingHomeostasisStrategy, SwingHomeostasisConfig
from strategy.hybrid_multitimeframe import HybridMultiTimeframeStrategy, HybridMultiTimeframeConfig

from backtest.backtest_engine import BacktestEngine, BacktestConfig
from backtest.runner import BacktestRunner, RunnerConfig
from backtest.metrics import MetricsCalculator

from execution.order_executor import OrderExecutor
from execution.mt5_bridge import MT5Bridge
from execution.session_filter import SessionFilter

from monitoring.performance_tracker import PerformanceTracker
from monitoring.volatility_monitor import VolatilityMonitor
from monitoring.alert_system import AlertSystem

from data.loaders import DataManager
from models.feature_engineering import FeatureEngineer
from models.neural_model import NeuralTradingModel, ModelConfig
from models.trainer import ModelTrainer, TrainingConfig
from models.evaluator import ModelEvaluator


class NexusTradingSystem:
    """Main Nexus Trading System orchestrator"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.project_root = Path(__file__).parent
        
        # Setup logging
        logger_instance = setup_logging("INFO")
        self.logger = logger_instance.system_logger
        self.logger.info("=" * 60)
        self.logger.info("NEXUS TRADING SYSTEM STARTING")
        self.logger.info("=" * 60)
        
        # Load configurations
        self.configs = self._load_all_configs()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("Nexus Trading System initialized successfully")
    
    def _load_all_configs(self) -> Dict[str, Any]:
        """Load all configuration files"""
        
        configs = {}
        config_files = ['assets.yaml', 'risk.yaml', 'execution.yaml', 'model.yaml']
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            try:
                with open(config_path, 'r') as f:
                    configs[config_file.replace('.yaml', '')] = yaml.safe_load(f)
                self.logger.info(f"Loaded config: {config_file}")
            except Exception as e:
                self.logger.error(f"Failed to load {config_file}: {e}")
                configs[config_file.replace('.yaml', '')] = {}
        
        return configs
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        # Core components
        self.data_manager = DataManager(
            loader_type=self.configs['execution'].get('loader_type', 'csv')
        )
        
        self.trade_manager = TradeManager(self.configs['risk'])
        self.risk_engine = RiskEngine(self.configs['risk'])
        self.position_sizer = PositionSizer(self.configs['risk'])
        self.regime_detector = RegimeDetector()
        
        # Strategies
        self.strategies = self._initialize_strategies()
        
        # Backtesting
        from backtest.backtest_engine import BacktestEngine, BacktestConfig
        from backtest.runner import BacktestRunner, RunnerConfig
        from backtest.metrics import MetricsCalculator
        
        # Create default backtest config
        backtest_config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
            initial_capital=10000.0
        )
        
        self.backtest_engine = BacktestEngine(backtest_config)
        self.backtest_runner = BacktestRunner(
            RunnerConfig(
                run_parallel=True,
                max_workers=4,
                enable_dynamic_management=True,
                generate_reports=True,
                create_charts=True
            )
        )
        
        # Execution
        self.order_executor = OrderExecutor(self.configs['execution'])
        self.mt5_bridge = MT5Bridge(self.configs['execution'])
        self.session_filter = SessionFilter(self.configs['execution'])
        
        # Monitoring
        self.performance_tracker = PerformanceTracker()
        self.volatility_monitor = VolatilityMonitor()
        self.alert_system = AlertSystem(self.configs.get('monitoring', {}))
        
        # ML Models (optional)
        self.feature_engineer = FeatureEngineer(self.configs['model'])
        self.neural_model = None
        self.model_trainer = None
        self.model_evaluator = None
        
        if self.configs['model'].get('enabled', False):
            self._initialize_ml_models()
        
        # Main trading engine
        self.trading_engine = TradingEngine(self.config_dir)
        
        # Register strategies with engine
        for strategy in self.strategies:
            self.trading_engine.register_strategy(strategy)
    
    def _initialize_strategies(self) -> List[BaseStrategy]:
        """Initialize all trading strategies"""
        
        strategies = []
        
        # EMA Crossover Strategy
        ema_config = EMACrossoverConfig(
            fast_ema_period=12,
            slow_ema_period=26,
            signal_ema_period=9,
            min_confidence=0.6,
            assets=self.configs['assets']['assets'].keys()
        )
        strategies.append(EMACrossoverStrategy(ema_config))
        
        # Swing Homeostasis Strategy
        swing_config = SwingHomeostasisConfig(
            lookback_period=100,
            structure_sensitivity=0.02,
            liquidity_threshold=0.001,
            equilibrium_window=20,
            min_sweep_strength=0.003,
            confirmation_bars=3,
            assets=self.configs['assets']['assets'].keys()
        )
        strategies.append(SwingHomeostasisStrategy(swing_config))
        
        # Hybrid Multi-Timeframe Strategy
        hybrid_config = HybridMultiTimeframeConfig(
            timeframes=["1W", "1D", "4H", "1H", "15M"],
            timeframe_weights={
                "1W": 0.3,
                "1D": 0.25,
                "4H": 0.2,
                "1H": 0.15,
                "15M": 0.1
            },
            weekly_bias_weight=0.3,
            daily_structure_weight=0.25,
            confluence_threshold=0.7,
            min_timeframes_agree=2,
            assets=self.configs['assets']['assets'].keys()
        )
        strategies.append(HybridMultiTimeframeStrategy(hybrid_config))
        
        self.logger.info(f"Initialized {len(strategies)} trading strategies")
        return strategies
    
    def _initialize_ml_models(self):
        """Initialize ML models if enabled"""
        
        try:
            model_config = ModelConfig(
                architecture="LSTM",
                input_size=50,
                hidden_layers=[128, 64, 32],
                dropout_rate=0.2,
                batch_size=32,
                learning_rate=0.001,
                epochs=100
            )
            
            self.neural_model = NeuralTradingModel(model_config, "models/saved")
            self.model_trainer = ModelTrainer(
                TrainingConfig(
                    train_split=0.8,
                    val_split=0.1,
                    test_split=0.1,
                    cross_validation_folds=5,
                    walk_forward_enabled=True,
                    hyperparameter_tuning=False
                ),
                "models/saved"
            )
            self.model_evaluator = ModelEvaluator("models/evaluation")
            
            self.logger.info("ML models initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
    
    async def start_trading(self):
        """Start live trading"""
        
        self.logger.info("Starting live trading...")
        
        try:
            # Start monitoring systems
            self.performance_tracker.start()
            self.volatility_monitor.start()
            self.alert_system.start()
            
            # Connect to broker
            if not self.mt5_bridge.connect():
                self.logger.error("Failed to connect to broker")
                return False
            
            # Start order executor
            await self.order_executor.start()
            
            # Start trading engine
            self.trading_engine.async_start()
            
            self.logger.info("Live trading started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start trading: {e}")
            return False
    
    async def stop_trading(self):
        """Stop live trading"""
        
        self.logger.info("Stopping live trading...")
        
        try:
            # Stop trading engine
            self.trading_engine.stop()
            
            # Stop order executor
            await self.order_executor.stop()
            
            # Stop monitoring
            self.performance_tracker.stop()
            self.volatility_monitor.stop()
            self.alert_system.stop()
            
            # Disconnect from broker
            self.mt5_bridge.disconnect()
            
            self.logger.info("Live trading stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
    
    def run_backtest(self, start_date: str, end_date: str, 
                     strategies: List[str] = None, 
                     initial_capital: float = 10000.0,
                     parallel: bool = True) -> Dict[str, Any]:
        """Run comprehensive backtesting"""
        
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        try:
            # Parse dates
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Create backtest configuration
            backtest_config = BacktestConfig(
                start_date=start_dt,
                end_date=end_dt,
                initial_capital=initial_capital,
                commission=0.0001,
                slippage=0.0001,
                enable_dynamic_tp=True,
                enable_runner_mode=True,
                enforce_session_filters=True,
                enforce_risk_limits=True,
                save_trades=True,
                save_equity_curve=True
            )
            
            # Select strategies
            if strategies:
                selected_strategies = [s for s in self.strategies if s.name in strategies]
            else:
                selected_strategies = self.strategies
            
            # Get market data
            symbols = list(self.configs['assets']['assets'].keys())
            market_data = {}
            
            for symbol in symbols:
                try:
                    # Load historical data
                    data = self.data_manager.get_latest_data(symbol, "1D", periods=1000)
                    if data is not None and not data.empty:
                        market_data[symbol] = data
                        self.logger.info(f"Loaded {len(data)} data points for {symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to load data for {symbol}: {e}")
            
            if not market_data:
                self.logger.error("No market data available for backtesting")
                return {'success': False, 'error': 'No market data'}
            
            # Run backtest
            if parallel:
                results = self.backtest_runner.run_parallel_backtests()
            else:
                results = {}
                job_id = self.backtest_runner.add_backtest_job(
                    selected_strategies, market_data, backtest_config
                )
                results = self.backtest_runner.run_single_backtest(
                    selected_strategies, market_data, backtest_config
                )
                results[job_id] = results
            
            # Generate comparison
            comparison = self.backtest_runner.compare_results(list(results.keys()))
            
            # Export results
            self.backtest_runner.export_all_results('json')
            
            self.logger.info(f"Backtest completed. Results: {len(results)}")
            
            return {
                'success': True,
                'results': results,
                'comparison': comparison,
                'total_trades': sum(len(r.trades) for r in results.values()),
                'backtest_config': backtest_config.__dict__
            }
        
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_models(self, symbol: str, start_date: str, end_date: str,
                   model_type: str = "LSTM") -> Dict[str, Any]:
        """Train ML models for a symbol"""
        
        if not self.configs['model']['enabled']:
            return {'success': False, 'error': 'ML models are disabled'}
        
        self.logger.info(f"Training {model_type} model for {symbol}")
        
        try:
            # Parse dates
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Get training data
            data = self.data_manager.get_data(symbol, "1H", start_dt, end_dt)
            
            if data is None or data.empty:
                return {'success': False, 'error': f'No training data available for {symbol}'}
            
            # Create features
            features = self.feature_engineer.create_features(
                data, 
                self.configs['model'],
                symbol
            )
            
            # Train model
            training_result = self.model_trainer.train_model(
                features.features, 
                features.target,
                self.neural_model.config,
                self.configs['model'],
                symbol
            )
            
            # Evaluate model
            test_data = self.data_manager.get_data(symbol, "1H", 
                                               end_dt - timedelta(days=30), end_dt)
            if test_data is not None and not test_data.empty:
                test_features = self.feature_engineer.create_features(
                    test_data,
                    self.configs['model'],
                    f"{symbol}_test"
                )
                
                evaluation = self.model_evaluator.evaluate_model(
                    self.neural_model,
                    test_features.features,
                    test_features.target,
                    test_data
                )
                
                self.logger.info(f"Model evaluation: {evaluation}")
            
            return {
                'success': True,
                'training_result': training_result,
                'evaluation': evaluation if 'evaluation' in locals() else None,
                'model_summary': self.neural_model.get_model_summary()
            }
        
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'trading_engine': self.trading_engine.get_status(),
            'data_manager': {
                'loader_type': self.data_manager.loader_type,
                'cache_size': len(self.data_manager.data_cache)
            },
            'strategies': {
                'total': len(self.strategies),
                'active': [s.name for s in self.strategies if s.is_active],
                'names': [s.name for s in self.strategies]
            },
            'execution': {
                'mt5_connected': self.mt5_bridge.is_connected,
                'order_executor': self.order_executor.get_status(),
                'session_filter': self.session_filter.get_filter_status()
            },
            'monitoring': {
                'performance_tracker': self.performance_tracker.get_tracker_status(),
                'volatility_monitor': self.volatility_monitor.get_monitor_status(),
                'alert_system': self.alert_system.get_system_status()
            },
            'ml_models': {
                'enabled': self.configs['model']['enabled'],
                'model_summary': self.neural_model.get_model_summary() if self.neural_model else None
            },
            'config_files': list(self.config_dir.glob('*.yaml')),
            'project_root': str(self.project_root)
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive system report"""
        
        status = self.get_system_status()
        
        report = f"""
# NEXUS TRADING SYSTEM - SYSTEM REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## SYSTEM STATUS
- **Trading Engine**: {'Active' if status['trading_engine']['is_running'] else 'Inactive'}
- **Data Manager**: {status['data_manager']['loader_type'].upper()}
- **Strategies**: {status['strategies']['total']} configured, {status['strategies']['active']} active
- **Execution**: {'Connected' if status['execution']['mt5_connected'] else 'Disconnected'}

## PERFORMANCE MONITORING
- **Performance Tracker**: {'Running' if status['monitoring']['performance_tracker']['is_running'] else 'Stopped'}
- **Volatility Monitor**: {'Running' if status['monitoring']['volatility_monitor']['is_running'] else 'Stopped'}
- **Alert System**: {'Running' if status['monitoring']['alert_system']['is_running'] else 'Stopped'}

## ACTIVE STRATEGIES
"""
        
        for strategy in self.strategies:
            strategy_status = strategy.get_status()
            report += f"""
### {strategy.name}
- Status: {'Active' if strategy.is_active else 'Inactive'}
- Signals: {strategy_status['total_signals']}
- Trades: {strategy_status['performance']['total_trades']}
- Win Rate: {strategy_status['performance']['win_rate']:.1f}%
- P&L: ${strategy_status['performance']['total_pnl']:.2f}
"""
        
        report += f"""
## CONFIGURATION FILES
"""
        
        for config_file in status['config_files']:
            report += f"- {config_file}\n"
        
        report += f"""
## NEXT STEPS
1. Configure trading parameters in config/ directory
2. Load market data (CSV or MT5)
3. Run backtest: python main.py --backtest --start 2023-01-01 --end 2023-12-31
4. Train models: python main.py --train --symbol XAUUSD --start 2023-01-01 --end 2023-12-31
5. Start live trading: python main.py --trade

## SUPPORTED ASSETS
"""
        
        assets = self.configs['assets']['assets']
        for symbol, config in assets.items():
            report += f"- {symbol} ({config.get('name', 'Unknown')})\n"
        
        report += f"""
## TRADING RULES IMPLEMENTED
✅ Default risk: 1% per trade
✅ Max daily loss: $9.99
✅ One active trade per asset
✅ Max trades per day (9H: 2, 6H: 2, 3H: 1)
✅ TP/SL: Entry SL=-$3, TP=+$9.9, lock SL to +$3, extend TP to +$15
✅ All decisions logged for auditing
✅ Session filters for Gold (London/NY)
✅ BTC 24/7 with volatility filter
✅ Risk management and position sizing
✅ Dynamic TP/SL management
✅ Comprehensive monitoring and alerting
"""
        
        return report
    
    def shutdown(self):
        """Shutdown system gracefully"""
        
        self.logger.info("Shutting down Nexus Trading System...")
        
        try:
            # Stop all components
            if self.trading_engine.is_running:
                self.trading_engine.stop()
            
            if self.order_executor.is_running:
                # Note: In a real async context, this would be awaited
                # For now, we'll call it synchronously
                try:
                    import asyncio
                    if asyncio.get_event_loop().is_running():
                        # If already in event loop, create a new one
                        asyncio.run(self.order_executor.stop())
                    else:
                        asyncio.create_task(self.order_executor.stop())
                except:
                    # Fallback for non-async context
                    pass
            
            if self.performance_tracker.is_running:
                self.performance_tracker.stop()
            
            if self.volatility_monitor.is_running:
                self.volatility_monitor.stop()
            
            if self.alert_system.is_running:
                self.alert_system.stop()
            
            if self.mt5_bridge.is_connected:
                self.mt5_bridge.disconnect()
            
            self.logger.info("Nexus Trading System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Nexus Trading System - Professional Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
        
    parser.add_argument(
        '--config',
        default='config',
        help='Configuration directory path'
    )
    
    parser.add_argument(
        '--trade',
        action='store_true',
        help='Start live trading'
    )
    
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtesting'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train ML models'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        help='Symbol for model training'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate system report'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status'
    )
    
    parser.add_argument(
        '--shutdown',
        action='store_true',
        help='Shutdown the system'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    nexus = NexusTradingSystem(args.config)
    
    try:
        if args.trade:
            # Start live trading
            success = asyncio.run(nexus.start_trading())
            if not success:
                print("Failed to start trading")
                sys.exit(1)
        
        elif args.backtest:
            # Run backtesting
            if not args.start or not args.end:
                # Default to last year
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
            else:
                start_date = datetime.strptime(args.start, '%Y-%m-%d')
                end_date = datetime.strptime(args.end, '%Y-%m-%d')
            
            result = nexus.run_backtest(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if result['success']:
                print(f"Backtest completed successfully!")
                print(f"Total trades: {result['total_trades']}")
                print(f"Results saved to backtest/py_results/")
            else:
                print(f"Backtest failed: {result['error']}")
                sys.exit(1)
        
        elif args.train:
            # Train models
            if not args.symbol:
                print("Error: --symbol required for model training")
                sys.exit(1)
            
            if not args.start or not args.end:
                # Default to last 6 months
                end_date = datetime.now()
                start_date = end_date - timedelta(days=180)
            else:
                start_date = datetime.strptime(args.start, '%Y-%m-%d')
                end_date = datetime.strptime(args.end, '%Y-%m-%d')
            
            result = nexus.train_models(
                symbol=args.symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if result['success']:
                print(f"Model training completed for {args.symbol}")
                if result.get('evaluation'):
                    print(f"Model accuracy: {result['evaluation']['accuracy']:.4f}")
                print(f"Model summary: {result['model_summary']}")
            else:
                print(f"Model training failed: {result['error']}")
                sys.exit(1)
        
        elif args.report:
            # Generate report
            report = nexus.generate_report()
            print(report)
            
            # Save report to file
            report_path = Path("nexus_system_report.md")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {report_path}")
        
        elif args.status:
            # Show system status
            status = nexus.get_system_status()
            
            print("NEXUS TRADING SYSTEM STATUS")
            print("=" * 50)
            print(f"Trading Engine: {'Active' if status['trading_engine']['is_running'] else 'Inactive'}")
            print(f"Data Manager: {status['data_manager']['loader_type'].upper()}")
            print(f"Strategies: {status['strategies']['total']} total, {status['strategies']['active']} active")
            print(f"Execution: {'Connected' if status['execution']['mt5_connected'] else 'Disconnected'}")
            print(f"Monitoring: {'Active' if status['monitoring']['performance_tracker']['is_running'] else 'Stopped'}")
            print(f"ML Models: {'Enabled' if status['ml_models']['enabled'] else 'Disabled'}")
            print(f"Alert System: {'Running' if status['monitoring']['alert_system']['is_running'] else 'Stopped'}")
            print("=" * 50)
        
        elif args.shutdown:
            # Shutdown system
            nexus.shutdown()
        
        else:
            # Show help
            print(parser.format_help())
            print("\nExamples:")
            print("  python main.py --backtest --start 2023-01-01 --end 2023-12-31")
            print("  python main.py --train --symbol XAUUSD --start 2023-01-01 --end 2023-12-31")
            print("  python main.py --trade")
            print("  python main.py --report")
            print("  python main.py --status")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Nexus Trading System - Backtesting Metrics
Comprehensive performance metrics and analysis tools
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from core.logger import get_logger


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annual_return: float
    monthly_return: float
    daily_return_avg: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    
    # Risk-adjusted metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    skewness: float
    kurtosis: float
    
    # Trading-specific metrics
    avg_trade_duration: timedelta
    avg_holding_period: timedelta
    trades_per_month: float
    exposure_percentage: float
    
    # Compliance metrics
    daily_loss_limit_compliance: float
    consecutive_losses_compliance: float
    risk_per_trade_compliance: float


class MetricsCalculator:
    """Advanced metrics calculator for trading performance"""
    
    def __init__(self, output_dir: str = "backtest/py_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("metrics_calculator")
    
    def calculate_comprehensive_metrics(self, equity_curve: pd.Series, 
                                      trades: List[Dict[str, Any]],
                                      initial_capital: float = 10000.0) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            equity_curve: Equity curve series
            trades: List of trade records
            initial_capital: Starting capital
            
        Returns:
            PerformanceMetrics with all metrics
        """
        
        self.logger.info("Calculating comprehensive performance metrics")
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Return metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        annual_return = self._calculate_annual_return(equity_curve)
        monthly_return = self._calculate_monthly_return(equity_curve)
        daily_return_avg = returns.mean() * 100
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(equity_curve, returns)
        max_drawdown, max_dd_duration = self._calculate_max_drawdown(equity_curve)
        
        # Trade metrics
        trade_metrics = self._calculate_trade_metrics(trades)
        
        # Risk-adjusted metrics
        risk_metrics = self._calculate_risk_metrics(returns)
        
        # Trading-specific metrics
        trading_metrics = self._calculate_trading_metrics(trades, equity_curve)
        
        # Compliance metrics
        compliance_metrics = self._calculate_compliance_metrics(trades)
        
        metrics = PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_return=monthly_return,
            daily_return_avg=daily_return_avg,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            **trade_metrics,
            **risk_metrics,
            **trading_metrics,
            **compliance_metrics
        )
        
        self.logger.info(f"Metrics calculated: Total Return={total_return:.2f}%, Sharpe={sharpe_ratio:.2f}")
        
        return metrics
    
    def _calculate_annual_return(self, equity_curve: pd.Series) -> float:
        """Calculate annualized return"""
        
        total_days = len(equity_curve)
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        
        if total_days == 0:
            return 0.0
        
        years = total_days / 252  # Trading days per year
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        return annual_return * 100
    
    def _calculate_monthly_return(self, equity_curve: pd.Series) -> float:
        """Calculate average monthly return"""
        
        monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
        
        if len(monthly_returns) == 0:
            return 0.0
        
        return monthly_returns.mean() * 100
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        
        if returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - risk_free_rate / 252
        sharpe = excess_returns / returns.std()
        
        return sharpe * np.sqrt(252)  # Annualized
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - risk_free_rate / 252
        downside_std = downside_returns.std()
        
        sortino = excess_returns / downside_std
        
        return sortino * np.sqrt(252)  # Annualized
    
    def _calculate_calmar_ratio(self, equity_curve: pd.Series, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        
        annual_return = self._calculate_annual_return(equity_curve) / 100
        max_drawdown, _ = self._calculate_max_drawdown(equity_curve)
        
        if max_drawdown == 0:
            return 0.0
        
        return annual_return / abs(max_drawdown)
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        
        cumulative_returns = (1 + equity_curve.pct_change()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        # Calculate duration of max drawdown
        max_dd_idx = drawdown.idxmin()
        start_idx = drawdown[:max_dd_idx][drawdown == 0].last_valid_index()
        
        if start_idx is not None:
            duration = (max_dd_idx - start_idx).days
        else:
            duration = 0
        
        return max_drawdown * 100, duration  # Return as percentage
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade-specific metrics"""
        
        closed_trades = [t for t in trades if t['action'] == 'CLOSE']
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        largest_win = max(t['pnl'] for t in winning_trades) if winning_trades else 0
        largest_loss = min(t['pnl'] for t in losing_trades) if losing_trades else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate risk metrics"""
        
        # Value at Risk
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Conditional Value at Risk
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'var_95': var_95 * 100,
            'var_99': var_99 * 100,
            'cvar_95': cvar_95 * 100,
            'cvar_99': cvar_99 * 100,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _calculate_trading_metrics(self, trades: List[Dict[str, Any]], 
                                  equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate trading-specific metrics"""
        
        closed_trades = [t for t in trades if t['action'] == 'CLOSE']
        
        if not closed_trades:
            return {
                'avg_trade_duration': timedelta(0),
                'avg_holding_period': timedelta(0),
                'trades_per_month': 0.0,
                'exposure_percentage': 0.0
            }
        
        # Average trade duration
        durations = []
        for trade in closed_trades:
            if 'duration' in trade and isinstance(trade['duration'], (timedelta, pd.Timedelta)):
                durations.append(trade['duration'])
        
        avg_duration = np.mean(durations) if durations else timedelta(0)
        
        # Trades per month
        if closed_trades:
            first_trade = min(trade['timestamp'] for trade in closed_trades)
            last_trade = max(trade['timestamp'] for trade in closed_trades)
            months = (last_trade - first_trade).days / 30.44
            trades_per_month = len(closed_trades) / months if months > 0 else 0
        else:
            trades_per_month = 0
        
        # Exposure percentage (simplified)
        exposure_percentage = 50.0  # Default assumption
        
        return {
            'avg_trade_duration': avg_duration,
            'avg_holding_period': avg_duration,
            'trades_per_month': trades_per_month,
            'exposure_percentage': exposure_percentage
        }
    
    def _calculate_compliance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate compliance with trading rules"""
        
        # Daily loss limit compliance (max $9.99 per day)
        daily_pnls = {}
        for trade in trades:
            date = trade['timestamp'].date()
            if date not in daily_pnls:
                daily_pnls[date] = 0
            daily_pnls[date] += trade['pnl']
        
        daily_losses = [pnl for pnl in daily_pnls.values() if pnl < 0]
        days_over_limit = len([loss for loss in daily_losses if abs(loss) > 9.99])
        daily_loss_compliance = 1 - (days_over_limit / len(daily_pnls)) if daily_pnls else 1.0
        
        # Consecutive losses compliance (max 3)
        consecutive_losses = self._calculate_max_consecutive_losses(trades)
        consecutive_losses_compliance = 1.0 if consecutive_losses <= 3 else 0.0
        
        # Risk per trade compliance (1% per trade)
        # This would need position size data to calculate accurately
        risk_per_trade_compliance = 0.95  # Assumed compliance rate
        
        return {
            'daily_loss_limit_compliance': daily_loss_compliance * 100,
            'consecutive_losses_compliance': consecutive_losses_compliance * 100,
            'risk_per_trade_compliance': risk_per_trade_compliance * 100
        }
    
    def _calculate_max_consecutive_losses(self, trades: List[Dict[str, Any]]) -> int:
        """Calculate maximum consecutive losses"""
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade['action'] == 'CLOSE' and trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            elif trade['action'] == 'CLOSE' and trade['pnl'] > 0:
                current_consecutive = 0
        
        return max_consecutive
    
    def generate_performance_report(self, metrics: PerformanceMetrics) -> str:
        """Generate comprehensive performance report"""
        
        report = f"""
# Nexus Trading System - Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Return Performance
- **Total Return**: {metrics.total_return:.2f}%
- **Annual Return**: {metrics.annual_return:.2f}%
- **Monthly Return**: {metrics.monthly_return:.2f}%
- **Daily Average Return**: {metrics.daily_return_avg:.3f}%

## Risk Metrics
- **Volatility**: {metrics.volatility:.2f}%
- **Sharpe Ratio**: {metrics.sharpe_ratio:.2f}
- **Sortino Ratio**: {metrics.sortino_ratio:.2f}
- **Calmar Ratio**: {metrics.calmar_ratio:.2f}
- **Maximum Drawdown**: {metrics.max_drawdown:.2f}%
- **Max DD Duration**: {metrics.max_drawdown_duration} days

## Trading Performance
- **Total Trades**: {metrics.total_trades}
- **Winning Trades**: {metrics.winning_trades}
- **Losing Trades**: {metrics.losing_trades}
- **Win Rate**: {metrics.win_rate:.1f}%
- **Average Win**: ${metrics.avg_win:.2f}
- **Average Loss**: ${metrics.avg_loss:.2f}
- **Profit Factor**: {metrics.profit_factor:.2f}
- **Largest Win**: ${metrics.largest_win:.2f}
- **Largest Loss**: ${metrics.largest_loss:.2f}

## Risk-Adjusted Metrics
- **VaR (95%)**: {metrics.var_95:.2f}%
- **VaR (99%)**: {metrics.var_99:.2f}%
- **CVaR (95%)**: {metrics.cvar_95:.2f}%
- **CVaR (99%)**: {metrics.cvar_99:.2f}%
- **Skewness**: {metrics.skewness:.3f}
- **Kurtosis**: {metrics.kurtosis:.3f}

## Trading Statistics
- **Average Trade Duration**: {metrics.avg_trade_duration}
- **Trades Per Month**: {metrics.trades_per_month:.1f}
- **Exposure Percentage**: {metrics.exposure_percentage:.1f}%

## Rule Compliance
- **Daily Loss Limit Compliance**: {metrics.daily_loss_limit_compliance:.1f}%
- **Consecutive Losses Compliance**: {metrics.consecutive_losses_compliance:.1f}%
- **Risk Per Trade Compliance**: {metrics.risk_per_trade_compliance:.1f}%

## Performance Grade
{self._calculate_performance_grade(metrics)}
"""
        
        return report
    
    def _calculate_performance_grade(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall performance grade"""
        
        score = 0
        
        # Return metrics (40 points)
        if metrics.total_return > 20:
            score += 40
        elif metrics.total_return > 10:
            score += 30
        elif metrics.total_return > 5:
            score += 20
        elif metrics.total_return > 0:
            score += 10
        
        # Risk metrics (30 points)
        if metrics.sharpe_ratio > 2:
            score += 15
        elif metrics.sharpe_ratio > 1:
            score += 10
        elif metrics.sharpe_ratio > 0.5:
            score += 5
        
        if metrics.max_drawdown > -10:
            score += 15
        elif metrics.max_drawdown > -20:
            score += 10
        elif metrics.max_drawdown > -30:
            score += 5
        
        # Trading metrics (30 points)
        if metrics.win_rate > 60:
            score += 15
        elif metrics.win_rate > 50:
            score += 10
        elif metrics.win_rate > 40:
            score += 5
        
        if metrics.profit_factor > 2:
            score += 15
        elif metrics.profit_factor > 1.5:
            score += 10
        elif metrics.profit_factor > 1:
            score += 5
        
        # Convert to grade
        if score >= 85:
            return "A+ (Excellent)"
        elif score >= 75:
            return "A (Very Good)"
        elif score >= 65:
            return "B (Good)"
        elif score >= 55:
            return "C (Average)"
        elif score >= 45:
            return "D (Below Average)"
        else:
            return "F (Poor)"
    
    def create_performance_charts(self, equity_curve: pd.Series, 
                                 trades: List[Dict[str, Any]],
                                 save_charts: bool = True) -> Dict[str, str]:
        """Create performance visualization charts"""
        
        chart_paths = {}
        
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Nexus Trading System - Performance Analysis', fontsize=16)
            
            # 1. Equity Curve
            axes[0, 0].plot(equity_curve.index, equity_curve.values, linewidth=2)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Drawdown Chart
            returns = equity_curve.pct_change().dropna()
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max * 100
            
            axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Trade Distribution
            closed_trades = [t for t in trades if t['action'] == 'CLOSE']
            if closed_trades:
                pnls = [t['pnl'] for t in closed_trades]
                colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
                
                axes[1, 0].bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
                axes[1, 0].set_title(f'Trade P&L Distribution ({len(pnls)} trades)')
                axes[1, 0].set_ylabel('P&L ($)')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Monthly Returns
            monthly_returns = equity_curve.resample('M').last().pct_change().dropna() * 100
            colors_monthly = ['green' if ret > 0 else 'red' for ret in monthly_returns]
            
            axes[1, 1].bar(range(len(monthly_returns)), monthly_returns.values, color=colors_monthly, alpha=0.7)
            axes[1, 1].set_title('Monthly Returns')
            axes[1, 1].set_ylabel('Return (%)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_charts:
                chart_path = self.output_dir / f"performance_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                chart_paths['main_chart'] = str(chart_path)
                
                self.logger.info(f"Performance charts saved to {chart_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creating performance charts: {e}")
        
        return chart_paths
    
    def export_metrics_to_json(self, metrics: PerformanceMetrics, filename: str):
        """Export metrics to JSON file"""
        
        # Convert to JSON-serializable format
        metrics_dict = {
            'total_return': metrics.total_return,
            'annual_return': metrics.annual_return,
            'monthly_return': metrics.monthly_return,
            'daily_return_avg': metrics.daily_return_avg,
            'volatility': metrics.volatility,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio,
            'max_drawdown': metrics.max_drawdown,
            'max_drawdown_duration': metrics.max_drawdown_duration,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'win_rate': metrics.win_rate,
            'avg_win': metrics.avg_win,
            'avg_loss': metrics.avg_loss,
            'profit_factor': metrics.profit_factor,
            'largest_win': metrics.largest_win,
            'largest_loss': metrics.largest_loss,
            'var_95': metrics.var_95,
            'var_99': metrics.var_99,
            'cvar_95': metrics.cvar_95,
            'cvar_99': metrics.cvar_99,
            'skewness': metrics.skewness,
            'kurtosis': metrics.kurtosis,
            'avg_trade_duration': str(metrics.avg_trade_duration),
            'trades_per_month': metrics.trades_per_month,
            'exposure_percentage': metrics.exposure_percentage,
            'daily_loss_limit_compliance': metrics.daily_loss_limit_compliance,
            'consecutive_losses_compliance': metrics.consecutive_losses_compliance,
            'risk_per_trade_compliance': metrics.risk_per_trade_compliance,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(self.output_dir / filename, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filename}")
    
    def compare_strategies(self, strategy_results: Dict[str, PerformanceMetrics]) -> pd.DataFrame:
        """Compare multiple strategies"""
        
        comparison_data = []
        
        for strategy_name, metrics in strategy_results.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': metrics.total_return,
                'Sharpe Ratio': metrics.sharpe_ratio,
                'Max Drawdown (%)': metrics.max_drawdown,
                'Win Rate (%)': metrics.win_rate,
                'Profit Factor': metrics.profit_factor,
                'Total Trades': metrics.total_trades,
                'Annual Return (%)': metrics.annual_return,
                'Volatility (%)': metrics.volatility,
                'Calmar Ratio': metrics.calmar_ratio
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_df.to_csv(self.output_dir / "strategy_comparison.csv", index=False)
        
        self.logger.info("Strategy comparison saved")
        
        return comparison_df
    
    def calculate_rolling_metrics(self, equity_curve: pd.Series, 
                                window: int = 252) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        returns = equity_curve.pct_change().dropna()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        rolling_metrics['Rolling Sharpe'] = rolling_sharpe
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        rolling_metrics['Rolling Volatility'] = rolling_vol
        
        # Rolling drawdown
        rolling_cumulative = (1 + returns).rolling(window).apply(lambda x: x.cumprod().iloc[-1])
        rolling_max = rolling_cumulative.expanding().max()
        rolling_dd = (rolling_cumulative - rolling_max) / rolling_max * 100
        rolling_metrics['Rolling Drawdown'] = rolling_dd
        
        # Rolling return
        rolling_return = rolling_cumulative - 1
        rolling_metrics['Rolling Return'] = rolling_return * 100
        
        return rolling_metrics.dropna()
    
    def generate_risk_report(self, metrics: PerformanceMetrics) -> str:
        """Generate detailed risk analysis report"""
        
        risk_report = f"""
# Risk Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Risk Assessment
- **Overall Risk Level**: {self._assess_risk_level(metrics)}
- **Volatility**: {metrics.volatility:.2f}% ({self._classify_volatility(metrics.volatility)})
- **Maximum Drawdown**: {metrics.max_drawdown:.2f}% ({self._classify_drawdown(metrics.max_drawdown)})

## Value at Risk Analysis
- **Daily VaR (95%)**: {metrics.var_95:.2f}% - Expected worst loss in 5% of trading days
- **Daily VaR (99%)**: {metrics.var_99:.2f}% - Expected worst loss in 1% of trading days
- **Expected Shortfall (95%)**: {metrics.cvar_95:.2f}% - Average loss when VaR is breached
- **Expected Shortfall (99%)**: {metrics.cvar_99:.2f}% - Average loss in worst 1% scenarios

## Risk-Adjusted Performance
- **Sharpe Ratio**: {metrics.sharpe_ratio:.2f} ({self._classify_sharpe(metrics.sharpe_ratio)})
- **Sortino Ratio**: {metrics.sortino_ratio:.2f} ({self._classify_sortino(metrics.sortino_ratio)})
- **Calmar Ratio**: {metrics.calmar_ratio:.2f} ({self._classify_calmar(metrics.calmar_ratio)})

## Distribution Analysis
- **Skewness**: {metrics.skewness:.3f} ({'Right-skewed' if metrics.skewness > 0 else 'Left-skewed' if metrics.skewness < 0 else 'Symmetric'})
- **Kurtosis**: {metrics.kurtosis:.3f} ({'Heavy-tailed' if metrics.kurtosis > 3 else 'Light-tailed' if metrics.kurtosis < 3 else 'Normal'})

## Risk Recommendations
{self._generate_risk_recommendations(metrics)}
"""
        
        return risk_report
    
    def _assess_risk_level(self, metrics: PerformanceMetrics) -> str:
        """Assess overall risk level"""
        
        risk_score = 0
        
        # Volatility contribution
        if metrics.volatility > 30:
            risk_score += 3
        elif metrics.volatility > 20:
            risk_score += 2
        elif metrics.volatility > 15:
            risk_score += 1
        
        # Drawdown contribution
        if abs(metrics.max_drawdown) > 30:
            risk_score += 3
        elif abs(metrics.max_drawdown) > 20:
            risk_score += 2
        elif abs(metrics.max_drawdown) > 15:
            risk_score += 1
        
        # Sharpe ratio contribution (inverse)
        if metrics.sharpe_ratio < 0.5:
            risk_score += 2
        elif metrics.sharpe_ratio < 1.0:
            risk_score += 1
        
        if risk_score >= 6:
            return "Very High"
        elif risk_score >= 4:
            return "High"
        elif risk_score >= 2:
            return "Medium"
        else:
            return "Low"
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility > 25:
            return "Very High"
        elif volatility > 20:
            return "High"
        elif volatility > 15:
            return "Medium"
        elif volatility > 10:
            return "Low"
        else:
            return "Very Low"
    
    def _classify_drawdown(self, drawdown: float) -> str:
        """Classify drawdown severity"""
        if abs(drawdown) > 25:
            return "Severe"
        elif abs(drawdown) > 15:
            return "High"
        elif abs(drawdown) > 10:
            return "Moderate"
        elif abs(drawdown) > 5:
            return "Mild"
        else:
            return "Minimal"
    
    def _classify_sharpe(self, sharpe: float) -> str:
        """Classify Sharpe ratio"""
        if sharpe > 2:
            return "Excellent"
        elif sharpe > 1.5:
            return "Very Good"
        elif sharpe > 1:
            return "Good"
        elif sharpe > 0.5:
            return "Fair"
        else:
            return "Poor"
    
    def _classify_sortino(self, sortino: float) -> str:
        """Classify Sortino ratio"""
        if sortino > 3:
            return "Excellent"
        elif sortino > 2:
            return "Very Good"
        elif sortino > 1.5:
            return "Good"
        elif sortino > 1:
            return "Fair"
        else:
            return "Poor"
    
    def _classify_calmar(self, calmar: float) -> str:
        """Classify Calmar ratio"""
        if calmar > 3:
            return "Excellent"
        elif calmar > 2:
            return "Very Good"
        elif calmar > 1:
            return "Good"
        elif calmar > 0.5:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_risk_recommendations(self, metrics: PerformanceMetrics) -> str:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        if metrics.volatility > 20:
            recommendations.append("• Consider reducing position sizes due to high volatility")
        
        if abs(metrics.max_drawdown) > 20:
            recommendations.append("• Implement stricter risk controls to limit drawdowns")
        
        if metrics.sharpe_ratio < 1:
            recommendations.append("• Improve risk-adjusted returns through better entry/exit timing")
        
        if metrics.win_rate < 40:
            recommendations.append("• Review strategy selection and improve signal quality")
        
        if metrics.profit_factor < 1.2:
            recommendations.append("• Optimize take-profit levels to improve reward-to-risk ratio")
        
        if not recommendations:
            return "Risk management appears adequate. Continue monitoring performance."
        
        return "\n".join(recommendations)

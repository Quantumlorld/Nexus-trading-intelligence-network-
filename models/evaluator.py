"""
Nexus Trading System - Model Evaluator
Comprehensive model evaluation and performance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)

from .neural_model import NeuralTradingModel
from .trainer import TrainingResult
from core.logger import get_logger


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    specificity: float
    matthews_corrcoef: float
    balanced_accuracy: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_return: float
    total_return: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc': self.auc,
            'specificity': self.specificity,
            'matthews_corrcoef': self.matthews_corrcoef,
            'balanced_accuracy': self.balanced_accuracy,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'classification_report': self.classification_report,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'avg_trade_return': self.avg_trade_return,
            'total_return': self.total_return
        }


@dataclass
class BacktestResult:
    """Results from backtesting model predictions"""
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: EvaluationMetrics
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int


class ModelEvaluator:
    """Comprehensive model evaluation system"""
    
    def __init__(self, output_dir: str = "models/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("model_evaluator")
        
        # Evaluation history
        self.evaluation_history: List[Dict[str, Any]] = []
        
        self.logger.info("Model evaluator initialized")
    
    def evaluate_model(self, model: NeuralTradingModel, 
                      test_features: pd.DataFrame, test_targets: pd.Series,
                      price_data: Optional[pd.DataFrame] = None,
                      model_name: str = "model") -> EvaluationMetrics:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained neural network model
            test_features: Test feature data
            test_targets: Test target data
            price_data: Price data for trading simulation
            model_name: Name of the model being evaluated
            
        Returns:
            EvaluationMetrics with comprehensive metrics
        """
        
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Get predictions
        predictions = model.predict(test_features)
        predictions_proba = predictions  # For binary classification
        
        # Convert to binary predictions
        binary_predictions = (predictions_proba > 0.5).astype(int)
        
        # Calculate classification metrics
        classification_metrics = self._calculate_classification_metrics(
            test_targets.values, binary_predictions, predictions_proba
        )
        
        # Calculate trading metrics if price data provided
        if price_data is not None:
            trading_metrics = self._calculate_trading_metrics(
                predictions_proba, test_targets, price_data
            )
            classification_metrics.update(trading_metrics)
        else:
            # Default trading metrics
            classification_metrics.update({
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_trade_return': 0.0,
                'total_return': 0.0
            })
        
        # Create metrics object
        metrics = EvaluationMetrics(**classification_metrics)
        
        # Save evaluation results
        self._save_evaluation_results(metrics, model_name)
        
        self.logger.info(f"Model evaluation completed: Accuracy={metrics.accuracy:.4f}")
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # AUC
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity (true negative rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Matthews correlation coefficient
        mcc = self._calculate_matthews_corrcoef(y_true, y_pred)
        
        # Balanced accuracy
        balanced_acc = (recall + specificity) / 2
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'specificity': specificity,
            'matthews_corrcoef': mcc,
            'balanced_accuracy': balanced_acc,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
    
    def _calculate_matthews_corrcoef(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Matthews correlation coefficient"""
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_trading_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                                 price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading-specific metrics"""
        
        # Align data
        min_length = min(len(predictions), len(targets), len(price_data))
        predictions = predictions[:min_length]
        targets = targets[:min_length]
        price_data = price_data.iloc[:min_length]
        
        # Simulate trading
        returns = self._simulate_trading(predictions, price_data)
        
        if len(returns) == 0:
            return {
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_trade_return': 0.0,
                'total_return': 0.0
            }
        
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        win_rate = (returns > 0).mean()
        avg_trade_return = returns.mean()
        
        # Sharpe ratio
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Profit factor
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        if len(losing_returns) > 0:
            profit_factor = winning_returns.sum() / abs(losing_returns.sum())
        else:
            profit_factor = float('inf') if len(winning_returns) > 0 else 0
        
        return {
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'total_return': total_return
        }
    
    def _simulate_trading(self, predictions: np.ndarray, price_data: pd.DataFrame) -> np.ndarray:
        """Simulate trading based on model predictions"""
        
        returns = []
        position = 0  # 0 = no position, 1 = long, -1 = short
        
        for i in range(len(predictions)):
            pred = predictions[i]
            current_price = price_data['close'].iloc[i]
            
            # Trading logic
            if pred > 0.6 and position == 0:  # Strong buy signal
                position = 1
                entry_price = current_price
                
            elif pred < 0.4 and position == 0:  # Strong sell signal
                position = -1
                entry_price = current_price
                
            elif (0.4 <= pred <= 0.6) and position != 0:  # Neutral signal, close position
                if position == 1:
                    trade_return = (current_price - entry_price) / entry_price
                else:
                    trade_return = (entry_price - current_price) / entry_price
                
                returns.append(trade_return)
                position = 0
        
        return np.array(returns)
    
    def backtest_model(self, model: NeuralTradingModel, 
                      features: pd.DataFrame, targets: pd.Series,
                      price_data: pd.DataFrame, 
                      initial_capital: float = 10000.0,
                      model_name: str = "model") -> BacktestResult:
        """
        Comprehensive backtesting of model predictions
        
        Args:
            model: Trained neural network model
            features: Feature data
            targets: Target data
            price_data: Price data for backtesting
            initial_capital: Starting capital for simulation
            model_name: Name of the model
            
        Returns:
            BacktestResult with detailed backtesting results
        """
        
        self.logger.info(f"Backtesting model: {model_name}")
        
        # Get predictions
        predictions = model.predict(features)
        
        # Simulate trading
        equity_curve, returns, trades = self._run_backtest(
            predictions, price_data, initial_capital
        )
        
        # Calculate metrics
        metrics = self.evaluate_model(model, features, targets, price_data, model_name)
        
        # Create result
        result = BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            metrics=metrics,
            start_date=price_data.index[0],
            end_date=price_data.index[-1],
            total_trades=len(trades),
            winning_trades=len(trades[trades['pnl'] > 0]),
            losing_trades=len(trades[trades['pnl'] < 0])
        )
        
        # Save backtest results
        self._save_backtest_results(result, model_name)
        
        self.logger.info(f"Backtest completed: Total return={metrics.total_return:.2%}")
        
        return result
    
    def _run_backtest(self, predictions: np.ndarray, price_data: pd.DataFrame,
                     initial_capital: float) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """Run detailed backtesting simulation"""
        
        # Align data
        min_length = min(len(predictions), len(price_data))
        predictions = predictions[:min_length]
        price_data = price_data.iloc[:min_length].copy()
        
        # Initialize tracking
        equity = [initial_capital]
        returns = []
        trades = []
        
        position = 0
        entry_price = 0
        entry_time = None
        shares = 0
        
        for i in range(len(predictions)):
            current_price = price_data['close'].iloc[i]
            current_time = price_data.index[i]
            pred = predictions[i]
            
            # Position sizing (simplified - 1% risk per trade)
            position_size = initial_capital * 0.01
            
            # Trading logic
            if pred > 0.6 and position == 0:  # Enter long
                position = 1
                entry_price = current_price
                entry_time = current_time
                shares = position_size / current_price
                
            elif pred < 0.4 and position == 0:  # Enter short
                position = -1
                entry_price = current_price
                entry_time = current_time
                shares = position_size / current_price
                
            elif (0.4 <= pred <= 0.6) and position != 0:  # Exit position
                if position == 1:
                    exit_price = current_price
                    pnl = (exit_price - entry_price) * shares
                else:
                    exit_price = current_price
                    pnl = (entry_price - exit_price) * shares
                
                # Record trade
                trade = {
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'direction': 'long' if position == 1 else 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl': pnl,
                    'return': pnl / position_size
                }
                trades.append(trade)
                
                # Update equity
                new_equity = equity[-1] + pnl
                equity.append(new_equity)
                returns.append(pnl / position_size)
                
                # Reset position
                position = 0
                entry_price = 0
                entry_time = None
                shares = 0
            
            else:
                # No trade, equity stays the same
                equity.append(equity[-1])
        
        # Convert to Series
        equity_curve = pd.Series(equity, index=[price_data.index[0]] + list(price_data.index[:len(equity)-1]))
        returns_series = pd.Series(returns)
        trades_df = pd.DataFrame(trades)
        
        return equity_curve, returns_series, trades_df
    
    def compare_models(self, models: Dict[str, NeuralTradingModel],
                       test_features: pd.DataFrame, test_targets: pd.Series,
                       price_data: Optional[pd.DataFrame] = None) -> Dict[str, EvaluationMetrics]:
        """Compare multiple models"""
        
        self.logger.info(f"Comparing {len(models)} models")
        
        results = {}
        
        for model_name, model in models.items():
            try:
                metrics = self.evaluate_model(
                    model, test_features, test_targets, price_data, model_name
                )
                results[model_name] = metrics
                
                self.logger.info(f"Evaluated {model_name}: Accuracy={metrics.accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {e}")
        
        # Save comparison results
        self._save_comparison_results(results)
        
        return results
    
    def _save_evaluation_results(self, metrics: EvaluationMetrics, model_name: str):
        """Save evaluation results to file"""
        
        results_file = self.output_dir / f"{model_name}_evaluation.json"
        
        with open(results_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        self.logger.debug(f"Evaluation results saved to {results_file}")
    
    def _save_backtest_results(self, result: BacktestResult, model_name: str):
        """Save backtest results to files"""
        
        backtest_dir = self.output_dir / f"{model_name}_backtest"
        backtest_dir.mkdir(exist_ok=True)
        
        # Save equity curve
        result.equity_curve.to_csv(backtest_dir / "equity_curve.csv")
        
        # Save returns
        result.returns.to_csv(backtest_dir / "returns.csv")
        
        # Save trades
        result.trades.to_csv(backtest_dir / "trades.csv", index=False)
        
        # Save metrics
        with open(backtest_dir / "metrics.json", 'w') as f:
            json.dump(result.metrics.to_dict(), f, indent=2)
        
        # Save summary
        summary = {
            'model_name': model_name,
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'total_return': result.metrics.total_return,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'max_drawdown': result.metrics.max_drawdown,
            'win_rate': result.metrics.win_rate
        }
        
        with open(backtest_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Backtest results saved to {backtest_dir}")
    
    def _save_comparison_results(self, results: Dict[str, EvaluationMetrics]):
        """Save model comparison results"""
        
        comparison_file = self.output_dir / "model_comparison.json"
        
        comparison_data = {}
        for model_name, metrics in results.items():
            comparison_data[model_name] = metrics.to_dict()
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        self.logger.info(f"Model comparison saved to {comparison_file}")
    
    def generate_evaluation_report(self, model_name: str) -> str:
        """Generate comprehensive evaluation report"""
        
        results_file = self.output_dir / f"{model_name}_evaluation.json"
        
        if not results_file.exists():
            return f"No evaluation results found for {model_name}"
        
        with open(results_file, 'r') as f:
            metrics = json.load(f)
        
        report = f"""
# Model Evaluation Report: {model_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Classification Metrics
- **Accuracy**: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})
- **Precision**: {metrics['precision']:.4f} ({metrics['precision']:.2%})
- **Recall**: {metrics['recall']:.4f} ({metrics['recall']:.2%})
- **F1-Score**: {metrics['f1_score']:.4f} ({metrics['f1_score']:.2%})
- **AUC**: {metrics['auc']:.4f} ({metrics['auc']:.2%})
- **Specificity**: {metrics['specificity']:.4f} ({metrics['specificity']:.2%})
- **Matthews Corr Coef**: {metrics['matthews_corrcoef']:.4f}
- **Balanced Accuracy**: {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']:.2%})

## Trading Metrics
- **Total Return**: {metrics['total_return']:.4f} ({metrics['total_return']:.2%})
- **Sharpe Ratio**: {metrics['sharpe_ratio']:.4f}
- **Maximum Drawdown**: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']:.2%})
- **Win Rate**: {metrics['win_rate']:.4f} ({metrics['win_rate']:.2%})
- **Average Trade Return**: {metrics['avg_trade_return']:.4f} ({metrics['avg_trade_return']:.2%})
- **Profit Factor**: {metrics['profit_factor']:.4f}

## Confusion Matrix
{metrics['confusion_matrix']}

## Classification Report
```
{json.dumps(metrics['classification_report'], indent=2)}
```
"""
        
        # Save report
        report_file = self.output_dir / f"{model_name}_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return report
    
    def plot_evaluation_results(self, model_name: str, save_plots: bool = True):
        """Generate evaluation plots"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Load backtest results
            backtest_dir = self.output_dir / f"{model_name}_backtest"
            
            if not backtest_dir.exists():
                self.logger.warning(f"No backtest results found for {model_name}")
                return
            
            # Load data
            equity_curve = pd.read_csv(backtest_dir / "equity_curve.csv", index_col=0, parse_dates=True)
            returns = pd.read_csv(backtest_dir / "returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
            trades = pd.read_csv(backtest_dir / "trades.csv", parse_dates=['entry_time', 'exit_time'])
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Model Evaluation: {model_name}', fontsize=16)
            
            # Equity curve
            axes[0, 0].plot(equity_curve.index, equity_curve.iloc[:, 0])
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Equity ($)')
            axes[0, 0].grid(True)
            
            # Returns distribution
            axes[0, 1].hist(returns, bins=50, alpha=0.7)
            axes[0, 1].set_title('Returns Distribution')
            axes[0, 1].set_xlabel('Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
            
            # Trade P&L
            if not trades.empty:
                axes[1, 0].bar(range(len(trades)), trades['pnl'])
                axes[1, 0].set_title('Trade P&L')
                axes[1, 0].set_xlabel('Trade Number')
                axes[1, 0].set_ylabel('P&L ($)')
                axes[1, 0].grid(True)
            
            # Cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            axes[1, 1].plot(cumulative_returns.index, cumulative_returns)
            axes[1, 1].set_title('Cumulative Returns')
            axes[1, 1].set_ylabel('Cumulative Return')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_plots:
                plot_file = self.output_dir / f"{model_name}_plots.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Evaluation plots saved to {plot_file}")
            
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Failed to generate plots: {e}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'models_evaluated': list(set(e['model_name'] for e in self.evaluation_history)),
            'evaluation_dates': [e['timestamp'] for e in self.evaluation_history],
            'average_accuracy': 0,
            'best_accuracy': 0,
            'worst_accuracy': 0
        }
        
        if self.evaluation_history:
            accuracies = [e['metrics']['accuracy'] for e in self.evaluation_history]
            summary['average_accuracy'] = np.mean(accuracies)
            summary['best_accuracy'] = max(accuracies)
            summary['worst_accuracy'] = min(accuracies)
        
        return summary

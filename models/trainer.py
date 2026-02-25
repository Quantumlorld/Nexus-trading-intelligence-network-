"""
Nexus Trading System - Model Trainer
Comprehensive training pipeline for neural network models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict

from .neural_model import NeuralTradingModel, ModelConfig
from .feature_engineering import FeatureEngineer, FeatureSet
from core.logger import get_logger


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    cross_validation_folds: int = 5
    walk_forward_enabled: bool = True
    walk_forward_window: int = 252  # Trading days
    retraining_frequency: int = 30  # Days
    ensemble_models: bool = True
    save_best_models: bool = True
    early_stopping: bool = True
    hyperparameter_tuning: bool = False
    performance_threshold: float = 0.6


@dataclass
class TrainingResult:
    """Results from model training"""
    model_id: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    training_time: timedelta
    model_config: ModelConfig
    feature_importance: Optional[Dict[str, float]] = None
    training_history: Optional[Dict[str, List[float]]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ModelTrainer:
    """Advanced model training pipeline"""
    
    def __init__(self, config: TrainingConfig, model_dir: str = "models/saved"):
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("model_trainer")
        
        # Training state
        self.training_results: List[TrainingResult] = []
        self.best_models: Dict[str, TrainingResult] = {}
        
        # Feature engineer
        self.feature_engineer = None
        
        self.logger.info("Model trainer initialized")
    
    def train_model(self, price_data: pd.DataFrame, target_data: pd.Series,
                   model_config: ModelConfig, feature_config: Dict[str, Any],
                   symbol: str = "DEFAULT") -> TrainingResult:
        """
        Train a single model with comprehensive pipeline
        
        Args:
            price_data: OHLCV price data
            target_data: Target variable for supervised learning
            model_config: Neural network configuration
            feature_config: Feature engineering configuration
            symbol: Trading symbol
            
        Returns:
            TrainingResult with metrics and model info
        """
        
        start_time = datetime.now()
        model_id = f"{symbol}_{model_config.architecture}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting training for {model_id}")
        
        try:
            # 1. Feature Engineering
            feature_set = self._create_features(price_data, target_data, feature_config, symbol)
            
            # 2. Data Splitting
            train_data, val_data, test_data = self._split_data(feature_set)
            
            # 3. Model Creation
            model = NeuralTradingModel(model_config, str(self.model_dir / model_id))
            
            # 4. Training
            training_history = self._train_model(model, train_data, val_data)
            
            # 5. Evaluation
            test_metrics = self._evaluate_model(model, test_data)
            
            # 6. Feature Importance
            feature_importance = self._calculate_feature_importance(feature_set, model)
            
            # 7. Save Model
            if self.config.save_best_models:
                self._save_trained_model(model, model_id, feature_set)
            
            # Create result
            training_time = datetime.now() - start_time
            result = TrainingResult(
                model_id=model_id,
                training_metrics=training_history.get('train_metrics', {}),
                validation_metrics=training_history.get('val_metrics', {}),
                test_metrics=test_metrics,
                training_time=training_time,
                model_config=model_config,
                feature_importance=feature_importance,
                training_history=training_history
            )
            
            # Store results
            self.training_results.append(result)
            self._update_best_models(result)
            
            self.logger.info(f"Training completed for {model_id} in {training_time}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed for {model_id}: {e}")
            raise
    
    def _create_features(self, price_data: pd.DataFrame, target_data: pd.Series,
                        feature_config: Dict[str, Any], symbol: str) -> FeatureSet:
        """Create features for training"""
        
        if self.feature_engineer is None:
            self.feature_engineer = FeatureEngineer(feature_config)
        
        feature_set = self.feature_engineer.create_features(price_data, target_data, symbol)
        
        # Feature selection if enabled
        if feature_config.get('feature_selection', False):
            feature_set = self.feature_engineer.select_features(
                feature_set, 
                method=feature_config.get('selection_method', 'correlation'),
                k_best=feature_config.get('k_best', 50)
            )
        
        # Normalization
        feature_set = self.feature_engineer.normalize_features(
            feature_set, 
            method=feature_config.get('normalization', 'standard')
        )
        
        return feature_set
    
    def _split_data(self, feature_set: FeatureSet) -> Tuple[Dict, Dict, Dict]:
        """Split data into train, validation, and test sets"""
        
        total_samples = len(feature_set.features)
        
        # Calculate split indices
        train_end = int(total_samples * self.config.train_split)
        val_end = int(total_samples * (self.config.train_split + self.config.val_split))
        
        # Split features
        train_features = feature_set.features.iloc[:train_end]
        val_features = feature_set.features.iloc[train_end:val_end]
        test_features = feature_set.features.iloc[val_end:]
        
        # Split targets
        if feature_set.target is not None:
            train_targets = feature_set.target.iloc[:train_end]
            val_targets = feature_set.target.iloc[train_end:val_end]
            test_targets = feature_set.target.iloc[val_end:]
        else:
            train_targets = val_targets = test_targets = None
        
        return (
            {'features': train_features, 'targets': train_targets},
            {'features': val_features, 'targets': val_targets},
            {'features': test_features, 'targets': test_targets}
        )
    
    def _train_model(self, model: NeuralTradingModel, train_data: Dict, val_data: Dict) -> Dict:
        """Train the model"""
        
        # Prepare data loaders
        train_loader, val_loader = model.prepare_data(
            train_data['features'], 
            train_data['targets'],
            train_split=1.0  # Already split
        )
        
        # Train the model
        training_history = model.train(train_loader, val_loader)
        
        return training_history
    
    def _evaluate_model(self, model: NeuralTradingModel, test_data: Dict) -> Dict[str, float]:
        """Evaluate model on test data"""
        
        if test_data['targets'] is None:
            self.logger.warning("No test targets available for evaluation")
            return {}
        
        # Prepare test data loader
        test_loader, _ = model.prepare_data(
            test_data['features'],
            test_data['targets'],
            train_split=1.0
        )
        
        # Evaluate
        metrics = model.evaluate(test_loader)
        
        return metrics
    
    def _calculate_feature_importance(self, feature_set: FeatureSet, 
                                    model: NeuralTradingModel) -> Dict[str, float]:
        """Calculate feature importance"""
        
        if self.feature_engineer is None:
            return {}
        
        # Get importance using correlation method
        importance = self.feature_engineer.get_feature_importance(
            feature_set, 
            method='correlation'
        )
        
        return importance.to_dict()
    
    def _save_trained_model(self, model: NeuralTradingModel, model_id: str, 
                           feature_set: FeatureSet):
        """Save trained model and related artifacts"""
        
        model_dir = self.model_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model._save_checkpoint("final_model.pth")
        
        # Save feature engineering state
        if self.feature_engineer:
            with open(model_dir / "feature_engineer.pkl", 'wb') as f:
                pickle.dump(self.feature_engineer, f)
        
        # Save feature names
        with open(model_dir / "feature_names.json", 'w') as f:
            json.dump(feature_set.feature_names, f)
        
        # Save metadata
        metadata = {
            'model_id': model_id,
            'created_at': datetime.now().isoformat(),
            'feature_count': len(feature_set.feature_names),
            'data_points': len(feature_set.features),
            'model_summary': model.get_model_summary()
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_dir}")
    
    def _update_best_models(self, result: TrainingResult):
        """Update best models registry"""
        
        model_type = result.model_config.architecture
        
        # Update if better than current best
        if (model_type not in self.best_models or 
            result.test_metrics.get('accuracy', 0) > 
            self.best_models[model_type].test_metrics.get('accuracy', 0)):
            
            self.best_models[model_type] = result
            self.logger.info(f"New best {model_type} model: {result.model_id}")
    
    def train_ensemble(self, price_data: pd.DataFrame, target_data: pd.Series,
                       model_configs: List[ModelConfig], feature_config: Dict[str, Any],
                       symbol: str = "DEFAULT") -> List[TrainingResult]:
        """Train ensemble of models"""
        
        self.logger.info(f"Training ensemble of {len(model_configs)} models")
        
        results = []
        
        for i, config in enumerate(model_configs):
            try:
                result = self.train_model(price_data, target_data, config, feature_config, symbol)
                results.append(result)
                
                self.logger.info(f"Ensemble model {i+1}/{len(model_configs)} completed")
                
            except Exception as e:
                self.logger.error(f"Ensemble model {i+1} failed: {e}")
        
        # Save ensemble metadata
        if results:
            self._save_ensemble_metadata(results, symbol)
        
        return results
    
    def _save_ensemble_metadata(self, results: List[TrainingResult], symbol: str):
        """Save ensemble metadata"""
        
        ensemble_dir = self.model_dir / f"ensemble_{symbol}"
        ensemble_dir.mkdir(exist_ok=True)
        
        metadata = {
            'symbol': symbol,
            'created_at': datetime.now().isoformat(),
            'model_count': len(results),
            'models': [result.model_id for result in results],
            'average_accuracy': np.mean([r.test_metrics.get('accuracy', 0) for r in results]),
            'best_accuracy': max([r.test_metrics.get('accuracy', 0) for r in results])
        }
        
        with open(ensemble_dir / "ensemble_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def walk_forward_training(self, price_data: pd.DataFrame, target_data: pd.Series,
                            model_config: ModelConfig, feature_config: Dict[str, Any],
                            symbol: str = "DEFAULT") -> List[TrainingResult]:
        """Perform walk-forward training"""
        
        self.logger.info("Starting walk-forward training")
        
        results = []
        window_size = self.config.walk_forward_window
        step_size = window_size // 4  # 25% overlap
        
        total_samples = len(price_data)
        
        for start_idx in range(0, total_samples - window_size, step_size):
            end_idx = start_idx + window_size
            
            if end_idx >= total_samples:
                break
            
            # Get window data
            window_price_data = price_data.iloc[start_idx:end_idx]
            window_target_data = target_data.iloc[start_idx:end_idx]
            
            # Train model on window
            try:
                result = self.train_model(
                    window_price_data, 
                    window_target_data,
                    model_config, 
                    feature_config,
                    f"{symbol}_walkforward_{start_idx}_{end_idx}"
                )
                
                results.append(result)
                
                self.logger.info(f"Walk-forward window {start_idx}-{end_idx} completed")
                
            except Exception as e:
                self.logger.error(f"Walk-forward window {start_idx}-{end_idx} failed: {e}")
        
        self.logger.info(f"Walk-forward training completed: {len(results)} models")
        
        return results
    
    def cross_validation_training(self, price_data: pd.DataFrame, target_data: pd.Series,
                                 model_config: ModelConfig, feature_config: Dict[str, Any],
                                 symbol: str = "DEFAULT") -> List[TrainingResult]:
        """Perform k-fold cross validation"""
        
        self.logger.info(f"Starting {self.config.cross_validation_folds}-fold cross validation")
        
        results = []
        fold_size = len(price_data) // self.config.cross_validation_folds
        
        for fold in range(self.config.cross_validation_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.config.cross_validation_folds - 1 else len(price_data)
            
            # Create validation set
            val_price_data = price_data.iloc[start_idx:end_idx]
            val_target_data = target_data.iloc[start_idx:end_idx]
            
            # Create training set (all other data)
            train_price_data = pd.concat([price_data.iloc[:start_idx], price_data.iloc[end_idx:]])
            train_target_data = pd.concat([target_data.iloc[:start_idx], target_data.iloc[end_idx:]])
            
            try:
                result = self.train_model(
                    train_price_data,
                    train_target_data,
                    model_config,
                    feature_config,
                    f"{symbol}_cv_fold_{fold+1}"
                )
                
                # Evaluate on validation set
                feature_set = self._create_features(val_price_data, val_target_data, feature_config, symbol)
                test_data = {'features': feature_set.features, 'targets': feature_set.target}
                
                # Load model and evaluate
                model = NeuralTradingModel(model_config, str(self.model_dir / result.model_id))
                model.load_model("final_model.pth")
                
                val_metrics = self._evaluate_model(model, test_data)
                result.validation_metrics = val_metrics
                
                results.append(result)
                
                self.logger.info(f"Cross-validation fold {fold+1} completed")
                
            except Exception as e:
                self.logger.error(f"Cross-validation fold {fold+1} failed: {e}")
        
        self.logger.info(f"Cross-validation completed: {len(results)} models")
        
        return results
    
    def hyperparameter_tuning(self, price_data: pd.DataFrame, target_data: pd.Series,
                            base_config: ModelConfig, feature_config: Dict[str, Any],
                            symbol: str = "DEFAULT") -> Tuple[ModelConfig, TrainingResult]:
        """Perform hyperparameter tuning"""
        
        self.logger.info("Starting hyperparameter tuning")
        
        # Define hyperparameter grid
        param_grid = {
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64],
            'hidden_layers': [
                [64, 32],
                [128, 64, 32],
                [256, 128, 64]
            ],
            'dropout_rate': [0.1, 0.2, 0.3]
        }
        
        best_result = None
        best_config = base_config
        best_score = 0
        
        # Grid search (simplified)
        for lr in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for hidden_layers in param_grid['hidden_layers']:
                    for dropout in param_grid['dropout_rate']:
                        
                        # Create config
                        config = ModelConfig(
                            architecture=base_config.architecture,
                            input_size=base_config.input_size,
                            hidden_layers=hidden_layers,
                            dropout_rate=dropout,
                            batch_size=batch_size,
                            learning_rate=lr
                        )
                        
                        try:
                            result = self.train_model(
                                price_data, target_data, config, feature_config,
                                f"{symbol}_tune_{datetime.now().strftime('%H%M%S')}"
                            )
                            
                            score = result.test_metrics.get('accuracy', 0)
                            
                            if score > best_score:
                                best_score = score
                                best_result = result
                                best_config = config
                                
                                self.logger.info(f"New best config found: accuracy={score:.4f}")
                            
                        except Exception as e:
                            self.logger.error(f"Hyperparameter test failed: {e}")
        
        self.logger.info(f"Hyperparameter tuning completed. Best accuracy: {best_score:.4f}")
        
        return best_config, best_result
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        
        if not self.training_results:
            return {'message': 'No training results available'}
        
        # Calculate statistics
        accuracies = [r.test_metrics.get('accuracy', 0) for r in self.training_results]
        training_times = [r.training_time.total_seconds() for r in self.training_results]
        
        summary = {
            'total_models_trained': len(self.training_results),
            'average_accuracy': np.mean(accuracies),
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies),
            'average_training_time': np.mean(training_times),
            'total_training_time': sum(training_times),
            'model_types': list(set(r.model_config.architecture for r in self.training_results)),
            'best_models': {k: v.model_id for k, v in self.best_models.items()},
            'training_date_range': {
                'start': min(r.created_at for r in self.training_results).isoformat(),
                'end': max(r.created_at for r in self.training_results).isoformat()
            }
        }
        
        return summary
    
    def load_model(self, model_id: str) -> NeuralTradingModel:
        """Load a trained model"""
        
        model_dir = self.model_dir / model_id
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load metadata
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create model
        model_config = ModelConfig(**metadata['model_summary'])
        model = NeuralTradingModel(model_config, str(model_dir))
        
        # Load weights
        model.load_model("final_model.pth")
        
        return model
    
    def export_results(self, filepath: str):
        """Export training results to file"""
        
        results_data = []
        for result in self.training_results:
            result_dict = asdict(result)
            result_dict['training_time'] = str(result.training_time)
            result_dict['created_at'] = result.created_at.isoformat()
            results_data.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Training results exported to {filepath}")

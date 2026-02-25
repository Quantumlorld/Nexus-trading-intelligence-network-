"""
Nexus Trading System - Neural Network Models
Advanced neural network models for trading signal generation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for neural network models"""
    architecture: str = "LSTM"
    input_size: int = 50
    hidden_layers: List[int] = None
    dropout_rate: float = 0.2
    activation: str = "relu"
    output_activation: str = "sigmoid"
    sequence_length: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]


class TradingDataset(Dataset):
    """PyTorch dataset for trading data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 20):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Create sequences
        self.sequences, self.sequence_targets = self._create_sequences()
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data"""
        sequences = []
        targets = []
        
        for i in range(len(self.features) - self.sequence_length):
            sequences.append(self.features[i:i + self.sequence_length])
            targets.append(self.targets[i + self.sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.sequence_targets[idx]])
        )


class LSTMModel(nn.Module):
    """LSTM-based neural network for trading"""
    
    def __init__(self, config: ModelConfig):
        super(LSTMModel, self).__init__()
        self.config = config
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        input_size = config.input_size
        for hidden_size in config.hidden_layers:
            self.lstm_layers.append(
                nn.LSTM(input_size, hidden_size, batch_first=True, dropout=config.dropout_rate)
            )
            input_size = hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        
        fc_input_size = config.hidden_layers[-1]
        for i in range(len(config.hidden_layers) - 1):
            self.fc_layers.append(
                nn.Linear(fc_input_size, config.hidden_layers[i + 1])
            )
            fc_input_size = config.hidden_layers[i + 1]
        
        # Output layer
        self.output_layer = nn.Linear(fc_input_size, 1)
        
        # Activation functions
        self.activation = self._get_activation(config.activation)
        self.output_activation = self._get_activation(config.output_activation)
    
    def _get_activation(self, activation_name: str):
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(activation_name, nn.ReLU())
    
    def forward(self, x):
        # Pass through LSTM layers
        lstm_out = x
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)
            lstm_out = self.dropout(lstm_out)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        fc_out = lstm_out
        for fc in self.fc_layers:
            fc_out = self.activation(fc(fc_out))
            fc_out = self.dropout(fc_out)
        
        # Output layer
        output = self.output_layer(fc_out)
        
        # Apply output activation if specified
        if self.output_activation:
            output = self.output_activation(output)
        
        return output


class TransformerModel(nn.Module):
    """Transformer-based model for trading"""
    
    def __init__(self, config: ModelConfig):
        super(TransformerModel, self).__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.input_size, config.hidden_layers[0])
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.hidden_layers[0], config.sequence_length
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_layers[0],
            nhead=8,
            dropout=config.dropout_rate,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=len(config.hidden_layers)
        )
        
        # Output layers
        self.fc = nn.Linear(config.hidden_layers[0], 1)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = x.transpose(0, 1)  # (batch, seq, feature) -> (seq, batch, feature)
        transformer_out = self.transformer(x)
        transformer_out = transformer_out.transpose(0, 1)
        
        # Take the last output
        output = transformer_out[:, -1, :]
        output = self.dropout(output)
        
        # Final output
        output = torch.sigmoid(self.fc(output))
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class EnsembleModel:
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make ensemble prediction"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred * weight)
        
        return sum(predictions)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make prediction with uncertainty estimation"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Calculate mean and std
        predictions_tensor = torch.stack(predictions)
        mean_pred = predictions_tensor.mean(dim=0)
        uncertainty = predictions_tensor.std(dim=0)
        
        return mean_pred, uncertainty


class NeuralTradingModel:
    """Main neural network model for trading signals"""
    
    def __init__(self, config: ModelConfig, model_dir: str = "models/saved"):
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = self._create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.BCELoss()
        
        # Training state
        self.training_history = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.logger.info(f"Initialized {config.architecture} model on {self.device}")
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        if self.config.architecture == "LSTM":
            return LSTMModel(self.config)
        elif self.config.architecture == "Transformer":
            return TransformerModel(self.config)
        else:
            raise ValueError(f"Unsupported architecture: {self.config.architecture}")
    
    def prepare_data(self, features: pd.DataFrame, targets: pd.Series,
                    train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training"""
        
        # Convert to numpy arrays
        features_array = features.values
        targets_array = targets.values
        
        # Split data
        split_idx = int(len(features_array) * train_split)
        
        train_features = features_array[:split_idx]
        train_targets = targets_array[:split_idx]
        val_features = features_array[split_idx:]
        val_targets = targets_array[split_idx:]
        
        # Create datasets
        train_dataset = TradingDataset(train_features, train_targets, self.config.sequence_length)
        val_dataset = TradingDataset(val_features, val_targets, self.config.sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Train the neural network model"""
        
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self._validate(val_loader)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(f"best_model_epoch_{epoch}.pth")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Log progress
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, "
                              f"Val Loss = {val_loss:.6f}")
        
        # Load best model
        self._load_best_checkpoint()
        
        # Save training history
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch + 1
        }
        
        self._save_training_history()
        
        return self.training_history
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        
        self.model.eval()
        
        # Prepare data
        features_array = features.values
        dataset = TradingDataset(features_array, np.zeros(len(features_array)), 
                              self.config.sequence_length)
        
        predictions = []
        
        with torch.no_grad():
            for sequence, _ in dataset:
                sequence = sequence.unsqueeze(0).to(self.device)
                prediction = self.model(sequence)
                predictions.append(prediction.cpu().numpy()[0, 0])
        
        return np.array(predictions)
    
    def predict_with_confidence(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals"""
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        features_array = features.values
        dataset = TradingDataset(features_array, np.zeros(len(features_array)), 
                              self.config.sequence_length)
        
        all_predictions = []
        
        # Multiple forward passes for uncertainty
        n_samples = 10
        for _ in range(n_samples):
            predictions = []
            with torch.no_grad():
                for sequence, _ in dataset:
                    sequence = sequence.unsqueeze(0).to(self.device)
                    prediction = self.model(sequence)
                    predictions.append(prediction.cpu().numpy()[0, 0])
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate mean and std
        mean_predictions = all_predictions.mean(axis=0)
        std_predictions = all_predictions.std(axis=0)
        
        return mean_predictions, std_predictions
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': len(self.training_history.get('train_losses', [])),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, self.model_dir / filename)
    
    def _load_best_checkpoint(self):
        """Load the best model checkpoint"""
        checkpoint_files = list(self.model_dir.glob("best_model_*.pth"))
        if not checkpoint_files:
            return
        
        # Find the best checkpoint
        best_checkpoint = min(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        checkpoint = torch.load(best_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Loaded best checkpoint: {best_checkpoint}")
    
    def _save_training_history(self):
        """Save training history to file"""
        history_file = self.model_dir / "training_history.json"
        
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_file}")
    
    def load_model(self, filename: str):
        """Load a saved model"""
        checkpoint_path = self.model_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Model loaded from {checkpoint_path}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(batch_targets.cpu().numpy().flatten())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Accuracy
        accuracy = np.mean(binary_predictions == targets)
        
        # Precision, Recall, F1
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(targets, binary_predictions, average='binary', zero_division=0)
        recall = recall_score(targets, binary_predictions, average='binary', zero_division=0)
        f1 = f1_score(targets, binary_predictions, average='binary', zero_division=0)
        
        # ROC AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(targets, predictions)
        except:
            auc = 0.0
        
        metrics = {
            'loss': total_loss / len(test_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        self.logger.info(f"Model evaluation: {metrics}")
        
        return metrics
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            'architecture': self.config.architecture,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.config.input_size,
            'hidden_layers': self.config.hidden_layers,
            'sequence_length': self.config.sequence_length,
            'device': str(self.device),
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'training_epochs': len(self.training_history.get('train_losses', [])),
            'best_val_loss': self.best_val_loss
        }
        
        return summary

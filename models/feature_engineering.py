"""
Nexus Trading System - Feature Engineering
Creates and manages features for ML models and signal generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import talib


@dataclass
class FeatureSet:
    """Container for engineered features"""
    features: pd.DataFrame
    feature_names: List[str]
    target: Optional[pd.Series] = None
    timestamps: pd.DatetimeIndex = None
    metadata: Dict[str, Any] = None


class FeatureEngineer:
    """Advanced feature engineering for trading signals"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature groups
        self.price_features = config.get('price_features', [])
        self.technical_features = config.get('technical_features', [])
        self.time_features = config.get('time_features', [])
        self.volatility_features = config.get('volatility_features', [])
        
        # Lookback periods
        self.lookback_periods = config.get('lookback_periods', {
            'short': 10,
            'medium': 50,
            'long': 200
        })
        
        # Scalers for normalization
        self.scalers = {}
        self.is_fitted = False
        
        # Feature cache
        self.feature_cache = {}
    
    def create_features(self, price_data: pd.DataFrame, 
                       target_data: Optional[pd.Series] = None,
                       symbol: str = "DEFAULT") -> FeatureSet:
        """
        Create comprehensive feature set from price data
        
        Args:
            price_data: OHLCV DataFrame
            target_data: Optional target variable for supervised learning
            symbol: Symbol identifier for caching
            
        Returns:
            FeatureSet with engineered features
        """
        
        self.logger.info(f"Creating features for {symbol} with {len(price_data)} data points")
        
        # Check cache first
        cache_key = f"{symbol}_{len(price_data)}_{price_data.index[0]}"
        if cache_key in self.feature_cache:
            self.logger.debug(f"Using cached features for {symbol}")
            return self.feature_cache[cache_key]
        
        # Initialize feature DataFrame
        features = pd.DataFrame(index=price_data.index)
        feature_names = []
        
        # 1. Price-based features
        price_features = self._create_price_features(price_data)
        features = pd.concat([features, price_features], axis=1)
        feature_names.extend(price_features.columns.tolist())
        
        # 2. Technical indicator features
        technical_features = self._create_technical_features(price_data)
        features = pd.concat([features, technical_features], axis=1)
        feature_names.extend(technical_features.columns.tolist())
        
        # 3. Time-based features
        time_features = self._create_time_features(price_data)
        features = pd.concat([features, time_features], axis=1)
        feature_names.extend(time_features.columns.tolist())
        
        # 4. Volatility features
        volatility_features = self._create_volatility_features(price_data)
        features = pd.concat([features, volatility_features], axis=1)
        feature_names.extend(volatility_features.columns.tolist())
        
        # 5. Advanced pattern features
        pattern_features = self._create_pattern_features(price_data)
        features = pd.concat([features, pattern_features], axis=1)
        feature_names.extend(pattern_features.columns.tolist())
        
        # 6. Market microstructure features
        microstructure_features = self._create_microstructure_features(price_data)
        features = pd.concat([features, microstructure_features], axis=1)
        feature_names.extend(microstructure_features.columns.tolist())
        
        # Remove NaN values
        features = features.dropna()
        
        # Create feature set
        feature_set = FeatureSet(
            features=features,
            feature_names=feature_names,
            target=target_data.loc[features.index] if target_data is not None else None,
            timestamps=features.index,
            metadata={
                'symbol': symbol,
                'created_at': datetime.now(),
                'original_data_points': len(price_data),
                'feature_count': len(feature_names),
                'lookback_used': self.lookback_periods
            }
        )
        
        # Cache the result
        self.feature_cache[cache_key] = feature_set
        
        self.logger.info(f"Created {len(feature_names)} features for {symbol}")
        
        return feature_set
    
    def _create_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Returns
        features['returns'] = price_data['close'].pct_change()
        features['log_returns'] = np.log(price_data['close'] / price_data['close'].shift(1))
        features['price_change_pct'] = (price_data['close'] - price_data['open']) / price_data['open']
        
        # Price ratios
        features['high_low_ratio'] = price_data['high'] / price_data['low']
        features['open_close_ratio'] = price_data['open'] / price_data['close']
        features['close_high_ratio'] = price_data['close'] / price_data['high']
        features['close_low_ratio'] = price_data['close'] / price_data['low']
        
        # Multi-period returns
        for period in [2, 3, 5, 10]:
            features[f'returns_{period}d'] = price_data['close'].pct_change(period)
            features[f'log_returns_{period}d'] = np.log(price_data['close'] / price_data['close'].shift(period))
        
        # Price position in range
        for window in [5, 10, 20]:
            features[f'price_position_{window}'] = (
                (price_data['close'] - price_data['low'].rolling(window).min()) /
                (price_data['high'].rolling(window).max() - price_data['low'].rolling(window).min())
            )
        
        return features
    
    def _create_technical_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            features[f'ma_{period}'] = price_data['close'].rolling(period).mean()
            features[f'price_to_ma_{period}'] = price_data['close'] / features[f'ma_{period}']
        
        # Exponential moving averages
        for period in [12, 26]:
            features[f'ema_{period}'] = price_data['close'].ewm(span=period).mean()
        
        # EMA crossover signals
        features['ema_12_26_cross'] = (features['ema_12'] > features['ema_26']).astype(int)
        features['ema_cross_signal'] = np.where(
            features['ema_12_26_cross'] != features['ema_12_26_cross'].shift(1),
            features['ema_12_26_cross'],
            0
        )
        
        # RSI
        features['rsi_14'] = talib.RSI(price_data['close'], timeperiod=14)
        features['rsi_7'] = talib.RSI(price_data['close'], timeperiod=7)
        features['rsi_21'] = talib.RSI(price_data['close'], timeperiod=21)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(price_data['close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(price_data['close'])
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_position'] = (price_data['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ATR (Average True Range)
        features['atr_14'] = talib.ATR(price_data['high'], price_data['low'], price_data['close'], timeperiod=14)
        features['atr_ratio'] = features['atr_14'] / price_data['close']
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(price_data['high'], price_data['low'], price_data['close'])
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd
        features['stoch_cross'] = np.where(slowk > slowd, 1, 0)
        
        # ADX (Average Directional Index)
        features['adx_14'] = talib.ADX(price_data['high'], price_data['low'], price_data['close'], timeperiod=14)
        features['plus_di'] = talib.PLUS_DI(price_data['high'], price_data['low'], price_data['close'], timeperiod=14)
        features['minus_di'] = talib.MINUS_DI(price_data['high'], price_data['low'], price_data['close'], timeperiod=14)
        features['di_diff'] = features['plus_di'] - features['minus_di']
        
        # CCI (Commodity Channel Index)
        features['cci_14'] = talib.CCI(price_data['high'], price_data['low'], price_data['close'], timeperiod=14)
        
        # Williams %R
        features['williams_r'] = talib.WILLR(price_data['high'], price_data['low'], price_data['close'], timeperiod=14)
        
        return features
    
    def _create_time_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Extract time components
        features['hour'] = price_data.index.hour
        features['day_of_week'] = price_data.index.dayofweek
        features['day_of_month'] = price_data.index.day
        features['month'] = price_data.index.month
        features['quarter'] = price_data.index.quarter
        
        # Cyclical encoding for time features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Session indicators
        features['is_london_session'] = ((features['hour'] >= 8) & (features['hour'] < 17)).astype(int)
        features['is_ny_session'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)
        features['is_asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        features['is_overlap_session'] = ((features['hour'] >= 13) & (features['hour'] < 17)).astype(int)
        
        # Weekend/holiday indicators
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)
        
        return features
    
    def _create_volatility_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Realized volatility at different windows
        returns = price_data['close'].pct_change()
        for window in [5, 10, 20, 50]:
            features[f'realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        # GARCH-like volatility
        features['volatility_trend'] = features['realized_vol_20'] / features['realized_vol_50']
        
        # Volatility regime
        vol_median = features['realized_vol_20'].median()
        features['vol_regime'] = np.where(features['realized_vol_20'] > vol_median * 1.5, 2,
                                         np.where(features['realized_vol_20'] < vol_median * 0.5, 0, 1))
        
        # Price volatility ratio
        features['price_vol_ratio'] = features['atr_14'] / price_data['close'] if 'atr_14' in features.columns else 0
        
        # Volatility of volatility
        features['vol_of_vol'] = features['realized_vol_20'].rolling(20).std()
        
        return features
    
    def _create_pattern_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create candlestick pattern features"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Doji patterns
        features['doji'] = talib.CDLDOJI(price_data['open'], price_data['high'], 
                                        price_data['low'], price_data['close'])
        features['dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(price_data['open'], price_data['high'],
                                                           price_data['low'], price_data['close'])
        features['gravestone_doji'] = talib.CDLGRAVESTONEDOJI(price_data['open'], price_data['high'],
                                                             price_data['low'], price_data['close'])
        
        # Engulfing patterns
        features['bullish_engulfing'] = talib.CDLENGULFING(price_data['open'], price_data['high'],
                                                           price_data['low'], price_data['close'])
        features['bearish_engulfing'] = talib.CDLENGULFING(price_data['open'], price_data['high'],
                                                           price_data['low'], price_data['close'])
        
        # Hammer patterns
        features['hammer'] = talib.CDLHAMMER(price_data['open'], price_data['high'],
                                           price_data['low'], price_data['close'])
        features['inverted_hammer'] = talib.CDLINVERTEDHAMMER(price_data['open'], price_data['high'],
                                                             price_data['low'], price_data['close'])
        
        # Star patterns
        features['morning_star'] = talib.CDLMORNINGSTAR(price_data['open'], price_data['high'],
                                                        price_data['low'], price_data['close'])
        features['evening_star'] = talib.CDLEVENINGSTAR(price_data['open'], price_data['high'],
                                                        price_data['low'], price_data['close'])
        
        # Harami patterns
        features['bullish_harami'] = talib.CDLHARAMI(price_data['open'], price_data['high'],
                                                      price_data['low'], price_data['close'])
        features['bearish_harami'] = talib.CDLHARAMI(price_data['open'], price_data['high'],
                                                      price_data['low'], price_data['close'])
        
        # Convert pattern signals to binary (0/1)
        pattern_cols = [col for col in features.columns if 'cdl' in col.lower()]
        for col in pattern_cols:
            features[col] = np.where(features[col] > 0, 1, np.where(features[col] < 0, -1, 0))
        
        return features
    
    def _create_microstructure_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Volume profile features
        if 'volume' in price_data.columns:
            features['volume_sma_10'] = price_data['volume'].rolling(10).mean()
            features['volume_ratio'] = price_data['volume'] / features['volume_sma_10']
            features['volume_trend'] = (features['volume_sma_10'] / 
                                       features['volume_sma_10'].shift(10) - 1)
            
            # Volume-price relationship
            features['vwap'] = (price_data['close'] * price_data['volume']).rolling(20).sum() / \
                             price_data['volume'].rolling(20).sum()
            features['price_to_vwap'] = price_data['close'] / features['vwap']
        
        # Liquidity indicators
        features['spread_estimate'] = (price_data['high'] - price_data['low']) / price_data['close']
        features['spread_trend'] = features['spread_estimate'].rolling(10).mean()
        
        # Order flow imbalance (proxy)
        features['buy_pressure'] = (price_data['close'] - price_data['low']) / (price_data['high'] - price_data['low'])
        features['sell_pressure'] = (price_data['high'] - price_data['close']) / (price_data['high'] - price_data['low'])
        features['flow_imbalance'] = features['buy_pressure'] - features['sell_pressure']
        
        # Price efficiency
        features['price_efficiency'] = abs(price_data['close'] - price_data['open']) / (price_data['high'] - price_data['low'])
        
        return features
    
    def create_target_variables(self, price_data: pd.DataFrame, 
                              horizon: int = 1,
                              threshold: float = 0.001) -> pd.Series:
        """
        Create target variables for supervised learning
        
        Args:
            price_data: OHLCV DataFrame
            horizon: Future horizon for target (number of periods)
            threshold: Minimum price change threshold for classification
            
        Returns:
            Series with target variables
        """
        
        # Future returns
        future_returns = price_data['close'].shift(-horizon) / price_data['close'] - 1
        
        # Classification target (1=buy, 0=hold, -1=sell)
        classification_target = np.where(future_returns > threshold, 1,
                                       np.where(future_returns < -threshold, -1, 0))
        
        # Binary classification (1=profitable, 0=not)
        binary_target = (future_returns > 0).astype(int)
        
        # Regression target (actual return)
        regression_target = future_returns
        
        # Choose target type based on use case
        # For now, return binary classification
        return pd.Series(binary_target, index=price_data.index, name='target')
    
    def normalize_features(self, feature_set: FeatureSet, 
                          method: str = 'standard') -> FeatureSet:
        """
        Normalize features using specified method
        
        Args:
            feature_set: FeatureSet to normalize
            method: Normalization method ('standard', 'minmax')
            
        Returns:
            Normalized FeatureSet
        """
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Fit scaler on training data if not already fitted
        if not self.is_fitted:
            scaler.fit(feature_set.features)
            self.scalers[method] = scaler
            self.is_fitted = True
        
        # Transform features
        normalized_features = pd.DataFrame(
            scaler.transform(feature_set.features),
            index=feature_set.features.index,
            columns=feature_set.features.columns
        )
        
        # Create new feature set
        normalized_feature_set = FeatureSet(
            features=normalized_features,
            feature_names=feature_set.feature_names,
            target=feature_set.target,
            timestamps=feature_set.timestamps,
            metadata={
                **feature_set.metadata,
                'normalized': True,
                'normalization_method': method,
                'normalized_at': datetime.now()
            }
        )
        
        return normalized_feature_set
    
    def select_features(self, feature_set: FeatureSet, 
                       method: str = 'correlation',
                       k_best: int = 50) -> FeatureSet:
        """
        Feature selection using various methods
        
        Args:
            feature_set: FeatureSet with features
            method: Selection method ('correlation', 'variance', 'mutual_info')
            k_best: Number of best features to select
            
        Returns:
            FeatureSet with selected features
        """
        
        if method == 'correlation' and feature_set.target is not None:
            # Select features with highest correlation to target
            correlations = feature_set.features.corrwith(feature_set.target).abs()
            top_features = correlations.nlargest(k_best).index
            
        elif method == 'variance':
            # Select features with highest variance
            variances = feature_set.features.var()
            top_features = variances.nlargest(k_best).index
            
        elif method == 'mutual_info' and feature_set.target is not None:
            from sklearn.feature_selection import mutual_info_classif
            
            mi_scores = mutual_info_classif(feature_set.features, feature_set.target)
            mi_series = pd.Series(mi_scores, index=feature_set.features.columns)
            top_features = mi_series.nlargest(k_best).index
            
        else:
            # Default: select first k_best features
            top_features = feature_set.features.columns[:k_best]
        
        # Create new feature set with selected features
        selected_feature_set = FeatureSet(
            features=feature_set.features[top_features],
            feature_names=top_features.tolist(),
            target=feature_set.target,
            timestamps=feature_set.timestamps,
            metadata={
                **feature_set.metadata,
                'feature_selection': True,
                'selection_method': method,
                'original_feature_count': len(feature_set.feature_names),
                'selected_feature_count': len(top_features)
            }
        )
        
        self.logger.info(f"Selected {len(top_features)} features using {method} method")
        
        return selected_feature_set
    
    def get_feature_importance(self, feature_set: FeatureSet, 
                             method: str = 'correlation') -> pd.Series:
        """
        Calculate feature importance scores
        
        Args:
            feature_set: FeatureSet with features and target
            method: Importance calculation method
            
        Returns:
            Series with feature importance scores
        """
        
        if method == 'correlation' and feature_set.target is not None:
            importance = feature_set.features.corrwith(feature_set.target).abs()
            
        elif method == 'variance':
            importance = feature_set.features.var()
            
        elif method == 'mutual_info' and feature_set.target is not None:
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(feature_set.features, feature_set.target)
            importance = pd.Series(mi_scores, index=feature_set.features.columns)
            
        else:
            raise ValueError(f"Unsupported importance method: {method}")
        
        return importance.sort_values(ascending=False)
    
    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
        self.logger.info("Feature cache cleared")
    
    def get_feature_summary(self, feature_set: FeatureSet) -> Dict[str, Any]:
        """Get summary statistics for feature set"""
        
        summary = {
            'total_features': len(feature_set.feature_names),
            'data_points': len(feature_set.features),
            'missing_values': feature_set.features.isnull().sum().sum(),
            'feature_types': {},
            'memory_usage_mb': feature_set.features.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Feature type breakdown
        for col in feature_set.features.columns:
            dtype = str(feature_set.features[col].dtype)
            summary['feature_types'][dtype] = summary['feature_types'].get(dtype, 0) + 1
        
        return summary

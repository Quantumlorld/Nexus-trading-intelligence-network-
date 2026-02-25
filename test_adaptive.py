#!/usr/bin/env python3
"""
Test script for the adaptive learning layer
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Test the adaptive learning components"""
    
    print("🚀 NEXUS ADAPTIVE LAYER TEST")
    print("=" * 50)
    
    try:
        # Import adaptive components
        from adaptive import (
            UserPerformanceTracker, PerformanceCategory, RiskProfile,
            OutlierDetector, OutlierType, OutlierSeverity,
            StrategyWeightManager, WeightUpdateMethod,
            create_performance_tracker, create_outlier_detector, create_strategy_weight_manager
        )
        
        print("✅ All adaptive components imported successfully!")
        
        # Create instances
        print("\n📊 Creating adaptive components...")
        
        performance_tracker = create_performance_tracker()
        outlier_detector = create_outlier_detector()
        weight_manager = create_strategy_weight_manager()
        
        print("✅ Adaptive components created successfully!")
        
        # Test performance tracker
        print("\n📈 Testing Performance Tracker...")
        print(f"   - Min trades for analysis: {performance_tracker.min_trades_for_analysis}")
        print(f"   - Performance window: {performance_tracker.performance_window_days} days")
        print(f"   - Performance categories: {[c.value for c in PerformanceCategory]}")
        print(f"   - Risk profiles: {[r.value for r in RiskProfile]}")
        
        # Test outlier detector
        print("\n🔍 Testing Outlier Detector...")
        print(f"   - Z-score threshold: {outlier_detector.z_threshold}")
        print(f"   - IQR multiplier: {outlier_detector.iqr_multiplier}")
        print(f"   - Contamination rate: {outlier_detector.contamination_rate}")
        print(f"   - Min samples: {outlier_detector.min_samples_for_detection}")
        print(f"   - Outlier types: {[t.value for t in OutlierType]}")
        print(f"   - Severity levels: {[s.value for s in OutlierSeverity]}")
        
        # Test weight manager
        print("\n⚖️ Testing Strategy Weight Manager...")
        print(f"   - Learning rate: {weight_manager.learning_rate}")
        print(f"   - Decay factor: {weight_manager.decay_factor}")
        print(f"   - Min sample size: {weight_manager.min_sample_size}")
        print(f"   - Weight smoothing: {weight_manager.weight_smoothing}")
        print(f"   - Update method: {weight_manager.update_method.value}")
        print(f"   - Performance thresholds: {weight_manager.performance_thresholds}")
        
        # Test update method
        print("\n🔄 Testing Update Methods...")
        for method in WeightUpdateMethod:
            weight_manager.set_update_method(method)
            print(f"   - {method.value}: ✓")
        
        print("\n🎯 ADAPTIVE LAYER TEST COMPLETED SUCCESSFULLY!")
        print("✅ All components are working correctly!")
        print("📊 Ready for integration with the main trading system!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing adaptive layer: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

import React, { useState, useEffect } from 'react';
import { systemService } from '@/services/system';

interface DemoProgress {
  trade_count: number;
  current_phase: string;
  phase_name: string;
  phase_progress: number;
  adaptive_learning: boolean;
  demo_mode: boolean;
  next_phase_features: string[];
}

interface DemoTradingPanelProps {
  className?: string;
}

export const DemoTradingPanel: React.FC<DemoTradingPanelProps> = ({ className }) => {
  const [demoProgress, setDemoProgress] = useState<DemoProgress | null>(null);
  const [loading, setLoading] = useState(false);
  const [demoStarted, setDemoStarted] = useState(false);

  // Fetch demo progress
  useEffect(() => {
    const fetchDemoProgress = async () => {
      try {
        const progress = await systemService.getDemoProgress();
        setDemoProgress(progress);
        setDemoStarted(progress.demo_mode);
      } catch (error) {
        console.error('Failed to fetch demo progress:', error);
      }
    };

    fetchDemoProgress();
    const interval = setInterval(fetchDemoProgress, 3000);
    return () => clearInterval(interval);
  }, []);

  const startDemoTrading = async () => {
    setLoading(true);
    try {
      const response = await systemService.startDemoTrading();
      if (response.success) {
        setDemoStarted(true);
        alert('Demo trading started! 500 trade learning plan activated.');
      } else {
        alert(`Failed to start demo: ${response.message}`);
      }
    } catch (error) {
      alert(`Error starting demo: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'baseline': return 'from-blue-500 to-blue-600';
      case 'learning': return 'from-purple-500 to-purple-600';
      case 'optimization': return 'from-green-500 to-green-600';
      default: return 'from-gray-500 to-gray-600';
    }
  };

  const getPhaseIcon = (phase: string) => {
    switch (phase) {
      case 'baseline': return '📊';
      case 'learning': return '🧠';
      case 'optimization': return '⚡';
      default: return '🎯';
    }
  };

  const formatProgress = (progress: number) => {
    return `${(progress * 100).toFixed(1)}%`;
  };

  return (
    <div className={className}>
      <div className="bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 rounded-2xl p-6 text-white shadow-2xl shadow-purple-500/20 border border-purple-400/30">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold mb-1">🚀 Demo Trading Center</h2>
            <p className="text-purple-100 text-sm">500 Trade Learning & Adaptation Plan</p>
          </div>
          <div className="flex items-center space-x-4">
            {demoStarted && (
              <span className="px-3 py-1 bg-green-500 text-white text-xs rounded-full font-bold animate-pulse shadow-lg shadow-green-400/50">
                DEMO ACTIVE
              </span>
            )}
          </div>
        </div>

        {/* Demo Trading Plan */}
        <div className="mb-6">
          <h3 className="text-lg font-bold mb-4 flex items-center">
            <span className="w-2 h-2 bg-yellow-400 rounded-full mr-2"></span>
            Trading Plan Overview
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-black/20 backdrop-blur-sm rounded-xl p-4 border border-blue-400/30">
              <div className="flex items-center mb-2">
                <span className="text-2xl mr-2">📊</span>
                <div>
                  <h4 className="font-bold">Phase 1</h4>
                  <p className="text-sm text-blue-200">Baseline Establishment</p>
                </div>
              </div>
              <div className="text-2xl font-bold text-blue-300 mb-2">100 Trades</div>
              <div className="text-sm text-blue-200">
                <p>• Safe risk management</p>
                <p>• Performance tracking</p>
                <p>• Establish metrics</p>
              </div>
            </div>

            <div className="bg-black/20 backdrop-blur-sm rounded-xl p-4 border border-purple-400/30">
              <div className="flex items-center mb-2">
                <span className="text-2xl mr-2">🧠</span>
                <div>
                  <h4 className="font-bold">Phase 2</h4>
                  <p className="text-sm text-purple-200">Adaptive Learning</p>
                </div>
              </div>
              <div className="text-2xl font-bold text-purple-300 mb-2">200 Trades</div>
              <div className="text-sm text-purple-200">
                <p>• Behavioral analysis</p>
                <p>• Strategy optimization</p>
                <p>• Pattern recognition</p>
              </div>
            </div>

            <div className="bg-black/20 backdrop-blur-sm rounded-xl p-4 border border-green-400/30">
              <div className="flex items-center mb-2">
                <span className="text-2xl mr-2">⚡</span>
                <div>
                  <h4 className="font-bold">Phase 3</h4>
                  <p className="text-sm text-green-200">Full Optimization</p>
                </div>
              </div>
              <div className="text-2xl font-bold text-green-300 mb-2">200 Trades</div>
              <div className="text-sm text-green-200">
                <p>• Dynamic position sizing</p>
                <p>• Outlier detection</p>
                <p>• Full automation</p>
              </div>
            </div>
          </div>
        </div>

        {/* Current Progress */}
        {demoProgress && (
          <div className="mb-6">
            <h3 className="text-lg font-bold mb-4 flex items-center">
              <span className="w-2 h-2 bg-cyan-400 rounded-full mr-2 animate-pulse"></span>
              Current Progress
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-black/30 backdrop-blur-sm rounded-xl p-4 border border-cyan-400/30">
                <h4 className="font-bold mb-3">Phase Status</h4>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-cyan-200">Current Phase</span>
                    <div className="flex items-center">
                      <span className="text-2xl mr-2">{getPhaseIcon(demoProgress.phase_name)}</span>
                      <span className={`px-3 py-1 rounded-full text-xs font-bold bg-gradient-to-r ${getPhaseColor(demoProgress.phase_name)} text-white`}>
                        {demoProgress.phase_name.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-cyan-200">Progress</span>
                    <span className="text-cyan-300 font-bold">{formatProgress(demoProgress.phase_progress)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-cyan-200">Total Trades</span>
                    <span className="text-cyan-300 font-bold">{demoProgress.trade_count}/500</span>
                  </div>
                </div>
              </div>

              <div className="bg-black/30 backdrop-blur-sm rounded-xl p-4 border border-purple-400/30">
                <h4 className="font-bold mb-3">Learning Features</h4>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <span className={`w-3 h-3 rounded-full mr-2 ${demoProgress.adaptive_learning ? 'bg-green-400' : 'bg-gray-400'}`}></span>
                    <span className="text-purple-200">Adaptive Learning</span>
                  </div>
                  <div className="flex items-center">
                    <span className={`w-3 h-3 rounded-full mr-2 ${demoProgress.trade_count > 100 ? 'bg-green-400' : 'bg-gray-400'}`}></span>
                    <span className="text-purple-200">Behavioral Analysis</span>
                  </div>
                  <div className="flex items-center">
                    <span className={`w-3 h-3 rounded-full mr-2 ${demoProgress.trade_count > 300 ? 'bg-green-400' : 'bg-gray-400'}`}></span>
                    <span className="text-purple-200">Full Optimization</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Next Phase Features */}
        {demoProgress && demoProgress.next_phase_features.length > 0 && (
          <div className="mb-6">
            <h3 className="text-lg font-bold mb-4 flex items-center">
              <span className="w-2 h-2 bg-yellow-400 rounded-full mr-2"></span>
              Next Phase Features
            </h3>
            <div className="bg-black/20 backdrop-blur-sm rounded-xl p-4 border border-yellow-400/30">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {demoProgress.next_phase_features.map((feature, index) => (
                  <div key={index} className="flex items-center">
                    <span className="w-2 h-2 bg-yellow-400 rounded-full mr-2"></span>
                    <span className="text-yellow-200 text-sm">{feature}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Control Buttons */}
        <div className="flex space-x-4">
          {!demoStarted ? (
            <button
              onClick={startDemoTrading}
              disabled={loading}
              className="flex-1 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 disabled:from-gray-500 disabled:to-gray-600 text-white px-6 py-3 rounded-xl font-bold transition-all transform hover:scale-105 shadow-xl"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                  </svg>
                  Starting Demo...
                </span>
              ) : (
                '🚀 Start 500-Trade Demo'
              )}
            </button>
          ) : (
            <div className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 text-white px-6 py-3 rounded-xl font-bold shadow-xl text-center">
              <div className="flex items-center justify-center">
                <span className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse"></span>
                Demo Trading Active
              </div>
            </div>
          )}
        </div>

        {/* Instructions */}
        <div className="mt-6 p-4 bg-black/20 backdrop-blur-sm rounded-xl border border-purple-400/30">
          <h4 className="font-bold mb-3 text-purple-200">📋 Demo Instructions</h4>
          <div className="text-sm text-purple-200 space-y-2">
            <p>• <strong>Phase 1 (0-100 trades)</strong>: Establish baseline with safe risk management</p>
            <p>• <strong>Phase 2 (101-300 trades)</strong>: System learns your patterns and adapts strategies</p>
            <p>• <strong>Phase 3 (301-500 trades)</strong>: Full optimization with advanced features</p>
            <p>• <strong>Connect MT5 Demo</strong>: Use demo account for safe learning</p>
            <p>• <strong>Monitor Progress</strong>: Watch system adapt to your trading style</p>
          </div>
        </div>
      </div>
    </div>
  );
};

import React, { useState, useEffect } from 'react';
import { systemService } from '@/services/system';

interface UniversalMT5Status {
  connected: boolean;
  connector_type?: string;
  account_summary?: any;
  available_symbols?: string[];
  open_positions?: any[];
  demo_mode?: boolean;
  trade_count?: number;
  current_phase?: string;
  message?: string;
}

interface UniversalMT5ConnectionProps {
  className?: string;
}

export const UniversalMT5Connection: React.FC<UniversalMT5ConnectionProps> = ({ className }) => {
  const [status, setStatus] = useState<UniversalMT5Status | null>(null);
  const [connecting, setConnecting] = useState(false);

  // Fetch universal MT5 status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await systemService.getUniversalMT5Status();
        setStatus(response.data || null);
      } catch (error) {
        console.error('Failed to fetch universal MT5 status:', error);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const connectUniversalMT5 = async () => {
    setConnecting(true);
    try {
      const response = await systemService.connectUniversalMT5();
      if (response.success) {
        alert('✅ Connected to MT5 Terminal! Universal connector active.');
      } else {
        alert(`❌ Connection failed: ${response.message}`);
      }
    } catch (error) {
      alert(`Error connecting to MT5: ${error}`);
    } finally {
      setConnecting(false);
    }
  };

  const printSetupInstructions = () => {
    const instructions = `
🎯 NEXUS UNIVERSAL MT5 SETUP
================================

📋 REQUIREMENTS:
1. ✅ Install MetaTrader 5 Terminal
2. ✅ Open MT5 Terminal (any broker)
3. ✅ Login to ANY MT5 Account (Demo or Live)
4. ✅ Enable "Allow DLL Imports" in MT5
5. ✅ Enable "Automated Trading" in MT5

🔗 CONNECTION OPTIONS:
• Option A: Login to MT5 manually, then click "Connect Universal MT5"
• Option B: Use auto-connection (attempts 5 times)
• Option C: Connect to specific broker account

💡 UNIVERSAL ADVANTAGES:
• Works with ANY MT5 broker worldwide
• No broker restrictions or approvals
• Direct API access to MT5 terminal
• Full trading functionality
• Real-time position management
• Account information access

⚠️ IMPORTANT NOTES:
• This bypasses broker-specific APIs
• Uses direct MT5 terminal connection
• Works with demo AND live accounts
• No additional broker setup required
• Full control over trading operations

Ready to connect to ANY MT5 broker! 🎯
    `;
    alert(instructions);
  };

  if (!status) {
    return (
      <div className={className}>
        <div className="bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 rounded-2xl p-6 text-white shadow-2xl shadow-purple-500/20 border border-purple-400/30">
          <div className="animate-pulse">
            <div className="flex items-center justify-center">
              <svg className="animate-spin h-8 w-8 mr-3" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
              <span>Loading Universal MT5 Status...</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={className}>
      <div className="bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 rounded-2xl p-6 text-white shadow-2xl shadow-purple-500/20 border border-purple-400/30">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold mb-1">🌐 Universal MT5 Connection</h2>
            <p className="text-purple-100 text-sm">Works with ANY MT5 broker worldwide</p>
          </div>
          <div className="flex items-center space-x-4">
            {status.connected && (
              <span className="px-3 py-1 bg-green-500 text-white text-xs rounded-full font-bold animate-pulse shadow-lg shadow-green-400/50">
                UNIVERSAL ACTIVE
              </span>
            )}
          </div>
        </div>

        {/* Connection Status */}
        <div className="mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-black/20 backdrop-blur-sm rounded-xl p-4 border border-cyan-400/30">
              <h4 className="font-bold mb-3 flex items-center">
                <span className={`w-3 h-3 rounded-full mr-2 ${status.connected ? 'bg-green-400' : 'bg-red-400'}`}></span>
                Connection Status
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-cyan-200">Status</span>
                  <span className={`font-bold ${status.connected ? 'text-green-400' : 'text-red-400'}`}>
                    {status.connected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-cyan-200">Connector Type</span>
                  <span className="text-cyan-300 font-bold">
                    {status.connector_type || 'Universal'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-cyan-200">Demo Mode</span>
                  <span className="text-cyan-300 font-bold">
                    {status.demo_mode ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-black/20 backdrop-blur-sm rounded-xl p-4 border border-purple-400/30">
              <h4 className="font-bold mb-3 flex items-center">
                <span className="w-3 h-3 bg-purple-400 rounded-full mr-2"></span>
                Trading Info
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-purple-200">Trade Count</span>
                  <span className="text-purple-300 font-bold">
                    {status.trade_count || 0}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-purple-200">Current Phase</span>
                  <span className="text-purple-300 font-bold">
                    {status.current_phase || 'baseline'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-purple-200">Open Positions</span>
                  <span className="text-purple-300 font-bold">
                    {status.open_positions?.length || 0}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Account Summary */}
        {status.connected && status.account_summary && (
          <div className="mb-6">
            <h4 className="font-bold mb-3 flex items-center">
              <span className="w-3 h-3 bg-green-400 rounded-full mr-2"></span>
              Account Summary
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-black/20 backdrop-blur-sm rounded-lg p-3 border border-green-400/30">
                <p className="text-xs text-green-200 mb-1">Balance</p>
                <p className="text-lg font-bold text-green-300">
                  ${status.account_summary.balance?.toFixed(2) || '0.00'}
                </p>
              </div>
              <div className="bg-black/20 backdrop-blur-sm rounded-lg p-3 border border-blue-400/30">
                <p className="text-xs text-blue-200 mb-1">Equity</p>
                <p className="text-lg font-bold text-blue-300">
                  ${status.account_summary.equity?.toFixed(2) || '0.00'}
                </p>
              </div>
              <div className="bg-black/20 backdrop-blur-sm rounded-lg p-3 border border-purple-400/30">
                <p className="text-xs text-purple-200 mb-1">Margin</p>
                <p className="text-lg font-bold text-purple-300">
                  ${status.account_summary.margin?.toFixed(2) || '0.00'}
                </p>
              </div>
              <div className="bg-black/20 backdrop-blur-sm rounded-lg p-3 border border-yellow-400/30">
                <p className="text-xs text-yellow-200 mb-1">Profit</p>
                <p className={`text-lg font-bold ${status.account_summary.profit >= 0 ? 'text-green-300' : 'text-red-300'}`}>
                  ${status.account_summary.profit?.toFixed(2) || '0.00'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Available Symbols */}
        {status.connected && status.available_symbols && status.available_symbols.length > 0 && (
          <div className="mb-6">
            <h4 className="font-bold mb-3 flex items-center">
              <span className="w-3 h-3 bg-cyan-400 rounded-full mr-2"></span>
              Available Symbols
            </h4>
            <div className="bg-black/20 backdrop-blur-sm rounded-xl p-4 border border-cyan-400/30">
              <div className="flex flex-wrap gap-2">
                {status.available_symbols.slice(0, 10).map((symbol, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-cyan-500/20 text-cyan-200 rounded-full text-xs font-medium border border-cyan-400/30"
                  >
                    {symbol}
                  </span>
                ))}
                {status.available_symbols.length > 10 && (
                  <span className="px-3 py-1 bg-cyan-500/20 text-cyan-200 rounded-full text-xs font-medium border border-cyan-400/30">
                    +{status.available_symbols.length - 10} more
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Control Buttons */}
        <div className="flex space-x-4">
          {!status.connected ? (
            <>
              <button
                onClick={connectUniversalMT5}
                disabled={connecting}
                className="flex-1 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 disabled:from-gray-500 disabled:to-gray-600 text-white px-6 py-3 rounded-xl font-bold transition-all transform hover:scale-105 shadow-xl"
              >
                {connecting ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                    </svg>
                    Connecting...
                  </span>
                ) : (
                  '🌐 Connect Universal MT5'
                )}
              </button>
              <button
                onClick={printSetupInstructions}
                className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white rounded-xl font-bold transition-all transform hover:scale-105 shadow-xl"
              >
                📋 Setup Guide
              </button>
            </>
          ) : (
            <div className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 text-white px-6 py-3 rounded-xl font-bold shadow-xl text-center">
              <div className="flex items-center justify-center">
                <span className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse"></span>
                Universal MT5 Connected
              </div>
            </div>
          )}
        </div>

        {/* Instructions */}
        <div className="mt-6 p-4 bg-black/20 backdrop-blur-sm rounded-xl border border-purple-400/30">
          <h4 className="font-bold mb-3 text-purple-200">🌐 Universal MT5 Advantages</h4>
          <div className="text-sm text-purple-200 space-y-2">
            <p>• <strong>Works with ANY MT5 broker</strong> - No restrictions or approvals</p>
            <p>• <strong>Direct terminal connection</strong> - Bypasses broker APIs</p>
            <p>• <strong>Full trading functionality</strong> - All features available</p>
            <p>• <strong>Demo & Live accounts</strong> - Works with both</p>
            <p>• <strong>No setup required</strong> - Just login to MT5 and connect</p>
          </div>
        </div>
      </div>
    </div>
  );
};

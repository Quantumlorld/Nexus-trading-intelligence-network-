import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

const SimpleDashboard: React.FC = () => {
  const [mt5Status, setMt5Status] = React.useState({
    connected: false,
    loading: false,
    message: 'Not Connected'
  });

  const [demoStatus, setDemoStatus] = React.useState({
    active: false,
    trades: 0,
    phase: 'Baseline',
    winRate: 0,
    profit: 0
  });

  const [tradeHistory, setTradeHistory] = React.useState<any[]>([]);

  // Poll MT5 status from backend (bridge-aware)
  React.useEffect(() => {
    const fetchStatus = async () => {
      try {
        const resp = await fetch('http://localhost:8000/admin/mt5-status');
        const data = await resp.json();
        setMt5Status({
          connected: !!data.connected,
          loading: false,
          message: data.message || 'Not Connected'
        });
      } catch {
        setMt5Status(prev => ({ ...prev, loading: false }));
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Poll trade history
  React.useEffect(() => {
    const fetchHistory = async () => {
      try {
        const resp = await fetch('http://localhost:8000/trade/history');
        const data = await resp.json();
        setTradeHistory(data.history || []);
      } catch {
        // ignore
      }
    };
    fetchHistory();
    const interval = setInterval(fetchHistory, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleConnectMT5 = async () => {
    setMt5Status(prev => ({ ...prev, loading: true, message: 'Connecting...' }));
    
    try {
      const response = await fetch('http://localhost:8000/admin/mt5-connect-xm', {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (data.success) {
        setMt5Status({
          connected: true,
          loading: false,
          message: 'Connected'
        });
      } else {
        setMt5Status({
          connected: false,
          loading: false,
          message: data.message || 'Connection Failed'
        });
      }
    } catch (error) {
      setMt5Status({
        connected: false,
        loading: false,
        message: 'Connection Error'
      });
    }
  };

  const handleStartDemo = async () => {
    try {
      const response = await fetch('http://localhost:8000/admin/demo/start', {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (data.success) {
        setDemoStatus({
          active: true,
          trades: 0,
          phase: 'Baseline',
          winRate: 0,
          profit: 0
        });
      }
    } catch (error) {
      console.error('Demo start error:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            Nexus Trading System
          </h1>
          <p className="text-purple-200 text-lg">
            Professional Trading Intelligence Platform
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* System Status Card */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4">
              System Status
            </h2>
            <div className="space-y-3">
              <div className="flex justify-between text-white">
                <span>Backend API</span>
                <span className="text-green-400">✅ Online</span>
              </div>
              <div className="flex justify-between text-white">
                <span>Database</span>
                <span className="text-green-400">✅ Connected</span>
              </div>
              <div className="flex justify-between text-white">
                <span>MT5 Connection</span>
                <span className={mt5Status.connected ? "text-green-400" : "text-yellow-400"}>
                  {mt5Status.connected ? "✅ Connected" : "⚠️ " + mt5Status.message}
                </span>
              </div>
              <div className="flex justify-between text-white">
                <span>Demo Trading</span>
                <span className="text-green-400">✅ Ready</span>
              </div>
            </div>
          </div>
          
          {/* MT5 Connection Card */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4">
              MT5 Connection
            </h2>
            <div className="space-y-3">
              <input
                type="number"
                placeholder="Account Number"
                className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50"
                defaultValue="5047475068"
              />
              <input
                type="password"
                placeholder="Password"
                className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50"
                defaultValue=""
              />
              <input
                type="text"
                placeholder="Server"
                className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50"
                defaultValue="MetaQuotes-Demo"
              />
              <button 
                onClick={handleConnectMT5}
                disabled={mt5Status.loading}
                className="w-full bg-gradient-to-r from-green-400 to-emerald-400 text-white py-2 rounded-lg font-semibold hover:from-green-500 hover:to-emerald-500 transition-all disabled:opacity-50"
              >
                {mt5Status.loading ? 'Connecting...' : 'Connect to XM Global MT5'}
              </button>
            </div>
          </div>
          
          {/* Demo Trading Card */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4">
              Demo Trading
            </h2>
            <div className="space-y-3">
              <div className="text-white">
                <div className="flex justify-between mb-2">
                  <span>Progress</span>
                  <span>0/500 trades</span>
                </div>
                <div className="w-full bg-white/20 rounded-full h-2">
                  <div className="bg-gradient-to-r from-green-400 to-emerald-400 h-2 rounded-full" style={{width: '0%'}}></div>
                </div>
              </div>
              <div className="text-white space-y-2">
                <div className="flex justify-between">
                  <span>Current Phase</span>
                  <span className="text-purple-300">Baseline</span>
                </div>
                <div className="flex justify-between">
                  <span>Win Rate</span>
                  <span className="text-green-300">0%</span>
                </div>
                <div className="flex justify-between">
                  <span>Net Profit</span>
                  <span className="text-green-300">$0.00</span>
                </div>
              </div>
              <button 
                onClick={handleStartDemo}
                className="w-full bg-gradient-to-r from-purple-400 to-pink-400 text-white py-2 rounded-lg font-semibold hover:from-purple-500 hover:to-pink-500 transition-all"
              >
                Start 500-Trade Demo
              </button>
            </div>
          </div>
          
          {/* Trading Panel Card */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4">
              Quick Trade
            </h2>
            <div className="space-y-3">
              <select className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white">
                <option value="EUR/USD">EUR/USD</option>
                <option value="GBP/USD">GBP/USD</option>
                <option value="USD/JPY">USD/JPY</option>
                <option value="XAU/USD">XAU/USD</option>
                <option value="BTC/USD">BTC/USD</option>
              </select>
              <div className="grid grid-cols-2 gap-2">
                <button className="bg-green-500 text-white py-2 rounded-lg font-semibold hover:bg-green-600 transition-all">
                  BUY
                </button>
                <button className="bg-red-500 text-white py-2 rounded-lg font-semibold hover:bg-red-600 transition-all">
                  SELL
                </button>
              </div>
              <input
                type="number"
                placeholder="Volume (lots)"
                className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50"
                defaultValue="0.01"
              />
            </div>
          </div>
          
          {/* Performance Card */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4">
              Performance
            </h2>
            <div className="space-y-3">
              <div className="text-white space-y-2">
                <div className="flex justify-between">
                  <span>Total Trades</span>
                  <span className="text-blue-300">0</span>
                </div>
                <div className="flex justify-between">
                  <span>Winning Trades</span>
                  <span className="text-green-300">0</span>
                </div>
                <div className="flex justify-between">
                  <span>Losing Trades</span>
                  <span className="text-red-300">0</span>
                </div>
                <div className="flex justify-between">
                  <span>Win Rate</span>
                  <span className="text-yellow-300">0%</span>
                </div>
                <div className="flex justify-between">
                  <span>Total Profit</span>
                  <span className="text-green-300">$0.00</span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Settings Card */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h2 className="text-xl font-semibold text-white mb-4">
              Settings
            </h2>
            <div className="space-y-3">
              <div className="text-white space-y-2">
                <label className="flex items-center space-x-2">
                  <input type="checkbox" className="rounded" defaultChecked />
                  <span>Enable Trading</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input type="checkbox" className="rounded" defaultChecked />
                  <span>Risk Management</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input type="checkbox" className="rounded" />
                  <span>Auto Trading</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input type="checkbox" className="rounded" defaultChecked />
                  <span>Notifications</span>
                </label>
              </div>
            </div>
          </div>
        </div>
        
        {/* Footer */}
        <div className="mt-12 text-center text-white/60">
          <p>Nexus Trading Intelligence Network - Professional Trading Platform</p>
          <p className="text-sm mt-2">Ready for Production Use • 500-Trade Demo System • Universal MT5 Integration</p>
        </div>
      </div>

      {/* Trade History */}
      <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
        <h2 className="text-xl font-bold text-white mb-4">Trade History</h2>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {tradeHistory.length === 0 ? (
            <p className="text-white/50">No trades yet.</p>
          ) : (
            tradeHistory.map((trade, idx) => (
              <div key={idx} className="bg-black/30 rounded p-3 text-sm text-white">
                <div className="flex justify-between">
                  <span>{trade.symbol} {trade.action} {trade.volume}</span>
                  <span className={trade.status === 'executed' ? 'text-green-400' : 'text-red-400'}>
                    {trade.status}
                  </span>
                </div>
                <div className="text-xs text-white/50 mt-1">
                  ID: {trade.id} | Retcode: {trade.retcode}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SimpleDashboard />} />
        <Route path="/dashboard" element={<SimpleDashboard />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Router>
  );
};

export default App;

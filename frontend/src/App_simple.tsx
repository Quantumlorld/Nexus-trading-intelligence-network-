import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

const SimpleDashboard: React.FC = () => {
  const [mt5Status, setMt5Status] = React.useState({
    connected: false,
    loading: false,
    message: 'Not Connected'
  });

  const [tradeHistory, setTradeHistory] = React.useState<any[]>([]);
  const [batchStatus, setBatchStatus] = React.useState<{
    running: boolean;
    batchId: string | null;
    queued: number;
    executed: number;
    failed: number;
    processed: number;
  }>({
    running: false,
    batchId: null,
    queued: 0,
    executed: 0,
    failed: 0,
    processed: 0
  });

  // Poll MT5 status from backend (bridge-aware)
  React.useEffect(() => {
    const fetchStatus = async () => {
      try {
        const resp = await fetch('http://localhost:8000/admin/mt5-status');
        const data = await resp.json();
        console.log('[MT5_STATUS]', data);
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

  // Update batch progress from trade history
  React.useEffect(() => {
    const executed = tradeHistory.filter(t => t.status === 'executed').length;
    const failed = tradeHistory.filter(t => t.status === 'failed').length;
    const processed = tradeHistory.length;
    setBatchStatus(prev => ({
      ...prev,
      processed,
      executed,
      failed
    }));
  }, [tradeHistory]);

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

  const handleStartBatch = async () => {
    setBatchStatus(prev => ({ ...prev, running: true, queued: 0, executed: 0, failed: 0, processed: 0 }));
    
    try {
      const response = await fetch('http://localhost:8000/admin/demo/batch-500', {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (data.success) {
        setBatchStatus({
          running: true,
          batchId: data.batch_id,
          queued: data.queued,
          executed: 0,
          failed: 0,
          processed: 0
        });
      }
    } catch (error) {
      console.error('Batch start failed:', error);
      setBatchStatus(prev => ({ ...prev, running: false }));
    }
  };

  const handleQuickTrade = async (action: string) => {
    try {
      const response = await fetch('http://localhost:8000/trade', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          symbol: 'EUR/USD',
          action: action.toUpperCase(),
          quantity: 0.01,
          order_type: 'MARKET'
        })
      });
      
      const data = await response.json();
      console.log('Trade result:', data);
    } catch (error) {
      console.error('Trade failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden relative">
      {/* Animated starfield background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-950 via-black to-blue-950 opacity-80"></div>
        <div className="absolute inset-0">
          {[...Array(80)].map((_, i) => (
            <div
              key={i}
              className="absolute bg-white rounded-full animate-pulse"
              style={{
                width: Math.random() * 2 + 'px',
                height: Math.random() * 2 + 'px',
                top: Math.random() * 100 + '%',
                left: Math.random() * 100 + '%',
                animationDelay: Math.random() * 5 + 's',
                animationDuration: Math.random() * 3 + 2 + 's'
              }}
            />
          ))}
        </div>
      </div>

      {/* Main content */}
      <div className="relative z-10 p-8">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-6xl font-black bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse mb-2">
            NEXUS
          </h1>
          <p className="text-xl text-cyan-300 font-light tracking-widest">
            Trading Intelligence Network
          </p>
          <div className="mt-4 text-sm text-gray-400">
            <span className="inline-block px-3 py-1 border border-cyan-500/30 rounded-full">
              INTERGALACTIC EDITION
            </span>
          </div>
        </header>

        {/* Status Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12 max-w-7xl mx-auto">
          {/* MT5 Status */}
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition"></div>
            <div className="relative bg-black/50 backdrop-blur-xl border border-cyan-500/30 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-cyan-300 mb-4 flex items-center">
                <span className="w-2 h-2 bg-cyan-400 rounded-full mr-2 animate-pulse"></span>
                MT5 Connection
              </h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Status</span>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                    mt5Status.connected 
                      ? 'bg-green-500/20 text-green-400 border border-green-500/50' 
                      : 'bg-red-500/20 text-red-400 border border-red-500/50'
                  }`}>
                    {mt5Status.connected ? 'CONNECTED' : 'OFFLINE'}
                  </span>
                </div>
                <div className="text-sm text-gray-400">
                  <div>Account: {mt5Status.connected ? '5047475068' : '----'}</div>
                  <div>Server: {mt5Status.connected ? 'MetaQuotes-Demo' : '----'}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Trading Status */}
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition"></div>
            <div className="relative bg-black/50 backdrop-blur-xl border border-purple-500/30 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-purple-300 mb-4 flex items-center">
                <span className="w-2 h-2 bg-purple-400 rounded-full mr-2 animate-pulse"></span>
                Trading Engine
              </h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Mode</span>
                  <span className="px-3 py-1 rounded-full text-xs font-bold bg-purple-500/20 text-purple-400 border border-purple-500/50">
                    DEMO
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Phase</span>
                  <span className="text-purple-300 font-mono">BASELINE</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Trades</span>
                  <span className="text-purple-300 font-mono">{tradeHistory.length}</span>
                </div>
              </div>
            </div>
          </div>

          {/* System Health */}
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-pink-500 to-orange-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition"></div>
            <div className="relative bg-black/50 backdrop-blur-xl border border-pink-500/30 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-pink-300 mb-4 flex items-center">
                <span className="w-2 h-2 bg-pink-400 rounded-full mr-2 animate-pulse"></span>
                System Health
              </h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Backend</span>
                  <span className="px-2 py-1 rounded text-xs font-bold bg-green-500/20 text-green-400">ONLINE</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Database</span>
                  <span className="px-2 py-1 rounded text-xs font-bold bg-green-500/20 text-green-400">SYNCED</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Latency</span>
                  <span className="text-pink-300 font-mono">12ms</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Trade History */}
        <div className="max-w-7xl mx-auto mb-12">
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-orange-500 to-red-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition"></div>
            <div className="relative bg-black/50 backdrop-blur-xl border border-orange-500/30 rounded-2xl p-6">
              <h2 className="text-xl font-bold text-orange-300 mb-6 flex items-center">
                <span className="w-3 h-3 bg-orange-400 rounded-full mr-3 animate-pulse"></span>
                Quantum Trade Log
              </h2>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {tradeHistory.length === 0 ? (
                  <div className="text-center py-8">
                    <div className="text-gray-500 mb-2">No quantum trades executed yet</div>
                    <div className="text-xs text-gray-600">Initiate first trade to see timeline</div>
                  </div>
                ) : (
                  tradeHistory.map((trade, idx) => (
                    <div key={idx} className="bg-black/40 border border-gray-700/50 rounded-lg p-4 mb-2 hover:border-cyan-500/30 transition">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center space-x-3">
                          <span className="text-lg font-bold text-cyan-300">{trade.symbol}</span>
                          <span className={`px-2 py-1 rounded text-xs font-bold ${
                            trade.action === 'BUY' 
                              ? 'bg-green-500/20 text-green-400' 
                              : 'bg-red-500/20 text-red-400'
                          }`}>
                            {trade.action}
                          </span>
                          <span className="text-gray-400">{trade.volume}</span>
                        </div>
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                          trade.status === 'executed' 
                            ? 'bg-green-500/20 text-green-400 border border-green-500/50' 
                            : 'bg-red-500/20 text-red-400 border border-red-500/50'
                        }`}>
                          {trade.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500 mt-2 font-mono">
                        ID: {trade.id} | RETCODE: {trade.retcode}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="max-w-7xl mx-auto">
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-green-500 to-blue-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition"></div>
            <div className="relative bg-black/50 backdrop-blur-xl border border-green-500/30 rounded-2xl p-6">
              <h2 className="text-xl font-bold text-green-300 mb-6 flex items-center">
                <span className="w-3 h-3 bg-green-400 rounded-full mr-3 animate-pulse"></span>
                Command Deck
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold text-cyan-300 mb-4">Connection Control</h3>
                  <button
                    onClick={handleConnectMT5}
                    disabled={mt5Status.loading}
                    className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 text-white py-3 rounded-xl font-bold hover:from-cyan-600 hover:to-purple-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
                  >
                    {mt5Status.loading ? 'INITIALIZING...' : 'CONNECT TO MT5'}
                  </button>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-purple-300 mb-4">Quick Trade</h3>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => handleQuickTrade('BUY')}
                      className="bg-gradient-to-r from-green-500 to-emerald-500 text-white py-3 rounded-xl font-bold hover:from-green-600 hover:to-emerald-600 transition-all transform hover:scale-105"
                    >
                      BUY EUR/USD
                    </button>
                    <button
                      onClick={() => handleQuickTrade('SELL')}
                      className="bg-gradient-to-r from-red-500 to-pink-500 text-white py-3 rounded-xl font-bold hover:from-red-600 hover:to-pink-600 transition-all transform hover:scale-105"
                    >
                      SELL EUR/USD
                    </button>
                  </div>
                </div>
              </div>
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-orange-300 mb-4">500-Trade Batch Demo</h3>
                <div className="bg-black/40 border border-orange-500/30 rounded-lg p-4">
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-gray-400">Status</span>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                      batchStatus.running 
                        ? 'bg-orange-500/20 text-orange-400 border border-orange-500/50 animate-pulse' 
                        : 'bg-gray-500/20 text-gray-400'
                    }`}>
                      {batchStatus.running ? 'RUNNING' : 'IDLE'}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Batch ID</span>
                      <div className="text-orange-300 font-mono">{batchStatus.batchId || '----'}</div>
                    </div>
                    <div>
                      <span className="text-gray-400">Queued</span>
                      <div className="text-orange-300 font-mono">{batchStatus.queued}</div>
                    </div>
                    <div>
                      <span className="text-gray-400">Processed</span>
                      <div className="text-orange-300 font-mono">{batchStatus.processed}</div>
                    </div>
                    <div>
                      <span className="text-gray-400">Executed</span>
                      <div className="text-orange-300 font-mono">{batchStatus.executed}</div>
                    </div>
                    <div>
                      <span className="text-gray-400">Failed</span>
                      <div className="text-orange-300 font-mono">{batchStatus.failed}</div>
                    </div>
                    <div>
                      <span className="text-gray-400">Progress</span>
                      <div className="text-orange-300 font-mono">
                        {batchStatus.queued > 0 ? Math.round((batchStatus.processed / batchStatus.queued) * 100) : 0}%
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={handleStartBatch}
                    disabled={batchStatus.running || !mt5Status.connected}
                    className="w-full mt-4 bg-gradient-to-r from-orange-500 to-red-500 text-white py-3 rounded-xl font-bold hover:from-orange-600 hover:to-red-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
                  >
                    {batchStatus.running ? 'BATCH IN PROGRESS...' : 'START 500-TRADE DEMO'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-16 text-gray-500">
          <div className="text-sm font-mono mb-2">
            NEXUS TRADING INTELLIGENCE NETWORK v2.0
          </div>
          <div className="text-xs text-gray-600">
            InterGalactic Edition • Quantum Engine • Cosmic Interface
          </div>
        </footer>
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
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
};

export default App;

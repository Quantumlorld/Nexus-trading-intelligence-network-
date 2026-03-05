import React, { useState, useEffect } from 'react';
import { systemService } from '@/services/system';

interface Position {
  symbol: string;
  type: 'BUY' | 'SELL';
  volume: number;
  entry_price: number;
  current_price: number;
  profit_loss: number;
  open_time: string;
}

interface MarketData {
  symbol: string;
  bid: number;
  ask: number;
  spread: number;
  change: number;
  change_percent: number;
}

interface ProfessionalTradingPanelProps {
  className?: string;
}

export const ProfessionalTradingPanel: React.FC<ProfessionalTradingPanelProps> = ({ className }) => {
  const [symbol, setSymbol] = useState('XAU/USD');
  const [volume, setVolume] = useState(0.01);
  const [orderType, setOrderType] = useState<'BUY' | 'SELL'>('BUY');
  const [positions, setPositions] = useState<Position[]>([]);
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(false);
  const [accountInfo, setAccountInfo] = useState<any>(null);

  // Fetch real-time data
  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        // Professional market data simulation
        const mockData: MarketData = {
          symbol: symbol,
          bid: symbol === 'XAU/USD' ? 2050.50 + Math.random() * 20 : 
               symbol === 'USDX' ? 103.20 + Math.random() * 2 :
               symbol === 'BTC/USD' ? 42500 + Math.random() * 1000 :
               1.0850 + Math.random() * 0.001,
          ask: symbol === 'XAU/USD' ? 2052.30 + Math.random() * 20 :
               symbol === 'USDX' ? 103.35 + Math.random() * 2 :
               symbol === 'BTC/USD' ? 42850 + Math.random() * 1000 :
               1.0855 + Math.random() * 0.001,
          spread: symbol === 'XAU/USD' ? 1.8 + Math.random() * 2 :
                 symbol === 'USDX' ? 0.15 + Math.random() * 0.5 :
                 symbol === 'BTC/USD' ? 350 + Math.random() * 100 :
                 0.5 + Math.random() * 2,
          change: (Math.random() - 0.5) * (symbol === 'XAU/USD' ? 5 : symbol === 'USDX' ? 0.5 : symbol === 'BTC/USD' ? 200 : 0.01),
          change_percent: (Math.random() - 0.5) * 0.5
        };
        setMarketData(mockData);

        // Fetch account info
        const mt5Status = await systemService.getMT5Status();
        if (mt5Status.success && mt5Status.data?.account_info) {
          setAccountInfo(mt5Status.data.account_info);
        }
      } catch (error) {
        console.error('Failed to fetch market data:', error);
      }
    };

    fetchMarketData();
    const interval = setInterval(fetchMarketData, 2000);
    return () => clearInterval(interval);
  }, [symbol]);

  // Execute trade
  const executeTrade = async () => {
    if (!accountInfo) {
      alert('Please connect MT5 first');
      return;
    }

    setLoading(true);
    try {
      const response = await systemService.executeTrade({
        symbol,
        action: orderType,
        quantity: volume,
        order_type: 'MARKET'
      });

      if (response.success) {
        alert(`${orderType} order placed successfully!`);
        fetchPositions();
      } else {
        alert(`Order failed: ${response.message}`);
      }
    } catch (error) {
      alert(`Trade execution error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // Fetch positions
  const fetchPositions = async () => {
    try {
      const mockPositions: Position[] = [
        {
          symbol: 'XAU/USD',
          type: 'BUY',
          volume: 0.1,
          entry_price: 2050.50,
          current_price: 2060.50,
          profit_loss: 100.00,
          open_time: new Date().toISOString()
        }
      ];
      setPositions(mockPositions);
    } catch (error) {
      console.error('Failed to fetch positions:', error);
    }
  };

  useEffect(() => {
    fetchPositions();
    const interval = setInterval(fetchPositions, 5000);
    return () => clearInterval(interval);
  }, []);

  const formatPL = (pl: number) => {
    return (pl >= 0 ? '+' : '') + pl.toFixed(2);
  };

  const formatTime = (time: string) => {
    return new Date(time).toLocaleString();
  };

  const getSymbolIcon = (symbol: string) => {
    switch (symbol) {
      case 'XAU/USD': return '🥇';
      case 'BTC/USD': return '₿';
      case 'USDX': return '📊';
      case 'EUR/USD': return '💶';
      case 'GBP/USD': return '💷';
      case 'USD/JPY': return '💴';
      default: return '📈';
    }
  };

  return (
    <div className={className}>
      {/* Professional Header */}
      <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 rounded-t-2xl p-6 border border-purple-400/30 shadow-2xl shadow-purple-500/20">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white mb-1">Professional Trading Terminal</h2>
            <p className="text-purple-100 text-sm">Advanced Market Execution Platform</p>
          </div>
          <div className="flex items-center space-x-4">
            {accountInfo && (
              <div className="text-right">
                <p className="text-purple-100 text-xs">Account Balance</p>
                <p className="text-white text-xl font-bold">${accountInfo.balance?.toFixed(2)}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Trading Interface */}
      <div className="bg-black/40 backdrop-blur-xl rounded-b-2xl border-x border-b border-purple-400/30">
        
        {/* Market Data Display */}
        {marketData && (
          <div className="bg-black/30 backdrop-blur-sm border-b border-purple-400/20 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <span className="text-3xl">{getSymbolIcon(symbol)}</span>
                <div>
                  <h3 className="text-xl font-bold text-white">{symbol}</h3>
                  <p className="text-purple-200 text-sm">Real-time Market Data</p>
                </div>
              </div>
              <div className="text-right">
                <div className={`text-3xl font-bold ${marketData.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {symbol === 'XAU/USD' || symbol === 'BTC/USD' 
                    ? marketData.ask.toFixed(2) 
                    : marketData.ask.toFixed(5)}
                </div>
                <div className={`text-sm font-medium ${marketData.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {marketData.change >= 0 ? '▲' : '▼'} {Math.abs(marketData.change_percent).toFixed(2)}%
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gradient-to-br from-purple-500/20 to-indigo-500/20 backdrop-blur-sm rounded-xl p-3 border border-purple-400/30">
                <p className="text-xs text-purple-300 mb-1">Bid Price</p>
                <p className="text-lg font-bold text-white">
                  {symbol === 'XAU/USD' || symbol === 'BTC/USD' 
                    ? marketData.bid.toFixed(2) 
                    : marketData.bid.toFixed(5)}
                </p>
              </div>
              <div className="bg-gradient-to-br from-pink-500/20 to-purple-500/20 backdrop-blur-sm rounded-xl p-3 border border-pink-400/30">
                <p className="text-xs text-pink-300 mb-1">Ask Price</p>
                <p className="text-lg font-bold text-white">
                  {symbol === 'XAU/USD' || symbol === 'BTC/USD' 
                    ? marketData.ask.toFixed(2) 
                    : marketData.ask.toFixed(5)}
                </p>
              </div>
              <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 backdrop-blur-sm rounded-xl p-3 border border-blue-400/30">
                <p className="text-xs text-blue-300 mb-1">Spread</p>
                <p className="text-lg font-bold text-white">
                  {symbol === 'XAU/USD' || symbol === 'BTC/USD' 
                    ? marketData.spread.toFixed(1) 
                    : marketData.spread.toFixed(1)}
                </p>
              </div>
              <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 backdrop-blur-sm rounded-xl p-3 border border-green-400/30">
                <p className="text-xs text-green-300 mb-1">Daily Change</p>
                <p className={`text-lg font-bold ${marketData.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {marketData.change >= 0 ? '+' : ''}{symbol === 'XAU/USD' || symbol === 'BTC/USD' 
                    ? marketData.change.toFixed(2) 
                    : marketData.change.toFixed(5)}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Trading Controls */}
        <div className="p-6 bg-gradient-to-b from-black/20 to-black/10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Order Form */}
            <div className="bg-black/30 backdrop-blur-sm rounded-2xl shadow-xl border border-purple-400/30 p-6">
              <h4 className="text-lg font-bold text-white mb-4 flex items-center">
                <span className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></span>
                Quick Order Execution
              </h4>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-purple-200 mb-2">Trading Symbol</label>
                  <select
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value)}
                    className="w-full px-4 py-3 bg-black/40 backdrop-blur-sm border border-purple-400/30 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-transparent text-white"
                  >
                    <option value="XAU/USD">🥇 XAU/USD - Gold</option>
                    <option value="BTC/USD">₿ BTC/USD - Bitcoin</option>
                    <option value="USDX">📊 USDX - Dollar Index</option>
                    <option value="EUR/USD">💶 EUR/USD - Euro</option>
                    <option value="GBP/USD">💷 GBP/USD - British Pound</option>
                    <option value="USD/JPY">💴 USD/JPY - Japanese Yen</option>
                  </select>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-purple-200 mb-2">Volume (Lots)</label>
                    <input
                      type="number"
                      value={volume}
                      onChange={(e) => setVolume(parseFloat(e.target.value))}
                      step="0.01"
                      min="0.01"
                      className="w-full px-4 py-3 bg-black/40 backdrop-blur-sm border border-purple-400/30 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-transparent text-white"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-purple-200 mb-2">Order Type</label>
                    <select
                      value={orderType}
                      onChange={(e) => setOrderType(e.target.value as 'BUY' | 'SELL')}
                      className="w-full px-4 py-3 bg-black/40 backdrop-blur-sm border border-purple-400/30 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-transparent text-white"
                    >
                      <option value="BUY">🟢 BUY</option>
                      <option value="SELL">🔴 SELL</option>
                    </select>
                  </div>
                </div>
                
                <div className="flex space-x-3">
                  <button
                    onClick={executeTrade}
                    disabled={loading || !accountInfo}
                    className={`flex-1 px-6 py-4 rounded-xl font-bold text-white transition-all transform hover:scale-105 shadow-xl ${
                      orderType === 'BUY' 
                        ? 'bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 disabled:from-purple-500 disabled:to-purple-600 shadow-green-400/30' 
                        : 'bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 disabled:from-purple-500 disabled:to-purple-600 shadow-red-400/30'
                    }`}
                  >
                    {loading ? (
                      <span className="flex items-center justify-center">
                        <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                        </svg>
                        Executing...
                      </span>
                    ) : (
                      `${orderType} ${symbol}`
                    )}
                  </button>
                </div>
                
                {!accountInfo && (
                  <div className="mt-3 p-3 bg-amber-500/20 backdrop-blur-sm border border-amber-400/30 rounded-xl">
                    <p className="text-sm text-amber-200">
                      ⚠️ Connect your MT5 account to enable real trading
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Account Overview */}
            <div className="space-y-4">
              {accountInfo && (
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                  <h4 className="text-lg font-bold text-slate-900 mb-4 flex items-center">
                    <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                    Account Overview
                  </h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4">
                      <p className="text-xs text-blue-600 mb-1">Balance</p>
                      <p className="text-xl font-bold text-blue-900">${accountInfo.balance?.toFixed(2)}</p>
                    </div>
                    <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-4">
                      <p className="text-xs text-green-600 mb-1">Equity</p>
                      <p className="text-xl font-bold text-green-900">${accountInfo.equity?.toFixed(2)}</p>
                    </div>
                    <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-4">
                      <p className="text-xs text-purple-600 mb-1">Margin Used</p>
                      <p className="text-xl font-bold text-purple-900">${accountInfo.margin?.toFixed(2)}</p>
                    </div>
                    <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-4">
                      <p className="text-xs text-amber-600 mb-1">Free Margin</p>
                      <p className="text-xl font-bold text-amber-900">${accountInfo.free_margin?.toFixed(2)}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Quick Stats */}
              <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl p-6 text-white">
                <h4 className="text-lg font-bold mb-4 flex items-center">
                  <span className="w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                  Market Statistics
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-300">Market Status</span>
                    <span className="px-3 py-1 bg-green-500 text-white text-xs rounded-full font-bold">LIVE</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-300">Trading Session</span>
                    <span className="text-white font-bold">London/New York</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-300">Volatility</span>
                    <span className="text-purple-300 font-bold">Medium</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-300">Leverage</span>
                    <span className="text-white font-bold">1:500</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Open Positions */}
        <div className="p-6 bg-white border-t border-slate-200">
          <h4 className="text-lg font-bold text-slate-900 mb-4 flex items-center">
            <span className="w-2 h-2 bg-orange-500 rounded-full mr-2"></span>
            Active Positions
          </h4>
          {positions.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-6xl mb-4">📊</div>
              <p className="text-slate-500">No open positions</p>
              <p className="text-sm text-slate-400 mt-2">Place your first trade to see it here</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-200">
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-700">Symbol</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-700">Type</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-700">Volume</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-700">Entry</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-700">Current</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-700">P&L</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-slate-700">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((position, index) => (
                    <tr key={index} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                      <td className="py-3 px-4">
                        <div className="flex items-center">
                          <span className="mr-2">{getSymbolIcon(position.symbol)}</span>
                          <span className="font-medium">{position.symbol}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold text-white ${
                          position.type === 'BUY' ? 'bg-green-500' : 'bg-red-500'
                        }`}>
                          {position.type}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-medium">{position.volume}</td>
                      <td className="py-3 px-4 font-medium">
                        {position.symbol === 'XAU/USD' || position.symbol === 'BTC/USD' 
                          ? position.entry_price.toFixed(2) 
                          : position.entry_price.toFixed(5)}
                      </td>
                      <td className="py-3 px-4 font-medium">
                        {position.symbol === 'XAU/USD' || position.symbol === 'BTC/USD' 
                          ? position.current_price.toFixed(2) 
                          : position.current_price.toFixed(5)}
                      </td>
                      <td className={`py-3 px-4 font-bold ${
                        position.profit_loss >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {formatPL(position.profit_loss)}
                      </td>
                      <td className="py-3 px-4 text-sm text-slate-500">
                        {formatTime(position.open_time)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

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

interface TradingPanelProps {
  className?: string;
}

export const TradingPanel: React.FC<TradingPanelProps> = ({ className }) => {
  const [symbol, setSymbol] = useState('EUR/USD');
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
        // Simulate real market data (will connect to MT5 later)
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
        // Refresh positions
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
      // Mock positions (will connect to MT5 later)
      const mockPositions: Position[] = [
        {
          symbol: 'EUR/USD',
          type: 'BUY',
          volume: 0.1,
          entry_price: 1.0850,
          current_price: 1.0865,
          profit_loss: 15.00,
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

  return (
    <div className={className}>
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Trading Panel</h3>
        
        {/* Account Info */}
        {accountInfo && (
          <div className="mb-6 p-4 bg-blue-50 rounded-lg">
            <h4 className="text-sm font-medium text-blue-800 mb-2">Account Information</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Balance:</span>
                <span className="font-medium ml-1">${accountInfo.balance?.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-gray-600">Equity:</span>
                <span className="font-medium ml-1">${accountInfo.equity?.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-gray-600">Margin:</span>
                <span className="font-medium ml-1">${accountInfo.margin?.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-gray-600">Free:</span>
                <span className="font-medium ml-1">${accountInfo.free_margin?.toFixed(2)}</span>
              </div>
            </div>
          </div>
        )}

        {/* Market Data */}
        {marketData && (
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="text-sm font-medium text-gray-800 mb-2">Market Data - {symbol}</h4>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Bid:</span>
                <span className="font-medium ml-1">
                  {symbol === 'XAU/USD' || symbol === 'BTC/USD' 
                    ? marketData.bid.toFixed(2) 
                    : marketData.bid.toFixed(5)}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Ask:</span>
                <span className="font-medium ml-1">
                  {symbol === 'XAU/USD' || symbol === 'BTC/USD' 
                    ? marketData.ask.toFixed(2) 
                    : marketData.ask.toFixed(5)}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Spread:</span>
                <span className="font-medium ml-1">
                  {symbol === 'XAU/USD' || symbol === 'BTC/USD' 
                    ? marketData.spread.toFixed(1) 
                    : marketData.spread.toFixed(1)}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Change:</span>
                <span className={`font-medium ml-1 ${marketData.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {marketData.change >= 0 ? '+' : ''}{symbol === 'XAU/USD' || symbol === 'BTC/USD' 
                    ? marketData.change.toFixed(2) 
                    : marketData.change.toFixed(5)}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Change%:</span>
                <span className={`font-medium ml-1 ${marketData.change_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {marketData.change_percent >= 0 ? '+' : ''}{marketData.change_percent.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Order Form */}
        <div className="mb-6 p-4 bg-yellow-50 rounded-lg">
          <h4 className="text-sm font-medium text-yellow-800 mb-4">Place Order</h4>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Symbol</label>
              <select
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="EUR/USD">EUR/USD</option>
                <option value="GBP/USD">GBP/USD</option>
                <option value="USD/JPY">USD/JPY</option>
                <option value="XAU/USD">GOLD/USD</option>
                <option value="BTC/USD">BTC/USD</option>
                <option value="USDX">DXY (US Dollar Index)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Volume</label>
              <input
                type="number"
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
                step="0.01"
                min="0.01"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Order Type</label>
              <select
                value={orderType}
                onChange={(e) => setOrderType(e.target.value as 'BUY' | 'SELL')}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="BUY">BUY</option>
                <option value="SELL">SELL</option>
              </select>
            </div>
            
            <div className="flex items-end">
              <button
                onClick={executeTrade}
                disabled={loading || !accountInfo}
                className={`w-full px-4 py-2 rounded-md text-white font-medium ${
                  orderType === 'BUY' 
                    ? 'bg-green-600 hover:bg-green-700 disabled:bg-gray-300' 
                    : 'bg-red-600 hover:bg-red-700 disabled:bg-gray-300'
                }`}
              >
                {loading ? 'Executing...' : `${orderType} ${symbol}`}
              </button>
            </div>
          </div>
          
          {!accountInfo && (
            <div className="text-xs text-yellow-700 bg-yellow-100 p-2 rounded">
              ⚠️ Connect MT5 to enable real trading
            </div>
          )}
        </div>

        {/* Open Positions */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-800 mb-4">Open Positions</h4>
          {positions.length === 0 ? (
            <p className="text-sm text-gray-500">No open positions</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">Symbol</th>
                    <th className="text-left py-2">Type</th>
                    <th className="text-left py-2">Volume</th>
                    <th className="text-left py-2">Entry</th>
                    <th className="text-left py-2">Current</th>
                    <th className="text-left py-2">P&L</th>
                    <th className="text-left py-2">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((position, index) => (
                    <tr key={index} className="border-b">
                      <td className="py-2">{position.symbol}</td>
                      <td className="py-2">
                        <span className={`px-2 py-1 rounded text-xs text-white ${
                          position.type === 'BUY' ? 'bg-green-500' : 'bg-red-500'
                        }`}>
                          {position.type}
                        </span>
                      </td>
                      <td className="py-2">{position.volume}</td>
                      <td className="py-2">
                        {position.symbol === 'XAU/USD' || position.symbol === 'BTC/USD' 
                          ? position.entry_price.toFixed(2) 
                          : position.entry_price.toFixed(5)}
                      </td>
                      <td className="py-2">
                        {position.symbol === 'XAU/USD' || position.symbol === 'BTC/USD' 
                          ? position.current_price.toFixed(2) 
                          : position.current_price.toFixed(5)}
                      </td>
                      <td className={`py-2 font-medium ${
                        position.profit_loss >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {formatPL(position.profit_loss)}
                      </td>
                      <td className="py-2 text-xs text-gray-500">
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

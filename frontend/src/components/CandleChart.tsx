import React, { useState, useEffect } from 'react';
import { candleService, CandleData } from '@/services/candles';

interface CandleChartProps {
  className?: string;
}

export const CandleChart: React.FC<CandleChartProps> = ({ className }) => {
  const [candles, setCandles] = useState<CandleData[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('EUR/USD');
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1h');
  const [loading, setLoading] = useState(true);

  const fetchCandles = async () => {
    try {
      setLoading(true);
      const response = await candleService.getCandles(selectedSymbol, selectedTimeframe, 50);
      
      if (response.success && response.data) {
        setCandles(response.data.candles);
      }
    } catch (error) {
      console.error('Failed to fetch candles:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCandles();
  }, [selectedSymbol, selectedTimeframe]);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatPrice = (price: number) => {
    return selectedSymbol === 'BTC/USD' ? price.toFixed(0) : price.toFixed(4);
  };

  if (loading) {
    return (
      <div className={className}>
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-4">Candle Chart</h3>
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={className}>
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Candle Chart</h3>
          <div className="flex space-x-2">
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="px-2 py-1 text-sm border rounded"
            >
              {candleService.getAvailableSymbols().map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
            <select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="px-2 py-1 text-sm border rounded"
            >
              {candleService.getAvailableTimeframes().map(timeframe => (
                <option key={timeframe} value={timeframe}>{timeframe}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Simple Candle Display */}
        <div className="space-y-2">
          <div className="grid grid-cols-6 text-xs font-medium text-gray-500 pb-2 border-b">
            <div>Time</div>
            <div>Open</div>
            <div>High</div>
            <div>Low</div>
            <div>Close</div>
            <div>Volume</div>
          </div>
          
          <div className="max-h-64 overflow-y-auto space-y-1">
            {candles.slice().reverse().map((candle, index) => (
              <div key={index} className="grid grid-cols-6 text-xs hover:bg-gray-50 p-1 rounded">
                <div className="font-mono text-gray-600">
                  {formatTimestamp(candle.timestamp)}
                </div>
                <div className={candle.close > candle.open ? 'text-green-600' : 'text-red-600'}>
                  {formatPrice(candle.open)}
                </div>
                <div className="text-blue-600">
                  {formatPrice(candle.high)}
                </div>
                <div className="text-purple-600">
                  {formatPrice(candle.low)}
                </div>
                <div className={candle.close > candle.open ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
                  {formatPrice(candle.close)}
                </div>
                <div className="text-gray-500">
                  {candle.volume.toFixed(0)}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Summary Stats */}
        <div className="border-t pt-4 mt-4">
          <h4 className="text-sm font-medium mb-2">Summary</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Latest Price:</span>
              <span className="ml-2 font-medium">
                {candles.length > 0 ? formatPrice(candles[candles.length - 1].close) : 'N/A'}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Candles Shown:</span>
              <span className="ml-2 font-medium">{candles.length}</span>
            </div>
            <div>
              <span className="text-gray-500">Symbol:</span>
              <span className="ml-2 font-medium">{selectedSymbol}</span>
            </div>
            <div>
              <span className="text-gray-500">Timeframe:</span>
              <span className="ml-2 font-medium">{selectedTimeframe}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CandleChart;

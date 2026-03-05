import apiService from './api';
import { ApiResponse } from '@/types/api';

export interface CandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface CandleResponse {
  success: boolean;
  candles: CandleData[];
  symbol: string;
  timeframe: string;
}

export class CandleService {
  // Get candle data
  async getCandles(symbol: string, timeframe: string = '1h', limit: number = 100): Promise<ApiResponse<CandleResponse>> {
    return apiService.get<CandleResponse>('/candles', { symbol, timeframe, limit });
  }

  // Get 9H candles specifically
  async get9HCandles(symbol: string, limit: number = 100): Promise<ApiResponse<CandleResponse>> {
    return this.getCandles(symbol, '9h', limit);
  }

  // Get available symbols
  getAvailableSymbols(): string[] {
    return ['EUR/USD', 'BTC/USD'];
  }

  // Get available timeframes
  getAvailableTimeframes(): string[] {
    return ['1h', '5m', '9h'];
  }

  // Validate symbol
  isValidSymbol(symbol: string): boolean {
    return this.getAvailableSymbols().includes(symbol);
  }

  // Validate timeframe
  isValidTimeframe(timeframe: string): boolean {
    return this.getAvailableTimeframes().includes(timeframe);
  }
}

export const candleService = new CandleService();
export default candleService;

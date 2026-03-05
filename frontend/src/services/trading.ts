import apiService from './api';
import {
  TradeRequest,
  TradeResponse,
  Position,
  Balance,
  RiskStats,
  TradeLedger,
  ApiResponse,
} from '@/types/api';

export class TradingService {
  // Trade Execution
  async executeTrade(trade: TradeRequest): Promise<ApiResponse<TradeResponse>> {
    return apiService.post<TradeResponse>('/trade', trade);
  }

  // Get user positions
  async getPositions(): Promise<ApiResponse<Position[]>> {
    return apiService.get<Position[]>('/trading/positions');
  }

  // Get user balance
  async getBalance(): Promise<ApiResponse<Balance>> {
    return apiService.get<Balance>('/trading/balance');
  }

  // Get risk statistics
  async getRiskStats(): Promise<ApiResponse<RiskStats>> {
    return apiService.get<RiskStats>('/trading/risk-stats');
  }

  // Get trade ledger
  async getTradeLedger(params?: {
    limit?: number;
    offset?: number;
    status?: string;
  }): Promise<ApiResponse<{ trades: TradeLedger[]; total: number }>> {
    return apiService.get<{ trades: TradeLedger[]; total: number }>('/trading/ledger', params);
  }

  // Get trade history
  async getTradeHistory(params?: {
    limit?: number;
    offset?: number;
    symbol?: string;
  }): Promise<ApiResponse<{ trades: TradeResponse[]; total: number }>> {
    return apiService.get<{ trades: TradeResponse[]; total: number }>('/trading/history', params);
  }

  // Cancel trade
  async cancelTrade(tradeId: string): Promise<ApiResponse<void>> {
    return apiService.delete<void>(`/trading/trades/${tradeId}`);
  }

  // Modify position
  async modifyPosition(
    positionId: string,
    modifications: {
      stop_loss?: number;
      take_profit?: number;
    }
  ): Promise<ApiResponse<Position>> {
    return apiService.put<Position>(`/trading/positions/${positionId}`, modifications);
  }

  // Close position
  async closePosition(positionId: string): Promise<ApiResponse<void>> {
    return apiService.delete<void>(`/trading/positions/${positionId}`);
  }

  // Get available symbols
  async getSymbols(): Promise<ApiResponse<string[]>> {
    return apiService.get<string[]>('/trading/symbols');
  }

  // Get symbol info
  async getSymbolInfo(symbol: string): Promise<ApiResponse<{
    symbol: string;
    name: string;
    min_lot_size: number;
    max_lot_size: number;
    lot_step: number;
    min_stop_loss: number;
    max_stop_loss: number;
    swap_long: number;
    swap_short: number;
  }>> {
    return apiService.get(`/trading/symbols/${symbol}`);
  }

  // Calculate margin
  async calculateMargin(params: {
    symbol: string;
    volume: number;
    leverage: number;
  }): Promise<ApiResponse<{
    required_margin: number;
    commission: number;
    swap: number;
  }>> {
    return apiService.post('/trading/calculate-margin', params);
  }
}

export const tradingService = new TradingService();

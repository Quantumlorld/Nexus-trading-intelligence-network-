import apiService from './api';
import { ApiResponse } from '@/types/api';

export interface SystemStatus {
  trading_enabled: boolean;
  broker_connected: boolean;
  db_connected: boolean;
  uptime: string;
  start_time: string;
  supported_symbols: string[];
  supported_timeframes: string[];
  '9h_candles_available': Record<string, number>;
}

export interface HealthStatus {
  status: string;
  db_status: string;
  broker_status: string;
  trading_enabled: boolean;
  uptime: string;
}

export class SystemService {
  // Get system health
  async getHealth(): Promise<ApiResponse<HealthStatus>> {
    return apiService.get<HealthStatus>('/health');
  }

  // Get detailed system status
  async getSystemStatus(): Promise<ApiResponse<SystemStatus>> {
    return apiService.get<SystemStatus>('/admin/system-status');
  }

  // Enable trading
  async enableTrading(): Promise<ApiResponse<{ success: boolean; message: string }>> {
    return apiService.post<{ success: boolean; message: string }>('/admin/enable-trading');
  }

  // Disable trading
  async disableTrading(): Promise<ApiResponse<{ success: boolean; message: string }>> {
    return apiService.post<{ success: boolean; message: string }>('/admin/disable-trading');
  }

  // Simulate broker failure
  async simulateBrokerFailure(): Promise<ApiResponse<{ success: boolean; message: string }>> {
    return apiService.post<{ success: boolean; message: string }>('/admin/simulate-broker-failure');
  }

  // Simulate broker recovery
  async simulateBrokerRecovery(): Promise<ApiResponse<{ success: boolean; message: string }>> {
    return apiService.post<{ success: boolean; message: string }>('/admin/simulate-broker-recovery');
  }

  // Get metrics
  async getMetrics(): Promise<string> {
    const response = await fetch('http://localhost:8000/metrics');
    return await response.text();
  }

  // Connect to MT5
  async connectMT5(account: number, password: string, server: string): Promise<ApiResponse<{ success: boolean; message: string }>> {
    return apiService.post<{ success: boolean; message: string }>('/admin/mt5-connect', {
      account,
      password,
      server
    });
  }

  // Get MT5 status
  async getMT5Status(): Promise<ApiResponse<{ connected: boolean; message?: string; account_info?: any; server?: string; account?: number }>> {
    return apiService.get('/admin/mt5-status');
  }

  // Execute trade
  async executeTrade(tradeRequest: { symbol: string; action: string; quantity: number; order_type: string }): Promise<ApiResponse<{ success: boolean; message: string; trade_id?: string }>> {
    return apiService.post('/admin/execute-trade', tradeRequest);
  }

  // Get demo progress
  async getDemoProgress(): Promise<ApiResponse<{
    trade_count: number;
    current_phase: string;
    phase_name: string;
    phase_progress: number;
    adaptive_learning: boolean;
    demo_mode: boolean;
    next_phase_features: string[];
  }>> {
    return apiService.get('/admin/demo/progress');
  }

  // Start demo trading
  async startDemoTrading(): Promise<ApiResponse<{ success: boolean; message: string; plan?: any }>> {
    return apiService.post('/admin/demo/start');
  }

  // Universal MT5 methods
  async getUniversalMT5Status(): Promise<ApiResponse<UniversalMT5Status>> {
    return apiService.get('/admin/mt5-universal-status');
  }

  async connectUniversalMT5(): Promise<ApiResponse<{ success: boolean; message: string; connector_type?: string; account_summary?: any }>> {
    return apiService.post('/admin/mt5-universal-connect');
  }

  async placeUniversalMT5Order(orderRequest: { symbol: string; volume: number; order_type: string; price?: number }): Promise<ApiResponse<{ success: boolean; message: string; order_id?: number }>> {
    return apiService.post('/admin/mt5-universal-order', orderRequest);
  }

  async getUniversalMT5Positions(): Promise<ApiResponse<{ success: boolean; positions: any[]; count: number; connector_type: string }>> {
    return apiService.get('/admin/mt5-universal-positions');
  }
}

// Type definitions
interface UniversalMT5Status {
  connected: boolean;
  connector_type?: string;
  account_summary?: {
    balance: number;
    equity: number;
    margin: number;
    free_margin: number;
    profit: number;
    leverage: number;
    open_positions: number;
    total_profit: number;
    total_loss: number;
    net_profit: number;
    margin_level: number;
    server: string;
    terminal_info?: {
      version: number;
      build: number;
      company: string;
    };
  };
  available_symbols?: string[];
  open_positions?: any[];
  demo_mode?: boolean;
  trade_count?: number;
  current_phase?: string;
  message?: string;
}

export const systemService = new SystemService();
export default systemService;

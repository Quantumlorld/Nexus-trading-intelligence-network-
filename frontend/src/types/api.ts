// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Authentication Types
export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface User {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'trader';
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

// Trading Types
export interface TradeRequest {
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  leverage?: number;
  stop_loss?: number;
  take_profit?: number;
}

export interface TradeResponse {
  id: string;
  user_id: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  executed_quantity: number;
  price: number;
  status: TradeStatus;
  created_at: string;
  updated_at: string;
  stop_loss?: number;
  take_profit?: number;
  slippage?: number;
  commission?: number;
}

export enum TradeStatus {
  PENDING = 'PENDING',
  PARTIALLY_FILLED = 'PARTIALLY_FILLED',
  FILLED = 'FILLED',
  REJECTED = 'REJECTED',
  CANCELLED = 'CANCELLED',
}

// Portfolio Types
export interface Position {
  id: string;
  user_id: string;
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  margin_used: number;
  created_at: string;
  updated_at: string;
}

export interface Balance {
  user_id: string;
  broker_balance: number;
  available_balance: number;
  margin_used: number;
  equity: number;
  free_margin: number;
  margin_level: number;
  updated_at: string;
}

// Risk Management Types
export interface RiskStats {
  user_id: string;
  daily_trades: number;
  daily_volume: number;
  daily_loss: number;
  daily_profit: number;
  max_daily_loss: number;
  exposure: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  updated_at: string;
}

// Reconciliation Types
export interface ReconciliationLog {
  id: string;
  user_id: string;
  discrepancy_type: 'QUANTITY' | 'PRICE' | 'POSITION';
  ledger_value: number;
  broker_value: number;
  difference: number;
  risk_impact: 'LOW' | 'MEDIUM' | 'HIGH';
  action: 'AUTO_CORRECTED' | 'MANUAL_REVIEW' | 'BROKER_SYNCED' | 'EMERGENCY_STOP';
  created_at: string;
  resolved_at?: string;
}

export interface ReconciliationSummary {
  total_discrepancies: number;
  low_risk: number;
  medium_risk: number;
  high_risk: number;
  auto_corrected: number;
  manual_review: number;
  last_reconciliation: string;
}

// Ledger Types
export interface TradeLedger {
  id: string;
  trade_id: string;
  user_id: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  status: TradeStatus;
  broker_reference?: string;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

// Alert Types
export interface Alert {
  id: string;
  user_id: string;
  type: 'SLIPPAGE' | 'PARTIAL_FILL' | 'REJECTION' | 'RISK_LIMIT' | 'RECONCILIATION';
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  message: string;
  data?: Record<string, any>;
  is_read: boolean;
  created_at: string;
}

// System Status Types
export interface SystemHealth {
  status: 'HEALTHY' | 'DEGRADED' | 'DOWN';
  services: {
    api: 'UP' | 'DOWN';
    database: 'UP' | 'DOWN';
    broker: 'UP' | 'DOWN';
    reconciliation: 'ACTIVE' | 'INACTIVE';
  };
  uptime: number;
  last_check: string;
}

export interface TradingControl {
  trading_enabled: boolean;
  emergency_stop: boolean;
  maintenance_mode: boolean;
  message?: string;
  updated_at: string;
}

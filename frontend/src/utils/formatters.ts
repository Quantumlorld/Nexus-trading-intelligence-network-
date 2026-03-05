import { format } from 'date-fns';

// Currency formatting
export const formatCurrency = (amount: number, currency = 'USD'): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount);
};

// Percentage formatting
export const formatPercentage = (value: number, decimals = 2): string => {
  return `${value.toFixed(decimals)}%`;
};

// Number formatting with commas
export const formatNumber = (value: number, decimals = 2): string => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

// Date formatting
export const formatDate = (date: string | Date, formatStr = 'MMM dd, yyyy HH:mm'): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return format(dateObj, formatStr);
};

// Time ago formatting
export const formatTimeAgo = (date: string | Date): string => {
  const now = new Date();
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const diffInSeconds = Math.floor((now.getTime() - dateObj.getTime()) / 1000);

  if (diffInSeconds < 60) return 'just now';
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)}d ago`;
  
  return formatDate(dateObj);
};

// Trade status formatting
export const formatTradeStatus = (status: string): string => {
  return status.replace(/_/g, ' ');
};

// Risk level formatting with colors
export const formatRiskLevel = (level: string): { text: string; color: string } => {
  switch (level) {
    case 'LOW':
      return { text: 'Low Risk', color: 'text-green-600' };
    case 'MEDIUM':
      return { text: 'Medium Risk', color: 'text-yellow-600' };
    case 'HIGH':
      return { text: 'High Risk', color: 'text-red-600' };
    default:
      return { text: level, color: 'text-gray-600' };
  }
};

// Symbol formatting
export const formatSymbol = (symbol: string): string => {
  return symbol.replace('/', '');
};

// Leverage formatting
export const formatLeverage = (leverage: number): string => {
  return `1:${leverage}`;
};

// P&L formatting with sign
export const formatPnL = (value: number): { text: string; color: string } => {
  const formatted = formatCurrency(Math.abs(value));
  const sign = value >= 0 ? '+' : '-';
  const color = value >= 0 ? 'text-green-600' : 'text-red-600';
  
  return {
    text: `${sign}${formatted}`,
    color,
  };
};

// Volume formatting
export const formatVolume = (volume: number): string => {
  if (volume >= 1000000) {
    return `${(volume / 1000000).toFixed(1)}M`;
  }
  if (volume >= 1000) {
    return `${(volume / 1000).toFixed(1)}K`;
  }
  return formatNumber(volume, 0);
};

// Price formatting
export const formatPrice = (price: number, decimals = 5): string => {
  return formatNumber(price, decimals);
};

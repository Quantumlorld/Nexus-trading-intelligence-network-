import { useMutation, useQuery } from '@tanstack/react-query';
import { tradingService } from '@/services/trading';
import { TradeRequest, Position } from '@/types/api';
import { toast } from 'react-hot-toast';

export const useExecuteTrade = () => {
  return useMutation({
    mutationFn: (tradeData: TradeRequest) => tradingService.executeTrade(tradeData),
    onSuccess: (response) => {
      if (response.success) {
        toast.success('Trade executed successfully!');
      } else {
        toast.error(response.error || 'Trade execution failed');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Trade execution failed');
    },
  });
};

export const usePositions = () => {
  return useQuery({
    queryKey: ['positions'],
    queryFn: () => tradingService.getPositions(),
    refetchInterval: 30000, // Refresh every 30 seconds
    onError: (error: any) => {
      toast.error(error.message || 'Failed to load positions');
    },
  });
};

export const useBalance = () => {
  return useQuery({
    queryKey: ['balance'],
    queryFn: () => tradingService.getBalance(),
    refetchInterval: 15000, // Refresh every 15 seconds
    onError: (error: any) => {
      toast.error(error.message || 'Failed to load balance');
    },
  });
};

export const useRiskStats = () => {
  return useQuery({
    queryKey: ['risk-stats'],
    queryFn: () => tradingService.getRiskStats(),
    refetchInterval: 60000, // Refresh every minute
    onError: (error: any) => {
      toast.error(error.message || 'Failed to load risk statistics');
    },
  });
};

export const useTradeHistory = () => {
  return useQuery({
    queryKey: ['trade-history'],
    queryFn: () => tradingService.getTradeHistory(),
    onError: (error: any) => {
      toast.error(error.message || 'Failed to load trade history');
    },
  });
};

export const useCancelTrade = () => {
  return useMutation({
    mutationFn: (tradeId: string) => tradingService.cancelTrade(tradeId),
    onSuccess: (response) => {
      if (response.success) {
        toast.success('Trade cancelled successfully!');
      } else {
        toast.error(response.error || 'Trade cancellation failed');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Trade cancellation failed');
    },
  });
};

export const useClosePosition = () => {
  return useMutation({
    mutationFn: (positionId: string) => tradingService.closePosition(positionId),
    onSuccess: (response) => {
      if (response.success) {
        toast.success('Position closed successfully!');
      } else {
        toast.error(response.error || 'Position closure failed');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Position closure failed');
    },
  });
};

export const useSymbolInfo = (symbol: string) => {
  return useQuery({
    queryKey: ['symbol-info', symbol],
    queryFn: () => tradingService.getSymbolInfo(symbol),
    enabled: !!symbol,
    onError: (error: any) => {
      toast.error(error.message || 'Failed to load symbol information');
    },
  });
};

export const useCalculateMargin = () => {
  return useMutation({
    mutationFn: (params: { symbol: string; quantity: number; leverage: number }) =>
      tradingService.calculateMargin(params),
    onError: (error: any) => {
      toast.error(error.message || 'Margin calculation failed');
    },
  });
};

import { useMutation, useQuery } from '@tanstack/react-query';
import { reconciliationService } from '@/services/reconciliation';
import { toast } from 'react-hot-toast';

export const useReconciliationLogs = (filters?: { status?: string }) => {
  return useQuery({
    queryKey: ['reconciliation-logs', filters],
    queryFn: () => reconciliationService.getLogs(filters),
    refetchInterval: 30000, // Refresh every 30 seconds
    onError: (error: any) => {
      toast.error(error.message || 'Failed to load reconciliation logs');
    },
  });
};

export const useReconciliationSummary = () => {
  return useQuery({
    queryKey: ['reconciliation-summary'],
    queryFn: () => reconciliationService.getSummary(),
    refetchInterval: 60000, // Refresh every minute
    onError: (error: any) => {
      toast.error(error.message || 'Failed to load reconciliation summary');
    },
  });
};

export const useTriggerReconciliation = () => {
  return useMutation({
    mutationFn: () => reconciliationService.triggerManual(),
    onSuccess: (response) => {
      if (response.success) {
        toast.success('Reconciliation triggered successfully!');
      } else {
        toast.error(response.error || 'Failed to trigger reconciliation');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to trigger reconciliation');
    },
  });
};

export const useResolveDiscrepancy = () => {
  return useMutation({
    mutationFn: (discrepancyId: string) => reconciliationService.resolveDiscrepancy(discrepancyId),
    onSuccess: (response) => {
      if (response.success) {
        toast.success('Discrepancy resolved successfully!');
      } else {
        toast.error(response.error || 'Failed to resolve discrepancy');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to resolve discrepancy');
    },
  });
};

export const useReconciliationStats = () => {
  return useQuery({
    queryKey: ['reconciliation-stats'],
    queryFn: () => reconciliationService.getStats(),
    refetchInterval: 120000, // Refresh every 2 minutes
    onError: (error: any) => {
      toast.error(error.message || 'Failed to load reconciliation statistics');
    },
  });
};

export const useExportReconciliationReport = () => {
  return useMutation({
    mutationFn: (params: { start_date: string; end_date: string; format: 'csv' | 'pdf' }) =>
      reconciliationService.exportReport(params),
    onSuccess: (response) => {
      if (response.success && response.data?.download_url) {
        // Create download link
        const link = document.createElement('a');
        link.href = response.data.download_url;
        link.download = `reconciliation-report.${params.format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        toast.success('Report exported successfully!');
      } else {
        toast.error(response.error || 'Report export failed');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Report export failed');
    },
  });
};

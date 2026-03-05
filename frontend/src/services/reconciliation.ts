import apiService from './api';
import {
  ReconciliationLog,
  ReconciliationSummary,
  ApiResponse,
} from '@/types/api';

export class ReconciliationService {
  // Get reconciliation logs
  async getReconciliationLogs(params?: {
    limit?: number;
    offset?: number;
    risk_impact?: string;
    action?: string;
  }): Promise<ApiResponse<{ logs: ReconciliationLog[]; total: number }>> {
    return apiService.get<{ logs: ReconciliationLog[]; total: number }>('/reconciliation/logs', params);
  }

  // Get reconciliation summary
  async getReconciliationSummary(): Promise<ApiResponse<ReconciliationSummary>> {
    return apiService.get<ReconciliationSummary>('/reconciliation/summary');
  }

  // Trigger manual reconciliation
  async triggerReconciliation(params?: {
    user_id?: string;
    force?: boolean;
  }): Promise<ApiResponse<{ message: string; discrepancies_found: number }>> {
    return apiService.post<{ message: string; discrepancies_found: number }>('/reconciliation/trigger', params);
  }

  // Get unresolved discrepancies
  async getUnresolvedDiscrepancies(): Promise<ApiResponse<ReconciliationLog[]>> {
    return apiService.get<ReconciliationLog[]>('/reconciliation/unresolved');
  }

  // Resolve discrepancy manually
  async resolveDiscrepancy(
    logId: string,
    resolution: {
      action: 'AUTO_CORRECTED' | 'MANUAL_REVIEW' | 'BROKER_SYNCED';
      notes?: string;
    }
  ): Promise<ApiResponse<void>> {
    return apiService.put<void>(`/reconciliation/logs/${logId}/resolve`, resolution);
  }

  // Get reconciliation statistics
  async getReconciliationStats(params?: {
    period?: 'hour' | 'day' | 'week' | 'month';
  }): Promise<ApiResponse<{
    total_reconciliations: number;
    successful_reconciliations: number;
    failed_reconciliations: number;
    average_discrepancies: number;
    resolution_time_avg: number;
    trend: Array<{
      timestamp: string;
      discrepancies: number;
      resolutions: number;
    }>;
  }>> {
    return apiService.get('/reconciliation/stats', params);
  }

  // Export reconciliation report
  async exportReconciliationReport(params?: {
    start_date?: string;
    end_date?: string;
    format?: 'csv' | 'json' | 'pdf';
  }): Promise<Blob> {
    const response = await apiService.getClient().get('/reconciliation/export', {
      params,
      responseType: 'blob',
    });
    return response.data;
  }
}

export const reconciliationService = new ReconciliationService();

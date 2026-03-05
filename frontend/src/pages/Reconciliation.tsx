import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { reconciliationService } from '@/services/reconciliation';
import { formatCurrency, formatTimeAgo, formatTradeStatus } from '@/utils/formatters';
import { Button } from '@/components/ui/Button';
import { useUIStore } from '@/store/uiStore';
import { toast } from 'react-hot-toast';

interface DiscrepancyLog {
  id: string;
  trade_id: string;
  broker_trade_id?: string;
  discrepancy_type: 'MISSING_BROKER' | 'MISSING_LEDGER' | 'PRICE_DIFFERENCE' | 'QUANTITY_DIFFERENCE';
  description: string;
  broker_value?: number;
  ledger_value?: number;
  difference?: number;
  status: 'PENDING' | 'RESOLVED' | 'IGNORED';
  created_at: string;
  resolved_at?: string;
}

const Reconciliation: React.FC = () => {
  const { closeModal } = useUIStore();
  const [selectedFilter, setSelectedFilter] = useState<'all' | 'pending' | 'resolved'>('all');

  // Fetch reconciliation logs
  const { data: logs, isLoading: logsLoading, refetch } = useQuery({
    queryKey: ['reconciliation-logs', selectedFilter],
    queryFn: () => reconciliationService.getLogs({ status: selectedFilter === 'all' ? undefined : selectedFilter }),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch reconciliation summary
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['reconciliation-summary'],
    queryFn: () => reconciliationService.getSummary(),
    refetchInterval: 60000, // Refresh every minute
  });

  // Manual reconciliation trigger
  const triggerReconciliation = useMutation({
    mutationFn: () => reconciliationService.triggerManual(),
    onSuccess: (response) => {
      if (response.success) {
        toast.success('Reconciliation triggered successfully!');
        refetch();
      } else {
        toast.error(response.error || 'Failed to trigger reconciliation');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to trigger reconciliation');
    },
  });

  // Resolve discrepancy
  const resolveDiscrepancy = useMutation({
    mutationFn: (discrepancyId: string) => reconciliationService.resolveDiscrepancy(discrepancyId),
    onSuccess: (response) => {
      if (response.success) {
        toast.success('Discrepancy resolved successfully!');
        refetch();
      } else {
        toast.error(response.error || 'Failed to resolve discrepancy');
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to resolve discrepancy');
    },
  });

  const filteredLogs = logs?.data?.filter(log => 
    selectedFilter === 'all' || log.status === selectedFilter
  ) || [];

  const getDiscrepancyColor = (type: string) => {
    switch (type) {
      case 'MISSING_BROKER':
        return 'text-red-600 bg-red-50';
      case 'MISSING_LEDGER':
        return 'text-orange-600 bg-orange-50';
      case 'PRICE_DIFFERENCE':
        return 'text-yellow-600 bg-yellow-50';
      case 'QUANTITY_DIFFERENCE':
        return 'text-blue-600 bg-blue-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'PENDING':
        return 'text-yellow-600 bg-yellow-50';
      case 'RESOLVED':
        return 'text-green-600 bg-green-50';
      case 'IGNORED':
        return 'text-gray-600 bg-gray-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Reconciliation Dashboard</h2>
          <Button
            variant="ghost"
            onClick={() => closeModal('reconcileModal')}
            className="p-2"
          >
            ×
          </Button>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white p-4 border rounded-lg">
            <div className="text-sm font-medium text-gray-500">Total Discrepancies</div>
            <div className="mt-1 text-2xl font-bold text-gray-900">
              {summaryLoading ? '...' : summary?.data?.total_discrepancies || 0}
            </div>
          </div>
          
          <div className="bg-white p-4 border rounded-lg">
            <div className="text-sm font-medium text-yellow-600">Pending</div>
            <div className="mt-1 text-2xl font-bold text-yellow-600">
              {summaryLoading ? '...' : summary?.data?.pending_discrepancies || 0}
            </div>
          </div>
          
          <div className="bg-white p-4 border rounded-lg">
            <div className="text-sm font-medium text-green-600">Resolved</div>
            <div className="mt-1 text-2xl font-bold text-green-600">
              {summaryLoading ? '...' : summary?.data?.resolved_discrepancies || 0}
            </div>
          </div>
          
          <div className="bg-white p-4 border rounded-lg">
            <div className="text-sm font-medium text-gray-500">Last Run</div>
            <div className="mt-1 text-sm font-bold text-gray-900">
              {summaryLoading ? '...' : formatTimeAgo(summary?.data?.last_run || '')}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-4 mb-6">
          <Button
            onClick={() => triggerReconciliation.mutate()}
            loading={triggerReconciliation.isPending}
            disabled={triggerReconciliation.isPending}
          >
            {triggerReconciliation.isPending ? 'Running...' : 'Run Reconciliation'}
          </Button>
          
          <div className="flex gap-2">
            <Button
              variant={selectedFilter === 'all' ? 'primary' : 'outline'}
              onClick={() => setSelectedFilter('all')}
              size="sm"
            >
              All ({logs?.data?.length || 0})
            </Button>
            <Button
              variant={selectedFilter === 'pending' ? 'primary' : 'outline'}
              onClick={() => setSelectedFilter('pending')}
              size="sm"
            >
              Pending ({logs?.data?.filter(l => l.status === 'PENDING').length || 0})
            </Button>
            <Button
              variant={selectedFilter === 'resolved' ? 'primary' : 'outline'}
              onClick={() => setSelectedFilter('resolved')}
              size="sm"
            >
              Resolved ({logs?.data?.filter(l => l.status === 'RESOLVED').length || 0})
            </Button>
          </div>
        </div>

        {/* Discrepancy Logs Table */}
        <div className="flex-1 overflow-auto">
          <div className="bg-white border rounded-lg">
            <div className="px-4 py-3 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">Discrepancy Logs</h3>
            </div>
            
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Trade ID
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Description
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Difference
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {logsLoading ? (
                    <tr>
                      <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                        Loading discrepancy logs...
                      </td>
                    </tr>
                  ) : filteredLogs.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                        No discrepancies found
                      </td>
                    </tr>
                  ) : (
                    filteredLogs.map((log: DiscrepancyLog) => (
                      <tr key={log.id} className="hover:bg-gray-50">
                        <td className="px-4 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {log.trade_id}
                          {log.broker_trade_id && (
                            <div className="text-xs text-gray-500">Broker: {log.broker_trade_id}</div>
                          )}
                        </td>
                        <td className="px-4 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${getDiscrepancyColor(log.discrepancy_type)}`}>
                            {formatTradeStatus(log.discrepancy_type)}
                          </span>
                        </td>
                        <td className="px-4 py-4 text-sm text-gray-500 max-w-xs truncate">
                          {log.description}
                        </td>
                        <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-500">
                          {log.difference !== undefined && (
                            <span className={log.difference > 0 ? 'text-red-600' : 'text-green-600'}>
                              {formatCurrency(Math.abs(log.difference))}
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(log.status)}`}>
                            {log.status}
                          </span>
                        </td>
                        <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatTimeAgo(log.created_at)}
                        </td>
                        <td className="px-4 py-4 whitespace-nowrap text-sm">
                          {log.status === 'PENDING' && (
                            <Button
                              size="sm"
                              onClick={() => resolveDiscrepancy.mutate(log.id)}
                              loading={resolveDiscrepancy.isPending}
                              disabled={resolveDiscrepancy.isPending}
                            >
                              Resolve
                            </Button>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Reconciliation;

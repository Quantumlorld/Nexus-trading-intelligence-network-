import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { tradingService } from '@/services/trading';
import { formatCurrency, formatPnL, formatTimeAgo } from '@/utils/formatters';
import { Button } from '@/components/ui/Button';
import { useUIStore } from '@/store/uiStore';
import SimpleSystemStatus from '@/components/SimpleSystemStatus';
import CandleChart from '@/components/CandleChart';

const Dashboard: React.FC = () => {
  const { openModal } = useUIStore();

  // Fetch portfolio data
  const { data: positions, isLoading: positionsLoading } = useQuery({
    queryKey: ['positions'],
    queryFn: () => tradingService.getPositions(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch balance data
  const { data: balance, isLoading: balanceLoading } = useQuery({
    queryKey: ['balance'],
    queryFn: () => tradingService.getBalance(),
    refetchInterval: 15000, // Refresh every 15 seconds
  });

  // Fetch risk stats
  const { data: riskStats, isLoading: riskLoading } = useQuery({
    queryKey: ['risk-stats'],
    queryFn: () => tradingService.getRiskStats(),
    refetchInterval: 60000, // Refresh every minute
  });

  const totalPnL = positions?.data?.reduce(
    (sum, position) => sum + position.unrealized_pnl,
    0
  ) || 0;

  const pnlColor = totalPnL >= 0 ? 'text-green-600' : 'text-red-600';

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Nexus Trading Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Real-time system monitoring and trading interface
          </p>
        </div>

        {/* System Status and Chart */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <SimpleSystemStatus />
          <CandleChart />
        </div>

        {/* Quick Info */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-900">Backend Status</h3>
            <p className="mt-2 text-sm text-green-600">✓ Connected</p>
            <p className="text-xs text-gray-500">Port: 8000</p>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-900">Frontend Status</h3>
            <p className="mt-2 text-sm text-green-600">✓ Running</p>
            <p className="text-xs text-gray-500">Port: 3000</p>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-900">System Health</h3>
            <p className="mt-2 text-sm text-green-600">✓ Operational</p>
            <p className="text-xs text-gray-500">All services online</p>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="text-sm font-medium text-gray-500">Account Balance</div>
            <div className="mt-2 text-3xl font-bold text-gray-900">
              {balanceLoading ? '...' : formatCurrency(balance?.data?.broker_balance || 0)}
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="text-sm font-medium text-gray-500">Available Margin</div>
            <div className="mt-2 text-3xl font-bold text-gray-900">
              {balanceLoading ? '...' : formatCurrency(balance?.data?.free_margin || 0)}
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="text-sm font-medium text-gray-500">Total P&L</div>
            <div className={`mt-2 text-3xl font-bold ${pnlColor}`}>
              {positionsLoading ? '...' : formatPnL(totalPnL).text}
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="text-sm font-medium text-gray-500">Daily Loss</div>
            <div className="mt-2 text-3xl font-bold text-gray-900">
              {riskLoading ? '...' : formatCurrency(riskStats?.data?.daily_loss || 0)}
            </div>
            <div className="mt-1 text-sm text-gray-500">
              Limit: {formatCurrency(riskStats?.data?.max_daily_loss || 0)}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 mb-8">
          <Button onClick={() => openModal('tradeForm')}>
            New Trade
          </Button>
          <Button variant="outline" onClick={() => openModal('reconcileModal')}>
            Reconcile
          </Button>
        </div>

        {/* Positions Table */}
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">Open Positions</h2>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quantity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Entry Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Current Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    P&L
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Opened
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {positionsLoading ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                      Loading positions...
                    </td>
                  </tr>
                ) : positions?.data?.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                      No open positions
                    </td>
                  </tr>
                ) : (
                  positions?.data?.map((position) => (
                    <tr key={position.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {position.symbol}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {position.quantity > 0 ? 'BUY' : 'SELL'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {Math.abs(position.quantity)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatCurrency(position.entry_price)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatCurrency(position.current_price)}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${formatPnL(position.unrealized_pnl).color}`}>
                        {formatPnL(position.unrealized_pnl).text}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatTimeAgo(position.created_at)}
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
  );
};

export default Dashboard;

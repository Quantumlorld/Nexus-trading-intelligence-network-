import React, { useState, useEffect } from 'react';
import { systemService } from '@/services/system';

interface SimpleSystemStatusProps {
  className?: string;
}

export const SimpleSystemStatus: React.FC<SimpleSystemStatusProps> = ({ className }) => {
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchStatus = async () => {
    try {
      setRefreshing(true);
      const healthResponse = await systemService.getHealth();

      if (healthResponse.success) {
        setHealth(healthResponse.data);
      }
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleTrading = async () => {
    try {
      if (health?.trading_enabled) {
        await systemService.disableTrading();
      } else {
        await systemService.enableTrading();
      }
      await fetchStatus();
    } catch (error) {
      console.error('Failed to toggle trading:', error);
    }
  };

  if (loading) {
    return (
      <div className={className}>
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-4">System Status</h3>
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        </div>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      case 'unhealthy':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className={className}>
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">System Status</h3>
          <button
            onClick={fetchStatus}
            disabled={refreshing}
            className="p-2 text-gray-500 hover:text-gray-700"
          >
            <svg className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>

        <div className="space-y-3">
          {/* Overall Status */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Overall Status</span>
            <span className={`px-2 py-1 rounded text-xs text-white ${getStatusColor(health?.status || 'unknown')}`}>
              {health?.status ? health.status.charAt(0).toUpperCase() + health.status.slice(1) : 'Unknown'}
            </span>
          </div>

          {/* Trading Status */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Trading</span>
            <div className="flex items-center space-x-2">
              <span className={`w-2 h-2 rounded-full ${health?.trading_enabled ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <span className="text-sm">{health?.trading_enabled ? 'Enabled' : 'Disabled'}</span>
              <button
                onClick={handleToggleTrading}
                className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Toggle
              </button>
            </div>
          </div>

          {/* Broker Status */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Broker</span>
            <div className="flex items-center space-x-2">
              <span className={`w-2 h-2 rounded-full ${health?.broker_status === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <span className="text-sm">{health?.broker_status ? health.broker_status.charAt(0).toUpperCase() + health.broker_status.slice(1) : 'Unknown'}</span>
            </div>
          </div>

          {/* Database Status */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Database</span>
            <div className="flex items-center space-x-2">
              <span className={`w-2 h-2 rounded-full ${health?.db_status === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <span className="text-sm">{health?.db_status ? health.db_status.charAt(0).toUpperCase() + health.db_status.slice(1) : 'Unknown'}</span>
            </div>
          </div>

          {/* Uptime */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Uptime</span>
            <span className="text-sm text-gray-500">{health?.uptime || 'Unknown'}</span>
          </div>
        </div>

        {/* Simulation Controls */}
        <div className="border-t pt-4 mt-4">
          <h4 className="text-sm font-medium mb-2">Simulation Controls</h4>
          <div className="flex space-x-2">
            <button
              onClick={async () => {
                await systemService.simulateBrokerFailure();
                await fetchStatus();
              }}
              disabled={!health?.broker_status || health?.broker_status === 'unhealthy'}
              className="px-2 py-1 text-xs bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-300"
            >
              Simulate Broker Failure
            </button>
            <button
              onClick={async () => {
                await systemService.simulateBrokerRecovery();
                await fetchStatus();
              }}
              disabled={health?.broker_status === 'healthy'}
              className="px-2 py-1 text-xs bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-300"
            >
              Simulate Broker Recovery
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimpleSystemStatus;

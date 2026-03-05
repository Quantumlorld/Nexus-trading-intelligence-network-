import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { systemService, SystemStatus, HealthStatus } from '@/services/system';
import { RefreshCw, Power, AlertTriangle, CheckCircle } from 'lucide-react';

interface SystemStatusProps {
  className?: string;
}

export const SystemStatus: React.FC<SystemStatusProps> = ({ className }) => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchStatus = async () => {
    try {
      setRefreshing(true);
      const [healthResponse, systemResponse] = await Promise.all([
        systemService.getHealth(),
        systemService.getSystemStatus()
      ]);

      if (healthResponse.success) {
        setHealth(healthResponse.data);
      }
      if (systemResponse.success) {
        setSystemStatus(systemResponse.data);
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
    const interval = setInterval(fetchStatus, 30000); // Refresh every 30 seconds
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

  const handleSimulateBrokerFailure = async () => {
    try {
      await systemService.simulateBrokerFailure();
      await fetchStatus();
    } catch (error) {
      console.error('Failed to simulate broker failure:', error);
    }
  };

  const handleSimulateBrokerRecovery = async () => {
    try {
      await systemService.simulateBrokerRecovery();
      await fetchStatus();
    } catch (error) {
      console.error('Failed to simulate broker recovery:', error);
    }
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32">
            <RefreshCw className="animate-spin" />
          </div>
        </CardContent>
      </Card>
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

  const getStatusIcon = (connected: boolean) => {
    return connected ? (
      <CheckCircle className="h-4 w-4 text-green-500" />
    ) : (
      <AlertTriangle className="h-4 w-4 text-red-500" />
    );
  };

  return (
    <Card className={className}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle>System Status</CardTitle>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchStatus}
          disabled={refreshing}
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Overall Status */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Overall Status</span>
          <Badge className={getStatusColor(health?.status || 'unknown')}>
            {health?.status || 'Unknown'}
          </Badge>
        </div>

        {/* Trading Status */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Trading</span>
          <div className="flex items-center space-x-2">
            {getStatusIcon(health?.trading_enabled || false)}
            <Badge variant={health?.trading_enabled ? 'default' : 'secondary'}>
              {health?.trading_enabled ? 'Enabled' : 'Disabled'}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={handleToggleTrading}
            >
              <Power className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Broker Status */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Broker</span>
          <div className="flex items-center space-x-2">
            {getStatusIcon(health?.broker_status === 'healthy')}
            <Badge variant={health?.broker_status === 'healthy' ? 'default' : 'destructive'}>
              {health?.broker_status || 'Unknown'}
            </Badge>
          </div>
        </div>

        {/* Database Status */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Database</span>
          <div className="flex items-center space-x-2">
            {getStatusIcon(health?.db_status === 'healthy')}
            <Badge variant={health?.db_status === 'healthy' ? 'default' : 'destructive'}>
              {health?.db_status || 'Unknown'}
            </Badge>
          </div>
        </div>

        {/* Uptime */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Uptime</span>
          <span className="text-sm text-gray-500">{health?.uptime || 'Unknown'}</span>
        </div>

        {/* Simulation Controls */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium mb-2">Simulation Controls</h4>
          <div className="flex space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleSimulateBrokerFailure}
              disabled={!health?.broker_status || health?.broker_status === 'unhealthy'}
            >
              Simulate Broker Failure
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleSimulateBrokerRecovery}
              disabled={health?.broker_status === 'healthy'}
            >
              Simulate Broker Recovery
            </Button>
          </div>
        </div>

        {/* System Details */}
        {systemStatus && (
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium mb-2">System Details</h4>
            <div className="space-y-1 text-sm text-gray-500">
              <div>Supported Symbols: {systemStatus.supported_symbols.join(', ')}</div>
              <div>Supported Timeframes: {systemStatus.supported_timeframes.join(', ')}</div>
              <div>9H Candles Available:</div>
              {Object.entries(systemStatus['9h_candles_available']).map(([symbol, count]) => (
                <div key={symbol} className="ml-4">
                  {symbol}: {count} candles
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SystemStatus;

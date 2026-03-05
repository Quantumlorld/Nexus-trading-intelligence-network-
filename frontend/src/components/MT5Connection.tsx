import React, { useState } from 'react';
import { systemService } from '@/services/system';

interface MT5ConnectionProps {
  className?: string;
}

export const MT5Connection: React.FC<MT5ConnectionProps> = ({ className }) => {
  const [account, setAccount] = useState('');
  const [password, setPassword] = useState('');
  const [server, setServer] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<any>(null);

  const handleConnect = async () => {
    if (!account || !password || !server) {
      alert('Please fill in all fields');
      return;
    }

    setLoading(true);
    try {
      const response = await systemService.connectMT5(parseInt(account), password, server);
      if (response.success) {
        setStatus({ success: true, message: response.message });
        setTimeout(() => window.location.reload(), 2000);
      } else {
        setStatus({ success: false, message: response.message });
      }
    } catch (error) {
      setStatus({ success: false, message: 'Connection failed' });
    } finally {
      setLoading(false);
    }
  };

  const handleCheckStatus = async () => {
    setLoading(true);
    try {
      const mt5Status = await systemService.getMT5Status();
      setStatus(mt5Status);
    } catch (error) {
      setStatus({ success: false, message: 'Failed to get status' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={className}>
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold mb-4">MetaTrader 5 Connection</h3>
        
        {status && (
          <div className={`mb-4 p-3 rounded text-sm ${
            status.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
          }`}>
            {status.message}
          </div>
        )}

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Account Number
            </label>
            <input
              type="number"
              value={account}
              onChange={(e) => setAccount(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Your MT5 account number"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Your MT5 password"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Server
            </label>
            <input
              type="text"
              value={server}
              onChange={(e) => setServer(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Your broker server (e.g., MetaQuotes-Demo)"
            />
          </div>

          <div className="flex space-x-2">
            <button
              onClick={handleConnect}
              disabled={loading}
              className="flex-1 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300"
            >
              {loading ? 'Connecting...' : 'Connect MT5'}
            </button>
            
            <button
              onClick={handleCheckStatus}
              disabled={loading}
              className="flex-1 bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600 disabled:bg-gray-300"
            >
              {loading ? 'Checking...' : 'Check Status'}
            </button>
          </div>
        </div>

        <div className="mt-4 p-3 bg-yellow-50 rounded-md">
          <h4 className="text-sm font-medium text-yellow-800 mb-2">⚠️ Important Notes:</h4>
          <ul className="text-xs text-yellow-700 space-y-1">
            <li>• Use a DEMO account first for testing</li>
            <li>• Ensure MT5 terminal is installed</li>
            <li>• Enable "Allow DLL imports" in MT5</li>
            <li>• Never share your real account credentials</li>
            <li>• Start with small amounts for real trading</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

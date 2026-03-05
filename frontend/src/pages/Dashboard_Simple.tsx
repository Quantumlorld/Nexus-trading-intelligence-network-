import React from 'react';
import { SimpleSystemStatus } from '../components/SimpleSystemStatus';
import { CandleChart } from '../components/CandleChart';
import { MT5Connection } from '../components/MT5Connection';
import { ProfessionalTradingPanel } from '../components/ProfessionalTradingPanel';
import { DemoTradingPanel } from '../components/DemoTradingPanel';
import { UniversalMT5Connection } from '../components/UniversalMT5Connection';

const Dashboard: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute top-0 right-0 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse animation-delay-2000"></div>
        <div className="absolute bottom-0 left-1/2 w-96 h-96 bg-indigo-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse animation-delay-4000"></div>
      </div>

      {/* Professional Header */}
      <div className="relative bg-black/30 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-1 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                Nexus Trading System
              </h1>
              <p className="text-purple-200">Professional Trading Intelligence Platform</p>
            </div>
            <div className="flex items-center space-x-4">
              <span className="px-3 py-1 bg-gradient-to-r from-green-400 to-emerald-400 text-white text-xs rounded-full font-bold animate-pulse shadow-lg shadow-green-400/50">
                SYSTEM LIVE
              </span>
              <span className="text-purple-300 text-sm bg-white/10 px-3 py-1 rounded-full backdrop-blur-sm">
                v2.0 Professional
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* System Status and Chart */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <SimpleSystemStatus />
            </div>
            
            <div className="lg:col-span-1 space-y-6">
              <MT5Connection />
              <UniversalMT5Connection />
            </div>
          </div>
          <CandleChart />
        </div>

        {/* Professional Trading Panel */}
        <div className="mb-8">
          <ProfessionalTradingPanel />
        </div>

        {/* Demo Trading Panel */}
        <div className="mb-8">
          <DemoTradingPanel />
        </div>

        {/* Professional Footer */}
        <div className="mt-12 bg-gradient-to-r from-black/40 to-black/20 backdrop-blur-xl rounded-2xl p-6 text-white border border-white/10">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 backdrop-blur-sm rounded-xl p-4 border border-green-400/30">
              <h4 className="font-bold mb-2 flex items-center">
                <span className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></span>
                System Status
              </h4>
              <p className="text-sm text-green-200">All systems operational</p>
              <p className="text-xs text-green-300 mt-1">Uptime: 99.9%</p>
            </div>
            <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 backdrop-blur-sm rounded-xl p-4 border border-blue-400/30">
              <h4 className="font-bold mb-2 flex items-center">
                <span className="w-2 h-2 bg-blue-400 rounded-full mr-2 animate-pulse"></span>
                Trading Engine
              </h4>
              <p className="text-sm text-blue-200">Ready for execution</p>
              <p className="text-xs text-blue-300 mt-1">Latency: 12ms</p>
            </div>
            <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 backdrop-blur-sm rounded-xl p-4 border border-purple-400/30">
              <h4 className="font-bold mb-2 flex items-center">
                <span className="w-2 h-2 bg-purple-400 rounded-full mr-2 animate-pulse"></span>
                Market Data
              </h4>
              <p className="text-sm text-purple-200">Real-time feeds active</p>
              <p className="text-xs text-purple-300 mt-1">Update: 2ms</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

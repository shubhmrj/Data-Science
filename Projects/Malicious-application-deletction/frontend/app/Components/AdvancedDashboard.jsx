"use client"
import React, { useState, useEffect } from 'react';
import { 
  Brain, Activity, Globe, Target, Shield, Zap, 
  TrendingUp, AlertTriangle, Eye, Radar, Settings,
  BarChart3, PieChart, Clock, Database, Cpu
} from 'lucide-react';
import ThreatVisualization from './ThreatVisualization';
import AIThreatHunting from './AIThreatHunting';
import LiveStatsPanel from './LiveStatsPanel';
import AdvancedThreatIntelligence from './AdvancedThreatIntelligence';

const AdvancedDashboard = () => {
  const [activeView, setActiveView] = useState('overview');
  const [systemStatus, setSystemStatus] = useState({
    aiHunting: 'ACTIVE',
    threatVisualization: 'ACTIVE',
    realTimeMonitoring: 'ACTIVE',
    mlModels: 'OPERATIONAL'
  });

  const dashboardViews = [
    {
      id: 'overview',
      name: 'System Overview',
      icon: Activity,
      description: 'Complete system status and metrics'
    },
    {
      id: 'threat-viz',
      name: 'Threat Visualization',
      icon: Globe,
      description: 'Real-time global threat mapping'
    },
    {
      id: 'ai-hunting',
      name: 'AI Threat Hunting',
      icon: Brain,
      description: 'Automated threat detection and prediction'
    },
    {
      id: 'threat-intel',
      name: 'Threat Intelligence',
      icon: Target,
      description: 'APK source tracking and link analysis'
    },
    {
      id: 'analytics',
      name: 'Advanced Analytics',
      icon: BarChart3,
      description: 'Deep insights and trend analysis'
    }
  ];

  const systemMetrics = [
    {
      title: 'AI Models Active',
      value: '4',
      change: '+1',
      color: 'text-purple-400',
      bgColor: 'bg-purple-900/20',
      borderColor: 'border-purple-500/30',
      icon: Brain
    },
    {
      title: 'Threats Detected',
      value: '1,247',
      change: '+23',
      color: 'text-red-400',
      bgColor: 'bg-red-900/20',
      borderColor: 'border-red-500/30',
      icon: AlertTriangle
    },
    {
      title: 'Predictions Made',
      value: '89',
      change: '+7',
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-900/20',
      borderColor: 'border-cyan-500/30',
      icon: Target
    },
    {
      title: 'System Uptime',
      value: '99.9%',
      change: '0%',
      color: 'text-green-400',
      bgColor: 'bg-green-900/20',
      borderColor: 'border-green-500/30',
      icon: Shield
    }
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'ACTIVE':
      case 'OPERATIONAL':
        return 'text-green-400 bg-green-900/20 border-green-500/30';
      case 'WARNING':
        return 'text-yellow-400 bg-yellow-900/20 border-yellow-500/30';
      case 'ERROR':
        return 'text-red-400 bg-red-900/20 border-red-500/30';
      default:
        return 'text-gray-400 bg-gray-900/20 border-gray-500/30';
    }
  };

  const renderOverview = () => (
    <div className="space-y-8">
      {/* System Status Header */}
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Activity className="w-8 h-8 text-cyan-400 animate-pulse" />
              <div className="absolute inset-0 bg-cyan-400/20 rounded-full animate-ping"></div>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Advanced APK Security Platform</h2>
              <p className="text-gray-400">Real-time threat detection with AI-powered analysis</p>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-sm text-gray-400">Last Updated</div>
            <div className="text-white font-mono">{new Date().toLocaleTimeString()}</div>
          </div>
        </div>

        {/* System Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {systemMetrics.map((metric, index) => (
            <div
              key={index}
              className={`p-6 rounded-xl border transition-all duration-300 hover:scale-105 ${metric.bgColor} ${metric.borderColor}`}
            >
              <div className="flex items-center justify-between mb-4">
                <metric.icon className={`w-8 h-8 ${metric.color}`} />
                <div className={`text-xs font-bold px-2 py-1 rounded-full ${metric.bgColor} ${metric.borderColor} border`}>
                  {metric.change}
                </div>
              </div>
              <div className={`text-3xl font-black ${metric.color} mb-2`}>
                {metric.value}
              </div>
              <div className="text-sm text-gray-400">{metric.title}</div>
            </div>
          ))}
        </div>
      </div>

      {/* System Status Components */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* System Components Status */}
        <div className="glass rounded-2xl p-6 border border-gray-600">
          <div className="flex items-center space-x-3 mb-6">
            <Settings className="w-6 h-6 text-gray-400" />
            <h3 className="text-xl font-bold text-white">System Components</h3>
          </div>
          
          <div className="space-y-4">
            {Object.entries(systemStatus).map(([component, status]) => (
              <div key={component} className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    status === 'ACTIVE' || status === 'OPERATIONAL' ? 'bg-green-400 animate-pulse' : 'bg-red-400'
                  }`}></div>
                  <span className="text-white font-medium">
                    {component.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                  </span>
                </div>
                <div className={`px-3 py-1 rounded-full text-xs font-bold border ${getStatusColor(status)}`}>
                  {status}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="glass rounded-2xl p-6 border border-gray-600">
          <div className="flex items-center space-x-3 mb-6">
            <Zap className="w-6 h-6 text-yellow-400" />
            <h3 className="text-xl font-bold text-white">Quick Actions</h3>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <button
              onClick={() => setActiveView('threat-viz')}
              className="p-4 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-lg hover:from-blue-700 hover:to-cyan-700 transition-all duration-300 group"
            >
              <Globe className="w-6 h-6 text-white mb-2 group-hover:scale-110 transition-transform" />
              <div className="text-white font-bold text-sm">Threat Map</div>
            </button>
            
            <button
              onClick={() => setActiveView('ai-hunting')}
              className="p-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-300 group"
            >
              <Brain className="w-6 h-6 text-white mb-2 group-hover:scale-110 transition-transform" />
              <div className="text-white font-bold text-sm">AI Hunting</div>
            </button>
            
            <button
              onClick={() => setActiveView('analytics')}
              className="p-4 bg-gradient-to-r from-green-600 to-emerald-600 rounded-lg hover:from-green-700 hover:to-emerald-700 transition-all duration-300 group"
            >
              <BarChart3 className="w-6 h-6 text-white mb-2 group-hover:scale-110 transition-transform" />
              <div className="text-white font-bold text-sm">Analytics</div>
            </button>
            
            <button className="p-4 bg-gradient-to-r from-orange-600 to-red-600 rounded-lg hover:from-orange-700 hover:to-red-700 transition-all duration-300 group">
              <AlertTriangle className="w-6 h-6 text-white mb-2 group-hover:scale-110 transition-transform" />
              <div className="text-white font-bold text-sm">Alerts</div>
            </button>
          </div>
        </div>
      </div>

      {/* Live Statistics */}
      <LiveStatsPanel />
    </div>
  );

  const renderAnalytics = () => (
    <div className="space-y-8">
      {/* Analytics Header */}
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center space-x-3 mb-4">
          <BarChart3 className="w-8 h-8 text-green-400" />
          <div>
            <h2 className="text-2xl font-bold text-white">Advanced Analytics</h2>
            <p className="text-gray-400">Deep insights and predictive intelligence</p>
          </div>
        </div>
      </div>

      {/* Analytics Placeholder - Can be expanded with real charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="glass rounded-2xl p-6 border border-gray-600">
          <h3 className="text-xl font-bold text-white mb-4">Threat Trends</h3>
          <div className="h-64 bg-gray-800/50 rounded-lg border border-gray-700 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <TrendingUp className="w-12 h-12 mx-auto mb-4" />
              <p>Advanced analytics charts will be displayed here</p>
              <p className="text-sm">Integration with Chart.js/D3.js coming soon</p>
            </div>
          </div>
        </div>
        
        <div className="glass rounded-2xl p-6 border border-gray-600">
          <h3 className="text-xl font-bold text-white mb-4">ML Model Performance</h3>
          <div className="h-64 bg-gray-800/50 rounded-lg border border-gray-700 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <PieChart className="w-12 h-12 mx-auto mb-4" />
              <p>Model accuracy and performance metrics</p>
              <p className="text-sm">Real-time model monitoring dashboard</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
      {/* Navigation Header */}
      <div className="sticky top-0 z-40 bg-black/20 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-lg flex items-center justify-center">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold">APK<span className="text-cyan-400">Guard</span> Advanced</h1>
                <p className="text-xs text-gray-400">AI Security Platform</p>
              </div>
            </div>
            
            {/* View Selector */}
            <div className="flex items-center space-x-2">
              {dashboardViews.map((view) => (
                <button
                  key={view.id}
                  onClick={() => setActiveView(view.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
                    activeView === view.id
                      ? 'bg-gradient-to-r from-purple-500 to-cyan-500 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-white/10'
                  }`}
                  title={view.description}
                >
                  <view.icon className="w-4 h-4" />
                  <span className="hidden md:inline">{view.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {activeView === 'overview' && renderOverview()}
        {activeView === 'threat-viz' && <ThreatVisualization />}
        {activeView === 'ai-hunting' && <AIThreatHunting />}
        {activeView === 'threat-intel' && <AdvancedThreatIntelligence />}
        {activeView === 'analytics' && renderAnalytics()}
      </div>
    </div>
  );
};

export default AdvancedDashboard;

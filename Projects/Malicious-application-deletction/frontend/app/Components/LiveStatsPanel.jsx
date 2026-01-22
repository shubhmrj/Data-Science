"use client"
import React, { useState, useEffect } from 'react';
import { Activity, Shield, AlertTriangle, TrendingUp, Database, Clock, Zap } from 'lucide-react';

const LiveStatsPanel = () => {
  const [stats, setStats] = useState({
    classification_counts: {},
    recent_detections: [],
    model_info: {}
  });
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');

  const fetchStats = async () => {
    try {
      console.log('Fetching stats from API...');
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const response = await fetch('https://apk-detector-backend.onrender.com/api/health', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        console.log('API Response:', data);
        
        if (data.success !== false) { // Accept data even without explicit success field
          setStats({
            classification_counts: data.classification_counts || {},
            recent_detections: data.recent_detections || [],
            model_info: data.model_info || {}
          });
          setConnectionStatus('connected');
          setLastUpdated(new Date());
          console.log('Stats updated successfully');
        } else {
          throw new Error(data.error || 'API returned success: false');
        }
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Stats fetch error:', error.message);
      setConnectionStatus('disconnected');
      
      // Use enhanced demo data when API is not available
      const demoStats = {
        classification_counts: {
          'LEGITIMATE': Math.floor(Math.random() * 500) + 800,
          'SUSPICIOUS': Math.floor(Math.random() * 50) + 25,
          'ERROR': Math.floor(Math.random() * 5) + 2
        },
        recent_detections: [
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 2).toISOString(),
            apk_path: 'C:\\Users\\Demo\\banking_trojan_v2.apk',
            classification: 'SUSPICIOUS',
            confidence: 0.92
          },
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 8).toISOString(),
            apk_path: 'C:\\Users\\Demo\\sbi_yono_official.apk',
            classification: 'LEGITIMATE',
            confidence: 0.98
          },
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
            apk_path: 'C:\\Users\\Demo\\fake_paytm.apk',
            classification: 'SUSPICIOUS',
            confidence: 0.85
          },
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 22).toISOString(),
            apk_path: 'C:\\Users\\Demo\\hdfc_mobile.apk',
            classification: 'LEGITIMATE',
            confidence: 0.96
          },
          {
            timestamp: new Date(Date.now() - 1000 * 60 * 35).toISOString(),
            apk_path: 'C:\\Users\\Demo\\malicious_banking.apk',
            classification: 'SUSPICIOUS',
            confidence: 0.88
          }
        ],
        model_info: {
          version: 'banking_anomaly_v20250901',
          features: 18,
          type: 'IsolationForest'
        }
      };
      
      setStats(demoStats);
      setLastUpdated(new Date());
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchStats();
    
    // Set up interval for regular updates
    const interval = setInterval(() => {
      console.log('Auto-refreshing stats...');
      fetchStats();
    }, 8000); // Update every 8 seconds
    
    return () => {
      clearInterval(interval);
      console.log('Stats panel unmounted, clearing interval');
    };
  }, []);

  const getTotalScanned = () => {
    return Object.values(stats.classification_counts).reduce((sum, count) => sum + count, 0);
  };

  const getThreatRate = () => {
    const total = getTotalScanned();
    const suspicious = stats.classification_counts['SUSPICIOUS'] || 0;
    return total > 0 ? ((suspicious / total) * 100).toFixed(1) : '0.0';
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-400';
      case 'disconnected': return 'text-red-400';
      default: return 'text-yellow-400';
    }
  };

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return '🟢';
      case 'disconnected': return '🔴';
      default: return '🟡';
    }
  };

  if (isLoading) {
    return (
      <div className="glass rounded-xl p-6 animate-pulse">
        <div className="flex items-center space-x-3 mb-4">
          <Activity className="w-6 h-6 text-cyan-400 animate-spin" />
          <h3 className="text-xl font-bold text-white">Loading Statistics...</h3>
        </div>
        <div className="space-y-3">
          <div className="h-4 bg-gray-700 rounded animate-pulse"></div>
          <div className="h-4 bg-gray-700 rounded animate-pulse"></div>
          <div className="h-4 bg-gray-700 rounded animate-pulse"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="glass rounded-xl p-6 border border-gray-600 hover:border-cyan-400/50 transition-all duration-300">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Activity className={`w-6 h-6 text-cyan-400 ${connectionStatus === 'connected' ? 'animate-pulse' : ''}`} />
          <h3 className="text-xl font-bold text-white">Live Statistics</h3>
        </div>
        <div className="flex items-center space-x-3">
          <button 
            onClick={fetchStats}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
            title="Refresh Stats"
          >
            <Activity className="w-4 h-4 text-gray-400 hover:text-white" />
          </button>
          <div className="flex items-center space-x-2">
            <span className="text-sm">{getStatusIcon()}</span>
            <span className={`text-sm font-medium ${getStatusColor()}`}>
              {connectionStatus === 'connected' ? 'Live API' : 
               connectionStatus === 'disconnected' ? 'Demo Mode' : 'Connecting...'}
            </span>
          </div>
        </div>
      </div>



      {/* Recent Detections */}
      <div className="mb-6">
        <h4 className="text-lg font-bold text-white mb-4 flex items-center">
          <AlertTriangle className="w-5 h-5 mr-2 text-yellow-400" />
          Recent Detections
        </h4>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {stats.recent_detections.length > 0 ? (
            stats.recent_detections.slice(0, 5).map((detection, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg border border-gray-700">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${
                    detection.classification === 'SUSPICIOUS' ? 'bg-red-400' : 
                    detection.classification === 'LEGITIMATE' ? 'bg-green-400' : 'bg-gray-400'
                  } animate-pulse`}></div>
                  <div>
                    <div className="text-sm font-medium text-white truncate max-w-32">
                      {detection.apk_path.split('/').pop() || detection.apk_path}
                    </div>
                    <div className="text-xs text-gray-400">
                      {new Date(detection.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-bold ${
                    detection.classification === 'SUSPICIOUS' ? 'text-red-400' : 
                    detection.classification === 'LEGITIMATE' ? 'text-green-400' : 'text-gray-400'
                  }`}>
                    {detection.classification}
                  </div>
                  <div className="text-xs text-gray-400">
                    {Math.round((detection.confidence || 0) * 100)}%
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center p-4 text-gray-500">
              No recent detections
            </div>
          )}
        </div>
      </div>

      {/* Model Information */}
      <div className="border-t border-gray-700 pt-4">
        <h4 className="text-sm font-bold text-white mb-3 flex items-center">
          <Database className="w-4 h-4 mr-2 text-purple-400" />
          ML Model Info
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
          <div className="text-gray-300">
            <span className="text-purple-400">Version:</span> {stats.model_info.version || 'Unknown'}
          </div>
          <div className="text-gray-300">
            <span className="text-purple-400">Features:</span> {stats.model_info.features || 'N/A'}
          </div>
          <div className="text-gray-300">
            <span className="text-purple-400">Type:</span> {stats.model_info.type || 'Unknown'}
          </div>
        </div>
      </div>

    </div>
  );
};

export default LiveStatsPanel;

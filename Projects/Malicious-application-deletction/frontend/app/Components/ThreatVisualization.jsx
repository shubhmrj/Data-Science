"use client"
import React, { useState, useEffect, useRef } from 'react';
import { 
  Globe, Activity, AlertTriangle, Shield, Zap, Target, 
  TrendingUp, Eye, Brain, Radar, MapPin, Clock
} from 'lucide-react';

const ThreatVisualization = () => {
  const [threats, setThreats] = useState([]);
  const [activeThreats, setActiveThreats] = useState(0);
  const [threatLevel, setThreatLevel] = useState('MEDIUM');
  const [isScanning, setIsScanning] = useState(true);
  const mapRef = useRef(null);
  const [selectedThreat, setSelectedThreat] = useState(null);
  const [threatStats, setThreatStats] = useState({
    total: 0,
    blocked: 0,
    active: 0,
    severity: { high: 0, medium: 0, low: 0 }
  });

  // Simulated global threat locations
  const threatLocations = [
    { country: 'India', city: 'Mumbai', lat: 19.0760, lng: 72.8777, threats: [] },
    { country: 'USA', city: 'New York', lat: 40.7128, lng: -74.0060, threats: [] },
    { country: 'China', city: 'Beijing', lat: 39.9042, lng: 116.4074, threats: [] },
    { country: 'Russia', city: 'Moscow', lat: 55.7558, lng: 37.6176, threats: [] },
    { country: 'Brazil', city: 'São Paulo', lat: -23.5505, lng: -46.6333, threats: [] },
    { country: 'UK', city: 'London', lat: 51.5074, lng: -0.1278, threats: [] },
    { country: 'Germany', city: 'Berlin', lat: 52.5200, lng: 13.4050, threats: [] },
    { country: 'Japan', city: 'Tokyo', lat: 35.6762, lng: 139.6503, threats: [] },
  ];

  const threatTypes = [
    { type: 'Banking Trojan', severity: 'HIGH', color: 'bg-red-500', icon: '🏦' },
    { type: 'SMS Interceptor', severity: 'MEDIUM', color: 'bg-yellow-500', icon: '📱' },
    { type: 'Fake Certificate', severity: 'HIGH', color: 'bg-red-500', icon: '🔒' },
    { type: 'Data Stealer', severity: 'HIGH', color: 'bg-red-500', icon: '💳' },
    { type: 'Adware', severity: 'LOW', color: 'bg-green-500', icon: '📢' },
    { type: 'Spyware', severity: 'MEDIUM', color: 'bg-yellow-500', icon: '👁️' },
    { type: 'Ransomware', severity: 'HIGH', color: 'bg-red-500', icon: '🔐' },
    { type: 'Keylogger', severity: 'MEDIUM', color: 'bg-yellow-500', icon: '⌨️' }
  ];

  // Generate realistic threat data
  const generateThreat = () => {
    const location = threatLocations[Math.floor(Math.random() * threatLocations.length)];
    const threatType = threatTypes[Math.floor(Math.random() * threatTypes.length)];
    
    return {
      id: Date.now() + Math.random(),
      ...threatType,
      location: location,
      timestamp: new Date(),
      status: Math.random() > 0.3 ? 'BLOCKED' : 'ACTIVE',
      confidence: Math.random() * 0.4 + 0.6, // 60-100%
      sourceIP: `${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
      targetAPK: `${threatType.type.toLowerCase().replace(' ', '_')}_${Math.floor(Math.random() * 1000)}.apk`
    };
  };

  // Fetch threat data from backend API
  const fetchThreatData = async () => {
    try {
      const response = await fetch('https://apk-detector-backend.onrender.com/api/ai-hunting/threat-feed');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.success) {
        setThreats(data.threats);
        
        // Update stats based on real data
        const stats = {
          total: data.threats.length,
          blocked: data.threats.filter(t => t.status === 'BLOCKED').length,
          active: data.threats.filter(t => t.status === 'ACTIVE').length,
          severity: {
            high: data.threats.filter(t => t.severity === 'HIGH').length,
            medium: data.threats.filter(t => t.severity === 'MEDIUM').length,
            low: data.threats.filter(t => t.severity === 'LOW').length
          }
        };
        setThreatStats(stats);
        setActiveThreats(stats.active);
        
        // Determine overall threat level
        if (stats.severity.high > 5) setThreatLevel('CRITICAL');
        else if (stats.severity.high > 2) setThreatLevel('HIGH');
        else if (stats.severity.medium > 3) setThreatLevel('MEDIUM');
        else setThreatLevel('LOW');
      } else {
        throw new Error(data.error || 'Failed to fetch threat data');
      }
    } catch (err) {
      console.error('Error fetching threat data:', err);
      
      // Fallback to simulated data on error
      if (Math.random() > 0.4) { // 60% chance of new threat
        const newThreat = generateThreat();
        setThreats(prev => {
          const updated = [newThreat, ...prev].slice(0, 50); // Keep last 50 threats
          
          // Update stats
          const stats = {
            total: updated.length,
            blocked: updated.filter(t => t.status === 'BLOCKED').length,
            active: updated.filter(t => t.status === 'ACTIVE').length,
            severity: {
              high: updated.filter(t => t.severity === 'HIGH').length,
              medium: updated.filter(t => t.severity === 'MEDIUM').length,
              low: updated.filter(t => t.severity === 'LOW').length
            }
          };
          setThreatStats(stats);
          setActiveThreats(stats.active);
          
          // Determine overall threat level
          if (stats.severity.high > 5) setThreatLevel('CRITICAL');
          else if (stats.severity.high > 2) setThreatLevel('HIGH');
          else if (stats.severity.medium > 3) setThreatLevel('MEDIUM');
          else setThreatLevel('LOW');
          
          return updated;
        });
      }
    }
  };

  // Real-time threat detection with API integration
  useEffect(() => {
    fetchThreatData(); // Initial fetch
    const interval = setInterval(fetchThreatData, 8000); // Fetch every 8 seconds
    return () => clearInterval(interval);
  }, []);

  const getThreatLevelColor = () => {
    switch (threatLevel) {
      case 'CRITICAL': return 'text-red-400 animate-pulse';
      case 'HIGH': return 'text-red-400';
      case 'MEDIUM': return 'text-yellow-400';
      case 'LOW': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  const getThreatLevelBg = () => {
    switch (threatLevel) {
      case 'CRITICAL': return 'bg-red-900/30 border-red-500/50';
      case 'HIGH': return 'bg-red-900/20 border-red-500/30';
      case 'MEDIUM': return 'bg-yellow-900/20 border-yellow-500/30';
      case 'LOW': return 'bg-green-900/20 border-green-500/30';
      default: return 'bg-gray-900/20 border-gray-500/30';
    }
  };

  return (
    <div className="space-y-8">
      {/* Threat Overview Header */}
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Radar className="w-8 h-8 text-cyan-400 animate-spin" style={{ animationDuration: '3s' }} />
              <div className="absolute inset-0 bg-cyan-400/20 rounded-full animate-ping"></div>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Global Threat Intelligence</h2>
              <p className="text-gray-400">Real-time APK threat monitoring worldwide</p>
            </div>
          </div>
          
          <div className={`px-6 py-3 rounded-xl border-2 ${getThreatLevelBg()}`}>
            <div className="text-center">
              <div className={`text-2xl font-black ${getThreatLevelColor()}`}>
                {threatLevel}
              </div>
              <div className="text-sm text-gray-300">Threat Level</div>
            </div>
          </div>
        </div>

        {/* Real-time Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3">
              <Activity className="w-6 h-6 text-red-400 animate-pulse" />
              <div>
                <div className="text-2xl font-bold text-red-400">{activeThreats}</div>
                <div className="text-sm text-gray-400">Active Threats</div>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3">
              <Shield className="w-6 h-6 text-green-400" />
              <div>
                <div className="text-2xl font-bold text-green-400">{threatStats.blocked}</div>
                <div className="text-sm text-gray-400">Blocked</div>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3">
              <Target className="w-6 h-6 text-yellow-400" />
              <div>
                <div className="text-2xl font-bold text-yellow-400">{threatStats.severity.high}</div>
                <div className="text-sm text-gray-400">Critical</div>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3">
              <TrendingUp className="w-6 h-6 text-cyan-400" />
              <div>
                <div className="text-2xl font-bold text-cyan-400">{threatStats.total}</div>
                <div className="text-sm text-gray-400">Total Detected</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Interactive World Map */}
        <div className="glass rounded-2xl p-6 border border-gray-600">
          <div className="flex items-center space-x-3 mb-6">
            <Globe className="w-6 h-6 text-cyan-400" />
            <h3 className="text-xl font-bold text-white">Global Threat Map</h3>
            <div className="flex items-center space-x-2 ml-auto">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-green-400">Live</span>
            </div>
          </div>
          
          {/* Simplified World Map Visualization */}
          <div className="relative h-80 bg-gray-900/50 rounded-xl border border-gray-700 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 to-purple-900/20"></div>
            
            {/* Animated threat pulses */}
            {threatLocations.map((location, index) => {
              const locationThreats = threats.filter(t => t.location.city === location.city);
              const hasActiveThreats = locationThreats.some(t => t.status === 'ACTIVE');
              
              return (
                <div
                  key={location.city}
                  className={`absolute w-4 h-4 rounded-full cursor-pointer transition-all duration-300 hover:scale-150 ${
                    hasActiveThreats ? 'bg-red-500 animate-pulse' : 'bg-green-500'
                  }`}
                  style={{
                    left: `${((location.lng + 180) / 360) * 100}%`,
                    top: `${((90 - location.lat) / 180) * 100}%`,
                  }}
                  onClick={() => setSelectedThreat(location)}
                  title={`${location.city}, ${location.country} - ${locationThreats.length} threats`}
                >
                  {hasActiveThreats && (
                    <div className="absolute inset-0 bg-red-500/30 rounded-full animate-ping scale-150"></div>
                  )}
                </div>
              );
            })}
            
            {/* Threat connection lines */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              {threats.slice(0, 5).map((threat, index) => (
                <line
                  key={threat.id}
                  x1={`${((threat.location.lng + 180) / 360) * 100}%`}
                  y1={`${((90 - threat.location.lat) / 180) * 100}%`}
                  x2="50%"
                  y2="50%"
                  stroke={threat.severity === 'HIGH' ? '#ef4444' : '#eab308'}
                  strokeWidth="1"
                  strokeOpacity="0.3"
                  className="animate-pulse"
                />
              ))}
            </svg>
          </div>
          
          {/* Map Legend */}
          <div className="mt-4 flex items-center justify-between text-sm">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                <span className="text-gray-400">Active Threats</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-gray-400">Secure</span>
              </div>
            </div>
            <div className="text-gray-500">
              {isScanning && (
                <div className="flex items-center space-x-2">
                  <Eye className="w-4 h-4 animate-pulse" />
                  <span>Scanning...</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Live Threat Feed */}
        <div className="glass rounded-2xl p-6 border border-gray-600">
          <div className="flex items-center space-x-3 mb-6">
            <AlertTriangle className="w-6 h-6 text-yellow-400 animate-pulse" />
            <h3 className="text-xl font-bold text-white">Live Threat Feed</h3>
            <div className="ml-auto text-sm text-gray-400">
              Last update: {new Date().toLocaleTimeString()}
            </div>
          </div>
          
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {threats.slice(0, 8).map((threat) => (
              <div
                key={threat.id}
                className={`p-4 rounded-lg border transition-all duration-300 hover:scale-105 cursor-pointer ${
                  threat.status === 'ACTIVE' 
                    ? 'bg-red-900/20 border-red-500/30 hover:border-red-400' 
                    : 'bg-green-900/20 border-green-500/30 hover:border-green-400'
                }`}
                onClick={() => setSelectedThreat(threat)}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{threat.icon}</span>
                    <div>
                      <div className="font-bold text-white">{threat.type}</div>
                      <div className="text-sm text-gray-400">
                        {threat.location.city}, {threat.location.country}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-bold ${
                      threat.status === 'ACTIVE' ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {threat.status}
                    </div>
                    <div className="text-xs text-gray-500">
                      {Math.round(threat.confidence * 100)}% confidence
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>{threat.sourceIP}</span>
                  <span>{threat.timestamp.toLocaleTimeString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Threat Details Modal */}
      {selectedThreat && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass rounded-2xl p-8 max-w-2xl w-full border border-gray-600">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold text-white">Threat Details</h3>
              <button
                onClick={() => setSelectedThreat(null)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>
            
            {selectedThreat.type ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-gray-400">Threat Type</label>
                    <div className="text-white font-bold">{selectedThreat.type}</div>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Severity</label>
                    <div className={`font-bold ${
                      selectedThreat.severity === 'HIGH' ? 'text-red-400' :
                      selectedThreat.severity === 'MEDIUM' ? 'text-yellow-400' : 'text-green-400'
                    }`}>
                      {selectedThreat.severity}
                    </div>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Location</label>
                    <div className="text-white">{selectedThreat.location.city}, {selectedThreat.location.country}</div>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Status</label>
                    <div className={`font-bold ${
                      selectedThreat.status === 'ACTIVE' ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {selectedThreat.status}
                    </div>
                  </div>
                </div>
                
                <div>
                  <label className="text-sm text-gray-400">Target APK</label>
                  <div className="text-white font-mono text-sm bg-gray-800/50 p-2 rounded">
                    {selectedThreat.targetAPK}
                  </div>
                </div>
                
                <div>
                  <label className="text-sm text-gray-400">Source IP</label>
                  <div className="text-white font-mono">{selectedThreat.sourceIP}</div>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-400">
                <MapPin className="w-12 h-12 mx-auto mb-4" />
                <h4 className="text-lg font-bold text-white mb-2">
                  {selectedThreat.city}, {selectedThreat.country}
                </h4>
                <p>Click on threat indicators to view detailed information</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ThreatVisualization;

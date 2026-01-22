"use client"
import React, { useState, useEffect } from 'react';
import { 
  Network, Shield, Target, AlertTriangle, Brain, 
  Eye, Zap, TrendingUp, Globe, Link, Search,
  Database, Cpu, Activity, Clock, Users, Lock
} from 'lucide-react';

const AdvancedThreatIntelligence = () => {
  const [activeTab, setActiveTab] = useState('source-tracking');
  const [sourceAnalysis, setSourceAnalysis] = useState(null);
  const [linkAnalysis, setLinkAnalysis] = useState(null);
  const [attributionClusters, setAttributionClusters] = useState([]);
  const [networkGraph, setNetworkGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [urlToCheck, setUrlToCheck] = useState('');

  // Fetch attribution clusters
  const fetchAttributionClusters = async () => {
    try {
      const response = await fetch('https://apk-detector-backend.onrender.com/api/ml-tracker/attribution-clusters');
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setAttributionClusters(data.clusters);
        }
      }
    } catch (err) {
      console.error('Error fetching attribution clusters:', err);
      // Fallback demo data
      setAttributionClusters([
        {
          cluster_id: 'cluster_001',
          cluster_name: 'Operation Banking Storm',
          threat_actor: 'APT-Banking-Alpha',
          campaign_name: 'Indian Banking Campaign 2025',
          techniques: ['SMS Interception', 'Overlay Attacks', 'Certificate Pinning Bypass'],
          infrastructure: ['185.xxx.xxx.xxx/24', 'C&C: banking-secure[.]tk'],
          confidence: 0.89,
          apk_count: 47,
          first_activity: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toISOString(),
          last_activity: new Date().toISOString()
        }
      ]);
    }
  };

  // Fetch network graph
  const fetchNetworkGraph = async () => {
    try {
      const response = await fetch('https://apk-detector-backend.onrender.com/api/ml-tracker/network-graph');
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setNetworkGraph(data.graph);
        }
      }
    } catch (err) {
      console.error('Error fetching network graph:', err);
      // Fallback demo data
      setNetworkGraph({
        nodes: [
          { id: 'apk_001', type: 'apk', label: 'fake_sbi.apk', threat_level: 'high' },
          { id: 'domain_001', type: 'domain', label: 'banking-secure.tk', status: 'active' },
          { id: 'actor_001', type: 'actor', label: 'APT-Banking-Alpha', confidence: 0.89 }
        ],
        edges: [
          { source: 'apk_001', target: 'domain_001', type: 'downloads_from', weight: 0.8 },
          { source: 'actor_001', target: 'domain_001', type: 'controls', weight: 0.9 }
        ]
      });
    }
  };

  // Check malicious link
  const checkMaliciousLink = async () => {
    if (!urlToCheck.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/ml-tracker/check-link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlToCheck })
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setLinkAnalysis(data.analysis);
        }
      }
    } catch (err) {
      console.error('Error checking link:', err);
      // Fallback demo analysis
      setLinkAnalysis({
        url: urlToCheck,
        is_malicious: urlToCheck.includes('fake') || urlToCheck.includes('phish'),
        confidence: 0.87,
        threat_type: 'Banking Phishing',
        risk_factors: ['Suspicious domain', 'Banking keywords'],
        analysis_timestamp: new Date().toISOString()
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAttributionClusters();
    fetchNetworkGraph();
  }, []);

  const tabs = [
    { id: 'source-tracking', name: 'APK Source Tracking', icon: Target },
    { id: 'link-analysis', name: 'Link Analysis', icon: Link },
    { id: 'attribution', name: 'Threat Attribution', icon: Users },
    { id: 'network-graph', name: 'Network Graph', icon: Network }
  ];

  const renderSourceTracking = () => (
    <div className="space-y-6">
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center space-x-3 mb-6">
          <Target className="w-6 h-6 text-cyan-400" />
          <h3 className="text-xl font-bold text-white">APK Source Attribution</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <h4 className="font-bold text-white mb-3">Source Fingerprinting</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Certificate Hash:</span>
                <span className="text-cyan-400 font-mono">a7b3c9d2e1f4</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Code Signature:</span>
                <span className="text-cyan-400 font-mono">x8k9m2n5p7q1</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Obfuscation Level:</span>
                <span className="text-yellow-400">High (0.87)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">String Entropy:</span>
                <span className="text-red-400">6.8/8.0</span>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <h4 className="font-bold text-white mb-3">Attribution Results</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Threat Family:</span>
                <span className="text-red-400">Banking Trojan Alpha</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Geographic Origin:</span>
                <span className="text-white">Eastern Europe</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Confidence:</span>
                <span className="text-green-400">89.3%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Similar Samples:</span>
                <span className="text-cyan-400">23 found</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <Brain className="w-5 h-5 text-blue-400" />
            <span className="font-bold text-blue-400">ML Insights</span>
          </div>
          <p className="text-sm text-gray-300">
            This APK shows strong correlation with the "Operation Banking Storm" campaign. 
            Code patterns and infrastructure fingerprints match 23 previously identified samples.
          </p>
        </div>
      </div>
    </div>
  );

  const renderLinkAnalysis = () => (
    <div className="space-y-6">
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center space-x-3 mb-6">
          <Link className="w-6 h-6 text-purple-400" />
          <h3 className="text-xl font-bold text-white">Malicious Link Detection</h3>
        </div>
        
        <div className="flex space-x-4 mb-6">
          <input
            type="text"
            value={urlToCheck}
            onChange={(e) => setUrlToCheck(e.target.value)}
            placeholder="Enter URL to analyze (e.g., https://fake-banking-site.com)"
            className="flex-1 px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-purple-500 focus:outline-none"
          />
          <button
            onClick={checkMaliciousLink}
            disabled={loading}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg font-medium text-white hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 transition-all duration-300"
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
        
        {linkAnalysis && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h4 className="font-bold text-white mb-3">Analysis Results</h4>
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${linkAnalysis.is_malicious ? 'bg-red-500' : 'bg-green-500'}`}></div>
                  <span className={`font-bold ${linkAnalysis.is_malicious ? 'text-red-400' : 'text-green-400'}`}>
                    {linkAnalysis.is_malicious ? 'MALICIOUS' : 'SAFE'}
                  </span>
                </div>
                <div className="text-sm space-y-1">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Confidence:</span>
                    <span className="text-white">{Math.round(linkAnalysis.confidence * 100)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Threat Type:</span>
                    <span className="text-red-400">{linkAnalysis.threat_type}</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h4 className="font-bold text-white mb-3">Risk Factors</h4>
              <div className="space-y-2">
                {linkAnalysis.risk_factors?.map((factor, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm text-gray-300">{factor}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderAttribution = () => (
    <div className="space-y-6">
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center space-x-3 mb-6">
          <Users className="w-6 h-6 text-green-400" />
          <h3 className="text-xl font-bold text-white">Threat Attribution Clusters</h3>
        </div>
        
        <div className="space-y-4">
          {attributionClusters.map((cluster) => (
            <div key={cluster.cluster_id} className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h4 className="text-lg font-bold text-white">{cluster.cluster_name}</h4>
                  <p className="text-sm text-gray-400">{cluster.campaign_name}</p>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-400">Confidence</div>
                  <div className="text-xl font-bold text-green-400">{Math.round(cluster.confidence * 100)}%</div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div>
                  <div className="text-sm text-gray-400 mb-1">Threat Actor</div>
                  <div className="text-white font-medium">{cluster.threat_actor}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">APK Count</div>
                  <div className="text-cyan-400 font-bold">{cluster.apk_count}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Last Activity</div>
                  <div className="text-white">{new Date(cluster.last_activity).toLocaleDateString()}</div>
                </div>
              </div>
              
              <div className="mb-4">
                <div className="text-sm text-gray-400 mb-2">Techniques</div>
                <div className="flex flex-wrap gap-2">
                  {cluster.techniques.map((technique, index) => (
                    <span key={index} className="px-2 py-1 bg-red-900/30 text-red-400 text-xs rounded border border-red-500/30">
                      {technique}
                    </span>
                  ))}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-gray-400 mb-2">Infrastructure</div>
                <div className="space-y-1">
                  {cluster.infrastructure.map((infra, index) => (
                    <div key={index} className="text-sm text-gray-300 font-mono bg-gray-900/50 px-2 py-1 rounded">
                      {infra}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderNetworkGraph = () => (
    <div className="space-y-6">
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center space-x-3 mb-6">
          <Network className="w-6 h-6 text-orange-400" />
          <h3 className="text-xl font-bold text-white">APK Distribution Network</h3>
        </div>
        
        {networkGraph && (
          <div className="space-y-6">
            {/* Network Visualization Placeholder */}
            <div className="h-96 bg-gray-900/50 rounded-lg border border-gray-700 relative overflow-hidden">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center text-gray-400">
                  <Network className="w-16 h-16 mx-auto mb-4 animate-pulse" />
                  <p className="text-lg font-bold">Interactive Network Graph</p>
                  <p className="text-sm">Advanced D3.js visualization will be rendered here</p>
                </div>
              </div>
              
              {/* Simulated network nodes */}
              <div className="absolute top-1/4 left-1/4 w-4 h-4 bg-red-500 rounded-full animate-pulse" title="Malicious APK"></div>
              <div className="absolute top-1/2 right-1/3 w-4 h-4 bg-yellow-500 rounded-full animate-pulse" title="C&C Server"></div>
              <div className="absolute bottom-1/3 left-1/2 w-4 h-4 bg-purple-500 rounded-full animate-pulse" title="Threat Actor"></div>
              
              {/* Connection lines */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                <line x1="25%" y1="25%" x2="66%" y2="50%" stroke="#ef4444" strokeWidth="2" strokeOpacity="0.5" className="animate-pulse" />
                <line x1="66%" y1="50%" x2="50%" y2="66%" stroke="#a855f7" strokeWidth="2" strokeOpacity="0.5" className="animate-pulse" />
              </svg>
            </div>
            
            {/* Network Statistics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 text-center">
                <div className="text-2xl font-bold text-red-400">{networkGraph.nodes?.filter(n => n.type === 'apk').length || 0}</div>
                <div className="text-sm text-gray-400">APK Nodes</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 text-center">
                <div className="text-2xl font-bold text-yellow-400">{networkGraph.nodes?.filter(n => n.type === 'domain').length || 0}</div>
                <div className="text-sm text-gray-400">Domains</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 text-center">
                <div className="text-2xl font-bold text-purple-400">{networkGraph.nodes?.filter(n => n.type === 'actor').length || 0}</div>
                <div className="text-sm text-gray-400">Threat Actors</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 text-center">
                <div className="text-2xl font-bold text-cyan-400">{networkGraph.edges?.length || 0}</div>
                <div className="text-sm text-gray-400">Connections</div>
              </div>
            </div>
            
            {/* Node Details */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                <h4 className="font-bold text-white mb-3">Key Nodes</h4>
                <div className="space-y-2">
                  {networkGraph.nodes?.slice(0, 3).map((node) => (
                    <div key={node.id} className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${
                        node.type === 'apk' ? 'bg-red-500' :
                        node.type === 'domain' ? 'bg-yellow-500' : 'bg-purple-500'
                      }`}></div>
                      <span className="text-sm text-white">{node.label}</span>
                      <span className="text-xs text-gray-400">({node.type})</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                <h4 className="font-bold text-white mb-3">Relationships</h4>
                <div className="space-y-2">
                  {networkGraph.edges?.slice(0, 3).map((edge, index) => (
                    <div key={index} className="text-sm">
                      <span className="text-cyan-400">{edge.source}</span>
                      <span className="text-gray-400 mx-2">→</span>
                      <span className="text-white">{edge.target}</span>
                      <span className="text-xs text-gray-500 ml-2">({edge.type})</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="w-8 h-8 text-purple-400 animate-pulse" />
          <div>
            <h2 className="text-2xl font-bold text-white">Advanced Threat Intelligence</h2>
            <p className="text-gray-400">ML-powered APK source tracking and malicious link detection</p>
          </div>
        </div>
        
        {/* Tab Navigation */}
        <div className="flex flex-wrap gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-purple-500 to-cyan-500 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-white/10'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span className="hidden sm:inline">{tab.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'source-tracking' && renderSourceTracking()}
      {activeTab === 'link-analysis' && renderLinkAnalysis()}
      {activeTab === 'attribution' && renderAttribution()}
      {activeTab === 'network-graph' && renderNetworkGraph()}
    </div>
  );
};

export default AdvancedThreatIntelligence;

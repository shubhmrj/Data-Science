"use client"
import React, { useState, useEffect, useRef } from 'react';
import { 
  Brain, Zap, Target, AlertTriangle, TrendingUp, Activity,
  Search, Filter, Eye, Cpu, Database, Clock, Shield,
  CheckCircle, XCircle, ArrowRight, BarChart3, PieChart
} from 'lucide-react';

const AIThreatHunting = () => {
  const [huntingStatus, setHuntingStatus] = useState('ACTIVE');
  const [predictions, setPredictions] = useState([]);
  const [patterns, setPatterns] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [huntingStats, setHuntingStats] = useState({
    threatsFound: 0,
    patternsIdentified: 0,
    predictionsAccuracy: 0,
    anomaliesDetected: 0
  });
  const [selectedHunt, setSelectedHunt] = useState(null);
  const [aiInsights, setAiInsights] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(true);

  // AI Threat Hunting Patterns
  const threatPatterns = [
    {
      id: 'banking_trojan_v2',
      name: 'Advanced Banking Trojan Pattern',
      description: 'Sophisticated overlay attacks targeting Indian banking apps',
      confidence: 0.94,
      severity: 'CRITICAL',
      indicators: ['SMS_INTERCEPT', 'OVERLAY_ATTACK', 'BANKING_PERMISSIONS'],
      predictedTargets: ['SBI YONO', 'HDFC Mobile', 'ICICI iMobile'],
      riskScore: 95,
      firstSeen: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
      samples: 23
    },
    {
      id: 'fake_certificate_campaign',
      name: 'Fake Certificate Campaign',
      description: 'Mass distribution of APKs with forged banking certificates',
      confidence: 0.87,
      severity: 'HIGH',
      indicators: ['FAKE_CERT', 'PHISHING', 'SOCIAL_ENGINEERING'],
      predictedTargets: ['Paytm', 'PhonePe', 'Google Pay'],
      riskScore: 82,
      firstSeen: new Date(Date.now() - 1000 * 60 * 60 * 6), // 6 hours ago
      samples: 45
    },
    {
      id: 'sms_stealer_evolution',
      name: 'SMS Stealer Evolution',
      description: 'New variant bypassing Android security restrictions',
      confidence: 0.91,
      severity: 'HIGH',
      indicators: ['SMS_READ', 'NOTIFICATION_ACCESS', 'DEVICE_ADMIN'],
      predictedTargets: ['OTP Systems', '2FA Apps', 'Banking SMS'],
      riskScore: 88,
      firstSeen: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
      samples: 12
    }
  ];

  // AI Predictions
  const aiPredictions = [
    {
      id: 'prediction_1',
      type: 'THREAT_EMERGENCE',
      title: 'New Banking Trojan Variant Predicted',
      description: 'AI models predict emergence of new trojan targeting UPI apps within 48 hours',
      probability: 0.78,
      timeframe: '24-48 hours',
      impact: 'HIGH',
      confidence: 0.85,
      basedOn: ['Pattern analysis', 'Historical data', 'Threat intelligence'],
      recommendations: [
        'Increase monitoring of UPI-related APKs',
        'Update detection signatures',
        'Alert security teams'
      ]
    },
    {
      id: 'prediction_2',
      type: 'ATTACK_CAMPAIGN',
      title: 'Coordinated Phishing Campaign Expected',
      description: 'ML algorithms detect preparation phase of large-scale phishing campaign',
      probability: 0.82,
      timeframe: '12-24 hours',
      impact: 'CRITICAL',
      confidence: 0.91,
      basedOn: ['Network traffic analysis', 'Domain registrations', 'Social media monitoring'],
      recommendations: [
        'Activate emergency response protocol',
        'Notify partner organizations',
        'Prepare countermeasures'
      ]
    }
  ];

  // Anomaly Detection Results
  const detectedAnomalies = [
    {
      id: 'anomaly_1',
      type: 'BEHAVIORAL',
      title: 'Unusual APK Distribution Pattern',
      description: 'Spike in APK downloads from suspicious domains',
      severity: 'MEDIUM',
      confidence: 0.76,
      affectedRegions: ['Mumbai', 'Delhi', 'Bangalore'],
      timeline: 'Last 4 hours',
      metrics: {
        normalRate: '150 APKs/hour',
        currentRate: '890 APKs/hour',
        increase: '493%'
      }
    },
    {
      id: 'anomaly_2',
      type: 'SIGNATURE',
      title: 'Certificate Authority Anomaly',
      description: 'Unusual certificate signing patterns detected',
      severity: 'HIGH',
      confidence: 0.89,
      affectedRegions: ['Global'],
      timeline: 'Last 2 hours',
      metrics: {
        suspiciousCerts: 34,
        legitimateCerts: 156,
        anomalyRatio: '21.8%'
      }
    }
  ];

  // AI Insights Generation
  const generateAIInsights = () => {
    const insights = [
      {
        type: 'TREND_ANALYSIS',
        title: 'Banking Trojans Evolving Rapidly',
        description: 'AI analysis shows 340% increase in sophisticated banking malware over the past month',
        impact: 'HIGH',
        actionRequired: true,
        details: 'New variants are using advanced evasion techniques and targeting specific Indian banking apps'
      },
      {
        type: 'PREDICTIVE_MODEL',
        title: 'Peak Attack Window Identified',
        description: 'ML models predict highest attack probability between 2-4 PM IST',
        impact: 'MEDIUM',
        actionRequired: false,
        details: 'Historical data shows 67% of successful attacks occur during business hours'
      },
      {
        type: 'THREAT_INTELLIGENCE',
        title: 'New Attack Vector Discovered',
        description: 'AI identified previously unknown exploitation method in Android WebView',
        impact: 'CRITICAL',
        actionRequired: true,
        details: 'Zero-day vulnerability being actively exploited by banking trojans'
      }
    ];
    
    setAiInsights(insights);
  };

  // Simulate AI hunting process
  useEffect(() => {
    const interval = setInterval(() => {
      // Update hunting stats
      setHuntingStats(prev => ({
        threatsFound: prev.threatsFound + Math.floor(Math.random() * 3),
        patternsIdentified: prev.patternsIdentified + Math.floor(Math.random() * 2),
        predictionsAccuracy: Math.min(prev.predictionsAccuracy + Math.random() * 0.5, 99.8),
        anomaliesDetected: prev.anomaliesDetected + Math.floor(Math.random() * 2)
      }));

      // Generate new insights periodically
      if (Math.random() > 0.7) {
        generateAIInsights();
      }
    }, 3000);

    // Initial data load
    setPatterns(threatPatterns);
    setPredictions(aiPredictions);
    setAnomalies(detectedAnomalies);
    generateAIInsights();

    return () => clearInterval(interval);
  }, []);

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'CRITICAL': return 'text-red-400 bg-red-900/20 border-red-500/30';
      case 'HIGH': return 'text-orange-400 bg-orange-900/20 border-orange-500/30';
      case 'MEDIUM': return 'text-yellow-400 bg-yellow-900/20 border-yellow-500/30';
      case 'LOW': return 'text-green-400 bg-green-900/20 border-green-500/30';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-500/30';
    }
  };

  const getImpactIcon = (impact) => {
    switch (impact) {
      case 'CRITICAL': return <AlertTriangle className="w-5 h-5 text-red-400 animate-pulse" />;
      case 'HIGH': return <AlertTriangle className="w-5 h-5 text-orange-400" />;
      case 'MEDIUM': return <Target className="w-5 h-5 text-yellow-400" />;
      default: return <Shield className="w-5 h-5 text-green-400" />;
    }
  };

  return (
    <div className="space-y-8">
      {/* AI Threat Hunting Header */}
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Brain className="w-8 h-8 text-purple-400 animate-pulse" />
              <div className="absolute inset-0 bg-purple-400/20 rounded-full animate-ping"></div>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">AI Threat Hunting</h2>
              <p className="text-gray-400">Autonomous threat detection and predictive analysis</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className={`px-4 py-2 rounded-lg border ${
              huntingStatus === 'ACTIVE' 
                ? 'bg-green-900/20 border-green-500/30 text-green-400' 
                : 'bg-gray-900/20 border-gray-500/30 text-gray-400'
            }`}>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  huntingStatus === 'ACTIVE' ? 'bg-green-400 animate-pulse' : 'bg-gray-400'
                }`}></div>
                <span className="font-medium">{huntingStatus}</span>
              </div>
            </div>
          </div>
        </div>

        {/* AI Hunting Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3">
              <Target className="w-6 h-6 text-red-400" />
              <div>
                <div className="text-2xl font-bold text-red-400">{huntingStats.threatsFound}</div>
                <div className="text-sm text-gray-400">Threats Found</div>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3">
              <Search className="w-6 h-6 text-blue-400" />
              <div>
                <div className="text-2xl font-bold text-blue-400">{huntingStats.patternsIdentified}</div>
                <div className="text-sm text-gray-400">Patterns ID'd</div>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3">
              <Brain className="w-6 h-6 text-purple-400" />
              <div>
                <div className="text-2xl font-bold text-purple-400">{huntingStats.predictionsAccuracy.toFixed(1)}%</div>
                <div className="text-sm text-gray-400">AI Accuracy</div>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center space-x-3">
              <Eye className="w-6 h-6 text-cyan-400" />
              <div>
                <div className="text-2xl font-bold text-cyan-400">{huntingStats.anomaliesDetected}</div>
                <div className="text-sm text-gray-400">Anomalies</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Threat Patterns */}
        <div className="glass rounded-2xl p-6 border border-gray-600">
          <div className="flex items-center space-x-3 mb-6">
            <Search className="w-6 h-6 text-blue-400" />
            <h3 className="text-xl font-bold text-white">Identified Threat Patterns</h3>
            <div className="ml-auto text-sm text-gray-400">
              {patterns.length} active patterns
            </div>
          </div>
          
          <div className="space-y-4">
            {patterns.map((pattern) => (
              <div
                key={pattern.id}
                className="p-4 rounded-lg border border-gray-700 hover:border-blue-500/50 transition-all duration-300 cursor-pointer"
                onClick={() => setSelectedHunt(pattern)}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`px-3 py-1 rounded-full text-xs font-bold border ${getSeverityColor(pattern.severity)}`}>
                      {pattern.severity}
                    </div>
                    <div className="text-white font-bold">{pattern.name}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-bold text-blue-400">
                      {Math.round(pattern.confidence * 100)}%
                    </div>
                    <div className="text-xs text-gray-500">confidence</div>
                  </div>
                </div>
                
                <p className="text-gray-400 text-sm mb-3">{pattern.description}</p>
                
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-4">
                    <span className="text-gray-500">
                      <Clock className="w-3 h-3 inline mr-1" />
                      {pattern.firstSeen.toLocaleTimeString()}
                    </span>
                    <span className="text-gray-500">
                      <Database className="w-3 h-3 inline mr-1" />
                      {pattern.samples} samples
                    </span>
                  </div>
                  <div className="text-orange-400 font-bold">
                    Risk: {pattern.riskScore}/100
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* AI Predictions */}
        <div className="glass rounded-2xl p-6 border border-gray-600">
          <div className="flex items-center space-x-3 mb-6">
            <Brain className="w-6 h-6 text-purple-400 animate-pulse" />
            <h3 className="text-xl font-bold text-white">AI Predictions</h3>
            <div className="ml-auto text-sm text-gray-400">
              Next 48 hours
            </div>
          </div>
          
          <div className="space-y-4">
            {predictions.map((prediction) => (
              <div
                key={prediction.id}
                className="p-4 rounded-lg border border-gray-700 hover:border-purple-500/50 transition-all duration-300"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    {getImpactIcon(prediction.impact)}
                    <div className="text-white font-bold">{prediction.title}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-bold text-purple-400">
                      {Math.round(prediction.probability * 100)}%
                    </div>
                    <div className="text-xs text-gray-500">probability</div>
                  </div>
                </div>
                
                <p className="text-gray-400 text-sm mb-3">{prediction.description}</p>
                
                <div className="flex items-center justify-between text-xs mb-3">
                  <span className="text-gray-500">Timeframe: {prediction.timeframe}</span>
                  <span className="text-gray-500">Confidence: {Math.round(prediction.confidence * 100)}%</span>
                </div>
                
                <div className="space-y-1">
                  <div className="text-xs text-gray-400">Recommendations:</div>
                  {prediction.recommendations.slice(0, 2).map((rec, index) => (
                    <div key={index} className="text-xs text-cyan-400 flex items-center">
                      <ArrowRight className="w-3 h-3 mr-1" />
                      {rec}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Anomaly Detection */}
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center space-x-3 mb-6">
          <Activity className="w-6 h-6 text-yellow-400 animate-pulse" />
          <h3 className="text-xl font-bold text-white">Anomaly Detection</h3>
          <div className="ml-auto flex items-center space-x-2">
            <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
            <span className="text-sm text-yellow-400">Monitoring</span>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {anomalies.map((anomaly) => (
            <div
              key={anomaly.id}
              className="p-4 rounded-lg border border-gray-700 hover:border-yellow-500/50 transition-all duration-300"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <div className={`px-3 py-1 rounded-full text-xs font-bold border ${getSeverityColor(anomaly.severity)}`}>
                    {anomaly.severity}
                  </div>
                  <div className="text-white font-bold">{anomaly.title}</div>
                </div>
                <div className="text-sm font-bold text-yellow-400">
                  {Math.round(anomaly.confidence * 100)}%
                </div>
              </div>
              
              <p className="text-gray-400 text-sm mb-3">{anomaly.description}</p>
              
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <div className="text-gray-400">Timeline</div>
                  <div className="text-white">{anomaly.timeline}</div>
                </div>
                <div>
                  <div className="text-gray-400">Regions</div>
                  <div className="text-white">{anomaly.affectedRegions.join(', ')}</div>
                </div>
              </div>
              
              {anomaly.metrics && (
                <div className="mt-3 p-2 bg-gray-800/50 rounded text-xs">
                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <div className="text-gray-400">Normal</div>
                      <div className="text-green-400">{anomaly.metrics.normalRate || anomaly.metrics.legitimateCerts}</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Current</div>
                      <div className="text-red-400">{anomaly.metrics.currentRate || anomaly.metrics.suspiciousCerts}</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Change</div>
                      <div className="text-yellow-400">{anomaly.metrics.increase || anomaly.metrics.anomalyRatio}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* AI Insights */}
      <div className="glass rounded-2xl p-6 border border-gray-600">
        <div className="flex items-center space-x-3 mb-6">
          <Cpu className="w-6 h-6 text-cyan-400" />
          <h3 className="text-xl font-bold text-white">AI Insights & Intelligence</h3>
          <div className="ml-auto text-sm text-gray-400">
            Powered by ML algorithms
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {aiInsights.map((insight, index) => (
            <div
              key={index}
              className="p-4 rounded-lg border border-gray-700 hover:border-cyan-500/50 transition-all duration-300"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  {getImpactIcon(insight.impact)}
                  <div className={`px-2 py-1 rounded text-xs font-bold ${getSeverityColor(insight.impact)}`}>
                    {insight.type}
                  </div>
                </div>
                {insight.actionRequired && (
                  <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse"></div>
                )}
              </div>
              
              <h4 className="text-white font-bold mb-2">{insight.title}</h4>
              <p className="text-gray-400 text-sm mb-3">{insight.description}</p>
              <p className="text-gray-500 text-xs">{insight.details}</p>
              
              {insight.actionRequired && (
                <div className="mt-3 px-3 py-2 bg-red-900/20 border border-red-500/30 rounded text-xs text-red-400">
                  <CheckCircle className="w-3 h-3 inline mr-1" />
                  Action Required
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Detailed Hunt Modal */}
      {selectedHunt && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass rounded-2xl p-8 max-w-4xl w-full border border-gray-600 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold text-white">Threat Pattern Details</h3>
              <button
                onClick={() => setSelectedHunt(null)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4">Pattern Information</h4>
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm text-gray-400">Pattern Name</label>
                      <div className="text-white font-bold">{selectedHunt.name}</div>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Severity</label>
                      <div className={`inline-block px-3 py-1 rounded-full text-xs font-bold border ${getSeverityColor(selectedHunt.severity)}`}>
                        {selectedHunt.severity}
                      </div>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Confidence</label>
                      <div className="text-white">{Math.round(selectedHunt.confidence * 100)}%</div>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Risk Score</label>
                      <div className="text-red-400 font-bold">{selectedHunt.riskScore}/100</div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4">Detection Metrics</h4>
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm text-gray-400">First Detected</label>
                      <div className="text-white">{selectedHunt.firstSeen.toLocaleString()}</div>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Sample Count</label>
                      <div className="text-white">{selectedHunt.samples} APKs</div>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Status</label>
                      <div className="text-red-400 font-bold">ACTIVE THREAT</div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="text-lg font-bold text-white mb-4">Description</h4>
                <p className="text-gray-300">{selectedHunt.description}</p>
              </div>
              
              <div>
                <h4 className="text-lg font-bold text-white mb-4">Threat Indicators</h4>
                <div className="flex flex-wrap gap-2">
                  {selectedHunt.indicators.map((indicator, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-red-900/20 border border-red-500/30 text-red-400 rounded-full text-sm"
                    >
                      {indicator}
                    </span>
                  ))}
                </div>
              </div>
              
              <div>
                <h4 className="text-lg font-bold text-white mb-4">Predicted Targets</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {selectedHunt.predictedTargets.map((target, index) => (
                    <div
                      key={index}
                      className="p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg text-center"
                    >
                      <div className="text-yellow-400 font-bold">{target}</div>
                      <div className="text-xs text-gray-400">High Risk</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIThreatHunting;

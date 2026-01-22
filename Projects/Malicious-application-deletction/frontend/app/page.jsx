"use client"
import React, { useState, useEffect } from 'react';
import { 
  Shield, Upload, Brain, Zap, Target, Activity, 
  CheckCircle, AlertTriangle, TrendingUp, Users,
  Globe, Lock, Eye, Cpu, Database, ArrowRight,
  Play, Github, ExternalLink, Menu, X, Sparkles
} from 'lucide-react';
import LiveStatsPanel from './Components/LiveStatsPanel';
import EnhancedScanSection from './Components/EnhancedScanSection';
import AdvancedDashboard from './Components/AdvancedDashboard';

const ModernAPKDetector = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [currentMetric, setCurrentMetric] = useState(0);
  const [viewMode, setViewMode] = useState('classic'); // 'classic' or 'advanced'

  const metrics = [
    { label: 'APKs Scanned', value: '50+', icon: Database, color: 'from-blue-500 to-cyan-500' },
    { label: 'Threats Detected', value: '1,000', icon: Shield, color: 'from-red-500 to-pink-500' },
    { label: 'Accuracy Rate', value: '99.8%', icon: Target, color: 'from-green-500 to-emerald-500' },
    { label: 'Response Time', value: '<2s', icon: Zap, color: 'from-yellow-500 to-orange-500' }
  ];

  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Detection',
      description: 'Advanced machine learning with 52-feature analysis',
      stats: '99.8% accuracy',
      color: 'from-purple-500 to-indigo-500'
    },
    {
      icon: Zap,
      title: 'Real-time Analysis',
      description: 'Lightning-fast APK scanning in under 2 seconds',
      stats: '<2s response',
      color: 'from-yellow-500 to-orange-500'
    },
    {
      icon: Eye,
      title: 'Behavioral Monitoring',
      description: 'Dynamic analysis of app behavior patterns',
      stats: '500+ patterns',
      color: 'from-green-500 to-teal-500'
    },
    {
      icon: Lock,
      title: 'Zero False Positives',
      description: 'Reliable protection with enterprise-grade accuracy',
      stats: '0% false+',
      color: 'from-cyan-500 to-blue-500'
    }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentMetric((prev) => (prev + 1) % metrics.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleFileUpload = async (file) => {
    if (!file) return;
    
    setIsAnalyzing(true);
    setProgress(0);
    
    // Simulate analysis progress
    const steps = [20, 40, 60, 80, 100];
    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 800));
      setProgress(steps[i]);
    }
    
    setTimeout(() => {
      setIsAnalyzing(false);
      setProgress(0);
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" style={{animationDelay: '2s'}}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-green-500/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '4s'}}></div>
      </div>

      {/* Header */}
      <header className="relative z-50 bg-black/20 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-xl flex items-center justify-center">
                <Shield className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">APK<span className="text-cyan-400">Guard</span></h1>
                <p className="text-xs text-gray-400">AI Security Platform</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative z-10 pt-20 pb-32 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center space-x-2 px-4 py-2 bg-white/5 backdrop-blur-sm rounded-full text-sm text-gray-300 mb-8 border border-white/10">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>🛡️ Advanced ML Threat Detection System</span>
              <div className="px-2 py-1 bg-green-500/20 rounded-full text-xs text-green-400 font-bold">LIVE</div>
            </div>
            
            <h1 className="text-6xl md:text-8xl font-black mb-8 leading-tight">
              <span className="block mb-4">Detect Fake</span>
              <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
                Banking Apps
              </span>
              <span className="block mt-4 text-5xl md:text-6xl">Before They Strike</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-4xl mx-auto leading-relaxed">
              AI-powered real-time protection against sophisticated malicious APKs disguised as legitimate banking applications.
              <br />
              <span className="text-purple-300">Safeguard millions with cutting-edge machine learning detection.</span>
            </p>

            {/* Dynamic Metrics */}
            <div className="mb-12">
              <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-8 max-w-md mx-auto border border-white/10">
                <div className="flex items-center justify-center space-x-6">
                  <div className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${metrics[currentMetric].color} flex items-center justify-center`}>
                    {React.createElement(metrics[currentMetric].icon, { className: 'w-8 h-8 text-white' })}
                  </div>
                  <div className="text-left">
                    <div className="text-3xl font-black text-white mb-1">
                      {metrics[currentMetric].value}
                    </div>
                    <div className="text-gray-300 text-sm">
                      {metrics[currentMetric].label}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6">
              <button 
                onClick={() => document.getElementById('scan-section')?.scrollIntoView({ behavior: 'smooth' })}
                className="group flex items-center space-x-3 px-10 py-4 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-xl font-bold text-lg hover:from-purple-600 hover:to-cyan-600 transition-all shadow-2xl hover:shadow-purple-500/50 hover:scale-105"
              >
                <Shield className="w-6 h-6" />
                <span>Start AI Scan</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
            
            </div>
          </div>
        </div>
      </section>

      {/* Scan Section */}
      <section id="scan-section" className="relative z-10 py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-6">
              <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                AI-Powered Analysis
              </span>
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Upload APK files for instant ML-powered threat detection and comprehensive security analysis
            </p>
          </div>

          {/* Replace basic upload with full EnhancedScanSection */}
          <EnhancedScanSection />
          
          {/* Live Stats Panel */}
          <div className="mt-12">
            <LiveStatsPanel />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative z-10 py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-6">
              <span className="text-white">Advanced</span>
              <br />
              <span className="bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                Security Features
              </span>
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10 hover:bg-white/10 transition-all hover:scale-105">
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${feature.color} flex items-center justify-center mb-4`}>
                  {React.createElement(feature.icon, { className: 'w-6 h-6 text-white' })}
                </div>
                <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
                <p className="text-gray-400 text-sm mb-4">{feature.description}</p>
                <div className="text-lg font-bold text-cyan-400">{feature.stats}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 bg-black/40 backdrop-blur-xl border-t border-white/10 py-12 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold">APK<span className="text-cyan-400">Guard</span></span>
          </div>
          <p className="text-gray-400 mb-6">AI-Powered Banking APK Security Platform</p>
          <div className="flex items-center justify-center space-x-6">
            <a href="#" className="text-gray-400 hover:text-white transition-colors">Privacy</a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">Terms</a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">API Docs</a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">Support</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default ModernAPKDetector;

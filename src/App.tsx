import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Package, Zap, BarChart3 } from 'lucide-react';

// Import page components
import Dashboard from './pages/Dashboard';
import Products from './pages/Products';
import Recommendations from './pages/Recommendations';
import Results from './pages/Results';

// Navigation Component
function Navigation() {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/products', label: 'Products', icon: Package },
    { path: '/recommendations', label: 'Recommendations', icon: Zap },
    { path: '/results', label: 'Results', icon: BarChart3 },
  ];

  return (
    <nav className="sticky top-0 z-50 bg-gradient-to-r from-indigo-600 to-purple-600 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 group">
            <div className="bg-white p-2 rounded-lg group-hover:scale-110 transition-transform">
              <Zap className="w-5 h-5 text-indigo-600" />
            </div>
            <span className="text-xl font-bold text-white tracking-tight">
              SmartRec
            </span>
            <span className="ml-2 px-2 py-0.5 rounded text-xs font-medium bg-white/20 text-white border border-white/30">
              AI Recommendation Engine
            </span>
          </Link>

          {/* Nav Links */}
          <div className="flex items-center gap-1">
            {navItems.map(({ path, label, icon: Icon }) => {
              const isActive = location.pathname === path;
              return (
                <Link
                  key={path}
                  to={path}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                    isActive
                      ? 'bg-white text-indigo-600 shadow-lg'
                      : 'text-white hover:bg-white/10'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="inline">{label}</span>
                </Link>
              );
            })}
          </div>

          {/* Status Badge */}
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
            <span className="text-white text-sm font-medium">System Live</span>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-neutral-50 font-sans text-neutral-900">
        <Navigation />
        
        {/* Page Routes */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/products" element={<Products />} />
            <Route path="/recommendations" element={<Recommendations />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t border-neutral-200 mt-16 py-8">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              <div>
                <h3 className="font-bold text-neutral-900 mb-4">SmartRec</h3>
                <p className="text-neutral-500 text-sm">
                  AI-powered recommendation engine using PyTorch & Neural Collaborative Filtering
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold text-neutral-900 mb-4">Features</h4>
                <ul className="space-y-2 text-sm text-neutral-500">
                  <li><Link to="/" className="hover:text-indigo-600">Analytics Dashboard</Link></li>
                  <li><Link to="/products" className="hover:text-indigo-600">Product Catalog</Link></li>
                  <li><Link to="/recommendations" className="hover:text-indigo-600">Personalization</Link></li>
                  <li><Link to="/results" className="hover:text-indigo-600">Model Metrics</Link></li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold text-neutral-900 mb-4">Tech Stack</h4>
                <ul className="space-y-2 text-sm text-neutral-500">
                  <li>Python 3.12 + PyTorch 2.2</li>
                  <li>FastAPI + Node.js</li>
                  <li>React 19 + Vite</li>
                  <li>TailwindCSS + Recharts</li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold text-neutral-900 mb-4">Models</h4>
                <ul className="space-y-2 text-sm text-neutral-500">
                  <li>Matrix Factorization</li>
                  <li>Neural Collaborative Filtering</li>
                  <li>Scoring (CTR/CVR-style)</li>
                  <li>Real-time Inference</li>
                </ul>
              </div>
            </div>
            
            <div className="border-t border-neutral-200 mt-8 pt-8 flex flex-col sm:flex-row justify-between items-center text-sm text-neutral-500">
              <p>&copy; 2026 SmartRec. Open Source AI Project – Matrix Factorization & NCF.</p>
              <p>Scale: 500 Users × 500 Products × 50K Interactions</p>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
}

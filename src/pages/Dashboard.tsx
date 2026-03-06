import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { Activity, Zap, Server, ShieldCheck, Loader2, ArrowUpRight, Cpu, Radio, Database } from 'lucide-react';
import { motion } from 'motion/react';

export default function DashboardPage() {
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalProducts: 0,
    totalInteractions: 0
  });
  const [loading, setLoading] = useState(true);
  const [liveTraffic, setLiveTraffic] = useState<{time: string, req: number}[]>([]);

  // Simulate live traffic data
  useEffect(() => {
    const initialData = Array.from({length: 20}, (_, i) => ({
      time: `-${20-i}s`,
      req: Math.floor(Math.random() * 50) + 120
    }));
    setLiveTraffic(initialData);

    const interval = setInterval(() => {
      setLiveTraffic(prev => {
        const newData = [...prev.slice(1)];
        newData.push({
          time: 'Now',
          req: Math.floor(Math.random() * 50) + 120
        });
        return newData;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    Promise.all([
      fetch('/api/users').then(res => res.json()),
      fetch('/api/products').then(res => res.json()).catch(() => [])
    ]).then(([users, products = []]) => {
      setStats({
        totalUsers: users.length,
        totalProducts: products.length || 0,
        totalInteractions: 100000 // Actual movielens size
      });
      setLoading(false);
    }).catch(() => {
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
      </div>
    );
  }

  const latencyData = [
    { name: 'P50', ms: 42 },
    { name: 'P90', ms: 85 },
    { name: 'P95', ms: 142 },
    { name: 'P99', ms: 210 }
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900 flex items-center gap-3">
             <Server className="w-8 h-8 text-indigo-600" /> System Overview
          </h1>
          <p className="text-neutral-500 mt-1">SmartRec MLOps Control Center</p>
        </div>
        <div className="flex items-center gap-3 bg-emerald-50 px-4 py-2 rounded-full border border-emerald-200">
           <Radio className="w-4 h-4 text-emerald-600 animate-pulse" />
           <span className="text-sm font-semibold text-emerald-700">All Systems Operational</span>
        </div>
      </div>

      {/* Primary Telemetry Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div initial={{opacity:0, y:20}} animate={{opacity:1, y:0}} className="bg-slate-900 rounded-2xl p-6 text-white relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Cpu className="w-16 h-16" />
          </div>
          <p className="text-indigo-300 text-sm font-medium mb-1 uppercase tracking-wider">Active Model</p>
          <p className="text-3xl font-bold tracking-tight">NCF-v2.1</p>
          <div className="mt-4 flex items-center gap-2 text-xs font-mono text-emerald-400">
            <ArrowUpRight className="w-3 h-3" /> Deployed
          </div>
        </motion.div>

        <motion.div initial={{opacity:0, y:20}} animate={{opacity:1, y:0}} transition={{delay:0.1}} className="bg-white rounded-2xl p-6 border border-neutral-200 shadow-sm">
          <div className="flex justify-between items-start mb-1">
             <p className="text-neutral-500 text-sm font-medium uppercase tracking-wider">P99 Latency</p>
             <Zap className="w-5 h-5 text-amber-500" />
          </div>
          <p className="text-3xl font-bold text-neutral-900">142<span className="text-lg text-neutral-400 font-normal">ms</span></p>
          <div className="mt-4 w-full bg-neutral-100 rounded-full h-1.5">
            <div className="bg-amber-500 h-1.5 rounded-full" style={{ width: '45%' }}></div>
          </div>
        </motion.div>

        <motion.div initial={{opacity:0, y:20}} animate={{opacity:1, y:0}} transition={{delay:0.2}} className="bg-white rounded-2xl p-6 border border-neutral-200 shadow-sm">
          <div className="flex justify-between items-start mb-1">
             <p className="text-neutral-500 text-sm font-medium uppercase tracking-wider">Data Lake</p>
             <Database className="w-5 h-5 text-blue-500" />
          </div>
          <p className="text-3xl font-bold text-neutral-900">{stats.totalInteractions.toLocaleString()}</p>
          <p className="text-sm text-neutral-400 mt-1">Processed Interactions</p>
        </motion.div>
        
        <motion.div initial={{opacity:0, y:20}} animate={{opacity:1, y:0}} transition={{delay:0.3}} className="bg-white rounded-2xl p-6 border border-neutral-200 shadow-sm">
          <div className="flex justify-between items-start mb-1">
             <p className="text-neutral-500 text-sm font-medium uppercase tracking-wider">Vectors Indexed</p>
             <Activity className="w-5 h-5 text-indigo-500" />
          </div>
          <p className="text-3xl font-bold text-neutral-900">{(stats.totalUsers + stats.totalProducts).toLocaleString()}</p>
          <p className="text-sm text-neutral-400 mt-1">Embeddings (Dim: 64)</p>
        </motion.div>
      </div>

      {/* Main Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Live Traffic */}
        <div className="lg:col-span-2 bg-white rounded-2xl p-6 shadow-sm border border-neutral-200">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-lg font-bold text-neutral-900 flex items-center gap-2">
               <Activity className="w-5 h-5 text-indigo-600" /> Live Inference Traffic
            </h2>
            <span className="flex h-3 w-3 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500"></span>
            </span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={liveTraffic} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorReq" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4f46e5" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#4f46e5" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
              <XAxis dataKey="time" stroke="#9ca3af" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#9ca3af" fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
              <Area type="monotone" dataKey="req" stroke="#4f46e5" strokeWidth={3} fillOpacity={1} fill="url(#colorReq)" animationDuration={300} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Latency Distribution */}
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-neutral-200 flex flex-col">
          <h2 className="text-lg font-bold text-neutral-900 mb-6 flex items-center gap-2">
             <Zap className="w-5 h-5 text-amber-500" /> Engine Latency Distribution
          </h2>
          <div className="flex-1">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={latencyData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                <XAxis dataKey="name" stroke="#9ca3af" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#9ca3af" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip cursor={{fill: '#f3f4f6'}} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                <Bar dataKey="ms" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                  {latencyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === 3 ? '#ef4444' : index === 2 ? '#f59e0b' : '#3b82f6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

// Remove old local functions
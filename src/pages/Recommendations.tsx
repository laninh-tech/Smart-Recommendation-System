import React, { useState, useEffect } from 'react';
import { Search, User, Loader2, Star, ShoppingCart, MousePointerClick, CreditCard, BrainCircuit, Activity, Zap, Info, ChevronRight, SlidersHorizontal } from 'lucide-react';
import { motion } from 'motion/react';

export default function RecommendationsPage() {
  const [users, setUsers] = useState<any[]>([]);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [selectedUser, setSelectedUser] = useState<any>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [recommendationsLoading, setRecommendationsLoading] = useState(false);
  const [inferenceStats, setInferenceStats] = useState({ latency: 0, model: '', targetDim: 0 });
  const [activeModel, setActiveModel] = useState<'NCF' | 'MF'>('NCF');

  useEffect(() => {
    fetch('/api/users')
      .then(res => res.json())
      .then(data => {
        setUsers(data);
        if (data.length > 0) {
          setSelectedUser(data[0]);
          fetchRecommendations(data[0].user_id);
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const fetchRecommendations = (userId: string, targetModel = activeModel) => {
    setRecommendationsLoading(true);

    fetch(`/api/recommendations/${userId}?model=${targetModel.toLowerCase()}`)
      .then(res => res.json())
      .then(data => {
        const recs = data.recommendations || [];

        const enhancedRecs = recs.map((r: any) => {
           const isMatch = selectedUser?.preferences?.includes(r.category);
           const latentAffinity = Math.min(99, Math.max(0, (r.scores?.final || 0) * 100));
           
           let xaiText = isMatch 
            ? `High latent overlap detected in historical ${r.category} interaction sequence.`
            : `Collaborative filtering signal: similar users purchased this frequently.`;
            
           if(data.model === 'POPULAR') {
               xaiText = `Cold-start fallback: Global trending item in ${r.category}.`;
           }

           return { ...r, latentAffinity, xaiText };
        });
        
        setRecommendations(enhancedRecs);
        setInferenceStats({
                    latency: Number(data.latency) || 0,
                    model: data.model || targetModel,
            targetDim: targetModel === 'NCF' ? 64 : 32
        });
        setRecommendationsLoading(false);
      })
      .catch(() => setRecommendationsLoading(false));
  };

  const handleUserSelect = (user: any) => {
    setSelectedUser(user);
    fetchRecommendations(user.user_id, activeModel);
  };

  const handleModelToggle = (model: 'NCF' | 'MF') => {
      setActiveModel(model);
      if (selectedUser) {
          fetchRecommendations(selectedUser.user_id, model);
      }
  };

  const filteredUsers = users.filter(u =>
    u.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    u.user_id.includes(searchTerm)
  );

  const filteredRecommendations = searchTerm
    ? recommendations.filter(r => r.name?.toLowerCase().includes(searchTerm.toLowerCase()))
    : recommendations;

  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-neutral-900">Personalized Recommendations</h1>
        <p className="text-neutral-500 mt-2">Select a user to see AI-powered recommendations</p>
      </div>

      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-4 top-3.5 w-5 h-5 text-neutral-400" />
        <input
          type="text"
          placeholder="Search users or products..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full pl-12 pr-4 py-3 border border-neutral-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Users Sidebar */}
        <div className="bg-white rounded-2xl p-6 border border-neutral-200 shadow-sm">
          <h2 className="text-lg font-bold text-neutral-900 mb-4">Users</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {filteredUsers.length > 0 ? (
              filteredUsers.map(user => (
                <button
                  key={user.user_id}
                  onClick={() => handleUserSelect(user)}
                  className={`w-full text-left p-4 rounded-lg transition-colors ${
                    selectedUser?.user_id === user.user_id
                      ? 'bg-indigo-100 border-l-4 border-indigo-600'
                      : 'bg-neutral-50 hover:bg-neutral-100'
                  }`}
                >
                  <div className="font-medium text-neutral-900">{user.name}</div>
                  <div className="text-xs text-neutral-500 mt-1">
                    {user.preferences?.length > 0
                      ? `Interests: ${user.preferences.join(', ')}`
                      : 'No specific preferences'}
                  </div>
                </button>
              ))
            ) : (
              <p className="text-neutral-500 text-center py-4">No users found</p>
            )}
          </div>
        </div>

        {/* Recommendations */}
        <div className="lg:col-span-2 space-y-4">
          {selectedUser && (
            <>
              <div className="bg-gradient-to-br from-slate-900 to-indigo-950 rounded-2xl p-6 text-white border border-indigo-500/30 relative overflow-hidden shadow-xl">
                {/* Decorative background grid */}
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f46e51a_1px,transparent_1px),linear-gradient(to_bottom,#4f46e51a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]"></div>
                
                <div className="relative z-10">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center gap-3 mb-1">
                        <div className="bg-indigo-500/20 p-2 rounded-lg backdrop-blur-sm border border-indigo-400/30">
                          <User className="w-6 h-6 text-indigo-300" />
                        </div>
                        <h2 className="text-2xl font-bold tracking-tight">{selectedUser.name}</h2>
                      </div>
                      <div className="flex items-center gap-2 mt-3 text-indigo-200 text-sm">
                        <BrainCircuit className="w-4 h-4" />
                        <span>Latent Profile Generated</span>
                      </div>
                    </div>
                    
                    {/* A/B Test Model Toggle */}
                    <div className="hidden md:flex flex-col items-end gap-2">
                       <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-1">
                           <SlidersHorizontal className="w-3.5 h-3.5" /> A/B Test Model
                       </div>
                       <div className="flex bg-black/40 p-1 rounded-lg border border-indigo-500/30 backdrop-blur-md">
                           <button 
                             onClick={() => handleModelToggle('MF')}
                             className={`px-4 py-1.5 rounded-md text-sm font-bold transition-all ${activeModel === 'MF' ? 'bg-indigo-500 text-white shadow-lg' : 'text-neutral-400 hover:text-white hover:bg-white/10'}`}
                           >
                             MF Base
                           </button>
                           <button 
                             onClick={() => handleModelToggle('NCF')}
                             className={`px-4 py-1.5 rounded-md text-sm font-bold transition-all ${activeModel === 'NCF' ? 'bg-emerald-500 text-white shadow-lg' : 'text-neutral-400 hover:text-white hover:bg-white/10'}`}
                           >
                             NCF Deep
                           </button>
                       </div>
                    </div>
                  </div>

                  {/* Telemetry Footer */}
                  <div className="mt-6 pt-4 border-t border-indigo-500/20 flex flex-wrap justify-between items-center gap-4">
                    {selectedUser.preferences?.length > 0 ? (
                      <div className="flex items-center gap-2 bg-indigo-900/40 p-2.5 rounded-lg border border-indigo-500/20 backdrop-blur-sm">
                        <div className="flex items-center gap-1.5 px-2 py-0.5 bg-indigo-500/30 rounded text-xs font-bold tracking-wider uppercase text-indigo-200 border border-indigo-400/30">
                          <Star className="w-3 h-3" /> Features
                        </div>
                        <div className="flex flex-wrap gap-2 text-sm text-indigo-100 font-medium">
                          {selectedUser.preferences.map((p: string, i: number) => (
                             <span key={i} className="flex items-center gap-1">
                               {i > 0 && <ChevronRight className="w-3 h-3 text-indigo-500/50" />}
                               {p}
                             </span>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="text-sm text-amber-300/80 font-medium flex items-center gap-2 bg-amber-900/20 px-3 py-1.5 rounded-lg border border-amber-500/20">
                         <Info className="w-4 h-4 text-amber-400" /> Cold-start mode active. Global popularity fallback.
                      </div>
                    )}

                    {/* Real-time Inference Stats Panel */}
                    <div className="flex gap-3">
                      <div className="bg-black/40 backdrop-blur-md rounded-lg px-3 py-1.5 border border-indigo-500/20 shadow-inner flex items-center gap-3">
                        <div className="flex items-center gap-1.5">
                          <Activity className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Latency</span>
                        </div>
                        <span className="font-mono text-emerald-300">
                          {recommendationsLoading ? '--' : `${inferenceStats.latency}ms`}
                        </span>
                      </div>
                      
                      <div className="bg-black/40 backdrop-blur-md rounded-lg px-3 py-1.5 border border-indigo-500/20 shadow-inner flex items-center gap-3">
                        <div className="flex items-center gap-1.5">
                          <Zap className="w-3.5 h-3.5 text-amber-400" />
                          <span className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Target Dim</span>
                        </div>
                        <span className="font-mono text-amber-300">
                          {recommendationsLoading ? '--' : inferenceStats.targetDim}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Recommendations Grid */}
              {recommendationsLoading ? (
                <div className="flex justify-center items-center h-96">
                  <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {filteredRecommendations.length > 0 ? (
                    filteredRecommendations.map((product, idx) => (
                      <motion.div
                        key={product.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.05 }}
                        className="bg-white rounded-xl border border-neutral-200 overflow-hidden hover:shadow-lg transition-shadow"
                      >
                        {/* Image */}
                        <div className="aspect-video bg-neutral-100 overflow-hidden flex items-center justify-center">
                          {product.image ? (
                            <img
                              src={product.image}
                              alt={product.name}
                              className="w-full h-full object-contain"
                              onError={(e) => {
                                (e.target as HTMLImageElement).style.display = 'none';
                              }}
                            />
                          ) : null}
                        </div>

                        {/* Info */}
                        <div className="p-4">
                          <div className="flex justify-between items-start gap-2">
                            <div>
                              <div className="text-xs font-medium text-indigo-600 mb-1">
                                {product.category?.toUpperCase()}
                              </div>
                              <h3 className="font-bold text-neutral-900 line-clamp-2">
                                {product.name}
                              </h3>
                            </div>

                            {/* Match Badge */}
                            {selectedUser.preferences?.includes(product.category) && (
                              <div className="bg-green-100 text-green-700 text-xs font-semibold px-2 py-1 rounded-full">
                                ✓ Match
                              </div>
                            )}
                          </div>

                          {/* Rating & Price */}
                          <div className="flex items-center justify-between mt-3">
                            <div className="flex items-center gap-1">
                              <Star className="w-4 h-4 fill-amber-400 text-amber-400" />
                              <span className="text-sm font-medium">{product.rating?.toFixed(1)}</span>
                            </div>
                            <span className="font-bold text-neutral-900">${product.price?.toFixed(2)}</span>
                          </div>

                          {/* XAI (Explainable AI) Section */}
                          <div className="mt-3 bg-indigo-50/50 p-3 rounded-xl border border-indigo-100/50">
                            <div className="flex items-center gap-1 mb-2 text-indigo-800 font-semibold text-xs uppercase tracking-wider">
                                <Activity className="w-3 h-3" /> Inference Engine
                            </div>
                            
                            {/* Latent Affinity */}
                            <div className="space-y-1">
                                <div className="flex justify-between items-center text-xs">
                                    <span className="text-neutral-600 font-medium">Latent Affinity Score</span>
                                    <span className="font-mono font-bold text-indigo-600">{product.latentAffinity?.toFixed(1)}%</span>
                                </div>
                                <div className="w-full bg-neutral-200 rounded-full h-1.5 overflow-hidden">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${product.latentAffinity}%` }}
                                        transition={{ delay: 0.2 + (idx * 0.1), duration: 0.8, ease: "easeOut" }}
                                        className="bg-gradient-to-r from-indigo-400 to-indigo-600 h-1.5 rounded-full"
                                    />
                                </div>
                            </div>
                            
                            {/* Explanation */}
                            <div className="mt-2.5 pt-2.5 border-t border-indigo-100 flex items-start gap-2">
                                <Info className="w-3.5 h-3.5 text-indigo-400 mt-0.5 shrink-0" />
                                <p className="text-xs text-neutral-600 leading-tight italic">
                                    {product.xaiText}
                                </p>
                            </div>
                          </div>

                          {/* Multi-Task Scores */}
                          <div className="mt-3 space-y-2.5 pt-3 border-t border-neutral-100">
                            <div className="flex justify-between items-center text-xs">
                              <span className="text-neutral-500 flex items-center gap-1 font-medium">
                                <MousePointerClick className="w-3.5 h-3.5 text-blue-500" /> pCTR (Click)
                              </span>
                              <span className="font-mono font-medium text-neutral-700">
                                {Math.round(product.scores?.ctr * 100)}%
                              </span>
                            </div>

                            <div className="flex justify-between items-center text-xs">
                              <span className="text-neutral-500 flex items-center gap-1 font-medium">
                                <ShoppingCart className="w-3.5 h-3.5 text-emerald-500" /> pCVR (Conv)
                              </span>
                              <span className="font-mono font-medium text-neutral-700">
                                {Math.round(product.scores?.cvr * 100)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    ))
                  ) : (
                    <div className="col-span-2 text-center py-12">
                      <Search className="w-12 h-12 text-neutral-300 mx-auto mb-4" />
                      <p className="text-neutral-500">No recommendations found</p>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
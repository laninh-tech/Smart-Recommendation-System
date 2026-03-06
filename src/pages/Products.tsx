import React, { useState, useEffect } from 'react';
import { Search, Package, Star, Loader2 } from 'lucide-react';

export default function ProductsPage() {
  const [products, setProducts] = useState<any[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [loading, setLoading] = useState(true);
  const [categories, setCategories] = useState<string[]>([]);

  useEffect(() => {
    fetch('/api/products')
      .then(res => res.json())
      .then(data => {
        setProducts(data);
        const cats = [...new Set(data.map((p: any) => p.category))];
        setCategories(cats as string[]);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const filteredProducts = products.filter(p => {
    const matchesSearch = p.title?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         p.category?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || p.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

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
        <h1 className="text-4xl font-bold text-neutral-900">All Products</h1>
        <p className="text-neutral-500 mt-2">Explore {products.length}+ products in our catalog</p>
      </div>

      {/* Search Section */}
      <div className="space-y-4">
        <div className="relative">
          <Search className="absolute left-4 top-3.5 w-5 h-5 text-neutral-400" />
          <input
            type="text"
            placeholder="Search products, categories..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-12 pr-4 py-3 border border-neutral-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setSelectedCategory('all')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedCategory === 'all'
                ? 'bg-indigo-600 text-white'
                : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
            }`}
          >
            All Categories
          </button>
          {categories.map(cat => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedCategory === cat
                  ? 'bg-indigo-600 text-white'
                  : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Products Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {filteredProducts.length > 0 ? (
          filteredProducts.map(product => (
            <ProductCard key={product.product_id} product={product} />
          ))
        ) : (
          <div className="col-span-full text-center py-12">
            <Search className="w-12 h-12 text-neutral-300 mx-auto mb-4" />
            <p className="text-neutral-500 font-medium">No products found</p>
          </div>
        )}
      </div>

      {/* Results Count */}
      <div className="text-center text-neutral-600">
        Showing {filteredProducts.length} of {products.length} products
      </div>
    </div>
  );
}

function ProductCard({ product }: { product: any }) {
  return (
    <div className="bg-white rounded-xl border border-neutral-200 overflow-hidden hover:shadow-lg transition-shadow group">
      {/* Image */}
      <div className="aspect-square bg-neutral-100 overflow-hidden flex items-center justify-center p-4">
        {product.thumbnail ? (
          <img
            src={product.thumbnail}
            alt={product.title}
            className="w-full h-full object-contain group-hover:scale-105 transition-transform"
            onError={(e) => {
              (e.target as HTMLImageElement).src = '/placeholder.png';
            }}
          />
        ) : (
          <Package className="w-12 h-12 text-neutral-300" />
        )}
      </div>

      {/* Info */}
      <div className="p-4">
        <div className="text-xs font-medium text-indigo-600 mb-1">
          {product.category?.toUpperCase()}
        </div>
        <h3 className="font-bold text-neutral-900 line-clamp-2" title={product.title}>
          {product.title}
        </h3>

        {/* Rating */}
        <div className="flex items-center gap-1 mt-2">
          <Star className="w-4 h-4 fill-amber-400 text-amber-400" />
          <span className="text-sm font-medium text-neutral-700">
            {product.rating?.toFixed(1)}
          </span>
        </div>

        {/* Price */}
        <p className="text-lg font-bold text-neutral-900 mt-3">
          ${product.price?.toFixed(2)}
        </p>
      </div>
    </div>
  );
}
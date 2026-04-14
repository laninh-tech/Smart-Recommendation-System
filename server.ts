/**
 * Optional dev server: serves Vite app and proxies /api to SmartRec FastAPI backend.
 * Run with: npm run dev:server
 * Default: backend at http://localhost:8000, app at http://localhost:3001
 */
import express from 'express';
import { createServer as createViteServer } from 'vite';

const app = express();
const PORT = 3001;
const BACKEND_URL = 'http://localhost:8000';

console.log('SmartRec Node.js Proxy Server');
console.log('Forwarding /api to:', BACKEND_URL);
console.log('='.repeat(60));

async function startServer() {
  app.use(express.json());

  // --- API Routes (Proxy to Python FastAPI) ---
  
  app.get('/api/users', async (req, res) => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/users`);
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching users:', error);
      res.status(500).json({ error: 'Failed to fetch users' });
    }
  });

  app.get('/api/recommendations/:userId', async (req, res) => {
    try {
      const userId = req.params.userId;
      const query = new URLSearchParams(req.query as Record<string, string>).toString();
      const url = `${BACKEND_URL}/api/recommendations/${userId}${query ? `?${query}` : ''}`;
      const response = await fetch(url);
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      res.status(500).json({ error: 'Failed to fetch recommendations' });
    }
  });

  app.get('/api/metrics', async (req, res) => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/metrics`);
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
      res.status(500).json({ error: 'Failed to fetch metrics' });
    }
  });

  app.get('/api/clickstream', async (req, res) => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/clickstream`);
      const data = await response.json();
      res.json(data);
    } catch (error) {
      console.error('Error fetching clickstream:', error);
      res.status(500).json({ error: 'Failed to fetch clickstream' });
    }
  });

  // --- Vite Middleware ---
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static('dist'));
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n✓ Node.js proxy server running on http://localhost:${PORT}`);
    console.log(`✓ Backend: ${BACKEND_URL}`);
    console.log(`✓ Ready to serve requests!\n`);
  });
}

startServer();

# Smart Recommendation System

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)
![React 19](https://img.shields.io/badge/React-19-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end machine learning recommendation system with three collaborative filtering models (Baseline heuristic, Matrix Factorization, Neural Collaborative Filtering), interactive web UI, and production-ready architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Models](#models)
- [Results](#results)
- [API Reference](#api-reference)
- [Documentation](#documentation)

## 🎯 Overview

This project implements a complete recommendation system pipeline:
- **Dataset**: MovieLens 100K (943 users, 1,682 items, 100K interactions)
- **Models**: 3 algorithms compared (Baseline, MF, NCF)
- **Best Performance**: NCF with **RMSE 1.2117** (42.9% better than MF)
- **Architecture**: Full-stack web application with React frontend, FastAPI backend, Node proxy
- **Reproducibility**: Deterministic scoring, no randomness, temporal data split

## ✨ Features

### Backend
- ✅ **Model Selection**: Query parameter `?model=ncf|mf|baseline` for runtime model choice
- ✅ **Deterministic Scoring**: No random components, fully reproducible results
- ✅ **Real Latency Measurement**: Accurate millisecond-level inference timing
- ✅ **Cold-Start Handling**: Popular items fallback for new users
- ✅ **Lazy Model Loading**: Load models on-demand, save memory
- ✅ **REST API**: Clean, documented endpoints for all features

### Frontend
- ✅ **User Selection Dashboard**: Choose from 943 users with search
- ✅ **Model Comparison**: Toggle between 3 models, compare side-by-side
- ✅ **Recommendations View**: 12 products with scores, ratings, images
- ✅ **Metrics Dashboard**: RMSE, Precision@K, Recall@K comparison
- ✅ **Real-time Performance**: Latency display per request

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python 3.12, FastAPI, PyTorch 2.10, scikit-learn |
| **Frontend** | React 19, TypeScript, Vite, Node.js |
| **API Gateway** | Node.js Express (proxy) |
| **Database** | JSON/CSV (MovieLens 100K) |
| **Infrastructure** | Docker, Docker Compose |
| **ML Training** | Pandas, NumPy, SGD/Adam optimizers |

## 📁 Project Structure

```
.
├── backend/                   # Python FastAPI backend
│   ├── api/
│   │   ├── main.py           # FastAPI app entry point
│   │   ├── endpoints.py       # API route handlers
│   │   └── models/            # ML model implementations
│   ├── data/                  # MovieLens 100K data
│   │   ├── interactions.csv   # User-item interactions
│   │   ├── users.json         # User metadata
│   │   └── products.json      # Product metadata
│   ├── checkpoints/           # Trained model artifacts (MF, NCF)
│   ├── requirements.txt       # Python dependencies
│   └── tests/                 # Unit tests
│
├── src/                       # React frontend (TypeScript)
│   ├── components/            # React components
│   ├── pages/                 # Page components
│   ├── App.tsx                # Root component
│   └── main.tsx               # Entry point
│
├── docs/                      # Documentation & reports
│   ├── ARCHITECTURE.md        # System architecture details
│   ├── Baocao_v3.tex         # Full LaTeX report
│   ├── Baocao_v3.pdf         # Compiled PDF report
│   └── images/                # Screenshots, diagrams
│
├── docker-compose.yml         # Container orchestration
├── Dockerfile                 # Backend container
├── package.json               # Node.js dependencies
├── tsconfig.json             # TypeScript config
├── vite.config.ts            # Vite bundler config
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+ LTS
- Git
- RAM: 2GB minimum, 4GB recommended

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Smart-Recommendation-System.git
cd Smart-Recommendation-System

# Create Python virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r backend/requirements.txt
npm install
```

### Running the Application

Open 3 terminals and run each command:

**Terminal 1 - Backend (Port 8000)**
```bash
cd backend
python api/main.py
# Output: INFO: Uvicorn running on 0.0.0.0:8000
```

**Terminal 2 - Proxy (Port 3001)**
```bash
npm run dev:server
# Output: Node.js proxy running on http://localhost:3001
```

**Terminal 3 - Frontend (Port 5173)**
```bash
npm run dev
# Output: Local: http://localhost:5173
```

**Access the Application**
Open browser: http://localhost:5173

## 🏗️ Architecture

### 4-Layer Architecture

```
┌─────────────────────────────────┐
│ Layer 1: Presentation (React)   │ Port 5173
│ - User selection, model toggle  │
│ - Recommendations display       │
│ - Metrics dashboard             │
└────────────┬────────────────────┘
						 │
┌────────────▼────────────────────┐
│ Layer 2: API Gateway (Express)  │ Port 3001
│ - Request routing               │
│ - CORS handling                 │
│ - Load balancing                │
└────────────┬────────────────────┘
						 │
┌────────────▼────────────────────┐
│ Layer 3: Backend API (FastAPI)  │ Port 8000
│ - /recommendations endpoint     │
│ - /metrics endpoint             │
│ - /users, /products endpoints   │
│ - Inference engines             │
└────────────┬────────────────────┘
						 │
┌────────────▼────────────────────┐
│ Layer 4: Data & Models          │
│ - MovieLens 100K dataset        │
│ - Trained models (MF, NCF)      │
│ - User/product metadata         │
└─────────────────────────────────┘
```

## 🤖 Models

### 1. Baseline Heuristic
- **Type**: Rule-based
- **Formula**: `score = 0.40×popularity + 0.35×rating + 0.25×preference`
- **Advantages**: Fast O(n), good for cold-start
- **Disadvantages**: No user-item personalization

### 2. Matrix Factorization (MF)
- **Type**: Collaborative Filtering
- **RMSE**: 2.1220
- **Architecture**: 943×50 user embeddings, 1682×50 item embeddings
- **Optimizer**: SGD, Learning rate: 0.005, Epochs: 50
- **Loss**: `L2(factors) + λ(regularization)`

### 3. Neural Collaborative Filtering (NCF) ⭐ Best
- **Type**: Deep Learning
- **RMSE**: 1.2117 (42.9% improvement vs MF)
- **Architecture**: 
	- Embeddings: 64-dim (user + item)
	- MLP: Dense(128)→ReLU→Dense(64)→ReLU→Dense(32)→ReLU→Output
- **Optimizer**: Adam, Learning rate: 0.001, Epochs: 50
- **Loss**: Binary Cross-Entropy
- **Advantages**: Captures non-linear relationships

## 📊 Results

### Model Comparison
| Metric | NCF | MF | Baseline |
|--------|-----|-----|----------|
| RMSE | **1.2117** | 2.1220 | — |
| Improvement | +42.9% | baseline | reference |
| Inference Time | 15-25ms | 12-20ms | <10ms |
| Memory | ~150MB | ~80MB | <10MB |

### Dataset Split
- **Training**: 80,000 interactions (80%)
- **Validation**: 10,000 interactions (10%)
- **Test**: 10,000 interactions (10%)
- **Temporal Split**: Chronological (no data leakage)

## 🔌 API Reference

### Get Recommendations
```bash
GET /api/recommendations/{user_id}?model=ncf&top_k=12
```

**Example:**
```bash
curl "http://localhost:3001/api/recommendations/u259?model=ncf&top_k=5"
```

**Response:**
```json
{
	"user_id": "u259",
	"model": "NCF",
	"recommendations": [
		{"id": "p1467", "scores": {"final": 0.9432}},
		{"id": "p119", "scores": {"final": 0.7582}}
	],
	"latency_ms": 19
}
```

### Get Metrics
```bash
GET /api/metrics
```

**Response:**
```json
{
	"ncf": {"rmse": 1.2117, "precision_10": 0.45},
	"mf": {"rmse": 2.1220, "precision_10": 0.38},
	"baseline": {"rmse": null}
}
```

### Get Users
```bash
GET /api/users?limit=10&offset=0
```

### Get Products
```bash
GET /api/products?limit=10&offset=0
```

## 📚 Documentation

- **Full Report**: [docs/Baocao_v3.pdf](docs/Baocao_v3.pdf) - Complete technical report (English/Vietnamese)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Detailed architecture design
- **Setup Guide**: [SETUP.md](SETUP.md) - Installation and deployment
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines

## 🔧 Configuration

### Environment Variables
Create `.env` file (copy from `.env.example`):
```
PYTHON_VERSION=3.12
NODE_ENV=development
BACKEND_PORT=8000
PROXY_PORT=3001
FRONTEND_PORT=5173
```

### Model Configuration
See `backend/config.py` for:
- Latent factor dimensions
- Learning rates
- Regularization parameters
- Batch sizes

## 📈 Performance

### Benchmarks (Local Development)
- **Inference Latency**: 15-25ms (NCF), 12-20ms (MF)
- **Model Loading**: <100ms (lazy loading)
- **Memory Usage**: ~150MB backend + models
- **Requests/Second**: ~10 (single-threaded development server)

### Production Scaling
- **Docker**: Containerized backend ready for deployment
- **Docker Compose**: Multi-container orchestration
- **Kubernetes Ready**: Horizontal scaling support

## 🐛 Known Limitations

- **Data**: MovieLens 100K is old (1990s), doesn't reflect modern patterns
- **Scale**: Developed for local testing, needs optimization for millions of users
- **Features**: Only uses user-item interactions, not content-based features
- **Cold-Start**: Basic popular-items fallback (could use content-based)

## 🚀 Future Improvements

- [ ] Implement hybrid recommendation (collaborative + content-based)
- [ ] Add vector database for semantic search (Milvus, Weaviate)
- [ ] Online learning for real-time model updates
- [ ] A/B testing framework
- [ ] Explainability layer (understand why items recommended)
- [ ] Kubernetes deployment configs
- [ ] Performance monitoring and alerts
- [ ] GraphQL API option
- [ ] Mobile app support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📞 Support

For issues, questions, or suggestions:
1. Open an [Issue](https://github.com/yourusername/Smart-Recommendation-System/issues)
2. Check [documentation](docs/)
3. Review technical [report](docs/Baocao_v3.pdf)

## 🙏 Acknowledgments

- **Dataset**: MovieLens 100K by GroupLens Research
- **Frameworks**: FastAPI, React, PyTorch, scikit-learn
- **Inspiration**: Collaborative filtering research papers

---

**Last Updated**: April 2026  
**Status**: Production-Ready Demo
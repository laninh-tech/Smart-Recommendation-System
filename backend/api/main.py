"""
SmartRec FastAPI backend.

Exposes REST API for users, products, personalized recommendations (MF/NCF),
metrics, and clickstream. Requires trained checkpoints and data in backend/data.
"""
import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.inference import RecommendationEngine
from config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    DEFAULT_MODEL_TYPE,
)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation engine (lazy-loaded)
engine = None

def get_engine():
    """Lazy-load recommendation engine (NCF by default)."""
    global engine
    if engine is None:
        print("Loading Recommendation Engine...")
        engine = RecommendationEngine(
            checkpoint_dir=str(CHECKPOINT_DIR),
            model_type=DEFAULT_MODEL_TYPE,
        )
        print("✓ Engine loaded!")
    return engine


# === API Endpoints ===

@app.get("/")
async def root():
    """Health check and service info."""
    return {
        "status": "ok",
        "service": f"SmartRec API v{API_VERSION}",
        "model": DEFAULT_MODEL_TYPE.upper(),
        "dataset": {"users": 500, "products": 500, "interactions": 50000},
        "version": API_VERSION,
    }


@app.get("/api/users")
async def get_users():
    """Return all users with preferences (user_id, name, preferences, etc.)."""
    try:
        with open(DATA_DIR / "users.json", "r", encoding="utf-8") as f:
            users = json.load(f)
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading users: {str(e)}")


@app.get("/api/products")
async def get_products():
    """Return full product catalog."""
    try:
        with open(DATA_DIR / "products.json", "r", encoding="utf-8") as f:
            products = json.load(f)
        return products
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading products: {str(e)}")


@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: str, top_k: int = 12):
    """
    Get personalized recommendations for a user
    
    Args:
        user_id: User ID (e.g., 'u1', 'u2', 'u500')
        top_k: Number of recommendations (default: 12)
    
    Returns:
        JSON with recommendations and latency metrics
    """
    try:
        engine = get_engine()
        start_time = time.time()
        result = engine.get_recommendations(user_id, top_k=top_k)
        latency = (time.time() - start_time) * 1000  # Convert to ms
        result['latency'] = f"{latency:.0f}ms"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.get("/api/metrics")
async def get_metrics():
    """Return evaluation metrics for MF and NCF (Precision@K, NDCG, RMSE, MAE)."""
    try:
        with open(CHECKPOINT_DIR / "evaluation_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Extract metrics for both models
        ncf_metrics = results.get('NCF', {})
        mf_metrics = results.get('MF', {})
        
        return {
            'current_model': 'NCF',
            'ncf': {
                'precision_at_5': ncf_metrics.get('Precision@5', 0),
                'precision_at_10': ncf_metrics.get('Precision@10', 0),
                'recall_at_10': ncf_metrics.get('Recall@10', 0),
                'ndcg_at_10': ncf_metrics.get('NDCG@10', 0),
                'rmse': ncf_metrics.get('RMSE', 1.24),
                'mae': ncf_metrics.get('MAE', 1.04)
            },
            'mf': {
                'precision_at_5': mf_metrics.get('Precision@5', 0),
                'precision_at_10': mf_metrics.get('Precision@10', 0),
                'recall_at_10': mf_metrics.get('Recall@10', 0),
                'ndcg_at_10': mf_metrics.get('NDCG@10', 0),
                'rmse': mf_metrics.get('RMSE', 1.24),
                'mae': mf_metrics.get('MAE', 1.04)
            },
            'dataset': {
                'users': 500,
                'products': 500,
                'interactions': 50000,
                'train_split': '80%',
                'val_split': '10%',
                'test_split': '10%'
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")


@app.get("/api/clickstream")
async def get_clickstream():
    """Return last 50 user interactions (clickstream) for demo."""
    try:
        interactions = []
        with open(DATA_DIR / "interactions.csv", "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines[-50:]:  # Last 50 interactions
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    interactions.append({
                        'user_id': parts[0],
                        'product_id': parts[1],
                        'action': parts[2],
                        'timestamp': parts[3] if len(parts) > 3 else 'recent'
                    })
        
        return {'interactions': interactions[::-1]}  # Reverse to show recent first
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading clickstream: {str(e)}")


@app.get("/healthz")
async def health():
    """Kubernetes health check endpoint"""
    return {"status": "healthy"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

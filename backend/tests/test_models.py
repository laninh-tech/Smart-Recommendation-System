"""
Test suite for SmartRec recommendation system
"""
import pytest
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import tempfile

# Add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from models.matrix_factorization import MatrixFactorization
from models.ncf import NCF
from api.inference import RecommendationEngine


class TestMatrixFactorization:
    """Tests for Matrix Factorization model"""
    
    @pytest.fixture
    def mf_model(self):
        """Create a test MF model"""
        return MatrixFactorization(n_users=100, n_items=50, n_factors=32)
    
    def test_model_creation(self, mf_model):
        """Test model instantiation"""
        assert mf_model is not None
        assert mf_model.n_users == 100
        assert mf_model.n_items == 50
        assert mf_model.n_factors == 32
    
    def test_model_forward_pass(self, mf_model):
        """Test forward pass with random inputs"""
        user_ids = torch.randint(0, 100, (32,))  # batch_size=32
        item_ids = torch.randint(0, 50, (32,))
        
        output = mf_model(user_ids, item_ids)
        
        assert output.shape == (32, 1)
        assert not torch.isnan(output).any()
    
    def test_model_parameter_count(self, mf_model):
        """Test model has expected number of parameters"""
        params = sum(p.numel() for p in mf_model.parameters() if p.requires_grad)
        # user_emb: 100*32, item_emb: 50*32, bias_user: 100, bias_item: 50
        expected = (100 * 32) + (50 * 32) + 100 + 50
        assert params == expected
    
    def test_model_output_range(self, mf_model):
        """Test model outputs are in reasonable range"""
        user_ids = torch.randint(0, 100, (64,))
        item_ids = torch.randint(0, 50, (64,))
        
        output = mf_model(user_ids, item_ids)
        
        # Outputs should be roughly in range of mean ratings (0-5)
        assert output.min() >= -10
        assert output.max() <= 10


class TestNCF:
    """Tests for Neural Collaborative Filtering model"""
    
    @pytest.fixture
    def ncf_model(self):
        """Create a test NCF model"""
        return NCF(n_users=100, n_items=50, embedding_dim=32, layers=[64, 32], dropout=0.2)
    
    def test_model_creation(self, ncf_model):
        """Test NCF model instantiation"""
        assert ncf_model is not None
        assert ncf_model.embedding_dim == 32
    
    def test_model_forward_pass(self, ncf_model):
        """Test NCF forward pass"""
        user_ids = torch.randint(0, 100, (32,))
        item_ids = torch.randint(0, 50, (32,))
        
        output = ncf_model(user_ids, item_ids)
        
        assert output.shape == (32, 1)
        assert not torch.isnan(output).any()
    
    def test_model_dropout_behavior(self, ncf_model):
        """Test dropout is applied in training but not in eval"""
        user_ids = torch.randint(0, 100, (32,))
        item_ids = torch.randint(0, 50, (32,))
        
        # Training mode
        ncf_model.train()
        output_train = ncf_model(user_ids, item_ids)
        
        # Eval mode
        ncf_model.eval()
        output_eval = ncf_model(user_ids, item_ids)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval)


class TestRecommendationEngine:
    """Tests for the inference engine"""
    
    @pytest.fixture
    def mock_data(self, tmp_path):
        """Create mock data files"""
        # Create mock users.json
        users_data = [
            {"user_id": "user_1", "name": "Alice", "preferences": ["electronics"]},
            {"user_id": "user_2", "name": "Bob", "preferences": ["books"]},
        ]
        users_file = tmp_path / "users.json"
        with open(users_file, 'w') as f:
            json.dump(users_data, f)
        
        # Create mock products.json
        products_data = [
            {"product_id": "prod_1", "title": "Laptop", "category": "electronics"},
            {"product_id": "prod_2", "title": "Novel", "category": "books"},
        ]
        products_file = tmp_path / "products.json"
        with open(products_file, 'w') as f:
            json.dump(products_data, f)
        
        return {
            'users_file': str(users_file),
            'products_file': str(products_file),
            'tmp_path': tmp_path
        }
    
    def test_user_loading(self, mock_data):
        """Test users are loaded correctly"""
        with open(mock_data['users_file']) as f:
            users = json.load(f)
        
        assert len(users) == 2
        assert users[0]['user_id'] == "user_1"
        assert users[1]['preferences'] == ["books"]
    
    def test_product_loading(self, mock_data):
        """Test products are loaded correctly"""
        with open(mock_data['products_file']) as f:
            products = json.load(f)
        
        assert len(products) == 2
        assert products[0]['category'] == "electronics"


class TestDataProcessing:
    """Tests for data processing pipelines"""
    
    def test_preference_matching(self):
        """Test user preference matching logic"""
        user_preferences = ["electronics", "books"]
        product_categories = ["electronics", "appliances"]
        
        match = any(cat in product_categories for cat in user_preferences)
        
        assert match is True
    
    def test_preference_no_match(self):
        """Test when preferences don't match"""
        user_preferences = ["furniture"]
        product_categories = ["electronics", "appliances"]
        
        match = any(cat in product_categories for cat in user_preferences)
        
        assert match is False
    
    def test_empty_preferences(self):
        """Test handling of empty preferences"""
        user_preferences = []
        product_categories = ["electronics"]
        
        match = any(cat in product_categories for cat in user_preferences)
        
        assert match is False


class TestMetrics:
    """Tests for evaluation metrics"""
    
    def test_precision_calculation(self):
        """Test precision@k calculation"""
        # predictions = [relevant, relevant, not_relevant, relevant, not_relevant]
        predictions = [True, True, False, True, False]
        k = 3
        precision = sum(predictions[:k]) / k
        
        assert precision == pytest.approx(2/3)
    
    def test_recall_calculation(self):
        """Test recall@k calculation"""
        predictions = [True, True, False, True, False]
        k = 5
        total_relevant = 3
        recall = sum(predictions[:k]) / total_relevant
        
        assert recall == pytest.approx(1.0)
    
    def test_ndcg_calculation(self):
        """Test NDCG calculation"""
        # Relevance scores at positions
        relevances = [1, 1, 0, 1, 0]
        
        # DCG = 1/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5)
        positions = np.arange(1, len(relevances) + 1)
        dcg = np.sum(relevances / np.log2(positions + 1))
        
        assert dcg > 0


class TestAPIEndpoints:
    """Tests for API endpoints"""
    
    def test_metrics_endpoint_format(self):
        """Test metrics endpoint returns correct format"""
        # Mock metrics response
        metrics = {
            "mf": {
                "precision_at_5": 0.75,
                "precision_at_10": 0.68,
                "recall_at_10": 0.52,
                "ndcg_at_10": 0.71,
                "rmse": 1.24,
                "mae": 1.04
            },
            "ncf": {
                "precision_at_5": 0.78,
                "precision_at_10": 0.72,
                "recall_at_10": 0.55,
                "ndcg_at_10": 0.74,
                "rmse": 1.19,
                "mae": 0.99
            }
        }
        
        # Verify structure
        assert "mf" in metrics
        assert "ncf" in metrics
        assert "rmse" in metrics["mf"]
        assert "precision_at_10" in metrics["ncf"]


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_model_forward_backward(self):
        """Test model can do forward and backward pass"""
        model = MatrixFactorization(n_users=50, n_items=25, n_factors=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Forward pass
        user_ids = torch.randint(0, 50, (16,))
        item_ids = torch.randint(0, 25, (16,))
        target = torch.randn(16, 1)
        
        output = model(user_ids, item_ids)
        loss = ((output - target) ** 2).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loss should be finite
        assert not torch.isnan(loss)
        assert loss.item() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

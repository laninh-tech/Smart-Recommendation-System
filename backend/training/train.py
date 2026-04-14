"""
Training Script for Recommendation Models
"""
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np
from tqdm import tqdm
import time

from models.matrix_factorization import MatrixFactorization, MatrixFactorizationTrainer
from models.ncf import NCF, NeuMF
from data.dataset_loader import DataLoader as RecDataLoader
from training.evaluate import RecommendationEvaluator


class Trainer:
    """Universal trainer for recommendation models"""
    
    def __init__(self, model, train_loader, val_loader, 
                 learning_rate=0.001, weight_decay=1e-5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            user_ids = batch['user']
            product_ids = batch['product']
            ratings = batch['rating']
            
            # Forward pass
            predictions = self.model(user_ids, product_ids)
            loss = self.criterion(predictions, ratings)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                user_ids = batch['user']
                product_ids = batch['product']
                ratings = batch['rating']
                
                predictions = self.model(user_ids, product_ids)
                loss = self.criterion(predictions, ratings)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, epochs=50, early_stopping_patience=5):
        """Train model with early stopping"""
        print(f"\nTraining {self.model.__class__.__name__}...")
        print(f"Epochs: {epochs}, Patience: {early_stopping_patience}")
        print("=" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        self.model.load_state_dict(self.best_model_state)
        print(f"\nBest validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': epoch + 1
        }


def main():
    """Main training pipeline"""
    print("="*60)
    print("SmartRec Model Training Pipeline")
    print("="*60)
    
    # Configuration
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EPOCHS = 50
    PATIENCE = 5
    
    # Create checkpoint directory
    checkpoint_dir = Path(backend_dir) / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n1. Loading dataset...")
    data_loader = RecDataLoader(data_dir=str(Path(backend_dir) / 'data'))
    data = data_loader.prepare_datasets()
    
    print(f"   Users: {data['n_users']}")
    print(f"   Products: {data['n_products']}")
    print(f"   Train samples: {len(data['train'])}")
    print(f"   Val samples: {len(data['val'])}")
    print(f"   Test samples: {len(data['test'])}")
    
    # Create data loaders
    train_loader = DataLoader(data['train'], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(data['val'], batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(data['test'], batch_size=BATCH_SIZE, shuffle=False)
    
    results = {}

    # Build history for fair ranking evaluation (exclude train/val seen items only)
    seen_items_by_user = {}
    for row in data['train'].interactions[['user_id', 'product_id']].itertuples(index=False):
        uid = row.user_id
        pid = row.product_id
        if pid in data['product_to_idx']:
            seen_items_by_user.setdefault(uid, set()).add(data['product_to_idx'][pid])
    for row in data['val'].interactions[['user_id', 'product_id']].itertuples(index=False):
        uid = row.user_id
        pid = row.product_id
        if pid in data['product_to_idx']:
            seen_items_by_user.setdefault(uid, set()).add(data['product_to_idx'][pid])
    seen_items_by_user = {uid: sorted(list(items)) for uid, items in seen_items_by_user.items()}
    
    # ========================
    # Train Matrix Factorization
    # ========================
    print("\n" + "="*60)
    print("2. Training Matrix Factorization")
    print("="*60)
    
    mf_model = MatrixFactorization(
        n_users=data['n_users'],
        n_items=data['n_products'],
        n_factors=50
    )
    
    mf_trainer = Trainer(
        mf_model, train_loader, val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    mf_history = mf_trainer.train(epochs=EPOCHS, early_stopping_patience=PATIENCE)
    
    # Save model
    torch.save({
        'model_state_dict': mf_model.state_dict(),
        'n_users': data['n_users'],
        'n_products': data['n_products'],
        'n_factors': 50,
        'history': mf_history
    }, checkpoint_dir / 'mf_model.pth')
    print(f"✓ Saved to {checkpoint_dir / 'mf_model.pth'}")
    
    # Evaluate MF
    print("\nEvaluating Matrix Factorization...")
    mf_evaluator = RecommendationEvaluator(
        mf_model, data['test'],
        data['user_to_idx'], data['product_to_idx'],
        data['idx_to_user'], data['idx_to_product'],
        seen_items_by_user=seen_items_by_user,
    )
    mf_metrics = mf_evaluator.evaluate_all(k_values=[5, 10, 20])
    results['MF'] = mf_metrics
    
    # ========================
    # Train NCF
    # ========================
    print("\n" + "="*60)
    print("3. Training Neural Collaborative Filtering (NCF)")
    print("="*60)
    
    ncf_model = NCF(
        n_users=data['n_users'],
        n_items=data['n_products'],
        embedding_dim=64,
        layers=[128, 64, 32],
        dropout=0.2
    )
    
    ncf_trainer = Trainer(
        ncf_model, train_loader, val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    ncf_history = ncf_trainer.train(epochs=EPOCHS, early_stopping_patience=PATIENCE)
    
    # Save model
    torch.save({
        'model_state_dict': ncf_model.state_dict(),
        'n_users': data['n_users'],
        'n_products': data['n_products'],
        'embedding_dim': 64,
        'layers': [128, 64, 32],
        'history': ncf_history
    }, checkpoint_dir / 'ncf_model.pth')
    print(f"✓ Saved to {checkpoint_dir / 'ncf_model.pth'}")
    
    # Evaluate NCF
    print("\nEvaluating NCF...")
    ncf_evaluator = RecommendationEvaluator(
        ncf_model, data['test'],
        data['user_to_idx'], data['product_to_idx'],
        data['idx_to_user'], data['idx_to_product'],
        seen_items_by_user=seen_items_by_user,
    )
    ncf_metrics = ncf_evaluator.evaluate_all(k_values=[5, 10, 20])
    results['NCF'] = ncf_metrics
    
    # ========================
    # Save metadata and results
    # ========================
    metadata = {
        'n_users': data['n_users'],
        'n_products': data['n_products'],
        'user_to_idx': data['user_to_idx'],
        'product_to_idx': data['product_to_idx'],
        'idx_to_user': data['idx_to_user'],
        'idx_to_product': data['idx_to_product'],
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'epochs': EPOCHS,
            'patience': PATIENCE
        }
    }
    
    with open(checkpoint_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(checkpoint_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    # ========================
    # Final Summary
    # ========================
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    print("\nModel Comparison:")
    print("\n  Matrix Factorization:")
    print(f"    Precision@10: {results['MF']['Precision@10']:.4f}")
    print(f"    Recall@10:    {results['MF']['Recall@10']:.4f}")
    print(f"    NDCG@10:      {results['MF']['NDCG@10']:.4f}")
    print(f"    RMSE:         {results['MF']['RMSE']:.4f}")
    
    print("\n  Neural Collaborative Filtering:")
    print(f"    Precision@10: {results['NCF']['Precision@10']:.4f}")
    print(f"    Recall@10:    {results['NCF']['Recall@10']:.4f}")
    print(f"    NDCG@10:      {results['NCF']['NDCG@10']:.4f}")
    print(f"    RMSE:         {results['NCF']['RMSE']:.4f}")
    
    print(f"\n✓ All models saved to: {checkpoint_dir}")
    print(f"✓ Evaluation results saved to: {checkpoint_dir / 'evaluation_results.json'}")


if __name__ == '__main__':
    main()

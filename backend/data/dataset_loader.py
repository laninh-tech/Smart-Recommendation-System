"""
Dataset Loader - Load và preprocess data cho training
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

class RecommendationDataset(Dataset):
    """PyTorch Dataset cho recommendation"""
    
    def __init__(self, interactions_df, user_map, product_map):
        self.interactions = interactions_df
        self.user_map = user_map
        self.product_map = product_map
        
        # Convert to indices
        self.user_indices = interactions_df['user_id'].map(user_map).values
        self.product_indices = interactions_df['product_id'].map(product_map).values
        self.ratings = interactions_df['rating_score'].values
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.user_indices[idx], dtype=torch.long),
            'product': torch.tensor(self.product_indices[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float32)
        }


class DataLoader:
    """Load and preprocess data"""
    
    def __init__(self, data_dir='./data'):
        self.data_dir = Path(data_dir)
        
    def load_data(self):
        """Load all data files"""
        print("Loading data...")
        
        # Load users
        with open(self.data_dir / 'users.json', 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        # Load products
        with open(self.data_dir / 'products.json', 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        # Load interactions
        interactions = pd.read_csv(self.data_dir / 'interactions.csv')
        
        print(f"✓ Loaded {len(users)} users")
        print(f"✓ Loaded {len(products)} products")
        print(f"✓ Loaded {len(interactions)} interactions")
        
        return users, products, interactions
    
    def preprocess_interactions(self, interactions):
        """
        Preprocess interactions for training
        
        Convert actions to rating scores:
        - view: 1
        - add_to_cart: 3
        - purchase: 5 (hoặc actual rating nếu có)
        """
        df = interactions.copy()
        
        # Convert action to score
        def action_to_score(row):
            if row['action'] == 'purchase' and pd.notna(row['rating']):
                return row['rating']
            elif row['action'] == 'purchase':
                return 5
            elif row['action'] == 'add_to_cart':
                return 3
            else:  # view
                return 1
        
        df['rating_score'] = df.apply(action_to_score, axis=1)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    def create_mappings(self, interactions):
        """Create user and product ID mappings"""
        unique_users = interactions['user_id'].unique()
        unique_products = interactions['product_id'].unique()
        
        user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        product_to_idx = {pid: idx for idx, pid in enumerate(unique_products)}
        
        idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
        idx_to_product = {idx: pid for pid, idx in product_to_idx.items()}
        
        return user_to_idx, product_to_idx, idx_to_user, idx_to_product
    
    def split_data(self, interactions, test_size=0.1, val_size=0.1):
        """
        Split data into train/val/test sets
        
        Sử dụng temporal split: 
        - Train: 80% interactions đầu
        - Val: 10% tiếp theo
        - Test: 10% cuối (most recent)
        """
        # Sort by timestamp
        df = interactions.sort_values('timestamp')
        
        n = len(df)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        print(f"\nData split:")
        print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        print(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
        print(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def prepare_datasets(self):
        """Prepare complete datasets for training"""
        # Load data
        users, products, interactions = self.load_data()
        
        # Preprocess
        interactions = self.preprocess_interactions(interactions)
        
        # Create mappings
        user_to_idx, product_to_idx, idx_to_user, idx_to_product = \
            self.create_mappings(interactions)
        
        # Split data
        train_df, val_df, test_df = self.split_data(interactions)
        
        # Create PyTorch datasets
        train_dataset = RecommendationDataset(train_df, user_to_idx, product_to_idx)
        val_dataset = RecommendationDataset(val_df, user_to_idx, product_to_idx)
        test_dataset = RecommendationDataset(test_df, user_to_idx, product_to_idx)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'n_users': len(user_to_idx),
            'n_products': len(product_to_idx),
            'user_to_idx': user_to_idx,
            'product_to_idx': product_to_idx,
            'idx_to_user': idx_to_user,
            'idx_to_product': idx_to_product,
            'users': users,
            'products': products
        }


if __name__ == '__main__':
    loader = DataLoader()
    data = loader.prepare_datasets()
    
    print(f"\n✓ Datasets ready!")
    print(f"  Users: {data['n_users']}")
    print(f"  Products: {data['n_products']}")
    print(f"  Train samples: {len(data['train'])}")
    print(f"  Val samples: {len(data['val'])}")
    print(f"  Test samples: {len(data['test'])}")

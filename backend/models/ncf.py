"""
Neural Collaborative Filtering (NCF) Model
Deep learning approach to collaborative filtering
"""
import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Neural Collaborative Filtering
    
    Architecture:
    1. User & Item Embeddings
    2. Concatenate embeddings
    3. Feed through MLP layers
    4. Output: Predicted rating
    
    Paper: "Neural Collaborative Filtering" (WWW 2017)
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, 
                 layers=[128, 64, 32], dropout=0.2):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of embeddings
            layers: List of hidden layer sizes
            dropout: Dropout rate
        """
        super(NCF, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_size = embedding_dim * 2
        
        for hidden_size in layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.BatchNorm1d(hidden_size))
            mlp_layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        
        # Output layer
        mlp_layers.append(nn.Linear(input_size, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass
        
        Args:
            user_ids: [batch_size]
            item_ids: [batch_size]
        
        Returns:
            predictions: [batch_size]
        """
        # Get embeddings
        user_embed = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        item_embed = self.item_embedding(item_ids)  # [batch_size, embedding_dim]
        
        # Concatenate
        concat = torch.cat([user_embed, item_embed], dim=1)  # [batch_size, embedding_dim*2]
        
        # Pass through MLP
        output = self.mlp(concat).squeeze()  # [batch_size]
        
        return output
    
    def predict(self, user_ids, item_ids):
        """Predict ratings"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(user_ids, item_ids)
        return predictions
    
    def recommend(self, user_id, top_k=10, exclude_items=None):
        """
        Generate top-K recommendations
        
        Args:
            user_id: User index
            top_k: Number of recommendations
            exclude_items: Items to exclude
        
        Returns:
            top_items: Recommended item indices
            scores: Prediction scores
        """
        self.eval()
        with torch.no_grad():
            # Prepare tensors
            user_tensor = torch.tensor([user_id] * self.n_items)
            item_tensor = torch.arange(self.n_items)
            
            # Predict scores
            scores = self.forward(user_tensor, item_tensor)
            
            # Exclude items
            if exclude_items is not None:
                scores[exclude_items] = float('-inf')
            
            # Get top-K
            top_scores, top_items = torch.topk(scores, min(top_k, len(scores)))
        
        return top_items.numpy(), top_scores.numpy()


class GMF(nn.Module):
    """
    Generalized Matrix Factorization (GMF)
    Component của NeuMF (Neural Matrix Factorization)
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64):
        super(GMF, self).__init__()
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.output_layer.weight)
    
    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        
        # Element-wise product
        element_product = user_embed * item_embed
        output = self.output_layer(element_product).squeeze()
        
        return output


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF)
    Combines GMF and MLP
    
    Paper: "Neural Collaborative Filtering" (WWW 2017)
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, 
                 mlp_layers=[128, 64, 32], dropout=0.2):
        super(NeuMF, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        
        # GMF part
        self.gmf_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP part
        self.mlp_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp = []
        input_size = embedding_dim * 2
        for hidden_size in mlp_layers:
            mlp.append(nn.Linear(input_size, hidden_size))
            mlp.append(nn.ReLU())
            mlp.append(nn.BatchNorm1d(hidden_size))
            mlp.append(nn.Dropout(dropout))
            input_size = hidden_size
        
        self.mlp = nn.Sequential(*mlp)
        
        # Final prediction layer
        self.output_layer = nn.Linear(embedding_dim + mlp_layers[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
    
    def forward(self, user_ids, item_ids):
        # GMF part
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item
        
        # MLP part
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_concat = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp(mlp_concat)
        
        # Concatenate GMF and MLP
        concat = torch.cat([gmf_output, mlp_output], dim=1)
        
        # Final prediction
        prediction = self.output_layer(concat).squeeze()
        
        return prediction
    
    def recommend(self, user_id, top_k=10, exclude_items=None):
        """Generate recommendations"""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id] * self.n_items)
            item_tensor = torch.arange(self.n_items)
            
            scores = self.forward(user_tensor, item_tensor)
            
            if exclude_items is not None:
                scores[exclude_items] = float('-inf')
            
            top_scores, top_items = torch.topk(scores, min(top_k, len(scores)))
        
        return top_items.numpy(), top_scores.numpy()


if __name__ == '__main__':
    print("Testing NCF Models...")
    
    n_users = 100
    n_items = 50
    
    # Test NCF
    print("\n1. Testing NCF...")
    ncf = NCF(n_users, n_items, embedding_dim=64, layers=[128, 64, 32])
    user_ids = torch.randint(0, n_users, (8,))
    item_ids = torch.randint(0, n_items, (8,))
    
    predictions = ncf(user_ids, item_ids)
    print(f"✓ NCF forward pass: {predictions.shape}")
    print(f"  Parameters: {sum(p.numel() for p in ncf.parameters()):,}")
    
    # Test recommendations
    top_items, scores = ncf.recommend(user_id=0, top_k=10)
    print(f"✓ Top 5 recommendations: {top_items[:5]}")
    
    # Test NeuMF
    print("\n2. Testing NeuMF...")
    neumf = NeuMF(n_users, n_items, embedding_dim=64, mlp_layers=[128, 64, 32])
    predictions = neumf(user_ids, item_ids)
    print(f"✓ NeuMF forward pass: {predictions.shape}")
    print(f"  Parameters: {sum(p.numel() for p in neumf.parameters()):,}")

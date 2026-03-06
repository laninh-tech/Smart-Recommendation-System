"""
Matrix Factorization Model
Collaborative Filtering sử dụng matrix factorization
"""
import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """
    Matrix Factorization Model
    
    Decompose user-item interaction matrix thành:
    R ≈ U × V^T
    
    Trong đó:
    - R: interaction matrix (n_users × n_items)
    - U: user embedding matrix (n_users × n_factors)
    - V: item embedding matrix (n_items × n_factors)
    
    Prediction: r_ui = u_u · v_i (dot product)
    """
    
    def __init__(self, n_users, n_items, n_factors=50, dropout=0.02):
        """
        Args:
            n_users: Số lượng users
            n_items: Số lượng items/products
            n_factors: Số dimensions của embeddings
            dropout: Dropout rate để prevent overfitting
        """
        super(MatrixFactorization, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        # User embeddings
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        
        # Item embeddings
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        
        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with normal distribution"""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass
        
        Args:
            user_ids: Tensor of user indices [batch_size]
            item_ids: Tensor of item indices [batch_size]
        
        Returns:
            predictions: Predicted ratings [batch_size]
        """
        # Get embeddings
        user_embeds = self.user_embeddings(user_ids)  # [batch_size, n_factors]
        item_embeds = self.item_embeddings(item_ids)  # [batch_size, n_factors]
        
        # Apply dropout
        user_embeds = self.dropout(user_embeds)
        item_embeds = self.dropout(item_embeds)
        
        # Dot product
        dot_product = (user_embeds * item_embeds).sum(dim=1)  # [batch_size]
        
        # Add biases
        user_bias = self.user_bias(user_ids).squeeze()  # [batch_size]
        item_bias = self.item_bias(item_ids).squeeze()  # [batch_size]
        
        # Final prediction
        prediction = dot_product + user_bias + item_bias + self.global_bias
        
        return prediction
    
    def predict(self, user_ids, item_ids):
        """Predict ratings (inference mode)"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(user_ids, item_ids)
        return predictions
    
    def recommend(self, user_id, top_k=10, exclude_items=None):
        """
        Recommend top-K items cho user
        
        Args:
            user_id: User index
            top_k: Number of items to recommend
            exclude_items: List of item indices to exclude (already interacted)
        
        Returns:
            top_items: Recommended item indices
            scores: Prediction scores
        """
        self.eval()
        with torch.no_grad():
            # Get user embedding
            user_tensor = torch.tensor([user_id] * self.n_items)
            item_tensor = torch.arange(self.n_items)
            
            # Predict scores for all items
            scores = self.forward(user_tensor, item_tensor)
            
            # Exclude already interacted items
            if exclude_items is not None:
                scores[exclude_items] = float('-inf')
            
            # Get top-K
            top_scores, top_items = torch.topk(scores, min(top_k, len(scores)))
            
        return top_items.numpy(), top_scores.numpy()
    
    def get_user_embedding(self, user_id):
        """Get embedding vector cho user"""
        return self.user_embeddings(torch.tensor([user_id]))[0].detach().numpy()
    
    def get_item_embedding(self, item_id):
        """Get embedding vector cho item"""
        return self.item_embeddings(torch.tensor([item_id]))[0].detach().numpy()


class MatrixFactorizationTrainer:
    """Trainer cho MF model"""
    
    def __init__(self, model, lr=0.001, weight_decay=1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader, device='cpu'):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            user_ids = batch['user'].to(device)
            item_ids = batch['product'].to(device)
            ratings = batch['rating'].to(device)
            
            # Forward pass
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, device='cpu'):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                user_ids = batch['user'].to(device)
                item_ids = batch['product'].to(device)
                ratings = batch['rating'].to(device)
                
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)


if __name__ == '__main__':
    # Test model
    print("Testing Matrix Factorization Model...")
    
    n_users = 100
    n_items = 50
    n_factors = 32
    
    model = MatrixFactorization(n_users, n_items, n_factors)
    
    # Test forward pass
    user_ids = torch.randint(0, n_users, (8,))
    item_ids = torch.randint(0, n_items, (8,))
    
    predictions = model(user_ids, item_ids)
    print(f"✓ Forward pass successful: {predictions.shape}")
    
    # Test recommendations
    top_items, scores = model.recommend(user_id=0, top_k=10)
    print(f"✓ Recommendations for user 0: {top_items[:5]}")
    
    print("\nModel Summary:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  User embeddings: {model.user_embeddings.weight.shape}")
    print(f"  Item embeddings: {model.item_embeddings.weight.shape}")

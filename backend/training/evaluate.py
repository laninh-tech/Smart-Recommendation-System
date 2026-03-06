"""
Evaluation Metrics for Recommendation Systems
"""
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
import torch


def hit_rate_at_k(recommended_items, relevant_items, k):
    """
    Hit Rate@K: % users có ít nhất 1 relevant item trong top-K
    
    Args:
        recommended_items: List of recommended item lists
        relevant_items: List of relevant item lists
        k: Top-K cutoff
    
    Returns:
        hit_rate: Float between 0 and 1
    """
    hits = 0
    for rec, rel in zip(recommended_items, relevant_items):
        rec_k = set(rec[:k])
        rel_set = set(rel)
        if len(rec_k.intersection(rel_set)) > 0:
            hits += 1
    
    return hits / len(recommended_items)


def precision_at_k(recommended_items, relevant_items, k):
    """
    Precision@K: % recommended items that are relevant
    
    Precision@K = (# relevant items in top-K) / K
    
    Args:
        recommended_items: List of recommended item lists  
        relevant_items: List of relevant item lists
        k: Top-K cutoff
    
    Returns:
        precision: Average precision across all users
    """
    precisions = []
    
    for rec, rel in zip(recommended_items, relevant_items):
        rec_k = set(rec[:k])
        rel_set = set(rel)
        
        # Count relevant items in top-K
        n_relevant = len(rec_k.intersection(rel_set))
        precision = n_relevant / min(k, len(rec))
        precisions.append(precision)
    
    return np.mean(precisions)


def recall_at_k(recommended_items, relevant_items, k):
    """
    Recall@K: % relevant items that are recommended
    
    Recall@K = (# relevant items in top-K) / (total # relevant items)
    
    Args:
        recommended_items: List of recommended item lists
        relevant_items: List of relevant item lists
        k: Top-K cutoff
    
    Returns:
        recall: Average recall across all users
    """
    recalls = []
    
    for rec, rel in zip(recommended_items, relevant_items):
        if len(rel) == 0:
            continue
            
        rec_k = set(rec[:k])
        rel_set = set(rel)
        
        # Count relevant items in top-K
        n_relevant = len(rec_k.intersection(rel_set))
        recall = n_relevant / len(rel_set)
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def ndcg_at_k(recommended_items, relevant_items, k):
    """
    NDCG@K: Normalized Discounted Cumulative Gain
    
    Considers both relevance and ranking position
    
    DCG@K = Σ (rel_i / log2(i+1))
    NDCG@K = DCG@K / IDCG@K
    
    Args:
        recommended_items: List of recommended item lists
        relevant_items: List of relevant item lists  
        k: Top-K cutoff
    
    Returns:
        ndcg: Average NDCG across all users
    """
    ndcgs = []
    
    for rec, rel in zip(recommended_items, relevant_items):
        if len(rel) == 0:
            continue
        
        rec_k = rec[:k]
        rel_set = set(rel)
        
        # DCG calculation
        dcg = 0.0
        for i, item in enumerate(rec_k):
            if item in rel_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0
        
        # IDCG calculation (ideal ranking)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0


def mean_average_precision_at_k(recommended_items, relevant_items, k):
    """
    MAP@K: Mean Average Precision
    
    Average of precision values at each relevant item position
    """
    aps = []
    
    for rec, rel in zip(recommended_items, relevant_items):
        if len(rel) == 0:
            continue
        
        rec_k = rec[:k]
        rel_set = set(rel)
        
        # Calculate AP
        precisions = []
        n_relevant = 0
        
        for i, item in enumerate(rec_k):
            if item in rel_set:
                n_relevant += 1
                precisions.append(n_relevant / (i + 1))
        
        ap = np.mean(precisions) if precisions else 0.0
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def rmse(predictions, targets):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(targets, predictions))


def mae(predictions, targets):
    """Mean Absolute Error"""
    return np.mean(np.abs(predictions - targets))


class RecommendationEvaluator:
    """Comprehensive evaluator for recommendation models"""
    
    def __init__(self, model, test_data, user_to_idx, product_to_idx, 
                 idx_to_user, idx_to_product):
        self.model = model
        self.test_data = test_data
        self.user_to_idx = user_to_idx
        self.product_to_idx = product_to_idx
        self.idx_to_user = idx_to_user
        self.idx_to_product = idx_to_product
    
    def evaluate_ranking(self, k_values=[5, 10, 20]):
        """
        Evaluate ranking metrics
        
        Returns dict with Precision@K, Recall@K, NDCG@K, Hit Rate@K
        """
        print(f"\nEvaluating ranking metrics...")
        
        # Group test data by user
        test_df = self.test_data.interactions
        user_relevant_items = {}
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            rating = row['rating_score']
            
            # Consider rating >= 3 as relevant
            if rating >= 3:
                if user_id not in user_relevant_items:
                    user_relevant_items[user_id] = []
                product_idx = self.product_to_idx[product_id]
                user_relevant_items[user_id].append(product_idx)
        
        # Generate recommendations for each user
        all_recommended = []
        all_relevant = []
        
        for user_id in user_relevant_items.keys():
            user_idx = self.user_to_idx[user_id]
            
            # Get items user has already interacted with (to exclude)
            interacted = test_df[test_df['user_id'] == user_id]['product_id'].tolist()
            interacted_idx = [self.product_to_idx[pid] for pid in interacted]
            
            # Generate recommendations
            recommended, _ = self.model.recommend(
                user_idx, 
                top_k=max(k_values),
                exclude_items=interacted_idx
            )
            
            all_recommended.append(recommended.tolist())
            all_relevant.append(user_relevant_items[user_id])
        
        # Calculate metrics for each K
        results = {}
        for k in k_values:
            results[f'Precision@{k}'] = precision_at_k(all_recommended, all_relevant, k)
            results[f'Recall@{k}'] = recall_at_k(all_recommended, all_relevant, k)
            results[f'NDCG@{k}'] = ndcg_at_k(all_recommended, all_relevant, k)
            results[f'HitRate@{k}'] = hit_rate_at_k(all_recommended, all_relevant, k)
        
        return results
    
    def evaluate_rating_prediction(self):
        """Evaluate rating prediction (RMSE, MAE)"""
        print(f"\nEvaluating rating prediction...")
        
        self.model.eval()
        predictions = []
        actuals = []
        
        test_loader = torch.utils.data.DataLoader(
            self.test_data, 
            batch_size=256, 
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user']
                product_ids = batch['product']
                ratings = batch['rating']
                
                preds = self.model(user_ids, product_ids)
                
                predictions.extend(preds.numpy())
                actuals.extend(ratings.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return {
            'RMSE': rmse(predictions, actuals),
            'MAE': mae(predictions, actuals)
        }
    
    def evaluate_all(self, k_values=[5, 10, 20]):
        """Run full evaluation"""
        print("="*60)
        print("Running Full Evaluation")
        print("="*60)
        
        # Ranking metrics
        ranking_metrics = self.evaluate_ranking(k_values)
        
        # Rating prediction metrics
        prediction_metrics = self.evaluate_rating_prediction()
        
        # Combine results
        all_metrics = {**ranking_metrics, **prediction_metrics}
        
        # Print results
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        
        print("\nRanking Metrics:")
        for k in k_values:
            print(f"  @{k}:")
            print(f"    Precision:  {ranking_metrics[f'Precision@{k}']:.4f}")
            print(f"    Recall:     {ranking_metrics[f'Recall@{k}']:.4f}")
            print(f"    NDCG:       {ranking_metrics[f'NDCG@{k}']:.4f}")
            print(f"    Hit Rate:   {ranking_metrics[f'HitRate@{k}']:.4f}")
        
        print(f"\nRating Prediction:")
        print(f"  RMSE: {prediction_metrics['RMSE']:.4f}")
        print(f"  MAE:  {prediction_metrics['MAE']:.4f}")
        
        return all_metrics


if __name__ == '__main__':
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Example data
    recommended = [
        [1, 2, 3, 4, 5],
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24]
    ]
    
    relevant = [
        [1, 3, 6],
        [11, 15, 16],
        [20, 22, 25, 30]
    ]
    
    print(f"Precision@5: {precision_at_k(recommended, relevant, 5):.4f}")
    print(f"Recall@5: {recall_at_k(recommended, relevant, 5):.4f}")
    print(f"NDCG@5: {ndcg_at_k(recommended, relevant, 5):.4f}")
    print(f"Hit Rate@5: {hit_rate_at_k(recommended, relevant, 5):.4f}")

"""
MovieLens 100k Dataset Loader for SmartRec
Downloads the public MovieLens dataset and converts it into the SmartRec e-commerce format
(users.json, products.json, interactions.csv).
"""
import os
import json
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def get_movielens_genres():
    return [
        "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]

def download_and_extract(url, extract_to):
    """Download and extract zip file"""
    data_dir = Path(extract_to)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-100k.zip"
    
    if not (data_dir / "ml-100k").exists():
        print(f"Downloading {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        # Clean up zip
        os.remove(zip_path)
    return data_dir / "ml-100k"

def process_movielens(ml_dir, output_dir):
    """Process ml-100k files into SmartRec format"""
    output_dir = Path(output_dir)
    genres_list = get_movielens_genres()
    
    print("Processing Movies -> Products...")
    # Process Items (u.item)
    # Format: movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | ...
    products = []
    item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genres_list
    df_items = pd.read_csv(ml_dir / 'u.item', sep='|', names=item_cols, encoding='latin-1')
    
    for _, row in df_items.iterrows():
        # Get genres (categories) where the flag is 1
        active_genres = [g for g in genres_list if row[g] == 1]
        primary_category = active_genres[0] if active_genres else "General"
        
        products.append({
            'product_id': f"p{row['movie_id']}",
            'title': row['title'].strip(),
            'category': primary_category,
            'price': float(np.random.randint(5, 50)), # Fake price since ML doesn't have it
            'rating': 0.0, # Will be calculated from interactions
            'thumbnail': "https://dummyimage.com/200x200/4f46e5/ffffff.png&text=Movie",
            'brand': "MovieLens"
        })
    
    print("Processing Users...")
    # Process Users (u.user)
    # Format: user id | age | gender | occupation | zip code
    users = []
    df_users = pd.read_csv(ml_dir / 'u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip'])
    
    for _, row in df_users.iterrows():
        age = int(row['age'])
        if age < 25: age_group = "18-25"
        elif age < 35: age_group = "26-35"
        elif age < 45: age_group = "36-45"
        else: age_group = "46+"
            
        users.append({
            'user_id': f"u{row['user_id']}",
            'name': f"User_{row['user_id']} ({row['occupation']})",
            'preferences': [], # Will populate based on interactions
            'age_group': age_group,
            'active_level': "high"
        })
        
    print("Processing Ratings -> Interactions...")
    # Process Ratings (u.data)
    # Format: user id | item id | rating | timestamp
    df_ratings = pd.read_csv(ml_dir / 'u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    interactions = []
    user_category_counts = {} # To infer preferences
    product_ratings = {p['product_id']: [] for p in products}
    
    # Map ML ratings (1-5) to E-commerce Actions
    # Rating 1-2: view (didn't like it enough to buy, but interacted)
    # Rating 3-4: add_to_cart (showed strong interest)
    # Rating 5: purchase (loved it)
    def map_action(rating):
        if rating <= 2: return "view"
        elif rating <= 4: return "add_to_cart"
        else: return "purchase"
        
    # Get product category map
    prod_cat_map = {p['product_id']: p['category'] for p in products}

    for _, row in df_ratings.iterrows():
        uid = f"u{row['user_id']}"
        pid = f"p{row['movie_id']}"
        action = map_action(row['rating'])
        
        interactions.append({
            'user_id': uid,
            'product_id': pid,
            'action': action,
            'rating': float(row['rating']) if action == 'purchase' else None,
            'timestamp': datetime.fromtimestamp(row['timestamp']).isoformat()
        })
        
        # Track product ratings to calculate average later
        product_ratings[pid].append(row['rating'])
        
        # Track user's category interactions to infer preferences
        cat = prod_cat_map.get(pid)
        if cat:
            if uid not in user_category_counts:
                user_category_counts[uid] = {}
            user_category_counts[uid][cat] = user_category_counts[uid].get(cat, 0) + 1
            
    # Post-process User Preferences (top 2 categories they interact with)
    for user in users:
        uid = user['user_id']
        if uid in user_category_counts:
            sorted_cats = sorted(user_category_counts[uid].items(), key=lambda x: x[1], reverse=True)
            user['preferences'] = [cat for cat, _ in sorted_cats[:2]]
            
    # Post-process Product Ratings (average)
    for product in products:
        pid = product['product_id']
        ratings = product_ratings[pid]
        if ratings:
            product['rating'] = round(sum(ratings) / len(ratings), 1)
        else:
            product['rating'] = 3.0
            
    # Sort interactions by timestamp
    df_interactions = pd.DataFrame(interactions).sort_values('timestamp')
    
    print(f"✓ Created {len(users)} users")
    print(f"✓ Created {len(products)} products")
    print(f"✓ Created {len(df_interactions)} interactions")
    
    # Save files
    with open(output_dir / 'users.json', 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'products.json', 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    df_interactions.to_csv(output_dir / 'interactions.csv', index=False)
    
    print("\nDataset generation complete!")
    print(f"Files saved to {output_dir}")

if __name__ == '__main__':
    URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    DATA_DIR = Path(__file__).parent
    
    ml_dir = download_and_extract(URL, DATA_DIR)
    process_movielens(ml_dir, DATA_DIR)

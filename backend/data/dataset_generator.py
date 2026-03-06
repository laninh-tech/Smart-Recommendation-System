"""
Dataset Loader and Generator
Tạo synthetic user-product interactions data realistic cho training
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class DatasetGenerator:
    def __init__(self, n_users=30, n_products=100, n_interactions=8000):
        self.n_users = n_users
        self.n_products = n_products
        self.n_interactions = n_interactions
        self.data_dir = Path(__file__).parent
        
        # Product categories will be loaded from actual products
        self.categories = []
        
    def generate_users(self, actual_categories):
        """Generate user profiles with preferences"""
        users = []
        
        # Vietnamese names for realistic profiles
        vn_names = [
            'Nguyễn Văn Minh', 'Trần Thị Linh', 'Lê Hoàng Khoa', 'Phạm Thu Hương',
            'Hoàng Văn Tuấn', 'Đỗ Thanh Nga', 'Vũ Minh Dũng', 'Bùi Thị Mai',
            'Đặng Quốc Phúc', 'Ngô Thanh Lan', 'Đinh Văn Hưng', 'Lý Thị Hoa',
            'Phan Hữu Hoàng', 'Mai Thanh Thảo', 'Dương Văn Nam', 'Trương Thị Anh',
            'Võ Minh Long', 'Cao Thị Trang', 'Tạ Văn Hải', 'Lâm Thanh Phương',
            'Hồ Văn Quân', 'Trịnh Thị Vân', 'Phan Thị Hồng', 'Lương Văn Đức',
            'Đào Thị Bình', 'Tô Văn Kiên', 'Đoàn Thị Dung', 'Huỳnh Văn Thắng',
            'Chu Thị Yến', 'La Văn Thành', 'Tăng Thị Xuân', 'Ông Văn Tài',
            'Triệu Thị Nhung', 'Bạch Văn Bảo', 'Khuất Thị Chi', 'Nghiêm Văn Tiến',
            'Quách Thị Diệu', 'Mạc Văn Hùng', 'Thân Thị My', 'Lưu Văn Đạt',
            'Nghiêm Thị Hiền', 'Từ Văn Cường', 'Quang Thị Loan', 'Hà Văn Thọ',
            'Phan Thị Vy', 'Châu Văn Phong', 'Tôn Thị Hà', 'Kim Văn Sơn',
            'Uông Thị Ngọc', 'Lạc Văn Khải', 'Ninh Thị Duyên', 'Vi Văn Trí',
            'Lục Thị Thanh', 'Tống Văn Bình', 'An Thị Tú', 'Khương Văn Lâm',
            'Cung Thị Quyên', 'Thái Văn Nhân', 'Thiều Thị Như', 'Lạc Văn Hiếu',
            'Đường Thị Oanh', 'Khổng Văn Duy'
        ]
        
        for i in range(self.n_users):
            # 20% users không có preferences (cold start - guests)
            if i < self.n_users * 0.2:
                preferences = []
                name = f'Khách {i} (Chưa có sở thích)'
            else:
                # Mỗi user có 1-3 categories yêu thích
                n_prefs = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                preferences = list(np.random.choice(actual_categories, min(n_prefs, len(actual_categories)), replace=False))
                
                # Use Vietnamese name with preferences
                base_name = vn_names[(i - int(self.n_users * 0.2)) % len(vn_names)]
                pref_str = ' & '.join([cat.replace('-', ' ').title() for cat in preferences])
                name = f'{base_name} ({pref_str})'
            
            users.append({
                'user_id': f'u{i}',
                'name': name,
                'preferences': preferences,
                'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+']),
                'active_level': np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
            })
        
        # Save users
        with open(self.data_dir / 'users.json', 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
        
        return users
    
    def fetch_products(self):
        """Fetch real products from DummyJSON"""
        import urllib.request
        
        print("Fetching products from DummyJSON...")
        url = 'https://dummyjson.com/products?limit=100'
        
        # Add User-Agent header to avoid 403
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            products = data['products']
        
        # Simplify product data
        simplified = []
        for p in products:
            simplified.append({
                'product_id': f'p{p["id"]}',
                'title': p['title'],
                'category': p['category'],
                'price': p['price'],
                'rating': p['rating'],
                'thumbnail': p['thumbnail'],
                'brand': p.get('brand', 'Unknown')
            })
        
        # Save products
        with open(self.data_dir / 'products.json', 'w', encoding='utf-8') as f:
            json.dump(simplified, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(simplified)} products")
        return simplified
    
    def generate_interactions(self, users, products):
        """
        Generate realistic user-product interactions
        
        Interaction types:
        - view: 70%
        - add_to_cart: 20%
        - purchase: 10%
        
        Logic:
        - Users prefer products matching their preferences (category)
        - Higher rated products get more interactions
        - Recent interactions more likely (temporal pattern)
        """
        interactions = []
        
        # Create product lookup
        product_by_category = {}
        for p in products:
            cat = p['category']
            if cat not in product_by_category:
                product_by_category[cat] = []
            product_by_category[cat].append(p)
        
        # Base timestamp (30 days ago)
        base_time = datetime.now() - timedelta(days=30)
        
        for _ in range(self.n_interactions):
            # Random user
            user = np.random.choice(users)
            user_id = user['user_id']
            
            # Select product based on user preferences
            if user['preferences'] and np.random.random() < 0.85:
                # 85% chance to pick from preferred categories
                pref_cat = np.random.choice(user['preferences'])
                if pref_cat in product_by_category:
                    # Weighted sampling: prefer higher-rated products
                    pref_products = product_by_category[pref_cat]
                    ratings = np.array([p['rating'] for p in pref_products])
                    weights = ratings / ratings.sum()  # Normalize to probabilities
                    product = np.random.choice(pref_products, p=weights)
                else:
                    product = np.random.choice(products)
            else:
                # 15% chance to explore other categories
                product = np.random.choice(products)
            
            # Action type with realistic distribution
            action = np.random.choice(
                ['view', 'add_to_cart', 'purchase'],
                p=[0.70, 0.20, 0.10]
            )
            
            # Rating (only for purchase)
            rating = None
            if action == 'purchase':
                # Rating bias: higher for preferred categories
                if user['preferences'] and product['category'] in user['preferences']:
                    rating = np.random.choice([4, 5], p=[0.3, 0.7])
                else:
                    rating = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
            
            # Timestamp (more recent interactions more likely)
            # Exponential distribution favoring recent times
            days_ago = int(np.random.exponential(scale=10))
            days_ago = min(days_ago, 30)  # Cap at 30 days
            timestamp = base_time + timedelta(days=days_ago, 
                                             seconds=np.random.randint(0, 86400))
            
            interactions.append({
                'user_id': user_id,
                'product_id': product['product_id'],
                'action': action,
                'rating': rating,
                'timestamp': timestamp.isoformat()
            })
        
        # Sort by timestamp
        interactions.sort(key=lambda x: x['timestamp'])
        
        # Save interactions
        df = pd.DataFrame(interactions)
        df.to_csv(self.data_dir / 'interactions.csv', index=False)
        
        print(f"✓ Generated {len(interactions)} interactions")
        print(f"  - Views: {len(df[df['action']=='view'])}")
        print(f"  - Add to cart: {len(df[df['action']=='add_to_cart'])}")
        print(f"  - Purchases: {len(df[df['action']=='purchase'])}")
        
        return df
    
    def generate_all(self):
        """Generate complete dataset"""
        print("=" * 60)
        print("Generating SmartRec Training Dataset")
        print("=" * 60)
        
        # Fetch products first to get actual categories
        print("\n1. Fetching products...")
        products = self.fetch_products()
        
        # Extract actual categories from products
        actual_categories = list(set([p['category'] for p in products]))
        print(f"\u2713 Found {len(actual_categories)} categories: {', '.join(sorted(actual_categories))}")
        
        print("\n2. Generating users...")
        users = self.generate_users(actual_categories)
        print(f"\u2713 Created {len(users)} users")
        
        print("\n3. Generating interactions...")
        interactions = self.generate_interactions(users, products)
        
        print("\n" + "=" * 60)
        print("Dataset Generation Complete!")
        print("=" * 60)
        print(f"\nFiles created:")
        print(f"  - {self.data_dir / 'users.json'}")
        print(f"  - {self.data_dir / 'products.json'}")
        print(f"  - {self.data_dir / 'interactions.csv'}")
        
        return users, products, interactions


if __name__ == '__main__':
    # Generate dataset
    generator = DatasetGenerator(
        n_users=500,      # Increased from 30
        n_products=500,   # Increased from 100
        n_interactions=50000  # Increased proportionally
    )
    generator.generate_all()

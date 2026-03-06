"""
Verify training data quality: fraction of interactions that match user preferences.
Run from backend directory: python verify_data_matching.py
"""
import json
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent / "data"

df = pd.read_csv(_DATA_DIR / "interactions.csv")
with open(_DATA_DIR / "users.json", "r", encoding="utf-8") as f:
    users = {u["user_id"]: u for u in json.load(f)}
with open(_DATA_DIR / "products.json", "r", encoding="utf-8") as f:
    products = {p["product_id"]: p for p in json.load(f)}

# Calculate matching rate
matches = 0
non_matches = 0
match_by_action = {'view': [0, 0], 'add_to_cart': [0, 0], 'purchase': [0, 0]}

for _, row in df.iterrows():
    user = users.get(row['user_id'])
    product = products.get(row['product_id'])
    
    if user and product and user['preferences']:
        action = row['action']
        if product['category'] in user['preferences']:
            matches += 1
            match_by_action[action][0] += 1
        else:
            non_matches += 1
            match_by_action[action][1] += 1

total = matches + non_matches
print('=' * 60)
print('OVERALL DATA QUALITY')
print('=' * 60)
print(f'Total interactions (users with preferences): {total}')
print(f'Matching preferences: {matches} ({matches/total*100:.1f}%)')
print(f'Non-matching: {non_matches} ({non_matches/total*100:.1f}%)')
print()

print('By Action Type:')
for action in ['view', 'add_to_cart', 'purchase']:
    m, nm = match_by_action[action]
    total_action = m + nm
    if total_action > 0:
        print(f'  {action:12} => {m}/{total_action} matching ({m/total_action*100:5.1f}%)')

print()
print('=' * 60)
print('SAMPLE USERS')
print('=' * 60)

# Analyze 3 sample users
for sample_user in ['u6', 'u7', 'u10']:
    user_data = users[sample_user]
    user_interactions = df[df['user_id'] == sample_user]
    
    if len(user_interactions) == 0:
        continue
    
    print(f'\nUser: {user_data["name"]}')
    print(f'Preferences: {", ".join(user_data["preferences"])}')
    print(f'Total interactions: {len(user_interactions)}')
    
    # Count matches
    user_matches = 0
    user_purchases = 0
    purchase_matches = 0
    
    for _, row in user_interactions.iterrows():
        product = products[row['product_id']]
        if product['category'] in user_data['preferences']:
            user_matches += 1
            if row['action'] == 'purchase':
                purchase_matches += 1
        if row['action'] == 'purchase':
            user_purchases += 1
    
    print(f'Matching category: {user_matches}/{len(user_interactions)} ({user_matches/len(user_interactions)*100:.1f}%)')
    if user_purchases > 0:
        print(f'Purchase matching: {purchase_matches}/{user_purchases} ({purchase_matches/user_purchases*100:.1f}%)')
    
    print('  Recent interactions:')
    for idx, (_, row) in enumerate(user_interactions.tail(5).iterrows()):
        product = products[row['product_id']]
        is_match = '✓' if product['category'] in user_data['preferences'] else '✗'
        rating_str = f' (rating: {row["rating"]})' if pd.notna(row['rating']) else ''
        print(f'    {is_match} {product["title"][:35]:35} [{product["category"]:20}] {row["action"]:12}{rating_str}')

print('\n' + '=' * 60)

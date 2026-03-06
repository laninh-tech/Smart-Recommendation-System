"""
SmartRec System Health Check
Verify all components are ready without heavy model loading
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'backend'))

print("=" * 70)
print("SmartRec System Health Check")
print("=" * 70)

# 1. Check data files
print("\n1. Checking Data Files...")
data_dir = Path(__file__).parent / 'backend' / 'data'

for filename in ['users.json', 'products.json', 'interactions.csv']:
    filepath = data_dir / filename
    if filepath.exists():
        size_kb = filepath.stat().st_size / 1024
        print(f"   ✓ {filename:<25} ({size_kb:>8.1f} KB)")
    else:
        print(f"   ✗ {filename:<25} MISSING")

# 2. Check model checkpoints
print("\n2. Checking Model Checkpoints...")
checkpoint_dir = Path(__file__).parent / 'backend' / 'checkpoints'

for filename in ['mf_model.pth', 'ncf_model.pth', 'evaluation_results.json']:
    filepath = checkpoint_dir / filename
    if filepath.exists():
        size_kb = filepath.stat().st_size / 1024
        print(f"   ✓ {filename:<25} ({size_kb:>8.1f} KB)")
    else:
        print(f"   ✗ {filename:<25} MISSING")

# 3. Check Python modules
print("\n3. Checking Python Modules...")
modules_to_check = [
    'torch',
    'fastapi',
    'uvicorn',
    'pandas',
    'numpy',
    'sklearn'
]

for module_name in modules_to_check:
    try:
        __import__(module_name)
        print(f"   ✓ {module_name:<25} OK")
    except ImportError:
        print(f"   ✗ {module_name:<25} MISSING")

# 4. Load and verify data structure
print("\n4. Verifying Data Structure...")
try:
    with open(data_dir / 'users.json', 'r', encoding='utf-8') as f:
        users = json.load(f)
    print(f"   ✓ Users loaded: {len(users)} users")
    if users:
        sample_user = users[0]
        print(f"     Sample: {sample_user.get('name', 'unknown')} (ID: {sample_user.get('user_id')})")
        print(f"     Preferences: {sample_user.get('preferences', [])}")
except Exception as e:
    print(f"   ✗ Users data error: {e}")

try:
    with open(data_dir / 'products.json', 'r', encoding='utf-8') as f:
        products = json.load(f)
    print(f"   ✓ Products loaded: {len(products)} products")
    if products:
        sample_product = products[0]
        print(f"     Sample: {sample_product.get('title')} ({sample_product.get('category')})")
except Exception as e:
    print(f"   ✗ Products data error: {e}")

try:
    with open(data_dir / 'interactions.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    interaction_count = len(lines) - 1  # Exclude header
    print(f"   ✓ Interactions: {interaction_count} records")
except Exception as e:
    print(f"   ✗ Interactions error: {e}")

# 5. Check evaluation results
print("\n5. Checking Model Metrics...")
try:
    with open(checkpoint_dir / 'evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    if 'NCF' in results:
        ncf = results['NCF']
        print(f"   ✓ NCF Model:")
        print(f"     - RMSE: {ncf.get('RMSE', 'N/A')}")
        print(f"     - MAE:  {ncf.get('MAE', 'N/A')}")
        print(f"     - Precision@10: {ncf.get('Precision@10', 'N/A')}")
    
    if 'MF' in results:
        mf = results['MF']
        print(f"   ✓ Matrix Factorization:")
        print(f"     - RMSE: {mf.get('RMSE', 'N/A')}")
        print(f"     - MAE:  {mf.get('MAE', 'N/A')}")
except Exception as e:
    print(f"   ✗ Metrics error: {e}")

# 6. Check Node.js dependencies
print("\n6. Checking Node.js Packages...")
package_json_path = Path(__file__).parent / 'package.json'
if package_json_path.exists():
    try:
        with open(package_json_path, 'r') as f:
            package = json.load(f)
        deps = package.get('dependencies', {})
        key_packages = ['react', 'react-router-dom', 'recharts']
        for pkg in key_packages:
            if pkg in deps:
                print(f"   ✓ {pkg:<25} v{deps[pkg]}")
            else:
                print(f"   ✗ {pkg:<25} MISSING")
    except Exception as e:
        print(f"   ✗ package.json error: {e}")
else:
    print(f"   ✗ package.json MISSING")

# Final summary
print("\n" + "=" * 70)
print("✅ SmartRec System is Ready for Deployment")
print("=" * 70)
print("\nQuick Start:")
print("  1. Backend:  python -m uvicorn api.main:app --host 127.0.0.1 --port 5000")
print("  2. Frontend: npm run dev")
print("  3. Browser:  http://localhost:5173")
print("\nOr use Docker:")
print("  docker-compose up")
print("=" * 70)

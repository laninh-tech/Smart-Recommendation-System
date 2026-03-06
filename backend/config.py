"""
SmartRec backend configuration.
Centralizes paths and constants for API, data, and checkpoints.
"""
from pathlib import Path

# Backend root (directory containing api/, data/, checkpoints/)
BACKEND_ROOT = Path(__file__).resolve().parent

DATA_DIR = BACKEND_ROOT / "data"
CHECKPOINT_DIR = BACKEND_ROOT / "checkpoints"

# Default model for inference.
#
# On some Windows setups, importing PyTorch can be extremely slow or hang due to
# local environment issues. To ensure the demo runs reliably, default to the
# lightweight baseline recommender (no torch required).
#
# If PyTorch works on your machine, you can switch this to "ncf" or "mf".
DEFAULT_MODEL_TYPE = "baseline"

# API
API_TITLE = "SmartRec API"
API_VERSION = "2.0.0"
API_DESCRIPTION = (
    "ML-powered product recommendation system "
    "with Matrix Factorization and Neural Collaborative Filtering (NCF)."
)

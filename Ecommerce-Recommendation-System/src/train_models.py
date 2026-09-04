"""
train_models.py
=================
Entrypoint that fits and persists all Phase 3 models.

Deliberately kept as a THIN separate script rather than a
`if __name__ == "__main__":` block inside `recommender.py` itself: running
`recommender.py` directly would rebind its classes under the `__main__`
module path for pickling, breaking `joblib.load(...)` later from any other
entrypoint (a classic, easy-to-miss pickle footgun). Because this script
only *imports* the classes from `src.recommender`, they keep their real
`src.recommender.ClassName` identity in the pickle, and `load_all_models()`
can unpickle them from anywhere (the Streamlit app, the evaluation
notebook, a test suite, etc.).

Usage:
    python -m src.train_models
"""

from src.recommender import train_and_save_all_models
from src.utils import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    train_and_save_all_models()
    logger.info("All models trained and saved to models/.")

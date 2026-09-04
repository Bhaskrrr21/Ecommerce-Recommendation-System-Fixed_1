"""
config.py
=========
Centralized configuration for the E-commerce Recommendation System.

Design rationale
----------------
Hard-coding paths, thresholds, and hyperparameters throughout a codebase makes a
project brittle: a single change (e.g. moving the data directory, or tuning a
decay constant for "recency" scoring) requires touching many files. Instead we
expose one `Config` object that every module imports from. This mirrors how
production ML services typically manage settings (12-factor config, or a
dataclass-based settings object), and makes the project trivially portable
between a laptop, a CI pipeline, and Streamlit Community Cloud.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
LOG_DIR = BASE_DIR / "logs"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, ASSETS_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Raw Retailrocket file names
# ---------------------------------------------------------------------------
RAW_FILES = {
    "events": DATA_RAW_DIR / "events.csv",
    "item_properties_part1": DATA_RAW_DIR / "item_properties_part1.csv",
    "item_properties_part2": DATA_RAW_DIR / "item_properties_part2.csv",
    "category_tree": DATA_RAW_DIR / "category_tree.csv",
}

# Processed / feature-engineered artifacts produced by src/preprocessing.py
# and src/feature_engineering.py in later phases.
PROCESSED_FILES = {
    "events_clean": DATA_PROCESSED_DIR / "events_clean.parquet",
    "item_properties_clean": DATA_PROCESSED_DIR / "item_attributes.parquet",
    "category_tree_clean": DATA_PROCESSED_DIR / "category_tree_clean.parquet",
    "sessions": DATA_PROCESSED_DIR / "sessions.parquet",
    "user_id_map": DATA_PROCESSED_DIR / "user_id_map.parquet",
    "item_id_map": DATA_PROCESSED_DIR / "item_id_map.parquet",
    "user_features": DATA_PROCESSED_DIR / "user_features.parquet",
    "item_features": DATA_PROCESSED_DIR / "item_features.parquet",
    "category_features": DATA_PROCESSED_DIR / "category_features.parquet",
}


# ---------------------------------------------------------------------------
# Event semantics
# ---------------------------------------------------------------------------
# Retailrocket's `events.csv` encodes three event types. We define an implicit
# "value" per event, used later for weighting interaction strength in
# collaborative filtering (a purchase is worth far more signal than a view).
EVENT_TYPES = ("view", "addtocart", "transaction")

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}

# Special parent-category id used by Retailrocket's category_tree.csv to mark
# root-level categories (no parent).
ROOT_CATEGORY_MARKER = None


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------
# A "session" is not given directly in Retailrocket; we derive it. Industry
# standard (also used by Google Analytics) is a 30-minute inactivity cutoff.
SESSION_TIMEOUT_MINUTES = 30


# ---------------------------------------------------------------------------
# Modeling configuration (used from Phase 3 onward; declared here so config
# stays the single source of truth)
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    top_k: int = 10                     # Default recommendation list length
    min_interactions_per_user: int = 3  # Filters out near-cold-start users for CF
    min_interactions_per_item: int = 3  # Filters out near-cold-start items for CF
    svd_n_factors: int = 50
    svd_random_state: int = 42
    tfidf_max_features: int = 5000
    popularity_half_life_days: float = 14.0  # Recency decay for "trending"
    outlier_cap_percentile: float = 0.99     # Winsorize activity/interaction counts above this pct


MODEL_CONFIG = ModelConfig()


# ---------------------------------------------------------------------------
# Synthetic data generation (used only when real Retailrocket files are
# absent — see src/generate_sample_data.py). This lets every downstream
# script run end-to-end without the ~2.7M-row real dataset present.
# ---------------------------------------------------------------------------
@dataclass
class SyntheticDataConfig:
    n_users: int = 4000
    n_items: int = 3000
    n_categories: int = 120
    n_events: int = 120_000
    random_seed: int = 42
    days_span: int = 120


SYNTHETIC_CONFIG = SyntheticDataConfig()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("RECSYS_LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "recsys.log"

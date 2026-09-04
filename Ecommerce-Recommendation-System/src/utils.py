"""
utils.py
========
Cross-cutting utilities shared across the pipeline: logging setup, timing
decorators, and small I/O helpers. Keeping these here avoids duplicating
boilerplate in preprocessing.py, feature_engineering.py, recommender.py, etc.
"""

from __future__ import annotations

import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

from src.config import LOG_FILE, LOG_LEVEL

F = TypeVar("F", bound=Callable[..., Any])


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger configured to write to both stdout and a
    shared log file (logs/recsys.log).

    Using `getLogger(__name__)` per-module (rather than the root logger)
    is standard practice: it lets us filter/route logs by component if the
    project grows (e.g. silencing DEBUG logs from feature_engineering while
    keeping them for recommender).
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # Avoid duplicate handlers on re-import / notebook re-run
        return logger

    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # In read-only or ephemeral environments (e.g. some CI runners),
        # fall back silently to console-only logging.
        pass

    return logger


def timeit(func: F) -> F:
    """Decorator that logs the execution time of a function call.

    Used on the heavier pipeline steps (loading multi-million-row CSVs,
    fitting SVD, computing TF-IDF matrices) so that performance regressions
    are visible in logs rather than only discovered from user complaints.
    """
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2fs", func.__qualname__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]


def ensure_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    """Raise a clear, actionable error if a DataFrame is missing expected columns.

    Fails fast with a descriptive message instead of letting a downstream
    KeyError surface deep inside, e.g., a similarity computation.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def read_csv_safely(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Wrapper around pd.read_csv with a friendlier error for missing files.

    Points the user at the exact fix (drop the Retailrocket CSVs into
    data/raw/, or run generate_sample_data.py) instead of a bare
    FileNotFoundError.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n"
            f"→ Either place the real Retailrocket CSV at this path, or run "
            f"`python -m src.generate_sample_data` to create a schema-accurate "
            f"synthetic dataset for local development."
        )
    return pd.read_csv(path, **kwargs)

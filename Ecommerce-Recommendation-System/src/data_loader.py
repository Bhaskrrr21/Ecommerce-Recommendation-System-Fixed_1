"""
data_loader.py
==============
Thin, well-documented loading layer for the four Retailrocket files.

Isolating "how do I read the raw files" from "what do I do with them" (EDA,
preprocessing, feature engineering) means that if the on-disk format ever
changes (e.g. moving from CSV to Parquet, or reading from S3 instead of
local disk), only this module needs to change.
"""

from __future__ import annotations

import pandas as pd

from src.config import RAW_FILES
from src.utils import ensure_columns, get_logger, read_csv_safely, timeit

logger = get_logger(__name__)

EVENTS_SCHEMA = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
ITEM_PROPS_SCHEMA = ["timestamp", "itemid", "property", "value"]
CATEGORY_TREE_SCHEMA = ["categoryid", "parentid"]


@timeit
def load_events() -> pd.DataFrame:
    """Load events.csv and parse its epoch-millisecond timestamp column."""
    df = read_csv_safely(RAW_FILES["events"])
    ensure_columns(df, EVENTS_SCHEMA, context="load_events")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


@timeit
def load_item_properties() -> pd.DataFrame:
    """Load and concatenate the two item_properties parts.

    Retailrocket splits this file in two purely for file-size reasons on
    Kaggle — there's no semantic difference between part1 and part2, so we
    always treat them as a single logical table.
    """
    part1 = read_csv_safely(RAW_FILES["item_properties_part1"])
    part2 = read_csv_safely(RAW_FILES["item_properties_part2"])
    df = pd.concat([part1, part2], ignore_index=True)
    ensure_columns(df, ITEM_PROPS_SCHEMA, context="load_item_properties")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


@timeit
def load_category_tree() -> pd.DataFrame:
    """Load category_tree.csv (categoryid -> parentid, NaN parent = root)."""
    df = read_csv_safely(RAW_FILES["category_tree"])
    ensure_columns(df, CATEGORY_TREE_SCHEMA, context="load_category_tree")
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Convenience loader returning all four tables in one dict."""
    return {
        "events": load_events(),
        "item_properties": load_item_properties(),
        "category_tree": load_category_tree(),
    }

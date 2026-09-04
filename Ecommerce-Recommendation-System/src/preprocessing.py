"""
preprocessing.py
================
Cleaning, ID encoding, session construction, and interaction aggregation
for the Retailrocket dataset.

This module turns the three raw tables (events, item_properties,
category_tree) into clean, analysis-ready tables. It does NOT compute the
final modeling features (purchase counts, conversion rates, activity
scores, etc.) — that's `feature_engineering.py`. The split mirrors a
standard production pipeline: preprocessing is about *correctness and
structure*, feature engineering is about *signal*.

Design decisions (see Phase 1 EDA for the evidence behind each one)
---------------------------------------------------------------------
1. Duplicate events are dropped exactly once (exact row duplicates only —
   two genuinely separate views of the same item one second apart are
   NOT duplicates and must NOT be collapsed).
2. Outliers are NOT dropped from the raw event log. The Phase 1 EDA
   confirmed that "extreme" users/items are legitimate power users and
   best-sellers, not bots — removing their rows would delete the exact
   signal collaborative filtering needs most. Instead, extreme *aggregate
   counts* are winsorized (capped at the 99th percentile) only when they
   feed into a *score* that could otherwise be dominated by one outlier
   (implemented in feature_engineering.py, not here).
3. Sessions are derived, not given. We use the industry-standard 30-minute
   inactivity cutoff (src.config.SESSION_TIMEOUT_MINUTES), matching
   Google Analytics' definition — a defensible, well-known convention
   rather than an arbitrary one.
4. IDs are label-encoded (visitorid/itemid -> contiguous 0..N-1 indices).
   Raw Retailrocket IDs are large, sparse integers; contiguous indices are
   required for the sparse matrices used in Phase 3 collaborative
   filtering (a user-item matrix indexed by raw IDs would be absurdly
   wide). The original <-> encoded mapping is persisted so predictions can
   always be mapped back to real IDs.
5. Item properties are a time-varying "EAV" log (one row per property
   change). We take the LATEST value per (item, property) as of the
   snapshot date, because for recommendation purposes we care about the
   product's current state, not its full change history.
6. Missing values are handled explicitly and per-column, not with a
   blanket `fillna(0)` — each decision is justified inline below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_FILES,
    SESSION_TIMEOUT_MINUTES,
)
from src.data_loader import load_all
from src.utils import get_logger, timeit

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Events cleaning
# ---------------------------------------------------------------------------
@timeit
def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and structurally validate the events table.

    Missing-value handling: `transactionid` is NaN for every non-transaction
    row by design (confirmed in Phase 1 EDA) — this is NOT missing data to
    impute, so we leave it untouched. We only assert the invariant holds.
    """
    before = len(events)
    df = events.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    dropped = before - len(df)
    logger.info("clean_events: dropped %d exact duplicate rows (%.3f%%)", dropped, 100 * dropped / before)

    invariant_ok = (df["transactionid"].isna() == (df["event"] != "transaction")).all()
    if not invariant_ok:
        logger.warning(
            "clean_events: transactionid/event invariant violated for some rows — "
            "flagging rather than silently fixing, since this suggests a genuine "
            "data quality issue upstream."
        )

    # itemid/visitorid should never be missing — if they are, the row carries
    # no usable signal for any downstream model and is safe to drop.
    na_before = len(df)
    df = df.dropna(subset=["visitorid", "itemid", "event"])
    na_dropped = na_before - len(df)
    if na_dropped:
        logger.info("clean_events: dropped %d rows missing visitorid/itemid/event", na_dropped)

    df["visitorid"] = df["visitorid"].astype(np.int64)
    df["itemid"] = df["itemid"].astype(np.int64)
    df["event"] = df["event"].astype("category")

    return df


@timeit
def encode_ids(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label-encode visitorid/itemid to contiguous integer indices.

    Returns (events_with_idx, user_id_map, item_id_map). The maps are
    persisted to parquet so any later step (or the Streamlit app) can
    translate a model's internal index back to the original Retailrocket ID.
    """
    user_codes, user_uniques = pd.factorize(events["visitorid"], sort=True)
    item_codes, item_uniques = pd.factorize(events["itemid"], sort=True)

    df = events.copy()
    df["user_idx"] = user_codes
    df["item_idx"] = item_codes

    user_id_map = pd.DataFrame({"visitorid": user_uniques, "user_idx": np.arange(len(user_uniques))})
    item_id_map = pd.DataFrame({"itemid": item_uniques, "item_idx": np.arange(len(item_uniques))})

    logger.info(
        "encode_ids: %d unique visitors -> user_idx, %d unique items -> item_idx",
        len(user_id_map), len(item_id_map),
    )
    return df, user_id_map, item_id_map


# ---------------------------------------------------------------------------
# 2. Session construction
# ---------------------------------------------------------------------------
@timeit
def build_sessions(events: pd.DataFrame, timeout_minutes: int = SESSION_TIMEOUT_MINUTES) -> pd.DataFrame:
    """Assign a session_id to every event using a 30-minute inactivity cutoff.

    Retailrocket has no session field, so we derive one: within a visitor's
    event stream (sorted by time), a new session starts whenever the gap
    since the previous event exceeds `timeout_minutes`. This is the same
    definition used by Google Analytics, chosen for defensibility over an
    arbitrary threshold.
    """
    df = events.sort_values(["visitorid", "timestamp"]).copy()
    gap = df.groupby("visitorid")["datetime"].diff()
    new_session = (gap.isna()) | (gap > pd.Timedelta(minutes=timeout_minutes))
    df["session_seq"] = new_session.groupby(df["visitorid"]).cumsum().astype(np.int64)
    # Human/machine-readable global session key: "{visitorid}_{session_seq}"
    df["session_id"] = df["visitorid"].astype(str) + "_" + df["session_seq"].astype(str)

    n_sessions = df["session_id"].nunique()
    logger.info(
        "build_sessions: %d sessions derived from %d events across %d visitors (timeout=%dmin)",
        n_sessions, len(df), df["visitorid"].nunique(), timeout_minutes,
    )
    return df


@timeit
def summarize_sessions(events_with_sessions: pd.DataFrame) -> pd.DataFrame:
    """One row per session: duration, event count, event-type breakdown.

    Session duration for a single-event session is defined as 0 seconds
    (not NaN) — a session did happen, it was just instantaneous; treating
    it as missing would incorrectly drop it from duration averages.

    Implementation note: we compute the event-type breakdown via a pivoted
    crosstab rather than per-group lambdas — algebraically identical, but
    orders of magnitude faster on 100k+ event logs since it avoids Python-level
    function calls per group.
    """
    df = events_with_sessions
    keys = ["visitorid", "session_id"]

    timing = df.groupby(keys, observed=True)["datetime"].agg(
        session_start="min", session_end="max", n_events="size"
    ).reset_index()

    event_counts = (
        pd.crosstab([df["visitorid"], df["session_id"]], df["event"])
        .rename(columns={"view": "n_views", "addtocart": "n_addtocart", "transaction": "n_transactions"})
        .reset_index()
    )
    for col in ["n_views", "n_addtocart", "n_transactions"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    sessions = timing.merge(event_counts, on=keys, how="left")
    sessions["duration_sec"] = (
        (sessions["session_end"] - sessions["session_start"]).dt.total_seconds()
    )
    return sessions


# ---------------------------------------------------------------------------
# 3. Item properties cleaning
# ---------------------------------------------------------------------------
def _parse_hashed_numeric(value: object) -> float:
    """Retailrocket hashes some numeric properties (e.g. price) with a
    leading 'n' to anonymize the true scale while preserving relative order,
    e.g. 'n39146.000' -> 39146.0. Non-numeric / non-hashed values return NaN.
    """
    if not isinstance(value, str) or not value.startswith("n"):
        return np.nan
    try:
        return float(value[1:])
    except ValueError:
        return np.nan


@timeit
def clean_item_properties(item_properties: pd.DataFrame) -> pd.DataFrame:
    """Collapse the time-varying EAV property log into one row per item
    with the LATEST known value of each property of interest.

    We extract exactly the properties this project uses downstream:
    'categoryid', 'available', and the hashed numeric property '790'
    (treated as price, per the project's data dictionary). Other hashed
    properties exist in the real dataset but aren't used by any model in
    this project, so we don't carry their full cardinality forward.

    Missing-value handling:
      - Items with no 'available' property recorded: default to
        available=1 (unknown availability shouldn't silently hide a
        product from recommendations — the conservative choice for a
        recommender is to assume it CAN be recommended, and let downstream
        real inventory checks be the final gate).
      - Items with no 'categoryid': kept as categoryid=NaN and flagged via
        `has_category=False`, rather than imputed — assigning a fake
        category would corrupt content-based similarity.
      - Items with no price signal: price=NaN, `has_price=False`. Models
        that need price (none in this project yet) must handle NaN
        explicitly rather than have it silently zero-filled (zero looks
        like "free", which is a meaningfully different value).
    """
    df = item_properties.sort_values("timestamp")

    latest = (
        df[df["property"].isin(["categoryid", "available", "790"])]
        .drop_duplicates(subset=["itemid", "property"], keep="last")
    )

    wide = latest.pivot(index="itemid", columns="property", values="value").reset_index()
    wide.columns.name = None
    for col in ["categoryid", "available", "790"]:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["categoryid"] = pd.to_numeric(wide["categoryid"], errors="coerce")
    wide["has_category"] = wide["categoryid"].notna()

    wide["price"] = wide["790"].apply(_parse_hashed_numeric)
    wide["has_price"] = wide["price"].notna()
    wide = wide.drop(columns=["790"])

    wide["available"] = pd.to_numeric(wide["available"], errors="coerce")
    n_missing_avail = wide["available"].isna().sum()
    wide["available"] = wide["available"].fillna(1).astype(int)

    logger.info(
        "clean_item_properties: %d items | missing category: %d | missing price: %d | "
        "missing availability (defaulted to available=1): %d",
        len(wide), (~wide["has_category"]).sum(), (~wide["has_price"]).sum(), n_missing_avail,
    )
    return wide


# ---------------------------------------------------------------------------
# 4. Category tree cleaning
# ---------------------------------------------------------------------------
@timeit
def clean_category_tree(category_tree: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate the category tree and flag root categories.

    A NaN `parentid` is not missing data — it's how Retailrocket marks a
    top-level ("root") category. We make this explicit with an `is_root`
    flag rather than leaving analysts to rediscover the convention.
    """
    df = category_tree.drop_duplicates(subset="categoryid").copy()
    df["is_root"] = df["parentid"].isna()
    logger.info(
        "clean_category_tree: %d categories (%d root, %d child)",
        len(df), df["is_root"].sum(), (~df["is_root"]).sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@timeit
def run_preprocessing_pipeline(persist: bool = True) -> dict[str, pd.DataFrame]:
    """Run the full Phase 2 preprocessing pipeline end to end and optionally
    persist every intermediate table to data/processed/ as parquet.
    """
    raw = load_all()

    events_clean = clean_events(raw["events"])
    events_encoded, user_id_map, item_id_map = encode_ids(events_clean)
    events_with_sessions = build_sessions(events_encoded)
    sessions = summarize_sessions(events_with_sessions)

    item_attributes = clean_item_properties(raw["item_properties"])
    category_tree_clean = clean_category_tree(raw["category_tree"])

    outputs = {
        "events_clean": events_with_sessions,
        "sessions": sessions,
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "item_properties_clean": item_attributes,
        "category_tree_clean": category_tree_clean,
    }

    if persist:
        for key, df in outputs.items():
            path = PROCESSED_FILES[key]
            df.to_parquet(path, index=False)
            logger.info("Persisted %s -> %s (%d rows)", key, path, len(df))

    return outputs


if __name__ == "__main__":
    run_preprocessing_pipeline(persist=True)

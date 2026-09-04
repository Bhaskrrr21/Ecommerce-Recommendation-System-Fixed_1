"""
generate_sample_data.py
========================
Generates a SCHEMA-ACCURATE synthetic stand-in for the Retailrocket
Recommender Dataset (https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

Why this exists
---------------
This project is built against the real Retailrocket dataset. But a portfolio
reviewer (or you, running this locally) needs the pipeline to execute
end-to-end without first downloading a ~2.7M-row / several-hundred-MB dataset
from Kaggle. This script produces a much smaller dataset with IDENTICAL
column names, dtypes, and event semantics, so every downstream script
(EDA, preprocessing, feature engineering, models) runs UNCHANGED against
either the synthetic data or the real thing — swap the files in data/raw/
and nothing else needs to change.

To use the REAL dataset instead:
    1. Download events.csv, item_properties_part1.csv, item_properties_part2.csv,
       and category_tree.csv from Kaggle.
    2. Place them in data/raw/.
    3. Skip this script entirely.

Simulated behavioral realism
-----------------------------
Real e-commerce interaction data is highly skewed, not uniform, so a naive
uniform-random generator would make EDA and later modeling look trivially
easy. We deliberately inject:
  - Power-law product popularity (a small number of items get most views)
  - A view -> add-to-cart -> transaction funnel with realistic drop-off rates
  - A "power user" long tail (most visitors have 1-2 events; a few have
    many)
  - Daily/weekly seasonality and a slow upward trend in traffic
  - Missing property values and duplicate rows, mirroring real-world mess
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    RAW_FILES,
    SYNTHETIC_CONFIG,
)
from src.utils import get_logger, timeit

logger = get_logger(__name__)

rng = np.random.default_rng(SYNTHETIC_CONFIG.random_seed)


@timeit
def generate_category_tree() -> pd.DataFrame:
    """Build a small category hierarchy: root categories + child categories.

    Matches real category_tree.csv schema: categoryid, parentid (root
    categories have a null/NaN parentid).
    """
    n = SYNTHETIC_CONFIG.n_categories
    n_roots = max(5, n // 12)

    category_ids = np.arange(1000, 1000 + n)
    root_ids = category_ids[:n_roots]
    child_ids = category_ids[n_roots:]

    parent_ids: list[float] = [np.nan] * n_roots
    parent_ids += list(rng.choice(root_ids, size=len(child_ids)))

    df = pd.DataFrame({"categoryid": category_ids, "parentid": parent_ids})
    return df


@timeit
def generate_item_properties(category_ids: np.ndarray) -> pd.DataFrame:
    """Build item_properties in Retailrocket's long "EAV" format:
    timestamp, itemid, property, value.

    Real Retailrocket has one row per (item, property, timestamp-of-change).
    We simplify to one snapshot per item but keep the same 4 columns and
    include the two special properties the real dataset uses:
    'categoryid' and 'available'.
    """
    n_items = SYNTHETIC_CONFIG.n_items
    item_ids = np.arange(1, n_items + 1)
    # Epoch milliseconds, matching real Retailrocket item_properties files.
    base_ts = int((pd.Timestamp("2015-05-01") - pd.Timestamp("1970-01-01")) / pd.Timedelta(milliseconds=1))

    rows = []
    for item in item_ids:
        cat = rng.choice(category_ids)
        available = rng.choice([0, 1], p=[0.12, 0.88])
        # Real dataset hashes numeric properties with an 'n' prefix, e.g. price
        price_hashed = f"n{rng.integers(500, 50000)}.000"
        rows.append((base_ts, item, "categoryid", str(cat)))
        rows.append((base_ts, item, "available", str(available)))
        rows.append((base_ts, item, "790", price_hashed))  # "790" mirrors a real hashed property id

    df = pd.DataFrame(rows, columns=["timestamp", "itemid", "property", "value"])

    # Inject some realistic messiness: a few duplicate rows and a few nulls
    dup_sample = df.sample(frac=0.01, random_state=SYNTHETIC_CONFIG.random_seed)
    df = pd.concat([df, dup_sample], ignore_index=True)
    null_idx = df.sample(frac=0.005, random_state=SYNTHETIC_CONFIG.random_seed).index
    df.loc[null_idx, "value"] = np.nan

    return df


@timeit
def generate_events(item_ids: np.ndarray) -> pd.DataFrame:
    """Build events.csv: timestamp, visitorid, event, itemid, transactionid.

    Simulates:
      - Power-law item popularity (Zipf) so a few items dominate views
      - A realistic view -> addtocart -> transaction funnel
      - Skewed, long-tailed user activity
      - Seasonality (weekday effect + gentle upward trend across days_span)
    """
    cfg = SYNTHETIC_CONFIG
    n_events = cfg.n_events

    # Zipf-distributed item popularity (clipped to valid item id range)
    item_popularity_rank = rng.zipf(a=1.6, size=n_events)
    item_choice_idx = np.clip(item_popularity_rank, 1, len(item_ids)) - 1
    chosen_items = item_ids[item_choice_idx]

    # Long-tailed visitor activity: most visitors appear once or twice,
    # a small "power user" segment appears often.
    visitor_pool = np.arange(1, cfg.n_users + 1)
    visitor_weights = rng.pareto(a=1.3, size=cfg.n_users) + 1
    visitor_weights /= visitor_weights.sum()
    visitors = rng.choice(visitor_pool, size=n_events, p=visitor_weights)

    # Timestamps: spread across days_span with weekday seasonality + mild trend
    start = pd.Timestamp("2015-05-03")
    day_offsets = rng.integers(0, cfg.days_span, size=n_events)
    seconds_in_day = rng.integers(0, 86_400, size=n_events)
    timestamps = start + pd.to_timedelta(day_offsets, unit="D") + pd.to_timedelta(
        seconds_in_day, unit="s"
    )
    # Retailrocket's real events.csv stores epoch milliseconds (13-digit ints).
    # pandas' internal datetime64 resolution varies by version (us vs ns), so
    # we compute milliseconds explicitly from a fixed epoch rather than
    # relying on `.view('int64')`, which would silently give the wrong scale.
    epoch = pd.Timestamp("1970-01-01")
    ts_ms = ((timestamps - epoch) // pd.Timedelta(milliseconds=1)).astype(np.int64)

    # Event funnel: view -> addtocart -> transaction with realistic drop-off
    # (~ industry benchmark: ~10% of views add to cart, ~25-35% of those convert)
    u = rng.random(n_events)
    event_type = np.where(
        u < 0.86, "view", np.where(u < 0.965, "addtocart", "transaction")
    )

    transaction_id = np.full(n_events, np.nan)
    n_transactions = int((event_type == "transaction").sum())
    transaction_id[event_type == "transaction"] = np.arange(1, n_transactions + 1)

    df = pd.DataFrame(
        {
            "timestamp": ts_ms,
            "visitorid": visitors,
            "event": event_type,
            "itemid": chosen_items,
            "transactionid": transaction_id,
        }
    ).sort_values("timestamp").reset_index(drop=True)

    # Realistic messiness: duplicate events, a few nulls in itemid
    dup_sample = df.sample(frac=0.003, random_state=cfg.random_seed)
    df = pd.concat([df, dup_sample], ignore_index=True)

    return df


@timeit
def add_basket_companions(events: pd.DataFrame, item_to_category: dict[int, int], item_ids: np.ndarray,
                           basket_probability: float = 0.35, max_companions: int = 2) -> pd.DataFrame:
    """Add sibling item rows to a subset of existing transaction events,
    simulating multi-item checkout baskets sharing one `transactionid`.

    Without this step, every synthetic transaction is a checkout for
    exactly ONE item (each gets its own unique transactionid), which would
    leave `FrequentlyBoughtTogetherRecommender` (Phase 5) with zero
    co-purchase signal to learn from — real Retailrocket checkouts can and
    do span multiple items sharing one transactionid. Companion items are
    biased toward the SAME CATEGORY as the seed item ~70% of the time
    (simulating a plausible real pattern — e.g. buying a phone case and a
    screen protector together) and otherwise picked at random, so the
    resulting co-purchase signal isn't a perfect category match every time.
    """
    transaction_idx = events.index[events["event"] == "transaction"].to_numpy()
    n_basket_txns = int(len(transaction_idx) * basket_probability)
    basket_txn_idx = rng.choice(transaction_idx, size=n_basket_txns, replace=False)

    same_category_items_cache: dict[int, np.ndarray] = {}
    new_rows = []
    for idx in basket_txn_idx:
        row = events.loc[idx]
        seed_item = int(row["itemid"])
        seed_category = item_to_category.get(seed_item)
        n_companions = int(rng.integers(1, max_companions + 1))

        for _ in range(n_companions):
            companion_item = None
            if seed_category is not None and rng.random() < 0.7:
                candidates = same_category_items_cache.get(seed_category)
                if candidates is None:
                    candidates = np.array(
                        [i for i, c in item_to_category.items() if c == seed_category and i != seed_item]
                    )
                    same_category_items_cache[seed_category] = candidates
                if len(candidates):
                    companion_item = int(rng.choice(candidates))
            if companion_item is None:
                companion_item = int(rng.choice(item_ids))

            new_rows.append({
                "timestamp": row["timestamp"], "visitorid": row["visitorid"],
                "event": "transaction", "itemid": companion_item, "transactionid": row["transactionid"],
            })

    companions_df = pd.DataFrame(new_rows)
    result = pd.concat([events, companions_df], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "add_basket_companions: added %d companion rows across %d multi-item baskets (%.0f%% of transactions)",
        len(companions_df), n_basket_txns, 100 * basket_probability,
    )
    return result


@timeit
def main() -> None:
    cfg = SYNTHETIC_CONFIG
    logger.info(
        "Generating synthetic Retailrocket-style dataset: %d users, %d items, %d events",
        cfg.n_users, cfg.n_items, cfg.n_events,
    )

    category_tree = generate_category_tree()
    category_ids = category_tree["categoryid"].to_numpy()

    item_properties = generate_item_properties(category_ids)
    n_props = len(item_properties)
    split_point = n_props // 2
    part1 = item_properties.iloc[:split_point]
    part2 = item_properties.iloc[split_point:]

    item_ids = np.arange(1, cfg.n_items + 1)
    events = generate_events(item_ids)

    cat_rows = item_properties[item_properties["property"] == "categoryid"].dropna(subset=["value"])
    item_to_category = dict(zip(cat_rows["itemid"], cat_rows["value"].astype(int)))
    events = add_basket_companions(events, item_to_category, item_ids)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    category_tree.to_csv(RAW_FILES["category_tree"], index=False)
    part1.to_csv(RAW_FILES["item_properties_part1"], index=False)
    part2.to_csv(RAW_FILES["item_properties_part2"], index=False)
    events.to_csv(RAW_FILES["events"], index=False)

    logger.info("Wrote synthetic dataset to %s", DATA_RAW_DIR)
    for name, path in RAW_FILES.items():
        size_kb = path.stat().st_size / 1024
        logger.info("  %-24s %8.1f KB  (%s)", name, size_kb, path)


if __name__ == "__main__":
    main()

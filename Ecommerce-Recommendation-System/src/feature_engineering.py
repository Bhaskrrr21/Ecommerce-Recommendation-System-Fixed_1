"""
feature_engineering.py
=======================
Builds the user-level, item-level, and category-level feature tables that
every recommendation model in Phase 3 will consume.

Feature list and rationale
---------------------------
User features:
  - user_purchase_count      : direct measure of buying intent/value; used to
                                weight a user's profile more heavily in
                                collaborative filtering and to segment
                                high-value customers for retention targeting.
  - user_view_count           : breadth of browsing activity.
  - unique_items_viewed / unique_categories_viewed : signal for how
                                exploratory vs. narrow a shopper is — narrow
                                shoppers are better served by item-level
                                recommendations, broad shoppers by
                                category-level ones.
  - n_sessions / avg_session_duration_sec : engagement depth; short,
                                frequent sessions vs. long, rare ones imply
                                different UX (mobile quick-check vs.
                                desktop research session).
  - conversion_rate_user       : this user's personal view->purchase rate —
                                useful for identifying "browsers" who need a
                                nudge (e.g. a discount) vs. "buyers" who
                                just need the right item surfaced.
  - click_to_purchase_ratio_user : addtocart->purchase rate — isolates cart
                                abandonment specifically, a different
                                intervention point than browsing behavior.
  - recency_days               : days since last activity — the single
                                strongest predictor of near-term return
                                likelihood in most e-commerce cohort studies.
  - user_activity_score        : a single composite ranking signal (event-type
                                weighted + recency-decayed) used wherever a
                                simple "how engaged is this user" number is
                                needed (e.g. deciding how much to trust
                                collaborative filtering vs. fall back to
                                popularity for a given user).

Item features:
  - item_view_count (product popularity), item_addtocart_count,
    item_transaction_count : raw funnel volumes per product.
  - product_conversion_rate    : separates "popular to browse" from "popular
                                to buy" (Phase 1 EDA showed these differ).
  - click_to_purchase_ratio_item : cart-to-purchase rate per product — flags
                                products with a specific cart-abandonment
                                problem (e.g. shipping-cost surprise, low
                                stock messaging) distinct from a discovery
                                problem.
  - category_popularity        : this item's category's aggregate popularity
                                — used as a smoothed prior for items with too
                                little individual signal (new/cold items).
  - product_interaction_score  : composite, recency-decayed popularity score
                                — the core input to the "Trending Now" /
                                "Best Sellers" business features.
  - recency_days               : days since last interaction — used to
                                distinguish "currently trending" from
                                "was popular once."

Normalization
-------------
Raw counts and composite scores are heavy-tailed (Phase 1 EDA: power-law
popularity). We winsorize at the 99th percentile before min-max scaling so
a single extreme outlier can't compress every other value into a rounding
error near 0 — this is the concrete implementation of the Phase 1 "cap,
don't drop" decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EVENT_WEIGHTS, MODEL_CONFIG, PROCESSED_FILES
from src.utils import get_logger, timeit

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared normalization helper
# ---------------------------------------------------------------------------
def winsorize_and_scale(series: pd.Series, percentile: float = MODEL_CONFIG.outlier_cap_percentile) -> pd.Series:
    """Cap extreme values at the given percentile, then min-max scale to [0, 1].

    Winsorizing (rather than dropping) preserves every row while preventing
    one power-user/best-seller from dominating the scale — exactly the
    "cap, don't drop" decision justified by the Phase 1 outlier analysis.
    """
    if series.empty:
        return series
    cap = series.quantile(percentile)
    capped = series.clip(upper=cap)
    lo, hi = capped.min(), capped.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (capped - lo) / (hi - lo)


def _time_decayed_weight(events: pd.DataFrame, half_life_days: float) -> pd.Series:
    """Per-event weight = event-type importance (EVENT_WEIGHTS) × recency
    decay (exponential half-life). Recent transactions score highest;
    old views score lowest — this single formula is the basis for both
    user_activity_score and product_interaction_score below.
    """
    reference_date = events["datetime"].max()
    age_days = (reference_date - events["datetime"]).dt.total_seconds() / 86400
    decay = 0.5 ** (age_days / half_life_days)
    event_weight = events["event"].astype(str).map(EVENT_WEIGHTS).fillna(0.0)
    return event_weight * decay


# ---------------------------------------------------------------------------
# User features
# ---------------------------------------------------------------------------
@timeit
def build_user_features(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    item_categories: pd.DataFrame,
) -> pd.DataFrame:
    """One row per visitorid with the features listed in the module docstring."""
    events = events.merge(item_categories[["itemid", "categoryid"]], on="itemid", how="left")

    pivot = (
        events.pivot_table(index="visitorid", columns="event", aggfunc="size", fill_value=0)
    )
    for col in ["view", "addtocart", "transaction"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.rename(
        columns={"view": "user_view_count", "addtocart": "user_addtocart_count", "transaction": "user_purchase_count"}
    )[["user_view_count", "user_addtocart_count", "user_purchase_count"]]

    breadth = events.groupby("visitorid").agg(
        unique_items_viewed=("itemid", "nunique"),
        unique_categories_viewed=("categoryid", "nunique"),
        first_seen=("datetime", "min"),
        last_seen=("datetime", "max"),
    )

    session_stats = sessions.groupby("visitorid").agg(
        n_sessions=("session_id", "nunique"),
        avg_session_duration_sec=("duration_sec", "mean"),
        median_session_duration_sec=("duration_sec", "median"),
    )

    weighted = events.assign(_w=_time_decayed_weight(events, MODEL_CONFIG.popularity_half_life_days))
    activity_raw = weighted.groupby("visitorid")["_w"].sum().rename("activity_score_raw")

    features = pivot.join(breadth, how="outer").join(session_stats, how="outer").join(activity_raw, how="outer")
    features = features.fillna(
        {"user_view_count": 0, "user_addtocart_count": 0, "user_purchase_count": 0, "activity_score_raw": 0}
    )

    reference_date = events["datetime"].max()
    features["recency_days"] = (reference_date - features["last_seen"]).dt.total_seconds() / 86400

    features["conversion_rate_user"] = (
        features["user_purchase_count"] / features["user_view_count"].replace(0, np.nan)
    ).fillna(0.0)
    features["click_to_purchase_ratio_user"] = (
        features["user_purchase_count"] / features["user_addtocart_count"].replace(0, np.nan)
    ).fillna(0.0)

    features["user_activity_score"] = winsorize_and_scale(features["activity_score_raw"])

    features = features.reset_index()
    logger.info("build_user_features: %d users, %d features", len(features), features.shape[1] - 1)
    return features


# ---------------------------------------------------------------------------
# Category features (built first — item features consume category_popularity)
# ---------------------------------------------------------------------------
@timeit
def build_category_features(events: pd.DataFrame, item_categories: pd.DataFrame) -> pd.DataFrame:
    """One row per categoryid: engagement volumes + a normalized popularity score."""
    events = events.merge(item_categories[["itemid", "categoryid"]], on="itemid", how="left")
    events = events.dropna(subset=["categoryid"])

    pivot = events.pivot_table(index="categoryid", columns="event", aggfunc="size", fill_value=0)
    for col in ["view", "addtocart", "transaction"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.rename(
        columns={
            "view": "category_view_count",
            "addtocart": "category_addtocart_count",
            "transaction": "category_transaction_count",
        }
    )[["category_view_count", "category_addtocart_count", "category_transaction_count"]]

    pivot["total_interactions"] = pivot.sum(axis=1)
    num_items = item_categories.groupby("categoryid")["itemid"].nunique().rename("num_items_in_category")

    features = pivot.join(num_items, how="left").fillna({"num_items_in_category": 0})
    features["category_popularity_score"] = winsorize_and_scale(features["total_interactions"])
    features = features.reset_index()

    logger.info("build_category_features: %d categories", len(features))
    return features


# ---------------------------------------------------------------------------
# Item features
# ---------------------------------------------------------------------------
@timeit
def build_item_features(
    events: pd.DataFrame,
    item_attributes: pd.DataFrame,
    category_features: pd.DataFrame,
) -> pd.DataFrame:
    """One row per itemid with the features listed in the module docstring."""
    pivot = events.pivot_table(index="itemid", columns="event", aggfunc="size", fill_value=0)
    for col in ["view", "addtocart", "transaction"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.rename(
        columns={"view": "item_view_count", "addtocart": "item_addtocart_count", "transaction": "item_transaction_count"}
    )[["item_view_count", "item_addtocart_count", "item_transaction_count"]]

    activity = events.groupby("itemid").agg(
        unique_visitors=("visitorid", "nunique"),
        last_seen=("datetime", "max"),
    )

    weighted = events.assign(_w=_time_decayed_weight(events, MODEL_CONFIG.popularity_half_life_days))
    interaction_raw = weighted.groupby("itemid")["_w"].sum().rename("interaction_score_raw")

    # Start from the full item catalog (item_attributes), not just items with
    # events — items with ZERO interactions are exactly the cold-start items
    # this project needs to handle explicitly, not silently drop.
    features = (
        item_attributes.set_index("itemid")
        .join(pivot, how="left")
        .join(activity, how="left")
        .join(interaction_raw, how="left")
    )
    features[["item_view_count", "item_addtocart_count", "item_transaction_count"]] = features[
        ["item_view_count", "item_addtocart_count", "item_transaction_count"]
    ].fillna(0)
    features["unique_visitors"] = features["unique_visitors"].fillna(0)
    features["interaction_score_raw"] = features["interaction_score_raw"].fillna(0)
    features["is_cold_start"] = features["item_view_count"] + features["item_addtocart_count"] + features["item_transaction_count"] == 0

    reference_date = events["datetime"].max()
    features["recency_days"] = (reference_date - features["last_seen"]).dt.total_seconds() / 86400
    features["recency_days"] = features["recency_days"].fillna(np.inf)  # never interacted with = infinitely stale

    features["product_conversion_rate"] = (
        features["item_transaction_count"] / features["item_view_count"].replace(0, np.nan)
    ).fillna(0.0)
    features["click_to_purchase_ratio_item"] = (
        features["item_transaction_count"] / features["item_addtocart_count"].replace(0, np.nan)
    ).fillna(0.0)

    features["product_interaction_score"] = winsorize_and_scale(features["interaction_score_raw"])
    features["popularity_rank"] = features["item_view_count"].rank(method="min", ascending=False).astype(int)

    features = features.reset_index().merge(
        category_features[["categoryid", "category_popularity_score"]], on="categoryid", how="left"
    )
    features["category_popularity_score"] = features["category_popularity_score"].fillna(0.0)

    logger.info(
        "build_item_features: %d items (%d cold-start / zero interactions)",
        len(features), int(features["is_cold_start"].sum()),
    )
    return features


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@timeit
def run_feature_engineering_pipeline(persist: bool = True) -> dict[str, pd.DataFrame]:
    """Load Phase 2 preprocessing outputs and build all three feature tables."""
    events = pd.read_parquet(PROCESSED_FILES["events_clean"])
    sessions = pd.read_parquet(PROCESSED_FILES["sessions"])
    item_attributes = pd.read_parquet(PROCESSED_FILES["item_properties_clean"])

    item_categories = item_attributes[["itemid", "categoryid"]]

    category_features = build_category_features(events, item_categories)
    user_features = build_user_features(events, sessions, item_categories)
    item_features = build_item_features(events, item_attributes, category_features)

    outputs = {
        "user_features": user_features,
        "item_features": item_features,
        "category_features": category_features,
    }

    if persist:
        for key, df in outputs.items():
            path = PROCESSED_FILES[key]
            df.to_parquet(path, index=False)
            logger.info("Persisted %s -> %s (%d rows, %d cols)", key, path, len(df), df.shape[1])

    return outputs


if __name__ == "__main__":
    run_feature_engineering_pipeline(persist=True)

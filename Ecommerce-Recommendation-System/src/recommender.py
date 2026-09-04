"""
recommender.py
===============
Four recommendation strategies, built directly on top of the Phase 2
feature tables (`user_features`, `item_features`, `category_features`) and
the cleaned, ID-encoded event log (`events_clean`).

    1. PopularityRecommender          — non-personalized baseline
    2. ContentBasedRecommender        — TF-IDF + cosine similarity over
                                         item attributes
    3. CollaborativeFilteringRecommender — user-based, item-based, and SVD
                                         matrix factorization
    4. HybridRecommender              — confidence-weighted blend of all three

Each class documents its own advantages/disadvantages inline (see class
docstrings) — the short version is that no single strategy above is
sufficient on its own for this dataset: Phase 1's EDA showed 36% of the
catalog has zero interactions (kills collaborative filtering alone) and a
severe power-law popularity distribution (kills popularity-only ranking as
a *personalization* strategy, even though it's a fine fallback). This is
exactly why the hybrid model exists, not because "hybrid" sounds more
sophisticated on a resume.

A note on content-based similarity and this specific dataset
--------------------------------------------------------------
Retailrocket's item_properties file does not include a labeled "brand"
field or free-text product descriptions (unlike, say, an Amazon or
Flipkart product catalog) — every property beyond `categoryid` and
`available` is an anonymized, undocumented hash. This project is honest
about that constraint rather than inventing a fake brand/description field:

  - "Category Similarity" is implemented directly (category ancestor-path
    overlap) — this dataset genuinely supports it well via category_tree.csv.
  - "Description Similarity" is implemented as TF-IDF + cosine similarity
    over a *token document* built from category-path tokens and a
    price-bucket token — the same algorithmic technique used for free-text
    descriptions, applied to the structured attributes this dataset
    actually provides.
  - "Brand Similarity" is architecturally supported (ContentBasedRecommender
    accepts an optional `brand_column`) but left unpopulated for this
    dataset, and clearly logged as such at fit time — if a real brand field
    is later identified among Retailrocket's undocumented hashed
    properties (or added for a different retailer's data), no code change
    is required, only passing that column name in.
"""

from __future__ import annotations

import itertools
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.config import EVENT_WEIGHTS, MODEL_CONFIG, MODELS_DIR, PROCESSED_FILES
from src.utils import get_logger, timeit

logger = get_logger(__name__)


# ===========================================================================
# 1. Popularity-Based Recommender
# ===========================================================================
class PopularityRecommender:
    """Non-personalized ranking by (recency-weighted) popularity.

    Pros
    ----
    - Zero cold-start problem: works for a brand-new anonymous visitor with
      no history at all, which is exactly the visitor segment collaborative
      filtering cannot serve (Phase 1 EDA: a large share of visitors have
      exactly one event).
    - Cheap, fast, and trivially explainable ("this is what's popular").
    - An essential *fallback/safety-net* inside the hybrid model.

    Cons
    ----
    - Zero personalization — every user in the same segment/category sees
      the same list.
    - Actively reinforces popularity bias: recommending what's already
      popular tends to make it more popular still (a feedback loop),
      starving the long tail of exposure — this is a real cost, tracked
      explicitly via the catalog-coverage metric in Phase 4.
    """

    def __init__(self, item_features: pd.DataFrame, events: pd.DataFrame):
        self.item_features = item_features
        # Only itemid/datetime are needed for recently_popular(); keeping the
        # full event log (with visitorid, session_id, etc.) on this object
        # would needlessly bloat the pickled model size.
        self.events = events[["itemid", "datetime"]].copy()
        self._ranked_overall: Optional[pd.DataFrame] = None

    @timeit
    def fit(self) -> "PopularityRecommender":
        self._ranked_overall = self.item_features.sort_values(
            "product_interaction_score", ascending=False
        ).reset_index(drop=True)
        return self

    def trending(self, n: int = MODEL_CONFIG.top_k, category_id: Optional[int] = None,
                 exclude_items: Optional[set] = None) -> pd.DataFrame:
        """'Trending Products' — ranked by the recency-decayed interaction
        score from Phase 2 (recent transactions/carts/views count more than
        old ones), NOT a static all-time count.
        """
        df = self._ranked_overall
        if category_id is not None:
            df = df[df["categoryid"] == category_id]
        if exclude_items:
            df = df[~df["itemid"].isin(exclude_items)]
        return df.head(n)[
            ["itemid", "categoryid", "product_interaction_score", "item_view_count", "product_conversion_rate"]
        ]

    def best_sellers(self, n: int = MODEL_CONFIG.top_k, category_id: Optional[int] = None) -> pd.DataFrame:
        """'Best Sellers' — ranked by raw all-time transaction count, i.e.
        actual purchases rather than the blended, recency-weighted score.
        Deliberately a different signal from `trending()` (Phase 1 EDA
        showed "popular to browse" and "popular to buy" are not the same
        products).
        """
        df = self.item_features
        if category_id is not None:
            df = df[df["categoryid"] == category_id]
        return (
            df.sort_values("item_transaction_count", ascending=False)
            .head(n)[["itemid", "categoryid", "item_transaction_count", "product_conversion_rate"]]
        )

    def recently_popular(self, n: int = MODEL_CONFIG.top_k, days: int = 7,
                          category_id: Optional[int] = None) -> pd.DataFrame:
        """'Recently Popular' — recomputed from only the last `days` of raw
        events, a short, hard window rather than an exponential decay. This
        is the right signal for "what's hot *right now*" widgets, distinct
        from the longer-memory `trending()` score above.
        """
        cutoff = self.events["datetime"].max() - pd.Timedelta(days=days)
        recent = self.events[self.events["datetime"] >= cutoff]
        counts = recent.groupby("itemid").size().rename("recent_event_count").reset_index()
        df = counts.merge(self.item_features[["itemid", "categoryid"]], on="itemid", how="left")
        if category_id is not None:
            df = df[df["categoryid"] == category_id]
        return df.sort_values("recent_event_count", ascending=False).head(n)


# ===========================================================================
# 2. Content-Based Filtering
# ===========================================================================
class ContentBasedRecommender:
    """TF-IDF + cosine similarity over item attributes (category path,
    price bucket, and optionally brand — see module docstring).

    Pros
    ----
    - No cold-start problem for new ITEMS: as long as a product has a
      category and price, it can be recommended/compared immediately —
      unlike collaborative filtering, which needs interaction history.
    - Recommendations are naturally explainable ("similar category and
      price range") — directly feeds the "recommendation explanation"
      business feature.
    - Independent of any other user's behavior, so it can't be manipulated
      by fake interactions the way collaborative filtering can.

    Cons
    ----
    - Still has a cold-start problem for new USERS with no interaction
      history to build a content profile from.
    - Limited serendipity: tends to recommend "more of the same" rather
      than surfacing a genuinely novel but relevant item a collaborative
      signal might catch.
    - Quality is capped by how much attribute data actually exists — this
      dataset's real limitation, discussed candidly in the module docstring.
    """

    def __init__(self, item_features: pd.DataFrame, category_tree: pd.DataFrame,
                 price_bins: int = 5, brand_column: Optional[str] = None,
                 max_features: int = MODEL_CONFIG.tfidf_max_features):
        self.item_features = item_features.reset_index(drop=True)
        self.category_tree = category_tree
        self.price_bins = price_bins
        self.brand_column = brand_column
        self.max_features = max_features

        self._parent_map = dict(zip(category_tree["categoryid"], category_tree["parentid"]))
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._nn: Optional[NearestNeighbors] = None
        self._itemid_to_row: dict[int, int] = {}
        self._row_to_itemid: dict[int, int] = {}

    def _category_path_tokens(self, categoryid: float) -> list[str]:
        """Walk the category tree from `categoryid` up to its root,
        returning one token per ancestor (including itself). Two items in
        sibling leaf categories under the same parent will still share the
        parent-level token — this is what makes the similarity "category
        family" aware rather than only exact-category aware.
        """
        tokens: list[str] = []
        current = categoryid
        seen = set()
        while pd.notna(current) and current not in seen:
            tokens.append(f"cat_{int(current)}")
            seen.add(current)
            current = self._parent_map.get(current, np.nan)
        return tokens

    def _build_documents(self) -> pd.Series:
        df = self.item_features
        price_token = pd.Series("price_unknown", index=df.index, dtype=object)
        has_price = df["has_price"] if "has_price" in df.columns else df["price"].notna()
        if has_price.any():
            bucket_labels = [f"price_q{i+1}" for i in range(self.price_bins)]
            buckets = pd.qcut(df.loc[has_price, "price"], q=self.price_bins, labels=bucket_labels, duplicates="drop")
            price_token.loc[has_price] = buckets.astype(str)

        if self.brand_column and self.brand_column in df.columns and df[self.brand_column].notna().any():
            brand_token = "brand_" + df[self.brand_column].astype(str)
            logger.info("ContentBasedRecommender: using brand column '%s' for brand similarity", self.brand_column)
        else:
            brand_token = pd.Series("", index=df.index, dtype=object)
            logger.info(
                "ContentBasedRecommender: no usable brand column found (Retailrocket does not label one) — "
                "brand similarity is architecturally supported but inactive for this dataset."
            )

        docs = []
        for idx, row in df.iterrows():
            tokens = self._category_path_tokens(row["categoryid"])
            tokens.append(price_token.loc[idx])
            if brand_token.loc[idx]:
                tokens.append(brand_token.loc[idx])
            docs.append(" ".join(tokens))
        return pd.Series(docs, index=df.index)

    @timeit
    def fit(self) -> "ContentBasedRecommender":
        documents = self._build_documents()
        self._vectorizer = TfidfVectorizer(max_features=self.max_features, token_pattern=r"(?u)\b\w+\b")
        self._matrix = self._vectorizer.fit_transform(documents)

        self._itemid_to_row = {itemid: row for row, itemid in enumerate(self.item_features["itemid"])}
        self._row_to_itemid = {row: itemid for itemid, row in self._itemid_to_row.items()}

        n_neighbors = min(50, self._matrix.shape[0])
        self._nn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors, algorithm="brute")
        self._nn.fit(self._matrix)
        logger.info(
            "ContentBasedRecommender: fit TF-IDF over %d items, vocabulary size %d",
            self._matrix.shape[0], self._matrix.shape[1],
        )
        return self

    def similar_items(self, item_id: int, n: int = MODEL_CONFIG.top_k) -> pd.DataFrame:
        """'Similar Products' / 'Customers Also Viewed'-style content
        similarity — the closest items in TF-IDF/cosine space, excluding
        the query item itself.
        """
        row = self._itemid_to_row.get(item_id)
        if row is None:
            raise KeyError(f"item_id {item_id} not found in content-based item index")

        distances, indices = self._nn.kneighbors(self._matrix[row], n_neighbors=min(n + 1, self._matrix.shape[0]))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            candidate_id = self._row_to_itemid[idx]
            if candidate_id == item_id:
                continue
            results.append({"itemid": candidate_id, "content_similarity": 1 - dist})
        return pd.DataFrame(results).head(n)

    def recommend_for_user(self, item_ids: list[int], weights: Optional[list[float]] = None,
                            n: int = MODEL_CONFIG.top_k, exclude_items: Optional[set] = None) -> list[tuple[int, float]]:
        """Standard "content-based user profile" approach: build one profile
        vector as the (optionally weighted) average of the TF-IDF vectors of
        every item in `item_ids`, then rank ALL catalog items by cosine
        similarity to that profile.

        This exists so ContentBasedRecommender can be evaluated as a full
        personalized recommender in Phase 4 (given a user's history, rank
        the whole catalog), not only as an item-to-item "similar products"
        feature the way `similar_items()` is used on a product page.
        """
        rows = [self._itemid_to_row[i] for i in item_ids if i in self._itemid_to_row]
        if not rows:
            return []
        if weights is None:
            weights = [1.0] * len(rows)
        else:
            weights = [w for i, w in zip(item_ids, weights) if i in self._itemid_to_row]

        profile = np.asarray(self._matrix[rows].multiply(np.array(weights)[:, None]).mean(axis=0))
        profile_sparse = sp.csr_matrix(profile)

        sims = cosine_similarity(profile_sparse, self._matrix).ravel()

        exclude_rows = {self._itemid_to_row[i] for i in (exclude_items or set()) if i in self._itemid_to_row}
        exclude_rows |= set(rows)  # never re-recommend items already in the profile itself

        ranked_rows = np.argsort(-sims)
        results = []
        for row in ranked_rows:
            if row in exclude_rows:
                continue
            results.append((self._row_to_itemid[row], float(sims[row])))
            if len(results) >= n:
                break
        return results

    def category_similarity(self, item_id_a: int, item_id_b: int) -> float:
        """Explicit 'Category Similarity' sub-feature: Jaccard overlap of
        the two items' category ancestor-path token sets, independent of
        the full TF-IDF pipeline (useful as a fast, interpretable
        standalone signal, e.g. for the recommendation-explanation text).
        """
        cat_a = self.item_features.loc[self.item_features["itemid"] == item_id_a, "categoryid"].iloc[0]
        cat_b = self.item_features.loc[self.item_features["itemid"] == item_id_b, "categoryid"].iloc[0]
        set_a, set_b = set(self._category_path_tokens(cat_a)), set(self._category_path_tokens(cat_b))
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def explain(self, item_id: int, similar_item_id: int) -> str:
        """Human-readable reason for a content-based recommendation —
        feeds the 'recommendation explanation' business feature (Phase 5).
        """
        cat_sim = self.category_similarity(item_id, similar_item_id)
        row_a = self.item_features.loc[self.item_features["itemid"] == item_id].iloc[0]
        row_b = self.item_features.loc[self.item_features["itemid"] == similar_item_id].iloc[0]
        if cat_sim >= 0.99:
            reason = "same category"
        elif cat_sim > 0:
            reason = "related category"
        else:
            reason = "similar attributes"
        if row_a.get("has_price") and row_b.get("has_price"):
            price_ratio = min(row_a["price"], row_b["price"]) / max(row_a["price"], row_b["price"])
            if price_ratio > 0.7:
                reason += " and similar price range"
        return f"Recommended because it has {reason} to an item you viewed."


# ===========================================================================
# 3. Collaborative Filtering
# ===========================================================================
class CollaborativeFilteringRecommender:
    """User-based, item-based, and SVD matrix-factorization collaborative
    filtering over an implicit-feedback interaction matrix weighted by
    EVENT_WEIGHTS (view=1, addtocart=3, transaction=5) — a purchase moves
    the needle far more than a view, rather than treating every event as
    equally informative.

    Pros
    ----
    - Captures genuine collective behavior invisible to content attributes
      (e.g. "people who buy X also buy Y" even across unrelated categories).
    - Improves as more interaction data accumulates — the only strategy of
      the four whose quality compounds with scale/time.

    Cons
    ----
    - Severe cold start: Phase 2 found 36% of this catalog has zero
      interactions, and any brand-new user has none either — CF literally
      cannot rank these; excluded users/items are tracked explicitly
      (`self.excluded_users`, `self.excluded_items`) so the hybrid model
      knows exactly when to fall back.
    - Heavier to compute and refresh at scale than the other two strategies.
    - Can still concentrate on already-popular items unless explicitly
      counterbalanced (this project measures that via Phase 4's diversity
      and coverage metrics rather than assuming it away).
    """

    def __init__(self, events: pd.DataFrame, n_users: int, n_items: int,
                 min_interactions_per_user: int = MODEL_CONFIG.min_interactions_per_user,
                 min_interactions_per_item: int = MODEL_CONFIG.min_interactions_per_item,
                 n_factors: int = MODEL_CONFIG.svd_n_factors,
                 random_state: int = MODEL_CONFIG.svd_random_state):
        self.events = events
        self.n_users = n_users
        self.n_items = n_items
        self.min_interactions_per_user = min_interactions_per_user
        self.min_interactions_per_item = min_interactions_per_item
        self.n_factors = n_factors
        self.random_state = random_state

        self.interaction_matrix: Optional[sp.csr_matrix] = None
        self.excluded_users: set[int] = set()
        self.excluded_items: set[int] = set()
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self._item_nn: Optional[NearestNeighbors] = None
        self._user_nn: Optional[NearestNeighbors] = None
        self._item_vectors: Optional[sp.csr_matrix] = None
        self._user_vectors: Optional[sp.csr_matrix] = None

    @timeit
    def _build_interaction_matrix(self) -> sp.csr_matrix:
        weights = self.events["event"].astype(str).map(EVENT_WEIGHTS).fillna(0.0).to_numpy()
        rows = self.events["user_idx"].to_numpy()
        cols = self.events["item_idx"].to_numpy()

        mat = sp.coo_matrix((weights, (rows, cols)), shape=(self.n_users, self.n_items)).tocsr()
        mat.sum_duplicates()  # multiple events between the same user/item add up, as intended

        user_activity = np.asarray(mat.sum(axis=1)).ravel()
        item_activity = np.asarray(mat.sum(axis=0)).ravel()
        self.excluded_users = set(np.where(user_activity < self.min_interactions_per_user)[0].tolist())
        self.excluded_items = set(np.where(item_activity < self.min_interactions_per_item)[0].tolist())

        logger.info(
            "CF interaction matrix: %d users x %d items, %d nonzero entries | "
            "%d users and %d items below the min-interaction threshold (excluded from CF, "
            "routed to hybrid fallback instead)",
            self.n_users, self.n_items, mat.nnz, len(self.excluded_users), len(self.excluded_items),
        )
        return mat

    @timeit
    def fit(self) -> "CollaborativeFilteringRecommender":
        self.interaction_matrix = self._build_interaction_matrix()

        # --- Matrix factorization (SVD) ---
        svd = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        self.user_factors = svd.fit_transform(self.interaction_matrix)
        self.item_factors = svd.components_.T
        logger.info(
            "SVD: %d factors explain %.1f%% of variance",
            self.n_factors, 100 * svd.explained_variance_ratio_.sum(),
        )

        # --- Item-based CF: nearest neighbors over item columns ---
        # Normalized vectors are computed ONCE here and cached, not
        # recomputed inside every recommend_*() call — this matters a lot
        # once Phase 4 evaluation calls these methods for hundreds of users.
        self._item_vectors = normalize(self.interaction_matrix.T, norm="l2", axis=1)
        self._item_nn = NearestNeighbors(metric="cosine", n_neighbors=min(50, self.n_items), algorithm="brute")
        self._item_nn.fit(self._item_vectors)

        # --- User-based CF: nearest neighbors over user rows ---
        self._user_vectors = normalize(self.interaction_matrix, norm="l2", axis=1)
        self._user_nn = NearestNeighbors(metric="cosine", n_neighbors=min(50, self.n_users), algorithm="brute")
        self._user_nn.fit(self._user_vectors)

        return self

    def recommend_item_based(self, user_idx: int, n: int = MODEL_CONFIG.top_k) -> list[tuple[int, float]]:
        """For each item the user has interacted with, find its nearest
        neighbor items (co-interaction pattern) and aggregate, weighted by
        how strongly the user engaged with the seed item.
        """
        if user_idx in self.excluded_users or user_idx >= self.n_users:
            return []
        user_row = self.interaction_matrix[user_idx]
        seed_items = user_row.indices
        if len(seed_items) == 0:
            return []

        scores: dict[int, float] = {}
        for item_idx, weight in zip(seed_items, user_row.data):
            distances, indices = self._item_nn.kneighbors(self._item_vectors[item_idx], n_neighbors=min(n + 5, self.n_items))
            for dist, neighbor_idx in zip(distances[0], indices[0]):
                if neighbor_idx == item_idx or neighbor_idx in set(seed_items):
                    continue
                scores[neighbor_idx] = scores.get(neighbor_idx, 0.0) + weight * (1 - dist)

        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def recommend_user_based(self, user_idx: int, n: int = MODEL_CONFIG.top_k, k_neighbors: int = 20) -> list[tuple[int, float]]:
        """Find the k most similar users (by interaction pattern) and
        recommend items they engaged with, weighted by user similarity and
        excluding items the target user has already interacted with.
        """
        if user_idx in self.excluded_users or user_idx >= self.n_users:
            return []
        distances, indices = self._user_nn.kneighbors(self._user_vectors[user_idx], n_neighbors=min(k_neighbors + 1, self.n_users))

        already_seen = set(self.interaction_matrix[user_idx].indices)
        scores: dict[int, float] = {}
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            if neighbor_idx == user_idx:
                continue
            similarity = 1 - dist
            if similarity <= 0:
                continue
            neighbor_row = self.interaction_matrix[neighbor_idx]
            for item_idx, weight in zip(neighbor_row.indices, neighbor_row.data):
                if item_idx in already_seen:
                    continue
                scores[item_idx] = scores.get(item_idx, 0.0) + similarity * weight

        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def recommend_svd(self, user_idx: int, n: int = MODEL_CONFIG.top_k) -> list[tuple[int, float]]:
        """Rank all items by the SVD-reconstructed user·item score,
        excluding items already interacted with.
        """
        if user_idx in self.excluded_users or user_idx >= self.n_users:
            return []
        scores = self.user_factors[user_idx] @ self.item_factors.T
        already_seen = set(self.interaction_matrix[user_idx].indices)
        ranked_idx = np.argsort(-scores)
        results = []
        for item_idx in ranked_idx:
            if item_idx in already_seen:
                continue
            results.append((int(item_idx), float(scores[item_idx])))
            if len(results) >= n:
                break
        return results

    def similar_items(self, item_idx: int, n: int = MODEL_CONFIG.top_k) -> list[tuple[int, float]]:
        """Items most similar to `item_idx` by co-interaction pattern —
        nearest neighbors in the item-vector space built from the
        interaction matrix. This is the BEHAVIORAL analogue of
        `ContentBasedRecommender.similar_items()`: it powers "Customers
        Also Viewed" from actual co-view/co-purchase data, whereas the
        content-based version is driven by category/price attributes —
        two genuinely different signals that can (and often do) disagree,
        which is exactly why the app surfaces both.
        """
        if item_idx >= self.n_items or item_idx in self.excluded_items:
            return []
        distances, indices = self._item_nn.kneighbors(self._item_vectors[item_idx], n_neighbors=min(n + 1, self.n_items))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == item_idx:
                continue
            results.append((int(idx), float(1 - dist)))
        return results[:n]


# ===========================================================================
# 4. Frequently Bought Together (market-basket / co-purchase)
# ===========================================================================
class FrequentlyBoughtTogetherRecommender:
    """"Frequently Bought Together" via market-basket analysis over actual
    co-PURCHASES — the one recommender in this project built from
    `transactionid`, rather than from view/cart co-interaction (item-based
    CF) or item attributes (content-based). Multiple event rows sharing the
    same `transactionid` are items bought in the same checkout; this is the
    literal record of "goes well together," not a proxy for it.

    Pros
    ----
    - The highest-precision, most business-intuitive signal available:
      "customers who bought X also bought Y" is a directly observed fact,
      not an inferred similarity.
    - Trivially explainable — no algorithm to justify to a stakeholder.

    Cons
    ----
    - Requires an actual multi-item TRANSACTION to produce a signal at all;
      single-item checkouts contribute nothing, and Phase 1 found
      transactions are already the rarest event type (~4% of all events).
    - As a result, coverage is inherently sparse for a catalog this size —
      most items will have few or zero recorded co-purchase partners,
      reported explicitly via `coverage()` rather than assumed away.
    """

    def __init__(self, events: pd.DataFrame):
        self.events = events
        self._pair_counts: dict[int, dict[int, int]] = {}
        self._item_basket_counts: dict[int, int] = {}
        self._n_multi_item_baskets = 0

    @timeit
    def fit(self) -> "FrequentlyBoughtTogetherRecommender":
        transactions = self.events[self.events["event"] == "transaction"].dropna(subset=["transactionid"])
        baskets = transactions.groupby("transactionid")["itemid"].apply(lambda s: sorted(set(s.astype(int))))

        for basket in baskets:
            for item in basket:
                self._item_basket_counts[item] = self._item_basket_counts.get(item, 0) + 1
            if len(basket) < 2:
                continue  # a single-item checkout has no "together" signal
            self._n_multi_item_baskets += 1
            for item_a, item_b in itertools.permutations(basket, 2):
                self._pair_counts.setdefault(item_a, {})
                self._pair_counts[item_a][item_b] = self._pair_counts[item_a].get(item_b, 0) + 1

        logger.info(
            "FrequentlyBoughtTogetherRecommender: %d transactions -> %d baskets (%d with >=2 items), "
            "%d items have at least one co-purchase partner",
            len(transactions), len(baskets), self._n_multi_item_baskets, len(self._pair_counts),
        )
        return self

    def get_fbt(self, item_id: int, n: int = MODEL_CONFIG.top_k, min_support: int = 1) -> list[tuple[int, int, float]]:
        """Top-n items frequently co-purchased with `item_id`, ranked by
        confidence = P(B bought | A bought) = count(A,B) / count(A) — the
        standard association-rule metric, not raw co-occurrence count
        alone (which would just favor generically popular items regardless
        of any real association with A).

        Returns a list of (item_id, co_purchase_count, confidence) tuples.
        """
        neighbors = self._pair_counts.get(item_id, {})
        item_total = self._item_basket_counts.get(item_id, 0)
        if not neighbors or item_total == 0:
            return []

        scored = [
            (other_item, count, count / item_total)
            for other_item, count in neighbors.items()
            if count >= min_support
        ]
        scored.sort(key=lambda t: t[2], reverse=True)
        return scored[:n]

    def coverage(self) -> float:
        """Fraction of items that appear in at least one multi-item basket
        (i.e. have SOME co-purchase signal) — reported explicitly per the
        "Cons" above, rather than silently returning an empty list for the
        majority of a catalog without saying so.
        """
        if not self._item_basket_counts:
            return 0.0
        return len(self._pair_counts) / len(self._item_basket_counts)


# ===========================================================================
# 5. Hybrid Recommender
# ===========================================================================
class HybridRecommender:
    """Confidence-weighted blend of collaborative filtering (item-based by
    default — see the `cf_method` note below), content-based similarity, and popularity.

    The blend weight on collaborative filtering scales up smoothly with how
    much interaction history a user has (via Phase 2's `user_activity_score`)
    rather than hard-switching between strategies at an arbitrary
    threshold — a user with 4 events isn't meaningfully different from one
    with 6, so a hard cutoff at "5 interactions = now use CF only" would
    produce a visible, unjustified jump in recommendation character.
    Popularity always keeps a small constant weight, even for highly active
    users, as a deliberate anti-popularity-bias / diversity safety valve
    (a pure top-CF-score list tends to collapse onto the same handful of
    items — see Phase 4's coverage metric).

    Pros
    ----
    - Closes collaborative filtering's cold-start gap with content-based
      and popularity fallbacks, while still capturing CF's collective-
      behavior signal once a user has enough history.
    - One consistent API regardless of a user's history depth.

    Cons
    ----
    - Three models to maintain, retrain, and keep in sync instead of one.
    - Harder to explain a specific recommendation to a stakeholder — "why
      this item" now has up to three contributing reasons instead of one
      (mitigated here by `explain()`, but it's real added complexity).
    - Blend weights are a tuning surface of their own; poor weights can let
      one component silently dominate — evaluated explicitly in Phase 4
      rather than assumed correct.

    A note on `cf_method` (set from Phase 4 evaluation findings)
    ---------------------------------------------------------------
    Phase 4's evaluation found that plain `TruncatedSVD` on a raw
    implicit-feedback matrix tends to collapse onto the single dominant
    latent factor — which, empirically, is just overall item popularity
    (its top recommendations for an active user landed at global
    popularity ranks 120–250 out of 1,914 items, not a personalized
    pattern). This is a known, well-documented limitation of applying
    plain reconstruction-objective SVD to implicit feedback (it has no
    notion of "unobserved != negative," unlike a proper confidence-weighted
    ALS formulation — see the `implicit`/`scikit-surprise` note in
    requirements.txt). Item-based nearest-neighbor CF measurably
    outperformed it on this dataset (NDCG@10 0.0995 vs. 0.0111), so
    `cf_method` defaults to `"item_based"` here; `"svd"` and `"user_based"`
    remain available for comparison/education (see notebook 04).
    """

    def __init__(self, popularity_model: PopularityRecommender, content_model: ContentBasedRecommender,
                 cf_model: CollaborativeFilteringRecommender, user_features: pd.DataFrame,
                 item_id_map: pd.DataFrame, user_id_map: pd.DataFrame,
                 w_cf: float = 0.6, w_content: float = 0.25, w_popularity: float = 0.15,
                 cf_method: str = "item_based"):
        self.popularity_model = popularity_model
        self.content_model = content_model
        self.cf_model = cf_model
        self.user_features = user_features.set_index("visitorid")
        self.item_id_map = item_id_map
        self.user_id_map = user_id_map
        self.w_cf, self.w_content, self.w_popularity = w_cf, w_content, w_popularity
        if cf_method not in ("item_based", "user_based", "svd"):
            raise ValueError(f"cf_method must be 'item_based', 'user_based', or 'svd', got {cf_method!r}")
        self.cf_method = cf_method

        self._visitorid_to_idx = dict(zip(user_id_map["visitorid"], user_id_map["user_idx"]))
        self._itemid_to_idx = dict(zip(item_id_map["itemid"], item_id_map["item_idx"]))
        self._idx_to_itemid = dict(zip(item_id_map["item_idx"], item_id_map["itemid"]))

    def _cf_confidence(self, visitorid: int) -> float:
        """0..1 confidence that CF has enough signal for this user — reuses
        Phase 2's `user_activity_score`, so "how much do we trust CF" and
        "how engaged is this user" are the same underlying measurement,
        not two independently-tuned numbers that could drift apart.

        `user_activity_score` is heavily right-skewed by construction
        (Phase 2: winsorized + min-max scaled against a population where
        most users have only 1-2 total events — see Phase 1's long-tail
        finding). Used linearly, this compresses CF's weight too
        aggressively for exactly the moderately-active users we most want
        it to help: Phase 4 evaluation found the median confidence among
        users who actually completed a test-period purchase was only
        ~0.10, silently capping CF's contribution to ~6% of the blend for
        half of them even though CF alone was the strongest individual
        model. A square-root transform spreads out that compressed middle
        of the range while still keeping true near-zero-activity users
        near-zero confidence.
        """
        if visitorid not in self.user_features.index:
            return 0.0
        raw = float(self.user_features.loc[visitorid, "user_activity_score"])
        return raw ** 0.5

    def recommend(self, visitorid: int, n: int = MODEL_CONFIG.top_k,
                  seed_item_id: Optional[int] = None) -> pd.DataFrame:
        """Return the top-n blended recommendations for a visitor.

        `seed_item_id`, if given (e.g. the product page the visitor is
        currently on), also pulls in content-based "similar items" for
        that specific product — this is what powers a "Similar Products"
        rail on a live product page, as distinct from a homepage
        "Recommended for You" rail that only uses `visitorid`.
        """
        confidence = self._cf_confidence(visitorid)
        user_idx = self._visitorid_to_idx.get(visitorid)

        scores: dict[int, float] = {}

        if user_idx is not None:
            cf_method_fn = getattr(self.cf_model, f"recommend_{self.cf_method}")
            cf_candidates = cf_method_fn(user_idx, n=n * 3)
            if cf_candidates:
                # Normalize to [0, 1] before weighting — item-based/user-based CF
                # scores (cumulative similarity*weight sums) and SVD dot products
                # live on very different absolute scales from each other AND from
                # content similarity / popularity (both already 0-1-ish). Blending
                # raw scores would let whichever component happens to have the
                # largest numbers dominate regardless of w_cf/w_content/w_popularity.
                max_cf_score = max(score for _, score in cf_candidates) or 1.0
                for item_idx, raw_score in cf_candidates:
                    itemid = self._idx_to_itemid.get(item_idx)
                    if itemid is not None:
                        scores[itemid] = scores.get(itemid, 0.0) + self.w_cf * confidence * (raw_score / max_cf_score)

        content_candidates = pd.DataFrame()
        if seed_item_id is not None:
            content_candidates = self.content_model.similar_items(seed_item_id, n=n * 3)
        elif user_idx is not None and self.cf_model.interaction_matrix is not None and user_idx < self.cf_model.n_users:
            recent_items = self.cf_model.interaction_matrix[user_idx].indices
            if len(recent_items) > 0:
                last_item_id = self._idx_to_itemid.get(recent_items[-1])
                if last_item_id is not None:
                    content_candidates = self.content_model.similar_items(last_item_id, n=n * 3)

        for _, row in content_candidates.iterrows():
            scores[row["itemid"]] = scores.get(row["itemid"], 0.0) + self.w_content * row["content_similarity"]

        popularity_candidates = self.popularity_model.trending(n=n * 3)
        max_pop = popularity_candidates["product_interaction_score"].max() or 1.0
        for _, row in popularity_candidates.iterrows():
            norm_score = row["product_interaction_score"] / max_pop
            scores[row["itemid"]] = scores.get(row["itemid"], 0.0) + self.w_popularity * norm_score

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
        result = pd.DataFrame(ranked, columns=["itemid", "hybrid_score"])
        result["cf_confidence_used"] = confidence
        return result

    def explain(self, visitorid: int, item_id: int, seed_item_id: Optional[int] = None) -> str:
        """Human-readable reason combining whichever signal(s) contributed
        most to this item's score — the 'recommendation explanation'
        business feature, hybrid-aware version.
        """
        confidence = self._cf_confidence(visitorid)
        if seed_item_id is not None:
            return self.content_model.explain(seed_item_id, item_id)
        if confidence > 0.3:
            return "Recommended based on your browsing and purchase history."
        return "Recommended because it's popular among shoppers with similar interests."


# ===========================================================================
# Training orchestration — fits all models and persists them to models/
# ===========================================================================
@timeit
def train_and_save_all_models() -> dict[str, object]:
    """Fit the popularity, content-based, collaborative-filtering, and
    frequently-bought-together models from the Phase 2 processed tables,
    then persist each to `models/` via joblib so the Streamlit app
    (Phase 5) and evaluation harness (Phase 4) can load pre-fit models
    instead of refitting on every run.

    The hybrid model is intentionally NOT pickled on its own — it's a thin
    wrapper holding references to the other three, so it's cheaper and less
    error-prone to reconstruct it at load time (see `load_all_models`) than
    to serialize a second copy of the same underlying objects.
    """
    item_features = pd.read_parquet(PROCESSED_FILES["item_features"])
    category_tree = pd.read_parquet(PROCESSED_FILES["category_tree_clean"])
    events = pd.read_parquet(PROCESSED_FILES["events_clean"])
    user_id_map = pd.read_parquet(PROCESSED_FILES["user_id_map"])
    item_id_map = pd.read_parquet(PROCESSED_FILES["item_id_map"])

    popularity_model = PopularityRecommender(item_features, events).fit()
    content_model = ContentBasedRecommender(item_features, category_tree).fit()
    cf_model = CollaborativeFilteringRecommender(
        events, n_users=len(user_id_map), n_items=len(item_id_map)
    ).fit()
    fbt_model = FrequentlyBoughtTogetherRecommender(events).fit()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(popularity_model, MODELS_DIR / "popularity_model.joblib")
    joblib.dump(content_model, MODELS_DIR / "content_model.joblib")
    joblib.dump(cf_model, MODELS_DIR / "cf_model.joblib")
    joblib.dump(fbt_model, MODELS_DIR / "fbt_model.joblib")
    logger.info("Persisted popularity_model, content_model, cf_model, fbt_model to %s", MODELS_DIR)

    return {
        "popularity_model": popularity_model, "content_model": content_model,
        "cf_model": cf_model, "fbt_model": fbt_model,
    }


def load_all_models() -> dict[str, object]:
    """Load all persisted models from `models/`, reassemble the hybrid
    recommender around them, and return everything the Streamlit app needs
    in one dict — keys: 'popularity', 'content', 'cf', 'fbt', 'hybrid'.
    """
    popularity_model = joblib.load(MODELS_DIR / "popularity_model.joblib")
    content_model = joblib.load(MODELS_DIR / "content_model.joblib")
    cf_model = joblib.load(MODELS_DIR / "cf_model.joblib")
    fbt_model = joblib.load(MODELS_DIR / "fbt_model.joblib")

    user_features = pd.read_parquet(PROCESSED_FILES["user_features"])
    user_id_map = pd.read_parquet(PROCESSED_FILES["user_id_map"])
    item_id_map = pd.read_parquet(PROCESSED_FILES["item_id_map"])

    hybrid_model = HybridRecommender(popularity_model, content_model, cf_model, user_features, item_id_map, user_id_map)

    return {
        "popularity": popularity_model,
        "content": content_model,
        "cf": cf_model,
        "fbt": fbt_model,
        "hybrid": hybrid_model,
    }

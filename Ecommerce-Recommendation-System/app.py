"""
app.py
=======
Streamlit web application for the E-commerce Recommendation System.

"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MODEL_CONFIG, PROCESSED_FILES
from src.recommender import load_all_models
from src.visualization import funnel_chart, grouped_bar_chart, heatmap_chart, histogram_chart, horizontal_bar_chart

# ---------------------------------------------------------------------------
# Page config & light theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="E-commerce Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background-color: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 10px;
        padding: 14px 16px 8px 16px;
    }
    .product-card {
        border: 1px solid rgba(120,120,120,0.2);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .explanation-text {
        color: rgb(99, 102, 241);
        font-size: 0.85rem;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EVAL_RESULTS_PATH = PROCESSED_FILES["events_clean"].parent / "evaluation_results.csv"


# ---------------------------------------------------------------------------
# Cached data & model loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading recommendation models…")
def get_models() -> dict:
    return load_all_models()


@st.cache_data(show_spinner="Loading product catalog…")
def get_item_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_FILES["item_features"])


@st.cache_data(show_spinner=False)
def get_user_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_FILES["user_features"])


@st.cache_data(show_spinner=False)
def get_category_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_FILES["category_features"])


@st.cache_data(show_spinner=False)
def get_category_tree() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_FILES["category_tree_clean"])


@st.cache_data(show_spinner="Loading interaction history…")
def get_events() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_FILES["events_clean"])


@st.cache_data(show_spinner=False)
def get_id_maps() -> tuple[pd.DataFrame, pd.DataFrame]:
    user_id_map = pd.read_parquet(PROCESSED_FILES["user_id_map"])
    item_id_map = pd.read_parquet(PROCESSED_FILES["item_id_map"])
    return user_id_map, item_id_map


@st.cache_data(show_spinner=False)
def get_evaluation_results() -> pd.DataFrame | None:
    if EVAL_RESULTS_PATH.exists():
        return pd.read_csv(EVAL_RESULTS_PATH)
    return None


@st.cache_data(show_spinner=False)
def get_demo_visitors(_user_features: pd.DataFrame) -> dict[str, list[int]]:
    """A small, labeled set of real visitor IDs spanning the activity
    spectrum, so the Recommendations page can demo "what does this look
    like for an active shopper vs. a brand-new visitor" without the person
    needing to already know a specific ID.
    """
    uf = _user_features
    power_users = uf.sort_values("user_activity_score", ascending=False).head(5)["visitorid"].tolist()
    mid_band = uf[(uf["user_activity_score"] > 0.05) & (uf["user_activity_score"] < 0.25)]
    casual_users = mid_band.sample(min(5, len(mid_band)), random_state=42)["visitorid"].tolist()
    return {"Active shoppers": power_users, "Casual visitors": casual_users, "Brand-new visitor (cold start)": [999_999_999]}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def item_label(itemid: int, item_features: pd.DataFrame) -> str:
    row = item_features.loc[item_features["itemid"] == itemid]
    if row.empty:
        return f"Product #{itemid}"
    cat = row.iloc[0].get("categoryid")
    cat_str = f"Cat. {int(cat)}" if pd.notna(cat) else "Uncategorized"
    return f"Product #{itemid} · {cat_str}"


def price_display(price: float | None) -> str:
    """Retailrocket hashes price to anonymize the real scale — we label it
    a relative 'Price Index' rather than fabricating a currency amount.
    """
    if price is None or pd.isna(price):
        return "No price data"
    return f"Price Index {price:,.0f}"


def render_product_card(itemid: int, item_features: pd.DataFrame, score: float | None = None,
                         score_label: str = "Score", explanation: str | None = None) -> None:
    row = item_features.loc[item_features["itemid"] == itemid]
    if row.empty:
        st.warning(f"Product #{itemid} not found in catalog.")
        return
    row = row.iloc[0]
    with st.container():
        cols = st.columns([3, 2, 2, 2] if score is None else [3, 2, 2, 2, 2])
        cols[0].markdown(f"**Product #{itemid}**")
        cols[1].caption(f"Category {int(row['categoryid'])}" if pd.notna(row["categoryid"]) else "Uncategorized")
        cols[2].caption(price_display(row.get("price")))
        cols[3].caption("✅ In stock" if row.get("available", 1) == 1 else "⛔ Out of stock")
        if score is not None:
            cols[4].caption(f"{score_label}: {score:.3f}")
        if explanation:
            st.markdown(f"<span class='explanation-text'>{explanation}</span>", unsafe_allow_html=True)
        st.divider()


def add_recently_viewed(itemid: int) -> None:
    if "recently_viewed" not in st.session_state:
        st.session_state.recently_viewed = []
    if itemid in st.session_state.recently_viewed:
        st.session_state.recently_viewed.remove(itemid)
    st.session_state.recently_viewed.insert(0, itemid)
    st.session_state.recently_viewed = st.session_state.recently_viewed[:10]


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
def render_dashboard(models, item_features, user_features, category_features, events, eval_results):
    st.title("🏠 Dashboard")
    st.caption("Business overview — E-commerce Product Recommendation System")

    total_views = int((events["event"] == "view").sum())
    total_carts = int((events["event"] == "addtocart").sum())
    total_txns = int((events["event"] == "transaction").sum())
    conversion_rate = total_txns / total_views if total_views else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Products", f"{len(item_features):,}")
    c2.metric("Visitors", f"{len(user_features):,}")
    c3.metric("Total Events", f"{len(events):,}")
    c4.metric("Transactions", f"{total_txns:,}")
    c5.metric("View → Purchase Rate", f"{conversion_rate:.2%}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Conversion Funnel")
        fig = funnel_chart(["View", "Add to Cart", "Transaction"], [total_views, total_carts, total_txns])
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Top 10 Categories by Engagement")
        top_cats = category_features.sort_values("total_interactions", ascending=False).head(10).copy()
        top_cats["categoryid"] = top_cats["categoryid"].astype(int).astype(str)
        fig2 = horizontal_bar_chart(top_cats, "total_interactions", "categoryid", "Total events", "Category")
        st.plotly_chart(fig2, width='stretch')

    st.divider()
    st.subheader("Model Performance Snapshot")
    if eval_results is not None:
        display_cols = ["model", "ndcg@10", "precision@10", "catalog_coverage", "user_reach"]
        st.dataframe(
            eval_results[display_cols].set_index("model").style.format(
                {"ndcg@10": "{:.4f}", "precision@10": "{:.4f}", "catalog_coverage": "{:.1%}", "user_reach": "{:.1%}"}
            ),
            width='stretch',
        )
        st.caption(
            "Full methodology, the two bugs found and fixed during evaluation, and the accuracy-vs-coverage "
            "trade-off discussion are in the Analytics page and `notebooks/04_model_evaluation.ipynb`."
        )
    else:
        st.info("Run `python -m src.evaluation` to generate the model comparison table.")


# ---------------------------------------------------------------------------
# Page: Search Products
# ---------------------------------------------------------------------------
def render_search(models, item_features, category_features):
    st.title("🔍 Search Products")
    st.caption(
        "Retailrocket's catalog is fully anonymized — no product names, images, or brand labels — "
        "so search works by Product ID, category, price index, and availability."
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        item_id_query = st.text_input("Search by Product ID", placeholder="e.g. 1024")
    with col2:
        cat_ids = sorted(category_features["categoryid"].dropna().astype(int).unique().tolist())
        selected_cat = st.selectbox("Category filter", ["All"] + cat_ids)
    with col3:
        avail_only = st.checkbox("In stock only")

    price_series = item_features["price"].dropna()
    if len(price_series):
        p_min, p_max = float(price_series.min()), float(price_series.max())
        price_range = st.slider("Price Index range", p_min, p_max, (p_min, p_max))
    else:
        price_range = (0.0, 1.0)

    filtered = item_features.copy()
    if item_id_query.strip():
        try:
            qid = int(item_id_query.strip())
            filtered = filtered[filtered["itemid"] == qid]
        except ValueError:
            st.warning("Product ID must be a number.")
    if selected_cat != "All":
        filtered = filtered[filtered["categoryid"] == selected_cat]
    if avail_only:
        filtered = filtered[filtered["available"] == 1]
    filtered = filtered[
        filtered["price"].isna() | filtered["price"].between(price_range[0], price_range[1])
    ]

    st.write(f"**{len(filtered):,}** products match your filters")
    display_cols = ["itemid", "categoryid", "price", "available", "item_view_count",
                     "product_conversion_rate", "popularity_rank"]
    st.dataframe(
        filtered[display_cols].sort_values("item_view_count", ascending=False).head(100).rename(columns={
            "itemid": "Product ID", "categoryid": "Category", "price": "Price Index",
            "available": "In Stock", "item_view_count": "Views",
            "product_conversion_rate": "Conversion Rate", "popularity_rank": "Popularity Rank",
        }).style.format({"Conversion Rate": "{:.2%}"}),
        width='stretch', height=320,
    )

    st.divider()
    st.subheader("Product Details")
    valid_ids = item_features["itemid"].astype(int)
    default_id = int(filtered["itemid"].iloc[0]) if len(filtered) else int(valid_ids.iloc[0])
    selected_item = st.number_input("Enter a Product ID to inspect", min_value=int(valid_ids.min()),
                                     max_value=int(valid_ids.max()), value=default_id, step=1)

    if st.button("View Product", type="primary"):
        st.session_state.detail_item_id = int(selected_item)

    if st.session_state.get("detail_item_id") is not None:
        render_product_detail(st.session_state.detail_item_id, models, item_features)


def render_product_detail(itemid: int, models, item_features: pd.DataFrame) -> None:
    row = item_features.loc[item_features["itemid"] == itemid]
    if row.empty:
        st.error(f"Product #{itemid} does not exist in the catalog.")
        return
    row = row.iloc[0]
    add_recently_viewed(itemid)

    st.markdown(f"### Product #{itemid}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Category", f"{int(row['categoryid'])}" if pd.notna(row["categoryid"]) else "—")
    c2.metric("Price Index", f"{row['price']:.0f}" if pd.notna(row["price"]) else "—")
    c3.metric("Views", f"{int(row['item_view_count']):,}")
    c4.metric("Conversion Rate", f"{row['product_conversion_rate']:.2%}")

    tab1, tab2, tab3 = st.tabs(["🧩 Similar Products", "👀 Customers Also Viewed", "🛒 Frequently Bought Together"])

    with tab1:
        st.caption("Content-based: similar category, price range, and attributes.")
        try:
            similar = models["content"].similar_items(itemid, n=5)
            if similar.empty:
                st.info("No similar products found.")
            for _, r in similar.iterrows():
                explanation = models["content"].explain(itemid, int(r["itemid"]))
                render_product_card(int(r["itemid"]), item_features, r["content_similarity"],
                                     "Similarity", explanation)
        except KeyError:
            st.info("This product isn't in the content model's index.")

    with tab2:
        st.caption("Collaborative: what other shoppers who viewed this also viewed (co-interaction behavior).")
        _, item_id_map = get_id_maps()
        itemid_to_idx = dict(zip(item_id_map["itemid"], item_id_map["item_idx"]))
        idx_to_itemid = dict(zip(item_id_map["item_idx"], item_id_map["itemid"]))
        item_idx = itemid_to_idx.get(itemid)
        if item_idx is None:
            st.info("No interaction history for this product yet.")
        else:
            sims = models["cf"].similar_items(item_idx, n=5)
            if not sims:
                st.info("Not enough interaction data for this product yet (cold-start item).")
            for neighbor_idx, sim_score in sims:
                neighbor_itemid = idx_to_itemid.get(neighbor_idx)
                if neighbor_itemid is not None:
                    render_product_card(int(neighbor_itemid), item_features, sim_score, "Co-view similarity")

    with tab3:
        st.caption("Market-basket: items literally purchased together in the same checkout (uses `transactionid`).")
        fbt_results = models["fbt"].get_fbt(itemid, n=5)
        if not fbt_results:
            st.info("No recorded co-purchases for this product — most transactions in this dataset are "
                     "single-item checkouts, so coverage is inherently sparse (see the Analytics page).")
        for other_item, count, confidence in fbt_results:
            render_product_card(int(other_item), item_features, confidence,
                                 "Confidence", f"Bought together {count} time(s) — {confidence:.0%} of the time "
                                               f"someone buys Product #{itemid}, they also buy this.")


# ---------------------------------------------------------------------------
# Page: Recommendations
# ---------------------------------------------------------------------------
def render_recommendations(models, item_features, user_features):
    st.title("❤️ Recommendations")
    st.caption("Personalized recommendations from the hybrid model (Phase 3), with an optional side-by-side "
               "comparison against the other strategies evaluated in Phase 4.")

    demo_visitors = get_demo_visitors(user_features)
    col1, col2 = st.columns([2, 1])
    with col1:
        segment = st.selectbox("Choose a demo visitor segment", list(demo_visitors.keys()))
        visitor_options = demo_visitors[segment]
        visitorid = st.selectbox("Visitor ID", visitor_options)
    with col2:
        custom_id = st.text_input("...or enter any Visitor ID")
        if custom_id.strip():
            try:
                visitorid = int(custom_id.strip())
            except ValueError:
                st.warning("Visitor ID must be numeric.")

    compare_mode = st.toggle("Compare all strategies side-by-side", value=False)

    st.divider()
    hybrid_recs = models["hybrid"].recommend(int(visitorid), n=MODEL_CONFIG.top_k)
    confidence = hybrid_recs["cf_confidence_used"].iloc[0] if len(hybrid_recs) else 0.0

    st.subheader(f"Recommended for Visitor #{visitorid}")
    st.caption(f"Collaborative-filtering confidence used in this blend: **{confidence:.2f}** "
               f"({'established shopper — CF-weighted' if confidence > 0.3 else 'limited history — leaning on content & popularity'})")

    if hybrid_recs.empty:
        st.info("No recommendations available.")
    else:
        for _, r in hybrid_recs.iterrows():
            explanation = models["hybrid"].explain(int(visitorid), int(r["itemid"]))
            render_product_card(int(r["itemid"]), item_features, r["hybrid_score"], "Hybrid Score", explanation)

    if compare_mode:
        st.divider()
        st.subheader("Strategy Comparison")
        user_id_map, item_id_map = get_id_maps()
        idx_to_itemid = dict(zip(item_id_map["item_idx"], item_id_map["itemid"]))
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Popularity (baseline)**")
            for _, r in models["popularity"].trending(5).iterrows():
                st.caption(f"Product #{int(r['itemid'])} — score {r['product_interaction_score']:.3f}")
        with cols[1]:
            st.markdown("**Content-Based**")
            recent_ids = st.session_state.get("recently_viewed", [])
            if recent_ids:
                for iid, score in models["content"].recommend_for_user(recent_ids, n=5):
                    st.caption(f"Product #{int(iid)} — similarity {score:.3f}")
            else:
                st.caption("View a product first to seed content-based recommendations.")
        with cols[2]:
            st.markdown("**Collaborative Filtering (item-based)** — most accurate in Phase 4 evaluation")
            user_idx_row = user_id_map.loc[user_id_map["visitorid"] == visitorid, "user_idx"]
            if len(user_idx_row):
                recs = models["cf"].recommend_item_based(int(user_idx_row.iloc[0]), n=5)
                if recs:
                    for item_idx, score in recs:
                        real_id = idx_to_itemid.get(item_idx)
                        if real_id is not None:
                            st.caption(f"Product #{int(real_id)} — score {score:.3f}")
                else:
                    st.caption("Not enough history for CF (cold-start user/item).")
            else:
                st.caption("Unknown visitor — CF cannot serve this user (see Phase 4's user_reach metric).")

    st.divider()
    st.subheader("🕒 Recently Viewed (this session)")
    recently_viewed = st.session_state.get("recently_viewed", [])
    if recently_viewed:
        for iid in recently_viewed:
            render_product_card(iid, item_features)
    else:
        st.caption("Products you view on the Search page will appear here.")


# ---------------------------------------------------------------------------
# Page: Trending Products
# ---------------------------------------------------------------------------
def render_trending(models, item_features, category_features):
    st.title("📈 Trending Products")

    cat_ids = sorted(category_features["categoryid"].dropna().astype(int).unique().tolist())
    selected_cat = st.selectbox("Filter by category", ["All"] + cat_ids)
    cat_filter = None if selected_cat == "All" else selected_cat

    tab1, tab2, tab3 = st.tabs(["📈 Trending Now", "🏆 Best Sellers", "🔥 Recently Popular (7d)"])

    with tab1:
        st.caption("Recency-weighted popularity — recent transactions/carts/views count more than old ones.")
        df = models["popularity"].trending(15, category_id=cat_filter)
        if df.empty:
            st.info("No products in this category.")
        else:
            fig = horizontal_bar_chart(df, "product_interaction_score", "itemid", "Trending score", "Product ID",
                                        color="#6366F1")
            st.plotly_chart(fig, width='stretch')

    with tab2:
        st.caption("All-time transaction count — actual purchases, not just views (a different ranking, by design).")
        df = models["popularity"].best_sellers(15, category_id=cat_filter)
        if df.empty:
            st.info("No products in this category.")
        else:
            fig = horizontal_bar_chart(df, "item_transaction_count", "itemid", "Transactions", "Product ID",
                                        color="#10B981")
            st.plotly_chart(fig, width='stretch')

    with tab3:
        st.caption("Raw event count in the last 7 days only — a short, hard window for \"what's hot right now.\"")
        df = models["popularity"].recently_popular(15, days=7, category_id=cat_filter)
        if df.empty:
            st.info("No recent activity in this category.")
        else:
            fig = horizontal_bar_chart(df, "recent_event_count", "itemid", "Events (last 7 days)", "Product ID",
                                        color="#F59E0B")
            st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Page: Frequently Bought Together
# ---------------------------------------------------------------------------
def render_fbt(models, item_features):
    st.title("🛒 Frequently Bought Together")
    st.caption(
        "Built from actual co-purchase data (`transactionid` groups multiple items bought in one checkout) — "
        "the highest-precision signal in this project, distinct from view-based similarity."
    )

    fbt_model = models["fbt"]
    coverage = fbt_model.coverage()
    st.metric("Catalog coverage (items with ≥1 recorded co-purchase partner)", f"{coverage:.1%}")

    itemid = st.number_input("Enter a Product ID", min_value=int(item_features["itemid"].min()),
                              max_value=int(item_features["itemid"].max()), step=1)

    results = fbt_model.get_fbt(int(itemid), n=8)
    if not results:
        st.info("No recorded co-purchases for this product. Try another Product ID, or browse the Search page "
                "for products with high view counts, which are more likely to have transaction history.")
    else:
        st.subheader(f"Customers who bought Product #{int(itemid)} also bought:")
        chart_df = pd.DataFrame(results, columns=["itemid", "co_purchase_count", "confidence"])
        fig = horizontal_bar_chart(chart_df, "confidence", "itemid",
                                    "Confidence  P(also bought | bought this)", "Product ID")
        st.plotly_chart(fig, width='stretch')

        for other_item, count, confidence in results:
            render_product_card(int(other_item), item_features, confidence, "Confidence",
                                 f"Co-purchased {count} time(s) — {confidence:.0%} confidence.")


# ---------------------------------------------------------------------------
# Page: Analytics
# ---------------------------------------------------------------------------
def render_analytics(events, eval_results, item_features):
    st.title("📊 Analytics")

    tab1, tab2, tab3 = st.tabs(["📆 Interaction Patterns", "📊 Model Comparison", "🎯 Coverage & Diversity"])

    with tab1:
        st.subheader("Interaction Heatmap — Day of Week × Hour of Day")
        events_local = events.copy()
        events_local["dow"] = events_local["datetime"].dt.day_name()
        events_local["hour"] = events_local["datetime"].dt.hour
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_data = (
            events_local.groupby(["dow", "hour"]).size().unstack(fill_value=0).reindex(dow_order)
        )
        fig = heatmap_chart(heatmap_data, "Hour of day", "Day of week", "Events")
        st.plotly_chart(fig, width='stretch')
        st.caption("Darker cells = more interaction volume. Informs when to schedule 'flash trending' refreshes "
                   "and promotional pushes.")

    with tab2:
        st.subheader("Model Comparison (Phase 4 Evaluation)")
        if eval_results is None:
            st.info("Run `python -m src.evaluation` to generate this table.")
        else:
            metric_choice = st.selectbox("Metric", ["ndcg@10", "precision@10", "recall@10", "map@10", "mrr@10"])
            fig = px.bar(eval_results.sort_values(metric_choice, ascending=False), x="model", y=metric_choice,
                         color_discrete_sequence=["#6366F1"])
            st.plotly_chart(fig, width='stretch')
            st.dataframe(eval_results.set_index("model"), width='stretch')
            st.caption(
                "Item-based CF is the most accurate standalone model (see README Section 8 / notebook 04 for the "
                "full debugging story — a naive SVD collapsing onto popularity, and an over-aggressive confidence "
                "weighting in the hybrid blend, both found and fixed during evaluation)."
            )

    with tab3:
        st.subheader("Catalog Coverage, Diversity & Reach")
        if eval_results is None:
            st.info("Run `python -m src.evaluation` to generate this table.")
        else:
            fig = grouped_bar_chart(eval_results, "model", ["catalog_coverage", "diversity", "user_reach"])
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Catalog coverage is where the accuracy-vs-diversity trade-off actually shows up on this dataset — "
                "the hybrid model surfaces more of the catalog than pure collaborative filtering, at a real, "
                "quantified accuracy cost (see README Section 8)."
            )

        st.subheader("Product Popularity Distribution")
        fig2 = histogram_chart(item_features["item_view_count"], "Views")
        st.plotly_chart(fig2, width='stretch')
        st.caption(f"{item_features['is_cold_start'].mean():.1%} of the catalog has zero recorded interactions — "
                   "the empirical basis for why this project uses a hybrid, not a pure collaborative, model.")


# ---------------------------------------------------------------------------
# Main app: sidebar navigation & routing
# ---------------------------------------------------------------------------
def main() -> None:
    models = get_models()
    item_features = get_item_features()
    user_features = get_user_features()
    category_features = get_category_features()
    events = get_events()
    eval_results = get_evaluation_results()

    st.sidebar.title("🛒 RecSys Demo")
    st.sidebar.caption("E-commerce Product Recommendation System")
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Dashboard", "🔍 Search Products", "❤️ Recommendations", "📈 Trending Products",
         "🛒 Frequently Bought Together", "📊 Analytics"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Built on the Retailrocket Recommender Dataset (or a schema-accurate synthetic stand-in — "
        "see `data/raw/`). Models: popularity, content-based, collaborative filtering "
        "(item/user/SVD), frequently-bought-together, and a confidence-weighted hybrid."
    )

    if page == "🏠 Dashboard":
        render_dashboard(models, item_features, user_features, category_features, events, eval_results)
    elif page == "🔍 Search Products":
        render_search(models, item_features, category_features)
    elif page == "❤️ Recommendations":
        render_recommendations(models, item_features, user_features)
    elif page == "📈 Trending Products":
        render_trending(models, item_features, category_features)
    elif page == "🛒 Frequently Bought Together":
        render_fbt(models, item_features)
    elif page == "📊 Analytics":
        render_analytics(events, eval_results, item_features)


if __name__ == "__main__":
    main()

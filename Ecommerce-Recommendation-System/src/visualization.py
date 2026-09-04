"""
visualization.py
==================
Reusable Plotly chart-building functions shared by the Streamlit app
(`app.py`) — kept in one module so a chart's look-and-feel (colors, fonts,
margins) only needs to change in one place, rather than being copy-pasted
across the Dashboard, Trending, Frequently Bought Together, and Analytics
pages. Each function returns a `plotly.graph_objects.Figure`; callers pass
it to `st.plotly_chart(fig, width="stretch")`.

Static matplotlib/seaborn figures used in the Phase 1-4 notebooks are NOT
duplicated here — those live in the notebooks themselves, since they're
one-off exploratory outputs rather than something the live app re-renders
on every page load.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BRAND_COLORS = ["#6366F1", "#F59E0B", "#10B981", "#C44E52", "#8172B2"]


def funnel_chart(stage_labels: list[str], values: list[int], height: int = 320) -> go.Figure:
    """Conversion funnel (e.g. view -> add-to-cart -> transaction)."""
    fig = go.Figure(go.Funnel(y=stage_labels, x=values, marker={"color": BRAND_COLORS[: len(stage_labels)]}))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=height)
    return fig


def horizontal_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str,
                          color: str = BRAND_COLORS[0], height: int = 320, categorical_y: bool = True) -> go.Figure:
    """Horizontal bar chart — the recurring pattern for "top N" rankings
    (top categories, trending products, best sellers, FBT confidence).
    """
    fig = px.bar(df.sort_values(x_col), x=x_col, y=y_col, orientation="h",
                 labels={x_col: x_label, y_col: y_label}, color_discrete_sequence=[color])
    if categorical_y:
        fig.update_yaxes(type="category")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=height)
    return fig


def grouped_bar_chart(df: pd.DataFrame, x_col: str, y_cols: list[str], height: int = 380) -> go.Figure:
    """Grouped bar chart for comparing several metrics across models at
    once (e.g. catalog coverage / diversity / user reach side by side).
    """
    fig = px.bar(df, x=x_col, y=y_cols, barmode="group", color_discrete_sequence=BRAND_COLORS)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=height)
    return fig


def heatmap_chart(matrix: pd.DataFrame, x_label: str, y_label: str, color_label: str,
                   height: int = 400) -> go.Figure:
    """Day-of-week x hour-of-day (or any 2D) interaction heatmap."""
    fig = px.imshow(matrix, labels=dict(x=x_label, y=y_label, color=color_label),
                     color_continuous_scale="Purples", aspect="auto")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=height)
    return fig


def histogram_chart(series: pd.Series, x_label: str, nbins: int = 40,
                     color: str = BRAND_COLORS[0], height: int = 320) -> go.Figure:
    """Distribution histogram (e.g. product view-count distribution)."""
    fig = px.histogram(series, nbins=nbins, color_discrete_sequence=[color])
    fig.update_layout(xaxis_title=x_label, yaxis_title="Count", showlegend=False,
                       margin=dict(l=10, r=10, t=10, b=10), height=height)
    return fig

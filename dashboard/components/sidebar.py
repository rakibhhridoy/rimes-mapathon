"""
Sidebar filters + floating analytics overlay for the map panel.
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import numpy as np


def render_sidebar(infra: gpd.GeoDataFrame,
                    union_gdf: gpd.GeoDataFrame = None,
                    grid_gdf: gpd.GeoDataFrame = None,
                    is_dark: bool = True):
    """Render sidebar with filters."""
    accent = "#38bdf8"
    text2 = "#94a3b8" if is_dark else "#64748b"

    st.sidebar.markdown(
        f'<h3 style="color:{accent}; margin-bottom:8px;">Filters</h3>',
        unsafe_allow_html=True,
    )

    # Asset type filter
    asset_types = sorted(infra["asset_type"].unique().tolist()) if "asset_type" in infra.columns else []
    selected_types = st.sidebar.multiselect(
        "Asset Types", asset_types, default=asset_types
    )

    # Risk slider
    risk_min, risk_max = st.sidebar.slider(
        "Risk Range", 0.0, 1.0, (0.0, 1.0), step=0.05
    )

    # Division filter
    if "division" in infra.columns:
        divisions = sorted(infra["division"].unique().tolist())
        selected_divs = st.sidebar.multiselect(
            "Divisions", divisions, default=divisions
        )
    else:
        selected_divs = None

    # Apply filters
    filtered = infra.copy()
    if "asset_type" in filtered.columns:
        filtered = filtered[filtered["asset_type"].isin(selected_types)]
    if "flood_risk" in filtered.columns:
        filtered = filtered[
            (filtered["flood_risk"] >= risk_min) &
            (filtered["flood_risk"] <= risk_max)
        ]
    if selected_divs is not None and "division" in filtered.columns:
        filtered = filtered[filtered["division"].isin(selected_divs)]

    # Stats
    st.sidebar.markdown("---")
    st.sidebar.metric("Showing", f"{len(filtered):,} / {len(infra):,}")

    if "division" in filtered.columns:
        for div, cnt in filtered["division"].value_counts().items():
            st.sidebar.markdown(
                f'<span style="color:{accent};">{div}:</span> '
                f'<b style="color:{"#e2e8f0" if is_dark else "#1e293b"};">{cnt:,}</b>',
                unsafe_allow_html=True,
            )

    return filtered, selected_types, risk_min, risk_max


def render_analytics_overlay(infra: gpd.GeoDataFrame,
                              union_gdf: gpd.GeoDataFrame = None,
                              is_dark: bool = True):
    """Floating analytics panel next to the map."""
    bg = "#1e293b" if is_dark else "#ffffff"
    border = "#334155" if is_dark else "#e2e8f0"
    text = "#e2e8f0" if is_dark else "#1e293b"
    text2 = "#94a3b8" if is_dark else "#64748b"

    layout_opts = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=text,
        margin=dict(l=5, r=5, t=24, b=5),
    )

    # --- Asset type bar chart ---
    if "asset_type" in infra.columns:
        st.markdown(
            f'<p style="color:{text2}; font-size:0.7rem; margin:0 0 4px 0; '
            f'text-transform:uppercase; letter-spacing:0.5px;">Assets by Type</p>',
            unsafe_allow_html=True,
        )
        tc = infra["asset_type"].value_counts().reset_index()
        tc.columns = ["Type", "Count"]

        fig = px.bar(
            tc, x="Count", y="Type", orientation="h",
            color="Count",
            color_continuous_scale=["#1e3a5f", "#38bdf8"] if is_dark else ["#bfdbfe", "#2563eb"],
        )
        fig.update_layout(
            height=max(180, len(tc) * 22 + 40),
            showlegend=False, coloraxis_showscale=False,
            **layout_opts,
        )
        fig.update_xaxes(gridcolor=border, showgrid=True)
        fig.update_yaxes(gridcolor=border)
        st.plotly_chart(fig, use_container_width=True)

    # --- Risk distribution mini-histogram ---
    if "flood_risk" in infra.columns and len(infra) > 0:
        st.markdown(
            f'<p style="color:{text2}; font-size:0.7rem; margin:12px 0 4px 0; '
            f'text-transform:uppercase; letter-spacing:0.5px;">Risk Distribution</p>',
            unsafe_allow_html=True,
        )
        fig_h = px.histogram(
            infra, x="flood_risk", nbins=20,
            color_discrete_sequence=["#38bdf8"] if is_dark else ["#3b82f6"],
        )
        fig_h.update_layout(height=160, showlegend=False, **layout_opts)
        fig_h.update_xaxes(gridcolor=border, title="")
        fig_h.update_yaxes(gridcolor=border, title="")
        st.plotly_chart(fig_h, use_container_width=True)

    # --- Division pie ---
    if "division" in infra.columns:
        st.markdown(
            f'<p style="color:{text2}; font-size:0.7rem; margin:12px 0 4px 0; '
            f'text-transform:uppercase; letter-spacing:0.5px;">By Division</p>',
            unsafe_allow_html=True,
        )
        div_counts = infra["division"].value_counts().reset_index()
        div_counts.columns = ["Division", "Count"]
        fig_p = px.pie(
            div_counts, names="Division", values="Count", hole=0.6,
            color_discrete_sequence=["#38bdf8", "#22c55e", "#f59e0b", "#ef4444"],
        )
        fig_p.update_layout(height=180, **layout_opts, legend=dict(font_size=9))
        fig_p.update_traces(textinfo="percent", textfont_size=10)
        st.plotly_chart(fig_p, use_container_width=True)

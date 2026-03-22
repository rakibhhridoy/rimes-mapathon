"""
Sidebar analytics panel with filters, ranked tables, and charts.
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def render_sidebar(infra: gpd.GeoDataFrame,
                    union_gdf: gpd.GeoDataFrame = None,
                    grid_gdf: gpd.GeoDataFrame = None):
    """Render the sidebar with filters and analytics."""

    st.sidebar.header("Filters")

    # --- Filters ---
    asset_types = sorted(infra["asset_type"].unique().tolist()) if "asset_type" in infra.columns else []
    selected_types = st.sidebar.multiselect(
        "Asset Types", asset_types, default=asset_types
    )

    risk_min, risk_max = st.sidebar.slider(
        "Risk Threshold", 0.0, 1.0, (0.0, 1.0), step=0.05
    )

    # Apply filters
    filtered = infra.copy()
    if "asset_type" in filtered.columns:
        filtered = filtered[filtered["asset_type"].isin(selected_types)]
    if "flood_risk" in filtered.columns:
        filtered = filtered[
            (filtered["flood_risk"] >= risk_min) &
            (filtered["flood_risk"] <= risk_max)
        ]

    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Assets", len(filtered))
    if "is_high_risk" in filtered.columns:
        st.sidebar.metric("High-Risk Assets", int(filtered["is_high_risk"].sum()))

    return filtered, selected_types, risk_min, risk_max


def render_analytics(infra: gpd.GeoDataFrame,
                      union_gdf: gpd.GeoDataFrame = None,
                      grid_gdf: gpd.GeoDataFrame = None):
    """Render analytics charts in the sidebar area."""

    # --- Risk Distribution ---
    st.subheader("Risk Distribution")
    if "flood_risk" in infra.columns and len(infra) > 0:
        fig_hist = px.histogram(
            infra, x="flood_risk", nbins=30,
            color_discrete_sequence=["#dc3545"],
            labels={"flood_risk": "Flood Risk Score"},
        )
        fig_hist.update_layout(
            height=250, margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, width="stretch")

    # --- Exposed Assets by Category ---
    st.subheader("Exposed Assets by Type")
    if "asset_type" in infra.columns:
        type_counts = infra["asset_type"].value_counts().reset_index()
        type_counts.columns = ["Asset Type", "Count"]

        fig_bar = px.bar(
            type_counts, x="Count", y="Asset Type",
            orientation="h",
            color="Count",
            color_continuous_scale="OrRd",
        )
        fig_bar.update_layout(
            height=300, margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, width="stretch")

    # --- Union risk vs. total assets scatter ---
    if union_gdf is not None and len(union_gdf) > 0:
        st.subheader("Risk vs. Assets by Union")
        fig_scatter = px.scatter(
            union_gdf,
            x="mean_risk",
            y="total_assets" if "total_assets" in union_gdf.columns else "n_high_risk",
            hover_name="admin_name" if "admin_name" in union_gdf.columns else None,
            size="n_high_risk" if "n_high_risk" in union_gdf.columns else None,
            color="mean_risk",
            color_continuous_scale="RdYlGn_r",
            labels={
                "mean_risk": "Mean Risk",
                "total_assets": "Total Assets",
            },
        )
        fig_scatter.update_layout(
            height=300, margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_scatter, width="stretch")

    # --- Top 10 ranked assets table ---
    st.subheader("Top At-Risk Assets")
    display_cols = [
        c for c in ["risk_rank", "asset_type", "name", "flood_risk"]
        if c in infra.columns
    ]
    if display_cols and "flood_risk" in infra.columns:
        top = infra.sort_values("flood_risk", ascending=False).head(20)
        st.dataframe(top[display_cols], width="stretch", hide_index=True)
    elif display_cols:
        st.dataframe(infra[display_cols].head(20), width="stretch", hide_index=True)

    # --- Export ---
    st.markdown("---")
    st.subheader("Export Data")
    col1, col2 = st.columns(2)

    with col1:
        if "flood_risk" in infra.columns:
            csv_data = infra.drop(columns=["geometry", "centroid"], errors="ignore")
            st.download_button(
                "Download CSV",
                csv_data.to_csv(index=False),
                "risk_assets.csv",
                "text/csv",
            )

    with col2:
        if len(infra) > 0:
            geojson_str = infra.head(200).to_json()
            st.download_button(
                "Download GeoJSON",
                geojson_str,
                "risk_assets.geojson",
                "application/json",
            )

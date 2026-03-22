"""
Sidebar analytics panel with filters, ranked tables, and charts.
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Plotly dark theme defaults
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    margin=dict(l=10, r=10, t=32, b=10),
)


def render_sidebar(infra: gpd.GeoDataFrame,
                    union_gdf: gpd.GeoDataFrame = None,
                    grid_gdf: gpd.GeoDataFrame = None):
    """Render the sidebar with filters and analytics."""

    st.sidebar.markdown(
        '<h2 style="color:#38bdf8; margin-bottom:4px;">Filters</h2>',
        unsafe_allow_html=True,
    )

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

    # Metrics
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Filtered", f"{len(filtered):,}")
    if "is_high_risk" in filtered.columns:
        col2.metric("High Risk", int(filtered["is_high_risk"].sum()))
    else:
        col2.metric("Types", filtered["asset_type"].nunique() if "asset_type" in filtered.columns else 0)

    # Division breakdown
    if "division" in filtered.columns:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**By Division**")
        for div, count in filtered["division"].value_counts().items():
            st.sidebar.markdown(
                f'<span style="color:#7dd3fc;">{div}</span>: '
                f'<span style="color:#e2e8f0; font-weight:600;">{count:,}</span>',
                unsafe_allow_html=True,
            )

    return filtered, selected_types, risk_min, risk_max


def render_analytics(infra: gpd.GeoDataFrame,
                      union_gdf: gpd.GeoDataFrame = None,
                      grid_gdf: gpd.GeoDataFrame = None):
    """Render analytics charts in the analytics column."""

    # --- Risk Distribution ---
    if "flood_risk" in infra.columns and len(infra) > 0:
        st.markdown('<p style="color:#94a3b8; font-size:0.85rem; margin-bottom:4px;">'
                    'RISK DISTRIBUTION</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            infra, x="flood_risk", nbins=30,
            color_discrete_sequence=["#38bdf8"],
            labels={"flood_risk": "Flood Risk Score"},
        )
        fig_hist.update_layout(height=200, showlegend=False, **PLOTLY_LAYOUT)
        fig_hist.update_xaxes(gridcolor="#334155")
        fig_hist.update_yaxes(gridcolor="#334155")
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- Exposed Assets by Category ---
    if "asset_type" in infra.columns:
        st.markdown('<p style="color:#94a3b8; font-size:0.85rem; margin-bottom:4px;">'
                    'ASSETS BY TYPE</p>', unsafe_allow_html=True)
        type_counts = infra["asset_type"].value_counts().reset_index()
        type_counts.columns = ["Asset Type", "Count"]

        fig_bar = px.bar(
            type_counts, x="Count", y="Asset Type",
            orientation="h",
            color="Count",
            color_continuous_scale=["#1e3a5f", "#38bdf8", "#7dd3fc"],
        )
        fig_bar.update_layout(
            height=max(200, len(type_counts) * 28 + 40),
            showlegend=False,
            coloraxis_showscale=False,
            **PLOTLY_LAYOUT,
        )
        fig_bar.update_xaxes(gridcolor="#334155")
        fig_bar.update_yaxes(gridcolor="#334155")
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Union risk scatter ---
    if union_gdf is not None and len(union_gdf) > 0:
        st.markdown('<p style="color:#94a3b8; font-size:0.85rem; margin-bottom:4px;">'
                    'RISK vs ASSETS BY UNION</p>', unsafe_allow_html=True)
        y_col = "total_assets" if "total_assets" in union_gdf.columns else "n_high_risk"
        fig_scatter = px.scatter(
            union_gdf,
            x="mean_risk",
            y=y_col,
            hover_name="admin_name" if "admin_name" in union_gdf.columns else None,
            size="n_high_risk" if "n_high_risk" in union_gdf.columns else None,
            color="mean_risk",
            color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
            labels={"mean_risk": "Mean Risk", y_col: y_col.replace("_", " ").title()},
        )
        fig_scatter.update_layout(height=250, **PLOTLY_LAYOUT)
        fig_scatter.update_xaxes(gridcolor="#334155")
        fig_scatter.update_yaxes(gridcolor="#334155")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Top ranked assets table ---
    st.markdown('<p style="color:#94a3b8; font-size:0.85rem; margin-bottom:4px;">'
                'TOP AT-RISK ASSETS</p>', unsafe_allow_html=True)
    display_cols = [
        c for c in ["risk_rank", "asset_type", "name", "flood_risk"]
        if c in infra.columns
    ]
    if display_cols and "flood_risk" in infra.columns:
        top = infra.sort_values("flood_risk", ascending=False).head(15)
        st.dataframe(top[display_cols], height=300, hide_index=True)
    elif display_cols:
        st.dataframe(infra[display_cols].head(15), height=300, hide_index=True)
    else:
        st.info("Run full pipeline to see ranked assets.")

    # --- Export ---
    st.markdown("---")
    st.markdown('<p style="color:#94a3b8; font-size:0.85rem; margin-bottom:8px;">'
                'EXPORT DATA</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        csv_data = infra.drop(columns=["geometry", "centroid"], errors="ignore")
        st.download_button(
            "📥 CSV",
            csv_data.to_csv(index=False),
            "risk_assets.csv",
            "text/csv",
            use_container_width=True,
        )

    with col2:
        if len(infra) > 0:
            # Drop non-serializable columns before export
            export_gdf = infra.drop(columns=["centroid"], errors="ignore")
            geojson_str = export_gdf.head(200).to_json()
            st.download_button(
                "📥 GeoJSON",
                geojson_str,
                "risk_assets.geojson",
                "application/json",
                use_container_width=True,
            )

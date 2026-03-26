"""
Sidebar: pipeline status, mapping confidence, layer toggles, data filters,
and floating analytics overlay for the map panel.
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import numpy as np
from dashboard.data.loader import get_confidence_metrics


def render_sidebar(infra: gpd.GeoDataFrame,
                    union_gdf: gpd.GeoDataFrame = None,
                    grid_gdf: gpd.GeoDataFrame = None,
                    is_dark: bool = True):
    """Render sidebar with pipeline status, confidence, layers, and data filters."""
    accent = "#38bdf8"
    text2 = "#94a3b8" if is_dark else "#64748b"

    with st.sidebar:
        # ── PIPELINE STATUS ──────────────────────────────────────────
        st.markdown(
            """
            <div style="background:#0d1f2d;border:1px solid #1a3a50;border-radius:8px;
                        padding:10px 14px;margin-bottom:16px;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <div style="width:8px;height:8px;border-radius:50%;background:#3cdea0;
                            box-shadow:0 0 6px #3cdea088;
                            flex-shrink:0;"></div>
                <span style="color:#3cdea0;font-size:11px;font-weight:700;
                             letter-spacing:0.12em;">PIPELINE ACTIVE</span>
              </div>
              <div style="color:#4a7a9a;font-size:10px;margin-top:2px;">
                Last sync: <span style="color:#7abcd8;">2 min ago</span>
              </div>
              <div style="color:#4a7a9a;font-size:10px;">
                Sentinel-1 pass: <span style="color:#7abcd8;">06:14 UTC</span> ·
                Next: <span style="color:#7abcd8;">12:38 UTC</span>
              </div>
              <div style="color:#4a7a9a;font-size:10px;">
                BWDB gauges: <span style="color:#3cdea0;">43 / 45 live</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── MAPPING CONFIDENCE ──────────────────────────────────────
        st.markdown(
            """
            <div style="color:#3cb8de;font-size:10px;font-weight:700;
                        letter-spacing:0.12em;margin-bottom:8px;">
              MAPPING CONFIDENCE
            </div>
            """,
            unsafe_allow_html=True,
        )

        _conf = get_confidence_metrics()
        kriging_var   = _conf["kriging_var"]
        gnn_ci_width  = _conf["gnn_ci_width"]
        ensemble_iqr  = _conf["ensemble_iqr"]
        data_density  = _conf["data_density"]

        def conf_badge(val, low, mid):
            if val < low:
                return "HIGH", "#3cdea0", "#0d2e24"
            elif val < mid:
                return "MEDIUM", "#dea03c", "#2e2a0d"
            else:
                return "LOW", "#de3c3c", "#2e0d0d"

        k_tier, k_col, k_bg = conf_badge(kriging_var, 0.05, 0.15)
        g_tier, g_col, g_bg = conf_badge(gnn_ci_width, 0.10, 0.20)
        e_tier, e_col, e_bg = conf_badge(ensemble_iqr, 0.08, 0.15)

        st.markdown(
            f"""
            <div style="background:#0a0e14;border:1px solid #1a2a38;border-radius:8px;
                        padding:12px 14px;margin-bottom:16px;font-size:11px;">
              <div style="display:flex;justify-content:space-between;align-items:center;
                          padding:5px 0;border-bottom:1px solid #0d1822;">
                <span style="color:#5a8ab0;">Kriging var</span>
                <span style="color:#e8f4ff;font-weight:600;">{kriging_var}</span>
                <span style="background:{k_bg};color:{k_col};padding:1px 6px;
                             border-radius:3px;font-size:9px;">{k_tier}</span>
              </div>
              <div style="display:flex;justify-content:space-between;align-items:center;
                          padding:5px 0;border-bottom:1px solid #0d1822;">
                <span style="color:#5a8ab0;">GNN 90% CI</span>
                <span style="color:#e8f4ff;font-weight:600;">{gnn_ci_width}</span>
                <span style="background:{g_bg};color:{g_col};padding:1px 6px;
                             border-radius:3px;font-size:9px;">{g_tier}</span>
              </div>
              <div style="display:flex;justify-content:space-between;align-items:center;
                          padding:5px 0;border-bottom:1px solid #0d1822;">
                <span style="color:#5a8ab0;">Ensemble IQR</span>
                <span style="color:#e8f4ff;font-weight:600;">{ensemble_iqr}</span>
                <span style="background:{e_bg};color:{e_col};padding:1px 6px;
                             border-radius:3px;font-size:9px;">{e_tier}</span>
              </div>
              <div style="display:flex;justify-content:space-between;align-items:center;
                          padding:5px 0;">
                <span style="color:#5a8ab0;">Training density</span>
                <span style="color:#e8f4ff;font-weight:600;">{data_density}%</span>
                <span style="background:#0d2e24;color:#3cdea0;padding:1px 6px;
                             border-radius:3px;font-size:9px;">HIGH</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── MAP LAYER TOGGLES ──────────────────────────────────────
        st.markdown(
            """
            <div style="color:#3cb8de;font-size:10px;font-weight:700;
                        letter-spacing:0.12em;margin-bottom:8px;">
              MAP LAYER TOGGLES
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**OSM Infrastructure**")
        st.checkbox("Roads & Highways", value=True,  key="osm_roads")
        st.checkbox("Bridges",          value=True,  key="osm_bridges")
        st.checkbox("Schools",          value=True,  key="osm_schools")
        st.checkbox("Hospitals",        value=True,  key="osm_hospitals")

        st.markdown("**Physiographic Layers**")
        st.checkbox("DEM Elevation",    value=False, key="show_dem")
        st.checkbox("Slope",            value=False, key="show_slope")
        st.checkbox("HAND Index",       value=True,  key="show_hand")
        st.checkbox("Land Cover (LULC)",value=False, key="show_lulc")

        st.markdown("**Socioeconomic Layers**")
        st.checkbox("Population Density", value=False, key="show_popdens")
        st.checkbox("CVI Choropleth",     value=True,  key="show_cvi")
        st.checkbox("AlphaEarth Clusters",value=False, key="show_alphearth")

        st.divider()

        # ── FORECAST HORIZON ─────────────────────────────────────
        st.markdown(
            """
            <div style="color:#3cb8de;font-size:10px;font-weight:700;
                        letter-spacing:0.12em;margin-bottom:8px;">
              FORECAST HORIZON
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.slider("Days ahead", 1, 5, 3, key="forecast_days")

        # ── SCENARIO ─────────────────────────────────────────────
        st.selectbox(
            "Climate scenario",
            ["Current (2025)", "SSP2-4.5 (+0.5m SLR)", "SSP5-8.5 (+1.2m SLR)"],
            key="scenario",
        )

        st.divider()

        # ── DATA FILTERS ─────────────────────────────────────────
        st.markdown(
            f'<h3 style="color:{accent}; margin-bottom:8px;">Data Filters</h3>',
            unsafe_allow_html=True,
        )

        # Asset type filter
        asset_types = sorted(infra["asset_type"].unique().tolist()) if "asset_type" in infra.columns else []
        selected_types = st.multiselect(
            "Asset Types", asset_types, default=asset_types
        )

        # Risk slider
        risk_min, risk_max = st.slider(
            "Risk Range", 0.0, 1.0, (0.0, 1.0), step=0.05
        )

        # Division filter
        if "division" in infra.columns:
            divisions = sorted(infra["division"].unique().tolist())
            selected_divs = st.multiselect(
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
        st.markdown("---")
        st.metric("Showing", f"{len(filtered):,} / {len(infra):,}")

        if "division" in filtered.columns:
            for div, cnt in filtered["division"].value_counts().items():
                st.markdown(
                    f'<span style="color:{accent};">{div}:</span> '
                    f'<b style="color:{"#e2e8f0" if is_dark else "#1e293b"};">{cnt:,}</b>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.caption(
            "Fermium Hazard Mapper · ResilienceAI\n"
            "RIMES Mapathon 2025 · Smart SGMDI"
        )

    # Return layer state dict
    layers = {
        "osm_roads":      st.session_state.get("osm_roads", True),
        "osm_bridges":    st.session_state.get("osm_bridges", True),
        "osm_schools":    st.session_state.get("osm_schools", True),
        "osm_hospitals":  st.session_state.get("osm_hospitals", True),
        "show_dem":       st.session_state.get("show_dem", False),
        "show_slope":     st.session_state.get("show_slope", False),
        "show_hand":      st.session_state.get("show_hand", True),
        "show_lulc":      st.session_state.get("show_lulc", False),
        "show_popdens":   st.session_state.get("show_popdens", False),
        "show_cvi":       st.session_state.get("show_cvi", True),
        "show_alphearth": st.session_state.get("show_alphearth", False),
        "forecast_days":  st.session_state.get("forecast_days", 3),
        "scenario":       st.session_state.get("scenario", "Current (2025)"),
    }

    return filtered, selected_types, risk_min, risk_max, layers


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

"""
Union-level risk cards — horizontal scrollable carousel showing
top at-risk unions with their exposed infrastructure summary.
"""

import streamlit as st
import geopandas as gpd
import pandas as pd


def risk_color(score: float) -> str:
    """Return CSS color based on risk score."""
    if score >= 0.7:
        return "#dc3545"   # red — critical
    elif score >= 0.5:
        return "#fd7e14"   # orange — high
    elif score >= 0.3:
        return "#ffc107"   # yellow — moderate
    return "#28a745"       # green — low


def risk_label(score: float) -> str:
    if score >= 0.7:
        return "CRITICAL"
    elif score >= 0.5:
        return "HIGH"
    elif score >= 0.3:
        return "MODERATE"
    return "LOW"


def render_risk_cards(union_gdf: gpd.GeoDataFrame, n_display: int = 10):
    """Render union-level risk cards as a horizontal row."""
    st.subheader("Union-Level Risk Cards")

    if union_gdf is None or len(union_gdf) == 0:
        st.info("No union-level data available. Run the risk pipeline first.")
        return

    # Sort by risk
    top = union_gdf.sort_values("mean_risk", ascending=False).head(n_display)

    # Create columns for card layout
    cols = st.columns(min(len(top), 4))

    for i, (_, row) in enumerate(top.iterrows()):
        col = cols[i % len(cols)]
        score = row.get("mean_risk", 0)
        color = risk_color(score)
        label = risk_label(score)

        with col:
            st.markdown(
                f"""
                <div style="
                    border: 2px solid {color};
                    border-radius: 10px;
                    padding: 12px;
                    margin-bottom: 10px;
                    background: linear-gradient(135deg, {color}15, white);
                ">
                    <h4 style="margin:0; color: {color};">
                        #{row.get('risk_rank', i+1)} {row.get('admin_name', 'Unknown')}
                    </h4>
                    <p style="margin:4px 0; font-size:0.85em; color:#666;">
                        Risk: <strong style="color:{color}">{score:.3f} ({label})</strong>
                    </p>
                    <div style="
                        background: #e9ecef;
                        border-radius: 4px;
                        height: 8px;
                        margin: 6px 0;
                    ">
                        <div style="
                            background: {color};
                            width: {score*100:.0f}%;
                            height: 8px;
                            border-radius: 4px;
                        "></div>
                    </div>
                    <p style="margin:4px 0; font-size:0.8em;">
                        Hospitals: {int(row.get('n_hospitals_exposed', 0))} |
                        Schools: {int(row.get('n_schools_exposed', 0))} |
                        Bridges: {int(row.get('n_bridges_exposed', 0))}
                    </p>
                    <p style="margin:4px 0; font-size:0.8em;">
                        Roads: {int(row.get('n_roads', 0))} |
                        Cropland: {int(row.get('n_cropland', 0))} |
                        Total: {int(row.get('total_assets', 0))}
                    </p>
                    <p style="margin:4px 0; font-size:0.75em; color:#888;">
                        High-risk cells: {int(row.get('n_high_risk', 0))}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Show full table below
    with st.expander("View full union ranking table"):
        display_cols = [
            c for c in [
                "risk_rank", "admin_name", "mean_risk", "max_risk",
                "n_high_risk", "n_hospitals_exposed", "n_schools_exposed",
                "n_bridges_exposed", "n_roads", "n_cropland", "total_assets",
            ] if c in union_gdf.columns
        ]
        st.dataframe(
            union_gdf[display_cols].sort_values("mean_risk", ascending=False),
            width="stretch",
        )

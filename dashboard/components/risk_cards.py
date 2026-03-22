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
        return "#ef4444"   # red — critical
    elif score >= 0.5:
        return "#f59e0b"   # amber — high
    elif score >= 0.3:
        return "#eab308"   # yellow — moderate
    return "#22c55e"       # green — low


def risk_label(score: float) -> str:
    if score >= 0.7:
        return "CRITICAL"
    elif score >= 0.5:
        return "HIGH"
    elif score >= 0.3:
        return "MODERATE"
    return "LOW"


def render_risk_cards(union_gdf: gpd.GeoDataFrame, n_display: int = 12):
    """Render union-level risk cards as a styled grid."""
    st.markdown(
        '<div class="section-header">Union-Level Risk Assessment</div>',
        unsafe_allow_html=True,
    )

    if union_gdf is None or len(union_gdf) == 0:
        st.info("No union-level data available. Run the risk pipeline first.")
        return

    # Sort by risk
    top = union_gdf.sort_values("mean_risk", ascending=False).head(n_display)

    # Render cards in rows of 4
    cards_per_row = 4
    rows = [top.iloc[i:i + cards_per_row] for i in range(0, len(top), cards_per_row)]

    for row_data in rows:
        cols = st.columns(cards_per_row)
        for i, (_, row) in enumerate(row_data.iterrows()):
            score = row.get("mean_risk", 0)
            color = risk_color(score)
            label = risk_label(score)
            rank = int(row.get("risk_rank", 0))
            name = row.get("admin_name", "Unknown")

            n_hosp = int(row.get("n_hospitals_exposed", 0))
            n_sch = int(row.get("n_schools_exposed", 0))
            n_br = int(row.get("n_bridges_exposed", 0))
            n_rd = int(row.get("n_roads", 0))
            n_crop = int(row.get("n_cropland", 0))
            total = int(row.get("total_assets", 0))
            n_high = int(row.get("n_high_risk", 0))

            with cols[i]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%);
                    border: 1px solid {color}44;
                    border-left: 4px solid {color};
                    border-radius: 10px;
                    padding: 16px 18px;
                    margin-bottom: 12px;
                    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
                    min-height: 200px;
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="
                            font-size: 0.65rem;
                            color: {color};
                            background: {color}18;
                            border: 1px solid {color}33;
                            padding: 2px 8px;
                            border-radius: 10px;
                            font-weight: 600;
                        ">{label}</span>
                        <span style="
                            font-size: 0.7rem;
                            color: #64748b;
                            font-weight: 600;
                        ">#{rank}</span>
                    </div>

                    <h4 style="
                        margin: 8px 0 4px 0;
                        color: #f1f5f9;
                        font-size: 0.95rem;
                        font-weight: 600;
                        line-height: 1.2;
                    ">{name}</h4>

                    <div style="
                        font-size: 1.4rem;
                        font-weight: 700;
                        color: {color};
                        margin: 4px 0;
                    ">{score:.3f}</div>

                    <!-- Progress bar -->
                    <div style="
                        background: #334155;
                        border-radius: 4px;
                        height: 6px;
                        margin: 8px 0;
                        overflow: hidden;
                    ">
                        <div style="
                            background: linear-gradient(90deg, {color}, {color}88);
                            width: {score*100:.0f}%;
                            height: 6px;
                            border-radius: 4px;
                        "></div>
                    </div>

                    <!-- Stats grid -->
                    <div style="
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 3px 12px;
                        font-size: 0.72rem;
                        color: #94a3b8;
                        margin-top: 8px;
                    ">
                        <span>🏥 {n_hosp} hospitals</span>
                        <span>🏫 {n_sch} schools</span>
                        <span>🌉 {n_br} bridges</span>
                        <span>🛣️ {n_rd} roads</span>
                        <span>🌾 {n_crop} cropland</span>
                        <span>📊 {total} total</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Expandable full table
    with st.expander("View full union ranking table"):
        display_cols = [
            c for c in [
                "risk_rank", "admin_name", "mean_risk", "max_risk",
                "n_high_risk", "n_hospitals_exposed", "n_schools_exposed",
                "n_bridges_exposed", "n_roads", "n_cropland", "total_assets",
            ] if c in union_gdf.columns
        ]
        if display_cols:
            st.dataframe(
                union_gdf[display_cols].sort_values("mean_risk", ascending=False),
                height=400,
                hide_index=True,
            )

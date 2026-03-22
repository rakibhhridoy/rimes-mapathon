"""
SGMDI Streamlit Dashboard — Redesigned layout:
  - Sticky navbar (title + search + theme toggle)
  - KPI ticker strip
  - Full-width map with floating analytics overlay
  - Tabbed summary below map (Assets | Unions | Export)
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.components.risk_cards import render_risk_cards
from dashboard.components.map_view import render_map
from dashboard.components.sidebar import render_sidebar, render_analytics_overlay

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SGMDI — Flood Risk Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme state
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

is_dark = st.session_state.theme == "dark"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
BG = "#0f172a" if is_dark else "#f8fafc"
BG2 = "#1e293b" if is_dark else "#ffffff"
BG3 = "#334155" if is_dark else "#e2e8f0"
TEXT = "#e2e8f0" if is_dark else "#1e293b"
TEXT2 = "#94a3b8" if is_dark else "#64748b"
ACCENT = "#38bdf8"
BORDER = "#334155" if is_dark else "#cbd5e1"

st.markdown(f"""
<style>
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    /* --- Navbar --- */
    .sgmdi-nav {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: {BG2};
        border-bottom: 1px solid {BORDER};
        padding: 10px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: -1rem -1rem 16px -1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    }}
    .sgmdi-nav .nav-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .sgmdi-nav .nav-brand h1 {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {ACCENT};
        margin: 0;
        white-space: nowrap;
    }}
    .sgmdi-nav .nav-brand .nav-sub {{
        font-size: 0.7rem;
        color: {TEXT2};
        margin: 0;
    }}
    .sgmdi-nav .nav-badge {{
        background: {ACCENT}18;
        border: 1px solid {ACCENT}33;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.65rem;
        color: {ACCENT};
        white-space: nowrap;
    }}

    /* --- KPI strip --- */
    .kpi-strip {{
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
        overflow-x: auto;
        padding: 4px 0;
    }}
    .kpi-item {{
        flex: 1;
        min-width: 130px;
        background: {BG2};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.15);
    }}
    .kpi-item .kpi-val {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {ACCENT};
        line-height: 1;
    }}
    .kpi-item .kpi-label {{
        font-size: 0.65rem;
        color: {TEXT2};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }}
    .kpi-item.danger .kpi-val {{ color: #ef4444; }}
    .kpi-item.warning .kpi-val {{ color: #f59e0b; }}
    .kpi-item.success .kpi-val {{ color: #22c55e; }}

    /* --- Section label --- */
    .sec-label {{
        font-size: 0.7rem;
        color: {TEXT2};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 8px 0;
        font-weight: 600;
    }}

    /* --- Tab styling --- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {BG2};
        border: 1px solid {BORDER};
        border-radius: 8px 8px 0 0;
        color: {TEXT2};
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {ACCENT}18;
        border-color: {ACCENT};
        color: {ACCENT};
    }}

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {{
        background: {BG2};
        border-right: 1px solid {BORDER};
    }}

    /* --- Footer --- */
    .sgmdi-footer {{
        text-align: center;
        color: {TEXT2};
        font-size: 0.7rem;
        padding: 16px 0;
        margin-top: 24px;
        border-top: 1px solid {BORDER};
    }}

    hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@st.cache_data
def load_gdf(path: str) -> gpd.GeoDataFrame:
    if Path(path).exists():
        return gpd.read_file(path)
    return gpd.GeoDataFrame()


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    output_dir = Path("data/output")

    # ── Navbar ──────────────────────────────────────────────────────────
    nav_l, nav_c, nav_r = st.columns([3, 4, 2])
    with nav_l:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:1.5rem;">🌊</span>
            <div>
                <div style="font-size:1.05rem; font-weight:700; color:{ACCENT}; line-height:1.2;">
                    SGMDI — Flood Risk Intelligence
                </div>
                <div style="font-size:0.65rem; color:{TEXT2};">
                    Smart Geospatial Mapping & Disaster Impact Intelligence
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with nav_c:
        search_query = st.text_input(
            "search", placeholder="Search assets by name or type...",
            label_visibility="collapsed",
        )

    with nav_r:
        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown(
                f'<span class="sgmdi-nav nav-badge" style="display:inline-block; '
                f'margin-top:8px;">Rangpur & Rajshahi</span>',
                unsafe_allow_html=True,
            )
        with r2:
            if st.button("🌙" if is_dark else "☀️", help="Toggle theme"):
                st.session_state.theme = "light" if is_dark else "dark"
                st.rerun()

    st.markdown(f'<hr style="margin:8px 0 16px 0; border-color:{BORDER};">',
                unsafe_allow_html=True)

    # ── Load data ───────────────────────────────────────────────────────
    infra = load_gdf(str(output_dir / "risk_ranked_assets.geojson"))
    if len(infra) == 0:
        infra = load_gdf("data/raw/infrastructure_raw.gpkg")
        csv_data = load_csv(str(output_dir / "risk_ranked_assets.csv"))
        if len(csv_data) > 0 and len(infra) > 0:
            for col in ["flood_risk", "risk_rank", "is_high_risk"]:
                if col in csv_data.columns:
                    infra[col] = csv_data[col].values[:len(infra)]

    union_gdf = load_gdf(str(output_dir / "union_risk_summary.geojson"))
    hotspot_gdf = load_gdf(str(output_dir / "hotspot_clusters.geojson"))
    grid_gdf = load_gdf(str(output_dir / "risk_grid.geojson"))

    if len(infra) == 0:
        st.error("No data found. Run: `python -m pipeline.cli -c config.yaml run`")
        st.stop()

    # Ensure coordinate columns
    if "lon" not in infra.columns:
        pts = infra.geometry.representative_point()
        infra["lon"] = pts.x
        infra["lat"] = pts.y

    # Drop non-serializable columns
    if "centroid" in infra.columns:
        infra = infra.drop(columns=["centroid"])

    # Apply search
    if search_query:
        mask = (
            infra["name"].str.contains(search_query, case=False, na=False) |
            infra["asset_type"].str.contains(search_query, case=False, na=False)
        )
        infra = infra[mask]
        if len(infra) == 0:
            st.warning(f'No assets match "{search_query}"')
            st.stop()

    # ── Sidebar filters ─────────────────────────────────────────────────
    filtered, _, _, _ = render_sidebar(infra, union_gdf, grid_gdf, is_dark)

    # ── KPI Strip ───────────────────────────────────────────────────────
    n_total = len(filtered)
    n_high = int(filtered["is_high_risk"].sum()) if "is_high_risk" in filtered.columns else 0
    n_types = filtered["asset_type"].nunique() if "asset_type" in filtered.columns else 0
    avg_risk = filtered["flood_risk"].mean() if "flood_risk" in filtered.columns else 0
    n_divisions = filtered["division"].nunique() if "division" in filtered.columns else 2

    type_counts = filtered["asset_type"].value_counts() if "asset_type" in filtered.columns else pd.Series()
    n_hospitals = int(type_counts.get("hospital", 0))
    n_schools = int(type_counts.get("school", 0))
    n_bridges = int(type_counts.get("bridge", 0))

    kpis = [
        (f"{n_total:,}", "Total Assets", ""),
        (f"{n_high:,}", "High Risk", "danger"),
        (f"{avg_risk:.3f}" if avg_risk else "N/A", "Avg Risk Score", "warning"),
        (f"{n_hospitals}", "Hospitals", ""),
        (f"{n_schools}", "Schools", ""),
        (f"{n_bridges}", "Bridges", "warning"),
        (str(n_types), "Asset Types", "success"),
        (str(n_divisions), "Divisions", ""),
    ]

    kpi_html = '<div class="kpi-strip">'
    for val, label, cls in kpis:
        kpi_html += (
            f'<div class="kpi-item {cls}">'
            f'<div class="kpi-val">{val}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>'
        )
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Map + floating overlay ──────────────────────────────────────────
    st.markdown(f'<div class="sec-label">Interactive Risk Map</div>',
                unsafe_allow_html=True)

    map_col, overlay_col = st.columns([5, 2], gap="small")

    with map_col:
        render_map(
            filtered,
            grid_gdf=grid_gdf,
            union_gdf=union_gdf if len(union_gdf) > 0 else None,
            hotspot_gdf=hotspot_gdf if len(hotspot_gdf) > 0 else None,
            cfg=cfg,
            is_dark=is_dark,
        )

    with overlay_col:
        render_analytics_overlay(filtered, union_gdf, is_dark)

    # ── Tabbed summary panel below map ──────────────────────────────────
    st.markdown(f'<div class="sec-label">Detailed Analysis</div>',
                unsafe_allow_html=True)

    tab_assets, tab_unions, tab_export = st.tabs(
        ["📊 Assets", "🏘️ Unions", "📥 Export"]
    )

    with tab_assets:
        _render_assets_tab(filtered, is_dark)

    with tab_unions:
        if len(union_gdf) > 0:
            render_risk_cards(union_gdf, is_dark)
        else:
            st.info("Union-level data available after full pipeline run.")

    with tab_export:
        _render_export_tab(filtered, union_gdf)

    # ── Footer ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sgmdi-footer">'
        'SGMDI &mdash; Smart Geospatial Mapping & Disaster Impact Intelligence '
        '| Rangpur & Rajshahi Divisions, Bangladesh'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------
def _render_assets_tab(infra: gpd.GeoDataFrame, is_dark: bool):
    """Assets tab: type breakdown + ranked table."""
    import plotly.express as px

    if "asset_type" not in infra.columns:
        st.info("No asset data available.")
        return

    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        # Donut chart of asset types
        type_counts = infra["asset_type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig = px.pie(
            type_counts, names="Type", values="Count", hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(font_size=10),
        )
        fig.update_traces(textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Ranked table
        display_cols = [
            c for c in ["risk_rank", "asset_type", "name", "flood_risk", "division"]
            if c in infra.columns
        ]
        if "flood_risk" in infra.columns:
            top = infra.sort_values("flood_risk", ascending=False).head(25)
        else:
            top = infra.head(25)

        if display_cols:
            st.dataframe(top[display_cols], height=350, hide_index=True)


def _render_export_tab(infra: gpd.GeoDataFrame, union_gdf: gpd.GeoDataFrame):
    """Export tab with download buttons."""
    st.markdown("Download filtered data in different formats.")

    c1, c2, c3 = st.columns(3)

    with c1:
        csv_data = infra.drop(columns=["geometry", "centroid"], errors="ignore")
        st.download_button(
            "📥 Download CSV",
            csv_data.to_csv(index=False),
            "sgmdi_risk_assets.csv",
            "text/csv",
            use_container_width=True,
        )

    with c2:
        export_gdf = infra.drop(columns=["centroid"], errors="ignore")
        st.download_button(
            "📥 Download GeoJSON",
            export_gdf.head(500).to_json(),
            "sgmdi_risk_assets.geojson",
            "application/json",
            use_container_width=True,
        )

    with c3:
        if len(union_gdf) > 0:
            union_csv = union_gdf.drop(columns=["geometry"], errors="ignore")
            st.download_button(
                "📥 Union Summary CSV",
                union_csv.to_csv(index=False),
                "sgmdi_union_summary.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.button("📥 Union Summary (N/A)", disabled=True,
                       use_container_width=True)

    # Preview
    st.markdown("---")
    st.markdown("**Data Preview**")
    preview_cols = [c for c in infra.columns if c not in ["geometry", "centroid"]]
    st.dataframe(infra[preview_cols].head(10), hide_index=True)


if __name__ == "__main__":
    main()

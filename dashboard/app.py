"""
SGMDI Streamlit Dashboard — 3-panel layout:
  1. Union-level risk cards (bottom carousel)
  2. Interactive Folium map (center)
  3. Sidebar analytics (right panel)
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.components.risk_cards import render_risk_cards
from dashboard.components.map_view import render_map
from dashboard.components.sidebar import render_sidebar, render_analytics


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SGMDI — Flood Risk Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* --- Page background & font --- */
    .stApp {
        background: linear-gradient(175deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }

    /* --- Header banner --- */
    .sgmdi-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0c4a6e 50%, #164e63 100%);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .sgmdi-header h1 {
        color: #f0f9ff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sgmdi-header .subtitle {
        color: #7dd3fc;
        font-size: 0.95rem;
        margin-top: 6px;
    }
    .sgmdi-header .badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.75rem;
        color: #7dd3fc;
        margin-top: 10px;
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: #1e293b;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* --- Card containers --- */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* --- Section headers --- */
    .section-header {
        color: #e2e8f0;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #334155;
    }

    /* --- Dataframes --- */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* --- Footer --- */
    .sgmdi-footer {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        padding: 20px 0;
        border-top: 1px solid #1e293b;
        margin-top: 32px;
    }

    /* --- Dividers --- */
    hr {
        border-color: #334155;
    }

    /* --- Expander --- */
    .streamlit-expanderHeader {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
@st.cache_data
def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_infrastructure(path: str) -> gpd.GeoDataFrame:
    if Path(path).exists():
        return gpd.read_file(path)
    return gpd.GeoDataFrame()


@st.cache_data
def load_geojson(path: str) -> gpd.GeoDataFrame:
    if Path(path).exists():
        return gpd.read_file(path)
    return gpd.GeoDataFrame()


@st.cache_data
def load_risk_csv(path: str) -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _clean_for_display(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop non-serializable columns like centroid Point objects."""
    drop = [c for c in gdf.columns if c == "centroid"]
    if drop:
        gdf = gdf.drop(columns=drop)
    return gdf


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    output_dir = Path("data/output")

    # Header
    st.markdown("""
    <div class="sgmdi-header">
        <h1>SGMDI — Flood Risk Intelligence</h1>
        <div class="subtitle">
            Smart Geospatial Mapping & Disaster Impact Intelligence
        </div>
        <span class="badge">Rangpur & Rajshahi Divisions, Bangladesh</span>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    infra_geojson_path = str(output_dir / "risk_ranked_assets.geojson")
    infra_csv_path = str(output_dir / "risk_ranked_assets.csv")
    union_path = str(output_dir / "union_risk_summary.geojson")
    hotspot_path = str(output_dir / "hotspot_clusters.geojson")
    grid_path = str(output_dir / "risk_grid.geojson")

    # Try GeoJSON first, fall back to raw GPKG + CSV
    infra = load_infrastructure(infra_geojson_path)
    if len(infra) == 0:
        raw_path = "data/raw/infrastructure_raw.gpkg"
        infra = load_infrastructure(raw_path)
        csv_data = load_risk_csv(infra_csv_path)
        if len(csv_data) > 0 and len(infra) > 0:
            for col in ["flood_risk", "risk_rank", "is_high_risk"]:
                if col in csv_data.columns:
                    infra[col] = csv_data[col].values[:len(infra)]

    union_gdf = load_geojson(union_path)
    hotspot_gdf = load_geojson(hotspot_path)
    grid_gdf = load_geojson(grid_path)

    # Check data availability
    if len(infra) == 0:
        st.error(
            "No pipeline output found. Run the pipeline first:\n\n"
            "```bash\npython -m pipeline.cli run --config config.yaml\n```"
        )
        st.stop()

    # Ensure centroid columns exist (but keep centroid as lon/lat floats, not Point)
    if "lon" not in infra.columns:
        pts = infra.geometry.representative_point()
        infra["lon"] = pts.x
        infra["lat"] = pts.y

    # --- Top metrics row ---
    n_total = len(infra)
    n_high = int(infra["is_high_risk"].sum()) if "is_high_risk" in infra.columns else 0
    n_types = infra["asset_type"].nunique() if "asset_type" in infra.columns else 0
    n_divisions = infra["division"].nunique() if "division" in infra.columns else 2

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, f"{n_total:,}", "Total Assets"),
        (m2, f"{n_high:,}", "High Risk"),
        (m3, str(n_types), "Asset Types"),
        (m4, str(n_divisions), "Divisions"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Sidebar (filters + metrics) ---
    filtered_infra, selected_types, risk_min, risk_max = render_sidebar(
        infra, union_gdf, grid_gdf
    )

    # Clean for display (remove Point objects)
    filtered_infra = _clean_for_display(filtered_infra)

    # --- Main content: Map + Analytics ---
    map_col, analytics_col = st.columns([5, 3], gap="medium")

    with map_col:
        st.markdown('<div class="section-header">Interactive Risk Map</div>',
                    unsafe_allow_html=True)
        render_map(
            filtered_infra,
            grid_gdf=grid_gdf,
            union_gdf=union_gdf if len(union_gdf) > 0 else None,
            hotspot_gdf=hotspot_gdf if len(hotspot_gdf) > 0 else None,
            cfg=cfg,
        )

    with analytics_col:
        st.markdown('<div class="section-header">Analytics</div>',
                    unsafe_allow_html=True)
        render_analytics(filtered_infra, union_gdf, grid_gdf)

    # --- Risk Cards (bottom) ---
    st.markdown("<br>", unsafe_allow_html=True)
    if len(union_gdf) > 0:
        render_risk_cards(union_gdf)
    else:
        # Show asset summary cards when union data isn't available yet
        render_risk_cards_from_infra(filtered_infra)

    # --- Footer ---
    st.markdown(
        '<div class="sgmdi-footer">'
        'SGMDI Pipeline &mdash; Smart Geospatial Mapping & Disaster Impact Intelligence<br>'
        'Flood Risk Assessment | Rangpur & Rajshahi Divisions, Bangladesh'
        '</div>',
        unsafe_allow_html=True,
    )


def render_risk_cards_from_infra(infra: gpd.GeoDataFrame):
    """Fallback: show asset type breakdown cards when union data is unavailable."""
    st.markdown('<div class="section-header">Infrastructure Summary</div>',
                unsafe_allow_html=True)

    if "asset_type" not in infra.columns:
        return

    type_counts = infra["asset_type"].value_counts()
    cols = st.columns(min(len(type_counts), 6))

    colors = {
        "hospital": "#ef4444", "school": "#3b82f6", "bridge": "#f59e0b",
        "road": "#6b7280", "cropland": "#22c55e", "irrigation": "#06b6d4",
        "flood_shelter": "#10b981", "embankment": "#059669",
        "railway": "#8b5cf6", "ferry_ghat": "#0ea5e9",
        "fishpond": "#67e8f9", "market": "#f43f5e",
    }

    for i, (atype, count) in enumerate(type_counts.items()):
        col = cols[i % len(cols)]
        color = colors.get(atype, "#64748b")
        col.markdown(
            f'<div style="'
            f'background: linear-gradient(135deg, {color}22, {color}08);'
            f'border: 1px solid {color}55;'
            f'border-radius: 10px;'
            f'padding: 14px 16px;'
            f'margin-bottom: 8px;'
            f'text-align: center;'
            f'">'
            f'<div style="font-size:1.5rem; font-weight:700; color:{color};">{count}</div>'
            f'<div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">'
            f'{atype.replace("_", " ")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

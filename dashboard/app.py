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


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    output_dir = Path("data/output")

    # Header
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0;">
            <h1 style="margin:0;">SGMDI — Smart Geospatial Mapping & Disaster Impact Intelligence</h1>
            <p style="color:#666; margin:4px 0;">
                Flood Risk Assessment | Rangpur & Rajshahi Divisions, Bangladesh
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        st.warning(
            "No pipeline output found. Run the pipeline first:\n\n"
            "```bash\npython -m pipeline.cli run --config config.yaml\n```"
        )
        st.stop()

    # Ensure centroid columns exist
    if "lon" not in infra.columns:
        infra["centroid"] = infra.geometry.representative_point()
        infra["lon"] = infra["centroid"].x
        infra["lat"] = infra["centroid"].y

    # --- Sidebar (filters + metrics) ---
    filtered_infra, selected_types, risk_min, risk_max = render_sidebar(
        infra, union_gdf, grid_gdf
    )

    # --- Main content ---
    # Map + Analytics split
    map_col, analytics_col = st.columns([2, 1])

    with map_col:
        st.subheader("Interactive Risk Map")
        render_map(
            filtered_infra,
            grid_gdf=grid_gdf,
            union_gdf=union_gdf if len(union_gdf) > 0 else None,
            hotspot_gdf=hotspot_gdf if len(hotspot_gdf) > 0 else None,
            cfg=cfg,
        )

    with analytics_col:
        render_analytics(filtered_infra, union_gdf, grid_gdf)

    # --- Risk Cards (bottom) ---
    st.markdown("---")
    if len(union_gdf) > 0:
        render_risk_cards(union_gdf)
    else:
        st.info(
            "Union-level risk cards will appear here once admin boundaries "
            "are available and the pipeline has been run."
        )

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#999; font-size:0.8em;'>"
        "SGMDI Pipeline — Smart Geospatial Mapping & Disaster Impact Intelligence | "
        "Rangpur & Rajshahi, Bangladesh"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

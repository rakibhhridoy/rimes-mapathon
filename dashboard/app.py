"""
Fermium Hazard Mapper — Multi-Hazard Dashboard
  Tabs: Pipeline Data (flood) | Flood Risk (regional) | Landslide Risk | ACTION
  Combines real pipeline output with Fermium-HazMapper reference panels.
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
from dashboard.components.map_view import render_map, render_region_map
from dashboard.components.sidebar import render_sidebar, render_analytics_overlay
from dashboard.components.cofactors import render_flood_cofactors, render_landslide_cofactors
from dashboard.components.risk_panels import (
    render_alert_banner,
    render_metric_row,
    render_infrastructure_table,
    render_vulnerability_chart,
    render_agri_risk,
    render_landslide_upazila,
    render_literature_panel,
)
from dashboard.data.constants import (
    DATA_SOURCES,
    LANDSLIDE_DATA,
    VULNERABILITY_CLASSES,
    MOBILE_FINANCING,
    PIPELINE_STAGES,
)
from dashboard.data.loader import (
    get_regional_assets,
    get_emergency_shelters,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fermium Hazard Mapper · SGMDI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme state
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

is_dark = st.session_state.theme == "dark"

# ---------------------------------------------------------------------------
# CSS — Fermium dark theme
# ---------------------------------------------------------------------------
BG = "#0a0e14" if is_dark else "#f8fafc"
BG2 = "#0d1822" if is_dark else "#ffffff"
BG3 = "#1a3a50" if is_dark else "#e2e8f0"
TEXT = "#c8d6e5" if is_dark else "#1e293b"
TEXT2 = "#5a8ab0" if is_dark else "#64748b"
ACCENT = "#3cb8de"
BORDER = "#1a3a50" if is_dark else "#cbd5e1"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');

    /* Global dark base */
    html, body, [class*="css"] {{
        background-color: {BG} !important;
        color: {TEXT} !important;
        font-family: 'DM Mono', 'Fira Code', 'Courier New', monospace !important;
    }}

    .main .block-container {{
        background: {BG} !important;
        padding-top: 0.5rem !important;
        max-width: 100% !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {BG2} !important;
        border-right: 1px solid {BORDER} !important;
    }}
    [data-testid="stSidebar"] .block-container {{ background: {BG2} !important; }}
    [data-testid="stSidebarContent"] {{ background: {BG2} !important; }}

    /* Tab styling */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background: #0d1f2d !important;
        border-bottom: 1px solid {BORDER} !important;
        gap: 2px;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        background: transparent !important;
        color: {TEXT2} !important;
        border: 1px solid #1a2a38 !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        padding: 6px 16px !important;
        font-family: 'DM Mono', monospace !important;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        background: {BG3} !important;
        color: {ACCENT} !important;
        border-color: {ACCENT} !important;
    }}
    [data-testid="stTabContent"] {{ background: {BG} !important; border: none !important; }}

    /* Radio buttons */
    [data-testid="stRadio"] label {{ color: {TEXT} !important; font-size: 12px !important; }}
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {{ color: {TEXT2} !important; }}

    /* Checkboxes */
    [data-testid="stCheckbox"] label {{ color: #a0c0d8 !important; font-size: 12px !important; }}

    /* Selectbox */
    [data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background: {BG2} !important;
        border-color: {BORDER} !important;
        color: {TEXT} !important;
    }}

    /* Slider */
    [data-testid="stSlider"] {{ color: {TEXT2} !important; }}

    /* DataFrames */
    [data-testid="stDataFrame"] {{ background: {BG2} !important; }}

    /* Expander */
    [data-testid="stExpander"] {{
        background: {BG2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] summary {{ color: {TEXT2} !important; font-size: 12px !important; }}

    /* Download button */
    [data-testid="stDownloadButton"] button {{
        background: #0d1f2d !important;
        color: {ACCENT} !important;
        border: 1px solid {BORDER} !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 11px !important;
    }}

    /* Divider */
    hr {{ border-color: {BORDER} !important; }}

    /* Captions */
    [data-testid="stCaptionContainer"] {{ color: #4a7a9a !important; }}

    /* Plotly charts */
    .js-plotly-plot .plotly {{ background: transparent !important; }}

    /* Metric */
    [data-testid="metric-container"] {{
        background: {BG2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 5px; background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

    /* KPI strip */
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

    .sec-label {{
        font-size: 0.7rem;
        color: {TEXT2};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 8px 0;
        font-weight: 600;
    }}

    .sgmdi-footer {{
        text-align: center;
        color: {TEXT2};
        font-size: 0.7rem;
        padding: 16px 0;
        margin-top: 24px;
        border-top: 1px solid {BORDER};
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}
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
        gdf = gpd.read_file(path)
        # Ensure all column names are strings (folium crashes on float keys)
        gdf.columns = [str(c) for c in gdf.columns]
        return gdf
    return gpd.GeoDataFrame()


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# HEADER / NAVBAR
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,#0d1f2d 0%,{BG} 70%);
                border-bottom:1px solid {BORDER};padding:16px 24px 12px;
                margin-bottom:0px;">
      <div style="display:flex;align-items:center;justify-content:space-between;
                  flex-wrap:wrap;gap:8px;">
        <div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#de3c3c;
                        box-shadow:0 0 6px #de3c3c88;animation:pulse 2s infinite;
                        flex-shrink:0;"></div>
            <span style="color:#de3c78;font-size:10px;font-weight:700;
                         letter-spacing:0.15em;">
              SGMDI · RESILIENCEAI · RIMES MAPATHON
            </span>
          </div>
          <h1 style="font-size:22px;font-weight:700;margin:0;color:#e8f4ff;
                     letter-spacing:-0.01em;font-family:'DM Mono',monospace;">
            FERMIUM HAZARD MAPPER
          </h1>
          <p style="color:{TEXT2};font-size:11px;margin:4px 0 0;letter-spacing:0.04em;">
            Smart Geospatial Mapping &amp; Disaster Impact Intelligence · Bangladesh
          </p>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <span style="background:#0d1f2d;border:1px solid {BORDER};color:{TEXT2};
                       font-size:9px;padding:3px 8px;border-radius:3px;">NW RANGPUR</span>
          <span style="background:#0d1f2d;border:1px solid {BORDER};color:{TEXT2};
                       font-size:9px;padding:3px 8px;border-radius:3px;">NE SYLHET</span>
          <span style="background:#0d1f2d;border:1px solid {BORDER};color:{TEXT2};
                       font-size:9px;padding:3px 8px;border-radius:3px;">SW COASTAL</span>
          <span style="background:#2e0d0d;border:1px solid #8a1a1a;color:#de3c3c;
                       font-size:9px;padding:3px 8px;border-radius:3px;">CHT LANDSLIDE</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search + theme toggle row
nav_l, nav_r = st.columns([6, 1])
with nav_l:
    search_query = st.text_input(
        "search", placeholder="Search assets by name or type...",
        label_visibility="collapsed",
    )
with nav_r:
    if st.button("Light" if is_dark else "Dark", help="Toggle theme"):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    output_dir = Path("data/output")

    # ── Load pipeline data ───────────────────────────────────────
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

    has_pipeline_data = len(infra) > 0

    if has_pipeline_data:
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

        # ── Sidebar filters ──────────────────────────────────────
        filtered, _, _, _, layers = render_sidebar(infra, union_gdf, grid_gdf, is_dark)
    else:
        # No pipeline data — create minimal sidebar with just layer toggles
        filtered = gpd.GeoDataFrame()
        layers = _render_minimal_sidebar()

    # ── MAIN TABS ────────────────────────────────────────────────
    if has_pipeline_data:
        tab_pipeline, tab_flood, tab_landslide, tab_action = st.tabs([
            "Pipeline Data",
            "Flood Risk (Regional)",
            "Landslide Risk",
            "ACTION — Emergency Response",
        ])
    else:
        tab_flood, tab_landslide, tab_action = st.tabs([
            "Flood Risk",
            "Landslide Risk",
            "ACTION — Emergency Response",
        ])
        tab_pipeline = None

    # ══════════════════════════════════════════════════════════════
    # TAB: PIPELINE DATA (real data from pipeline)
    # ══════════════════════════════════════════════════════════════
    if tab_pipeline is not None:
        with tab_pipeline:
            _render_pipeline_tab(filtered, union_gdf, hotspot_gdf, grid_gdf, cfg, is_dark, layers)

    # ══════════════════════════════════════════════════════════════
    # TAB: FLOOD RISK (regional, from Fermium-HazMapper)
    # ══════════════════════════════════════════════════════════════
    with tab_flood:
        _render_flood_tab(layers)

    # ══════════════════════════════════════════════════════════════
    # TAB: LANDSLIDE RISK
    # ══════════════════════════════════════════════════════════════
    with tab_landslide:
        _render_landslide_tab(layers)

    # ══════════════════════════════════════════════════════════════
    # TAB: ACTION — Emergency Response
    # ══════════════════════════════════════════════════════════════
    with tab_action:
        _render_action_tab()

    # ── Footer ────────────────────────────────────────────────────
    st.markdown(
        '<div class="sgmdi-footer">'
        'SGMDI — Smart Geospatial Mapping & Disaster Impact Intelligence '
        '| Multi-Hazard · Bangladesh'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_minimal_sidebar():
    """Minimal sidebar when no pipeline data is available."""
    import random

    with st.sidebar:
        st.markdown(
            """
            <div style="background:#0d1f2d;border:1px solid #1a3a50;border-radius:8px;
                        padding:10px 14px;margin-bottom:16px;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <div style="width:8px;height:8px;border-radius:50%;background:#dea03c;
                            flex-shrink:0;"></div>
                <span style="color:#dea03c;font-size:11px;font-weight:700;
                             letter-spacing:0.12em;">PIPELINE PENDING</span>
              </div>
              <div style="color:#4a7a9a;font-size:10px;margin-top:2px;">
                Run: <code>python -m pipeline.cli -c config.yaml run</code>
              </div>
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
        st.checkbox("HAND Index",       value=True,  key="show_hand")
        st.checkbox("Land Cover (LULC)",value=False, key="show_lulc")

        st.markdown("**Socioeconomic Layers**")
        st.checkbox("Population Density", value=False, key="show_popdens")
        st.checkbox("CVI Choropleth",     value=True,  key="show_cvi")
        st.checkbox("AlphaEarth Clusters",value=False, key="show_alphearth")

        st.divider()
        st.slider("Forecast horizon (days)", 1, 5, 3, key="forecast_days")
        st.selectbox(
            "Climate scenario",
            ["Current (2025)", "SSP2-4.5 (+0.5m SLR)", "SSP5-8.5 (+1.2m SLR)"],
            key="scenario",
        )
        st.divider()
        st.caption("Fermium Hazard Mapper · ResilienceAI")

    return {
        "osm_roads":      st.session_state.get("osm_roads", True),
        "osm_bridges":    st.session_state.get("osm_bridges", True),
        "osm_schools":    st.session_state.get("osm_schools", True),
        "osm_hospitals":  st.session_state.get("osm_hospitals", True),
        "show_dem":       st.session_state.get("show_dem", False),
        "show_hand":      st.session_state.get("show_hand", True),
        "show_lulc":      st.session_state.get("show_lulc", False),
        "show_popdens":   st.session_state.get("show_popdens", False),
        "show_cvi":       st.session_state.get("show_cvi", True),
        "show_alphearth": st.session_state.get("show_alphearth", False),
        "forecast_days":  st.session_state.get("forecast_days", 3),
        "scenario":       st.session_state.get("scenario", "Current (2025)"),
    }


# ---------------------------------------------------------------------------
# Pipeline Data tab (your existing functionality)
# ---------------------------------------------------------------------------
def _render_pipeline_tab(filtered, union_gdf, hotspot_gdf, grid_gdf, cfg, is_dark, layers):
    """Pipeline data tab with KPIs, map, analytics, and asset/union/export sub-tabs."""
    import plotly.express as px

    # ── KPI Strip ──────────────────────────────────────────────
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

    # ── Map + floating overlay ──────────────────────────────────
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
            layers=layers,
        )

    with overlay_col:
        render_analytics_overlay(filtered, union_gdf, is_dark)

    # ── Tabbed summary panel below map ──────────────────────────
    st.markdown(f'<div class="sec-label">Detailed Analysis</div>',
                unsafe_allow_html=True)

    tab_assets, tab_unions, tab_export = st.tabs(
        ["Assets", "Unions", "Export"]
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


# ---------------------------------------------------------------------------
# Flood Risk tab (Fermium-HazMapper regional)
# ---------------------------------------------------------------------------
def _render_flood_tab(layers):
    """Multi-region flood risk tab from Fermium-HazMapper."""

    flood_region = st.radio(
        "Flood sub-region",
        list(DATA_SOURCES.keys()),
        horizontal=True,
        key="flood_region",
        label_visibility="collapsed",
    )

    region_data = DATA_SOURCES[flood_region]

    # Alert banner
    render_alert_banner(
        region_data["alert_level"],
        region_data["hazard"],
        region_data["risk_score"],
    )

    # KPI metrics row
    render_metric_row(flood_region)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # Map + right panel
    map_col, panel_col = st.columns([3, 2], gap="medium")

    with map_col:
        st.markdown(
            f"<span style='color:{region_data['accent']};font-size:11px;"
            f"font-weight:700;letter-spacing:0.1em;'>"
            f"INTERACTIVE RISK MAP — {flood_region.upper()}</span>",
            unsafe_allow_html=True,
        )
        render_region_map(flood_region, region_data, layers, map_key=f"flood_map_{flood_region}")

    with panel_col:
        flood_assets = get_regional_assets(
            flood_region, region_data.get("center"), radius_deg=0.5)
        render_infrastructure_table(
            flood_assets,
            key_prefix=f"infra_{flood_region}",
        )

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

        with st.expander("Data Sources for this region", expanded=False):
            for name, src, use in region_data["sources"]:
                st.markdown(
                    f"<div style='border-bottom:1px solid #1a2a38;padding:5px 0;font-size:11px;'>"
                    f"<span style='color:#e8f4ff;font-weight:600;'>{name}</span>"
                    f"<span style='color:#5a8ab0;margin-left:8px;'>{src}</span><br>"
                    f"<span style='color:#4a7a9a;font-size:10px;'>{use}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # CVI chart + Agri risk
    chart_col, agri_col = st.columns([1, 1], gap="medium")

    with chart_col:
        render_vulnerability_chart(flood_region)

    with agri_col:
        render_agri_risk(flood_region)

    # Vulnerability class taxonomy
    with st.expander("5-Class Community Vulnerability Taxonomy", expanded=False):
        cvi_cols = st.columns(5)
        for col, vc in zip(cvi_cols, VULNERABILITY_CLASSES):
            with col:
                st.markdown(
                    f"""
                    <div style="background:{vc['color']}22;border:1px solid {vc['color']}66;
                                border-radius:8px;padding:10px;font-size:10px;">
                      <div style="color:{vc['text_color']};font-weight:700;font-size:12px;
                                  margin-bottom:6px;">{vc['cls']} · {vc['label']}</div>
                      <div style="color:#4a7a9a;font-size:9px;margin-bottom:4px;">
                        CVI: {vc['cvi_range']}</div>
                      <div style="color:#5a8ab0;margin-bottom:3px;">
                        Housing: {vc['housing'][:55]}...</div>
                      <div style="color:#5a8ab0;margin-bottom:3px;">
                        Agri: {vc['agri'][:55]}...</div>
                      <div style="color:#5a8ab0;margin-bottom:3px;">
                        Infra: {vc['infra'][:55]}...</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Cofactors
    render_flood_cofactors(flood_region)


# ---------------------------------------------------------------------------
# Landslide Risk tab
# ---------------------------------------------------------------------------
def _render_landslide_tab(layers):
    """Landslide risk tab from Fermium-HazMapper."""

    # Alert banner
    render_alert_banner(
        LANDSLIDE_DATA["alert_level"],
        LANDSLIDE_DATA["hazard"],
        LANDSLIDE_DATA["risk_score"],
    )

    # 5-day rainfall gauge
    rain_mm = LANDSLIDE_DATA["rainfall_5day_mm"]
    rain_tier = (
        "WARNING" if rain_mm > 161 else "WATCH" if rain_mm > 71 else "NORMAL"
    )
    rain_color = "#de7a3c" if rain_mm > 161 else "#dea03c" if rain_mm > 71 else "#3cdea0"

    st.markdown(
        f"""
        <div style="background:#0d1822;border:1px solid {rain_color}55;border-radius:8px;
                    padding:10px 16px;margin-bottom:16px;display:flex;
                    align-items:center;gap:16px;font-size:12px;">
          <span style="color:#5a8ab0;">5-day antecedent rainfall (BMD/CHIRPS):</span>
          <span style="color:{rain_color};font-weight:700;font-size:16px;">{rain_mm} mm</span>
          <span style="color:{rain_color};">{rain_tier}</span>
          <span style="color:#4a7a9a;margin-left:auto;font-size:10px;">
            Threshold: 71mm (Watch) · 161mm (Warning) · 250mm (Emergency)
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    render_metric_row(landslide=True)
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # Map + panel
    ls_map_col, ls_panel_col = st.columns([3, 2], gap="medium")

    with ls_map_col:
        st.markdown(
            "<span style='color:#de3c78;font-size:11px;font-weight:700;"
            "letter-spacing:0.1em;'>LANDSLIDE SUSCEPTIBILITY MAP — CHT</span>",
            unsafe_allow_html=True,
        )
        render_region_map("CHT Landslide", LANDSLIDE_DATA, layers, map_key="landslide_map")

    with ls_panel_col:
        ls_assets = get_regional_assets(
            "CHT Landslide", LANDSLIDE_DATA.get("center"), radius_deg=0.5)
        render_infrastructure_table(
            ls_assets,
            key_prefix="ls_infra",
        )

        with st.expander("Data Sources — CHT Landslide", expanded=False):
            for name, src, use in LANDSLIDE_DATA["sources"]:
                st.markdown(
                    f"<div style='border-bottom:1px solid #1a2a38;padding:5px 0;font-size:11px;'>"
                    f"<span style='color:#e8f4ff;font-weight:600;'>{name}</span>"
                    f"<span style='color:#5a8ab0;margin-left:8px;'>{src}</span><br>"
                    f"<span style='color:#4a7a9a;font-size:10px;'>{use}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # Upazila susceptibility table
    render_landslide_upazila()

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # Literature panel
    render_literature_panel()

    # Cofactors pane
    render_landslide_cofactors()


# ---------------------------------------------------------------------------
# ACTION tab
# ---------------------------------------------------------------------------
def _render_action_tab():
    """Emergency response coordination tab."""

    st.markdown(
        """
        <div style="background:#2e0d0d;border:1px solid #8a1a1a;border-radius:8px;
                    padding:14px 20px;margin-bottom:20px;">
          <span style="color:#de3c3c;font-size:13px;font-weight:700;
                       letter-spacing:0.1em;">
            MULTI-HAZARD EMERGENCY RESPONSE COORDINATION
          </span>
          <div style="color:#8a4a4a;font-size:11px;margin-top:6px;">
            Active alerts: NE Sylhet (RED · Flash Flood) · CHT (RED · Landslide Warning) ·
            NW Rangpur (ORANGE · Fluvial) · SW Coastal (YELLOW · Surge Watch)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    act_left, act_right = st.columns([1, 1], gap="medium")

    # Emergency Shelters
    with act_left:
        st.markdown(
            "<span style='color:#de3c78;font-size:11px;font-weight:700;"
            "letter-spacing:0.1em;'>EMERGENCY SHELTER STATUS</span>",
            unsafe_allow_html=True,
        )
        for s in get_emergency_shelters():
            status_col = "#de3c3c" if s["status"] == "AT CAPACITY" else "#3cdea0"
            cvi_col = "#de3c3c" if s["cvi_class"] == 5 else "#de7a3c" if s["cvi_class"] == 4 else "#dea03c"
            st.markdown(
                f"""
                <div style="background:#0d1822;border:1px solid {status_col}33;
                            border-radius:8px;padding:10px 14px;margin-bottom:8px;
                            display:flex;justify-content:space-between;align-items:center;">
                  <div>
                    <div style="color:#e8f4ff;font-size:12px;font-weight:600;">
                      {s['name']}</div>
                    <div style="color:#5a8ab0;font-size:10px;">{s['region']}</div>
                  </div>
                  <div style="text-align:right;">
                    <div style="color:{status_col};font-size:11px;font-weight:700;">
                      {s['status']}</div>
                    <div style="color:#4a7a9a;font-size:10px;">
                      Cap: {s['capacity']:,} · CVI
                      <span style="color:{cvi_col};">Class {s['cvi_class']}</span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Mobile Financing
    with act_right:
        st.markdown(
            "<span style='color:#3cdea0;font-size:11px;font-weight:700;"
            "letter-spacing:0.1em;'>MOBILE FINANCING & CASH TRANSFERS</span>",
            unsafe_allow_html=True,
        )
        for mf in MOBILE_FINANCING:
            status_col = "#3cdea0" if mf["status"] == "ACTIVE" else "#dea03c"
            st.markdown(
                f"""
                <div style="background:#0d1822;border:1px solid {status_col}22;
                            border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                      <div style="color:#e8f4ff;font-size:12px;font-weight:600;">
                        {mf['program']}</div>
                      <div style="color:#5a8ab0;font-size:10px;">{mf['target']}</div>
                    </div>
                    <div style="text-align:right;flex-shrink:0;margin-left:12px;">
                      <div style="color:{status_col};font-size:11px;font-weight:700;">
                        {mf['status']}</div>
                      <div style="color:#dea03c;font-size:12px;font-weight:700;">
                        BDT {mf['amount_bdt']:,}</div>
                    </div>
                  </div>
                  <div style="margin-top:8px;background:#0a0e14;border-radius:4px;
                              height:6px;overflow:hidden;">
                    <div style="width:{min(100, int(mf['disbursed'] / max(mf['disbursed'] + 1, 1) * 100) if mf['disbursed'] > 0 else 5)}%;
                                height:6px;background:{status_col};border-radius:4px;"></div>
                  </div>
                  <div style="color:#4a7a9a;font-size:9px;margin-top:3px;">
                    Disbursed: {mf['disbursed']:,} households
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Alert Dispatch Logic
    st.markdown(
        "<span style='color:#3cb8de;font-size:11px;font-weight:700;"
        "letter-spacing:0.1em;'>REAL-TIME ALERT DISPATCH TRIGGERS</span>",
        unsafe_allow_html=True,
    )
    dispatch_cols = st.columns(4)
    triggers = [
        ("Flood Trigger", "BWDB gauge > warning level OR Kriging hazard > 0.75 in populated zone", "#3cb8de"),
        ("Landslide Trigger", "5-day rainfall > 71mm AND RF susceptibility = High/Very High", "#dea03c"),
        ("Surge Trigger", "BIWTA gauge > polder height OR BGD Met cyclone track cone", "#de3c78"),
        ("Dispatch Channel", "Twilio SMS to upazila DDM + Union Parishad; SendGrid to BDRCS + UNDP", "#3cdea0"),
    ]
    for col, (title, val, color) in zip(dispatch_cols, triggers):
        with col:
            st.markdown(
                f"""
                <div style="background:#0d1822;border:1px solid {color}33;
                            border-radius:8px;padding:12px;">
                  <div style="color:{color};font-size:10px;font-weight:700;
                              letter-spacing:0.08em;margin-bottom:6px;">
                    {title.upper()}</div>
                  <div style="color:#7a9ab8;font-size:11px;line-height:1.6;">{val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Pipeline Stage Summary
    with st.expander("7-Stage Processing Pipeline — Full Spec", expanded=False):
        for s in PIPELINE_STAGES:
            with st.expander(f"Stage {s['id']:02d} · {s['title']} — {s['desc']}", expanded=False):
                for item in s["items"]:
                    st.markdown(
                        f"<div style='color:#a0c0d8;font-size:12px;padding:3px 0;"
                        f"border-left:2px solid #1a3a50;padding-left:10px;margin-bottom:3px;'>"
                        f"{item}</div>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Assets / Export sub-tabs (unchanged from original)
# ---------------------------------------------------------------------------
def _render_assets_tab(infra: gpd.GeoDataFrame, is_dark: bool):
    """Assets tab: type breakdown + ranked table."""
    import plotly.express as px

    if "asset_type" not in infra.columns:
        st.info("No asset data available.")
        return

    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
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
            "Download CSV",
            csv_data.to_csv(index=False),
            "sgmdi_risk_assets.csv",
            "text/csv",
            use_container_width=True,
        )

    with c2:
        export_gdf = infra.drop(columns=["centroid"], errors="ignore")
        st.download_button(
            "Download GeoJSON",
            export_gdf.head(500).to_json(),
            "sgmdi_risk_assets.geojson",
            "application/json",
            use_container_width=True,
        )

    with c3:
        if len(union_gdf) > 0:
            union_csv = union_gdf.drop(columns=["geometry"], errors="ignore")
            st.download_button(
                "Union Summary CSV",
                union_csv.to_csv(index=False),
                "sgmdi_union_summary.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.button("Union Summary (N/A)", disabled=True,
                       use_container_width=True)

    st.markdown("---")
    st.markdown("**Data Preview**")
    preview_cols = [c for c in infra.columns if c not in ["geometry", "centroid"]]
    st.dataframe(infra[preview_cols].head(10), hide_index=True)


if __name__ == "__main__":
    main()

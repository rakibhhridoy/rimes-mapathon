"""
Fermium Hazard Mapper — public flood risk dashboard (SGMDI).

Tabs: Risk map | Rankings & export | Preparedness | About the data.

The dashboard is open to everyone: there is no sign-in. Everything it shows
comes from pipeline outputs in data/output; when an output is missing the app
says so instead of substituting placeholder values.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.components.effects import inject_wave_animation

st.set_page_config(
    page_title="Fermium Hazard Mapper · SGMDI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

from dashboard.components.about import render_about_tab
from dashboard.components.detail_panel import render_detail_panel
from dashboard.components.landslide_view import render_landslide_tab
from dashboard.components.map_view import render_map
from dashboard.components.preparedness import render_preparedness_tab
from dashboard.components.risk_cards import render_risk_cards
from dashboard.components.sidebar import render_analytics_overlay, render_sidebar
from dashboard.data.loader import (
    load_gdf_fast,
    load_landslide_model,
    load_pipeline_metadata,
)
from dashboard.data.regions import (
    REGION_CONFIGS,
    available_regions,
    region_config,
    region_label,
)

is_dark = st.session_state.theme == "dark"

# ---------------------------------------------------------------------------
# CSS — Fermium dark theme
# ---------------------------------------------------------------------------
BG = "#0a0e14" if is_dark else "#f8fafc"
BG2 = "#0d1822" if is_dark else "#ffffff"
BG3 = "#1e3a52" if is_dark else "#e2e8f0"
TEXT = "#f0f6ff" if is_dark else "#1e293b"
TEXT2 = "#8ab4d4" if is_dark else "#64748b"
ACCENT = "#00d4ff"
BORDER = "#1e3a52" if is_dark else "#cbd5e1"
GLOW = "rgba(0,212,255,0.12)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

    /* Global dark base */
    html, body, [class*="css"] {{
        background-color: {BG} !important;
        color: {TEXT} !important;
        font-family: 'Inter', 'DM Sans', -apple-system, sans-serif !important;
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
        border-radius: 6px !important;
        font-size: 12px !important;
        padding: 6px 16px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
        background: rgba(0,212,255,0.06) !important;
        border-color: {BORDER} !important;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        background: rgba(0,212,255,0.10) !important;
        color: {ACCENT} !important;
        border-color: {ACCENT} !important;
        box-shadow: 0 0 12px {GLOW};
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
        background: rgba(13,24,34,0.7) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stExpander"] summary {{ color: {TEXT2} !important; font-size: 12px !important; }}

    /* Download button */
    [data-testid="stDownloadButton"] button {{
        background: rgba(13,31,45,0.8) !important;
        backdrop-filter: blur(8px) !important;
        color: {ACCENT} !important;
        border: 1px solid {BORDER} !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stDownloadButton"] button:hover {{
        box-shadow: 0 0 16px {GLOW};
        border-color: {ACCENT} !important;
    }}

    /* Divider */
    hr {{ border-color: {BORDER} !important; }}

    /* Captions */
    [data-testid="stCaptionContainer"] {{ color: #4a7a9a !important; }}

    /* Plotly charts */
    .js-plotly-plot .plotly {{ background: transparent !important; }}

    /* Metric */
    [data-testid="metric-container"] {{
        background: rgba(13,24,34,0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 5px; background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

    /* Glassmorphism KPI strip */
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
        background: rgba(13,24,34,0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04);
        transition: all 0.25s ease;
    }}
    .kpi-item:hover {{
        border-color: rgba(0,212,255,0.35);
        box-shadow: 0 4px 24px rgba(0,212,255,0.12), inset 0 1px 0 rgba(255,255,255,0.06);
        transform: translateY(-1px);
    }}
    .kpi-item .kpi-val {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {ACCENT};
        line-height: 1;
        font-family: 'DM Mono', monospace;
    }}
    .kpi-item .kpi-label {{
        font-size: 0.62rem;
        color: {TEXT2};
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 5px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }}
    .kpi-item.danger .kpi-val {{ color: #ef4444; }}
    .kpi-item.danger {{ border-color: rgba(239,68,68,0.2); }}
    .kpi-item.danger:hover {{ box-shadow: 0 4px 24px rgba(239,68,68,0.1); }}
    .kpi-item.warning .kpi-val {{ color: #f59e0b; }}
    .kpi-item.warning {{ border-color: rgba(245,158,11,0.2); }}
    .kpi-item.warning:hover {{ box-shadow: 0 4px 24px rgba(245,158,11,0.1); }}
    .kpi-item.success .kpi-val {{ color: #22c55e; }}
    .kpi-item.success {{ border-color: rgba(34,197,94,0.2); }}
    .kpi-item.success:hover {{ box-shadow: 0 4px 24px rgba(34,197,94,0.1); }}

    .sec-label {{
        font-size: 0.7rem;
        color: {TEXT2};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 8px 0;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }}

    .sgmdi-footer {{
        text-align: center;
        color: {TEXT2};
        font-size: 0.7rem;
        padding: 16px 0;
        margin-top: 24px;
        border-top: 1px solid {BORDER};
        font-family: 'Inter', sans-serif;
    }}

    /* Tab content fade-in */
    [data-testid="stTabContent"] > div {{
        animation: fadeIn 0.3s ease-out;
    }}

    /* Table row hover */
    [data-testid="stDataFrame"] tbody tr:hover {{
        background: rgba(0,212,255,0.04) !important;
    }}

    /* Expander hover glow */
    [data-testid="stExpander"]:hover {{
        border-color: rgba(0,212,255,0.25) !important;
        box-shadow: 0 0 12px {GLOW};
    }}

    /* Loading skeleton shimmer */
    .skeleton {{
        background: linear-gradient(90deg, {BG2} 25%, #14202e 50%, {BG2} 75%);
        background-size: 200% 100%;
        animation: shimmer 1.8s ease-in-out infinite;
        border-radius: 8px;
    }}
    .skeleton-strip {{
        display: flex; gap: 12px; margin-bottom: 16px;
    }}
    .skeleton-card {{
        flex: 1; min-width: 130px; height: 72px;
        border-radius: 12px;
    }}
    .skeleton-map {{
        width: 100%; height: 400px; border-radius: 10px;
        margin-bottom: 16px;
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}

    @keyframes glowPulse {{
        0%, 100% {{ box-shadow: 0 0 8px rgba(0,212,255,0.08); }}
        50% {{ box-shadow: 0 0 16px rgba(0,212,255,0.18); }}
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(4px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    /* ── Phase 5: Accessibility ─────────────────────────────── */

    /* Focus states for keyboard navigation */
    button:focus-visible,
    [data-testid="stCheckbox"] input:focus-visible + label,
    [data-baseweb="tab"]:focus-visible,
    [data-baseweb="select"]:focus-visible,
    a:focus-visible {{
        outline: 2px solid #00d4ff !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 4px rgba(0,212,255,0.18) !important;
    }}

    /* Ensure minimum contrast on secondary text (WCAG AA 4.5:1) */
    /* #8ab4d4 on #0a0e14 = ~5.8:1 ratio — passes AA */
    /* Bump sidebar muted text for readability */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: #7abcd8 !important;
    }}

    /* High-contrast mode for data values */
    [data-testid="stMetricValue"] {{
        color: #f0f6ff !important;
        font-family: 'DM Mono', monospace !important;
    }}

    /* ── Tablet responsive (< 1024px) ──────────────────────── */
    @media (max-width: 1024px) {{
        .kpi-strip {{
            flex-wrap: wrap;
        }}
        .kpi-item {{
            min-width: calc(50% - 8px);
            flex: 0 0 calc(50% - 8px);
        }}
    }}

</style>
""", unsafe_allow_html=True)

# Water wave background animation
inject_wave_animation()

# Fermium logo beside the Streamlit menu (top-right)
import base64
import os

_logo_path = os.path.join(os.path.dirname(__file__), "..", "fermium.png")
if os.path.exists(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
    st.markdown(
        f"""
        <div style="position:fixed;top:-4px;left:52px;z-index:999999;">
            <img src="data:image/png;base64,{_logo_b64}" alt="Fermium Systems"
                 style="height:70px;opacity:0.9;">
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,#0d1f2d 0%,{BG} 70%);
                border-bottom:1px solid {BORDER};padding:16px 24px 12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;
                  flex-wrap:wrap;gap:8px;">
        <div>
          <div style="color:#5a8ab0;font-size:10px;font-weight:700;
                      letter-spacing:0.15em;margin-bottom:4px;">
            SGMDI · SMART GEOSPATIAL MAPPING &amp; DISASTER IMPACT INTELLIGENCE
          </div>
          <h1 style="font-size:22px;font-weight:700;margin:0;color:#f0f6ff;
                     letter-spacing:-0.01em;font-family:'DM Mono',monospace;">
            FERMIUM HAZARD MAPPER
          </h1>
          <p style="color:{TEXT2};font-size:11px;margin:4px 0 0;letter-spacing:0.04em;">
            Flood risk to infrastructure · Rangpur &amp; Rajshahi divisions, Bangladesh
          </p>
        </div>
        <div style="max-width:420px;background:#2e2a0d;border:1px solid #dea03c66;
                    border-radius:8px;padding:10px 14px;">
          <div style="color:#dea03c;font-size:10px;font-weight:700;
                      letter-spacing:0.1em;margin-bottom:3px;">RESEARCH PROTOTYPE</div>
          <div style="color:#c8a86a;font-size:11px;line-height:1.5;">
            Modelled susceptibility, not a forecast or an official warning.
            For warnings use FFWC and BMD; in an emergency dial 999.
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    regions = available_regions()
    if not regions:
        _render_no_data()
        return

    region = _select_region(regions)
    cfg = region_config(region)

    infra = load_gdf_fast(region, "risk_ranked_assets")
    union_gdf = load_gdf_fast(region, "union_risk_summary")
    hotspot_gdf = load_gdf_fast(region, "hotspot_clusters")

    if len(infra) == 0:
        # A landslide region's result is a susceptibility surface, not scored
        # assets, so it gets its own view rather than an empty asset table.
        if load_landslide_model(region):
            _render_landslide_region(region)
            return
        st.info(
            f"{region_label(region)} has no results yet. "
            "Its pipeline may still be running."
        )
        render_about_tab(region)
        return

    if "lon" not in infra.columns or "lat" not in infra.columns:
        pts = infra.geometry.representative_point()
        infra = infra.assign(lon=pts.x, lat=pts.y)
    infra = infra.drop(columns=["centroid"], errors="ignore")

    search_query = st.text_input(
        "Search assets",
        placeholder="Search assets by name or type…",
        label_visibility="collapsed",
    )
    if search_query:
        # regex=False: the query is user input, not a pattern
        mask = infra["name"].str.contains(search_query, case=False, na=False, regex=False)
        if "asset_type" in infra.columns:
            mask |= infra["asset_type"].str.contains(
                search_query, case=False, na=False, regex=False
            )
        infra = infra[mask]
        if len(infra) == 0:
            st.warning(f'No assets match "{search_query}".')
            return

    filtered, layers = render_sidebar(region, infra, is_dark)

    tab_map, tab_rank, tab_prep, tab_about = st.tabs(
        ["Risk map", "Rankings & export", "Preparedness", "About the data"]
    )

    with tab_map:
        _render_map_tab(region, filtered, union_gdf, hotspot_gdf, cfg, layers)

    with tab_rank:
        _render_rankings_tab(region, filtered, union_gdf)

    with tab_prep:
        render_preparedness_tab(region)

    with tab_about:
        render_about_tab(region)

    meta = load_pipeline_metadata(region) or {}
    st.markdown(
        '<div class="sgmdi-footer">'
        'Fermium Hazard Mapper · SGMDI — open research prototype · '
        'Infrastructure data © OpenStreetMap contributors (ODbL) · '
        f'Data processed {str(meta.get("generated_at", "—"))[:10]}'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_landslide_region(region: str):
    """Tabs for a region whose model estimates slope failure, not flooding."""
    tab_slide, tab_prep, tab_about = st.tabs(
        ["Landslide susceptibility", "Preparedness", "About the data"]
    )
    with tab_slide:
        render_landslide_tab(region)
    with tab_prep:
        render_preparedness_tab(region)
    with tab_about:
        render_about_tab(region)


def _select_region(regions: list[str]) -> str:
    """Region picker. Hidden when only one region has results."""
    if len(regions) == 1:
        return regions[0]

    pending = [rid for rid in REGION_CONFIGS if rid not in regions]
    choice = st.radio(
        "Region",
        regions,
        format_func=region_label,
        horizontal=True,
        key="region_select",
    )
    if pending:
        st.caption(
            "Still being processed: "
            + ", ".join(REGION_CONFIGS[rid][0] for rid in pending)
        )
    return choice


def _render_no_data():
    """Shown when pipeline outputs are absent — never mock content."""
    st.warning(
        "No pipeline results are available yet, so there is nothing to map.\n\n"
        "Generate them with `python -m pipeline.cli -c config.yaml run`, then "
        "build the display caches with `python preprocess_cache.py`."
    )
    render_about_tab()


def _render_map_tab(region, filtered, union_gdf, hotspot_gdf, cfg, layers):
    """KPI strip, interactive map and analytics."""
    n_total = len(filtered)
    n_high = int(filtered["is_high_risk"].sum()) if "is_high_risk" in filtered.columns else 0
    avg_risk = filtered["flood_risk"].mean() if "flood_risk" in filtered.columns else None
    type_counts = (
        filtered["asset_type"].value_counts()
        if "asset_type" in filtered.columns else {}
    )

    kpis = [
        (f"{n_total:,}", "Assets shown", ""),
        (f"{n_high:,}", "High susceptibility", "danger"),
        (f"{avg_risk:.3f}" if avg_risk is not None else "—", "Mean score", "warning"),
        (f"{int(type_counts.get('hospital', 0)):,}", "Hospitals & clinics", ""),
        (f"{int(type_counts.get('school', 0)):,}", "Schools", ""),
        (f"{int(type_counts.get('bridge', 0)):,}", "Bridges", "warning"),
        (f"{int(type_counts.get('flood_shelter', 0)):,}", "Flood shelters", "success"),
        (f"{int(type_counts.get('cropland', 0)):,}", "Cropland parcels", ""),
    ]
    st.markdown(
        '<div class="kpi-strip">'
        + "".join(
            f'<div class="kpi-item {cls}"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-label">{label}</div></div>'
            for val, label, cls in kpis
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "“High susceptibility” counts assets whose modelled score exceeds the "
        "threshold set in config.yaml."
    )

    st.markdown('<div class="sec-label">Interactive risk map</div>',
                unsafe_allow_html=True)

    map_col, overlay_col = st.columns([5, 2], gap="small")
    with map_col:
        render_map(
            region,
            filtered,
            grid_gdf=None,
            union_gdf=union_gdf if len(union_gdf) > 0 else None,
            hotspot_gdf=hotspot_gdf if len(hotspot_gdf) > 0 else None,
            cfg=cfg,
            is_dark=is_dark,
            layers=layers,
        )
    with overlay_col:
        render_analytics_overlay(filtered, is_dark)


def _render_rankings_tab(region, filtered, union_gdf):
    """Ranked assets, union summaries and downloads."""
    import plotly.express as px

    st.markdown('<div class="sec-label">Highest-scoring assets</div>',
                unsafe_allow_html=True)

    if "asset_type" not in filtered.columns:
        st.info("No asset attributes available.")
        return

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        type_counts = filtered["asset_type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig = px.pie(type_counts, names="Type", values="Count", hole=0.55,
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT, margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(font_size=10),
        )
        fig.update_traces(textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig, width="stretch")

    with c2:
        cols = [c for c in ["risk_rank", "asset_type", "name", "flood_risk",
                            "flood_probability", "division"]
                if c in filtered.columns]
        top = (filtered.sort_values("flood_risk", ascending=False)
               if "flood_risk" in filtered.columns else filtered).head(25)
        if cols:
            st.dataframe(
                top[cols].rename(columns={
                    "risk_rank": "Rank", "asset_type": "Type", "name": "Name",
                    "flood_risk": "Score", "flood_probability": "P(flood)",
                    "division": "Division",
                }),
                height=350, hide_index=True,
            )

    if "name" in filtered.columns and "flood_risk" in filtered.columns:
        top_assets = filtered.sort_values("flood_risk", ascending=False).head(50)
        choice = st.selectbox(
            "Inspect an asset",
            ["— Select an asset —"] + top_assets["name"].fillna("unnamed").tolist(),
            key="asset_selector",
        )
        if choice != "— Select an asset —":
            row = top_assets[top_assets["name"].fillna("unnamed") == choice].iloc[0]
            from dashboard.data.loader import get_kriging_ci_at_point

            def _num(key):
                val = row.get(key)
                return float(val) if isinstance(val, (int, float)) and val == val else None

            st.session_state.selected_asset = {
                "name": str(row.get("name", "unnamed")),
                "asset_type": str(row.get("asset_type", "")),
                "flood_risk": _num("flood_risk"),
                "flood_probability": _num("flood_probability"),
                "risk_rank": int(row.get("risk_rank", 0) or 0),
                "division": str(row.get("division", "")),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "kriging_ci": get_kriging_ci_at_point(
                    region, float(row["lat"]), float(row["lon"])),
                "cell_hazard": _num("cell_hazard"),
                "cell_exposure": _num("cell_exposure"),
                "cell_vulnerability": _num("cell_vulnerability"),
                "cell_composite_risk": _num("cell_composite_risk"),
            }
            render_detail_panel()

    st.markdown('<div class="sec-label">Union summaries</div>', unsafe_allow_html=True)
    render_risk_cards(union_gdf, is_dark)

    st.markdown('<div class="sec-label">Download</div>', unsafe_allow_html=True)
    st.caption(
        "Outputs may be reused with attribution, subject to the source data "
        "licences listed under “About the data”."
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Assets (CSV)",
            filtered.drop(columns=["geometry"], errors="ignore").to_csv(index=False),
            "hazmapper_assets.csv", "text/csv", width="stretch",
        )
    with d2:
        st.download_button(
            "Assets, top 500 (GeoJSON)",
            filtered.head(500).to_json(),
            "hazmapper_assets.geojson", "application/geo+json", width="stretch",
        )
    with d3:
        if len(union_gdf) > 0:
            st.download_button(
                "Union summary (CSV)",
                union_gdf.drop(columns=["geometry"], errors="ignore").to_csv(index=False),
                "hazmapper_union_summary.csv", "text/csv", width="stretch",
            )
        else:
            st.button("Union summary (unavailable)", disabled=True,
                      width="stretch")


if __name__ == "__main__":
    main()

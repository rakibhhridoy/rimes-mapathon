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

# ---------------------------------------------------------------------------
# CSS — light theme; every colour comes from dashboard/data/theme.py
# ---------------------------------------------------------------------------
from dashboard.data import theme

BG = theme.BG
BG2 = theme.SURFACE
BG3 = theme.SURFACE_ALT
TEXT = theme.TEXT
TEXT2 = theme.TEXT_MUTED
ACCENT = theme.ACCENT
BORDER = theme.BORDER
GLOW = theme.GLOW

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

    /* Global light base */
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
        background: #eaf2fc !important;
        border-bottom: 1px solid {BORDER} !important;
        gap: 2px;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        background: transparent !important;
        color: {TEXT2} !important;
        border: 1px solid #dbe3ec !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        padding: 6px 16px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
        background: rgba(42,120,214,0.06) !important;
        border-color: {BORDER} !important;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        background: rgba(42,120,214,0.10) !important;
        color: {ACCENT} !important;
        border-color: {ACCENT} !important;
        box-shadow: 0 0 12px {GLOW};
    }}
    [data-testid="stTabContent"] {{ background: {BG} !important; border: none !important; }}

    /* Radio buttons */
    [data-testid="stRadio"] label {{ color: {TEXT} !important; font-size: 12px !important; }}
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {{ color: {TEXT2} !important; }}

    /* Checkboxes */
    [data-testid="stCheckbox"] label {{ color: #475569 !important; font-size: 12px !important; }}

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
        background: rgba(255,255,255,0.95) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stExpander"] summary {{ color: {TEXT2} !important; font-size: 12px !important; }}

    /* Download button */
    [data-testid="stDownloadButton"] button {{
        background: rgba(255,255,255,0.94) !important;
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
        background: rgba(255,255,255,0.92) !important;
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
        flex-direction: column;
        gap: 2px;
    }}
    .kpi-item {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 10px;
        border: none;
        border-bottom: 1px solid {BORDER};
        border-radius: 0;
        background: transparent;
        box-shadow: none;
        padding: 5px 2px;
        text-align: left;
        transition: none;
    }}
    .kpi-item:last-child {{ border-bottom: none; }}
    .kpi-item:hover {{
        border-color: rgba(42,120,214,0.45);
        box-shadow: 0 4px 24px rgba(42,120,214,0.14), inset 0 1px 0 rgba(15,23,42,0.04);
        transform: translateY(-1px);
    }}
    .kpi-item .kpi-val {{
        font-size: 1.05rem;
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
        margin-top: 0;
        text-align: right;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }}
    .kpi-item.danger .kpi-val {{ color: #c62828; }}
    .kpi-item.danger {{ border-color: rgba(198,40,40,0.25); }}
    .kpi-item.danger:hover {{ box-shadow: 0 4px 24px rgba(198,40,40,0.12); }}
    .kpi-item.warning .kpi-val {{ color: #b45309; }}
    .kpi-item.warning {{ border-color: rgba(180,83,9,0.25); }}
    .kpi-item.warning:hover {{ box-shadow: 0 4px 24px rgba(180,83,9,0.12); }}
    .kpi-item.success .kpi-val {{ color: #15803d; }}
    .kpi-item.success {{ border-color: rgba(21,128,61,0.25); }}
    .kpi-item.success:hover {{ box-shadow: 0 4px 24px rgba(21,128,61,0.12); }}

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
        background: rgba(42,120,214,0.04) !important;
    }}

    /* Expander hover glow */
    [data-testid="stExpander"]:hover {{
        border-color: rgba(42,120,214,0.25) !important;
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
        0%, 100% {{ box-shadow: 0 0 8px rgba(42,120,214,0.08); }}
        50% {{ box-shadow: 0 0 16px rgba(42,120,214,0.25); }}
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
        outline: 2px solid #2a78d6 !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 4px rgba(42,120,214,0.25) !important;
    }}

    /* Ensure minimum contrast on secondary text (WCAG AA 4.5:1) */
    /* #55637a on #f7f9fb = ~5.8:1 ratio — passes AA */
    /* Bump sidebar muted text for readability */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: #7abcd8 !important;
    }}

    /* High-contrast mode for data values */
    [data-testid="stMetricValue"] {{
        color: #0f172a !important;
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

    /* ── Full-bleed map layout ──────────────────────────────────────────
       The map is the page. Everything else floats over it in glass cards
       positioned inside the main block, so the sidebar can collapse and
       the cards follow it. */
    html, body, [data-testid="stAppViewContainer"] {{
        overflow: hidden !important;
    }}
    [data-testid="stMain"] .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
        position: relative;
        height: 100vh !important;
        overflow: hidden;
    }}
    /* Streamlit's toolbar strip and the gaps between the (zero height)
       floating containers otherwise push the map down the page. */
    [data-testid="stHeader"] {{
        height: 0 !important; min-height: 0 !important;
        background: transparent !important;
    }}
    [data-testid="stMain"] .block-container > div,
    [data-testid="stMain"] [data-testid="stVerticalBlock"] {{
        gap: 0 !important;
    }}
    iframe[title="streamlit_folium.st_folium"] {{
        height: 100vh !important;
        width: 100% !important;
        border: none !important;
        display: block;
    }}
    /* the component wrapper Streamlit puts around the map */
    [data-testid="stCustomComponentV1"] {{
        height: 100vh !important;
    }}
    .fm-glass {{
        background: rgba(255,255,255,0.93);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid {BORDER};
        border-radius: 12px;
        box-shadow: 0 6px 24px rgba(15,23,42,0.10);
    }}
    .st-key-fm_title, .st-key-fm_controls, .st-key-fm_pills,
    .st-key-fm_kpis, .st-key-fm_detail, .st-key-fm_sheet,
    .st-key-fm_basemap, .st-key-fm_notice {{
        z-index: 600;
    }}
    /* Identity sits at the foot of the map, centred. */
    .st-key-fm_title {{
        position: absolute; bottom: 18px; left: 50%; right: auto;
        transform: translateX(-50%); width: 400px; text-align: center;
    }}
    /* Search and region, centred at the top. */
    .st-key-fm_controls {{
        position: absolute; top: 14px; left: 50%; transform: translateX(-50%);
        width: min(460px, 38vw);
    }}
    /* Basemap buttons, top left. */
    .st-key-fm_basemap {{
        position: absolute; top: 14px; left: 16px; width: 296px;
    }}
    .st-key-fm_basemap [data-testid="stButton"] button {{
        font-size: 11.5px; padding: 3px 10px; min-height: 0;
    }}
    /* Panels open from the foot of the left edge. */
    .st-key-fm_pills {{
        position: absolute; bottom: 18px; left: 16px; top: auto; right: auto;
        transform: none; width: auto;
    }}
    .st-key-fm_pills [data-testid="stButton"] button {{
        width: 176px; justify-content: flex-start;
        font-size: 12px; padding: 6px 12px;
    }}
    /* Counters run down the right edge. */
    .st-key-fm_kpis {{
        position: absolute; top: 50%; right: 16px; bottom: auto; left: auto;
        transform: translateY(-50%); width: 176px; padding: 10px 12px;
    }}
    .st-key-fm_detail {{
        position: absolute; bottom: 150px; left: 16px; width: 360px;
        max-height: 52vh; overflow-y: auto; z-index: 650;
    }}
    /* What the map is actually showing, above the identity card. */
    .st-key-fm_notice {{
        position: absolute; bottom: 146px; left: 50%;
        transform: translateX(-50%); width: auto; max-width: 460px;
        padding: 5px 12px;
    }}
    /* Glass treatment for the floating containers themselves */
    .st-key-fm_title, .st-key-fm_controls, .st-key-fm_pills,
    .st-key-fm_kpis, .st-key-fm_detail, .st-key-fm_notice,
    .st-key-fm_basemap {{
        background: rgba(255,255,255,0.93);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid {BORDER};
        border-radius: 12px;
        box-shadow: 0 6px 24px rgba(15,23,42,0.10);
        padding: 10px 14px;
    }}
    .st-key-fm_kpis {{ padding: 8px 12px; }}
    .st-key-fm_controls [data-testid="stTextInput"] input {{
        background: {BG2} !important;
    }}
    /* The prototype warning: a badge that opens on hover */
    .fm-badge {{
        display: inline-block; cursor: default;
        background: #fdf3e3; border: 1px solid #dea03c66; border-radius: 6px;
        color: #b45309; font-size: 9.5px; font-weight: 700;
        letter-spacing: 0.08em; padding: 3px 7px;
    }}
    .fm-badge .fm-badge-text {{
        display: none; font-weight: 400; letter-spacing: 0;
        color: #8a6524; font-size: 11px; line-height: 1.45;
    }}
    .fm-badge:hover .fm-badge-text {{ display: block; margin-top: 5px; }}
    /* Loading veil over the map */
    .fm-loading {{
        position: fixed; inset: 0; z-index: 900;
        background: rgba(247,249,251,0.72);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center; gap: 14px;
    }}
    .fm-spinner {{
        width: 44px; height: 44px; border-radius: 50%;
        border: 3px solid {ACCENT}33; border-top-color: {ACCENT};
        animation: fmspin 0.9s linear infinite;
    }}
    @keyframes fmspin {{ to {{ transform: rotate(360deg); }} }}
    .fm-loading-label {{
        color: {TEXT2}; font-size: 12px; letter-spacing: 0.04em;
    }}

</style>
""", unsafe_allow_html=True)

# Water wave background animation
inject_wave_animation()

# Fermium logo beside the Streamlit menu (top-right)
import base64
import os

# The dark recolour: the original logo is a pale tint for a dark page
# and is nearly invisible on white.
_logo_path = os.path.join(os.path.dirname(__file__), "..", "fermium_dark.png")
if os.path.exists(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
    st.markdown(
        f"""
        <div style="position:fixed;top:0;left:0;width:300px;padding:10px 0 8px 60px;background:#f7f9fb;border-bottom:1px solid #dbe3ec;z-index:999999;">
            <img src="data:image/png;base64,{_logo_b64}" alt="Fermium Systems"
                 style="height:34px;opacity:0.85;">
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

    # The floating controls carry the region picker, so they run first.
    region = _floating_controls(regions)
    cfg = region_config(region)
    panel = _floating_pills()

    veil = st.empty()
    veil.markdown(
        SPINNER_HTML.format(label=f"Loading {region_label(region)}…"),
        unsafe_allow_html=True,
    )
    infra = load_gdf_fast(region, "risk_ranked_assets")
    union_gdf = load_gdf_fast(region, "union_risk_summary")
    hotspot_gdf = load_gdf_fast(region, "hotspot_clusters")
    veil.empty()

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

    search_query = st.session_state.get("asset_search", "")
    if search_query:
        # regex=False: the query is user input, not a pattern
        mask = infra["name"].str.contains(search_query, case=False, na=False, regex=False)
        if "asset_type" in infra.columns:
            mask |= infra["asset_type"].str.contains(
                search_query, case=False, na=False, regex=False
            )
        infra = infra[mask]

    filtered, layers = render_sidebar(region, infra)

    sheet_slot = st.empty()
    notice_slot = st.empty()
    _floating_basemap()
    _floating_title(region, cfg)
    st.session_state.map_capped = False
    _render_map_view(region, filtered, union_gdf, hotspot_gdf, cfg, layers)
    if not panel:
        _floating_kpis(filtered)
        _floating_notice(notice_slot)
    else:
        notice_slot.empty()

    if st.session_state.get("selected_asset"):
        with st.container(key="fm_detail"):
            render_detail_panel()

    if panel:
        _render_sheet(sheet_slot, panel, region, filtered, union_gdf)


def _render_landslide_region(region: str):
    """A region whose model estimates slope failure, not flooding.

    It has no scored assets, so there are no KPI chips or rankings; the
    susceptibility map fills the page and the two remaining panels open in
    the same sheet as elsewhere.
    """
    from dashboard.components.landslide_view import render_landslide_map

    sheet_slot = st.empty()
    _floating_title(region, {})
    render_landslide_map(region)          # fills the page, like the flood map
    panel = st.session_state.get("open_panel")
    if panel == "Rankings & export":
        # A landslide region has no ranked assets; its model summary, fitted
        # coefficients and upazila table take that slot instead.
        with sheet_slot.container(key="fm_sheet"):
            head, shut = st.columns([4, 1])
            with head:
                st.markdown(
                    '<div class="sec-label" style="margin:2px 0 0;">'
                    'Landslide model</div>', unsafe_allow_html=True)
            with shut:
                if st.button("Close", key="close_sheet", use_container_width=True):
                    st.session_state.open_panel = None
                    st.rerun()
            render_landslide_tab(region, with_map=False)
    elif panel in ("Preparedness", "About the data"):
        _render_sheet(sheet_slot, panel, region, None, None)


def _render_no_data():
    """Shown when pipeline outputs are absent — never mock content."""
    st.warning(
        "No pipeline results are available yet, so there is nothing to map.\n\n"
        "Generate them with `python -m pipeline.cli -c config.yaml run`, then "
        "build the display caches with `python preprocess_cache.py`."
    )
    render_about_tab()


SPINNER_HTML = """
<div class="fm-loading">
  <div class="fm-spinner"></div>
  <div class="fm-loading-label">{label}</div>
</div>
"""

SHORT_REGION = {
    "rangpur_rajshahi": "Rangpur",
    "sylhet": "Sylhet",
    "sw_coastal": "Coast",
    "cht": "Hill Tracts",
}


def _short_region(region_id: str) -> str:
    """Names short enough for the floating picker, which has little room."""
    return SHORT_REGION.get(region_id, REGION_CONFIGS.get(region_id, (region_id,))[0])


PANELS = ["Rankings & export", "Preparedness", "About the data"]


def _floating_title(region: str, cfg: dict):
    """Name, region line and the prototype warning, top-left over the map."""

    with st.container(key="fm_title"):
        st.markdown(
            f"""
            <div style="font-family:'DM Mono',monospace;font-size:15px;
                        font-weight:700;color:{TEXT};letter-spacing:-0.01em;">
              FERMIUM HAZARD MAPPER
            </div>
            <div style="color:{TEXT2};font-size:10.5px;margin:3px 0 7px;
                        letter-spacing:0.03em;">
              {region_label(region)}
            </div>
            <div class="fm-badge">RESEARCH PROTOTYPE
              <div class="fm-badge-text">Modelled susceptibility, not a forecast
              or an official warning. For warnings use FFWC and BMD; in an
              emergency dial 999.</div>
            </div>
            <div style="color:#6b7a91;font-size:9px;margin-top:7px;
                        letter-spacing:0.02em;">
              Map data &copy; OpenStreetMap contributors &middot; Tiles &copy; Esri
            </div>
            """,
            unsafe_allow_html=True,
        )


def _floating_basemap():
    """Basemap buttons, top left, replacing Leaflet's own layer box."""
    from dashboard.components.map_view import BASEMAPS

    current = st.session_state.get("basemap", "Light")
    with st.container(key="fm_basemap"):
        columns = st.columns(len(BASEMAPS), gap="small")
        for column, name in zip(columns, BASEMAPS):
            with column:
                if st.button(name, key=f"base_{name}", use_container_width=True,
                             type="primary" if name == current else "secondary"):
                    st.session_state.basemap = name
                    st.rerun()


def _floating_notice(slot):
    """What the map is actually showing, above the identity card."""
    if not st.session_state.get("map_capped"):
        slot.empty()
        return
    with slot.container(key="fm_notice"):
        st.markdown(
            f'<div style="color:{TEXT2};font-size:10.5px;text-align:center;">'
            "Showing the 2,000 highest-scoring assets of the current filter"
            "</div>",
            unsafe_allow_html=True,
        )


def _floating_controls(regions: list[str]) -> str:
    """Region picker and asset search, floating at the top centre."""
    with st.container(key="fm_controls"):
        if len(regions) > 1:
            chosen = st.segmented_control(
                "Region", regions,
                format_func=_short_region,
                default=st.session_state.get("region_select", regions[0]),
                key="region_select", label_visibility="collapsed",
            )
            region = chosen or regions[0]
        else:
            region = regions[0]
        st.text_input(
            "Search assets", placeholder="Search assets by name or type…",
            label_visibility="collapsed", key="asset_search",
        )
    return region


def _floating_pills() -> str | None:
    """Panel switcher on the right edge. Returns the open panel, or None.

    Buttons rather than st.pills: the pill widget owns its key, and Streamlit
    refuses to let a Close button clear a key a widget already claimed, so the
    panel could never be closed. Here the open panel is plain session state
    and the buttons only toggle it.
    """
    open_panel = st.session_state.get("open_panel")
    with st.container(key="fm_pills"):
        for name in PANELS:
            active = name == open_panel
            if st.button(name, key=f"pill_{name}", use_container_width=True,
                         type="primary" if active else "secondary"):
                # Clicking the open panel's own button closes it again.
                st.session_state.open_panel = None if active else name
                st.rerun()
    return open_panel


def _floating_kpis(filtered):
    """The eight headline counts, along the bottom edge."""
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
    with st.container(key="fm_kpis"):
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


def _render_sheet(slot, panel: str, region, filtered, union_gdf):
    """The panel a button opens, over the whole map.

    It is drawn into a placeholder that exists on every run. Streamlit leaves
    a keyed container in the DOM when the script stops drawing it, so a panel
    rendered directly would stay on screen after Close, still covering the
    map; emptying a placeholder clears it properly.
    """
    with slot.container(key="fm_sheet"):
        head, shut = st.columns([5, 1])
        with head:
            st.markdown(
                f'<div class="sec-label" style="margin:2px 0 0;">{panel}</div>',
                unsafe_allow_html=True,
            )
        with shut:
            if st.button("Close", key="close_sheet", use_container_width=True):
                st.session_state.open_panel = None
                st.rerun()
        if panel == "Rankings & export" and filtered is not None:
            _render_rankings_tab(region, filtered, union_gdf)
        elif panel == "Preparedness":
            render_preparedness_tab(region)
        else:
            render_about_tab(region)


def _select_asset(region: str, row):
    """Put one asset's pipeline row into session state for the detail card."""
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


def _render_map_view(region, filtered, union_gdf, hotspot_gdf, cfg, layers):
    """The map itself, filling the viewport behind the floating panels.

    A marker click comes back as a coordinate, which is matched to the nearest
    asset so the detail card can show that asset's own pipeline row.
    """
    state = render_map(
        region,
        filtered,
        grid_gdf=None,
        union_gdf=union_gdf if len(union_gdf) > 0 else None,
        hotspot_gdf=hotspot_gdf if len(hotspot_gdf) > 0 else None,
        cfg=cfg,
        layers=layers,
    )
    clicked = (state or {}).get("last_object_clicked")
    if not clicked or "lat" not in clicked or len(filtered) == 0:
        return
    lat, lon = float(clicked["lat"]), float(clicked["lng"])
    # Degrees are fine for a nearest match at this scale, and the click lands
    # within a few metres of the marker it came from. The card is drawn after
    # the map in the same run, so setting the state here is enough: a rerun
    # would rebuild the whole map for every click.
    distance = (filtered["lat"] - lat) ** 2 + (filtered["lon"] - lon) ** 2
    nearest = filtered.loc[distance.idxmin()]
    if float(distance.min()) < 1e-6:
        _select_asset(region, nearest)


def _render_rankings_tab(region, filtered, union_gdf):
    """Charts, ranked assets, union summaries and downloads."""
    import plotly.express as px

    render_analytics_overlay(filtered)
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
            _select_asset(region, row)
            render_detail_panel()

    st.markdown('<div class="sec-label">Union summaries</div>', unsafe_allow_html=True)
    render_risk_cards(union_gdf)

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

"""
Sidebar: data provenance, model validation, layer toggles and filters.

Every value here is read from pipeline outputs. When a value is missing it is
shown as "—" rather than replaced with a placeholder.
"""

from datetime import datetime

import streamlit as st

from dashboard.data.loader import (
    load_landslide_model,
    load_pipeline_metadata,
    load_validation_metrics,
    raster_paths,
)
from dashboard.data import theme

ACCENT = "#2a78d6"


def _fmt(value, fmt="{:.3f}", dash="—"):
    """Format a number, or return a dash when it is missing."""
    if value is None:
        return dash
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return dash


def _generated_at(meta: dict) -> str:
    raw = (meta or {}).get("generated_at")
    if not raw:
        return "—"
    try:
        return datetime.fromisoformat(raw).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        return raw


def _row(label: str, value: str, note: str = "") -> str:
    return (
        '<div style="display:flex;justify-content:space-between;gap:8px;'
        'padding:5px 0;border-bottom:1px solid #ffffff;">'
        f'<span style="color:#55637a;">{label}</span>'
        f'<span style="color:#0f172a;font-weight:600;font-family:DM Mono,monospace;">'
        f'{value}</span></div>'
        + (f'<div style="color:#6b7a91;font-size:9px;margin:-2px 0 4px;">{note}</div>'
           if note else "")
    )


def render_provenance(meta: dict | None, validation: dict | None = None,
                      landslide: dict | None = None):
    """Where the numbers come from and how well the model did."""
    meta = meta or {}
    stats = meta.get("risk_score_stats") or {}
    val = meta.get("gnn_validation") or {}

    st.markdown(
        '<div style="color:#2a78d6;font-size:10px;font-weight:700;'
        'letter-spacing:0.12em;margin-bottom:8px;">DATA & MODEL</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="background:#f7f9fb;border:1px solid #dbe3ec;border-radius:8px;'
        'padding:12px 14px;margin-bottom:16px;font-size:11px;">'
        + _row("Processed", _generated_at(meta))
        + _row("Pipeline", str(meta.get("pipeline_version", "—")))
        + _row("Assets", f"{stats.get('n_assets', 0):,}" if stats else "—")
        + _row("Flood-prone labels",
               _fmt(meta.get("label_positive_rate"), "{:.1%}"),
               "share of assets on ground labelled flood-prone")
        + _row("Validation AUC", _fmt(val.get("val_auc_roc"), "{:.3f}"),
               "held-out 10 km blocks")
        + _row("Brier score", _fmt(val.get("val_brier"), "{:.3f}"),
               "lower is better")
        + _row("Kriging variance (mean)",
               _fmt(meta.get("kriging_variance_mean"), "{:.4f}"))
        + _row("Variance / sill",
               _fmt(meta.get("kriging_variance_sill_ratio"), "{:.2f}"),
               "1.0 = interpolation adds nothing between assets")
        + '</div>',
        unsafe_allow_html=True,
    )

    # Whether the map capped its markers is only known once the map has been
    # built, which happens after the sidebar. The row is written into this
    # placeholder at that point.
    st.session_state["cap_slot"] = st.empty()

    # Validation against observed floods, when it has been run.
    held = ((validation or {}).get("assets_vs_observed") or {}).get("held_out_blocks")
    if held and "auc_roc" in held:
        st.markdown(
            '<div style="background:#e8f6ee;border:1px solid #3cdea055;'
            'border-radius:8px;padding:10px 12px;margin-bottom:16px;font-size:11px;">'
            '<div style="color:#15803d;font-weight:700;letter-spacing:0.1em;'
            'margin-bottom:5px;">VS OBSERVED FLOODS</div>'
            + _row("AUC", _fmt(held.get("auc_roc"), "{:.3f}"))
            + _row("Precision lift", _fmt(held.get("ap_lift"), "{:.1f}x"))
            + '</div>',
            unsafe_allow_html=True,
        )

    # Landslide regions report their own fitted model instead.
    ls_val = (landslide or {}).get("validation") or {}
    if ls_val.get("val_auc_roc") is not None:
        inv = (landslide or {}).get("inventory", {})
        st.markdown(
            '<div style="background:#2e0d1a;border:1px solid #de3c7855;'
            'border-radius:8px;padding:10px 12px;margin-bottom:16px;font-size:11px;">'
            '<div style="color:#de3c78;font-weight:700;letter-spacing:0.1em;'
            'margin-bottom:5px;">LANDSLIDE MODEL</div>'
            + _row("Mapped landslides", f"{inv.get('n_landslides', 0):,}")
            + _row("Validation AUC", _fmt(ls_val.get("val_auc_roc"), "{:.3f}"),
                   "held-out 10 km blocks")
            + '</div>',
            unsafe_allow_html=True,
        )


def render_layer_toggles(region: str) -> dict:
    """Toggles for overlays, limited to layers whose data exists on disk."""
    st.markdown(
        '<div style="color:#2a78d6;font-size:10px;font-weight:700;'
        'letter-spacing:0.12em;margin-bottom:8px;">MAP LAYERS</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Infrastructure**")
    st.checkbox("Roads & railways", value=True, key="osm_roads")
    st.checkbox("Bridges", value=True, key="osm_bridges")
    st.checkbox("Schools", value=True, key="osm_schools")
    st.checkbox("Hospitals & clinics", value=True, key="osm_hospitals")

    st.markdown("**Hazard & terrain**")
    st.checkbox("Flood risk heatmap", value=True, key="show_heatmap")
    available = {name: path.exists()
                 for name, path in raster_paths(region).items()}
    for key, label, raster in [
        ("show_flood_surface", "Kriged hazard surface", "flood_risk"),
        ("show_landslide", "Landslide susceptibility", "landslide"),
        ("show_hand", "Height above drainage (HAND)", "hand"),
        ("show_slope", "Slope", "slope"),
        ("show_dem", "Elevation (SRTM 30 m)", "dem"),
    ]:
        st.checkbox(label, value=False, key=key, disabled=not available.get(raster),
                    help=None if available.get(raster) else "Raster not generated yet")

    st.markdown("**Population & boundaries**")
    st.checkbox("Population density (WorldPop)", value=False, key="show_popdens")
    st.checkbox("Union boundaries", value=True, key="show_unions")
    st.checkbox("Hotspots (Getis-Ord Gi*)", value=True, key="show_hotspots")

    return {
        key: st.session_state.get(key, default)
        for key, default in [
            ("osm_roads", True), ("osm_bridges", True), ("osm_schools", True),
            ("osm_hospitals", True), ("show_heatmap", True),
            ("show_flood_surface", False), ("show_landslide", False),
            ("show_hand", False),
            ("show_slope", False), ("show_dem", False), ("show_popdens", False),
            ("show_unions", True), ("show_hotspots", True),
        ]
    }


def render_sidebar(region: str, infra):
    """Sidebar contents. Returns (filtered assets, layer toggles)."""
    import pandas as pd

    with st.sidebar:
        render_provenance(
            load_pipeline_metadata(region),
            load_validation_metrics(region),
            load_landslide_model(region),
        )
        layers = render_layer_toggles(region)
        st.divider()

        st.markdown(
            f'<h3 style="color:{ACCENT};margin-bottom:8px;font-size:15px;">Filters</h3>',
            unsafe_allow_html=True,
        )

        asset_types = (
            sorted(infra["asset_type"].dropna().unique().tolist())
            if "asset_type" in infra.columns else []
        )
        selected_types = st.multiselect("Asset types", asset_types, default=asset_types)
        risk_min, risk_max = st.slider(
            "Flood susceptibility score", 0.0, 1.0, (0.0, 1.0), step=0.05
        )

        if "division" in infra.columns:
            divisions = sorted(infra["division"].dropna().unique().tolist())
            selected_divs = st.multiselect("Divisions", divisions, default=divisions)
        else:
            selected_divs = None

        mask = pd.Series(True, index=infra.index)
        if "asset_type" in infra.columns:
            mask &= infra["asset_type"].isin(selected_types)
        if "flood_risk" in infra.columns:
            mask &= infra["flood_risk"].between(risk_min, risk_max)
        if selected_divs is not None and "division" in infra.columns:
            mask &= infra["division"].isin(selected_divs)
        filtered = infra[mask]

        st.markdown("---")
        st.metric("Showing", f"{len(filtered):,} / {len(infra):,}")
        if "division" in filtered.columns:
            for div, cnt in filtered["division"].value_counts().items():
                st.markdown(
                    f'<span style="color:{ACCENT};">{div}:</span> '
                    f'<b style="color:{theme.TEXT};">{cnt:,}</b>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.caption(
            "Fermium Hazard Mapper · open research prototype. "
            "Not an official warning service."
        )

    return filtered, layers


def render_analytics_overlay(infra):
    """Compact charts beside the map."""
    import plotly.express as px

    border = theme.BORDER
    text = theme.TEXT
    text2 = theme.TEXT_MUTED
    layout_opts = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=text,
        margin=dict(l=5, r=5, t=24, b=5),
    )

    if "asset_type" in infra.columns and len(infra) > 0:
        st.markdown(
            f'<p style="color:{text2};font-size:0.7rem;margin:0 0 4px 0;'
            f'text-transform:uppercase;letter-spacing:0.5px;">Assets by type</p>',
            unsafe_allow_html=True,
        )
        tc = infra["asset_type"].value_counts().reset_index()
        tc.columns = ["Type", "Count"]
        fig = px.bar(
            tc, x="Count", y="Type", orientation="h", color="Count",
            color_continuous_scale=[theme.ACCENT_SOFT, theme.ACCENT],
        )
        fig.update_layout(height=max(180, len(tc) * 22 + 40), showlegend=False,
                          coloraxis_showscale=False, **layout_opts)
        fig.update_xaxes(gridcolor=border, showgrid=True)
        fig.update_yaxes(gridcolor=border)
        st.plotly_chart(fig, width="stretch")

    if "flood_risk" in infra.columns and len(infra) > 0:
        st.markdown(
            f'<p style="color:{text2};font-size:0.7rem;margin:12px 0 4px 0;'
            f'text-transform:uppercase;letter-spacing:0.5px;">'
            f'Susceptibility distribution</p>',
            unsafe_allow_html=True,
        )
        fig_h = px.histogram(infra, x="flood_risk", nbins=30,
                             color_discrete_sequence=["#2a78d6"])
        fig_h.update_layout(height=200, bargap=0.05, **layout_opts)
        fig_h.update_xaxes(gridcolor=border, title=None)
        fig_h.update_yaxes(gridcolor=border, title=None)
        st.plotly_chart(fig_h, width="stretch")

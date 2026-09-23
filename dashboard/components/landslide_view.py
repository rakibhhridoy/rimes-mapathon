"""
Landslide view for regions whose result is a susceptibility surface rather
than scored infrastructure.

Everything shown is read from the fitted model's own output: the inventory it
was trained on, its validation score on held-out blocks, its coefficients, and
the per-upazila statistics. Where the model has a known limit — a single-event
inventory, a mapped area smaller than the region — the page says so rather
than letting the map imply uniform coverage.
"""

import streamlit as st

from dashboard.data.loader import (
    get_raster_overlay,
    load_landslide_model,
    load_landslide_summary,
)
from dashboard.data.regions import region_config

# Plain-language reading of each terrain predictor.
FEATURE_MEANING = {
    "elevation": "height above sea level",
    "slope": "steepness of the ground",
    "twi": "how much water the terrain gathers (topographic wetness)",
    "hand": "height above the nearest drainage channel",
    "flow_acc": "how much upslope area drains through the point",
}


def _model_summary(model: dict):
    """What the model is, what it was fitted to, and how well it did."""
    inventory = model.get("inventory", {})
    validation = model.get("validation", {})

    auc = validation.get("val_auc_roc")
    auc_txt = f"{auc:.3f}" if isinstance(auc, (int, float)) else "—"

    st.markdown(
        f'<div style="background:#ffffff;border:1px solid #dbe3ec;'
        f'border-radius:8px;padding:16px 20px;margin-bottom:16px;">'
        f'<div style="color:#0f172a;font-size:13px;line-height:1.7;">'
        f'This map estimates <b>where slopes are prone to failure</b>. A '
        f'logistic regression was fitted to '
        f'<b>{inventory.get("n_landslides", 0):,} mapped landslides</b> '
        f'against {inventory.get("n_background", 0):,} background locations, '
        f'then scored on ground held out from training in whole 10 km blocks '
        f'(<b>AUC {auc_txt}</b>, where 0.5 would be chance).</div></div>',
        unsafe_allow_html=True,
    )

    if inventory.get("single_event"):
        dates = inventory.get("event_dates") or []
        when = dates[0] if dates else "a single episode"
        st.warning(
            f"Every mapped landslide in this inventory comes from one rainfall "
            f"episode ({when}). The map therefore describes which slopes failed "
            f"in that storm, which is a guide to susceptibility but not a "
            f"complete record of where landslides can happen."
        )

    domain = inventory.get("background_domain")
    if domain:
        st.caption(
            f"Background locations were drawn from the {domain}, because the "
            "inventory does not cover the whole region — treating unmapped "
            "ground as landslide-free would teach the model the mapping "
            "footprint instead of the terrain."
        )


def _coefficients(model: dict):
    """Which terrain properties drive the estimate, and in which direction."""
    coefficients = model.get("standardised_coefficients") or {}
    if not coefficients:
        return

    st.markdown('<div class="sec-label">What drives the estimate</div>',
                unsafe_allow_html=True)
    ordered = sorted(coefficients.items(), key=lambda kv: -abs(kv[1]))
    largest = max(abs(v) for _, v in ordered) or 1.0

    for name, value in ordered:
        direction = "raises" if value > 0 else "lowers"
        colour = "#c62828" if value > 0 else "#15803d"
        width = abs(value) / largest * 100
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'padding:6px 0;border-bottom:1px solid #dbe3ec;font-size:11px;">'
            f'<span style="color:#0f172a;width:150px;">{name}'
            f'<span style="color:#6b7a91;"> — '
            f'{FEATURE_MEANING.get(name, "")}</span></span>'
            f'<div style="flex:1;background:#ffffff;border-radius:3px;'
            f'height:7px;overflow:hidden;">'
            f'<div style="background:{colour};width:{width:.0f}%;height:7px;">'
            f'</div></div>'
            f'<span style="color:{colour};font-family:DM Mono,monospace;'
            f'width:120px;text-align:right;">{direction} risk ({value:+.2f})'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "Coefficients are standardised, so they can be compared with each "
        "other; they describe this model's fit, not physical causation."
    )


def _map(region: str, cfg: dict):
    """Susceptibility surface over a basemap."""
    from dashboard.components.map_view import (_add_base_layers, fill_frame,
                                                _get_map_imports)

    overlay = get_raster_overlay(region, "landslide")
    if not overlay:
        st.info(
            "The susceptibility raster has not been pre-rendered yet. Run "
            "`python preprocess_cache.py -c configs/cht.yaml`."
        )
        return

    folium, _, _, st_folium_fn = _get_map_imports()
    dash = cfg.get("dashboard", {})
    m = folium.Map(location=dash.get("map_center", [22.5, 92.1]),
                   zoom_start=dash.get("map_zoom", 9), tiles=None,
                   control_scale=False)
    fill_frame(folium, m)
    _add_base_layers(folium, m)
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{overlay['image_base64']}",
        bounds=overlay["bounds"], name="Landslide susceptibility", opacity=0.65,
    ).add_to(m)
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;top:56px;right:10px;z-index:9999;'
        'background:rgba(255,255,255,0.96);border:1px solid #dbe3ec;'
        'border-radius:6px;padding:8px 12px;font-size:10px;color:#475569;'
        'font-family:monospace;"><b style="color:#2a78d6;">Susceptibility</b>'
        '<br>darker orange = more prone to failure</div>'
    ))
    st_folium_fn(m, width=None, height=760, returned_objects=[])


def _upazila_table(region: str):
    """Per-upazila statistics, straight from the zonal summary."""
    import pandas as pd

    rows = load_landslide_summary(region)
    if not rows:
        return

    st.markdown('<div class="sec-label">By upazila</div>', unsafe_allow_html=True)
    df = pd.DataFrame(rows)
    keep = [c for c in ["admin_label", "susceptibility_mean",
                        "susceptibility_max", "population"] if c in df.columns]
    df = df[keep].rename(columns={
        "admin_label": "Upazila",
        "susceptibility_mean": "Mean susceptibility",
        "susceptibility_max": "Highest",
        "population": "Population",
    })
    st.dataframe(
        df, width="stretch", hide_index=True, height=360,
        column_config={
            "Mean susceptibility": st.column_config.ProgressColumn(
                "Mean susceptibility", min_value=0.0, max_value=1.0, format="%.3f"),
            "Population": st.column_config.NumberColumn("Population", format="%d"),
        },
    )
    st.download_button(
        "Download table (CSV)", df.to_csv(index=False),
        file_name=f"{region}_landslide_susceptibility.csv", mime="text/csv",
    )


def render_landslide_map(region: str):
    """Just the susceptibility map, for the full-page layout."""
    if not load_landslide_model(region):
        st.info(
            "No fitted landslide model for this region yet. Run "
            "`python -m pipeline.cli -c configs/cht.yaml landslide`."
        )
        return
    _map(region, region_config(region))


def render_landslide_tab(region: str, with_map: bool = True):
    """Model summary, map, fitted coefficients and the upazila table."""
    model = load_landslide_model(region)
    if not model:
        st.info(
            "No fitted landslide model for this region yet. Run "
            "`python -m pipeline.cli -c configs/cht.yaml landslide`."
        )
        return

    cfg = region_config(region)
    _model_summary(model)
    if with_map:
        _map(region, cfg)
    _coefficients(model)
    _upazila_table(region)

    citation = (model.get("inventory") or {}).get("citation")
    if citation:
        st.caption(f"Inventory: {citation}")

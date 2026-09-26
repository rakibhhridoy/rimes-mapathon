"""
About tab: what the numbers mean, how they were produced, what the data is,
what the limits are, and how to cite or reuse the work.
"""

import streamlit as st

from dashboard.data.constants import (
    DATA_SOURCES_USED,
    KNOWN_LIMITATIONS,
    METHOD_STEPS,
)
from dashboard.data.loader import (
    load_hazard_model,
    load_landslide_model,
    load_pipeline_metadata,
    load_validation_metrics,
)
from dashboard.data.regions import region_config, region_status

REPO_URL = "https://github.com/rakibhhridoy/rimes-mapathon"
DATA_DOI = "10.5281/zenodo.22978729"


def _section(title: str):
    st.markdown(f'<div class="sec-label">{title}</div>', unsafe_allow_html=True)


def render_about_tab(region: str = "rangpur_rajshahi"):
    st.markdown(
        '<div style="background:#ffffff;border:1px solid #dbe3ec;border-radius:8px;'
        'padding:16px 20px;margin-bottom:18px;">'
        '<div style="color:#0f172a;font-size:13px;line-height:1.7;">'
        'Fermium Hazard Mapper estimates <b>where flooding would hurt most</b> in '
        'northern Bangladesh. It combines three things for every 500 m of ground: '
        'how flood-prone the land is (<b>hazard</b>), how much built infrastructure '
        'stands there (<b>exposure</b>), and how hard it would be for people there '
        'to cope (<b>vulnerability</b>).'
        '</div>'
        '<div style="color:#55637a;font-size:12px;line-height:1.7;margin-top:10px;">'
        'Exposure is counted two ways: the infrastructure standing on the '
        'ground, and the people living on it. A city tops the first; a crowded '
        'floodplain village can top the second. Both are shown.'
        '</div>'
        '<div style="color:#55637a;font-size:12px;line-height:1.7;margin-top:10px;">'
        'Scores run from 0 to 1 and are meaningful <i>relative to each other</i>: '
        'a score of 0.8 marks ground far more exposed than ground scored 0.2. They '
        'describe typical monsoon conditions, not any particular flood, and they '
        'are not a forecast.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    _section("How the numbers are produced")
    for i, (title, detail) in enumerate(METHOD_STEPS, start=1):
        st.markdown(
            f'<div style="display:flex;gap:12px;padding:8px 0;'
            f'border-bottom:1px solid #dbe3ec;">'
            f'<span style="color:#2a78d6;font-family:DM Mono,monospace;font-size:11px;'
            f'flex-shrink:0;width:22px;">{i:02d}</span>'
            f'<div><span style="color:#0f172a;font-size:12px;font-weight:600;">{title}</span>'
            f'<div style="color:#55637a;font-size:11px;margin-top:2px;line-height:1.6;">'
            f'{detail}</div></div></div>',
            unsafe_allow_html=True,
        )

    meta = load_pipeline_metadata(region) or {}
    val = meta.get("gnn_validation") or {}
    if val:
        _section("How well the model does")
        st.markdown(
            f'<div style="color:#55637a;font-size:12px;line-height:1.8;">'
            f'On ground held out from training as whole 10 km blocks, the model '
            f'separates flood-prone from non-flood-prone locations with an '
            f'<b style="color:#0f172a;">AUC of {val.get("val_auc_roc", float("nan")):.3f}</b> '
            f'(0.5 would be chance). Blocks are held out whole because nearby '
            f'assets share terrain: a random split would flatter the model.'
            f'</div>',
            unsafe_allow_html=True,
        )

    hazard = load_hazard_model(region) or {}
    source = (region_config(region).get("risk", {}) or {}).get(
        "hazard_source", "kriged")
    if hazard or source:
        readable = ("each cell's own terrain" if source == "terrain_model"
                    else "interpolation between mapped assets (Kriging)")
        auc = hazard.get("val_auc_roc")
        extra = (f" That terrain model scores an AUC of {auc:.3f} on held-out "
                 f"blocks." if isinstance(auc, (int, float)) else "")
        st.markdown(
            f'<div style="color:#55637a;font-size:12px;line-height:1.8;'
            f'margin-top:8px;">Hazard for each cell comes from '
            f'<b style="color:#0f172a;">{readable}</b>.{extra} Kriging carries '
            f'information only where flood-prone ground is contiguous; where it '
            f'is patchy the interpolated surface flattens towards the regional '
            f'average, which is why the terrain model is used instead.</div>',
            unsafe_allow_html=True,
        )

    observed = load_validation_metrics(region) or {}
    held = (observed.get("assets_vs_observed") or {}).get("held_out_blocks") or {}

    # Say so when the hazard layer itself fails against observed flooding,
    # as it does on the coast, where surge follows embankments and tides
    # rather than terrain.
    surface = ((observed.get("hazard_surface_vs_observed") or {})
               .get("terrain") or {}).get("ranking") or {}
    surface_auc = surface.get("auc_roc")
    if isinstance(surface_auc, (int, float)) and surface_auc < 0.6:
        st.warning(
            f"In this region the hazard layer is barely better than chance "
            f"against observed flooding (AUC {surface_auc:.2f}). Flooding here "
            "is driven by embankments, tides and storm tracks that the terrain "
            "model does not see, so read the hazard and composite maps with "
            "caution. The asset rankings are more reliable."
        )
    if "auc_roc" in held:
        st.markdown(
            f'<div style="color:#55637a;font-size:12px;line-height:1.8;margin-top:8px;">'
            f'Against flood extents actually observed by Sentinel-1 radar during '
            f'{(observed.get("assets_vs_observed") or {}).get("n_events", "several")} '
            f'past floods, the same scores reach an '
            f'<b style="color:#0f172a;">AUC of {held["auc_roc"]:.3f}</b>, and rank '
            f'flooded places {held.get("ap_lift", float("nan")):.1f} times better '
            f'than chance would.</div>',
            unsafe_allow_html=True,
        )

    ls_model = load_landslide_model(region) or {}
    ls_val = ls_model.get("validation") or {}
    if ls_val.get("val_auc_roc") is not None:
        inv = ls_model.get("inventory", {})
        coef = ls_model.get("standardised_coefficients", {})
        drivers = ", ".join(
            f"{name} ({value:+.2f})" for name, value in
            sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:3]
        )
        _section("Landslide model")
        st.markdown(
            f'<div style="color:#55637a;font-size:12px;line-height:1.8;">'
            f'Susceptibility is a logistic regression fitted to '
            f'<b style="color:#0f172a;">{inv.get("n_landslides", 0):,} mapped '
            f'landslides</b> from NASA\'s COOLR inventory against '
            f'{inv.get("n_background", 0):,} background points, validated on '
            f'held-out 10 km blocks (<b style="color:#0f172a;">AUC '
            f'{ls_val["val_auc_roc"]:.3f}</b>). Strongest terrain predictors: '
            f'{drivers}.<br><span style="font-size:10px;color:#6b7a91;">'
            f'{inv.get("citation", "")}</span></div>',
            unsafe_allow_html=True,
        )

    _section("Coverage")
    for entry in region_status():
        color, text = (("#15803d", "available") if entry["available"]
                       else ("#b45309", "in preparation"))
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:7px 0;border-bottom:1px solid #dbe3ec;font-size:12px;">'
            f'<span style="color:#0f172a;">{entry["name"]}'
            f'<span style="color:#6b7a91;"> · {entry["hazard"]}</span></span>'
            f'<span style="color:{color};font-size:11px;">{text}</span></div>',
            unsafe_allow_html=True,
        )

    _section("Data sources")
    for src in DATA_SOURCES_USED:
        st.markdown(
            f'<div style="border-bottom:1px solid #dbe3ec;padding:8px 0;font-size:11px;">'
            f'<a href="{src["url"]}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#0f172a;font-weight:600;text-decoration:none;">{src["name"]}</a>'
            f'<div style="color:#55637a;margin-top:2px;">{src["use"]}</div>'
            f'<div style="color:#6b7a91;font-size:10px;margin-top:2px;">'
            f'{src["attribution"]} · {src["licence"]}</div></div>',
            unsafe_allow_html=True,
        )

    _section("What this cannot tell you")
    for limit in KNOWN_LIMITATIONS:
        st.markdown(
            f'<div style="color:#55637a;font-size:11px;line-height:1.7;'
            f'padding:5px 0 5px 12px;border-left:2px solid #8a3a1a;margin-bottom:6px;">'
            f'{limit}</div>',
            unsafe_allow_html=True,
        )

    _section("Code, data and reuse")
    st.markdown(
        f'<div style="color:#55637a;font-size:11px;line-height:1.8;">'
        f'Source code: <a href="{REPO_URL}" target="_blank" rel="noopener noreferrer" '
        f'style="color:#2a78d6;">{REPO_URL}</a> (MIT licence)<br>'
        f'Input data archive: <a href="https://doi.org/{DATA_DOI}" target="_blank" '
        f'rel="noopener noreferrer" style="color:#2a78d6;">doi.org/{DATA_DOI}</a><br>'
        f'Outputs on this site may be reused with attribution, subject to the '
        f'licences of the source datasets listed above.'
        f'</div>',
        unsafe_allow_html=True,
    )

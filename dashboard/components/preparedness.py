"""
Preparedness tab: shelters mapped in OpenStreetMap and the official
Bangladeshi sources for warnings and emergency help.

This dashboard is a research prototype. It issues no warnings and holds no
live information about which shelters are open, so it says so plainly and
points people to the authorities who do.
"""

import streamlit as st

from dashboard.data.constants import OFFICIAL_RESOURCES
from dashboard.data.loader import get_mapped_shelters


def render_official_resources():
    st.markdown(
        '<div style="background:#0d2e24;border:1px solid #3cdea0;border-radius:8px;'
        'padding:14px 18px;margin-bottom:18px;">'
        '<div style="color:#3cdea0;font-size:12px;font-weight:700;'
        'letter-spacing:0.08em;margin-bottom:6px;">'
        'FOR WARNINGS AND EMERGENCIES, USE THESE SOURCES</div>'
        '<div style="color:#8fd9bd;font-size:11px;">'
        'The maps here describe long-term susceptibility. They are not forecasts '
        'and are not monitored in real time.</div></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, res in enumerate(OFFICIAL_RESOURCES):
        with cols[i % 2]:
            link = (
                f'<a href="{res["url"]}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#00d4ff;font-size:10px;">{res["url"]}</a>'
                if res["url"] else
                '<span style="color:#8ab4d4;font-size:10px;">Toll-free telephone service</span>'
            )
            st.markdown(
                f'<div style="background:#0d1822;border:1px solid #1e3a52;'
                f'border-radius:8px;padding:12px 14px;margin-bottom:10px;">'
                f'<div style="color:#e8f4ff;font-size:12px;font-weight:600;">'
                f'{res["name"]}</div>'
                f'<div style="color:#8ab4d4;font-size:10px;margin:3px 0 5px;">'
                f'{res["detail"]}</div>{link}</div>',
                unsafe_allow_html=True,
            )


def render_shelters(region: str):
    """Shelters mapped in OSM within the study area, with their model scores."""
    st.markdown(
        '<div class="sec-label">Flood shelters mapped in OpenStreetMap</div>',
        unsafe_allow_html=True,
    )

    shelters = get_mapped_shelters(region)
    if len(shelters) == 0:
        st.info(
            "No flood shelters are mapped in OpenStreetMap for this area, or the "
            "pipeline has not been run. This says nothing about whether shelters "
            "exist — contact the Department of Disaster Management for the "
            "official list."
        )
        return

    st.caption(
        f"{len(shelters):,} shelters are mapped in the study area. OpenStreetMap "
        "records location and name only: capacity, condition and whether a shelter "
        "is currently open are not known here."
    )

    cols = ["name", "division", "lat", "lon", "flood_risk"]
    table = shelters[[c for c in cols if c in shelters.columns]].copy()
    table = table.rename(columns={
        "name": "Shelter", "division": "Division", "lat": "Latitude",
        "lon": "Longitude", "flood_risk": "Susceptibility of site",
    })
    if "Susceptibility of site" in table.columns:
        table = table.sort_values("Susceptibility of site", ascending=False)

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=320,
        column_config={
            "Susceptibility of site": st.column_config.ProgressColumn(
                "Susceptibility of site", min_value=0.0, max_value=1.0, format="%.2f",
                help="Modelled flood susceptibility of the ground the shelter "
                     "stands on. A high score means the shelter itself may be "
                     "affected.",
            ),
            "Latitude": st.column_config.NumberColumn(format="%.4f"),
            "Longitude": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.download_button(
        "Download shelter list (CSV)",
        table.to_csv(index=False),
        file_name="hazmapper_mapped_shelters.csv",
        mime="text/csv",
    )


def render_preparedness_tab(region: str):
    render_official_resources()
    render_shelters(region)

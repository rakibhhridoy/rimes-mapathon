"""
Interactive Folium map with:
  - Color-coded circle markers sized by risk
  - Donut-style cluster icons
  - Rich gauge-style popups
  - Risk heatmap, admin boundaries, hotspot overlays
"""

import folium
import geopandas as gpd
import numpy as np
import streamlit as st
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import json


# Asset type → (emoji, default color)
TYPE_COLORS = {
    "hospital": "#ef4444",
    "school": "#3b82f6",
    "bridge": "#f59e0b",
    "road": "#6b7280",
    "flood_shelter": "#10b981",
    "embankment": "#059669",
    "railway": "#8b5cf6",
    "ferry_ghat": "#0ea5e9",
    "cropland": "#22c55e",
    "fishpond": "#67e8f9",
    "irrigation": "#06b6d4",
    "market": "#f43f5e",
}

TYPE_EMOJI = {
    "hospital": "🏥", "school": "🏫", "bridge": "🌉", "road": "🛣️",
    "flood_shelter": "🏠", "embankment": "🔒", "railway": "🚂",
    "ferry_ghat": "⛴️", "cropland": "🌾", "fishpond": "🐟",
    "irrigation": "💧", "market": "🏪",
}


def _risk_color(score) -> str:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "#64748b"
    if score >= 0.7:
        return "#ef4444"
    elif score >= 0.5:
        return "#f59e0b"
    elif score >= 0.3:
        return "#eab308"
    return "#22c55e"


def _risk_label(score) -> str:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "N/A"
    if score >= 0.7:
        return "CRITICAL"
    elif score >= 0.5:
        return "HIGH"
    elif score >= 0.3:
        return "MODERATE"
    return "LOW"


def _gauge_popup(name, atype, risk, rank, division="") -> str:
    """Rich HTML popup with semicircle gauge and stats."""
    color = _risk_color(risk)
    label = _risk_label(risk)
    emoji = TYPE_EMOJI.get(atype, "📍")
    risk_val = f"{risk:.3f}" if isinstance(risk, (int, float)) and risk > 0 else "N/A"
    rank_val = f"#{int(rank)}" if isinstance(rank, (int, float)) and rank > 0 else "—"
    pct = min(float(risk) * 100, 100) if isinstance(risk, (int, float)) else 0

    # SVG semicircle gauge
    gauge_svg = f"""
    <svg width="100" height="55" viewBox="0 0 100 55">
      <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#334155" stroke-width="8" stroke-linecap="round"/>
      <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"
            stroke-dasharray="{pct * 1.26} 126"/>
      <text x="50" y="45" text-anchor="middle" font-size="14" font-weight="bold" fill="{color}">{risk_val}</text>
    </svg>
    """

    return f"""
    <div style="font-family:'Segoe UI',sans-serif; min-width:200px; padding:4px;">
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:6px;">
            <span style="font-size:20px;">{emoji}</span>
            <div>
                <div style="font-size:13px; font-weight:700; color:#1e293b; line-height:1.2;">
                    {name}
                </div>
                <div style="font-size:10px; color:#64748b;">
                    {atype.replace('_',' ').title()} {('| ' + division) if division else ''}
                </div>
            </div>
        </div>

        <div style="text-align:center; margin:4px 0;">
            {gauge_svg}
        </div>

        <div style="display:flex; justify-content:space-between; font-size:11px; margin-top:4px;">
            <span style="
                background:{color}18; color:{color}; padding:2px 8px;
                border-radius:10px; font-weight:600; font-size:10px;
            ">{label}</span>
            <span style="color:#64748b;">Rank: <b>{rank_val}</b></span>
        </div>
    </div>
    """


def render_map(infra: gpd.GeoDataFrame,
                grid_gdf: gpd.GeoDataFrame = None,
                union_gdf: gpd.GeoDataFrame = None,
                hotspot_gdf: gpd.GeoDataFrame = None,
                cfg: dict = None,
                is_dark: bool = True):
    """Render the main interactive map."""
    center = cfg.get("dashboard", {}).get("map_center", [25.5, 89.0])
    zoom = cfg.get("dashboard", {}).get("map_zoom", 8)

    m = folium.Map(location=center, zoom_start=zoom, tiles=None)

    # Base layers
    if is_dark:
        folium.TileLayer(
            tiles="https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png",
            attr="Stadia", name="Dark",
        ).add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="Street").add_to(m)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False,
    ).add_to(m)

    # --- Circle markers (color by type, size by risk) ---
    if infra is not None and len(infra) > 0:
        marker_cluster = MarkerCluster(
            name="Infrastructure",
            options={
                "maxClusterRadius": 40,
                "spiderfyOnMaxZoom": True,
                "showCoverageOnHover": False,
            },
        )

        for _, row in infra.iterrows():
            lat = row.get("lat", None)
            lon = row.get("lon", None)
            if lat is None or lon is None:
                pt = row.geometry.representative_point()
                lat, lon = pt.y, pt.x

            atype = row.get("asset_type", "other")
            risk = row.get("flood_risk", 0)
            name = row.get("name", "unnamed")
            rank = row.get("risk_rank", 0)
            division = row.get("division", "")

            type_color = TYPE_COLORS.get(atype, "#64748b")
            risk_radius = max(4, min(12, float(risk) * 15)) if isinstance(risk, (int, float)) and risk > 0 else 5

            popup_html = _gauge_popup(name, atype, risk, rank, division)

            folium.CircleMarker(
                location=[lat, lon],
                radius=risk_radius,
                color=type_color,
                fill=True,
                fill_color=type_color,
                fill_opacity=0.7,
                weight=1.5,
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"{TYPE_EMOJI.get(atype, '')} {name}",
            ).add_to(marker_cluster)

        marker_cluster.add_to(m)

    # --- Risk heatmap ---
    if grid_gdf is not None and "composite_risk" in grid_gdf.columns:
        heat_data = []
        for _, row in grid_gdf.iterrows():
            c = row.geometry.centroid
            risk_val = row.get("composite_risk", 0)
            if risk_val > 0.1:
                heat_data.append([c.y, c.x, risk_val])
        if heat_data:
            HeatMap(
                heat_data, name="Flood Risk Heatmap",
                min_opacity=0.25, radius=18, blur=12,
                gradient={0.2: "#0ea5e9", 0.4: "#22c55e",
                          0.6: "#eab308", 0.8: "#f59e0b", 1.0: "#ef4444"},
            ).add_to(m)

    # --- Admin boundaries ---
    if union_gdf is not None and len(union_gdf) > 0:
        folium.GeoJson(
            union_gdf.to_json(),
            name="Union Boundaries",
            style_function=lambda f: {
                "fillColor": _risk_color(f["properties"].get("mean_risk", 0)),
                "color": "#94a3b8" if is_dark else "#475569",
                "weight": 1,
                "fillOpacity": 0.25,
                "dashArray": "4",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["admin_name", "mean_risk", "risk_rank"],
                aliases=["Union:", "Risk:", "Rank:"],
            ),
        ).add_to(m)

    # --- Hotspots ---
    if hotspot_gdf is not None and len(hotspot_gdf) > 0:
        if "is_hotspot" in hotspot_gdf.columns:
            hotspots = hotspot_gdf[hotspot_gdf["is_hotspot"] == True]
        else:
            hotspots = hotspot_gdf
        if len(hotspots) > 0:
            folium.GeoJson(
                hotspots.to_json(),
                name="Hotspots (Gi*)",
                style_function=lambda x: {
                    "fillColor": "#ef4444",
                    "color": "#ef4444",
                    "weight": 2,
                    "fillOpacity": 0.35,
                },
            ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, width=None, height=620, returned_objects=[])

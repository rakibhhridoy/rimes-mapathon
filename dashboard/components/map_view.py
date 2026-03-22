"""
Interactive Folium map with toggleable risk layers, infrastructure markers,
admin boundaries, and hotspot overlays.
"""

import folium
import geopandas as gpd
import numpy as np
import streamlit as st
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium


ICON_MAP = {
    "hospital": ("plus-sign", "red"),
    "school": ("education", "blue"),
    "bridge": ("road", "orange"),
    "road": ("road", "gray"),
    "flood_shelter": ("home", "green"),
    "embankment": ("tower", "darkgreen"),
    "railway": ("road", "purple"),
    "ferry_ghat": ("plane", "cadetblue"),
    "cropland": ("leaf", "darkgreen"),
    "fishpond": ("tint", "lightblue"),
    "irrigation": ("tint", "blue"),
    "market": ("shopping-cart", "darkred"),
}


def render_map(infra: gpd.GeoDataFrame,
                grid_gdf: gpd.GeoDataFrame = None,
                union_gdf: gpd.GeoDataFrame = None,
                hotspot_gdf: gpd.GeoDataFrame = None,
                cfg: dict = None):
    """Render the main interactive map."""
    center = cfg.get("dashboard", {}).get("map_center", [25.5, 89.0])
    zoom = cfg.get("dashboard", {}).get("map_zoom", 8)

    # Dark-themed map
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,
    )

    # Base layers
    folium.TileLayer(
        tiles="https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png",
        attr="Stadia", name="Dark", overlay=False,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False,
    ).add_to(m)

    # --- Infrastructure markers (clustered) ---
    if infra is not None and len(infra) > 0:
        marker_cluster = MarkerCluster(name="Infrastructure Assets")

        for _, row in infra.iterrows():
            # Use lon/lat columns if available, otherwise compute
            if "lon" in row.index and "lat" in row.index:
                lat, lon = row["lat"], row["lon"]
            else:
                pt = row.geometry.representative_point()
                lat, lon = pt.y, pt.x

            atype = row.get("asset_type", "other")
            risk = row.get("flood_risk", 0)
            name = row.get("name", "unnamed")
            icon_name, icon_color = ICON_MAP.get(atype, ("info-sign", "gray"))

            # Style popup
            risk_val = f"{risk:.3f}" if isinstance(risk, (int, float)) else "N/A"
            popup_html = (
                f'<div style="font-family:sans-serif; min-width:160px;">'
                f'<b style="font-size:13px;">{name}</b><br>'
                f'<span style="color:#666;">Type:</span> {atype}<br>'
                f'<span style="color:#666;">Risk:</span> '
                f'<b style="color:{_risk_to_hex(risk) if isinstance(risk, (int, float)) else "#999"}">'
                f'{risk_val}</b><br>'
                f'<span style="color:#666;">Rank:</span> {row.get("risk_rank", "N/A")}'
                f'</div>'
            )

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix="glyphicon"),
            ).add_to(marker_cluster)

        marker_cluster.add_to(m)

    # --- Risk heatmap from grid ---
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
                min_opacity=0.3, radius=15, blur=10,
                gradient={0.2: "#0ea5e9", 0.4: "#22c55e", 0.6: "#eab308",
                          0.8: "#f59e0b", 1.0: "#ef4444"},
            ).add_to(m)

    # --- Admin boundaries (unions) ---
    if union_gdf is not None and len(union_gdf) > 0:
        style_function = lambda feature: {
            "fillColor": _risk_to_hex(
                feature["properties"].get("mean_risk", 0)
            ),
            "color": "#94a3b8",
            "weight": 1,
            "fillOpacity": 0.35,
            "dashArray": "3",
        }

        tooltip = folium.GeoJsonTooltip(
            fields=["admin_name", "mean_risk", "risk_rank"],
            aliases=["Union:", "Mean Risk:", "Rank:"],
            style="font-size:12px; background:#1e293b; color:#e2e8f0;",
        )

        folium.GeoJson(
            union_gdf.to_json(),
            name="Union Boundaries",
            style_function=style_function,
            tooltip=tooltip,
        ).add_to(m)

    # --- Hotspot clusters ---
    if hotspot_gdf is not None and len(hotspot_gdf) > 0:
        if "is_hotspot" in hotspot_gdf.columns:
            hotspots = hotspot_gdf[hotspot_gdf["is_hotspot"] == True]
        else:
            hotspots = hotspot_gdf
        if len(hotspots) > 0:
            folium.GeoJson(
                hotspots.to_json(),
                name="Hotspot Clusters (Gi*)",
                style_function=lambda x: {
                    "fillColor": "#ef4444",
                    "color": "#ef4444",
                    "weight": 2,
                    "fillOpacity": 0.4,
                },
            ).add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Render
    st_folium(m, width=None, height=650, returned_objects=[])


def _risk_to_hex(score) -> str:
    """Map risk score to hex color."""
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

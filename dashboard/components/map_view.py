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

    m = folium.Map(location=center, zoom_start=zoom, tiles=None)

    # Base layers
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png",
        attr="Stamen", name="Terrain", overlay=False,
    ).add_to(m)

    # --- Infrastructure markers (clustered) ---
    if infra is not None and len(infra) > 0:
        marker_cluster = MarkerCluster(name="Infrastructure Assets")
        infra_pts = infra.copy()
        if "centroid" not in infra_pts.columns:
            infra_pts["centroid"] = infra_pts.geometry.representative_point()

        for _, row in infra_pts.iterrows():
            pt = row.get("centroid", row.geometry.representative_point())
            atype = row.get("asset_type", "other")
            risk = row.get("flood_risk", 0)
            name = row.get("name", "unnamed")
            icon_name, icon_color = ICON_MAP.get(atype, ("info-sign", "gray"))

            popup_html = (
                f"<b>{name}</b><br>"
                f"Type: {atype}<br>"
                f"Risk: {risk:.3f}<br>"
                f"Rank: {row.get('risk_rank', 'N/A')}"
            )

            folium.Marker(
                location=[pt.y, pt.x],
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
                gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
            ).add_to(m)

    # --- Admin boundaries (unions) ---
    if union_gdf is not None and len(union_gdf) > 0:
        style_function = lambda feature: {
            "fillColor": _risk_to_hex(
                feature["properties"].get("mean_risk", 0)
            ),
            "color": "#333",
            "weight": 1,
            "fillOpacity": 0.4,
        }

        tooltip = folium.GeoJsonTooltip(
            fields=["admin_name", "mean_risk", "risk_rank"],
            aliases=["Union:", "Mean Risk:", "Rank:"],
            style="font-size:12px;",
        )

        folium.GeoJson(
            union_gdf.to_json(),
            name="Union Boundaries",
            style_function=style_function,
            tooltip=tooltip,
        ).add_to(m)

    # --- Hotspot clusters ---
    if hotspot_gdf is not None and len(hotspot_gdf) > 0:
        hotspots = hotspot_gdf[hotspot_gdf.get("is_hotspot", False) == True]
        if len(hotspots) > 0:
            folium.GeoJson(
                hotspots.to_json(),
                name="Hotspot Clusters (Gi*)",
                style_function=lambda x: {
                    "fillColor": "#dc3545",
                    "color": "#dc3545",
                    "weight": 2,
                    "fillOpacity": 0.5,
                },
            ).add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Render
    st_folium(m, width=None, height=600, returned_objects=[])


def _risk_to_hex(score: float) -> str:
    """Map risk score to hex color."""
    if score >= 0.7:
        return "#dc3545"
    elif score >= 0.5:
        return "#fd7e14"
    elif score >= 0.3:
        return "#ffc107"
    return "#28a745"

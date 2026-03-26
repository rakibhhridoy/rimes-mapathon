"""
Interactive Folium map with:
  - Color-coded circle markers sized by risk
  - Donut-style cluster icons
  - Rich gauge-style popups
  - Risk heatmap, admin boundaries, hotspot overlays
  - Kriging interpolation rings (from Fermium-HazMapper)
  - AlphaEarth cluster overlay + population density (from Fermium-HazMapper)
"""

import folium
import geopandas as gpd
import numpy as np
import streamlit as st
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from dashboard.data.loader import (
    get_alphaearth_clusters,
    get_pop_density_points,
    get_kriging_ci_at_point,
    get_raster_overlay,
)


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
    "shelter": "#10b981",
}

TYPE_EMOJI = {
    "hospital": "H", "school": "S", "bridge": "B", "road": "R",
    "flood_shelter": "FS", "embankment": "E", "railway": "Rl",
    "ferry_ghat": "F", "cropland": "C", "fishpond": "FP",
    "irrigation": "I", "market": "M", "shelter": "Sh",
}

RISK_COLOR_STOPS = [
    (0.00, "#1a6b52"),
    (0.25, "#3a9a3a"),
    (0.50, "#dea03c"),
    (0.75, "#de7a3c"),
    (1.00, "#de3c3c"),
]


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


def _risk_to_color(score: float) -> str:
    """Gradient risk color from Fermium-HazMapper."""
    for i in range(len(RISK_COLOR_STOPS) - 1):
        lo_val, lo_col = RISK_COLOR_STOPS[i]
        hi_val, hi_col = RISK_COLOR_STOPS[i + 1]
        if lo_val <= score <= hi_val:
            return hi_col if score > (lo_val + hi_val) / 2 else lo_col
    return "#de3c3c"


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


def _kriging_rings(m, lat, lon, score):
    """Draw concentric risk rings to simulate Kriging interpolation surface."""
    for r, alpha, offset in [
        (5000, 0.08, 0.0),
        (3000, 0.12, 0.05),
        (1500, 0.18, 0.10),
        (600,  0.28, 0.15),
    ]:
        folium.Circle(
            location=[lat, lon],
            radius=r,
            color=_risk_to_color(min(1.0, score + offset)),
            fill=True,
            fill_opacity=alpha,
            weight=0.5,
        ).add_to(m)


def _gauge_popup(name, atype, risk, rank, division="") -> str:
    """Rich HTML popup with semicircle gauge and stats."""
    color = _risk_color(risk)
    label = _risk_label(risk)
    emoji = TYPE_EMOJI.get(atype, "?")
    risk_val = f"{risk:.3f}" if isinstance(risk, (int, float)) and risk > 0 else "N/A"
    rank_val = f"#{int(rank)}" if isinstance(rank, (int, float)) and rank > 0 else "—"
    pct = min(float(risk) * 100, 100) if isinstance(risk, (int, float)) else 0
    kriging_ci = f"{get_kriging_ci_at_point(0, 0, risk):.3f}" if isinstance(risk, (int, float)) else "N/A"
    cvi_class = "IV" if isinstance(risk, (int, float)) and risk > 0.75 else "III" if isinstance(risk, (int, float)) and risk > 0.55 else "II"

    gauge_svg = f"""
    <svg width="100" height="55" viewBox="0 0 100 55">
      <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#334155" stroke-width="8" stroke-linecap="round"/>
      <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"
            stroke-dasharray="{pct * 1.26} 126"/>
      <text x="50" y="45" text-anchor="middle" font-size="14" font-weight="bold" fill="{color}">{risk_val}</text>
    </svg>
    """

    return f"""
    <div style="font-family:'Segoe UI',sans-serif; min-width:210px; padding:4px;">
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:6px;">
            <span style="font-size:14px;font-weight:700;background:{color}22;color:{color};
                         padding:2px 6px;border-radius:4px;">{emoji}</span>
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

        <div style="font-size:10px;color:#64748b;margin-top:4px;">
            <span>Kriging CI: +/-{kriging_ci}</span> |
            <span>CVI: {cvi_class}</span>
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
                is_dark: bool = True,
                layers: dict = None):
    """Render the main interactive map with all overlays."""
    if layers is None:
        layers = {}

    center = cfg.get("dashboard", {}).get("map_center", [25.5, 89.0]) if cfg else [25.5, 89.0]
    zoom = cfg.get("dashboard", {}).get("map_zoom", 8) if cfg else 8

    m = folium.Map(location=center, zoom_start=zoom, tiles=None,
                   control_scale=False)

    # Hide Leaflet attribution watermark
    m.get_root().html.add_child(folium.Element(
        "<style>.leaflet-control-attribution{display:none !important;}</style>"
    ))

    # Base layers
    if is_dark:
        folium.TileLayer(
            tiles="https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png",
            attr=" ", name="Dark",
        ).add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="Street", attr=" ").add_to(m)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, attr=" ").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr=" ", name="Satellite", overlay=False,
    ).add_to(m)

    # --- Kriging interpolation rings ---
    if layers.get("show_hand", True) or layers.get("show_cvi", True):
        if infra is not None and len(infra) > 0 and "flood_risk" in infra.columns:
            high_risk = infra[infra["flood_risk"] >= 0.6].head(20)
            for _, row in high_risk.iterrows():
                lat = row.get("lat", None)
                lon = row.get("lon", None)
                if lat is not None and lon is not None:
                    _kriging_rings(m, lat, lon, row["flood_risk"])

    # --- AlphaEarth cluster overlay ---
    if layers.get("show_alphearth", False):
        ae_data = get_alphaearth_clusters()
        if ae_data and "features" in ae_data:
            for feat in ae_data["features"]:
                props = feat.get("properties", {})
                coords = feat["geometry"]["coordinates"]
                color = props.get("color", "#a03cde")
                is_centroid = props.get("is_centroid", False)
                radius = 1500 if is_centroid else 600
                opacity = 0.15 if is_centroid else 0.08
                tooltip = (f"Cluster {props.get('cluster', '?')} centroid "
                          f"({props.get('n_points', '?')} pts)"
                          if is_centroid else
                          f"AlphaEarth cluster {props.get('cluster', '?')}")
                folium.Circle(
                    location=[coords[1], coords[0]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_opacity=opacity,
                    weight=0.8 if is_centroid else 0.3,
                    tooltip=tooltip,
                ).add_to(m)
        else:
            # No real data — show placeholder message
            folium.Marker(
                location=center,
                icon=folium.DivIcon(html='<div style="color:#a03cde;font-size:10px;">AlphaEarth: run pipeline</div>'),
            ).add_to(m)

    # --- Population density heatmap ---
    if layers.get("show_popdens", False):
        pop_points = get_pop_density_points(center, radius_deg=0.3, n_points=200)
        if pop_points:
            heat_data = [[lat, lon, w] for lat, lon, w in pop_points]
            HeatMap(
                heat_data, name="Population Density",
                min_opacity=0.15, radius=12, blur=10,
                gradient={"0.2": "#fce7f3", "0.5": "#f472b6", "0.8": "#db2777", "1.0": "#9d174d"},
            ).add_to(m)

    # --- Raster overlays ---
    for raster_key, layer_key in [("dem", "show_dem"), ("slope", "show_slope"),
                                   ("hand", "show_hand"), ("flood_risk", "show_cvi")]:
        if layers.get(layer_key, False) and raster_key not in ("hand",):  # hand uses kriging rings
            overlay = get_raster_overlay(raster_key)
            if overlay:
                import base64
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{overlay['image_base64']}",
                    bounds=overlay["bounds"],
                    name=raster_key.replace("_", " ").title(),
                    opacity=0.6,
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

            # Layer filter: skip if asset type is toggled off
            if atype == "hospital" and not layers.get("osm_hospitals", True):
                continue
            if atype == "bridge" and not layers.get("osm_bridges", True):
                continue
            if atype == "school" and not layers.get("osm_schools", True):
                continue
            if atype in ("road", "railway") and not layers.get("osm_roads", True):
                continue

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
                gradient={"0.2": "#0ea5e9", "0.4": "#22c55e",
                          "0.6": "#eab308", "0.8": "#f59e0b", "1.0": "#ef4444"},
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

    # --- LULC legend ---
    if layers.get("show_lulc", False):
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                    background:#0d1f2d;border:1px solid #1a3a50;border-radius:6px;
                    padding:8px 12px;font-size:10px;font-family:monospace;color:#a0c0d8;">
          <b style="color:#3cb8de;">LULC Classes</b><br>
          <span style="color:#3cdea0;">|</span> Cropland &nbsp;
          <span style="color:#3cb8de;">|</span> Water &nbsp;
          <span style="color:#8a6a1a;">|</span> Built-up &nbsp;
          <span style="color:#4a7a4a;">|</span> Forest
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    # --- DEM legend ---
    if layers.get("show_dem", False):
        dem_html = """
        <div style="position:fixed;bottom:30px;right:30px;z-index:9999;
                    background:#0d1f2d;border:1px solid #1a3a50;border-radius:6px;
                    padding:8px 12px;font-size:10px;font-family:monospace;color:#a0c0d8;">
          <b style="color:#3cb8de;">Elevation (SRTM 30m)</b><br>
          Low ------- High<br>
          <span style="color:#5a8ab0;">0m to 100m+ (simulated)</span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(dem_html))

    # --- Risk color legend ---
    risk_legend = """
    <div style="position:fixed;top:80px;right:10px;z-index:9999;
                background:#0d1f2d;border:1px solid #1a3a50;border-radius:6px;
                padding:8px 12px;font-size:10px;font-family:monospace;color:#a0c0d8;">
      <b style="color:#3cb8de;">Risk Score</b><br>
      <span style="color:#3cdea0;">o</span> Very Low (&lt;0.20)<br>
      <span style="color:#3a9a3a;">o</span> Low (0.20-0.40)<br>
      <span style="color:#dea03c;">o</span> Moderate (0.40-0.60)<br>
      <span style="color:#de7a3c;">o</span> High (0.60-0.80)<br>
      <span style="color:#de3c3c;">o</span> Very High (&gt;0.80)
    </div>
    """
    m.get_root().html.add_child(folium.Element(risk_legend))

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, width=None, height=620, returned_objects=[])


def render_region_map(region_key: str, region_data: dict, layers: dict, map_key: str = "region_map"):
    """Render an interactive Folium map for a specific region using real data with fallback."""
    from dashboard.data.loader import get_regional_assets

    center = region_data.get("center", [23.68, 90.35])
    zoom   = region_data.get("zoom",   9)
    assets = get_regional_assets(region_key, center, radius_deg=0.5)

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        attr=" ",
        prefer_canvas=True,
        control_scale=False,
    )

    # Hide Leaflet attribution watermark
    m.get_root().html.add_child(folium.Element(
        "<style>.leaflet-control-attribution{display:none !important;}</style>"
    ))

    # Kriging risk surface rings (from real kriging variance if available)
    if layers.get("show_hand", True) or layers.get("show_cvi", True):
        for asset in assets:
            _kriging_rings(m, asset[0], asset[1], asset[4])

    # AlphaEarth cluster overlay
    if layers.get("show_alphearth", False):
        ae_data = get_alphaearth_clusters()
        if ae_data and "features" in ae_data:
            # Filter to region bbox
            for feat in ae_data["features"]:
                coords = feat["geometry"]["coordinates"]
                props = feat.get("properties", {})
                if (abs(coords[1] - center[0]) < 0.5 and
                    abs(coords[0] - center[1]) < 0.5):
                    is_centroid = props.get("is_centroid", False)
                    folium.Circle(
                        location=[coords[1], coords[0]],
                        radius=1200 if is_centroid else 500,
                        color=props.get("color", "#a03cde"),
                        fill=True,
                        fill_opacity=0.12 if is_centroid else 0.06,
                        weight=0.8,
                        tooltip=f"AlphaEarth cluster {props.get('cluster', '?')}",
                    ).add_to(m)

    # Population density heatmap
    if layers.get("show_popdens", False):
        pop_points = get_pop_density_points(center, radius_deg=0.15, n_points=100)
        if pop_points:
            heat_data = [[lat, lon, w] for lat, lon, w in pop_points]
            HeatMap(
                heat_data, name="Population Density",
                min_opacity=0.15, radius=10, blur=8,
                gradient={"0.2": "#fce7f3", "0.5": "#f472b6", "0.8": "#db2777", "1.0": "#9d174d"},
            ).add_to(m)

    # Asset markers
    for lat, lon, name, atype, score in assets:
        show = True
        if atype == "hospital" and not layers.get("osm_hospitals", True):
            show = False
        if atype == "bridge"   and not layers.get("osm_bridges",   True):
            show = False
        if atype == "school"   and not layers.get("osm_schools",   True):
            show = False
        if atype == "road"     and not layers.get("osm_roads",     True):
            show = False

        if not show:
            continue

        col = _risk_to_color(score)
        risk_lbl = _risk_label(score)
        kriging_ci = get_kriging_ci_at_point(lat, lon, score)
        cvi_class = "IV" if score > 0.75 else "III" if score > 0.55 else "II"

        popup_html = f"""
        <div style="background:#0d1f2d;color:#e8f4ff;padding:8px 12px;
                    border-radius:6px;font-family:monospace;font-size:11px;min-width:180px;">
          <b style="color:#3cb8de;">{name}</b><br>
          <span style="color:#5a8ab0;">Type:</span> {atype.title()}<br>
          <span style="color:#5a8ab0;">GNN Risk:</span>
          <span style="color:{col};font-weight:700;">{score:.2f} — {risk_lbl}</span><br>
          <span style="color:#5a8ab0;">Kriging CI:</span> +/-{kriging_ci}<br>
          <span style="color:#5a8ab0;">CVI class:</span> {cvi_class}
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=8 + score * 10,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.85,
            weight=1.5,
            tooltip=f"{name} | Risk: {score:.2f}",
            popup=folium.Popup(popup_html, max_width=240),
        ).add_to(m)

    # Risk legend
    risk_legend = """
    <div style="position:fixed;top:80px;right:10px;z-index:9999;
                background:#0d1f2d;border:1px solid #1a3a50;border-radius:6px;
                padding:8px 12px;font-size:10px;font-family:monospace;color:#a0c0d8;">
      <b style="color:#3cb8de;">Risk Score</b><br>
      <span style="color:#3cdea0;">o</span> Very Low (&lt;0.20)<br>
      <span style="color:#3a9a3a;">o</span> Low (0.20-0.40)<br>
      <span style="color:#dea03c;">o</span> Moderate (0.40-0.60)<br>
      <span style="color:#de7a3c;">o</span> High (0.60-0.80)<br>
      <span style="color:#de3c3c;">o</span> Very High (&gt;0.80)
    </div>
    """
    m.get_root().html.add_child(folium.Element(risk_legend))

    return st_folium(m, width="100%", height=480, key=map_key, returned_objects=[])

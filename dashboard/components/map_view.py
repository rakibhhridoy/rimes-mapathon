"""
Interactive Folium map: asset markers with model scores, kriged hazard
surface, population density, union boundaries and Gi* hotspots.

Popups show values computed by the pipeline. Nothing on the map is derived
by rescaling one score into others.
"""

import streamlit as st

from dashboard.data.loader import (
    get_kriging_ci_batch,
    get_pop_density_points,
    get_raster_overlay,
    load_heatmap_points,
)
from dashboard.data import theme

# Tile providers require visible attribution — see their terms of use.
CARTO_ATTR = ('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
              'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>')
OSM_ATTR = ('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors')
ESRI_ATTR = ('Imagery &copy; <a href="https://www.esri.com">Esri</a>, Maxar, '
             'Earthstar Geographics, and the GIS User Community')


def _get_map_imports():
    """Lazy import folium and related heavy libs."""
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    from streamlit_folium import st_folium

    # Folium's LayerControl template uses `let`, which throws a SyntaxError
    # when st_folium re-evaluates the script on a Streamlit rerun.
    if not getattr(folium.LayerControl, '_template_patched', False):
        from folium.template import Template
        folium.LayerControl._template = Template("""
        {% macro script(this,kwargs) %}
            var {{ this.get_name() }}_layers = {
                base_layers : {
                    {%- for key, val in this.base_layers.items() %}
                    {{ key|tojson }} : {{val}},
                    {%- endfor %}
                },
                overlays :  {
                    {%- for key, val in this.overlays.items() %}
                    {{ key|tojson }} : {{val}},
                    {%- endfor %}
                },
            };
            var {{ this.get_name() }} = L.control.layers(
                {{ this.get_name() }}_layers.base_layers,
                {{ this.get_name() }}_layers.overlays,
                {{ this.options|tojavascript }}
            ).addTo({{this._parent.get_name()}});

            {%- if this.draggable %}
            new L.Draggable({{ this.get_name() }}.getContainer()).enable();
            {%- endif %}

        {% endmacro %}
        """)
        folium.LayerControl._template_patched = True

    return folium, MarkerCluster, HeatMap, st_folium


# JS guard injected into the folium HTML — runs in the iframe before any map
# code, so it works regardless of Python-side patching.
_BROWSER_JS_GUARD = """
<script>
(function(){
    // Guard HeatMap: prevent getImageData on 0-width canvas
    if(typeof L!=="undefined"&&L.HeatLayer){
        var _origDraw=L.HeatLayer.prototype._draw;
        L.HeatLayer.prototype._draw=function(){
            if(this._canvas&&this._canvas.width>0&&this._canvas.height>0){
                _origDraw.call(this);
            }
        };
    }
    // Guard Map: prevent "already initialized" error
    if(typeof L!=="undefined"&&L.Map){
        var _origInit=L.Map.prototype.initialize;
        L.Map.prototype.initialize=function(id,options){
            var container=typeof id==='string'?document.getElementById(id):id;
            if(container&&container._leaflet_id){
                container._leaflet_id=null;
                container.innerHTML='';
            }
            return _origInit.call(this,id,options);
        };
    }
})();
</script>
"""

TYPE_COLORS = {
    "hospital": "#c62828",
    "school": "#3b82f6",
    "bridge": "#b45309",
    "road": "#6b7280",
    "flood_shelter": "#10b981",
    "embankment": "#059669",
    "railway": "#6d28d9",
    "ferry_ghat": "#1c5cab",
    "cropland": "#15803d",
    "fishpond": "#67e8f9",
    "irrigation": "#06b6d4",
    "market": "#f43f5e",
}

TYPE_ABBR = {
    "hospital": "H", "school": "S", "bridge": "B", "road": "R",
    "flood_shelter": "FS", "embankment": "E", "railway": "Rl",
    "ferry_ghat": "F", "cropland": "C", "fishpond": "FP",
    "irrigation": "I", "market": "M",
}


def _risk_color(score) -> str:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "#64748b"
    if score >= 0.7:
        return "#c62828"
    if score >= 0.5:
        return "#b45309"
    if score >= 0.3:
        return "#a16207"
    return "#15803d"


def _risk_label(score) -> str:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "N/A"
    if score >= 0.7:
        return "VERY HIGH"
    if score >= 0.5:
        return "HIGH"
    if score >= 0.3:
        return "MODERATE"
    return "LOW"


def _factor_bar(label, value, color) -> str:
    """Horizontal bar for one pipeline-computed factor; blank when missing."""
    if value is None:
        return (
            f'<div style="display:flex;gap:6px;margin:3px 0;font-size:9px;">'
            f'<span style="color:#55637a;width:62px;text-align:right;">{label}</span>'
            f'<span style="color:#94a3b8;">not available</span></div>'
        )
    pct = max(0.0, min(float(value), 1.0)) * 100
    return (
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
        f'<span style="color:#55637a;font-size:9px;width:62px;text-align:right;'
        f'font-family:Inter,sans-serif;">{label}</span>'
        f'<div style="flex:1;background:#1e293b;border-radius:3px;height:6px;overflow:hidden;">'
        f'<div style="background:linear-gradient(90deg,{color}88,{color});'
        f'width:{pct:.0f}%;height:6px;border-radius:3px;"></div></div>'
        f'<span style="color:#0f172a;font-size:9px;font-family:DM Mono,monospace;'
        f'width:32px;">{float(value):.2f}</span></div>'
    )


def _popup_html(row, ci=None) -> str:
    """Popup for one asset, using only values present on the row."""
    name = row.get("name") or "unnamed"
    atype = str(row.get("asset_type", "other"))
    risk = row.get("flood_risk")
    rank = row.get("risk_rank")
    division = row.get("division", "")

    risk_f = float(risk) if isinstance(risk, (int, float)) else None
    color = _risk_color(risk_f if risk_f is not None else -1)
    pct = (risk_f or 0) * 100
    risk_txt = f"{risk_f:.3f}" if risk_f is not None else "N/A"
    rank_txt = f"#{int(rank)}" if isinstance(rank, (int, float)) and rank > 0 else "—"
    ci_txt = f"&plusmn;{ci:.3f}" if ci is not None else "not available"

    def cell(key):
        v = row.get(f"cell_{key}")
        return float(v) if isinstance(v, (int, float)) and v == v else None

    prob = row.get("flood_probability")
    prob_html = ""
    if isinstance(prob, (int, float)) and prob == prob:
        prob_html = (
            f'<div style="font-size:10px;color:#1e293b;margin:4px 0;">'
            f'Approximate flood probability: <b>{100 * float(prob):.0f}%</b>'
            f'<span style="color:#64748b;"> (calibrated on other areas of the region; '
            f'local rates can differ severalfold)</span></div>'
        )

    factors = (
        _factor_bar("Hazard", cell("hazard"), "#c62828")
        + _factor_bar("Exposure", cell("exposure"), "#b45309")
        + _factor_bar("Vulnerab.", cell("vulnerability"), "#6d28d9")
        + _factor_bar("Cell risk", cell("composite_risk"), "#2a78d6")
    )

    gauge = f"""
    <svg width="100" height="55" viewBox="0 0 100 55" role="img"
         aria-label="Susceptibility {risk_txt}">
      <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#1e293b"
            stroke-width="8" stroke-linecap="round"/>
      <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="{color}"
            stroke-width="8" stroke-linecap="round"
            stroke-dasharray="{pct * 1.26} 126"/>
      <text x="50" y="45" text-anchor="middle" font-size="14" font-weight="bold"
            fill="{color}" font-family="DM Mono,monospace">{risk_txt}</text>
    </svg>
    """

    return f"""
    <div style="font-family:Inter,'Segoe UI',sans-serif;min-width:235px;padding:4px;">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
            <span style="font-size:13px;font-weight:700;background:{color}22;color:{color};
                         padding:2px 7px;border-radius:5px;font-family:DM Mono,monospace;">
                {TYPE_ABBR.get(atype, '?')}</span>
            <div>
                <div style="font-size:13px;font-weight:600;color:#1e293b;line-height:1.2;">
                    {name}</div>
                <div style="font-size:10px;color:#64748b;">
                    {atype.replace('_', ' ').title()}{(' | ' + division) if division else ''}</div>
            </div>
        </div>
        <div style="text-align:center;margin:2px 0 4px 0;">{gauge}</div>
        <div style="font-size:9px;color:#64748b;text-align:center;margin-bottom:4px;">
            modelled flood susceptibility (ranking score)</div>
        {prob_html}
        <div style="background:#f7f9fb;border-radius:6px;padding:6px 8px;margin:4px 0;">
            {factors}
        </div>
        <div style="display:flex;gap:10px;font-size:9px;color:#64748b;margin-top:5px;
                    font-family:DM Mono,monospace;">
            <span>95% CI: {ci_txt}</span><span>Rank: {rank_txt}</span>
        </div>
        <div style="margin-top:6px;">
            <span style="background:{color}15;color:{color};padding:2px 10px;
                         border-radius:10px;font-weight:600;font-size:10px;">
                {_risk_label(risk_f if risk_f is not None else -1)}</span>
        </div>
    </div>
    """


def fill_frame(folium, m):
    """Make a folium map fill whatever frame the page gives it.

    Two things are needed. The map div carries a fixed pixel height baked into
    the HTML, which CSS overrides, and Leaflet measures its container once at
    construction, so it has to be told the size changed or it draws no tiles.
    """
    m.get_root().header.add_child(folium.Element(
        "<style>html,body,#root{height:100%!important;margin:0;padding:0}"
        ".float-container,.float-child,#map_div,.folium-map,"
        ".leaflet-container{height:100%!important;width:100%!important}"
        # Credits and scale are drawn by the page instead, so the map keeps
        # its whole surface for data.
        ".leaflet-control-attribution,.leaflet-control-scale{"
        "display:none!important}</style>"
    ))
    m.get_root().html.add_child(folium.Element(
        "<script>(function(){function fit(){for(var k in window){"
        "if(k.indexOf('map_')===0&&window[k]&&window[k].invalidateSize){"
        "try{window[k].invalidateSize();}catch(e){}}}}"
        "window.addEventListener('load',function(){setTimeout(fit,150);"
        "setTimeout(fit,600);});"
        "window.addEventListener('resize',fit);})();</script>"
    ))


def _add_base_layers(folium, m):
    """The three basemaps, switched from Leaflet's own layer box.

    The alternatives are added switched off: Leaflet draws base layers in the
    order they arrive, so without this the last one added covers the rest and
    the map always opens on satellite imagery.
    """
    folium.TileLayer(
        tiles=theme.TILE_URL, attr=theme.TILE_ATTR, name="Light",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets", attr=OSM_ATTR,
                     show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr=ESRI_ATTR, name="Satellite", show=False,
    ).add_to(m)


def _risk_legend() -> str:
    return """
    <div style="position:fixed;top:56px;right:10px;z-index:9999;
                background:rgba(255,255,255,0.96);border:1px solid #dbe3ec;border-radius:6px;
                padding:8px 12px;font-size:10px;font-family:monospace;color:#475569;">
      <b style="color:#2a78d6;">Susceptibility</b><br>
      <span style="color:#15803d;">&#9679;</span> Low (&lt;0.30)<br>
      <span style="color:#a16207;">&#9679;</span> Moderate (0.30-0.50)<br>
      <span style="color:#b45309;">&#9679;</span> High (0.50-0.70)<br>
      <span style="color:#c62828;">&#9679;</span> Very high (&ge;0.70)
    </div>
    """


def render_map(region, infra, grid_gdf=None, union_gdf=None, hotspot_gdf=None,
               cfg=None, layers=None, height=620):
    """Render the main interactive map."""
    layers = layers or {}
    _, _, _, st_folium_fn = _get_map_imports()
    m = _build_main_map(region, infra, grid_gdf, union_gdf, hotspot_gdf,
                        cfg, layers)
    # Only the click is returned: anything more (bounds, zoom, hover) makes
    # Streamlit rerun on every mouse move.
    return st_folium_fn(m, width=None, height=height,
                        returned_objects=["last_object_clicked"])


def _build_main_map(region, infra, grid_gdf, union_gdf, hotspot_gdf, cfg,
                    layers):
    """Build the folium Map object with all layers."""
    folium, MarkerCluster, HeatMap, _ = _get_map_imports()

    dash_cfg = (cfg or {}).get("dashboard", {})
    center = dash_cfg.get("map_center", [25.5, 89.0])
    zoom = dash_cfg.get("map_zoom", 8)
    bbox = (cfg or {}).get("aoi", {}).get("bbox", [88.0, 24.0, 89.9, 26.7])

    m = folium.Map(location=center, zoom_start=zoom, tiles=None,
                   control_scale=False, zoom_control=False)
    m.get_root().html.add_child(folium.Element(_BROWSER_JS_GUARD))

    from folium import MacroElement
    from jinja2 import Template
    zoom_br = MacroElement()
    zoom_br._template = Template(
        "{% macro script(this, kwargs) %}"
        "L.control.zoom({position: 'bottomright'}).addTo({{this._parent.get_name()}});"
        "{% endmacro %}"
    )
    m.add_child(zoom_br)

    fill_frame(folium, m)
    _add_base_layers(folium, m)

    # --- Raster overlays (pre-rendered in WGS84 by preprocess_cache.py) ---
    for layer_key, raster_key, label in [
        ("show_flood_surface", "flood_risk", "Kriged hazard surface"),
        ("show_landslide", "landslide", "Landslide susceptibility"),
        ("show_hand", "hand", "Height above drainage"),
        ("show_slope", "slope", "Slope"),
        ("show_dem", "dem", "Elevation"),
    ]:
        if layers.get(layer_key, False):
            overlay = get_raster_overlay(region, raster_key)
            if overlay:
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{overlay['image_base64']}",
                    bounds=overlay["bounds"], name=label, opacity=0.6,
                ).add_to(m)

    # --- Population density (sampled from WorldPop) ---
    if layers.get("show_popdens", False):
        pop_points = get_pop_density_points(region, tuple(bbox))
        if pop_points:
            HeatMap(
                [[lat, lon, w] for lat, lon, w in pop_points],
                name="Population density (WorldPop)",
                min_opacity=0.15, radius=12, blur=10,
                gradient={"0.2": "#cde2fb", "0.5": "#6da7ec",
                          "0.8": "#db2777", "1.0": "#9d174d"},
            ).add_to(m)

    # --- Asset markers ---
    if infra is not None and len(infra) > 0:
        display_infra = infra
        if "asset_type" in infra.columns:
            exclude = set()
            if not layers.get("osm_hospitals", True):
                exclude.update(("hospital", "clinic"))
            if not layers.get("osm_bridges", True):
                exclude.add("bridge")
            if not layers.get("osm_schools", True):
                exclude.update(("school", "college"))
            if not layers.get("osm_roads", True):
                exclude.update(("road", "railway"))
            if exclude:
                display_infra = display_infra[~display_infra["asset_type"].isin(exclude)]

        # Cap markers for browser performance; the table and exports keep all rows.
        capped = False
        if len(display_infra) > 2000 and "flood_risk" in display_infra.columns:
            display_infra = display_infra.nlargest(2000, "flood_risk")
            capped = True

        rows = []
        for _, row in display_infra.iterrows():
            lat, lon = row.get("lat"), row.get("lon")
            if lat is None or lon is None:
                pt = row.geometry.representative_point()
                lat, lon = pt.y, pt.x
            rows.append((lat, lon, row))

        ci_values = get_kriging_ci_batch(
            region, tuple((lat, lon) for lat, lon, _ in rows))

        marker_cluster = MarkerCluster(
            name="Infrastructure",
            options={"maxClusterRadius": 40, "spiderfyOnMaxZoom": True,
                     "showCoverageOnHover": False},
        )
        for i, (lat, lon, row) in enumerate(rows):
            atype = row.get("asset_type", "other")
            risk = row.get("flood_risk", 0)
            radius = (max(4, min(12, float(risk) * 15))
                      if isinstance(risk, (int, float)) and risk > 0 else 5)
            color = TYPE_COLORS.get(atype, "#64748b")
            folium.CircleMarker(
                location=[lat, lon], radius=radius, color=color, fill=True,
                fill_color=color, fill_opacity=0.7, weight=1.5,
                popup=folium.Popup(_popup_html(row, ci_values[i]), max_width=250),
                tooltip=f"{row.get('name', 'unnamed')}",
            ).add_to(marker_cluster)
        marker_cluster.add_to(m)

        if capped:
            st.session_state.map_capped = True

    # --- Composite risk heatmap ---
    if layers.get("show_heatmap", True):
        heat_data = load_heatmap_points(region)
        if not heat_data and grid_gdf is not None and "composite_risk" in getattr(grid_gdf, "columns", []):
            b = grid_gdf.geometry.bounds
            risks = grid_gdf["composite_risk"].values
            mask = risks > 0.1
            heat_data = list(zip(((b.miny + b.maxy) / 2)[mask],
                                 ((b.minx + b.maxx) / 2)[mask], risks[mask]))
        if heat_data:
            HeatMap(
                heat_data, name="Composite risk", min_opacity=0.25, radius=18, blur=12,
                gradient={"0.2": "#1c5cab", "0.4": "#15803d", "0.6": "#a16207",
                          "0.8": "#b45309", "1.0": "#c62828"},
            ).add_to(m)

    # --- Admin boundaries ---
    if layers.get("show_unions", True) and union_gdf is not None and len(union_gdf) > 0:
        name_field = ("admin_label" if "admin_label" in union_gdf.columns
                      else "admin_name")
        folium.GeoJson(
            union_gdf.to_json(),
            name="Union boundaries",
            style_function=lambda f: {
                # Grey where there is nothing to score, so missing data does
                # not read as low risk.
                "fillColor": ("#475569"
                              if f["properties"].get("mean_risk") is None
                              else _risk_color(f["properties"].get("mean_risk"))),
                "color": theme.TEXT_MUTED,
                "weight": 1, "fillOpacity": 0.25, "dashArray": "4",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[name_field, "mean_risk", "risk_rank"],
                aliases=["Union:", "Mean risk:", "Rank:"],
            ),
        ).add_to(m)

    # --- Hotspots ---
    if layers.get("show_hotspots", True) and hotspot_gdf is not None and len(hotspot_gdf) > 0:
        hotspots = (hotspot_gdf[hotspot_gdf["is_hotspot"] == True]
                    if "is_hotspot" in hotspot_gdf.columns else hotspot_gdf)
        if len(hotspots) > 0:
            folium.GeoJson(
                hotspots.to_json(),
                name="Hotspots (Gi*, 95%)",
                style_function=lambda x: {
                    "fillColor": "#c62828", "color": "#c62828",
                    "weight": 2, "fillOpacity": 0.35,
                },
            ).add_to(m)

    folium.LayerControl(collapsed=True, position='topright').add_to(m)
    m.get_root().html.add_child(folium.Element(_risk_legend()))
    return m

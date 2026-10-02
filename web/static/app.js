/* Fermium Hazard Mapper — front end.
 *
 * The map draws vector tiles the build script wrote, so panning and zooming
 * never ask the server for geometry again. Numbers come from small JSON
 * responses that the browser caches. Nothing here computes a risk figure:
 * every value shown is one the pipeline produced, and the counters beside the
 * map are counted by the API for whatever the filters leave visible.
 */

/* The app may be served at / or under a path such as /hazmapper, so every
 * URL is built from where the page itself was loaded. */
const BASE = document.baseURI.replace(/[^/]*$/, "");
const url = (path) => BASE + path.replace(/^\//, "");

const ALL = "all";

const CLASSES = [
  { value: "low", label: "Low", range: [0, 0.30], colour: "#15803d" },
  { value: "moderate", label: "Moderate", range: [0.30, 0.50], colour: "#a16207" },
  { value: "high", label: "High", range: [0.50, 0.70], colour: "#b45309" },
  { value: "very_high", label: "Very high", range: [0.70, 1.01], colour: "#c62828" },
];

const REGION_SHORT = {
  rangpur_rajshahi: "Rangpur", sylhet: "Sylhet", sw_coastal: "Coast", eastern_plains: "Eastern",
  north_west: "North-west", jamuna_east: "Jamuna east", haor: "Haor", central: "Central",
  west_central: "West-central", south_west: "South-west", chattogram_coast: "Chattogram",
  cht: "Hill Tracts",
};

// Regions whose results are too weak to read like the others.
const REGION_CAUTION = {
  west_central: "Low data. Few floods were mapped here, the model's skill varies widely " +
    "between held-out blocks (AUC 0.72 ± 0.19) and its scores agree only weakly with observed " +
    "floods (0.57). Read them as indicative, not as a ranking to act on.",
};

function showCaution(id) {
  const box = $("regionCaution");
  box.textContent = REGION_CAUTION[id] || "";
  box.hidden = !REGION_CAUTION[id];
}

const TYPE_LABELS = {
  bridge: "Bridge", cropland: "Cropland", embankment: "Embankment", fishpond: "Fish pond",
  flood_shelter: "Flood shelter", hospital: "Hospital & clinic", irrigation: "Irrigation channel",
  market: "Market", railway: "Railway", road: "Road", school: "School",
};

const OVERLAYS = {
  landslide: "Landslide susceptibility",
  flood_risk: "Kriged hazard surface",
  hand: "Height above drainage",
  slope: "Slope",
  dem: "Elevation",
  kriging_variance: "Kriging variance",
};

const state = {
  regions: [],
  hazard: "flood",
  view: ALL,
  summary: null,
  basemap: "topo",
  panel: null,
  layers: {
    assets: true, areas: true, area_lines: true, hotspots: true,
    landslide: true, ls_areas: false, ls_points: true,
  },
  filters: { regions: new Set(), types: new Set(), classes: new Set(), min: 0, max: 1 },
  // The area chosen in the pickers or on the map, and the choice at each level.
  area: null,
  picks: {},
  areaCache: {},
  // The landslide view filters areas and mapped landslides, not assets.
  ls: {
    summary: null, stats: null,
    filters: { categories: new Set(), min: 0, max: 1 },
  },
};

// The surface is drawn with matplotlib's OrRd; the upazilas use the same
// ramp over the same range, so one legend serves both.
const OR_RD = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"];
const LANDSLIDE_POINT = "#3b0764";

const $ = (id) => document.getElementById(id);
const fmt = (n) => (n === null || n === undefined ? "—" : Number(n).toLocaleString());
const fixed = (n, d = 3) => (n === null || n === undefined || Number.isNaN(Number(n)) ? "—" : Number(n).toFixed(d));
const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const regionName = (id) => state.regions.find((r) => r.id === id)?.name || id;
const typeLabel = (type) => TYPE_LABELS[type] || String(type || "asset").replace(/_/g, " ");

async function getJSON(address) {
  const response = await fetch(address);
  if (!response.ok) throw new Error(`${address}: ${response.status}`);
  return response.json();
}

function debounce(fn, wait) {
  let timer = null;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); };
}

function showLoading(label) {
  $("loadingLabel").textContent = label;
  $("loading").hidden = false;
}
const hideLoading = () => ($("loading").hidden = true);

/* ── Map and basemaps ─────────────────────────────────────────────────────
 * Three basemaps built from one set of sources. Satellite imagery is always
 * drawn; relief is a hillshade computed in the browser from elevation tiles,
 * laid over the imagery; the hybrid adds road and place-name labels on top.
 * Switching is a visibility change, so no tiles are refetched. */
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services";
const BASEMAPS = {
  satellite: { relief: false, labels: false },
  topo: { relief: true, labels: false },
  hybrid: { relief: true, labels: true },
};
const LABEL_LAYERS = ["basemap-roads", "basemap-places"];

const protocol = new pmtiles.Protocol();
maplibregl.addProtocol("pmtiles", protocol.tile);

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      imagery: {
        type: "raster",
        tiles: [`${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`],
        tileSize: 256, maxzoom: 19,
        attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics",
      },
      elevation: {
        type: "raster-dem",
        tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
        encoding: "terrarium", tileSize: 256, maxzoom: 14,
        attribution: "Relief: Mapzen terrain tiles, AWS Open Data",
      },
      roads: {
        type: "raster",
        tiles: [`${ESRI}/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}`],
        tileSize: 256, maxzoom: 19,
      },
      places: {
        type: "raster",
        tiles: [`${ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`],
        tileSize: 256, maxzoom: 19,
        attribution: "Labels &copy; Esri &middot; Asset data &copy; OpenStreetMap contributors",
      },
    },
    layers: [
      {
        id: "basemap-imagery", type: "raster", source: "imagery",
        // a touch less saturated, so the risk colours stand out from the fields
        paint: { "raster-saturation": -0.15, "raster-contrast": 0.05 },
      },
      {
        id: "basemap-relief", type: "hillshade", source: "elevation",
        paint: {
          "hillshade-exaggeration": 0.55,
          "hillshade-shadow-color": "#0b1220",
          "hillshade-highlight-color": "#fff6e8",
          "hillshade-accent-color": "#1b2a3a",
          "hillshade-illumination-direction": 315,
        },
      },
      { id: "basemap-roads", type: "raster", source: "roads", paint: { "raster-opacity": 0.75 } },
      { id: "basemap-places", type: "raster", source: "places" },
    ],
  },
  center: [90.3, 23.8],
  zoom: 6.3,
  attributionControl: { compact: true },
});
// Bottom right, stacked just above the attribution button.
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
map.addControl(new maplibregl.ScaleControl({ maxWidth: 110, unit: "metric" }), "bottom-left");

// Captured at creation: the region list can arrive after the map has loaded,
// and a listener attached then would wait for an event that already fired.
const mapReady = new Promise((resolve) => map.once("load", resolve));

// The credits are written out in the right panel, so on the map they stay folded
// into the ⓘ button, which Esri's terms still require to be on the map.
// MapLibre opens a compact attribution on wide screens; close it once loaded.
map.once("load", () =>
  document.querySelector(".maplibregl-ctrl-attrib")?.classList.remove("maplibregl-compact-show"));

function setBasemap(name) {
  const choice = BASEMAPS[name] || BASEMAPS.topo;
  state.basemap = name;
  const show = (on) => (on ? "visible" : "none");
  map.setLayoutProperty("basemap-relief", "visibility", show(choice.relief));
  LABEL_LAYERS.forEach((id) => map.setLayoutProperty(id, "visibility", show(choice.labels)));
  document.querySelectorAll("#basemaps button").forEach((button) =>
    button.setAttribute("aria-checked", String(button.dataset.basemap === name)));
}

/* ── Regions in view ──────────────────────────────────────────────────── */
const viewRegions = () => (state.view === ALL
  ? state.regions.filter((r) => r.has_assets).map((r) => r.id) : [state.view]);
const regionInfo = (id) => (state.summary?.regions || []).find((r) => r.id === id) || {};
const assetRegions = () => viewRegions().filter((id) => regionInfo(id).has_assets);
const regionShown = (id) => state.filters.regions.size === 0 || state.filters.regions.has(id);
const assetLayerIds = () => assetRegions().map((r) => `assets-${r}`).filter((id) => map.getLayer(id));

/* ── Region layers ────────────────────────────────────────────────────── */
function removeRegionLayers() {
  const prefixes = ["assets-", "adm-", "hotspots-", "overlay-", "ls-"];
  (map.getStyle().layers || []).forEach((layer) => {
    if (prefixes.some((p) => layer.id.startsWith(p))) map.removeLayer(layer.id);
  });
  Object.keys(map.getStyle().sources || {}).forEach((id) => {
    if (prefixes.some((p) => id.startsWith(p))) map.removeSource(id);
  });
}

function addRegionLayers(region) {
  const base = `pmtiles://${url(`tiles/${region}`)}`;
  // Layers sit under the hybrid's labels, above the administrative areas.
  map.addSource(`hotspots-${region}`, { type: "vector", url: `${base}/hotspots.pmtiles` });
  map.addLayer({
    id: `hotspots-${region}`, type: "fill", source: `hotspots-${region}`, "source-layer": "hotspots",
    paint: { "fill-color": "#c62828", "fill-opacity": 0.22 },
  }, "basemap-roads");

  map.addSource(`assets-${region}`, { type: "vector", url: `${base}/assets.pmtiles` });
  map.addLayer({
    id: `assets-${region}`, type: "circle", source: `assets-${region}`, "source-layer": "assets",
    // the most susceptible sites draw last, so they stay visible in dense areas
    layout: { "circle-sort-key": ["coalesce", ["get", "flood_risk"], 0] },
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 1.2, 9, 2.6, 11, 4, 14, 7],
      "circle-color": ["step", ["coalesce", ["get", "flood_risk"], 0],
        "#15803d", 0.30, "#a16207", 0.50, "#b45309", 0.70, "#c62828"],
      "circle-stroke-color": "#ffffff",
      // no outline at regional zoom, where outlines merge into white streaks
      "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 9, 0, 11, 0.8, 14, 1.2],
      "circle-stroke-opacity": 0.9,
      // faint at country scale, so the division and district colours read
      "circle-opacity": ["interpolate", ["linear"], ["zoom"], 6, 0.35, 8.5, 0.8, 10, 0.95],
    },
  }, "basemap-roads");        // under the hybrid's labels, which stay readable
}

/* ── Administrative areas ─────────────────────────────────────────────────
 * Divisions, districts, upazilas and unions are drawn one level at a time,
 * chosen by zoom, each coloured by its own figures. Picking an area, from
 * the dropdowns or by clicking it on the map, frames it, outlines it and
 * narrows the assets and the counters to it. */
const AREA_LEVELS = {
  flood: ["division", "district", "upazila", "union"],
  landslide: ["district", "upazila", "union"],
};
// The zooms at which each level's colours are drawn; its outlines carry on in.
const AREA_ZOOM = {
  flood: { division: [0, 7.4], district: [7.4, 8.8], upazila: [8.8, 10.4], union: [10.4, 24] },
  landslide: { district: [0, 8.6], upazila: [8.6, 10.4], union: [10.4, 24] },
};
const LEVEL_LABEL = { division: "Division", district: "District", upazila: "Upazila", union: "Union" };
const LINE_WIDTH = { division: 2.2, district: 1.5, upazila: 0.9, union: 0.5 };
// Flood areas are coloured by the share of their assets in the High or Very
// high class: mean scores sit below 0.30 almost everywhere, so they would
// paint every division the same.
const SHARE_STOPS = [[0, "#eef5ff"], [0.02, "#c9ddfb"], [0.05, "#97c0f6"], [0.10, "#5b9bec"],
  [0.20, "#2563c9"], [0.40, "#1e3a8a"]];
// An area with no mapped assets keeps only its outline.
const NO_DATA = "rgba(0, 0, 0, 0)";
const FILL_OPACITY = { flood: 0.5, landslide: 0.62 };
const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0)} %`);

const levelsOf = () => AREA_LEVELS[state.hazard];
function visibleLevel() {
  const z = map.getZoom();
  const bands = AREA_ZOOM[state.hazard];
  return levelsOf().find((l) => z >= bands[l][0] && z < bands[l][1]) || levelsOf().at(-1);
}

function areaColour() {
  if (state.hazard === "landslide") {
    const d = state.ls.summary?.display || {};
    return rampExpression("susceptibility_mean", d.vmin ?? 0, d.vmax ?? 1);
  }
  return ["case", ["has", "share_high"],
    ["interpolate", ["linear"], ["get", "share_high"], ...SHARE_STOPS.flat()], NO_DATA];
}

function addAreaLayers() {
  const hazard = state.hazard;
  const levels = levelsOf();
  levels.forEach((level) => {
    map.addSource(`adm-${level}`, {
      type: "vector", url: `pmtiles://${url(`tiles/admin/${hazard}_${level}.pmtiles`)}`,
    });
    const [from, to] = AREA_ZOOM[hazard][level];
    map.addLayer({
      id: `adm-fill-${level}`, type: "fill", source: `adm-${level}`, "source-layer": level,
      minzoom: from, maxzoom: to,
      paint: { "fill-color": areaColour(), "fill-opacity": FILL_OPACITY[hazard] },
    }, "basemap-roads");
  });
  // Outlines from the finest up, so a district's edge draws over its unions'.
  [...levels].reverse().forEach((level) => map.addLayer({
    id: `adm-line-${level}`, type: "line", source: `adm-${level}`, "source-layer": level,
    minzoom: AREA_ZOOM[hazard][level][0],
    paint: { "line-color": "#ffffff", "line-width": LINE_WIDTH[level], "line-opacity": 0.8 },
  }, "basemap-roads"));
  levels.forEach((level) => map.addLayer({
    id: `adm-sel-${level}`, type: "line", source: `adm-${level}`, "source-layer": level,
    filter: ["==", ["get", "id"], ""],
    paint: { "line-color": "#0f172a", "line-width": 2.6 },
  }, "basemap-roads"));
}

// The flood regions the view covers, or none for no restriction.
const shownRegions = () => (state.view === ALL ? [...state.filters.regions] : [state.view]);

function inShownRegions(area) {
  if (state.hazard === "landslide") return true;
  const ids = shownRegions();
  if (!ids.length) return true;
  if (area.level === "division") return (area.regions || "").split(",").some((r) => ids.includes(r));
  return ids.includes(area.region);
}

function applyAreaFilters() {
  const ids = state.hazard === "landslide" ? [] : shownRegions();
  const f = state.ls.filters;
  levelsOf().forEach((level) => {
    const conditions = ["all"];
    if (ids.length) {
      // regions are matched whole: "central" must not match "west_central"
      conditions.push(level === "division"
        ? ["any", ...ids.map((id) => ["in", `,${id},`, ["concat", ",", ["coalesce", ["get", "regions"], ""], ","]])]
        : ["in", ["coalesce", ["get", "region"], ""], ["literal", ids]]);
    }
    if (state.hazard === "landslide" && (f.min > 0 || f.max < 1)) {
      const value = ["coalesce", ["get", "susceptibility_mean"], 0];
      conditions.push([">=", value, f.min], ["<=", value, f.max]);
    }
    const filter = conditions.length > 1 ? conditions : null;
    [`adm-fill-${level}`, `adm-line-${level}`].forEach((id) => map.getLayer(id) && map.setFilter(id, filter));
    if (map.getLayer(`adm-sel-${level}`)) {
      map.setFilter(`adm-sel-${level}`, ["==", ["get", "id"], state.area?.level === level ? state.area.id : ""]);
    }
  });
}

async function areaLists() {
  const hazard = state.hazard;
  if (!state.areaCache[hazard]) {
    const levels = levelsOf().filter((l) => l !== "union").join(",");
    const { areas } = await getJSON(url(`api/areas/${hazard}?levels=${levels}`));
    const byLevel = {};
    areas.forEach((a) => (byLevel[a.level] ||= []).push(a));
    state.areaCache[hazard] = { byLevel, unions: {} };
  }
  return state.areaCache[hazard];
}

// Unions are many, so they are fetched one upazila at a time.
async function unionsOf(upazila) {
  const cache = await areaLists();
  if (!cache.unions[upazila]) {
    const { areas } = await getJSON(url(
      `api/areas/${state.hazard}?levels=union&parent=${encodeURIComponent(upazila)}`));
    cache.unions[upazila] = areas;
  }
  return cache.unions[upazila];
}

async function renderAreaPicker() {
  const cache = await areaLists();
  const picks = state.picks;
  const levels = levelsOf();
  const html = [];
  for (const [i, level] of levels.entries()) {
    const above = levels.slice(0, i).reverse().find((l) => picks[l]);
    let options = [];
    let waiting = "";
    if (level === "union") {
      if (picks.upazila) options = await unionsOf(picks.upazila);
      else waiting = "Choose an upazila first";
    } else {
      options = (cache.byLevel[level] || []).filter((a) => !above || a[above] === picks[above]);
    }
    options = options.filter(inShownRegions).sort((a, b) => a.name.localeCompare(b.name));
    // a long upazila list reads better grouped by district
    const grouped = level === "upazila" && !picks.district;
    let body = "";
    if (grouped) {
      const groups = {};
      options.forEach((a) => (groups[a.district] ||= []).push(a));
      body = Object.keys(groups).sort().map((d) => `<optgroup label="${esc(d)}">${groups[d].map((a) =>
        `<option value="${esc(a.id)}" ${a.id === picks[level] ? "selected" : ""}>${esc(a.name)}</option>`).join("")}</optgroup>`).join("");
    } else {
      body = options.map((a) =>
        `<option value="${esc(a.id)}" ${a.id === picks[level] ? "selected" : ""}>${esc(a.name)}</option>`).join("");
    }
    html.push(`<div class="area-row ${picks[level] ? "active" : ""}">
      <label for="area-${level}">${LEVEL_LABEL[level]}</label>
      <select id="area-${level}" data-level="${level}" ${waiting ? "disabled" : ""}>
        <option value="">${waiting || "All"}</option>${body}</select></div>`);
  }
  $("areaPicker").innerHTML = html.join("");
  $("areaPicker").querySelectorAll("select").forEach((select) =>
    select.addEventListener("change", () => pickArea(select.dataset.level, select.value)));
}

/* Choose an area at a level, or clear that level with an empty id, in which
 * case the choice falls back to the level above. */
async function pickArea(level, id, { fly = true } = {}) {
  const levels = levelsOf();
  const i = levels.indexOf(level);
  levels.slice(i).forEach((l) => (state.picks[l] = ""));
  let detail = null;
  if (id) {
    detail = await getJSON(url(`api/area/${state.hazard}/${level}/${encodeURIComponent(id)}`));
    levels.slice(0, i).forEach((l) => (state.picks[l] = detail[l] || ""));
    state.picks[level] = id;
  }
  const deepest = [...levels].reverse().find((l) => state.picks[l]);
  state.area = deepest ? { level: deepest, id: state.picks[deepest] } : null;
  if (state.area && !detail) {
    detail = await getJSON(url(`api/area/${state.hazard}/${deepest}/${encodeURIComponent(state.area.id)}`));
  }
  await renderAreaPicker();
  applyFilters();
  if (detail) {
    if (fly) {
      map.fitBounds([[detail.west, detail.south], [detail.east, detail.north]],
        { padding: framePadding(), maxZoom: 13, duration: 700 });
    }
    showAreaCard(detail);
  } else {
    $("detail").hidden = true;
    const extent = state.hazard === "landslide" ? state.ls.summary?.extent : state.summary?.extent;
    if (fly && extent) {
      const [w, s, e, n] = extent;
      map.fitBounds([[w, s], [e, n]], { padding: framePadding(), duration: 700 });
    }
  }
}

function clearArea() {
  state.picks = {};
  state.area = null;
}

function showAreaCard(d) {
  const levels = levelsOf();
  const next = levels[levels.indexOf(d.level) + 1];
  const chain = [d.level === "union" ? d.upazila_name : null,
    d.level !== "district" && d.level !== "division" ? d.district : null,
    d.level !== "division" && state.hazard === "flood" ? d.division : null].filter(Boolean);
  const tags = `<span class="tag">${LEVEL_LABEL[d.level]}</span>${chain.map((c) => `<span class="tag">${esc(c)}</span>`).join("")}`;
  const people = `${fmt(Math.round(d.population || 0))} people`;
  const children = (d.top_children || []).length ? `
    <div class="children"><div class="children-head">${state.hazard === "flood" ? "Most exposed" : "Most susceptible"} ${LEVEL_LABEL[next].toLowerCase()}s</div>
      ${d.top_children.map((c) => `<div><a href="#" data-level="${esc(c.level)}" data-id="${esc(c.id)}">${esc(c.name)}</a>
        <span class="num">${state.hazard === "flood" ? pct(c.share_high) : fixed(c.susceptibility_mean)}</span></div>`).join("")}</div>` : "";
  let body;
  if (state.hazard === "landslide") {
    body = `
      <div class="meta">${people} · ${fmt(d.n_landslides || 0)} landslides mapped on 6 August 2023</div>
      <div class="probability" style="color:#b30000">${fixed(d.susceptibility_mean)}</div>
      <div class="probability-note">mean modelled landslide susceptibility; the highest 30 m cell reaches ${fixed(d.susceptibility_max)}</div>`;
  } else if (!d.n_assets) {
    body = `<div class="meta">${people}</div>
      <div class="probability-note">No assets are mapped here: the area lies outside the flood regions
        or holds no infrastructure in OpenStreetMap.</div>`;
  } else {
    const counts = [["Hospitals &amp; clinics", d.n_hospital], ["Schools", d.n_school],
      ["Bridges", d.n_bridge], ["Flood shelters", d.n_shelter]];
    body = `
      <div class="meta">${people} · ${fmt(d.n_assets)} assets · ${esc(regionName(d.region))}${(d.regions || "").includes(",") ? " and others" : ""}</div>
      <div class="probability" style="color:#1e3a8a">${pct(d.share_high)}</div>
      <div class="probability-note">of its assets score High or Very high (0.50 and above), ${fmt(d.n_high)} in all</div>
      <div class="factor"><span>Mean score</span>
        <span class="bar"><span style="width:${Math.round((d.mean_score || 0) * 100)}%;background:${riskColour(d.mean_score || 0)}"></span></span>
        <span class="num">${fixed(d.mean_score)}</span></div>
      <div class="children">${counts.map(([label, n]) => `<div><span>${label}</span><span class="num">${fmt(n || 0)}</span></div>`).join("")}</div>`;
  }
  showCard(`${tags}<h3>${esc(d.name)}</h3>${body}${children}`);
  $("detail").querySelectorAll("a[data-id]").forEach((link) => link.addEventListener("click", (event) => {
    event.preventDefault();
    pickArea(link.dataset.level, link.dataset.id);
  }));
}

async function areaClick(event) {
  areaTip.remove();
  const layer = `adm-fill-${visibleLevel()}`;
  if (!map.getLayer(layer)) return;
  const [feature] = map.queryRenderedFeatures(event.point, { layers: [layer] });
  if (feature) await pickArea(feature.properties.level, feature.properties.id);
}

const areaTip = new maplibregl.Popup({ closeButton: false, closeOnClick: false, className: "area-tip", offset: 14, maxWidth: "260px" });

function areaHover(event) {
  const layer = `adm-fill-${visibleLevel()}`;
  const [feature] = map.getLayer(layer) ? map.queryRenderedFeatures(event.point, { layers: [layer] }) : [];
  if (!feature) { areaTip.remove(); return false; }
  const p = feature.properties;
  const value = state.hazard === "landslide"
    ? `mean susceptibility ${fixed(p.susceptibility_mean)} · ${fmt(p.n_landslides || 0)} landslides`
    : p.n_assets ? `${pct(p.share_high)} of ${fmt(p.n_assets)} assets High or Very high` : "no assets mapped";
  areaTip.setLngLat(event.lngLat)
    .setHTML(`<b>${esc(p.name)}</b> <span>${LEVEL_LABEL[p.level]}</span><br>${value}`).addTo(map);
  return true;
}

/* Overlays are PNGs of several megabytes each, so one is fetched only when
 * its layer is first switched on, and only for the regions in view. */
async function ensureOverlay(name, region) {
  const id = `overlay-${name}-${region}`;
  if (map.getSource(id)) return true;
  let meta;
  try {
    meta = await getJSON(url(`overlays/${region}/${name}.json`));
  } catch {
    return false;
  }
  if (map.getSource(id)) return true;         // added while this one waited
  const [[south, west], [north, east]] = meta.bounds;
  map.addSource(id, {
    type: "image",
    url: url(`overlays/${region}/${name}.png`),
    coordinates: [[west, north], [east, north], [east, south], [west, south]],
  });
  const below = (map.getStyle().layers || []).find((l) => l.id.startsWith("adm-fill-"))?.id
    || "basemap-roads";
  map.addLayer({
    id, type: "raster", source: id,
    paint: { "raster-opacity": name === "landslide" ? 0.68 : 0.55 },
  }, below);
  return true;
}

async function applyLayers() {
  const show = (on) => (on ? "visible" : "none");
  // Colours fade rather than hide, so a click still finds the area beneath.
  const fillOn = state.layers[state.hazard === "landslide" ? "ls_areas" : "areas"];
  levelsOf().forEach((level) => {
    if (map.getLayer(`adm-fill-${level}`)) {
      map.setPaintProperty(`adm-fill-${level}`, "fill-opacity", fillOn ? FILL_OPACITY[state.hazard] : 0);
    }
    if (map.getLayer(`adm-line-${level}`)) {
      map.setLayoutProperty(`adm-line-${level}`, "visibility", show(state.layers.area_lines));
    }
  });
  if (state.hazard === "landslide") {
    if (map.getLayer("ls-points")) map.setLayoutProperty("ls-points", "visibility", show(state.layers.ls_points));
    const region = state.view;
    for (const name of state.ls.summary?.region?.overlays || []) {
      const id = `overlay-${name}-${region}`;
      const wanted = Boolean(state.layers[name]);
      if (wanted && !map.getLayer(id)) await ensureOverlay(name, region);
      if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", show(wanted));
    }
    return;
  }
  for (const region of viewRegions()) {
    const shown = regionShown(region);
    for (const [key, ids] of Object.entries({
      assets: [`assets-${region}`],
      hotspots: [`hotspots-${region}`],
    })) {
      ids.forEach((id) => map.getLayer(id) && map.setLayoutProperty(id, "visibility", show(state.layers[key] && shown)));
    }
    for (const name of regionInfo(region).overlays || []) {
      const id = `overlay-${name}-${region}`;
      const wanted = Boolean(state.layers[name]) && shown;
      if (wanted && !map.getLayer(id)) await ensureOverlay(name, region);
      if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", show(wanted));
    }
  }
}

/* ── Filters ──────────────────────────────────────────────────────────── */
function filterExpression() {
  const f = state.filters;
  const score = ["coalesce", ["get", "flood_risk"], 0];
  const conditions = ["all", [">=", score, f.min], ["<=", score, f.max]];
  if (f.types.size) conditions.push(["in", ["get", "asset_type"], ["literal", [...f.types]]]);
  if (state.area) conditions.push(["==", ["get", `adm_${state.area.level}`], state.area.id]);
  if (f.classes.size) {
    conditions.push(["any", ...CLASSES.filter((c) => f.classes.has(c.value)).map((c) =>
      ["all", [">=", score, c.range[0]], ["<", score, c.range[1]]])]);
  }
  return conditions;
}

const activeFilters = () => (state.hazard === "landslide" ? state.ls.filters : state.filters);

const filtersActive = () => {
  if (state.hazard === "landslide") {
    const f = state.ls.filters;
    return Boolean(state.area) || f.categories.size > 0 || f.min > 0 || f.max < 1;
  }
  const f = state.filters;
  return Boolean(state.area) || f.regions.size + f.types.size + f.classes.size > 0 || f.min > 0 || f.max < 1;
};

function applyFilters() {
  if (state.hazard === "landslide") { applyLandslideFilters(); return; }
  const expression = filterExpression();
  assetLayerIds().forEach((id) => map.setFilter(id, expression));
  applyAreaFilters();
  $("resetFilters").hidden = !filtersActive();
  applyLayers();
  refreshCounters();
}

/* A multi-select dropdown. An empty selection means no restriction, which
 * the button reads as "All". The list opens inside the panel rather than as
 * a popover, so the panel's scrolling never clips it. */
function multiSelect(root, config) {
  const { label, selected, onChange, searchable = false } = config;
  let options = config.options;
  root.innerHTML = `
    <button type="button" class="ms-button" aria-expanded="false">
      <span class="ms-label">${esc(label)}</span>
      <span class="ms-summary"></span>
      <svg class="ms-chevron" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M6 9l6 6 6-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </button>
    <div class="ms-menu">
      ${searchable ? `<input type="search" class="ms-search" placeholder="Search ${esc(label.toLowerCase())}…">` : ""}
      <div class="ms-actions">
        <button type="button" class="link-button" data-act="all">Select all</button>
        <button type="button" class="link-button" data-act="none">Clear</button>
      </div>
      <div class="ms-list" role="listbox" aria-multiselectable="true"></div>
    </div>`;
  const button = root.querySelector(".ms-button");
  const list = root.querySelector(".ms-list");
  const search = root.querySelector(".ms-search");

  const summarise = () => {
    const summary = root.querySelector(".ms-summary");
    const picked = options.filter((o) => selected.has(o.value));
    summary.textContent = picked.length === 0 ? "All"
      : picked.length === 1 ? picked[0].label : `${picked.length} selected`;
    summary.classList.toggle("active", picked.length > 0);
  };

  const visible = () => {
    const query = (search?.value || "").trim().toLowerCase();
    return options.filter((o) => !query || o.label.toLowerCase().includes(query)
      || (o.group || "").toLowerCase().includes(query));
  };

  const render = () => {
    const shown = visible();
    let group = null;
    list.innerHTML = shown.map((o) => {
      const head = o.group && o.group !== group ? `<div class="ms-group">${esc(o.group)}</div>` : "";
      group = o.group || group;
      return `${head}<label class="ms-option">
        <input type="checkbox" value="${esc(o.value)}" ${selected.has(o.value) ? "checked" : ""}>
        ${o.swatch ? `<span class="swatch" style="background:${o.swatch}"></span>` : ""}
        <span class="name">${esc(o.label)}</span>
        ${o.count !== undefined ? `<span class="count">${fmt(o.count)}</span>` : ""}
      </label>`;
    }).join("") || `<div class="ms-empty">Nothing matches</div>`;
    summarise();
  };

  button.addEventListener("click", () => {
    const open = !root.classList.contains("open");
    document.querySelectorAll(".ms.open").forEach((other) => other !== root && other.classList.remove("open"));
    root.classList.toggle("open", open);
    button.setAttribute("aria-expanded", String(open));
    if (open && search) search.focus();
  });
  list.addEventListener("change", (event) => {
    const value = event.target.value;
    event.target.checked ? selected.add(value) : selected.delete(value);
    summarise();
    onChange();
  });
  root.querySelector(".ms-actions").addEventListener("click", (event) => {
    const act = event.target.dataset.act;
    if (!act) return;
    if (act === "all") visible().forEach((o) => selected.add(o.value));
    if (act === "none") selected.clear();
    render();
    onChange();
  });
  search?.addEventListener("input", render);

  render();
  return {
    setOptions(next) {
      options = next;
      const allowed = new Set(next.map((o) => o.value));
      [...selected].forEach((value) => allowed.has(value) || selected.delete(value));
      render();
    },
    refresh: render,
  };
}

const controls = {};

function buildFilters() {
  const summary = state.summary;
  const f = state.filters;
  const hasAssets = Boolean(summary.region.has_assets);
  $("filterSection").hidden = !hasAssets;
  if (!hasAssets) return;

  const assetTotals = {};
  (summary.counts.by_district || []).forEach((d) => (assetTotals[d.region] = (assetTotals[d.region] || 0) + d.n));
  controls.region = multiSelect($("fRegion"), {
    label: "Region", selected: f.regions,
    options: viewRegions().map((id) => ({
      value: id, label: regionName(id),
      count: regionInfo(id).has_assets ? assetTotals[id] || 0 : undefined,
    })),
    // the pickers list only the chosen regions' areas
    onChange: () => { clearArea(); renderAreaPicker(); applyFilters(); },
  });
  $("fRegion").hidden = state.view !== ALL;
  $("fClass").hidden = false;              // the landslide view may have hidden it
  $("fType").hidden = false;
  renderAreaPicker();

  controls.type = multiSelect($("fType"), {
    label: "Asset type", selected: f.types,
    options: Object.entries(summary.counts.by_type || {})
      .sort((a, b) => b[1] - a[1])
      .map(([type, n]) => ({ value: type, label: typeLabel(type), count: n })),
    onChange: applyFilters,
  });
  controls.class = multiSelect($("fClass"), {
    label: "Class", selected: f.classes,
    options: CLASSES.map((c) => ({ value: c.value, label: c.label, swatch: c.colour })),
    onChange: applyFilters,
  });
  syncScoreSlider();
}

function syncScoreSlider() {
  const f = activeFilters();
  $("scoreMin").value = f.min;
  $("scoreMax").value = f.max;
  $("scoreFill").style.left = `${f.min * 100}%`;
  $("scoreFill").style.right = `${(1 - f.max) * 100}%`;
  $("scoreOut").textContent = `${f.min.toFixed(2)} – ${f.max.toFixed(2)}`;
}

const scoreChanged = debounce(applyFilters, 120);
["scoreMin", "scoreMax"].forEach((id) => $(id).addEventListener("input", (event) => {
  const f = activeFilters();
  let min = Number($("scoreMin").value);
  let max = Number($("scoreMax").value);
  // the handles may not cross: the one being dragged stops at the other
  if (min > max) {
    if (event.target.id === "scoreMin") min = max; else max = min;
  }
  f.min = min;
  f.max = max;
  syncScoreSlider();
  scoreChanged();
}));

function resetFilters() {
  if (state.hazard === "landslide") {
    const f = state.ls.filters;
    f.categories.clear();
    f.min = 0;
    f.max = 1;
    clearArea();
    $("detail").hidden = true;
    buildLandslideFilters();
    applyFilters();
    return;
  }
  const f = state.filters;
  clearArea();
  $("detail").hidden = true;
  [f.regions, f.types, f.classes].forEach((set) => set.clear());
  f.min = 0;
  f.max = 1;
  if (state.summary) buildFilters();
  applyFilters();
}

/* ── Counters ─────────────────────────────────────────────────────────── */
const refreshCounters = debounce(async () => {
  if (state.hazard === "landslide") { await refreshLandslideCounters(); return; }
  const summary = state.summary;
  if (!summary) return;
  const cht = (summary.metrics_by_region?.cht || {}).landslide_model;

  if (!summary.region.has_assets) {
    const inventory = cht?.inventory || {};
    const validation = cht?.validation || {};
    renderCounters([
      [fmt(inventory.n_landslides), "Landslides mapped"],
      [fmt(inventory.n_background), "Background points"],
      [fixed(validation.val_auc_roc), "Validation AUC"],
      [fixed(validation.val_average_precision), "Average precision"],
    ], false);
    return;
  }

  const f = state.filters;
  const regions = assetRegions().filter(regionShown);
  const params = new URLSearchParams({
    regions: regions.join(","), types: [...f.types].join(","),
    area: state.area ? `${state.area.level}:${state.area.id}` : "",
    classes: [...f.classes].join(","), min_score: f.min, max_score: f.max,
  });
  const stats = regions.length
    ? await getJSON(url(`api/stats?${params}`))
    : { assets: 0, high_risk: 0, mean_risk: null, by_type: {} };
  const t = stats.by_type || {};
  const rows = [
    [fmt(stats.assets), "Assets shown"],
    [fmt(stats.high_risk), "Very high susceptibility"],
    [fixed(stats.mean_risk), "Mean score"],
    [fmt(t.hospital || 0), "Hospitals &amp; clinics"],
    [fmt(t.school || 0), "Schools"],
    [fmt(t.bridge || 0), "Bridges"],
    [fmt(t.flood_shelter || 0), "Flood shelters"],
    [fmt(t.cropland || 0), "Cropland parcels"],
  ];
  if (state.view === ALL && cht && regionShown("cht")) {
    rows.push([fmt((cht.inventory || {}).n_landslides), "Landslides mapped"]);
  }
  renderCounters(rows, filtersActive());
}, 150);

function renderCounters(rows, filtered) {
  $("counters").innerHTML = (filtered ? `<div class="filtered">Filtered view</div>` : "")
    + rows.map(([value, label]) =>
      `<div class="row"><span class="value">${value}</span><span class="label">${label}</span></div>`).join("");
}

/* ── Panel: layers and provenance ─────────────────────────────────────── */
function renderLayerToggles() {
  if (state.hazard === "landslide") {
    const overlays = new Set(state.ls.summary?.region?.overlays || []);
    const entries = [
      ...(overlays.has("landslide") ? [["landslide", "Susceptibility surface"]] : []),
      ["ls_areas", "Area colours"],
      ["area_lines", "Area boundaries"],
      ["ls_points", "Mapped landslides"],
      ...["hand", "slope", "dem"].filter((n) => overlays.has(n)).map((n) => [n, OVERLAYS[n]]),
    ];
    drawToggles(entries);
    return;
  }
  const available = new Set();
  viewRegions().forEach((id) => (regionInfo(id).overlays || []).forEach((name) => available.add(name)));
  const hasAssets = Boolean(state.summary?.region?.has_assets);
  const entries = [
    ...(hasAssets ? [["assets", "Asset markers"], ["areas", "Area colours"], ["area_lines", "Area boundaries"],
      ["hotspots", "Hotspots (Gi*, 95%)"]] : []),
    ...Object.keys(OVERLAYS).filter((name) => available.has(name)).map((name) => [name, OVERLAYS[name]]),
  ];
  drawToggles(entries);
}

function drawToggles(entries) {
  $("layers").innerHTML = entries.map(([key, label]) => `
    <label><span>${esc(label)}</span>
      <span class="switch"><input type="checkbox" data-layer="${key}" ${state.layers[key] ? "checked" : ""}><span></span></span>
    </label>`).join("");
  $("layers").querySelectorAll("input").forEach((input) =>
    input.addEventListener("change", () => {
      state.layers[input.dataset.layer] = input.checked;
      applyLayers();
    }));
}

function regionValidation(metrics) {
  const meta = metrics.pipeline_metadata || {};
  const held = ((metrics.validation || {}).assets_vs_observed || {}).held_out_blocks || {};
  return { meta, validation: meta.gnn_validation || {}, observed: held };
}

const statRows = (rows) => rows.map(([label, value, note]) => `
  <div class="row"><span class="label">${label}</span><span class="value">${value}</span></div>
  ${note ? `<div class="note">${note}</div>` : ""}`).join("");

function renderProvenance() {
  if (state.hazard === "landslide") {
    $("provenance").innerHTML = landslideModelRows(state.ls.summary?.model || {});
    return;
  }
  const summary = state.summary;
  const box = $("provenance");
  const byRegion = summary.metrics_by_region || {};

  if (state.view === ALL) {
    const totals = {};
    (summary.counts.by_district || []).forEach((d) => (totals[d.region] = (totals[d.region] || 0) + d.n));
    const flood = assetRegions().map((id) => {
      const { validation, observed } = regionValidation(byRegion[id] || {});
      return `<tr><td>${esc(REGION_SHORT[id] || regionName(id))}</td><td>${fmt(totals[id])}</td>
        <td>${fixed(validation.val_auc_roc)}</td><td>${fixed(observed.auc_roc)}</td></tr>`;
    }).join("");
    const cht = (byRegion.cht || {}).landslide_model;
    const landslide = cht ? `<tr><td>Hill Tracts</td><td>${fmt((cht.inventory || {}).n_landslides)}</td>
      <td>${fixed((cht.validation || {}).val_auc_roc)}</td><td>—</td></tr>` : "";
    box.innerHTML = `<table>
        <thead><tr><th>Region</th><th>Assets</th><th>AUC</th><th>vs obs.</th></tr></thead>
        <tbody>${flood}${landslide}</tbody></table>
      <div class="foot">AUC on 10 km blocks held out from training, and against flood extents
        observed by Sentinel-1. The Hill Tracts row counts mapped landslides.</div>`;
    return;
  }

  if (!summary.region.has_assets) {
    const model = summary.metrics.landslide_model || {};
    const inventory = model.inventory || {};
    const validation = model.validation || {};
    box.innerHTML = statRows([
      ["Landslides mapped", fmt(inventory.n_landslides)],
      ["Background points", fmt(inventory.n_background)],
      ["Validation AUC", fixed(validation.val_auc_roc), "held-out 10 km blocks"],
      ["Average precision", fixed(validation.val_average_precision)],
      ["Event dates", esc((inventory.event_dates || []).join(", ") || "—"),
        inventory.single_event ? "one rainfall episode, so this maps that storm" : ""],
    ]);
    return;
  }

  const { meta, validation, observed } = regionValidation(summary.metrics);
  box.innerHTML = statRows([
    ["Processed", esc((meta.generated_at || "").replace("T", " ").slice(0, 16) || "—")],
    ["Model", esc(validation.model_type || "—")],
    ["Assets", fmt(summary.counts.assets)],
    ["Flood-prone labels", meta.label_positive_rate != null ? `${(meta.label_positive_rate * 100).toFixed(1)} %` : "—",
      "share of assets on ground labelled flood-prone"],
    ["Validation AUC", fixed(validation.val_auc_roc), "held-out 10 km blocks"],
    ["Brier score", fixed(validation.val_brier), "lower is better"],
    ["AUC vs observed floods", fixed(observed.auc_roc), "Sentinel-1 flood extents"],
    ["Precision lift", observed.ap_lift != null ? `${fixed(observed.ap_lift, 1)}×` : "—", "over the base rate"],
  ]);
}

/* ── Asset detail ─────────────────────────────────────────────────────── */
function riskColour(score) {
  const found = CLASSES.find((c) => score < c.range[1]);
  return (found || CLASSES[CLASSES.length - 1]).colour;
}

function showDetail(asset) {
  const probability = asset.flood_probability;
  const factors = [
    ["Hazard", asset.cell_hazard, "#c62828"],
    ["Exposure", asset.cell_exposure, "#b45309"],
    ["Vulnerability", asset.cell_vulnerability, "#6d28d9"],
    ["Cell risk", asset.cell_composite_risk, "#2a78d6"],
  ];
  $("detail").innerHTML = `
    <button class="close" id="detailClose" aria-label="Close">×</button>
    <span class="tag">${esc(typeLabel(asset.asset_type))}</span>
    <span class="tag">${esc(REGION_SHORT[asset.region] || regionName(asset.region))}</span>
    <h3>${esc(asset.name || "unnamed")}</h3>
    <div class="meta">${esc(asset.division || "")} · ${Number(asset.lat).toFixed(4)}, ${Number(asset.lon).toFixed(4)} · Rank #${fmt(asset.risk_rank)} in its region</div>
    <div class="probability" style="color:${riskColour(asset.flood_risk || 0)}">
      ${probability != null ? `${Math.round(probability * 100)}%` : fixed(asset.flood_risk)}
    </div>
    <div class="probability-note">${probability != null
      ? "approximate probability that this ground floods, calibrated on other areas of the region; local rates can differ severalfold"
      : "modelled flood susceptibility (ranking score)"}</div>
    ${factors.map(([label, value, colour]) => `
      <div class="factor"><span>${label}</span>
        <span class="bar"><span style="width:${Math.round((value || 0) * 100)}%;background:${colour}"></span></span>
        <span class="num">${fixed(value)}</span></div>`).join("")}`;
  $("detail").hidden = false;
  $("detailClose").addEventListener("click", () => ($("detail").hidden = true));
}

/* ── Full panels ──────────────────────────────────────────────────────── */
async function openPanel(name) {
  if (state.panel === name) { closePanel(); return; }
  state.panel = name;
  document.querySelectorAll("#panelButtons button").forEach((button) =>
    button.setAttribute("aria-pressed", String(button.dataset.panel === name)));
  $("sheet").hidden = false;
  $("sheetTitle").textContent = {
    rankings: "Rankings & export", preparedness: "Preparedness", about: "About the data",
  }[name];
  $("sheetBody").innerHTML = "<p>Loading…</p>";
  $("sheetBody").innerHTML = await panelBody(name);
}

function closePanel() {
  state.panel = null;
  $("sheet").hidden = true;
  document.querySelectorAll("#panelButtons button").forEach((b) => b.setAttribute("aria-pressed", "false"));
}

async function panelBody(name) {
  if (name === "preparedness") return preparednessPanel();
  if (state.hazard === "landslide") {
    return name === "rankings" ? landslideRankings() : landslidePanel(state.ls.summary?.model) + DATA_SOURCES;
  }
  if (name === "rankings") return rankingsPanel();
  return state.view === ALL ? aboutAllPanel() : aboutPanel();
}

async function rankingsPanel() {
  if (!state.summary.region.has_assets) return landslidePanel(state.summary.metrics.landslide_model);
  const view = state.view;
  const [assets, admin] = await Promise.all([
    getJSON(url(`api/region/${view}/assets?limit=200`)),
    getJSON(url(`api/region/${view}/admin?level=union&limit=24`)),
  ]);
  const many = view === ALL;
  const table = assets.assets.slice(0, 50).map((a, i) => `
    <tr><td class="num">${many ? i + 1 : fmt(a.risk_rank)}</td>
    ${many ? `<td>${esc(REGION_SHORT[a.region] || a.region)}</td>` : ""}
    <td>${esc(typeLabel(a.asset_type))}</td><td>${esc(a.name)}</td>
    <td class="num">${fixed(a.flood_risk, 4)}</td><td class="num">${fixed(a.flood_probability, 3)}</td>
    <td>${esc(a.division || "")}</td></tr>`).join("");
  const cards = admin.units.map((u) => `
    <div class="card"><span class="tag">${u.mean_risk >= 0.3 ? "moderate" : "low"}</span>
      ${many ? `<span class="tag">${esc(REGION_SHORT[u.region] || u.region)}</span>` : ""}
      <div style="font-weight:600;margin:6px 0 2px">${esc(u.name)}</div>
      <div style="color:var(--dim);font-size:11px">${esc(u.parent || "")}</div>
      <div class="score" style="color:${riskColour(u.mean_risk || 0)}">${fixed(u.mean_risk)}</div>
      <div style="color:var(--muted);font-size:11px">${fmt(u.n_assets)} assets</div></div>`).join("");
  return `
    <h3>Highest-scoring assets${many ? " across all regions" : ""}</h3>
    <table><thead><tr><th class="num">${many ? "#" : "Rank"}</th>${many ? "<th>Region</th>" : ""}<th>Type</th><th>Name</th>
      <th class="num">Score</th><th class="num">P(flood)</th><th>District</th></tr></thead>
      <tbody>${table}</tbody></table>
    ${many ? `<p style="color:var(--dim);font-size:11.5px">Scores come from a separate model for each region, so across regions they are ordered by score, not compared as equals.</p>` : ""}
    <h3>Union summaries</h3><div class="cards">${cards}</div>
    <h3>Download</h3>
    <p>Outputs may be reused with attribution, subject to the source licences.
      <a href="${url(`api/region/${view}/export.csv`)}">Download ranked assets (CSV)</a></p>`;
}

function landslidePanel(model = {}) {
  const inventory = model.inventory || {};
  const validation = model.validation || {};
  const coefficients = model.standardised_coefficients || {};
  return `
    <h3>Landslide susceptibility model</h3>
    <p>A class-weighted logistic regression fitted to ${fmt(inventory.n_landslides)}
      landslide locations from ${esc(inventory.source || "the inventory")}, against
      ${fmt(inventory.n_background)} background points drawn from
      ${esc(inventory.background_domain || "the mapped area")}. On held-out 10 km blocks
      it reaches an AUC of ${fixed(validation.val_auc_roc)} and an average precision
      of ${fixed(validation.val_average_precision)}.</p>
    ${inventory.single_event ? `<p>Every point comes from one rainfall episode
      (${esc((inventory.event_dates || []).join(", "))}), so the surface describes where
      that storm triggered failures. It is a guide to susceptibility, not a
      complete record of where landslides can happen.</p>` : ""}
    <h3>Standardised coefficients</h3>
    <table><thead><tr><th>Predictor</th><th class="num">Coefficient</th></tr></thead><tbody>
      ${Object.entries(coefficients).map(([name, value]) =>
        `<tr><td>${esc(name)}</td><td class="num">${fixed(value)}</td></tr>`).join("")}
    </tbody></table>
    <p style="color:var(--dim);font-size:11px">${esc(inventory.citation || "")}</p>`;
}

async function preparednessPanel() {
  return `
    <h3>For warnings and emergencies, use these sources</h3>
    <p>The maps here describe long-term susceptibility. They are not forecasts
      and are not monitored in real time.</p>
    <div class="cards">
      <div class="card"><b>Flood Forecasting and Warning Centre (FFWC), BWDB</b>
        <p>Official river-level forecasts and flood warnings.<br>
        <a href="https://www.ffwc.gov.bd" rel="noopener">ffwc.gov.bd</a></p></div>
      <div class="card"><b>Bangladesh Meteorological Department</b>
        <p>Weather, rainfall and cyclone warnings.<br>
        <a href="https://www.bmd.gov.bd" rel="noopener">bmd.gov.bd</a></p></div>
      <div class="card"><b>Department of Disaster Management</b>
        <p>Shelters, relief and disaster response.<br>
        <a href="https://bangladesh.gov.bd" rel="noopener">bangladesh.gov.bd</a></p></div>
      <div class="card"><b>Disaster early-warning voice service</b>
        <p>Dial 1090, toll free: recorded weather and flood warnings.</p></div>
      <div class="card"><b>National Emergency Service</b>
        <p>Dial 999: police, fire service and ambulance.</p></div>
    </div>`;
}

const DATA_SOURCES = `
  <h3>Data sources</h3>
  <p>Infrastructure from OpenStreetMap (ODbL). Elevation from NASA SRTM.
    Flood extents from Copernicus Sentinel-1 via Google Earth Engine.
    Surface water from the EC Joint Research Centre. Population from WorldPop
    (CC BY 4.0). Boundaries from geoBoundaries (CC BY 4.0). Landslide
    locations from NASA COOLR.</p>
  <p>Code: <a href="https://github.com/rakibhhridoy/rimes-mapathon" rel="noopener">github.com/rakibhhridoy/rimes-mapathon</a></p>`;

function aboutAllPanel() {
  const byRegion = state.summary.metrics_by_region || {};
  const rows = assetRegions().map((id) => {
    const m = byRegion[id] || {};
    const { validation, observed } = regionValidation(m);
    const benchmark = ((m.benchmark || {}).summary || {}).graph_vs_best_baseline || {};
    const gap = ((m.label_comparison || {}).summary || {}).observed_minus_proxy || {};
    return `<tr><td>${esc(regionName(id))}</td>
      <td class="num">${fixed(validation.val_auc_roc)}</td>
      <td class="num">${fixed(observed.auc_roc)}</td>
      <td class="num">${observed.ap_lift != null ? `${fixed(observed.ap_lift, 1)}×` : "—"}</td>
      <td class="num">${gap.mean != null ? `+${gap.mean.toFixed(3)}` : "—"}</td>
      <td class="num">${benchmark.auc_difference_mean != null ? benchmark.auc_difference_mean.toFixed(3) : "—"}</td></tr>`;
  }).join("");
  const cht = byRegion.cht?.landslide_model;
  return `
    <p>Fermium Hazard Mapper estimates where flooding would hurt most in three
      regions of Bangladesh, and where slopes in the Chittagong Hill Tracts are
      prone to landslides. For every cell of a roughly 500 m grid it combines
      hazard, which is how likely the ground is to flood, exposure, which is what
      stands on it, and vulnerability, which is how hard it would be for the
      people there to cope. Each region has its own model, trained on flood
      extents Sentinel-1 radar observed there.</p>
    <h3>How well the models do</h3>
    <table><thead><tr><th>Region</th><th class="num">AUC, held-out blocks</th><th class="num">AUC vs observed floods</th>
      <th class="num">Precision lift</th><th class="num">Radar labels gain</th><th class="num">Graph network minus best</th></tr></thead>
      <tbody>${rows}</tbody></table>
    <p style="color:var(--dim);font-size:11.5px">An AUC of 0.5 would be chance. "Radar labels gain" is how
      much training on observed flood extents beats terrain-threshold labels; "graph network minus best"
      is how far the graph neural network trailed the best ordinary model, over repeated block assignments.</p>
    ${cht ? `<h3>Landslides in the Hill Tracts</h3>
      <p>A model fitted to ${fmt((cht.inventory || {}).n_landslides)} mapped landslides reaches an AUC
      of ${fixed((cht.validation || {}).val_auc_roc)} on held-out blocks. Every point comes from one
      storm, so the surface shows where that storm triggered failures.</p>` : ""}
    ${DATA_SOURCES}`;
}

async function aboutPanel() {
  const summary = state.summary;
  if (!summary.region.has_assets) return landslidePanel(summary.metrics.landslide_model) + DATA_SOURCES;
  const { validation, observed } = regionValidation(summary.metrics);
  const benchmark = (summary.metrics.benchmark || {}).summary || {};
  const labels = (summary.metrics.label_comparison || {}).summary || {};
  const gap = labels.observed_minus_proxy || {};
  const past = summary.metrics.past_model || {};
  const pastTest = ((summary.metrics.past_flooding_feature || {}).summary || {});

  return `
    <p>Fermium Hazard Mapper estimates where flooding would hurt most. For every
      cell of a roughly 500 m grid it combines hazard, which is how likely the
      ground is to flood, exposure, which is what stands on it, and
      vulnerability, which is how hard it would be for the people there to cope.</p>
    <h3>How well the model does</h3>
    <p>On ground held out from training as whole 10 km blocks, the model separates
      flood-prone from non-flood-prone locations with an <b>AUC of
      ${fixed(validation.val_auc_roc)}</b>, where 0.5 would be chance. Against flood
      extents observed by Sentinel-1 radar it reaches <b>${fixed(observed.auc_roc)}</b>
      and ranks flooded places ${fixed(observed.ap_lift, 1)} times better than chance.</p>
    ${past.val_auc_roc ? `<h3>What the map shows</h3>
      <p>In this region the asset scores also use where Sentinel-1 saw earlier
      floods reach, because flooding here tends to return to the same ground.
      Trained to predict the ${(past.target_events || []).length > 1 ? "floods" : "flood"}
      of the latest year from the years before it, that model reaches an
      <b>AUC of ${fixed(past.val_auc_roc)}</b> on held-out blocks.
      ${pastTest.with_past ? `Tested on the 2024 floods after training on earlier
      years it reached ${fixed(pastTest.with_past.auc_mean)}, against
      ${fixed(pastTest.past_only.auc_mean)} for the flood record alone and
      ${fixed(pastTest.without_past.auc_mean)} for the model without it.` : ""}
      The figures above, which leave the record out, are the ones to compare
      with other studies.</p>` : ""}
    ${benchmark.graph_vs_best_baseline ? `<h3>Why this model</h3>
      <p>Over ${benchmark.graph_vs_best_baseline.n_seeds} block assignments the graph
      neural network the project began with trailed the best ordinary model by
      ${Math.abs(benchmark.graph_vs_best_baseline.auc_difference_mean).toFixed(3)} in AUC,
      so a gradient-boosted tree is the model in use.</p>` : ""}
    ${gap.mean ? `<h3>Why radar labels</h3>
      <p>Training on flood extents observed by radar beats terrain-threshold labels
      by ${gap.mean.toFixed(3)} in AUC, ahead on ${gap.seeds_observed_ahead} of
      ${gap.n_seeds} block assignments.</p>` : ""}
    ${DATA_SOURCES}`;
}

/* ── Views ────────────────────────────────────────────────────────────── */
const isPhone = () => window.innerWidth <= 860;

function framePadding() {
  // Frame the regions in the part of the map the open panels leave visible.
  const side = (name) => {
    const panel = $(name === "left" ? "panelLeft" : "panelRight");
    const open = !document.body.classList.contains(`${name}-collapsed`) && !isPhone();
    return open ? panel.offsetWidth + 50 : 40;
  };
  return { top: 50, bottom: 50, left: side("left"), right: side("right") };
}

async function selectView(id) {
  showLoading(`Loading ${id === ALL ? "all regions" : regionName(id)}…`);
  state.view = id;
  $("detail").hidden = true;
  document.querySelectorAll("#regionChips button").forEach((button) =>
    button.setAttribute("aria-pressed", String(button.dataset.region === id)));

  const summary = await getJSON(url(`api/region/${id}/summary`));
  state.summary = summary;
  // Region and area choices belong to the view; the rest carry over.
  state.filters.regions.clear();
  clearArea();

  $("regionLine").textContent = id === ALL
    ? `${summary.regions.length} flood regions`
    : `${summary.region.name} — ${summary.region.hazard}`;
  showCaution(id);
  renderLegend();
  $("rangeName").textContent = "Score";
  $("search").placeholder = "Asset name, type or district…";

  removeRegionLayers();
  addAreaLayers();
  assetRegions().forEach(addRegionLayers);
  if (summary.extent) {
    const [w, s, e, n] = summary.extent;
    map.fitBounds([[w, s], [e, n]], { padding: framePadding(), duration: 0 });
  }
  buildFilters();
  renderLayerToggles();
  renderProvenance();
  applyFilters();
  await applyLayers();
  if (state.panel) $("sheetBody").innerHTML = await panelBody(state.panel);
  hideLoading();
}

/* ── Legend ───────────────────────────────────────────────────────── */
function rampExpression(property, vmin, vmax) {
  const stops = OR_RD.flatMap((colour, i) => [vmin + (vmax - vmin) * (i / (OR_RD.length - 1)), colour]);
  return ["interpolate", ["linear"], ["coalesce", ["get", property], vmin], ...stops];
}

function renderLegend() {
  if (state.hazard === "landslide") {
    const display = state.ls.summary?.display || {};
    const vmin = display.vmin ?? 0;
    const vmax = display.vmax ?? 1;
    const dates = (state.ls.summary?.model?.inventory || {}).event_dates || [];
    $("legendTitle").textContent = "Landslide susceptibility";
    $("legend").innerHTML = `
      <div class="ramp" style="background:linear-gradient(to right, ${OR_RD.join(", ")})"></div>
      <div class="ramp-labels"><span>${fixed(vmin, 2)} lower</span><span>higher ${fixed(vmax, 2)}</span></div>
      <div style="margin-top:8px"><span class="point"></span>Mapped landslide${dates.length ? `<span>${esc(dates.join(", "))}</span>` : ""}</div>
      <div class="note">The surface and the area colours share this scale; each area is coloured by its mean.
        Districts show when zoomed out, then upazilas and unions. Click an area to open it.</div>`;
    return;
  }
  $("legendTitle").textContent = "Flood susceptibility";
  $("legend").innerHTML = CLASSES.map((c) =>
    `<div><i style="background:${c.colour}"></i>${c.label}<span>${c.value === "low" ? "&lt; 0.30"
      : c.value === "very_high" ? "≥ 0.70" : `${c.range[0].toFixed(2)} – ${c.range[1].toFixed(2)}`}</span></div>`).join("")
    + `<div class="note" style="margin-top:10px">Areas: share of their assets scoring High or Very high</div>
      <div class="ramp" style="background:linear-gradient(to right, ${SHARE_STOPS.map(([, c]) => c).join(", ")})"></div>
      <div class="ramp-labels">${SHARE_STOPS.map(([v]) => `<span>${Math.round(v * 100)}${v === 0.4 ? "+" : ""}</span>`).join("")}</div>
      <div><i style="background:transparent;border-radius:3px;box-shadow:inset 0 0 0 1.5px #94a3b8"></i>No assets mapped (outline only)</div>
      <div class="note">Divisions show when zoomed out, then districts, upazilas and unions as you zoom in. Click an area to open it.</div>`;
}

/* ── Landslide view ───────────────────────────────────────────────────── */
function addLandslideLayers(region) {
  const base = `pmtiles://${url(`tiles/${region}`)}`;
  map.addSource("ls-points", { type: "vector", url: `${base}/ls_points.pmtiles` });
  map.addLayer({
    id: "ls-points", type: "circle", source: "ls-points", "source-layer": "ls_points",
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 2, 10, 3.5, 14, 6],
      "circle-color": LANDSLIDE_POINT,
      "circle-stroke-color": "#ffffff",
      "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 8, 0.5, 13, 1.2],
      "circle-opacity": 0.9,
    },
  }, "basemap-roads");
}

async function selectLandslide(region) {
  showLoading(`Loading ${regionName(region)}…`);
  state.view = region;
  $("detail").hidden = true;
  document.querySelectorAll("#regionChips button").forEach((button) =>
    button.setAttribute("aria-pressed", String(button.dataset.region === region)));

  const summary = await getJSON(url(`api/landslide/${region}/summary`));
  state.ls.summary = summary;
  state.summary = null;
  const f = state.ls.filters;
  f.categories.clear();
  f.min = 0;
  f.max = 1;
  clearArea();

  $("regionLine").textContent = `${summary.region.name} — ${summary.region.hazard}`;
  showCaution(region);
  $("rangeName").textContent = "Mean susceptibility";
  $("search").placeholder = "Upazila or district…";
  renderLegend();

  removeRegionLayers();
  addAreaLayers();
  addLandslideLayers(region);
  if (summary.extent) {
    const [w, s, e, n] = summary.extent;
    map.fitBounds([[w, s], [e, n]], { padding: framePadding(), duration: 0 });
  }
  buildLandslideFilters();
  renderLayerToggles();
  renderProvenance();
  applyFilters();
  await applyLayers();
  if (state.panel) $("sheetBody").innerHTML = await panelBody(state.panel);
  hideLoading();
}

function buildLandslideFilters() {
  const summary = state.ls.summary;
  const f = state.ls.filters;
  $("filterSection").hidden = false;
  $("fRegion").hidden = true;
  $("fType").hidden = true;
  renderAreaPicker();
  // One survey of one storm has one category; the filter appears when an
  // inventory holds several.
  const categories = Object.entries(summary.categories || {});
  $("fClass").hidden = categories.length < 2;
  controls.class = multiSelect($("fClass"), {
    label: "Category", selected: f.categories,
    options: categories.map(([c, n]) => ({ value: c, label: c, count: n })),
    onChange: applyFilters,
  });
  syncScoreSlider();
}

// The property of a mapped landslide that names its area at each level.
const POINT_AREA = { district: "district", upazila: "unit_id", union: "union_id" };

function landslidePointFilter() {
  const f = state.ls.filters;
  const points = ["all",
    [">=", ["coalesce", ["get", "unit_susceptibility"], 0], f.min],
    ["<=", ["coalesce", ["get", "unit_susceptibility"], 0], f.max]];
  if (state.area) points.push(["==", ["get", POINT_AREA[state.area.level]], state.area.id]);
  if (f.categories.size) points.push(["in", ["get", "category"], ["literal", [...f.categories]]]);
  return points;
}

function applyLandslideFilters() {
  if (map.getLayer("ls-points")) map.setFilter("ls-points", landslidePointFilter());
  applyAreaFilters();
  $("resetFilters").hidden = !filtersActive();
  applyLayers();
  refreshCounters();
}

async function refreshLandslideCounters() {
  const f = state.ls.filters;
  const params = new URLSearchParams({
    area: state.area ? `${state.area.level}:${state.area.id}` : "",
    categories: [...f.categories].join(","), min_score: f.min, max_score: f.max,
  });
  const stats = await getJSON(url(`api/landslide/${state.view}/stats?${params}`));
  state.ls.stats = stats;
  const top = stats.highest;
  renderCounters([
    [fmt(stats.upazilas), "Upazilas shown"],
    [fixed(stats.mean_susceptibility), "Mean susceptibility"],
    [fmt(stats.population), "People living there"],
    [fmt(stats.landslides), "Landslides mapped"],
    [fixed(stats.susceptibility_at_landslides), "Susceptibility where they struck"],
    [top ? esc(top.name) : "—", top ? `Highest upazila, ${fixed(top.susceptibility_mean)}` : "Highest upazila"],
  ], filtersActive());
}

function landslideModelRows(model) {
  const inventory = model.inventory || {};
  const validation = model.validation || {};
  return statRows([
    ["Landslides mapped", fmt(inventory.n_landslides)],
    ["Background points", fmt(inventory.n_background)],
    ["Validation AUC", fixed(validation.val_auc_roc), "held-out 10 km blocks"],
    ["Average precision", fixed(validation.val_average_precision)],
    ["Event dates", esc((inventory.event_dates || []).join(", ") || "—"),
      inventory.single_event ? "one rainfall episode, so this maps that storm" : ""],
  ]);
}

function landslideRankings() {
  const units = [...(state.ls.summary?.units || [])]
    .sort((a, b) => (b.susceptibility_mean || 0) - (a.susceptibility_mean || 0));
  const table = units.map((u, i) => `
    <tr><td class="num">${i + 1}</td><td>${esc(u.name)}</td><td>${esc(u.district)}</td>
    <td class="num">${fixed(u.susceptibility_mean)}</td><td class="num">${fixed(u.susceptibility_max)}</td>
    <td class="num">${fmt(u.population)}</td><td class="num">${fmt(u.n_landslides)}</td></tr>`).join("");
  return `
    <h3>Upazilas by mean landslide susceptibility</h3>
    <table><thead><tr><th class="num">#</th><th>Upazila</th><th>District</th>
      <th class="num">Mean</th><th class="num">Max</th><th class="num">People</th>
      <th class="num">Landslides mapped</th></tr></thead><tbody>${table}</tbody></table>
    <p style="color:var(--dim);font-size:11.5px">The mapped landslides all come from one storm, on
      6 August 2023, so an upazila with none may simply not have been struck by it.</p>
    <h3>Download</h3>
    <p><a href="${url(`api/landslide/${state.view}/upazilas.csv`)}">Download upazila figures (CSV)</a></p>`;
}

function searchUpazilas(query) {
  const q = query.toLowerCase();
  const found = (state.ls.summary?.units || []).filter((u) =>
    u.name.toLowerCase().includes(q) || (u.district || "").toLowerCase().includes(q)).slice(0, 12);
  const list = $("searchResults");
  list.innerHTML = found.map((u, i) => `
    <li data-index="${i}">${esc(u.name)}
      <span class="type">${esc(u.district)} · mean ${fixed(u.susceptibility_mean)} · ${fmt(u.n_landslides)} landslides</span></li>`).join("")
    || "<li>No upazila matches</li>";
  list.hidden = false;
  list.querySelectorAll("li[data-index]").forEach((item) =>
    item.addEventListener("click", () => {
      pickArea("upazila", found[Number(item.dataset.index)].unit_id);
      list.hidden = true;
      if (isPhone()) setSide("left", false);
    }));
}

function showCard(html) {
  $("detail").innerHTML = `<button class="close" id="detailClose" aria-label="Close">×</button>${html}`;
  $("detail").hidden = false;
  $("detailClose").addEventListener("click", () => ($("detail").hidden = true));
}

function landslideClick(event) {
  const layers = ["ls-points"].filter((id) => map.getLayer(id)
    && map.getLayoutProperty(id, "visibility") !== "none");
  const [feature] = layers.length ? map.queryRenderedFeatures(event.point, { layers }) : [];
  if (!feature) { areaClick(event); return; }
  const p = feature.properties;
  showCard(`
    <span class="tag">Mapped landslide</span><span class="tag">${esc(p.category || "")}</span>
    <h3>${esc(p.upazila || "Hill Tracts")}</h3>
    <div class="meta">${esc(p.district || "")} · ${esc(p.event_date || "")}</div>
    <div class="probability" style="color:#b30000">${fixed(p.susceptibility)}</div>
    <div class="probability-note">modelled susceptibility of the 30 m cell where this landslide struck,
      against ${fixed(p.unit_susceptibility)} for the upazila on average</div>`);
}

/* ── Search ───────────────────────────────────────────────────────────── */
let searchTimer = null;
$("search").addEventListener("input", (event) => {
  clearTimeout(searchTimer);
  const query = event.target.value.trim();
  if (query.length < 2) { $("searchResults").hidden = true; return; }
  if (state.hazard === "landslide") { searchUpazilas(query); return; }
  searchTimer = setTimeout(async () => {
    const found = await getJSON(
      url(`api/region/${state.view}/assets?q=${encodeURIComponent(query)}&limit=12`));
    const list = $("searchResults");
    list.innerHTML = found.assets.map((a, i) => `
      <li data-index="${i}">${esc(a.name)}
        <span class="type">${esc(typeLabel(a.asset_type))} · ${esc(a.division || "")}${state.view === ALL
          ? ` · ${esc(REGION_SHORT[a.region] || a.region)}` : ""} · ${fixed(a.flood_risk)}</span></li>`).join("")
      || "<li>No assets match</li>";
    list.hidden = false;
    list.querySelectorAll("li[data-index]").forEach((item) =>
      item.addEventListener("click", () => {
        const asset = found.assets[Number(item.dataset.index)];
        map.flyTo({ center: [asset.lon, asset.lat], zoom: 13 });
        showDetail(asset);
        list.hidden = true;
        if (isPhone()) setSide("left", false);
      }));
  }, 180);
});

/* ── Wiring ───────────────────────────────────────────────────────────── */
/* Each side panel folds away on its own; on a phone, where either covers the
 * map, opening one folds the other. */
function setSide(side, open) {
  document.body.classList.toggle(`${side}-collapsed`, !open);
  document.querySelector(`[data-open="${side}"]`).hidden = open;
  if (open && isPhone()) {
    const other = side === "left" ? "right" : "left";
    document.body.classList.add(`${other}-collapsed`);
    document.querySelector(`[data-open="${other}"]`).hidden = false;
  }
}
document.querySelectorAll("[data-close]").forEach((button) =>
  button.addEventListener("click", () => setSide(button.dataset.close, false)));
document.querySelectorAll("[data-open]").forEach((button) =>
  button.addEventListener("click", () => setSide(button.dataset.open, true)));
$("resetFilters").addEventListener("click", resetFilters);
document.querySelectorAll("#basemaps button").forEach((button) =>
  button.addEventListener("click", () => setBasemap(button.dataset.basemap)));
document.querySelectorAll("#panelButtons button").forEach((button) =>
  button.addEventListener("click", () => openPanel(button.dataset.panel)));
$("sheetClose").addEventListener("click", closePanel);
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (state.panel) closePanel();
  document.querySelectorAll(".ms.open").forEach((open) => open.classList.remove("open"));
});

// One handler for every region's asset layer: the layer says which region
// the clicked asset belongs to, and its rank finds the full row.
map.on("click", async (event) => {
  if (state.hazard === "landslide") { landslideClick(event); return; }
  const layers = assetLayerIds().filter((id) => map.getLayoutProperty(id, "visibility") !== "none");
  const [feature] = layers.length ? map.queryRenderedFeatures(event.point, { layers }) : [];
  if (!feature) { areaClick(event); return; }
  const region = feature.layer.id.slice("assets-".length);
  const rank = feature.properties.risk_rank;
  if (rank == null) return;
  showDetail(await getJSON(url(`api/region/${region}/rank/${rank}`)));
});
map.on("mousemove", (event) => {
  const layers = (state.hazard === "landslide" ? ["ls-points"] : assetLayerIds())
    .filter((id) => map.getLayer(id) && map.getLayoutProperty(id, "visibility") !== "none");
  const hit = layers.length && map.queryRenderedFeatures(event.point, { layers }).length;
  // over a marker the marker is what a click opens, so the area tip steps aside
  const area = hit ? (areaTip.remove(), false) : areaHover(event);
  map.getCanvas().style.cursor = hit || area ? "pointer" : "";
});
map.getCanvas().addEventListener("mouseleave", () => areaTip.remove());
map.on("movestart", () => areaTip.remove());

/* ── Hazard switch ────────────────────────────────────────────────────
 * Flood and landslide are separate views: each has its own regions,
 * filters, results, legend and model figures, so the two never mix. */
const floodRegions = () => state.regions.filter((r) => r.has_assets);
const landslideRegions = () => state.regions.filter((r) => r.has_landslide);

function renderChips() {
  const chips = state.hazard === "landslide"
    ? landslideRegions()
    : [{ id: ALL, name: "All" }, ...floodRegions()];
  $("regionChips").innerHTML = chips.map((region) =>
    `<button data-region="${esc(region.id)}" aria-pressed="${region.id === state.view}">${esc(region.id === ALL ? "All" : REGION_SHORT[region.id] || region.name)}</button>`).join("");
  document.querySelectorAll("#regionChips button").forEach((button) =>
    button.addEventListener("click", () => (state.hazard === "landslide"
      ? selectLandslide(button.dataset.region) : selectView(button.dataset.region))));
}

async function setHazard(hazard) {
  state.hazard = hazard;
  clearArea();
  areaTip.remove();
  document.querySelectorAll("#hazards button").forEach((button) =>
    button.setAttribute("aria-checked", String(button.dataset.hazard === hazard)));
  closePanel();
  $("search").value = "";
  $("searchResults").hidden = true;
  if (hazard === "landslide") {
    state.view = landslideRegions()[0]?.id;
    renderChips();
    if (state.view) await selectLandslide(state.view);
  } else {
    state.view = ALL;
    renderChips();
    await selectView(ALL);
  }
}

(async function start() {
  if (isPhone()) { setSide("left", false); setSide("right", false); }
  showLoading("Loading…");
  const { regions } = await getJSON(url("api/regions"));
  state.regions = regions;
  document.querySelectorAll("#hazards button").forEach((button) =>
    button.addEventListener("click", () => setHazard(button.dataset.hazard)));
  // with no landslide region the switch would lead nowhere
  if (!landslideRegions().length) $("hazards").closest("section").hidden = true;

  await mapReady;
  setBasemap(state.basemap);
  await setHazard("flood");
})();

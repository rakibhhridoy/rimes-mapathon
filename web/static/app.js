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
  rangpur_rajshahi: "Rangpur", sylhet: "Sylhet", sw_coastal: "Coast", cht: "Hill Tracts",
};

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
  view: ALL,
  summary: null,
  basemap: "topo",
  panel: null,
  layers: { assets: true, unions: true, hotspots: true, landslide: true },
  filters: { regions: new Set(), districts: new Set(), types: new Set(), classes: new Set(), min: 0, max: 1 },
};

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
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
map.addControl(new maplibregl.ScaleControl({ maxWidth: 110, unit: "metric" }), "bottom-left");

// Captured at creation: the region list can arrive after the map has loaded,
// and a listener attached then would wait for an event that already fired.
const mapReady = new Promise((resolve) => map.once("load", resolve));

// The credits are written out in the left panel, so on the map they stay folded
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
const viewRegions = () => (state.view === ALL ? state.regions.map((r) => r.id) : [state.view]);
const regionInfo = (id) => (state.summary?.regions || []).find((r) => r.id === id) || {};
const assetRegions = () => viewRegions().filter((id) => regionInfo(id).has_assets);
const regionShown = (id) => state.filters.regions.size === 0 || state.filters.regions.has(id);
const assetLayerIds = () => assetRegions().map((r) => `assets-${r}`).filter((id) => map.getLayer(id));

/* ── Region layers ────────────────────────────────────────────────────── */
function removeRegionLayers() {
  const prefixes = ["assets-", "unions-", "hotspots-", "overlay-"];
  (map.getStyle().layers || []).forEach((layer) => {
    if (prefixes.some((p) => layer.id.startsWith(p))) map.removeLayer(layer.id);
  });
  Object.keys(map.getStyle().sources || {}).forEach((id) => {
    if (prefixes.some((p) => id.startsWith(p))) map.removeSource(id);
  });
}

function addRegionLayers(region) {
  const base = `pmtiles://${url(`tiles/${region}`)}`;
  // Area layers sit under the hybrid's labels; the assets go above them.
  map.addSource(`unions-${region}`, { type: "vector", url: `${base}/unions.pmtiles` });
  map.addLayer({
    id: `unions-fill-${region}`, type: "fill", source: `unions-${region}`, "source-layer": "unions",
    paint: {
      "fill-color": ["interpolate", ["linear"], ["coalesce", ["get", "mean_risk"], 0],
        0, "#cde2fb", 0.2, "#9ec5f4", 0.35, "#6da7ec", 0.5, "#2a78d6"],
      "fill-opacity": 0.18,
    },
  }, "basemap-roads");
  map.addLayer({
    id: `unions-line-${region}`, type: "line", source: `unions-${region}`, "source-layer": "unions",
    paint: {
      "line-color": "#ffffff",
      "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.4, 12, 1.2],
      "line-opacity": 0.55,
    },
  }, "basemap-roads");

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
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 1.6, 9, 2.6, 11, 4, 14, 7],
      "circle-color": ["step", ["coalesce", ["get", "flood_risk"], 0],
        "#15803d", 0.30, "#a16207", 0.50, "#b45309", 0.70, "#c62828"],
      "circle-stroke-color": "#ffffff",
      // no outline at regional zoom, where outlines merge into white streaks
      "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 9, 0, 11, 0.8, 14, 1.2],
      "circle-stroke-opacity": 0.9,
      "circle-opacity": 0.95,
    },
  }, "basemap-roads");        // under the hybrid's labels, which stay readable
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
  const below = (map.getStyle().layers || []).find((l) => l.id.startsWith("unions-fill-"))?.id || "basemap-roads";
  map.addLayer({
    id, type: "raster", source: id,
    paint: { "raster-opacity": name === "landslide" ? 0.68 : 0.55 },
  }, below);
  return true;
}

async function applyLayers() {
  const show = (on) => (on ? "visible" : "none");
  for (const region of viewRegions()) {
    const shown = regionShown(region);
    for (const [key, ids] of Object.entries({
      assets: [`assets-${region}`],
      unions: [`unions-fill-${region}`, `unions-line-${region}`],
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
  if (f.districts.size) conditions.push(["in", ["get", "division"], ["literal", [...f.districts]]]);
  if (f.classes.size) {
    conditions.push(["any", ...CLASSES.filter((c) => f.classes.has(c.value)).map((c) =>
      ["all", [">=", score, c.range[0]], ["<", score, c.range[1]]])]);
  }
  return conditions;
}

const filtersActive = () => {
  const f = state.filters;
  return f.regions.size + f.districts.size + f.types.size + f.classes.size > 0 || f.min > 0 || f.max < 1;
};

function applyFilters() {
  const expression = filterExpression();
  assetLayerIds().forEach((id) => map.setFilter(id, expression));
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

function districtOptions() {
  const byDistrict = state.summary?.counts?.by_district || [];
  const regions = state.filters.regions;
  return byDistrict
    .filter((d) => d.district && (regions.size === 0 || regions.has(d.region)))
    .map((d) => ({
      value: d.district, label: d.district, count: d.n,
      group: state.view === ALL ? regionName(d.region) : undefined,
    }));
}

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
    onChange: () => { controls.district.setOptions(districtOptions()); applyFilters(); },
  });
  $("fRegion").hidden = state.view !== ALL;

  controls.district = multiSelect($("fDistrict"), {
    label: "District", selected: f.districts, searchable: true,
    options: districtOptions(), onChange: applyFilters,
  });
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
  const f = state.filters;
  $("scoreMin").value = f.min;
  $("scoreMax").value = f.max;
  $("scoreFill").style.left = `${f.min * 100}%`;
  $("scoreFill").style.right = `${(1 - f.max) * 100}%`;
  $("scoreOut").textContent = `${f.min.toFixed(2)} – ${f.max.toFixed(2)}`;
}

const scoreChanged = debounce(applyFilters, 120);
["scoreMin", "scoreMax"].forEach((id) => $(id).addEventListener("input", (event) => {
  const f = state.filters;
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
  const f = state.filters;
  [f.regions, f.districts, f.types, f.classes].forEach((set) => set.clear());
  f.min = 0;
  f.max = 1;
  if (state.summary) buildFilters();
  applyFilters();
}

/* ── Counters ─────────────────────────────────────────────────────────── */
const refreshCounters = debounce(async () => {
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
    regions: regions.join(","), types: [...f.types].join(","), districts: [...f.districts].join(","),
    classes: [...f.classes].join(","), min_score: f.min, max_score: f.max,
  });
  const stats = regions.length
    ? await getJSON(url(`api/stats?${params}`))
    : { assets: 0, high_risk: 0, mean_risk: null, by_type: {} };
  const t = stats.by_type || {};
  const rows = [
    [fmt(stats.assets), "Assets shown"],
    [fmt(stats.high_risk), "High susceptibility"],
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
  const available = new Set();
  viewRegions().forEach((id) => (regionInfo(id).overlays || []).forEach((name) => available.add(name)));
  const hasAssets = Boolean(state.summary?.region?.has_assets);
  const entries = [
    ...(hasAssets ? [["assets", "Asset markers"], ["unions", "Union boundaries"], ["hotspots", "Hotspots (Gi*, 95%)"]] : []),
    ...Object.keys(OVERLAYS).filter((name) => available.has(name)).map((name) => [name, OVERLAYS[name]]),
  ];
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
  if (name === "rankings") return rankingsPanel();
  if (name === "preparedness") return preparednessPanel();
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
  // Region and district choices belong to the view; the rest carry over.
  state.filters.regions.clear();
  state.filters.districts.clear();

  $("regionLine").textContent = id === ALL
    ? `${summary.regions.length} regions · flood and landslide`
    : `${summary.region.name} — ${summary.region.hazard}`;

  removeRegionLayers();
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

/* ── Search ───────────────────────────────────────────────────────────── */
let searchTimer = null;
$("search").addEventListener("input", (event) => {
  clearTimeout(searchTimer);
  const query = event.target.value.trim();
  if (query.length < 2) { $("searchResults").hidden = true; return; }
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
  const layers = assetLayerIds();
  if (!layers.length) return;
  const [feature] = map.queryRenderedFeatures(event.point, { layers });
  if (!feature) return;
  const region = feature.layer.id.slice("assets-".length);
  const rank = feature.properties.risk_rank;
  if (rank == null) return;
  showDetail(await getJSON(url(`api/region/${region}/rank/${rank}`)));
});
map.on("mousemove", (event) => {
  const layers = assetLayerIds();
  const hit = layers.length && map.queryRenderedFeatures(event.point, { layers }).length;
  map.getCanvas().style.cursor = hit ? "pointer" : "";
});

(async function start() {
  if (isPhone()) { setSide("left", false); setSide("right", false); }
  showLoading("Loading…");
  const { regions } = await getJSON(url("api/regions"));
  state.regions = regions;
  const chips = [{ id: ALL, name: "All" }, ...regions];
  $("regionChips").innerHTML = chips.map((region) =>
    `<button data-region="${esc(region.id)}" aria-pressed="false">${esc(region.id === ALL ? "All" : REGION_SHORT[region.id] || region.name)}</button>`).join("");
  document.querySelectorAll("#regionChips button").forEach((button) =>
    button.addEventListener("click", () => selectView(button.dataset.region)));

  await mapReady;
  setBasemap(state.basemap);
  await selectView(ALL);
})();

/* Fermium Hazard Mapper — front end.
 *
 * The map draws vector tiles the build script wrote, so panning and zooming
 * never ask the server for geometry again. Numbers come from small JSON
 * responses that the browser caches. Nothing here computes a risk figure:
 * every value shown is one the pipeline produced.
 */

/* The app may be served at / or under a path such as /hazmapper, so every
 * URL is built from where the page itself was loaded. */
const BASE = document.baseURI.replace(/[^/]*$/, "");
const url = (path) => BASE + path.replace(/^\//, "");

const RISK_COLOURS = [
  ["#15803d", 0.30],      // low
  ["#a16207", 0.50],      // moderate
  ["#b45309", 0.70],      // high
  ["#c62828", 1.01],      // very high
];

const state = {
  regions: [],
  region: null,
  summary: null,
  layers: { assets: true, unions: true, hotspots: true, landslide: true },
  minScore: 0,
  types: new Set(),
  panel: null,
  selected: null,
  availableOverlays: new Set(),
};

const $ = (id) => document.getElementById(id);
const fmt = (n) => (n === null || n === undefined ? "—" : n.toLocaleString());
const fixed = (n, d = 3) => (n === null || n === undefined ? "—" : Number(n).toFixed(d));

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

function showLoading(label) {
  $("loadingLabel").textContent = label;
  $("loading").hidden = false;
}
const hideLoading = () => ($("loading").hidden = true);

/* ── Map ──────────────────────────────────────────────────────────────── */
const protocol = new pmtiles.Protocol();
maplibregl.addProtocol("pmtiles", protocol.tile);

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    glyphs: "https://basemaps.cartocdn.com/gl/positron-gl-style/{fontstack}/{range}.pbf",
    sources: {
      basemap: {
        type: "raster",
        tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
        tileSize: 256,
        attribution: "Tiles &copy; Esri &middot; Map data &copy; OpenStreetMap contributors",
      },
    },
    layers: [{ id: "basemap", type: "raster", source: "basemap" }],
  },
  center: [89.2, 25.4],
  zoom: 7,
  attributionControl: { compact: true },
});
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
map.addControl(new maplibregl.ScaleControl({ maxWidth: 110, unit: "metric" }), "bottom-right");

/* ── Region layers ────────────────────────────────────────────────────── */
function riskExpression() {
  // Colour by the asset's own score, in the four classes the legend names.
  const stops = [];
  RISK_COLOURS.forEach(([colour, upper], index) => {
    if (index < RISK_COLOURS.length - 1) stops.push(upper, colour);
  });
  return ["step", ["coalesce", ["get", "flood_risk"], 0], RISK_COLOURS[0][0], ...stops.slice(0, -1).reverse().reverse()];
}

const OVERLAYS = {
  flood_risk: "Kriged hazard surface",
  landslide: "Landslide susceptibility",
  hand: "Height above drainage",
  slope: "Slope",
  kriging_variance: "Kriging variance",
};

async function addOverlays(region) {
  // Raster layers the pipeline pre-rendered. They are PNGs with bounds, so
  // the browser draws them without asking the server for anything else.
  const region_info = state.summary?.region || {};
  for (const [name, label] of Object.entries(OVERLAYS)) {
    // Do not ask for a layer this region cannot have: it would be a 404 in
    // the console on every load.
    if (name === "landslide" && !region_info.has_landslide) continue;
    if (name !== "landslide" && !region_info.has_assets && name !== "hand" && name !== "slope") continue;
    const id = `overlay-${name}`;
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
    let meta;
    try {
      meta = await getJSON(url(`overlays/${region}/${name}.json`));
    } catch {
      continue;                       // this region has no such layer
    }
    const [[south, west], [north, east]] = meta.bounds;
    map.addSource(id, {
      type: "image",
      url: url(`overlays/${region}/${name}.png`),
      coordinates: [[west, north], [east, north], [east, south], [west, south]],
    });
    map.addLayer({
      id, type: "raster", source: id,
      paint: { "raster-opacity": name === "landslide" ? 0.68 : 0.55 },
      layout: { visibility: state.layers[name] ? "visible" : "none" },
    }, map.getLayer("unions-fill") ? "unions-fill" : undefined);
    state.availableOverlays.add(name);
  }
}

function removeRegionLayers() {
  Object.keys(OVERLAYS).forEach((name) => {
    const id = `overlay-${name}`;
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
  });
  state.availableOverlays.clear();
  ["assets-circle", "unions-fill", "unions-line", "hotspots-fill", "heatmap"].forEach((id) => {
    if (map.getLayer(id)) map.removeLayer(id);
  });
  ["assets", "unions", "hotspots", "heat"].forEach((id) => {
    if (map.getSource(id)) map.removeSource(id);
  });
}

function addRegionLayers(region) {
  removeRegionLayers();
  const base = `pmtiles://${url(`tiles/${region}`)}`;

  map.addSource("unions", { type: "vector", url: `${base}/unions.pmtiles` });
  map.addLayer({
    id: "unions-fill", type: "fill", source: "unions", "source-layer": "unions",
    paint: {
      "fill-color": ["interpolate", ["linear"], ["coalesce", ["get", "mean_risk"], 0],
        0, "#cde2fb", 0.2, "#9ec5f4", 0.35, "#6da7ec", 0.5, "#2a78d6"],
      "fill-opacity": 0.45,
    },
  });
  map.addLayer({
    id: "unions-line", type: "line", source: "unions", "source-layer": "unions",
    paint: { "line-color": "#8f8d86", "line-width": 0.5, "line-opacity": 0.8 },
  });

  map.addSource("hotspots", { type: "vector", url: `${base}/hotspots.pmtiles` });
  map.addLayer({
    id: "hotspots-fill", type: "fill", source: "hotspots", "source-layer": "hotspots",
    paint: { "fill-color": "#c62828", "fill-opacity": 0.16 },
  });

  map.addSource("assets", { type: "vector", url: `${base}/assets.pmtiles` });
  map.addLayer({
    id: "assets-circle", type: "circle", source: "assets", "source-layer": "assets",
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 2.2, 10, 4, 14, 7],
      "circle-color": ["step", ["coalesce", ["get", "flood_risk"], 0],
        "#15803d", 0.30, "#a16207", 0.50, "#b45309", 0.70, "#c62828"],
      "circle-stroke-color": "#ffffff",
      "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 8, 0.3, 13, 1],
      "circle-opacity": 0.9,
    },
  });
  applyFilters();
}

function applyFilters() {
  Object.keys(OVERLAYS).forEach((name) => {
    const id = `overlay-${name}`;
    if (map.getLayer(id)) {
      map.setLayoutProperty(id, "visibility", state.layers[name] ? "visible" : "none");
    }
  });
  if (!map.getLayer("assets-circle")) return;
  const conditions = [">=", ["coalesce", ["get", "flood_risk"], 0], state.minScore];
  const filter = state.types.size
    ? ["all", conditions, ["in", ["get", "asset_type"], ["literal", [...state.types]]]]
    : conditions;
  map.setFilter("assets-circle", filter);
  ["assets-circle"].forEach((id) =>
    map.setLayoutProperty(id, "visibility", state.layers.assets ? "visible" : "none"));
  ["unions-fill", "unions-line"].forEach((id) => {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", state.layers.unions ? "visible" : "none");
  });
  if (map.getLayer("hotspots-fill")) {
    map.setLayoutProperty("hotspots-fill", "visibility", state.layers.hotspots ? "visible" : "none");
  }
}

/* ── Sidebar panels ───────────────────────────────────────────────────── */
function renderProvenance(summary) {
  // A landslide region has no scored assets, so it reports its own model
  // rather than rows of dashes where the flood figures would be.
  if (!summary.region.has_assets && summary.metrics.landslide_model) {
    const model = summary.metrics.landslide_model;
    const inventory = model.inventory || {};
    const validation = model.validation || {};
    const rows = [
      ["Landslides mapped", fmt(inventory.n_landslides)],
      ["Background points", fmt(inventory.n_background)],
      ["Validation AUC", fixed(validation.val_auc_roc), "held-out 10 km blocks"],
      ["Average precision", fixed(validation.val_average_precision)],
      ["Event dates", (inventory.event_dates || []).join(", ") || "—",
        inventory.single_event ? "one rainfall episode, so this maps that storm" : ""],
    ];
    $("provenance").innerHTML = rows.map(([label, value, note]) => `
      <div class="row"><span class="label">${label}</span><span class="value">${value}</span></div>
      ${note ? `<div class="note">${note}</div>` : ""}`).join("");
    $("observed").hidden = true;
    return;
  }
  const meta = summary.metrics.pipeline_metadata || {};
  const validation = meta.gnn_validation || {};
  const rows = [
    ["Processed", (meta.generated_at || "").replace("T", " ").slice(0, 16) || "—"],
    ["Model", validation.model_type || "—"],
    ["Assets", fmt(summary.counts.assets)],
    ["Flood-prone labels", meta.label_positive_rate != null ? `${(meta.label_positive_rate * 100).toFixed(1)} %` : "—",
      "share of assets on ground labelled flood-prone"],
    ["Validation AUC", fixed(validation.val_auc_roc), "held-out 10 km blocks"],
    ["Brier score", fixed(validation.val_brier), "lower is better"],
  ];
  $("provenance").innerHTML = rows.map(([label, value, note]) => `
    <div class="row"><span class="label">${label}</span><span class="value">${value}</span></div>
    ${note ? `<div class="note">${note}</div>` : ""}`).join("");

  const observed = ((summary.metrics.validation || {}).assets_vs_observed || {}).held_out_blocks;
  const box = $("observed");
  if (observed && observed.auc_roc != null) {
    box.hidden = false;
    box.innerHTML = `<div class="head">VS OBSERVED FLOODS</div>
      <div class="row"><span class="label">AUC</span><span class="value">${fixed(observed.auc_roc)}</span></div>
      <div class="row"><span class="label">Precision lift</span><span class="value">${fixed(observed.ap_lift, 1)}×</span></div>`;
  } else {
    box.hidden = true;
  }
}

function renderCounters(summary) {
  if (!summary.region.has_assets) {
    const model = summary.metrics.landslide_model || {};
    const inventory = model.inventory || {};
    const validation = model.validation || {};
    $("counters").innerHTML = [
      [fmt(inventory.n_landslides), "Landslides mapped"],
      [fmt(inventory.n_background), "Background points"],
      [fixed(validation.val_auc_roc), "Validation AUC"],
      [fixed(validation.val_average_precision), "Average precision"],
    ].map(([value, label]) =>
      `<div class="row"><span class="value">${value}</span><span class="label">${label}</span></div>`).join("");
    return;
  }
  const byType = summary.counts.by_type || {};
  const rows = [
    [fmt(summary.counts.assets), "Assets shown"],
    [fmt(summary.counts.high_risk), "High susceptibility"],
    [fixed(summary.counts.mean_risk), "Mean score"],
    [fmt(byType.hospital || 0), "Hospitals &amp; clinics"],
    [fmt(byType.school || 0), "Schools"],
    [fmt(byType.bridge || 0), "Bridges"],
    [fmt(byType.flood_shelter || 0), "Flood shelters"],
    [fmt(byType.cropland || 0), "Cropland parcels"],
  ];
  $("counters").innerHTML = rows.map(([value, label]) =>
    `<div class="row"><span class="value">${value}</span><span class="label">${label}</span></div>`).join("");
}

function renderLayerToggles() {
  const entries = [
    ["assets", "Asset markers"],
    ["unions", "Union boundaries"],
    ["hotspots", "Hotspots (Gi*, 95%)"],
    ...[...state.availableOverlays].map((name) => [name, OVERLAYS[name]]),
  ].filter(([key]) => key !== "assets" || state.summary?.region?.has_assets);
  $("layers").innerHTML = entries.map(([key, label]) =>
    `<label><input type="checkbox" data-layer="${key}" ${state.layers[key] ? "checked" : ""}>${label}</label>`).join("");
  $("layers").querySelectorAll("input").forEach((input) =>
    input.addEventListener("change", () => {
      state.layers[input.dataset.layer] = input.checked;
      applyFilters();
    }));
}

function renderTypeFilter(summary) {
  const types = Object.keys(summary.counts.by_type || {}).sort();
  $("typeFilter").innerHTML = types.map((type) =>
    `<label><input type="checkbox" data-type="${type}">${type}</label>`).join("");
  $("typeFilter").querySelectorAll("input").forEach((input) =>
    input.addEventListener("change", () => {
      input.checked ? state.types.add(input.dataset.type) : state.types.delete(input.dataset.type);
      applyFilters();
    }));
}

/* ── Asset detail ─────────────────────────────────────────────────────── */
function riskColour(score) {
  for (const [colour, upper] of RISK_COLOURS) if (score < upper) return colour;
  return RISK_COLOURS[RISK_COLOURS.length - 1][0];
}

function showDetail(asset) {
  state.selected = asset;
  const probability = asset.flood_probability;
  const factors = [
    ["Hazard", asset.cell_hazard, "#c62828"],
    ["Exposure", asset.cell_exposure, "#b45309"],
    ["Vulnerability", asset.cell_vulnerability, "#6d28d9"],
    ["Cell risk", asset.cell_composite_risk, "#2a78d6"],
  ];
  $("detail").innerHTML = `
    <button class="close" id="detailClose" aria-label="Close">×</button>
    <span class="tag">${asset.asset_type || "asset"}</span>
    <h3>${asset.name || "unnamed"}</h3>
    <div class="meta">${asset.division || ""} · ${Number(asset.lat).toFixed(4)}, ${Number(asset.lon).toFixed(4)} · Rank #${fmt(asset.risk_rank)}</div>
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

/* ── Panels ───────────────────────────────────────────────────────────── */
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
  return aboutPanel();
}

async function rankingsPanel() {
  if (!state.summary.region.has_assets) return landslidePanel();
  const [assets, admin] = await Promise.all([
    getJSON(url(`api/region/${state.region}/assets?limit=200`)),
    getJSON(url(`api/region/${state.region}/admin?level=union&limit=24`)),
  ]);
  const table = assets.assets.slice(0, 50).map((a) => `
    <tr><td class="num">${fmt(a.risk_rank)}</td><td>${a.asset_type}</td><td>${a.name}</td>
    <td class="num">${fixed(a.flood_risk, 4)}</td><td class="num">${fixed(a.flood_probability, 3)}</td>
    <td>${a.division || ""}</td></tr>`).join("");
  const cards = admin.units.map((u) => `
    <div class="card"><span class="tag">${u.mean_risk >= 0.3 ? "moderate" : "low"}</span>
      <div style="font-weight:600;margin:6px 0 2px">${u.name}</div>
      <div style="color:var(--dim);font-size:11px">${u.parent || ""}</div>
      <div class="score" style="color:${riskColour(u.mean_risk || 0)}">${fixed(u.mean_risk)}</div>
      <div style="color:var(--muted);font-size:11px">${fmt(u.n_assets)} assets</div></div>`).join("");
  return `
    <h3>Highest-scoring assets</h3>
    <table><thead><tr><th class="num">Rank</th><th>Type</th><th>Name</th>
      <th class="num">Score</th><th class="num">P(flood)</th><th>Division</th></tr></thead>
      <tbody>${table}</tbody></table>
    <h3>Union summaries</h3><div class="cards">${cards}</div>
    <h3>Download</h3>
    <p>Outputs may be reused with attribution, subject to the source licences.
      <a href="${url(`api/region/${state.region}/export.csv`)}">Download ranked assets (CSV)</a></p>`;
}

function landslidePanel() {
  const model = state.summary.metrics.landslide_model || {};
  const inventory = model.inventory || {};
  const validation = model.validation || {};
  const coefficients = model.standardised_coefficients || {};
  return `
    <h3>Landslide susceptibility model</h3>
    <p>A class-weighted logistic regression fitted to ${fmt(inventory.n_landslides)}
      landslide locations from ${inventory.source || "the inventory"}, against
      ${fmt(inventory.n_background)} background points drawn from
      ${inventory.background_domain || "the mapped area"}. On held-out 10 km blocks
      it reaches an AUC of ${fixed(validation.val_auc_roc)} and an average precision
      of ${fixed(validation.val_average_precision)}.</p>
    ${inventory.single_event ? `<p>Every point comes from one rainfall episode
      (${(inventory.event_dates || []).join(", ")}), so the surface describes where
      that storm triggered failures. It is a guide to susceptibility, not a
      complete record of where landslides can happen.</p>` : ""}
    <h3>Standardised coefficients</h3>
    <table><thead><tr><th>Predictor</th><th class="num">Coefficient</th></tr></thead><tbody>
      ${Object.entries(coefficients).map(([name, value]) =>
        `<tr><td>${name}</td><td class="num">${fixed(value)}</td></tr>`).join("")}
    </tbody></table>
    <p style="color:var(--dim);font-size:11px">${inventory.citation || ""}</p>`;
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

async function aboutPanel() {
  const summary = state.summary;
  const meta = summary.metrics.pipeline_metadata || {};
  const validation = meta.gnn_validation || {};
  const observed = ((summary.metrics.validation || {}).assets_vs_observed || {}).held_out_blocks || {};
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
      The scores therefore anticipate the next flood from the whole record, and
      the figures above, which leave the record out, are the ones to compare
      with other studies.</p>` : ""}
    ${benchmark.graph_vs_best_baseline ? `<h3>Why this model</h3>
      <p>Over ${benchmark.graph_vs_best_baseline.n_seeds} block assignments the graph
      neural network the project began with trailed the gradient-boosted tree by
      ${Math.abs(benchmark.graph_vs_best_baseline.auc_difference_mean).toFixed(3)} in AUC,
      so the tree is the model in use. The architecture is not what makes the
      system work; the features and the training labels are.</p>` : ""}
    ${gap.mean ? `<h3>Why radar labels</h3>
      <p>Training on flood extents observed by radar beats terrain-threshold labels
      by ${gap.mean.toFixed(3)} in AUC, ahead on ${gap.seeds_observed_ahead} of
      ${gap.n_seeds} block assignments.</p>` : ""}
    <h3>Data sources</h3>
    <p>Infrastructure from OpenStreetMap (ODbL). Elevation from NASA SRTM.
      Flood extents from Copernicus Sentinel-1 via Google Earth Engine.
      Surface water from the EC Joint Research Centre. Population from WorldPop
      (CC BY 4.0). Boundaries from geoBoundaries (CC BY 4.0). Landslide
      locations from NASA COOLR.</p>
    <p>Code: <a href="https://github.com/rakibhhridoy/rimes-mapathon" rel="noopener">github.com/rakibhhridoy/rimes-mapathon</a></p>`;
}

/* ── Region switching ─────────────────────────────────────────────────── */
async function selectRegion(id) {
  showLoading(`Loading ${state.regions.find((r) => r.id === id)?.name || id}…`);
  state.region = id;
  state.selected = null;
  $("detail").hidden = true;

  const summary = await getJSON(url(`api/region/${id}/summary`));
  state.summary = summary;
  const region = summary.region;

  $("regionLine").textContent = `${region.name} — ${region.hazard}`;
  document.querySelectorAll("#regionChips button").forEach((button) =>
    button.setAttribute("aria-pressed", String(button.dataset.region === id)));

  renderProvenance(summary);
  renderCounters(summary);
  renderLayerToggles();
  renderTypeFilter(summary);

  if (region.bbox) {
    map.fitBounds([[region.bbox[0], region.bbox[1]], [region.bbox[2], region.bbox[3]]],
      { padding: 60, duration: 0 });
  }
  if (region.has_assets) addRegionLayers(id); else removeRegionLayers();
  await addOverlays(id);
  // the landslide surface is the point of a landslide region, so it starts on
  if (!region.has_assets) state.layers.landslide = true;
  renderLayerToggles();
  applyFilters();
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
      url(`api/region/${state.region}/assets?q=${encodeURIComponent(query)}&limit=12`));
    const list = $("searchResults");
    list.innerHTML = found.assets.map((a) => `
      <li data-id="${a.asset_id}">${a.name}
        <span class="type">${a.asset_type} · ${fixed(a.flood_risk)}</span></li>`).join("")
      || "<li>No assets match</li>";
    list.hidden = false;
    list.querySelectorAll("li[data-id]").forEach((item) =>
      item.addEventListener("click", async () => {
        const asset = await getJSON(url(`api/region/${state.region}/asset/${item.dataset.id}`));
        map.flyTo({ center: [asset.lon, asset.lat], zoom: 13 });
        showDetail(asset);
        list.hidden = true;
      }));
  }, 180);
});

/* ── Wiring ───────────────────────────────────────────────────────────── */
$("minScore").addEventListener("input", (event) => {
  state.minScore = Number(event.target.value);
  $("minScoreOut").textContent = state.minScore.toFixed(2);
  applyFilters();
});
document.querySelectorAll("#panelButtons button").forEach((button) =>
  button.addEventListener("click", () => openPanel(button.dataset.panel)));
$("sheetClose").addEventListener("click", closePanel);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.panel) closePanel();
});

map.on("click", "assets-circle", async (event) => {
  const feature = event.features && event.features[0];
  if (!feature) return;
  // The tile carries the asset's own properties, but the detail card shows
  // the database row, which is the pipeline's output rather than a rounded
  // copy of it.
  const name = feature.properties.name;
  const found = await getJSON(
    url(`api/region/${state.region}/assets?q=${encodeURIComponent(name || "")}&limit=40`));
  const match = found.assets.find((a) =>
    Math.abs(a.lat - event.lngLat.lat) < 1e-4 && Math.abs(a.lon - event.lngLat.lng) < 1e-4)
    || found.assets[0];
  if (match) showDetail(match);
});
map.on("mouseenter", "assets-circle", () => (map.getCanvas().style.cursor = "pointer"));
map.on("mouseleave", "assets-circle", () => (map.getCanvas().style.cursor = ""));

(async function start() {
  showLoading("Loading…");
  const { regions } = await getJSON(url("api/regions"));
  state.regions = regions;
  $("regionChips").innerHTML = regions.map((region) => {
    const short = { rangpur_rajshahi: "Rangpur", sylhet: "Sylhet",
                    sw_coastal: "Coast", cht: "Hill Tracts" }[region.id] || region.name;
    return `<button data-region="${region.id}" aria-pressed="false">${short}</button>`;
  }).join("");
  document.querySelectorAll("#regionChips button").forEach((button) =>
    button.addEventListener("click", () => selectRegion(button.dataset.region)));

  await new Promise((resolve) => map.on("load", resolve));
  await selectRegion(regions[0].id);
})();

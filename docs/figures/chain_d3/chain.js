/* Fermium Hazard Mapper processing chain, drawn with D3.
   Layout and styling follow the GENESIS architecture figure: stage labels
   rotated down the left margin, source boxes with counts, a feature-chip row,
   an expanded model block with dashed bypass connectors, and output boxes.
   Flow is top -> bottom. Every count comes from counts.json, which
   make_figures.py writes from the pipeline outputs.
   drawChain(svg, d3, n): svg is a D3 selection of an <svg>, n the counts. */
function drawChain(svg, d3, n) {
  const W = 1100, H = 900;
  svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H)
    .attr("font-family", "Helvetica, Arial, sans-serif");

  const C = {
    text: "#283747", muted: "#5D6D7E", arrow: "#8A97A0",
    block: "#EEF2F4", blockStroke: "#B8C4CC",
    conv: "#F4A582", act: "#FCE3A0", head: "#A6CEE3",
    // No green or purple: groups are told apart by blue steps (pale, mid,
    // deep), with orange and yellow kept for the model layers.
    calib: "#86B0DD", risk: "#E89B6C", out: "#3F77B8",
    terrain: "#A6CEE3", people: "#D3E4F5", access: "#86B0DD", type: "#D7B8A0",
    label: "#5D6D7E", terrainTxt: "#2E6F9E", peopleTxt: "#4F7FB5", accessTxt: "#2B5C99",
    src: "#E5E8E8", srcStroke: "#AAB7B8", bar: "#D5DBDB",
    train: "#D6DEE3", calibBlk: "#86B0DD", test: "#3F77B8",
  };
  const fmt = d3.format(",");

  const defs = svg.append("defs");
  defs.append("marker").attr("id", "arr").attr("viewBox", "0 0 10 10")
    .attr("refX", 8).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7)
    .attr("orient", "auto-start-reverse")
    .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", C.arrow);
  const f = defs.append("filter").attr("id", "sh").attr("x", "-20%").attr("y", "-20%")
    .attr("width", "140%").attr("height", "140%");
  f.append("feDropShadow").attr("dx", 0).attr("dy", 1.2).attr("stdDeviation", 1.4)
    .attr("flood-color", "#000").attr("flood-opacity", 0.16);

  svg.append("rect").attr("width", W).attr("height", H).attr("fill", "#ffffff");

  // ---- helpers (as in the GENESIS figure) ----
  function rrect(x, y, w, h, fill, r = 9, stroke = null, shadow = true, dash = false) {
    const e = svg.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h)
      .attr("rx", r).attr("ry", r).attr("fill", fill);
    if (stroke) e.attr("stroke", stroke).attr("stroke-width", 1.1);
    if (dash) e.attr("stroke-dasharray", "5 3");
    if (shadow) e.attr("filter", "url(#sh)");
    return e;
  }
  function label(x, y, lines, { size = 14, weight = "normal", fill = C.text, anchor = "middle", italic = false } = {}) {
    const arr = Array.isArray(lines) ? lines : [lines];
    const t = svg.append("text").attr("x", x).attr("y", y).attr("text-anchor", anchor)
      .attr("font-size", size).attr("font-weight", weight).attr("fill", fill)
      .attr("dominant-baseline", "middle");
    if (italic) t.attr("font-style", "italic");
    arr.forEach((ln, i) => t.append("tspan").attr("x", x)
      .attr("dy", i === 0 ? -(arr.length - 1) * (size * 0.62) : size * 1.24).text(ln));
    return t;
  }
  function boxL(x, y, w, h, fill, lines, opt = {}) {
    rrect(x, y, w, h, fill, opt.r ?? 9, opt.stroke, opt.shadow ?? true, opt.dash);
    label(x + w / 2, y + h / 2, lines, { size: opt.size ?? 14, weight: opt.weight ?? "600", fill: opt.fill ?? C.text });
  }
  function vArrow(x, yTop, yBot) {
    svg.append("line").attr("x1", x).attr("y1", yTop).attr("x2", x).attr("y2", yBot)
      .attr("stroke", C.arrow).attr("stroke-width", 1.6).attr("marker-end", "url(#arr)");
  }
  function elbow(pts, r = 12, { dash = false, head = true } = {}) {
    let d = `M ${pts[0].x} ${pts[0].y}`;
    for (let i = 1; i < pts.length - 1; i++) {
      const p0 = pts[i - 1], p = pts[i], p1 = pts[i + 1];
      const l0 = Math.hypot(p.x - p0.x, p.y - p0.y), l1 = Math.hypot(p1.x - p.x, p1.y - p.y);
      const rr = Math.min(r, l0 / 2, l1 / 2);
      const a = { x: p.x - (p.x - p0.x) / l0 * rr, y: p.y - (p.y - p0.y) / l0 * rr };
      const b = { x: p.x + (p1.x - p.x) / l1 * rr, y: p.y + (p1.y - p.y) / l1 * rr };
      d += ` L ${a.x} ${a.y} Q ${p.x} ${p.y} ${b.x} ${b.y}`;
    }
    const last = pts[pts.length - 1];
    d += ` L ${last.x} ${last.y}`;
    const p = svg.append("path").attr("fill", "none").attr("stroke", C.arrow)
      .attr("stroke-width", 1.6).attr("d", d);
    if (head) p.attr("marker-end", "url(#arr)");
    if (dash) p.attr("stroke-dasharray", "4 3");
    return p;
  }
  function stage(y, txt) {
    label(30, y, txt, { size: 13.5, weight: "700", fill: C.muted })
      .attr("transform", `rotate(-90 30 ${y})`);
  }

  const L = 80, R = 1080;

  // ===================== DATA SOURCES =====================
  stage(62, "Data");
  const srcs = [
    ["OpenStreetMap", `${fmt(n.assets)} assets`],
    ["SRTM", `${n.srtm_m} m elevation`],
    ["Sentinel-1", `${n.events} flood events`],
    ["JRC GSW", `${n.jrc_m} m water`],
    ["WorldPop", `${n.worldpop_m} m population`],
    ["geoBoundaries", `${fmt(n.unions)} unions`],
    ["NASA COOLR", `${fmt(n.landslides)} landslides`],
  ];
  const sGap = 12, sW = (R - L - (srcs.length - 1) * sGap) / srcs.length;
  const srcX = srcs.map((_, i) => L + i * (sW + sGap));
  srcs.forEach(([name, detail], i) => {
    boxL(srcX[i], 34, sW, 56, C.src, [name, detail], { stroke: C.srcStroke, size: 14 });
    vArrow(srcX[i] + sW / 2, 90, 114);
  });

  // ===================== PREPROCESS =====================
  stage(136, "Preprocess");
  boxL(L, 114, R - L, 44, C.bar,
    `Clip to region · reproject to UTM 46N · slope, TWI, HAND, flow accumulation · district labels · ${n.grid_m} m grid`,
    { stroke: C.srcStroke, size: 14 });

  // ===================== FEATURES =====================
  stage(232, "Features");
  const fTop = 182, fH = 102, midX = (L + R) / 2;
  vArrow(midX, 158, fTop);
  rrect(L, fTop, R - L, fH, "#F7F9FA", 12, "#CBD5DB", false);
  const chipY = fTop + 14, chipH = 30, chipMid = chipY + chipH / 2;
  const chip = (x, w, fill, txt, tcol = C.text) => {
    rrect(x, chipY, w, chipH, fill, 6, null, false);
    label(x + w / 2, chipMid, txt, { size: 13, weight: "600", fill: tcol });
    return x + w;
  };
  const groups = [
    { toks: ["y"], w: 40, fill: C.label, tcol: "#fff", cap: "label", ccol: C.muted },
    { toks: ["elev", "slope", "TWI", "HAND", "flow"], w: 52, fill: C.terrain, cap: "terrain", ccol: C.terrainTxt },
    { toks: ["pop"], w: 46, fill: C.people, cap: "people", ccol: C.peopleTxt },
    { toks: ["hospital", "school", "shelter", "road", "canal"], w: 64, fill: C.access, cap: "access distances", ccol: C.accessTxt },
    { toks: ["type"], w: 50, fill: C.type, cap: "asset", ccol: "#8A6A50" },
  ];
  const gIn = 5, gSep = 24;
  const groupW = g => g.toks.length * g.w + (g.toks.length - 1) * gIn;
  const total = groups.reduce((a, g) => a + groupW(g), 0) + (groups.length - 1) * gSep;
  let x = midX - total / 2;
  groups.forEach(g => {
    const start = x;
    g.toks.forEach(t => { x = chip(x, g.w, g.fill, t, g.tcol) + gIn; });
    const end = x - gIn;
    label((start + end) / 2, chipY + chipH + 15, g.cap, { size: 12, fill: g.ccol, weight: "700" });
    x = end + gSep;
  });
  label(midX, fTop + fH - 14,
    "each asset = standardised feature vector · y = ground flooded in at least m of the mapped Sentinel-1 events (Hill Tracts: COOLR landslide points)",
    { size: 12, fill: C.muted, italic: true });

  // ===================== MODEL =====================
  stage(468, "Model");
  const mR = 830;                                   // models left of mR, validation right
  const bkY = 312, bkH = 26;
  const trainW = (mR - L) * n.train_pct / 100, restW = (mR - L - trainW) / 2;
  const seg = (x0, w, fill, txt) => {
    svg.append("rect").attr("x", x0).attr("y", bkY).attr("width", w).attr("height", bkH).attr("fill", fill);
    label(x0 + w / 2, bkY + bkH / 2, txt, { size: 12.5, weight: "600" });
  };
  seg(L, trainW, C.train, `${n.block_km} km spatial blocks · ${n.train_pct} % training, shared by both models`);
  seg(L + trainW, restW, C.calibBlk, "calibration");
  seg(L + trainW + restW, restW, C.test, "test");
  svg.selectAll("text").filter(function () { return this.textContent === "test"; }).attr("fill", "#FFFFFF");
  svg.append("rect").attr("x", L).attr("y", bkY).attr("width", mR - L).attr("height", bkH)
    .attr("rx", 6).attr("fill", "none").attr("stroke", "#CBD5DB");
  vArrow(midX, fTop + fH, bkY);

  // Asset model, expanded. The default is gradient boosting; the graph
  // network it replaced is kept as a configuration option.
  const gL = L, gR = L + 400, gcx = (gL + gR) / 2, gTop = 356, gBot = 604;
  rrect(gL, gTop, gR - gL, gBot - gTop, C.block, 14, C.blockStroke, false);
  label(gL + 14, gTop + 16, "Asset model", { size: 13, weight: "700", fill: C.muted, anchor: "start" });
  vArrow(gcx, bkY + bkH, gTop + 34);
  const lX = gcx - 140, lW = 280;
  boxL(lX, gTop + 34, lW, 46, C.conv, ["Class-weighted", "gradient boosting"], { size: 14, weight: "700" });
  boxL(lX, gTop + 94, lW, 30, C.act,
       `${n.boost_iterations} iterations · rate ${n.boost_lr}`, { size: 13 });
  boxL(lX, gTop + 138, lW, 30, C.head, "Score per asset", { size: 13, weight: "600" });
  vArrow(gcx, gTop + 80, gTop + 94);
  vArrow(gcx, gTop + 124, gTop + 138);
  boxL(lX, gTop + 186, lW, 50, "#FFFFFF",
       ["alternative: GraphSAGE on the", `k-NN graph (k = ${n.k}, edges \u2264 ${n.max_edge_km} km)`],
       { size: 12, weight: "600", stroke: C.blockStroke, shadow: false, dash: true, fill: C.muted });

  // Terrain hazard model.
  const tL = gR + 20, tR = mR, tcx = (tL + tR) / 2;
  rrect(tL, gTop, tR - tL, gBot - gTop, C.block, 14, C.blockStroke, false);
  label(tL + 14, gTop + 16, "Terrain hazard model", { size: 13, weight: "700", fill: C.muted, anchor: "start" });
  vArrow(tcx, bkY + bkH, gTop + 30);
  const tX = tcx - 150, tW = 300;
  boxL(tX, gTop + 30, tW, 32, C.terrain, "elev · slope · TWI · HAND · flow", { size: 13 });
  boxL(tX, gTop + 76, tW, 50, C.conv, ["Class-weighted", "logistic regression"], { size: 14, weight: "700" });
  boxL(tX, gTop + 140, tW, 44, C.head, ["Evaluated at every", `${n.srtm_m} m cell`], { size: 13 });
  vArrow(tcx, gTop + 62, gTop + 76);
  vArrow(tcx, gTop + 126, gTop + 140);
  boxL(tX + 40, gTop + 198, tW - 80, 38, "#FFFFFF", ["alternative: Kriged", "graph scores"],
    { size: 12, weight: "600", stroke: C.blockStroke, shadow: false, dash: true, fill: C.muted });

  // Validation column.
  const vL = mR + 26, vW = R - vL, vTop = gTop + 60, vH = 128;
  boxL(vL, vTop, vW, vH, "#FFFFFF",
    ["Validation", "on test blocks against", "observed Sentinel-1", "floods: AUC, AP lift,", "Brier score"],
    { size: 13, weight: "600", stroke: C.arrow, shadow: false, dash: true });
  elbow([{ x: mR, y: bkY + bkH / 2 }, { x: vL + vW / 2, y: bkY + bkH / 2 }, { x: vL + vW / 2, y: vTop }], 12, { dash: true });
  elbow([{ x: tR, y: gTop + 162 }, { x: vL, y: gTop + 162 }], 6, { dash: true });
  elbow([{ x: gcx + lW / 2, y: gTop + 153 }, { x: gR + 10, y: gTop + 225 }, { x: gR + 10, y: gBot + 12 },
         { x: vL + vW / 2, y: gBot + 12 }, { x: vL + vW / 2, y: vTop + vH }], 10, { dash: true });

  // ===================== CALIBRATE =====================
  stage(662, "Calibrate");
  const cY = 638, cH = 46;
  vArrow(gcx, gBot, cY);
  vArrow(tcx, gBot, cY);
  boxL(gcx - 190, cY, 380, cH, C.calib, ["Isotonic calibration on", "calibration blocks → probability"], { size: 13.5, weight: "700" });
  boxL(tcx - 150, cY, 300, cH, C.head, "Hazard surface H", { size: 14, weight: "700" });

  // ===================== RISK =====================
  stage(748, "Risk");
  const rY = 720, rH = 56;
  const kW = 380, kX = tcx - kW / 2;                       // composite under H
  const eW = kX - 24 - L, vX = kX + kW + 24, vWd = R - vX;
  boxL(L, rY, eW, rH, C.people, ["Exposure E", "mapped assets | people"], { size: 13.5, weight: "700" });
  boxL(kX, rY, kW, rH, C.risk, ["Composite risk ∛(H · E · V)", "two readings · quintile classes"],
    { size: 14, weight: "700", fill: "#3B2415" });
  boxL(vX, rY, vWd, rH, C.access, ["Vulnerability V", "population · access"], { size: 13.5, weight: "700" });
  vArrow(tcx, cY + cH, rY);
  vArrow(L + eW, rY + rH / 2, rY + rH / 2);
  svg.append("line").attr("x1", L + eW).attr("y1", rY + rH / 2).attr("x2", kX).attr("y2", rY + rH / 2)
    .attr("stroke", C.arrow).attr("stroke-width", 1.6).attr("marker-end", "url(#arr)");
  svg.append("line").attr("x1", vX).attr("y1", rY + rH / 2).attr("x2", kX + kW).attr("y2", rY + rH / 2)
    .attr("stroke", C.arrow).attr("stroke-width", 1.6).attr("marker-end", "url(#arr)");

  // ===================== OUTPUTS =====================
  stage(850, "Outputs");
  const oY = 818, oH = 64, oGap = 24, oW = (R - L - 2 * oGap) / 3;
  const oX = [L, L + oW + oGap, L + 2 * (oW + oGap)];
  boxL(oX[0], oY, oW, oH, C.out, ["Public dashboard", "scores, probabilities, maps"], { size: 13.5, weight: "700", fill: "#FFFFFF" });
  boxL(oX[1], oY, oW, oH, C.out, ["Union, upazila, district", "summaries · Gᵢ* hotspots"], { size: 13.5, weight: "700", fill: "#FFFFFF" });
  boxL(oX[2], oY, oW, oH, C.out, ["Data archive", "inputs, outputs, checksums"], { size: 13.5, weight: "700", fill: "#FFFFFF" });
  const busY = 800;
  oX.forEach(ox => elbow([{ x: tcx, y: rY + rH }, { x: tcx, y: busY }, { x: ox + oW / 2, y: busY }, { x: ox + oW / 2, y: oY }], 10));
  // The calibrated probability goes straight to the dashboard's asset view.
  elbow([{ x: gcx - 190, y: cY + cH / 2 }, { x: L - 16, y: cY + cH / 2 }, { x: L - 16, y: oY + oH / 2 }, { x: L, y: oY + oH / 2 }], 10);
}

if (typeof globalThis !== "undefined") globalThis.drawChain = drawChain;

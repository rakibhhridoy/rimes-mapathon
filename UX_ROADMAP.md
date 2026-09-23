# Fermium HazMapper — UX Enhancement Roadmap

## Confirmed Design Decisions

| Area | Choice |
|---|---|
| KPI Cards | Glassmorphism — frosted glass, blur backdrop, glow borders |
| Union Risk Cards | Gradient fill background based on risk level |
| Map Viewing | Fullscreen toggle button, default 620px |
| Map Popups | Rich card with risk gauge, factor breakdown bars, View Detail + Zoom buttons |
| Detail View | Slide-in panel from right on asset/card click |
| Panel Actions | Download CSV/PDF, Share link, Flag for review, Compare mode |
| Theme | Current dark refined — better contrast, consistent spacing, glassmorphism accents |
| Typography | Mixed — sans-serif (Inter/DM Sans) for labels, DM Mono for data values |
| Animations | Hover scale/glow effects, loading skeletons, smooth fade transitions |

---

## Phase 1: Cards & Typography

### 1.1 KPI Strip — Glassmorphism Cards
- Replace `.kpi-item` CSS with frosted glass: `backdrop-filter: blur(12px)`, semi-transparent bg
- Add soft glow border (`box-shadow: 0 0 15px rgba(56,189,248,0.15)`)
- Large metric value (DM Mono, 1.8rem, 700 weight)
- Small label (Inter/DM Sans, 0.65rem, uppercase)
- Progress bar below value showing percentage of max
- Category icon (SVG, not emoji) left-aligned
- **Files:** `app.py` (CSS block lines 46-170), KPI render section

### 1.2 Union Risk Cards — Gradient Fill
- Replace 4px left border with full gradient background
- Gradient: risk-color (left) fading to dark bg (right)
- Critical: `linear-gradient(135deg, rgba(239,68,68,0.3), #0d1822)`
- High: `linear-gradient(135deg, rgba(245,158,11,0.3), #0d1822)`
- Moderate: `linear-gradient(135deg, rgba(234,179,8,0.2), #0d1822)`
- Low: `linear-gradient(135deg, rgba(34,197,94,0.2), #0d1822)`
- Add hover: `transform: scale(1.02); box-shadow: 0 4px 20px rgba(0,0,0,0.3)`
- **Files:** `dashboard/components/risk_cards.py`

### 1.3 Mixed Typography
- Import Inter/DM Sans from Google Fonts alongside DM Mono
- Labels, headings, body text: Inter (sans-serif)
- Data values, metrics, code: DM Mono (monospace)
- **Files:** `app.py` (CSS block)

---

## Phase 2: Map Enhancements

### 2.1 Fullscreen Toggle
- Add a floating button (top-left of map) with expand icon
- On click: map takes 95vh height, sidebar collapses, overlay hides
- Second click: return to default 620px layout
- Use `st.session_state.map_fullscreen` to track state
- **Files:** `app.py`, `dashboard/components/map_view.py`

### 2.2 Rich Card Popups
Replace current gauge-only popup with structured card:
```
+------------------------+
| [icon] Asset Name      |
| Type | Division        |
| Risk: 0.825           |
| [========--] 82%      |
|------------------------|
| Hazard   [======] 0.9 |
| Exposure [====--] 0.7 |
| Vuln     [===---] 0.6 |
|------------------------|
| [View Detail] [Zoom]  |
+------------------------+
```
- Factor breakdown bars: hazard, exposure, vulnerability
- "View Detail" triggers slide-in panel via JS postMessage
- "Zoom" centers and zooms map to asset location
- **Files:** `dashboard/components/map_view.py` (`_gauge_popup` function)

---

## Phase 3: Slide-in Detail Panel

### 3.1 Panel Structure
- Width: 400px, slides from right edge
- Triggered by: clicking asset marker, union card, or table row
- Sections:
  - Header: asset name, type badge, risk score gauge
  - Risk breakdown: horizontal bars for each factor
  - Kriging confidence interval visualization
  - CVI class breakdown
  - Location mini-map (small static folium)
  - Historical context (if available)

### 3.2 Panel Actions
- **Download CSV/PDF:** Export selected asset/union data
- **Share link:** Copy deep link with asset ID as query param
- **Flag for review:** Mark asset with a note, save to `data/output/flagged.json`
- **Compare mode:** Toggle to select 2 items, show side-by-side in split panel

### 3.3 Implementation
- Use Streamlit `st.dialog` or custom CSS slide-in with `st.session_state`
- Store selected asset in `st.session_state.selected_asset`
- Panel renders conditionally based on session state
- Close button (X) or click outside to dismiss
- **Files:** New file `dashboard/components/detail_panel.py`, updates to `app.py`

---

## Phase 4: Visual Polish

### 4.1 Hover Animations
- Cards: `transition: transform 0.2s, box-shadow 0.2s`
- Buttons: subtle glow on hover
- Table rows: highlight row on hover
- Tab headers: underline slide animation
- **Files:** `app.py` (CSS block)

### 4.2 Loading Skeletons
- Shimmer placeholder blocks for:
  - KPI strip (8 rectangular shimmer blocks)
  - Map area (large shimmer rectangle)
  - Analytics charts (3 shimmer rectangles)
  - Union cards (4 shimmer card outlines)
- CSS animation: `@keyframes shimmer` with gradient sweep
- Show while `st.spinner` is active or data is loading
- **Files:** New utility `dashboard/components/skeletons.py`, `app.py`

### 4.3 Smooth Transitions
- Tab switch: fade-in content (`@keyframes fadeIn { from { opacity:0 } to { opacity:1 } }`)
- Card appearance: staggered fade-in on page load
- Panel open/close: CSS `transform: translateX()` with `transition: 0.3s ease`
- **Files:** `app.py` (CSS block)

---

## Phase 5: Accessibility & Refinement

### 5.1 Contrast Refinement
- Audit all text against WCAG AA (4.5:1 minimum)
- Current `#5a8ab0` on `#0a0e14` may fail — bump to `#7abcd8`
- Ensure all interactive elements have visible focus states

### 5.2 Responsive Considerations
- Desktop-first (primary target)
- Tablet: stack map and analytics vertically, reduce card columns to 2
- No mobile optimization required (dashboard is desktop-oriented)

---

## Implementation Priority

1. **Phase 1** — Cards & Typography (visual impact, low complexity)
2. **Phase 2** — Map Enhancements (core UX improvement)
3. **Phase 4** — Visual Polish (animations, skeletons)
4. **Phase 3** — Slide-in Panel (highest complexity, new component)
5. **Phase 5** — Accessibility (final pass)

---

## Technical Notes

- All CSS changes go through `app.py`'s global style block (lines 46-170)
- New components should follow existing pattern: one function per file, imported in `app.py`
- Use `st.session_state` for all UI state (fullscreen, selected asset, compare mode)
- Maintain dark/light theme support for all new components
- Test with real pipeline data (18MB assets, 71MB grid) for performance

"""
The dashboard's colours, defined once.

Every surface, text and accent colour comes from here, so a change to the
palette is a change to one file rather than to a hundred inline styles. The
blue is the same accent the technical document's figures use, and it passed
the contrast and colour-vision checks there.

Risk classes keep the conventional red, amber and green of hazard mapping,
darkened for a white background: on white the pale web defaults fall below
the 4.5:1 contrast that small text needs.
"""

# ── Surfaces ────────────────────────────────────────────────────────────────
BG = "#f7f9fb"          # page behind everything
SURFACE = "#ffffff"     # cards, panels, popups
SURFACE_ALT = "#eef2f7"  # inset areas, table stripes, disabled controls
SURFACE_TINT = "#eaf2fc"  # faint blue wash for highlighted panels

# ── Lines ───────────────────────────────────────────────────────────────────
BORDER = "#dbe3ec"
BORDER_STRONG = "#b9c6d6"

# ── Text ────────────────────────────────────────────────────────────────────
TEXT = "#0f172a"        # headings and body
TEXT_MUTED = "#55637a"  # labels, captions (4.9:1 on SURFACE)
TEXT_DIM = "#6b7a91"    # least important, never below 12px

# ── Accent ──────────────────────────────────────────────────────────────────
ACCENT = "#2a78d6"      # links, focus rings, the primary series in charts
ACCENT_DARK = "#1c5cab"
ACCENT_SOFT = "#cde2fb"
GLOW = "rgba(42,120,214,0.14)"

# ── Risk classes ────────────────────────────────────────────────────────────
RISK_HIGH = "#c62828"
RISK_MEDIUM = "#b45309"
RISK_LOW = "#15803d"
RISK_NONE = "#94a3b8"   # no mapped exposure: absence of data, not of risk

# ── Factor bars and asset types ─────────────────────────────────────────────
HAZARD = "#c62828"
EXPOSURE = "#b45309"
VULNERABILITY = "#6d28d9"
HEAT_GRADIENT = {"0.2": "#cde2fb", "0.5": "#6da7ec", "0.8": "#2a78d6", "1.0": "#104281"}

ASSET_COLORS = {
    "hospital": "#c62828",
    "school": "#b45309",
    "bridge": "#b45309",
    "road": "#55637a",
    "railway": "#6d28d9",
    "flood_shelter": "#15803d",
    "embankment": "#0f766e",
    "cropland": "#15803d",
    "other": "#6b7a91",
}

# ── Status ──────────────────────────────────────────────────────────────────
GOOD = "#15803d"
GOOD_BG = "#e8f6ee"
WARN = "#b45309"
WARN_BG = "#fdf3e3"
INFO_BG = SURFACE_TINT

# ── Map ─────────────────────────────────────────────────────────────────────
# A light grey basemap lets the coloured markers carry the signal. Esri's
# canvas is used rather than CARTO Positron, which now stamps "API KEY
# REQUIRED" across every tile served without a key.
TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")
TILE_ATTR = ("Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ")


def risk_color(score) -> str:
    """Colour for a composite risk score, or grey where there is no data."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return RISK_NONE
    if value != value or value < 0:      # NaN or the "missing" sentinel
        return RISK_NONE
    if value >= 0.6:
        return RISK_HIGH
    if value >= 0.3:
        return RISK_MEDIUM
    return RISK_LOW

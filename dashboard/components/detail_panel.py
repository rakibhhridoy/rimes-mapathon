# dashboard/components/detail_panel.py
"""
Slide-in detail panel for inspecting individual assets or unions.
Triggered via st.session_state.selected_asset.
"""

import streamlit as st


def _risk_color(score: float) -> str:
    if score >= 0.7:
        return "#c62828"
    elif score >= 0.5:
        return "#b45309"
    elif score >= 0.3:
        return "#a16207"
    return "#15803d"


def _risk_label(score: float) -> str:
    if score >= 0.7:
        return "VERY HIGH"
    elif score >= 0.5:
        return "HIGH"
    elif score >= 0.3:
        return "MODERATE"
    return "LOW"


def _bar_html(label: str, value, color: str) -> str:
    """Horizontal progress bar row; blank row when the value is missing."""
    if value is None or value != value:
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
            f'<span style="color:#55637a;font-size:10px;width:70px;text-align:right;'
            f'font-family:Inter,sans-serif;">{label}</span>'
            f'<span style="color:#94a3b8;font-size:10px;">not available</span></div>'
        )
    value = float(value)
    pct = min(value * 100, 100)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
        f'<span style="color:#55637a;font-size:10px;width:70px;text-align:right;'
        f'font-family:Inter,sans-serif;">{label}</span>'
        f'<div style="flex:1;background:rgba(15,23,42,0.04);border-radius:3px;'
        f'height:7px;overflow:hidden;">'
        f'<div style="background:linear-gradient(90deg,{color}88,{color});'
        f'width:{pct:.0f}%;height:7px;border-radius:3px;"></div></div>'
        f'<span style="color:#0f172a;font-size:10px;font-family:DM Mono,monospace;'
        f'width:38px;">{value:.3f}</span></div>'
    )


def _gauge_svg(score: float, color: str, size: int = 120) -> str:
    """Semi-circle gauge SVG."""
    pct = min(score * 100, 100)
    half = size // 2
    r = int(size * 0.35)
    cy = int(size * 0.42)
    sw = max(6, size // 12)
    dash = pct * 1.26
    val_str = f"{score:.3f}" if score > 0 else "N/A"
    return (
        f'<svg width="{size}" height="{int(size*0.5)}" viewBox="0 0 {size} {int(size*0.5)}">'
        f'<path d="M {size*0.1} {cy} A {r} {r} 0 0 1 {size*0.9} {cy}" '
        f'fill="none" stroke="#1e293b" stroke-width="{sw}" stroke-linecap="round"/>'
        f'<path d="M {size*0.1} {cy} A {r} {r} 0 0 1 {size*0.9} {cy}" '
        f'fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
        f'stroke-dasharray="{dash} 126"/>'
        f'<text x="{half}" y="{cy-2}" text-anchor="middle" font-size="{max(12, size//8)}" '
        f'font-weight="bold" fill="{color}" font-family="DM Mono,monospace">{val_str}</text>'
        f'</svg>'
    )


def inject_panel_css():
    """Inject the CSS for the slide-in detail panel. Call once in app.py."""
    st.markdown("""
    <style>
    .detail-panel {
        background: rgba(255,255,255,0.96);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid #dbe3ec;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(15,23,42,0.08);
        animation: slideIn 0.3s ease-out;
    }
    .detail-panel .dp-header {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 12px; padding-bottom: 10px;
        border-bottom: 1px solid #dbe3ec;
    }
    .detail-panel .dp-section {
        margin: 10px 0; padding: 10px 12px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(219,227,236,0.9);
        border-radius: 8px;
    }
    .detail-panel .dp-section-title {
        color: #2a78d6; font-size: 10px; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        font-family: 'Inter', sans-serif; margin-bottom: 8px;
    }
    .detail-panel .dp-actions {
        display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap;
    }
    .detail-panel .dp-actions button {
        background: rgba(255,255,255,0.94); backdrop-filter: blur(8px);
        border: 1px solid #dbe3ec; border-radius: 6px;
        color: #55637a; font-size: 10px; padding: 5px 12px;
        cursor: pointer; font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
    }
    .detail-panel .dp-actions button:hover {
        border-color: #2a78d6; color: #2a78d6;
        box-shadow: 0 0 12px rgba(42,120,214,0.14);
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    </style>
    """, unsafe_allow_html=True)


# Styles for the panel. Scoped to .detail-panel so they cannot leak into the
# rest of the page.
_PANEL_CSS = """
<style>
.detail-panel {
    background: rgba(255,255,255,0.96);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid #dbe3ec; border-radius: 12px;
    padding: 20px 22px; margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(15,23,42,0.08);
}
.detail-panel .dp-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid #dbe3ec;
}
.detail-panel .dp-section {
    margin: 10px 0; padding: 10px 12px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(219,227,236,0.9); border-radius: 8px;
}
.detail-panel .dp-section-title {
    color: #2a78d6; font-size: 10px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    font-family: 'Inter', sans-serif; margin-bottom: 8px;
}
</style>
"""


def render_detail_panel():
    """Render the detail panel for the asset selected in session state.

    Every field comes from the pipeline output row for that asset. Factors the
    pipeline did not produce are shown as unavailable rather than derived from
    the score.

    Expected session state shape:
        st.session_state.selected_asset = {
            "name", "asset_type", "flood_risk", "risk_rank", "division",
            "lat", "lon", "kriging_ci",
            "cell_hazard", "cell_exposure", "cell_vulnerability",
            "cell_composite_risk",
        }
    """
    asset = st.session_state.get("selected_asset")
    if not asset:
        return

    name = asset.get("name", "Unknown")
    atype = str(asset.get("asset_type", "unknown"))
    risk = float(asset.get("flood_risk") or 0)
    rank = asset.get("risk_rank", 0)
    division = asset.get("division", "")
    lat = asset.get("lat", 0)
    lon = asset.get("lon", 0)
    ci = asset.get("kriging_ci")

    color = _risk_color(risk)
    label = _risk_label(risk)
    ci_str = f"&plusmn;{ci:.3f}" if ci is not None else "not available"

    prob = asset.get("flood_probability")
    prob_block = ""
    if isinstance(prob, (int, float)) and prob == prob:
        prob_block = (
            f'<div style="text-align:center;margin-bottom:10px;font-family:Inter,sans-serif;">'
            f'<span style="color:#0f172a;font-size:20px;font-weight:700;'
            f'font-family:DM Mono,monospace;">{100 * float(prob):.0f}%</span>'
            f'<div style="color:#55637a;font-size:10px;">approximate probability that this '
            f'ground floods, calibrated on other areas of the region; local rates can '
            f'differ severalfold</div></div>'
        )

    factors = "".join(
        _bar_html(lbl, asset.get(key), col)
        for lbl, key, col in [
            ("Hazard", "cell_hazard", "#c62828"),
            ("Exposure", "cell_exposure", "#b45309"),
            ("Vulnerability", "cell_vulnerability", "#6d28d9"),
            ("Cell risk", "cell_composite_risk", "#2a78d6"),
        ]
    )

    panel_html = f"""
    <div class="detail-panel">
        <div class="dp-header">
            <div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="background:{color}20;color:{color};padding:3px 8px;
                                 border-radius:5px;font-size:11px;font-weight:700;
                                 font-family:DM Mono,monospace;">
                        {atype.replace('_', ' ').upper()}
                    </span>
                    <span style="background:{color}15;color:{color};padding:2px 8px;
                                 border-radius:10px;font-size:9px;font-weight:600;
                                 font-family:Inter,sans-serif;">{label}</span>
                </div>
                <div style="color:#0f172a;font-size:15px;font-weight:600;margin-top:6px;
                            font-family:Inter,sans-serif;">{name}</div>
                <div style="color:#55637a;font-size:10px;font-family:Inter,sans-serif;
                            margin-top:2px;">
                    {division} &middot; {lat:.4f}, {lon:.4f} &middot; Rank #{rank}
                </div>
            </div>
        </div>

        <div style="text-align:center;margin:4px 0 2px 0;">
            {_gauge_svg(risk, color, 130)}
        </div>
        <div style="text-align:center;color:#55637a;font-size:10px;
                    font-family:Inter,sans-serif;margin-bottom:8px;">
            modelled flood susceptibility of this asset (ranking score)
        </div>
        {prob_block}

        <div class="dp-section">
            <div class="dp-section-title">Risk factors of the surrounding 500 m cell</div>
            {factors}
        </div>

        <div class="dp-section">
            <div class="dp-section-title">Uncertainty</div>
            <div style="color:#0f172a;font-size:11px;font-family:DM Mono,monospace;">
                Kriging 95% CI: {ci_str}
            </div>
            <div style="color:#55637a;font-size:9px;margin-top:4px;
                        font-family:Inter,sans-serif;">
                Width of the 95% confidence interval of the kriged hazard surface
                at this location. Wider means sparser nearby data.
            </div>
        </div>
    </div>
    """

    # st.html renders inline, so the panel sizes itself to its content and
    # inherits the page background. It strips scripts, which this panel does
    # not need — the CSS classes below are scoped to .detail-panel.
    st.html(_PANEL_CSS + panel_html)

    act_cols = st.columns(2)
    with act_cols[0]:
        import pandas as pd
        st.download_button(
            "Download this asset (CSV)",
            pd.DataFrame([asset]).to_csv(index=False),
            file_name=f"{name.replace(' ', '_')}_risk.csv",
            mime="text/csv",
            key="dp_export_csv",
            width="stretch",
        )
    with act_cols[1]:
        if st.button("Close", key="dp_close", width="stretch"):
            st.session_state.selected_asset = None
            st.rerun()

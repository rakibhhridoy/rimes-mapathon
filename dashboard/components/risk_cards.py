"""
Union-level risk cards for the Unions tab.
"""

import streamlit as st
from dashboard.data import theme


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
        return "CRITICAL"
    elif score >= 0.5:
        return "HIGH"
    elif score >= 0.3:
        return "MODERATE"
    return "LOW"


def render_risk_cards(union_gdf,
                       n_display: int = 12):
    """Render union risk cards in a grid layout with gradient fill."""
    text = theme.TEXT
    text2 = theme.TEXT_MUTED

    if union_gdf is None or len(union_gdf) == 0:
        st.info("No area summaries yet. Run the full pipeline to generate them.")
        return

    # Units without mapped assets carry no score; they are listed separately
    # below rather than ranked as if they were safe.
    scored = union_gdf
    n_no_data = 0
    if "has_data" in union_gdf.columns:
        scored = union_gdf[union_gdf["has_data"].fillna(False)]
        n_no_data = int(len(union_gdf) - len(scored))
    elif "mean_risk" in union_gdf.columns:
        scored = union_gdf[union_gdf["mean_risk"].notna()]
        n_no_data = int(len(union_gdf) - len(scored))

    if len(scored) == 0:
        st.info("No area has mapped assets to score yet.")
        return

    top = scored.sort_values("mean_risk", ascending=False).head(n_display)

    cards_per_row = 4
    rows = [top.iloc[i:i + cards_per_row] for i in range(0, len(top), cards_per_row)]

    for row_data in rows:
        cols = st.columns(cards_per_row)
        for i, (_, row) in enumerate(row_data.iterrows()):
            score = row.get("mean_risk", 0)
            score = 0.0 if score != score else score  # NaN guard
            color = _risk_color(score)
            label = _risk_label(score)
            rank = int(row.get("risk_rank", 0))
            # Unit names repeat across the country, so show the parent too
            # when the boundary file provides it.
            name = row.get("admin_label") or row.get("admin_name") or "Unknown"
            pct = min(score * 100, 100)

            n_h = int(row.get("n_hospitals_exposed", 0))
            n_s = int(row.get("n_schools_exposed", 0))
            n_b = int(row.get("n_bridges_exposed", 0))
            n_r = int(row.get("n_roads", 0))
            n_c = int(row.get("n_cropland", 0))
            total = int(row.get("total_assets", 0))

            # The two readings answer different questions and are shown side
            # by side rather than one standing in for the other.
            people = row.get("mean_risk_people")
            people_txt = (f"{float(people):.3f}"
                          if people is not None and people == people else "—")

            with cols[i]:
                card_html = (
                    f'<div style="background:linear-gradient(145deg,{color}18 0%,#f7f9fb 70%);'
                    f'backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);'
                    f'border:1px solid {color}30;border-radius:12px;'
                    f'padding:14px 16px;margin-bottom:10px;'
                    f'box-shadow:0 4px 20px rgba(15,23,42,0.07),inset 0 1px 0 rgba(15,23,42,0.02);'
                    f'transition:all 0.25s ease;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-size:0.58rem;color:{color};background:{color}18;'
                    f'border:1px solid {color}35;padding:2px 8px;'
                    f'border-radius:10px;font-weight:600;font-family:Inter,sans-serif;'
                    f'letter-spacing:0.04em;">{label}</span>'
                    f'<span style="font-size:0.62rem;color:{text2};font-weight:600;'
                    f'font-family:DM Mono,monospace;">#{rank}</span>'
                    f'</div>'
                    f'<div style="font-size:0.88rem;font-weight:600;color:{text};'
                    f'margin:6px 0 2px 0;line-height:1.2;font-family:Inter,sans-serif;">{name}</div>'
                    f'<div style="font-size:1.3rem;font-weight:700;color:{color};'
                    f'margin:2px 0 6px 0;font-family:DM Mono,monospace;">{score:.3f}</div>'
                    f'<div style="background:rgba(15,23,42,0.04);border-radius:4px;height:4px;'
                    f'overflow:hidden;margin-bottom:8px;">'
                    f'<div style="background:linear-gradient(90deg,{color}88,{color});'
                    f'width:{pct:.0f}%;height:4px;border-radius:4px;"></div></div>'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.6rem;color:{text2};margin-bottom:6px;'
                    f'font-family:Inter,sans-serif;">'
                    f'<span>infrastructure</span>'
                    f'<span style="font-family:DM Mono,monospace;color:#55637a;">'
                    f'people {people_txt}</span></div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;'
                    f'gap:2px 10px;font-size:0.66rem;color:{text2};font-family:DM Mono,monospace;">'
                    f'<span>H {n_h}</span><span>S {n_s}</span>'
                    f'<span>B {n_b}</span><span>R {n_r}</span>'
                    f'<span>C {n_c}</span><span>T {total}</span>'
                    f'</div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

    if n_no_data:
        st.caption(
            f"{n_no_data:,} of {len(union_gdf):,} areas have no assets mapped in "
            "OpenStreetMap, so they cannot be scored. That is missing data, not "
            "an indication of safety."
        )

    # Full table
    with st.expander("View full ranking table"):
        display_cols = [
            c for c in [
                "risk_rank", "admin_label", "admin_name", "mean_risk",
                "mean_risk_people", "max_risk",
                "n_high_risk", "n_hospitals_exposed", "n_schools_exposed",
                "n_bridges_exposed", "n_roads", "n_cropland", "total_assets",
            ] if c in union_gdf.columns
        ]
        if display_cols:
            st.dataframe(
                union_gdf[display_cols].sort_values("mean_risk", ascending=False),
                height=400, hide_index=True,
            )

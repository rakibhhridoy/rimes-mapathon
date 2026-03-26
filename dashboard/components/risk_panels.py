# dashboard/components/risk_panels.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard.data.constants import (
    VULNERABILITY_CLASSES,
    LITERATURE,
)
from dashboard.data.loader import (
    get_upazila_risk,
    get_landslide_upazila,
    get_kriging_ci_at_point,
)

ALERT_COLORS = {
    "GREEN":  ("#0d2e24", "#3cdea0"),
    "YELLOW": ("#2e2a0d", "#dea03c"),
    "ORANGE": ("#2e1a0d", "#de7a3c"),
    "RED":    ("#2e0d0d", "#de3c3c"),
}

CVI_COLORS = {
    1: "#3cdea0", 2: "#88b860", 3: "#dea03c", 4: "#de7a3c", 5: "#de3c3c"
}


def render_alert_banner(alert_level: str, hazard: str, risk_score: float):
    bg, col = ALERT_COLORS.get(alert_level, ALERT_COLORS["YELLOW"])
    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {col};border-radius:8px;
                    padding:12px 20px;margin-bottom:16px;display:flex;
                    align-items:center;gap:16px;">
          <div style="width:14px;height:14px;border-radius:50%;background:{col};
                      flex-shrink:0;"></div>
          <div>
            <span style="color:{col};font-size:13px;font-weight:700;
                         letter-spacing:0.08em;">
              {alert_level} ALERT
            </span>
            <span style="color:{col}88;font-size:12px;margin-left:12px;">
              {hazard}
            </span>
          </div>
          <div style="margin-left:auto;text-align:right;">
            <span style="color:{col};font-size:20px;font-weight:700;">
              {risk_score:.2f}
            </span>
            <span style="color:{col}88;font-size:11px;margin-left:6px;">
              fused risk score
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(region_key: str = None, landslide: bool = False):
    """Top KPI cards row."""
    if landslide:
        metrics = [
            ("Area High-VHigh Risk", "79%", "CHT — Rahman et al. 2025", "#de3c78"),
            ("5-Day Rainfall", "187 mm", "CHIRPS + BMD - WATCH tier", "#dea03c"),
            ("Assets at Risk", "5", "GNN score > 0.70", "#de7a3c"),
            ("CVI Class 4-5 Pop.", "144k", "Exposed CHT population", "#de3c3c"),
        ]
    else:
        data = get_upazila_risk(region_key)
        exposed = sum(d["pop_exposed"] for d in data) if data else 0
        high_risk = sum(1 for d in data if d["flood_risk"] > 0.70) if data else 0
        avg_cvi   = round(sum(d["cvi"] for d in data) / len(data), 2) if data else 0.0
        metrics = [
            ("Upazilas Mapped", str(len(data)) if data else "—", "in active AOI", "#3cb8de"),
            ("High-Risk Upazilas", str(high_risk), "flood risk > 0.70", "#de7a3c"),
            ("Avg CVI Score", str(avg_cvi), "Community Vulnerability", "#de3c78"),
            ("Exposed Population", f"{exposed:,}", "within flood zone", "#dea03c"),
        ]

    cols = st.columns(4)
    for col, (label, value, sub, color) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div style="background:#0d1822;border:1px solid {color}33;
                            border-radius:8px;padding:12px 14px;text-align:center;">
                  <div style="color:#5a8ab0;font-size:10px;letter-spacing:0.08em;
                              margin-bottom:4px;">{label.upper()}</div>
                  <div style="color:{color};font-size:22px;font-weight:700;">{value}</div>
                  <div style="color:#4a7a9a;font-size:10px;margin-top:4px;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_infrastructure_table(assets: list, key_prefix: str = "infra"):
    """Sortable infrastructure exposure table."""
    st.markdown(
        "<span style='color:#3cb8de;font-size:11px;font-weight:700;"
        "letter-spacing:0.1em;'>INFRASTRUCTURE EXPOSURE — GNN RISK SCORES</span>",
        unsafe_allow_html=True,
    )
    rows = []
    for lat, lon, name, atype, score in assets:
        gnn_ci = get_kriging_ci_at_point(lat, lon, score)
        cvi_cls = 5 if score > 0.80 else 4 if score > 0.65 else 3 if score > 0.50 else 2
        rows.append({
            "Asset": name,
            "Type": atype.title(),
            "GNN Risk": round(score, 2),
            "Kriging CI": gnn_ci,
            "CVI Class": cvi_cls,
            "Status": "Critical" if score > 0.80 else "High" if score > 0.65 else "Moderate",
        })

    df = pd.DataFrame(rows)
    if df.empty or "GNN Risk" not in df.columns:
        st.info("No infrastructure assets to display.")
        return
    df = df.sort_values("GNN Risk", ascending=False)

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "GNN Risk": st.column_config.ProgressColumn(
                "GNN Risk Score",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "CVI Class": st.column_config.NumberColumn(
                "CVI Class",
                format="%d",
            ),
        },
        hide_index=True,
        height=220,
        key=f"{key_prefix}_table",
    )

    csv = df.to_csv(index=False)
    st.download_button(
        "Export CSV for field teams",
        csv,
        file_name="fermium_infrastructure_risk.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv",
    )


def render_vulnerability_chart(region_key: str):
    """CVI bar chart by upazila."""
    data = get_upazila_risk(region_key)
    if not data:
        st.info("No upazila data for this region.")
        return

    upazilas  = [d["upazila"] for d in data]
    cvi_vals  = [d["cvi"]        for d in data]
    risk_vals = [d["flood_risk"] for d in data]
    classes   = [d["class"]      for d in data]
    colors    = [CVI_COLORS.get(c, "#de3c3c") for c in classes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="CVI Score",
        x=upazilas,
        y=cvi_vals,
        marker=dict(color=colors, line=dict(width=0)),
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        name="Flood Risk",
        x=upazilas,
        y=risk_vals,
        mode="lines+markers",
        line=dict(color="#3cb8de", width=2),
        marker=dict(size=7, color="#3cb8de"),
        yaxis="y",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor="#0a0e14",
        plot_bgcolor="#0d1822",
        legend=dict(
            orientation="h", x=0, y=1.1,
            font=dict(color="#c8d6e5", size=10),
            bgcolor="#0a0e14",
        ),
        font=dict(color="#c8d6e5", size=11),
        xaxis=dict(gridcolor="#1a2a38", color="#5a8ab0"),
        yaxis=dict(
            gridcolor="#1a2a38", color="#5a8ab0",
            range=[0, 1.05], tickformat=".2f",
            title=dict(text="Score", font=dict(size=10)),
        ),
        title=dict(
            text="Upazila CVI vs Flood Risk",
            font=dict(color="#3cb8de", size=12),
        ),
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    # CVI domain weights mini display
    st.markdown(
        """
        <div style="background:#0d1f2d;border:1px solid #1a3a50;border-radius:8px;
                    padding:10px 14px;font-size:11px;display:flex;gap:12px;flex-wrap:wrap;">
          <span style="color:#de3c78;">Demo 30%</span>
          <span style="color:#dea03c;">Econ 25%</span>
          <span style="color:#a03cde;">Infra 20%</span>
          <span style="color:#3cb8de;">Housing 15%</span>
          <span style="color:#3cdea0;">Agri 10%</span>
          <span style="color:#4a7a9a;margin-left:auto;font-size:9px;">
            CVI = Exposure x Sensitivity / Adaptive Capacity
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agri_risk(region_key: str):
    """Agricultural risk monitor panel."""
    st.markdown(
        "<span style='color:#3cdea0;font-size:11px;font-weight:700;"
        "letter-spacing:0.1em;'>AGRICULTURAL RISK MONITOR</span>",
        unsafe_allow_html=True,
    )

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if "Coastal" in region_key:
        crop_line = [0.2, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.9, 0.8, 0.6, 0.4, 0.3]
        crop_name = "Saline rice / shrimp"
        ndvi_anom = [-0.02, -0.01, 0.01, 0.02, -0.03, -0.08, -0.15, -0.14, -0.09, -0.05, -0.02, -0.01]
    elif "Sylhet" in region_key:
        crop_line = [0.1, 0.2, 0.4, 0.7, 0.9, 0.95, 0.85, 0.75, 0.6, 0.4, 0.2, 0.1]
        crop_name = "Boro/Aman rice (haor)"
        ndvi_anom = [0.02, 0.01, -0.01, -0.08, -0.18, -0.22, -0.15, -0.10, -0.06, -0.02, 0.01, 0.02]
    else:
        crop_line = [0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.88, 0.70, 0.45, 0.2, 0.1]
        crop_name = "Aman/Boro rice (fluvial)"
        ndvi_anom = [0.01, 0.01, 0.00, -0.02, -0.05, -0.10, -0.14, -0.13, -0.08, -0.04, 0.00, 0.01]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=crop_line,
        name="Flood probability x crop calendar",
        fill="tozeroy",
        mode="lines",
        line=dict(color="#de7a3c", width=2),
        fillcolor="rgba(222,122,60,0.12)",
    ))
    fig.add_trace(go.Bar(
        x=months, y=ndvi_anom,
        name="NDVI anomaly vs 5yr baseline",
        marker=dict(
            color=["#de3c3c" if v < -0.08 else "#dea03c" if v < -0.03 else "#3cdea0"
                   for v in ndvi_anom],
            line=dict(width=0),
        ),
        opacity=0.7,
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor="#0a0e14",
        plot_bgcolor="#0d1822",
        font=dict(color="#c8d6e5", size=10),
        legend=dict(
            orientation="h", x=0, y=1.15,
            font=dict(size=9, color="#c8d6e5"),
            bgcolor="#0a0e14",
        ),
        xaxis=dict(gridcolor="#1a2a38", color="#5a8ab0"),
        yaxis=dict(
            gridcolor="#1a2a38", color="#5a8ab0",
            title=dict(text="Score / NDVI", font=dict(size=9)),
        ),
        title=dict(
            text=f"{crop_name} — Seasonal Flood x NDVI Risk",
            font=dict(color="#3cdea0", size=11),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_landslide_upazila():
    """Landslide upazila risk table."""
    st.markdown(
        "<span style='color:#de3c78;font-size:11px;font-weight:700;"
        "letter-spacing:0.1em;'>LANDSLIDE SUSCEPTIBILITY BY UPAZILA (CHT)</span>",
        unsafe_allow_html=True,
    )
    raw_data = get_landslide_upazila()
    df = pd.DataFrame(raw_data)
    # Normalize column names for display
    col_map = {"upazila": "Upazila", "susceptibility": "RF Susceptibility",
               "pop_exposed": "Exposed Population", "class": "CVI Class"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "Upazila" not in df.columns and len(df.columns) >= 4:
        df.columns = ["Upazila", "RF Susceptibility", "Exposed Population", "CVI Class"]
    df = df.sort_values("RF Susceptibility", ascending=False)

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "RF Susceptibility": st.column_config.ProgressColumn(
                "RF Susceptibility (AUC 0.93)",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "Exposed Population": st.column_config.NumberColumn(
                "Exposed Pop.",
                format="%d",
            ),
        },
        hide_index=True,
        height=200,
    )


def render_literature_panel():
    """Compact literature reference panel."""
    with st.expander("Contemporary Literature (2025-2026) — CHT Landslide Models", expanded=False):
        for lit in LITERATURE:
            st.markdown(
                f"""
                <div style="background:#0d1822;border:1px solid #1a2a38;border-radius:8px;
                            padding:12px 16px;margin-bottom:10px;">
                  <div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;">
                    <span style="background:#1a3a50;color:#3cb8de;padding:2px 8px;
                                 border-radius:4px;font-size:10px;font-weight:700;
                                 white-space:nowrap;flex-shrink:0;">{lit['cite']}</span>
                    <div>
                      <div style="color:#e8f4ff;font-size:12px;font-weight:500;">
                        {lit['title']}</div>
                      <div style="color:#5a8ab0;font-size:10px;margin-top:2px;">
                        {lit['journal']} - DOI: {lit['doi']}</div>
                      <div style="color:#7accd8;font-size:11px;margin-top:6px;">
                        {lit['relevance']}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

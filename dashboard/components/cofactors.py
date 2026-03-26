# dashboard/components/cofactors.py
import streamlit as st
from dashboard.data.constants import AHP_FLOOD_WEIGHTS, LANDSLIDE_DATA
import plotly.graph_objects as go


def render_flood_cofactors(region_key: str):
    """Expandable AHP weight matrix and conditioning factors for flood risk."""
    with st.expander("Conditioning Factors & Model Weights", expanded=False):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                "<span style='color:#3cb8de;font-size:11px;font-weight:700;"
                "letter-spacing:0.1em;'>AHP WEIGHT MATRIX — FLOOD RISK</span>",
                unsafe_allow_html=True,
            )
            weights = [w for _, w, _ in AHP_FLOOD_WEIGHTS]
            labels  = [f for f, _, _ in AHP_FLOOD_WEIGHTS]
            descs   = [d for _, _, d in AHP_FLOOD_WEIGHTS]

            fig = go.Figure(go.Bar(
                x=weights,
                y=labels,
                orientation="h",
                marker=dict(
                    color=["#3cb8de", "#3cb8de", "#3cdea0", "#dea03c", "#de7a3c", "#de3c78"],
                    line=dict(width=0),
                ),
                text=[f"{w:.2f}" for w in weights],
                textposition="auto",
                hovertext=descs,
                hoverinfo="text+x+y",
            ))
            fig.update_layout(
                height=240,
                margin=dict(l=0, r=10, t=10, b=10),
                paper_bgcolor="#0a0e14",
                plot_bgcolor="#0d1822",
                font=dict(color="#c8d6e5", size=11),
                xaxis=dict(
                    gridcolor="#1a2a38", tickformat=".0%",
                    range=[0, 0.30], color="#5a8ab0",
                ),
                yaxis=dict(color="#e8f4ff"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                "<span style='color:#4a7a9a;font-size:10px;'>"
                "AHP CR: 0.041 (consistent — CR &lt; 0.10)</span>",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                "<span style='color:#3cb8de;font-size:11px;font-weight:700;"
                "letter-spacing:0.1em;'>MODEL FUSION EQUATION</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="background:#0d1822;border:1px solid #1a3a50;border-radius:8px;
                            padding:14px;font-family:monospace;font-size:12px;
                            color:#a0c0d8;line-height:2.0;">
                  <span style="color:#3cb8de;">FinalRisk</span> =<br>
                  &nbsp;&nbsp;<span style="color:#dea03c;">0.35</span> x Kriging_hazard<br>
                  + <span style="color:#de3c78;">0.30</span> x GNN_infra_score<br>
                  + <span style="color:#a03cde;">0.20</span> x AHP_susceptibility<br>
                  + <span style="color:#3cdea0;">0.15</span> x CVI_vulnerability<br>
                  <br>
                  <span style="color:#5a8ab0;">Kriging model:</span>
                  <span style="color:#e8f4ff;">y(h) = c0 + c[1-e^(-h/a)]</span><br>
                  <span style="color:#5a8ab0;">GNN arch:</span>
                  <span style="color:#e8f4ff;">3-layer GraphSAGE + GAT</span><br>
                  <span style="color:#5a8ab0;">Validated vs:</span>
                  <span style="color:#e8f4ff;">2017 & 2019 BD flood records</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Proximity buffer summary
        st.markdown(
            "<span style='color:#3cb8de;font-size:11px;font-weight:700;"
            "letter-spacing:0.1em;'>PROXIMITY BUFFER THRESHOLDS</span>",
            unsafe_allow_html=True,
        )
        buf_cols = st.columns(5)
        buffers = [
            ("Rivers", "<1 km", "PopRisk Class 4-5", "#3cb8de"),
            ("Coast", "<5 km", "Surge + salinity zone", "#dea03c"),
            ("Roads", "<200 m", "CHT hill-cut risk", "#de7a3c"),
            ("Flood bodies", "<200 m", "Agri crop loss risk", "#de3c78"),
            ("Streams", "<50 m", "Landslide undercutting", "#a03cde"),
        ]
        for col, (name, dist, desc, color) in zip(buf_cols, buffers):
            with col:
                st.markdown(
                    f"""
                    <div style="background:#0d1822;border:1px solid {color}33;
                                border-radius:6px;padding:8px;text-align:center;">
                      <div style="color:{color};font-size:10px;font-weight:700;">{name}</div>
                      <div style="color:#e8f4ff;font-size:13px;font-weight:700;">{dist}</div>
                      <div style="color:#5a8ab0;font-size:9px;margin-top:3px;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_landslide_cofactors():
    """Expandable 15 conditioning factors for CHT landslide model."""
    factors = LANDSLIDE_DATA["conditioning_factors"]

    with st.expander("15 Conditioning Factors — Landslide Susceptibility (CHT)", expanded=False):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                "<span style='color:#de3c78;font-size:11px;font-weight:700;"
                "letter-spacing:0.1em;'>FEATURE IMPORTANCE (RF AUC 0.93)</span>",
                unsafe_allow_html=True,
            )
            names    = [f for f, _, _ in factors]
            weights  = [w for _, w, _ in factors]
            tooltips = [d for _, _, d in factors]

            reds = ["#de3c78", "#de7a3c", "#dea03c", "#c8a030", "#a09020",
                    "#888020", "#707020", "#586020", "#485020", "#384020",
                    "#303830"]
            fig2 = go.Figure(go.Bar(
                x=weights,
                y=names,
                orientation="h",
                marker=dict(color=reds[:len(names)], line=dict(width=0)),
                text=[f"{w:.2f}" for w in weights],
                textposition="auto",
                hovertext=tooltips,
                hoverinfo="text+x+y",
            ))
            fig2.update_layout(
                height=360,
                margin=dict(l=0, r=10, t=10, b=10),
                paper_bgcolor="#0a0e14",
                plot_bgcolor="#0d1822",
                font=dict(color="#c8d6e5", size=10),
                xaxis=dict(gridcolor="#1a2a38", color="#5a8ab0"),
                yaxis=dict(color="#e8f4ff"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown(
                "<span style='color:#de3c78;font-size:11px;font-weight:700;"
                "letter-spacing:0.1em;'>RAINFALL THRESHOLD MATRIX</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="background:#0d1822;border:1px solid #de3c7833;
                            border-radius:8px;padding:14px;font-size:11px;">
                  <table style="width:100%;border-collapse:collapse;font-family:monospace;">
                    <tr style="border-bottom:1px solid #1a2a38;">
                      <th style="color:#de3c78;padding:4px 8px;text-align:left;">
                        Alert Tier</th>
                      <th style="color:#de3c78;padding:4px 8px;text-align:right;">
                        5-day Rainfall</th>
                      <th style="color:#de3c78;padding:4px 8px;text-align:left;">
                        Action</th>
                    </tr>
                    <tr style="border-bottom:1px solid #0d1f2d;">
                      <td style="color:#3cdea0;padding:4px 8px;">No Warning</td>
                      <td style="color:#e8f4ff;padding:4px 8px;text-align:right;">&lt;71 mm</td>
                      <td style="color:#5a8ab0;padding:4px 8px;">Monitor only</td>
                    </tr>
                    <tr style="border-bottom:1px solid #0d1f2d;">
                      <td style="color:#dea03c;padding:4px 8px;">Watch</td>
                      <td style="color:#e8f4ff;padding:4px 8px;text-align:right;">71-160 mm</td>
                      <td style="color:#5a8ab0;padding:4px 8px;">Alert DDM, standby</td>
                    </tr>
                    <tr style="border-bottom:1px solid #0d1f2d;">
                      <td style="color:#de7a3c;padding:4px 8px;">Warning</td>
                      <td style="color:#e8f4ff;padding:4px 8px;text-align:right;">161-250 mm</td>
                      <td style="color:#5a8ab0;padding:4px 8px;">Evacuate high-risk zones</td>
                    </tr>
                    <tr>
                      <td style="color:#de3c3c;padding:4px 8px;">Emergency</td>
                      <td style="color:#e8f4ff;padding:4px 8px;text-align:right;">&gt;250 mm</td>
                      <td style="color:#5a8ab0;padding:4px 8px;">Full evacuation, close roads</td>
                    </tr>
                  </table>
                  <div style="color:#4a7a9a;font-size:9px;margin-top:8px;">
                    Source: Ahmed & Rahman (2018/updated 2025) - 1960-2025 BMD record
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<span style='color:#de3c78;font-size:11px;font-weight:700;"
                "letter-spacing:0.1em;'>MODEL ENSEMBLE</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="background:#0d1822;border:1px solid #de3c7833;
                            border-radius:8px;padding:14px;font-family:monospace;
                            font-size:11px;color:#a0c0d8;line-height:1.9;margin-top:8px;">
                  <span style="color:#de3c78;">Primary:</span>
                  RF (AUC 0.93, Rahman et al. 2025)<br>
                  <span style="color:#de7a3c;">Secondary:</span>
                  GBM (AUC 0.83, Roy et al. 2025)<br>
                  <span style="color:#dea03c;">Climate:</span>
                  LR-bNB + CMIP6 SSP245/585<br>
                  <span style="color:#a03cde;">Uncertainty:</span>
                  RF OOB + MC-Dropout IQR<br>
                  <span style="color:#5a8ab0;">Training pts:</span>
                  730 (primary) + 170 (CHT-wide)
                </div>
                """,
                unsafe_allow_html=True,
            )

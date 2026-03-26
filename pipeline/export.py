"""
Step 9 (cont.) — Export: GeoTIFF, GeoJSON, CSV, and PDF reports.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def export_ranked_csv(infra: gpd.GeoDataFrame, output_path: str) -> None:
    """Export ranked infrastructure assets as CSV for responders."""
    cols = [
        "risk_rank", "asset_type", "name", "priority",
        "flood_risk", "is_high_risk", "lon", "lat",
    ]
    available_cols = [c for c in cols if c in infra.columns]
    df = infra[available_cols].copy()
    df.to_csv(output_path, index=False)
    logger.info(f"Ranked CSV → {output_path} ({len(df)} assets)")


def export_geojson(gdf: gpd.GeoDataFrame, output_path: str,
                    max_features: int = None) -> None:
    """Export GeoDataFrame as GeoJSON."""
    if max_features:
        gdf = gdf.head(max_features)
    # Drop extra geometry columns (e.g. 'centroid') — GeoJSON supports only one
    extra_geom = [c for c in gdf.columns
                  if c != gdf.geometry.name and isinstance(gdf[c].dtype, gpd.array.GeometryDtype)]
    if extra_geom:
        gdf = gdf.drop(columns=extra_geom)
    gdf.to_file(output_path, driver="GeoJSON")
    logger.info(f"GeoJSON → {output_path} ({len(gdf)} features)")


def export_union_summary(union_gdf: gpd.GeoDataFrame, output_dir: Path) -> None:
    """Export union-level risk summary as GeoJSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # GeoJSON with geometry
    geojson_path = str(output_dir / "union_risk_summary.geojson")
    union_gdf.to_file(geojson_path, driver="GeoJSON")
    logger.info(f"Union summary GeoJSON → {geojson_path}")

    # CSV without geometry
    csv_path = str(output_dir / "union_risk_summary.csv")
    cols = [c for c in union_gdf.columns if c != "geometry"]
    union_gdf[cols].to_csv(csv_path, index=False)
    logger.info(f"Union summary CSV → {csv_path}")


def export_hotspots(grid_gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """Export hotspot clusters as GeoJSON."""
    hotspots = grid_gdf[grid_gdf["is_hotspot"]].copy()
    if len(hotspots) > 0:
        hotspots.to_file(output_path, driver="GeoJSON")
        logger.info(f"Hotspot GeoJSON → {output_path} ({len(hotspots)} cells)")
    else:
        logger.warning("No hotspot cells to export.")


def _safe_text(text) -> str:
    """Transliterate non-Latin text to ASCII for PDF output."""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("ascii", "replace").decode("ascii")


def generate_pdf_report(union_gdf: gpd.GeoDataFrame,
                         infra: gpd.GeoDataFrame,
                         output_path: str) -> None:
    """Generate a simple PDF situation report."""
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 not installed. Skipping PDF report.")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "SGMDI Flood Risk Situation Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Rangpur & Rajshahi Divisions, Bangladesh", ln=True, align="C")
    pdf.ln(10)

    # Summary stats
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)

    n_assets = len(infra)
    n_high = infra["is_high_risk"].sum() if "is_high_risk" in infra.columns else 0
    n_unions = len(union_gdf) if union_gdf is not None else 0

    pdf.cell(0, 6, f"Total infrastructure assets assessed: {n_assets}", ln=True)
    pdf.cell(0, 6, f"High-risk assets (risk > 0.7): {n_high}", ln=True)
    pdf.cell(0, 6, f"Administrative units (unions) analyzed: {n_unions}", ln=True)
    pdf.ln(5)

    # Top 10 at-risk unions
    if union_gdf is not None and len(union_gdf) > 0:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top 10 At-Risk Unions", ln=True)
        pdf.set_font("Helvetica", "", 9)

        top10 = union_gdf.head(10)
        for _, row in top10.iterrows():
            name = _safe_text(row.get("admin_name", "Unknown"))
            risk = row.get("mean_risk", 0)
            n_hosp = row.get("n_hospitals_exposed", 0)
            n_sch = row.get("n_schools_exposed", 0)
            n_br = row.get("n_bridges_exposed", 0)
            pdf.cell(
                0, 5,
                f"  {row.get('risk_rank', '-')}. {name} "
                f"(risk: {risk:.3f}) - "
                f"Hospitals: {n_hosp}, Schools: {n_sch}, Bridges: {n_br}",
                ln=True,
            )
        pdf.ln(5)

    # Top 20 at-risk assets
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Top 20 At-Risk Infrastructure Assets", ln=True)
    pdf.set_font("Helvetica", "", 9)

    top_assets = infra.head(20)
    for _, row in top_assets.iterrows():
        atype = _safe_text(row.get("asset_type", "unknown"))
        name = _safe_text(row.get("name", "unnamed"))
        risk = row.get("flood_risk", 0)
        pdf.cell(
            0, 5,
            f"  {row.get('risk_rank', '-')}. [{atype}] {name} "
            f"(risk: {risk:.3f})",
            ln=True,
        )

    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(
        0, 5,
        "Generated by SGMDI Pipeline - Smart Geospatial Mapping & "
        "Disaster Impact Intelligence",
        ln=True, align="C",
    )

    pdf.output(output_path)
    logger.info(f"PDF report → {output_path}")

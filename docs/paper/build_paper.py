"""Assemble docs/paper/paper.tex from the technical document plus new framing text."""
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
SRC = (DOCS / "sgmdi_technical_document.tex").read_text().split("\n")
NEW = Path(__file__).resolve().parent / "sections"


def _find(prefix, after=0):
    """Index of the first line at or after `after` that starts with `prefix`."""
    hits = [i for i in range(after, len(SRC)) if SRC[i].startswith(prefix)]
    assert hits, f"no line starts with {prefix!r}"
    return hits[0]


def R(start, stop=None, include_start=True):
    """Lines from the one starting with `start` up to, not including, the one
    starting with `stop` (or to the end). Passages are found by their text, so
    editing the technical document cannot shift what is copied."""
    i = _find(start)
    j = _find(stop, i + 1) if stop else len(SRC)
    return "\n".join(SRC[i if include_start else i + 1:j])


def P(start):
    """The single paragraph line that starts with `start`."""
    return SRC[_find(start)]


def N(name):
    return (NEW / f"{name}.tex").read_text().rstrip() + "\n"


parts = [
    R(r"\documentclass", "% ── Running header"),
    N("preamble"),
    R("% ── Code listings", r"\input{numbers.tex}"),
    r"\input{../numbers.tex}",
    "",
    r"\begin{document}",
    "",
    N("title"),
    N("introduction"),
    # The landslide model is left to its own article (docs/landslide_paper/).
    R(r"\section{Related work}", r"\subsection{Landslide susceptibility}"),
    R(r"\subsection{Exposure and vulnerability}", r"\subsection{Contribution}"),
    R(r"\section{Data}", r"\section{Methods}"),
    R(r"\section{Methods}", r"\subsection{Infrastructure}"),
    R(r"\subsection{Infrastructure}", r"\subsection{Features}"),
    r"\subsection{Flood extents from Sentinel-1}",
    R(r"\label{sec:observed}", "The validation step then"),
    "",
    R(r"\subsection{Features}", r"\section{Landslide susceptibility}"),
    r"\subsection{Validation design}",
    r"\label{sec:design}",
    "",
    P("Every model in the chain"),
    "",
    P("The validation step then"),
    "",
    N("design_extra"),
    r"\section{Results}",
    r"\label{sec:validation}",
    "",
    r"\subsection{Held-out results}",
    R(r"\subsection{Held-out results}", r"\subsection{Observed floods}", include_start=False),
    "",
    r"\subsection{Observed extents against terrain labels}",
    r"\label{sec:labelresult}",
    "",
    R("The comparison of the proxy labels", "The south-west coast behaves"),
    "",
    r"\subsection{The south-west coast}",
    "",
    P("The south-west coast behaves"),
    "",
    N("temporal_result"),
    R(r"\subsection{Model comparison}", r"\section{Dashboard}"),
    N("discussion"),
    N("backmatter"),
    R(r"\section*{Acknowledgements}"),
]
tex = "\n".join(parts) + "\n"

# Edits that make the reused text read as a paper rather than a versioned report.
EDITS = [
    # Floods only: the landslide model is reported separately.
    ("in the four areas it draws on: data-driven flood susceptibility mapping in Bangladesh, satellite flood mapping as a source of training labels, the validation of spatial models, and landslide and composite risk assessment.",
     "in the areas it draws on: data-driven flood susceptibility mapping in Bangladesh, satellite flood mapping as a source of training labels, the validation of spatial models, graph models and interpolation, and composite risk assessment."),
    ("how the flood or landslide locations were obtained", "how the flood locations were obtained"),
    ("Four regions are configured, each with its own bounding box, data directory and, where appropriate, flood events for validation (Table~\\ref{tab:regions}).",
     "Three flood regions are studied, each with its own bounding box, data directory and flood events for validation (Table~\\ref{tab:regions})."),
    ("Three of the four estimate flood risk, while the fourth, the Chittagong Hill Tracts, fits a landslide susceptibility model because its dominant hazard is slope failure, not inundation (Fig.~\\ref{fig:study}).",
     "They cover three contrasting flood regimes, riverine flooding in Rangpur and Rajshahi, flash flooding in the Sylhet haor basin and cyclone surge on the south-west coast (Fig.~\\ref{fig:study})."),
    ("figures/fig_study_area.pdf", "figures/fig_study_area_flood.pdf"),
    ("The four study regions, drawn as their configured bounding boxes over the district boundaries of geoBoundaries and tinted inside Bangladesh, blue for the three flood regions and orange for the landslide region.",
     "The three study regions, drawn as their configured bounding boxes over the district boundaries of geoBoundaries and tinted inside Bangladesh."),
    ("Chittagong Hill Tracts & rainfall-triggered landslide & 91.5, 21.5, 92.7, 23.5 & \\fileref{configs/cht.yaml} \\\\\n", ""),
    ("NASA COOLR & mapped landslide locations, Chittagong Hill Tracts & point & open \\\\\n", ""),
    ("figures/fig_chain.pdf", "figures/fig_chain_flood.pdf"),
    ("Source counts are totals over the three flood regions, except the landslide inventory.",
     "Source counts are totals over the three flood regions."),
    (" In the Chittagong Hill Tracts a landslide model fitted to the COOLR inventory takes the place of the flood labels (Section~\\ref{sec:landslide}).", ""),
    ("shared by the asset model, the terrain hazard model and the landslide model within a region.",
     "shared by the asset model and the terrain hazard model within a region."),
    ("for each flood region, and the landslide model for the Hill Tracts.", "for each flood region."),
    ("\\midrule\nChittagong Hill Tracts & \\CHTLandslides{} slides & \\CHTLSPositiveRate{}\\,\\% & \\multicolumn{2}{c}{landslide AUC \\CHTLSAUC{}, AP \\CHTLSAP{}} & -- & -- \\\\\n", ""),
    (" Landslide locations are from NASA's Cooperative Open Online Landslide Repository \\citep{juang2019}.", ""),
    ("The pipeline runs as a sequence of commands, each reading the previous step's files from the region's data directory, and this section follows that chain in order. Running a region end to end is \\texttt{python -m pipeline.cli -c configs/<region>.yaml run}, and each step can also be run alone. Figure~\\ref{fig:chain} shows",
     "The pipeline runs as a chain of steps, each reading the previous step's outputs, and this section follows that chain in order. Figure~\\ref{fig:chain} shows"),
    ("Table~\\ref{tab:data} lists every dataset the pipeline ingests, and licence terms decided two of the choices. Administrative",
     "Table~\\ref{tab:data} lists every dataset the pipeline ingests, all under open licences. Administrative"),
    (" Tile providers require visible attribution, which the dashboard shows on every map, where version 1 had suppressed it.", ""),
    ("Version 1 queried by place name (``Rangpur Division, Bangladesh''), which returned everything in the division whether or not it lay inside the study area, and which for the hill-tracts configuration would have pulled in Chattogram city and Cox's Bazar. The tiled query also avoids the timeouts that a whole division produced,",
     "Tiling avoids the timeouts that a query for a whole division produces."),
    (" and a query refused by the Overpass server, which allows two connections per address, is retried with a growing pause, since the public mirrors tested either hung or refused connections.", ""),
    ("Flow accumulation and HAND were the slowest steps of the chain in version 1, because each was written as a per-cell Python loop, which for the Rangpur and Rajshahi elevation model meant 80 million iterations. Both are now vectorised, with the sequential part of flow accumulation compiled with numba, and produce output identical to the original implementation. The Hill Tracts elevation model now preprocesses in about two minutes, where the earlier implementation had not finished after forty minutes.",
     ""),
    ("This was the only source in version 1, and it is kept so the two can be compared. ", ""),
    ("Sampling reprojects the query coordinates into each raster's coordinate system before reading it, and the absence of that step was the central error of version 1. Longitude and latitude pairs were passed to rasters stored in UTM, every query fell outside the raster's extent, and every sampled value came back as the nodata fill of zero, which left the five terrain features constant across all assets and every label at zero. The pipeline now stops",
     "Sampling reprojects the query coordinates into each raster's coordinate system before reading it, and the pipeline stops"),
    ("Version 1 compared the 5\\,km limit against distances in degrees, so it never removed an edge. ", ""),
    (", and it was the only surface in version 1.", "."),
    ("Version 1 filled those with a constant of 0.5, which together with an unsampled population term made 55\\,\\% of the vulnerability score a constant. Components without data are now excluded",
     "Components without data are excluded rather than filled with a constant,"),
    ("Version 1 took the raw product and divided by its maximum, which pushed the median cell to 0.0002 against a high-risk threshold of 0.7, so no cell could ever be flagged. ", ""),
    ("Version 1 filled them with zero, which the dashboard rendered as a green low-risk badge, and in Rangpur and Rajshahi that applied to \\RRNUnions{} minus \\RRNUnionsScored{} of the \\RRNUnions{} unions.",
     "In Rangpur and Rajshahi \\RRNUnionsScored{} of the \\RRNUnions{} unions hold mapped assets and carry a score."),
    (" The model of version 1 is not among these, since its labels were all zero (Section~\\ref{sec:corrections}) and it had nothing to learn from.", ""),
    ("The independent test of the flood models is agreement with flooding that was actually observed. The pipeline maps",
     "Flood extents observed by radar supply both the training labels and the reference against which the flood models are scored. The pipeline maps"),
    ("the graph network the system was built around is set against",
     "a graph neural network, the design this system began with, is set against"),
    ("Gradient boosting is consequently the default model of the system,",
     "Gradient boosting is consequently the default model,"),
    ("\\section{Data}\n", "\\section{Study area and data}\n"),
    # The radar caveats are in the discussion; the paper states them once.
    (" Two caveats apply to the radar extents themselves. Inundated paddy is open water to a radar and is counted as flooding, which is one reason the two-event threshold is preferred where the data allow it. And the minimum-backscatter composite over a window of two weeks records the largest extent reached at any pass in the window, not the extent on any one day.", ""),
    ("The original system was developed by Team Fermium, Shoumik Zubyer, Md Rakib Hasan and Fazla Zawadul Arabi, for the RIMES Mapathon, ResilienceAI track.",
     "The system was first developed for the RIMES Mapathon, ResilienceAI track."),
]
for old, new in EDITS:
    n = tex.count(old)
    assert n == 1, (n, old[:70])
    tex = tex.replace(old, new)
tex = tex.replace("this document", "this paper").replace("This document", "This paper")

out = DOCS / "paper" / "paper.tex"
out.parent.mkdir(exist_ok=True)
out.write_text(tex)
print("wrote", out, len(tex.split()), "words")

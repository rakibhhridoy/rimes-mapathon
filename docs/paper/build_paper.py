"""Assemble docs/paper/paper.tex from the technical document plus new framing text."""
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
SRC = (DOCS / "sgmdi_technical_document.tex").read_text().split("\n")
NEW = Path(__file__).resolve().parent / "sections"


def L(a, b, starts=None):
    """Source lines a..b inclusive (1-based); optionally assert how line a begins."""
    if starts:
        assert SRC[a - 1].startswith(starts), (a, SRC[a - 1][:60])
    return "\n".join(SRC[a - 1:b])


def N(name):
    return (NEW / f"{name}.tex").read_text().rstrip() + "\n"


parts = [
    L(1, 45, r"\documentclass"),
    N("preamble"),
    L(65, 78, "% ── Code listings"),
    r"\input{../numbers.tex}",
    "",
    r"\begin{document}",
    "",
    N("title"),
    N("introduction"),
    L(126, 184, r"\section{Related work}"),
    L(189, 241, r"\section{Data}"),
    L(243, 253, r"\section{Methods}"),
    L(255, 267, r"\subsection{Infrastructure}"),
    r"\subsection{Flood extents from Sentinel-1}",
    L(349, 388, r"\label{sec:observed}"),
    "",
    L(268, 311, r"\subsection{Features}"),
    r"\subsection{Validation design}",
    r"\label{sec:design}",
    "",
    L(324, 324, "Every model in the chain"),
    "",
    L(390, 390, "The validation step then"),
    "",
    N("design_extra"),
    L(312, 318, r"\section{Landslide susceptibility}"),
    r"\section{Results}",
    r"\label{sec:validation}",
    "",
    r"\subsection{Held-out results}",
    L(327, 346),
    "",
    r"\subsection{Observed extents against terrain labels}",
    r"\label{sec:labelresult}",
    "",
    L(392, 408, "The comparison of the proxy labels"),
    "",
    r"\subsection{The south-west coast}",
    "",
    L(410, 410, "The south-west coast"),
    "",
    N("temporal_result"),
    L(414, 456, r"\subsection{Model comparison}"),
    N("discussion"),
    N("backmatter"),
    L(516, 523, r"\section*{Acknowledgements}"),
]
tex = "\n".join(parts) + "\n"

# Edits that make the reused text read as a paper rather than a versioned report.
EDITS = [
    ("The pipeline runs as a sequence of commands, each reading the previous step's files from the region's data directory, and this section follows that chain in order. Running a region end to end is \\texttt{python -m pipeline.cli -c configs/<region>.yaml run}, and each step can also be run alone. Figure~\\ref{fig:chain} shows",
     "The pipeline runs as a chain of steps, each reading the previous step's outputs, and this section follows that chain in order. Figure~\\ref{fig:chain} shows"),
    ("Table~\\ref{tab:data} lists every dataset the pipeline ingests, and licence terms decided two of the choices. Administrative",
     "Table~\\ref{tab:data} lists every dataset the pipeline ingests, all under open licences. Administrative"),
    (" Tile providers require visible attribution, which the dashboard shows on every map, where version 1 had suppressed it.", ""),
    ("Version 1 queried by place name (``Rangpur Division, Bangladesh''), which returned everything in the division whether or not it lay inside the study area, and which for the hill-tracts configuration would have pulled in Chattogram city and Cox's Bazar. The tiled query also avoids the timeouts that a whole division produced,",
     "Tiling avoids the timeouts that a query for a whole division produces,"),
    ("Flow accumulation and HAND were the slowest steps of the chain in version 1, because each was written as a per-cell Python loop, which for the Rangpur and Rajshahi elevation model meant 80 million iterations. Both are now vectorised, with the sequential part of flow accumulation compiled with numba, and produce output identical to the original implementation. The Hill Tracts elevation model now preprocesses in about two minutes, where the earlier implementation had not finished after forty minutes.",
     "Both are vectorised, with the sequential part of flow accumulation compiled with numba."),
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
    ("Version 1 applied a logistic curve to slope alone, centred on 20$^\\circ$ with a scale of 5$^\\circ$, that had not been fitted to any data. Where upazila boundaries were unavailable it also sliced the raster into horizontal bands, labelled each with the name of a real upazila and drew a population for it from a random number generator, and neither survives. Per-upazila statistics now come from",
     "Per-upazila statistics come from"),
    (" The model of version 1 is not among these, since its labels were all zero (Section~\\ref{sec:corrections}) and it had nothing to learn from.", ""),
    ("The independent test of the flood models is agreement with flooding that was actually observed. The pipeline maps",
     "Flood extents observed by radar supply both the training labels and the reference against which the flood models are scored. The pipeline maps"),
    ("The claims below were checked against each study's full text, except where Table~\\ref{tab:litflood} marks otherwise.",
     "The claims below were checked against each study's full text, except where Table~\\ref{tab:litflood} marks otherwise. \\pending{read rahman2019, hasan2023 and islam2025review in full, which need library access.}"),
    ("the graph network the system was built around is set against",
     "a graph neural network, the design this system began with, is set against"),
    ("Gradient boosting is consequently the default model of the system,",
     "Gradient boosting is consequently the default model,"),
    ("\\section{Data}\n", "\\section{Study area and data}\n"),
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

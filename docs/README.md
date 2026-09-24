# Documentation

**`sgmdi_technical_document.pdf`** is the current technical document (version 2.0, a two-column article,
September 2026). It describes the system as it runs: data sources and licences,
the processing chain, the flood and landslide models, validation design and
results, the public dashboard, deployment, limitations, and a list of
corrections relative to version 1.

## Rebuilding it

Every number in the document is read from pipeline output. Regenerate the
figures, then compile:

```bash
python docs/build_numbers.py        # writes docs/numbers.tex from data/*/output
cd docs && latexmk -pdf sgmdi_technical_document.tex
```

A figure the pipeline has not produced renders as a visible `[TODO: missing]`
marker in the PDF. That is deliberate: a gap is better than a number carried
over from an earlier run.

## Contents of this folder

| File | Purpose |
|---|---|
| `sgmdi_technical_document.tex` / `.pdf` | the technical document |
| `numbers.tex` | generated macros, one per reported figure (do not edit) |
| `build_numbers.py` | generator for `numbers.tex` |
| `architecture_diagram.tex` / `.png` / `.svg` | pipeline diagram (predates version 2 and shows the March 2026 chain) |
| `paper/` | journal version of the document: `build_paper.py` assembles `paper.tex` from the technical document plus the new sections in `paper/sections/`; rerun it after editing either, then compile in `docs/paper/`. Red `[PENDING: …]` notes mark what remains before submission |
| `temporal_holdout.tex` | the temporal hold-out section, shared by both documents |
| `archive_v1/` | the March 2026 markdown docs and LaTeX source, kept for the record; they describe a system that no longer exists and contain figures that were later found to be wrong |

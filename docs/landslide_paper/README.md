# Landslide article: findings so far and plan

The landslide model of the Chittagong Hill Tracts was taken out of the flood
paper on 2026-09-26 (after commit `419eadb`), so that paper can make one
argument about one hazard. This folder keeps everything the model has shown
so far and the plan for turning it into its own article.

**It is not an article yet.** A single-storm inventory, a logistic regression
on five terrain features and an AUC lower than any published figure would be
an easy rejection on their own. The article is worth writing only with the
validation design below.

## Contents

| File | What it holds |
|---|---|
| `README.md` | this summary: results, literature, weaknesses, plan |
| `from_flood_paper.tex` | the landslide passages of the flood paper, verbatim, as they stood at `419eadb`, plus every other line that mentioned landslides |
| `references.bib` | the landslide and validation references already cited, all checked against full texts (see `docs/litreview/fulltext_checks.md`) |

The code and outputs stay where they are: `pipeline/landslide.py`,
`configs/cht.yaml`, and `data/cht/output/` (`landslide_model.json`,
`landslide_susceptibility.tif`, `landslide_upazila.json`). The technical
document (`docs/sgmdi_technical_document.tex`) still describes the model in
full. Run it with `python -m pipeline.cli --config configs/cht.yaml landslide`.

## What the current model shows

From `data/cht/output/landslide_model.json`, run of 2026-09-24:

| Item | Value |
|---|---|
| Inventory | NASA COOLR (Juang et al. 2019), 1,810 points, **all from one storm** (6 August 2023, mapped from PlanetScope) |
| Background points | 3,620 (twice the landslides), random, from the convex hull of the inventory buffered by 5 km, at least 500 m from any landslide |
| Model | class-weighted logistic regression on elevation, slope, TWI, HAND and flow accumulation (SRTM 30 m) |
| Validation | the flood models' 10 km spatial blocks, one assignment (seed 42) |
| Held-out AUC | **0.729** |
| Average precision | 0.390, against a test-block positive rate of 16.7 % (lift about 2.3) |
| Train / test points | 4,767 / 663 |
| Standardised coefficients | TWI −0.862, HAND +0.291, flow accumulation +0.285, elevation −0.217, slope −0.072 |

Upazila means of susceptibility are in `landslide_upazila.json`; the highest
are Kaptai (0.488), Alikadam (0.486) and Kawkhali (0.476).

Two things to understand before building on these numbers:

- **The test blocks hold 16.7 % landslides, against 33 % overall** (1,810 of
  5,430 points). The landslides cluster and the background points spread over
  the whole buffered hull, so the held-out blocks may be mostly background.
  Check the per-block balance and report the test prevalence with every
  score.
- **The coefficients cannot be read physically.** The five predictors are
  strongly collinear (TWI is computed from slope and flow accumulation), so
  the near-zero slope coefficient does not mean slope is unimportant. In the
  flood models, permutation importance found elevation carrying almost all the
  terrain signal; the same analysis should be run here.

## Where it sits against the literature

All checked against full texts on 2026-09-24:

| Study | Area | Inventory | Split | Reported accuracy |
|---|---|---|---|---|
| Hossain et al. 2025 | Hill Tracts | Rabby & Li, all 730 | not described | RF AUC 0.93; 78 % of the region high or very high; overall accuracy 0.98 |
| Roy et al. 2025 | Chittagong Hill Districts | 170 points from Rabby & Li | not recorded in the check; recheck | GBM and RF both AUC 0.83 |
| Islam et al. 2025 | Chattogram development area | 193 landslide, 165 non-landslide | random 70:30 | six classifiers > 90 % accuracy; LR–naive Bayes hybrid best; CMIP6 projections |
| Hasan et al. 2024 | Khagrachhari | 71 landslide, 56 non-landslide | random 70:30 | AUC BRT 0.95, RF 0.91, KNN 0.86 |
| **This model** | Hill Tracts | COOLR 1,810, one storm | **10 km blocks** | **AUC 0.729** |

None of the studies that state their split holds out areas. Inventory quality
limits all of them: positional error distorts both the fitted relationships
and the validation (Steger et al. 2016), and inventories biased towards
landslides that hit infrastructure give models that look good while encoding
the bias (Steger et al. 2021). Guidance on pseudo-absences stresses where
background points are drawn as much as how many (Barbet-Massin et al. 2012).
Reichenbach et al. (2018) found, over 565 studies, that uncertainty is rarely
evaluated and top-quality assessments are rare.

## Weaknesses a reviewer would raise now

1. One storm: the map shows which slopes failed in August 2023, not where
   landslides can happen.
2. The AUC is the lowest in the literature, and part of the gap is unexplained
   (stricter test, single storm, weak model, or the block prevalence above).
3. Five collinear terrain features only: no rainfall, lithology, land cover,
   distance to roads or streams, or aspect, which every comparable study uses.
4. One block assignment, so no spread and no paired tests.
5. A convex-hull background domain that still contains unmapped ground.

## The article worth writing

The same argument as the flood paper, carried to landslides: **what a
landslide susceptibility score means depends on how the test is separated from
the training data, and the published Hill Tracts figures all come from random
splits.**

1. **Get the multi-year inventory.** Rabby & Li (2019): 730 landslides,
   January 2001 to March 2017, from Google Earth, field mapping and literature.
   *First task: find out whether it can be downloaded or must be requested
   from the authors. Everything below depends on it.*
2. **Temporal test.** Train on 2001–2017 (Rabby & Li) and score on the
   August 2023 storm (COOLR), which is independent in time. This is the
   landslide counterpart of the 2024 flood test.
3. **Random against blocked against temporal.** Score the same model under a
   random 70:30 split, 10 km blocks and the temporal test, over 20 assignments
   with paired tests, to measure how much a random split inflates the AUC.
   This is the result that would carry the paper.
4. **Background sampling.** Compare background drawn from the whole region,
   the buffered hull and a tighter footprint, since this alone can move the
   score (Barbet-Massin et al. 2012; Steger et al. 2021).
5. **Predictors.** Add the usual set (lithology, land cover, distance to
   streams and roads, aspect, curvature, rainfall from GPM IMERG), then
   permutation importance with grouped features as in the flood paper.
6. **Models.** Logistic regression, random forest and gradient boosting on
   identical splits, with an area-of-applicability check (Meyer and Pebesma
   2021) for where the training data support a prediction.
7. **Rainfall.** Thresholds of the kind LHASA uses (Kirschbaum and Stanley
   2018) would be needed before the map could support warnings, and are out
   of scope unless the data allow it.

Candidate venues: *Landslides* (Springer), *NHESS*, or *Geomatics, Natural
Hazards and Risk*. The two articles share the pipeline and the validation
design, so each should cite the other and state what is new, to avoid any
appearance of splitting one study in two.

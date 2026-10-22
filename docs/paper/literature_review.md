# Critical literature review: novelty of the flood paper

*22 October 2026. Prompted by a coworker's comment that the paper is "not novel enough
for any journal". Covers four strands beyond Bangladesh flood susceptibility.*

## How the screen was done

- **OpenAlex:** 29 queries across four strands, 607 unique records from 2000 onwards,
  ranked by keywords on temporal tests, record baselines and reference models.
- **Web searches:** 14 targeted searches on the questions where a pre-empting study
  would most likely sit.
- **Earlier screens:** 6 and 9 October, about 540 OpenAlex records, reported in
  commits e850e50 and fa51cb7.
- **Not searched:** Scopus, Web of Science and Google Scholar, which are not available.
- **Evidence levels:** each claim below is marked as read in **full text**, read in the
  **abstract**, or seen only in a **search snippet**. Snippet-only claims must be checked
  before they are cited.

## 1. Verdict on the coworker's comment

No study found scores a flood susceptibility model against the record of earlier floods
on a later flood.

- **The closest designs** test a map on later floods without the record as a reference:
  - Swarup & Nury 2026, already cited.
  - Kumar et al. 2026 on Punjab. This one runs in reverse: trained on 2025 and tested
    on 2023.
- **Other designs** use the record only as labels or as a feature.
- **Feizbahr et al. 2025** (*Remote Sensing* 17:3471, full text) reports a
  "frequency-based" model scoring worst (AUC 0.65). That model is the frequency-ratio
  method over factor classes on a random split, not the flood record, so it does not
  contradict the paper.

The gap is real. The risk lies elsewhere: the comparison is new to *flood
susceptibility*, but not to hazard prediction. Other fields learned this lesson long
ago, and the paper does not yet use that history. Used well, the history is the paper's
strongest support. Ignored, it lets a reviewer call the result obvious.

## 2. What each strand says

### 2.1 Earthquake forecasting: the closest precedent, not yet cited

- **Zechar & Jordan 2008** (*GJI* 172:715, abstract; open access) scored alarm
  functions with the Molchan diagram and an **area skill score relative to a reference
  model**. The reference was *relative intensity* (RI), which is smoothed historical
  seismicity alone. In the 10-year California experiment, "neither PI nor NSHM provide
  significant performance gain relative to the RI reference model".
  - PI is a pattern-recognition model. NSHM is the USGS National Seismic Hazard Map.
  - This is the paper's result in another hazard: a sophisticated model and an official
    hazard map did not beat the map of where events already happened.
- **Taroni et al. 2017** (*Ann. Geophys.*, abstract) found the CN algorithm in Italy no
  better than a stationary Poisson reference.
- **RI as a reference:** CSEP treats RI as a reference against which models must be
  tested (search snippets on Nanjo 2010/2011, *EPS*).

**Use:** this gives the paper a ready-made framework, with a Molchan curve and an area
skill score against a past-event reference. It also turns "obvious" into "a test every
mature hazard-forecasting field already requires".

### 2.2 Crime hotspot prediction

- **Chainey, Tompson & Uhlig 2008** (*Security Journal*; title and abstract metadata,
  method from snippets) built hotspots from Time-1 crimes and scored them on Time-2
  crimes.
  - Measures: the hit rate, and the Prediction Accuracy Index (PAI), which is the hit
    rate divided by the share of area flagged.
  - Kernel density of past events was the most accurate method.
- **Levine 2008** criticised the method that PAI relies on.
- **A 2020 arXiv paper** proposes the PAI curve over all area shares. That curve is the
  Molchan curve read the other way up.

**Use:** the paper's planning-list result is one point on such a curve. Reporting the
whole curve, with PAI, makes the planning claim complete and puts it in established
terms.

### 2.3 Landslides

- **Already cited:** Chung & Fabbri 2003 and Guzzetti 2006 (temporal validation is
  standard).
- **Jones et al. 2021** (*JGR Earth Surface*, abstract; open access) is new to the paper.
  - It fitted logistic regression models in Nepal for 12 single years and hindcast the
    other years.
  - "Models trained on extreme years [were] unable to consistently hindcast landslide
    occurrence in other years."
  - Models built from more than 6–8 years of inventory hindcast well.
  - **This bears directly on the paper's main limit, a single test year.** It shows that
    the test year matters, and it gives a reason to test several years.
- **A chrono-validation paper** (*Eng. Geol.* 2020, snippet) and spatiotemporal
  leave-one-out schemes for Taiwan (*JAG* 2023, snippet) show that rolling temporal tests
  are routine for landslides.
- **Not found:** a landslide study that uses past-landslide density as a standalone
  baseline. Landslide work uses the inventory as labels and validation data, not as a
  rival model.

### 2.4 Wildfire

- **Shmuel & Heifetz 2022** (*Forests* 13:1050, global, 0.25° monthly): historical
  burned area was the most influential feature, and removing it lowered AUC from 0.97 to
  0.94. **Snippet only; check the full text before citing.**
  - This matches the paper's "record as a feature" result.
  - Fire history appears as an input, never as a benchmark the model must beat.
- **A Korean FIRMS study** (*JAG* 2025, snippet) validated a susceptibility map on the
  next year's fire perimeters (AUC 0.78). It noted that most wildfire susceptibility
  models are never checked against later fires.

### 2.5 Forecast verification

- **Already cited:** Murphy 1988 and Pappenberger 2015.
- **Pappenberger 2015** (full abstract) found that the skill a forecast earns depends
  on which benchmark is chosen. It recommends the benchmark "toughest to beat"
  (meteorological persistence, for EFAS).
  - **Use:** the record is the toughest-to-beat benchmark for susceptibility. A model
    that beats only random or a logistic regression (as in Feizbahr 2025 and most
    studies in Table 1) has earned "naive skill".
- **Seibert 2001** (*Hydrol. Process.*, title only, not open access) is the usual
  citation for benchmark models in hydrology. Add it if the full text can be obtained.

### 2.6 Satellite flood records and hazard models

- **Already cited:** Hawker et al. 2020, comparing records with global models as hazard
  maps.
- **Bernhofen et al. 2018** (*ERL*, snippet) checked six global flood models against
  satellite flood extents in Nigeria and Mozambique, with CSI 0.45–0.70.
- **Mester et al. 2021** (*ERL*, snippet) did the same for 11 global hydrological models
  on 8 floods.
- **What these share:** each scores how well a model *reproduces* observed floods. None
  asks whether the earlier record *predicts* the next flood better than the model.
  **That question is the paper's contribution, and it extends to physics-based hazard
  maps.**
- **US case studies** (news and grey literature, snippets only):
  - About two-thirds of Harvey's flooding in Houston lay outside the FEMA 100-year zone.
  - About 34 % of Helene's flooding in Tampa lay outside it.
  - These are not citable as they stand, but they show that official maps miss observed
    floods.
- **Global susceptibility products:**
  - A 30 m global susceptibility map (*Sci. Data* 2026) reports a median AUC of 0.95 on
    cross-validation.
  - A study using GFD labels (Rahimi et al. 2026, *Sci. Rep.*) reports AUCs of
    0.98–0.99.
  - Neither tests against later floods or against the record, so the inflated-accuracy
    pattern of Table 1 holds globally.

### 2.7 Bangladesh, after 9 October

- **Rahman et al. 2024** (*Prog. Disaster Sci.*, abstract) mapped the 2022 north-east
  flood with Sentinel-1 and fitted RF, LR and DT (AUC 0.94–0.97), from a single event.
  - **It is missing from Table 1 and should be added after its split has been checked.**
- **Islam et al. 2025** (*Earth Sci. Inform.*, review of ML flood prediction in
  Bangladesh, gaps and challenges, not open access) is worth citing for the gap
  statement once read.
- **No new study found** that tests a Bangladesh susceptibility model on a later flood
  against the record.

## 3. Gaps in the paper that the literature exposes

These are ranked by how likely a reviewer is to raise them.

1. **One test year.** All evidence for the record's lead rests on 2024, and Jones et al.
   2021 shows that hindcast skill varies strongly by year. A reviewer can say that 2024
   was a small, in-bank riverine year, so of course the record won.
2. **One firm region.** Rangpur–Rajshahi is firm, Sylhet depends on the target, the
   coast is open and Jamuna east is probable. The paper explains the pattern by how
   often floods return, but it shows this only by contrast across three regions.
3. **Only one kind of model.** The record beats a terrain ML model. Hawker 2020,
   Bernhofen 2018 and Mester 2021 compare records with physics-based hazard maps. A
   reviewer can ask whether the record also beats a hydrodynamic hazard map, the kind
   agencies actually use.
4. **Only the paper's own record.** The recommendation ("report the record as a
   baseline") assumes a processed Sentinel-1 archive. If a free global record (GFD,
   JRC Global Surface Water) works almost as well, the recommendation applies anywhere.
   If it does not, that is worth knowing too.
5. **Non-standard skill measure.** "Skill over the record" is defined in the SI from
   AUC. The earthquake and crime literature has standard tools for exactly this
   comparison: the Molchan diagram, the area skill score and the PAI curve. Not using
   them makes the measure look improvised.
6. **Missing precedent.** Zechar & Jordan 2008, the crime-hotspot work and Jones 2021
   are not cited. Without them, the "obvious" objection has no answer from the
   literature.
7. **A diffuse paper.** The graph network, Kriging and the composite risk index take
   space from the one argument that is new.

## 4. Enhancements, ranked by value for cost

| # | Enhancement | Answers gap | Cost | Data in hand? |
|---|---|---|---|---|
| E1 | **Rolling-origin temporal test**: every event year tested from the earlier ones only | 1, 2 | CPU, hours | yes |
| E2 | **Recurrence explains the lead**: pool all region–year tests and relate the record's lead to the share of newly flooded ground | 2 | cheap after E1 | yes |
| E3 | **More regions**: extend E1 to the built national regions with multi-year events, as validation regions | 2 | CPU, hours | yes |
| E4 | **Physics-based hazard map as a third contender**: JRC/GloFAS global river flood hazard maps (90 m, CC BY 4.0) on the same assets and test floods | 3 | download plus scoring | to download |
| E5 | **Global record as baseline**: GFD (2000–2018) and JRC Global Surface Water occurrence as alternative records | 4 | small | JRC GSW already used in S5 |
| E6 | **Molchan diagram and area skill score** with the record as reference model; PAI at the planning-list share | 5 | small | yes |
| E7 | **Cite the cross-hazard precedent** (Zechar & Jordan 2008, Chainey 2008, Jones 2021, Pappenberger 2015 benchmark choice) in the introduction and discussion | 6 | words only | yes |
| E8 | **Restructure**: graph network, Kriging and composite risk to the SI or a short section, so the paper makes one argument | 7 | editing | yes |
| E9 | A deep raster model on GPU (CNN or U-Net) as a stronger rival to the record | 3 (weakly) | GPU, days | yes |

### What each enhancement does

**E1 to E3 change the evidence most.**

- Event years available:
  - Rangpur–Rajshahi: 2017, 2019, 2020 and 2024.
  - Sylhet: 2017, 2022 and 2024.
  - Coast: 2020, 2021, 2022 and 2024.
  - Jamuna east: 2017, 2019, 2020 and 2024.
  - National central region: 2017, 2019 and 2020.
- Possible tests: 2019 and 2020 in the riverine regions, 2022 in Sylhet, and 2021 and
  2022 on the coast. That gives about 10–12 region–year tests instead of 4.
- **E2 then turns the qualitative explanation into a quantitative relationship:** the
  record's lead against the share of test-flooded assets on ground no earlier flood
  reached. If the relationship holds across years and regions, the paper's thesis
  becomes a rule that predicts where a model is worth building. Nothing found in any
  strand does this.
- **E4 broadens the claim** from "terrain ML models" to "hazard maps".
  - If the record also beats a hydrodynamic 100-year map on the 2024 floods, the result
    concerns how all flood hazard maps should be validated, which fits the Journal of
    Flood Risk Management closely.
  - If the hazard map wins, the paper gains a fair rival and a sharper limit.
- **E6 and E7 are cheap.** They place the result inside an established verification
  framework, which is the most direct answer to "not novel".

### Word budget

The paper is at 5,993 of 6,000 words.

- **E1 to E3:** fit if the region-by-region text of Results 5.4 becomes one figure (lead
  against recurrence) and one table, with per-region detail in the SI.
- **E8:** frees roughly 300–450 words. The model comparison, the Kriging sentences and
  most of the composite-risk method and implications would go.
- **E4 and E6:** each needs about 100–150 words and one panel.

## 5. Decisions needed from the author

1. Whether to remove the graph network, Kriging and composite risk from the main text
   (E8), since this frees the words E1 to E4 need.
2. Whether the national regions may join as validation regions (E3). The standing rule
   is not to change the three study regions before acceptance, and validation regions
   leave them untouched.
3. Whether to add the JRC hazard maps (E4). These are CC BY 4.0, compatible with the
   Zenodo archive, unlike FABDEM.
4. Whether E9 is worth the GPU time. The graph-network null already covers "a more
   complex model", so E9 is the lowest priority.
5. Whether to change the title if E1 to E4 succeed, for example to name the comparison
   with hazard maps.

## Sources

Read in full text:

- Feizbahr et al. 2025, *Remote Sensing* 17:3471, https://doi.org/10.3390/rs17203471

Read in abstract:

- Zechar & Jordan 2008, *GJI* 172:715, https://doi.org/10.1111/j.1365-246x.2007.03676.x
- Taroni et al. 2017, *Annals of Geophysics*, https://doi.org/10.4401/ag-6889
- Jones et al. 2021, *JGR Earth Surface*, https://doi.org/10.1029/2021JF006067
- Kumar et al. 2026, *Remote Sensing* 18:3204, https://doi.org/10.3390/rs18183204
- Pappenberger et al. 2015, *J. Hydrol.* 522:697, https://doi.org/10.1016/j.jhydrol.2015.01.024
- Rahman et al. 2024, *Prog. Disaster Sci.*, https://doi.org/10.1016/j.pdisas.2024.100402
- Rahimi et al. 2026, *Sci. Rep.*, https://doi.org/10.1038/s41598-026-41014-3
- Shmuel & Heifetz 2022, *Forests* 13:1050, https://doi.org/10.3390/f13071050

Seen in metadata or search snippets only (check before citing):

- Chainey, Tompson & Uhlig 2008, https://doi.org/10.1057/palgrave.sj.8350066
- Levine 2008, https://doi.org/10.1057/sj.2008.5
- Islam et al. 2025, https://doi.org/10.1007/s12145-025-01816-x
- Bernhofen et al. 2018, *ERL*: https://eprints.whiterose.ac.uk/135552
- Mester et al. 2021, *ERL* 16:094010: https://publications.pik-potsdam.de/pubman/item/item_25919_1/component/file_25920/Mester_2021_Environ._Res._Lett._16_094010.pdf
- Global 30 m flood susceptibility dataset 2026: https://www.nature.com/articles/s41597-026-08338-1
- Korean wildfire FIRMS study 2025: https://www.sciencedirect.com/science/article/pii/S1569843225005370
- Landslide chrono-validation 2020: https://www.sciencedirect.com/science/article/pii/S0013795220317154
- Taiwan landslide spatiotemporal model 2023: https://www.sciencedirect.com/science/article/pii/S1569843223004557
- JRC Global River Flood Hazard Maps v2.1: https://developers.google.com/earth-engine/datasets/catalog/JRC_CEMS_GLOFAS_FloodHazard_v2_1
- Houston Harvey flooding outside FEMA zones (news): https://www.npr.org/sections/thetwo-way/2017/09/01/547992600/early-data-from-harvey-shows-epic-flooding
- Tampa Helene flooding outside FEMA zones (news): https://www.nationalmortgagenews.com/articles/federal-flood-maps-are-no-match-for-floridas-double-hurricane

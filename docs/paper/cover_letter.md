Md Rakib Hasan
Department of Soil, Water and Environment, University of Dhaka, Dhaka-1000, Bangladesh
Fermium Systems, Dhaka-1207, Bangladesh
rakibhhridoy@fermium.systems · ORCID 0009-0002-4007-7590

23 October 2026

The Editor
Journal of Flood Risk Management

Dear Editor,

I am pleased to submit the manuscript "Do flood susceptibility models know more than the flood record? Evidence from radar-observed floods in Bangladesh" for consideration as an Original Paper in the Journal of Flood Risk Management.

Flood susceptibility maps are widely produced to guide where flood risk should be managed, and in Bangladesh the studies closest to this one report areas under the ROC curve (AUC) of 0.87 to 0.98. Weather and river forecasts are routinely scored against persistence, yet those figures are rarely set against its equivalent, which a flood manager already holds, the record of where floods have been, and they usually come from random splits of spatially clustered points, which test a model beside its own training data. The manuscript tests both points in three regions with contrasting flood regimes, using flood extents mapped from Sentinel-1 radar for thirteen events and scoring every model on 10 km blocks held out from training over twenty block assignments.

The main findings are these.

- Predicting each flood year from the years before it, in the three study regions and four validation regions that share no asset with them, gives 15 tests. The record's lead over a gradient-boosted terrain model fell as the share of a flood reaching ground no earlier flood had reached rose (Spearman ρ = −0.775, p < 0.001), whatever the length of the record, and the two were level at about 33 % of flooded assets on such new ground (95 % interval 18–41 %). The comparison therefore has an answer that depends on the flood regime rather than a single verdict.
- In Rangpur and Rajshahi, where few of the 2024 floods' assets stood on new ground, the record ranked the flooding at 0.896 against 0.801 for the model and kept its lead under every check the manuscript applies, including masks re-mapped from a single satellite and wider validation blocks. On the cyclone-affected coast, where most flooding reached new ground, the model led. Given the record as a feature, the model beat both of its parts in 5 of 8 tests, all riverine.
- A global hydrodynamic hazard map (JRC, LISFLOOD-FP) was no stronger a rival than the model, trailing the record significantly in 5 of 15 tests and leading it in 4, all on floods that reached much new ground. The comparison thus bears on hazard maps generally, as it has long done in earthquake forecasting, where maps of past seismicity remain the reference that forecasts must beat.
- Training on radar-observed extents rather than terrain-threshold labels raised the AUC by 0.080 to 0.101 on held-out areas, and by 0.024 to 0.071 against the 2024 floods, which neither label source had seen.

I believe the paper suits the Journal because its conclusions concern how flood risk is managed as much as how it is modelled. It recommends that agencies publish the radar flood record as a planning baseline wherever one exists, that the choice between that record and a model follow how much of past floods fell on new ground, and that maps used to site shelters, embankments or schools be judged on held-out areas and later floods. It also engages the Journal's own work on flood hazard mapping in Bangladesh, radar flood mapping, infrastructure exposure and vulnerability.

The manuscript states its limits plainly. The radar flood maps agree only moderately with an independent MODIS flood map (critical success index 0.29 and 0.34), the coastal hazard surface is weak, and the composite risk has not yet been checked against losses. Every number in the text is generated directly from the pipeline outputs, the code is openly available at https://github.com/rakibhhridoy/rimes-mapathon, and the input data and outputs are archived at https://doi.org/10.5281/zenodo.23573843. A Supporting Information file gives the full pipeline settings and a sensitivity analysis of the risk weights.

The main text runs to just under 6,000 words, excluding references, figures and tables, with 6 figures and 7 tables. The manuscript is original, has not been published, and is not under consideration elsewhere. The open-source software used here is described in a separate manuscript under review at SoftwareX (SOFTX-D-26-01375), which reports the software and its validation design and applies it across the whole country; this manuscript reports the scientific findings for the three study regions. The software paper repeats a few of the validation figures given here, as examples of what the software produces, but does not analyse them. I am its sole author and approve its submission. I declare an affiliation with Fermium Systems, which hosts the public dashboard described in the paper. The software is released under the MIT licence and the data under open licences, and no result depends on a proprietary component. The work received no specific funding.

Suggested reviewers:

- Dr Beth Tellman, Nelson Institute for Environmental Studies, University of Wisconsin–Madison, USA. Beth.tellman@wisc.edu
- Dr Sandro Martinis, German Remote Sensing Data Center, German Aerospace Center (DLR), Germany. Sandro.Martinis@dlr.de
- Prof. Hanna Meyer, Institute of Landscape Ecology, University of Münster, Germany. hanna.meyer@uni-muenster.de
- Dr Alexandre Wadoux, James Cook University, Australia. alexandre.wadoux@yahoo.fr

Thank you for considering the manuscript.

Yours sincerely,

Md Rakib Hasan

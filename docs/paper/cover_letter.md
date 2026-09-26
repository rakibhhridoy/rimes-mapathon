Md Rakib Hasan
Department of Soil, Water and Environment, University of Dhaka, Dhaka-1000, Bangladesh
Fermium Systems, Dhaka-1207, Bangladesh
rakibhhridoy@fermium.systems · ORCID 0009-0002-4007-7590

26 September 2026

The Editor
Journal of Flood Risk Management

Dear Editor,

I am pleased to submit the manuscript "Do flood susceptibility models know more than the flood record? Evidence from radar-observed floods in Bangladesh" for consideration as a Research Article in the Journal of Flood Risk Management.

Flood susceptibility maps are widely produced to guide where flood risk should be managed, and in Bangladesh the studies closest to this one report areas under the ROC curve (AUC) of 0.87 to 0.98. Those figures are rarely set against the simplest alternative a flood manager already holds, the record of where floods have been, and they usually come from random splits of spatially clustered points, which test a model beside its own training data. The manuscript tests both points in three regions with contrasting flood regimes, using flood extents mapped from Sentinel-1 radar for thirteen events and scoring every model on 10 km blocks held out from training over twenty block assignments.

The main findings are these.

- Trained on floods up to 2022, a gradient-boosted model ranked the ground the 2024 floods reached at 0.815 and 0.825 in the two riverine regions, below the radar record of earlier flooding at 0.905 and 0.868. Given that record as a feature, the model beat both, at 0.928 and 0.921, because terrain still ranks the ground no earlier flood reached. On the cyclone-affected coast, where most flooded assets stood on ground that had not flooded before, the model beat the record.
- Training on radar-observed extents rather than terrain-threshold labels raised the AUC by 0.064 to 0.125 on held-out areas, and by 0.042 to 0.123 against the 2024 floods, which neither label source had seen.
- A graph neural network linking neighbouring assets added nothing over ordinary tabular models, and elevation carried almost all of the terrain signal, while the topographic wetness index and height above nearest drainage added almost nothing.

I believe the paper suits the Journal because its conclusions concern how flood risk is managed as much as how it is modelled. It recommends that agencies publish the radar flood record as a planning baseline wherever one exists, that maps used to site shelters, embankments or schools be judged on held-out areas and later floods, and that areas without mapped infrastructure be treated as missing data rather than as safe. It also engages the Journal's own work on flood hazard mapping in Bangladesh, radar flood mapping, infrastructure exposure and vulnerability.

The manuscript states its limits plainly. The radar flood maps agree only moderately with an independent MODIS flood map (critical success index 0.29 and 0.34), the coastal hazard surface is close to chance, and the composite risk has not yet been checked against losses. Every number in the text is generated directly from the pipeline outputs, the code is openly available at https://github.com/rakibhhridoy/rimes-mapathon, and the input data and outputs are archived at https://doi.org/10.5281/zenodo.22978729. A Supporting Information file gives the full pipeline settings and a sensitivity analysis of the risk weights.

The main text runs to about 4,700 words, excluding references, figures and tables, with 8 figures and 7 tables. The manuscript is original, has not been published, and is not under consideration elsewhere. I am its sole author and approve its submission. I declare an affiliation with Fermium Systems, which hosts the public dashboard described in the paper. The software is released under the MIT licence and the data under open licences, and no result depends on a proprietary component. The work received no specific funding.

[Optional: suggested reviewers, with name, affiliation, email and a one-line reason each, and any reviewers to exclude.]

Thank you for considering the manuscript.

Yours sincerely,

Md Rakib Hasan

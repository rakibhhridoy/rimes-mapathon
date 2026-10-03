Md Rakib Hasan
Department of Soil, Water and Environment, University of Dhaka, Dhaka-1000, Bangladesh
Fermium Systems, Dhaka-1207, Bangladesh
rakibhhridoy@fermium.systems · ORCID 0009-0002-4007-7590

3 October 2026

The Editors-in-Chief
SoftwareX

Dear Editors,

We are pleased to submit "Fermium Hazard Mapper: an open pipeline for flood and landslide risk mapping, trained and tested on radar-observed floods" for consideration as an Original Software Publication in SoftwareX.

Flood susceptibility maps for Bangladesh are usually trained on a few hundred surveyed points and scored on a random share of the same points, so their accuracy says little about ground the model has not seen. Fermium Hazard Mapper is an open Python pipeline and web application that trains flood models on flood extents mapped from Sentinel-1 radar, scores every model on whole areas held out from training, and publishes risk for each mapped asset and administrative unit. One configuration file drives each region through a chain of command-line stages, from data download to calibrated scores, composite risk and a validation suite that compares models, label sources and time periods on identical held-out blocks.

The article describes the software and the evidence it reports.

- A national partition of eight flood regions, each holding one flood regime, covers all 61 lowland districts of Bangladesh and 114,569 mapped assets; a landslide model covers the Chittagong Hill Tracts.
- Over 20 assignments of held-out 10 km blocks, the default model reaches an AUC between 0.724 and 0.914 across the regions, and training on radar-observed floods beats training on terrain-threshold labels in every region. The graph neural network the system was first built around trails the best ordinary model in every region, by 0.005 to 0.088 in AUC, so the comparison stays in the pipeline for any future model to face.
- The web application serves the results from the division down to the union, beside the government's official scenario maps, at https://fermium.systems/hazmapper, without a sign-in.

We believe the software suits SoftwareX because it is built to be rerun and checked rather than only read. The code is released under the MIT licence at https://github.com/rakibhhridoy/rimes-mapathon, the described release is archived at https://doi.org/10.5281/zenodo.23108192, and the input data and outputs of the validation regions are archived at https://doi.org/10.5281/zenodo.23108572. A single command reproduces a validation region from that archive without an Earth Engine account and checks every headline figure against the archived outputs; from a fresh clone it reproduced the Sylhet region exactly. A suite of 159 tests guards the pipeline, and the 132 that need no data run in continuous integration on every change. No number in the article is typed by hand, since a build script writes each one from the pipeline outputs. The first version of the system was one of 12 finalists among 60 university teams in the RIMES Mapathon of March 2026, and RIMES has since expressed interest in building on it.

The main text runs to about 2,350 words, excluding the metadata tables, references and captions, with three figures and one table, and it follows the SoftwareX article template. The manuscript is original, has not been published, and is not under consideration elsewhere. Both authors have approved its submission. Md Rakib Hasan declares an affiliation with Fermium Systems, which hosts the public dashboard; the software is free and open, and no result depends on a proprietary component. The work received no specific funding.

Thank you for considering the manuscript.

Yours sincerely,

Md Rakib Hasan, on behalf of both authors

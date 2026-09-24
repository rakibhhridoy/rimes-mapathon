# Full-text checks for the journal paper

The related-work claims were first taken from abstracts. Each study below must
be checked against its full text before submission. Put downloaded PDFs in
`docs/litreview/pdfs/<key>.pdf` (the folder is git-ignored, since most of
them may not be redistributed) and each one can then be checked claim by
claim.

Status: **verified** = checked against the full text, **open** = open access,
not yet read, **paywalled** = needs library access.

| Key | Status | Access | Claims the paper makes |
|---|---|---|---|
| adnan2023 | **verified** 2026-09-24 | [Brunel repository PDF](https://bura.brunel.ac.uk/bitstream/2438/29114/1/FullText.pdf) | about 60 % of land flood-prone under every model; pixel-wise r 0.62–0.91; *added from the full text:* 965 points sampled from a satellite flood-frequency map covering 1988–2012 (Sect. 2.2); 70:30 train/test split, not spatial (Sect. 2.5); RF AUC 0.984 (Sect. 3) |
| rahman2019 | paywalled | [10.1007/s41748-019-00123-y](https://doi.org/10.1007/s41748-019-00123-y) | national scale; LR–FR, ANN, AHP; 475 locations from remote sensing and a MIKE-11 model; random 70:30 split; AUROC 0.881 |
| talukdar2020 | paywalled | [10.1007/s00477-020-01862-5](https://doi.org/10.1007/s00477-020-01862-5) | Teesta basin; bagging ensembles; AUC 0.945; flood locations and split unknown |
| hasan2023 | paywalled | [10.1016/j.ocecoaman.2023.106503](https://doi.org/10.1016/j.ocecoaman.2023.106503) | coastal area; RF, XGBoost, KNN; accuracy 86.7 %; flood locations and split unknown |
| islam2025review | paywalled | [10.1007/s12145-025-01816-x](https://doi.org/10.1007/s12145-025-01816-x) | review of ML flood prediction in Bangladesh; 42 conditioning factors; led by precipitation, distance from rivers, elevation, land cover, soil type |
| hasanm2024 | paywalled | [10.1007/s11356-024-34949-5](https://doi.org/10.1007/s11356-024-34949-5) | Khagrachhari; compared RF, boosted regression trees, kNN |
| islam2021 | open | [10.1016/j.gsf.2020.09.006](https://doi.org/10.1016/j.gsf.2020.09.006) | Teesta basin; Dagging, RS, RF, ANN, SVM; 413 current and former flood points; all AUC > 0.80; split unknown |
| islamr2024 | open | [10.1016/j.envc.2023.100833](https://doi.org/10.1016/j.envc.2023.100833) | north-east, local, flash floods; RF AUC 0.964; RF and SVM agree on 73.3 % of the map; flood locations and split unknown |
| hossain2025 | open | [10.1007/s43621-025-02084-x](https://doi.org/10.1007/s43621-025-02084-x) | Hill Tracts; RF on 730 landslide events; AUC 0.93; 78 % of region high or very high susceptibility; overall accuracy 98 %; same count as rabby2019 |
| roy2025 | open | [10.1007/s44288-025-00337-w](https://doi.org/10.1007/s44288-025-00337-w) | Chittagong Division; gradient boosting and RF on 170 locations; AUC 0.83 |
| islama2025 | open | [10.1016/j.geogeo.2025.100354](https://doi.org/10.1016/j.geogeo.2025.100354) | Chattogram development area; hybrid classifiers; > 90 % accuracy under current and projected climate |
| rabby2019 | open | [10.3390/data5010004](https://doi.org/10.3390/data5010004) | 730 landslides; Chittagong Hilly Areas; January 2001 to March 2017; Google Earth interpretation, field mapping and literature |

For every study, also record what the full text says about the train/test
split, specifically whether any area was held out, because the paper's
argument rests on it.

Automated retrieval failed for every publisher on 2026-09-24. Springer
requires cookies, while ScienceDirect and MDPI return 403 to scripts. A
browser download works for all the open-access ones.

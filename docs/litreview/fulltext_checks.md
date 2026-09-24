# Full-text checks for the journal paper

The related-work claims were first taken from abstracts. Each study below must
be checked against its full text before submission. Put downloaded PDFs in
`docs/litreview/pdfs/<key>.pdf` (the folder is git-ignored, since most of
them may not be redistributed) and each one can then be checked claim by
claim.

Checked 2026-09-24. **verified** = the paper's claims match the full text, **corrected** = the text or Table 1 was changed to match it, **extended** = details added from it, **paywalled** = still needs library access.

| Key | Status | What the full text showed |
|---|---|---|
| adnan2023 | **verified** | ~60 % flood-prone and r 0.62–0.91 confirmed. 965 points from a satellite flood-frequency map, 1988–2012 (Sect. 2.2); 70:30 split, no area held out (Sect. 2.5); RF AUC 0.984 |
| talukdar2020 | **corrected** | "413 flood points" in the abstract is 207 flood + 206 non-flood; from historical inundation maps, topographic maps, local survey; random 80:20 (Sect. 3.2). BgM5P AUC 0.945 (Sect. 4.4), not stated whether training or validation |
| islam2021 | **corrected** | abstract says 413 flood points, but the methods use 167 flood + 167 non-flood from records, fieldwork, residents, Google Earth; random 80:20 (Sect. 2.2.1). Validation AUC 0.81–0.873, all > 0.80 confirmed |
| islamr2024 | **corrected** | inventory from **Sentinel-1 in GEE for the June 2022 flash flood**, one Sylhet sub-district; 1,500 flooded + 1,500 dry pixels, random 70:30; RF AUC 0.964 and 73.3 % RF–SVM agreement confirmed. A direct precedent, now acknowledged in the text |
| hossain2025 | **verified** | uses the Rabby & Li inventory itself (730 landslides); RF AUC 0.93, 78 % high/very high, overall accuracy 0.98 confirmed; split not described |
| roy2025 | **verified** | 170 points sampled from the Rabby & Li inventory; GBM and RF both AUC 0.83; area Chittagong Hill Districts in Chittagong Division |
| islama2025 | **corrected** | the six base classifiers each exceed 90 % accuracy and the LR–bNB hybrid is best; the climate scenarios project susceptibility, not accuracy. 193 landslide + 165 non-landslide points, random 70:30 |
| hasanm2024 | **extended** | Khagrachari; 71 landslide + 56 non-landslide points, random 70:30; AUC BRT 0.95, RF 0.91, KNN 0.86 |
| rabby2019 | **verified** | 730 landslides, January 2001 to March 2017, Google Earth, field mapping, literature |
| rahman2019 | paywalled | [10.1007/s41748-019-00123-y](https://doi.org/10.1007/s41748-019-00123-y): 475 locations from remote sensing and a MIKE-11 model; random 70:30; AUROC 0.881 (from abstract) |
| hasan2023 | paywalled | [10.1016/j.ocecoaman.2023.106503](https://doi.org/10.1016/j.ocecoaman.2023.106503): coastal; RF, XGBoost, KNN; accuracy 86.7 % (from abstract) |
| islam2025review | paywalled | [10.1007/s12145-025-01816-x](https://doi.org/10.1007/s12145-025-01816-x): 42 conditioning factors, led by precipitation, distance from rivers, elevation, land cover, soil type (from abstract) |

None of the flood studies read in full holds out areas, reports calibrated
probabilities or a Brier score, or compares against the record of earlier
flooding.

For every study, also record what the full text says about the train/test
split, specifically whether any area was held out, because the paper's
argument rests on it.

Automated retrieval failed for every publisher on 2026-09-24. Springer
requires cookies, while ScienceDirect and MDPI return 403 to scripts. A
browser download works for all the open-access ones.

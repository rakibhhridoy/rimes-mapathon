# Zenodo record: new version of 10.5281/zenodo.19233968

**Published 2026-09-26 as 10.5281/zenodo.22978729** (concept DOI 10.5281/zenodo.19233967, which always resolves to the
latest version). The metadata below still has to be pasted into the record
(Edit on the record page); only the files are locked after publishing.

Create the new version from the existing record ("New version" on the record
page), so the concept DOI stays the same and the March 2026 version is kept
as history. Upload the eight files that `python scripts/make_archive.py`
writes to `dist/zenodo/`: one `.tar.gz` and one `_manifest.json` per region.

**Title**
Fermium Hazard Mapper: input data and outputs for flood and landslide susceptibility in Bangladesh (version 2)

**Version**
2.0

**Resource type**
Dataset

**Licence**
Creative Commons Attribution 4.0 International (CC BY 4.0) for the outputs.
Inputs keep their own licences, listed per file in each manifest (OpenStreetMap
under ODbL 1.0, WorldPop and geoBoundaries under CC BY 4.0, SRTM public domain,
JRC Global Surface Water free with attribution, NASA COOLR open, Sentinel-1
derived flood masks under the Copernicus open licence).

**Description**

Input data and pipeline outputs for Fermium Hazard Mapper, which estimates flood
risk to mapped infrastructure and people in three regions of Bangladesh
(Rangpur and Rajshahi, Sylhet, and the south-west coast) and landslide
susceptibility in the Chittagong Hill Tracts. Each region is one archive with
the inputs the pipeline ingests (`raw/`) and the outputs it produces (`output/`),
and a manifest giving the size, SHA-256 checksum and licence of every file.

The flood models are trained on flood extents mapped from Sentinel-1 radar for
thirteen events between 2017 and 2024 and validated on 10 km blocks held out
from training. The outputs include the per-event flood masks, the asset scores
and calibrated probabilities, the terrain and Kriged hazard surfaces, the
composite risk grid, union, upazila and district summaries, and the metrics of
every validation described in the accompanying paper: held-out blocks, observed
floods, the model comparison over twenty block assignments, the label-source
comparison, the temporal hold-out on the 2024 floods, the past-flooding feature
test and the weight sensitivity analysis.

Code: https://github.com/rakibhhridoy/rimes-mapathon (MIT). Every figure in the
paper is generated from these outputs.

**Changes from version 1 (March 2026)**

Version 1 contained GADM administrative boundaries, which GADM's licence does
not allow to be redistributed. They are removed and replaced by geoBoundaries
at the union, upazila and district levels. Version 1's outputs came from a
pipeline with a coordinate-system error that left every training label at zero,
and should not be used. All outputs here are from the corrected pipeline, run on
23 and 24 September 2026.

**Authors**
Md Rakib Hasan (ORCID 0009-0002-4007-7590), University of Dhaka and Fermium Systems

**Keywords**
flood susceptibility; Sentinel-1; landslide susceptibility; spatial cross-validation; Bangladesh; OpenStreetMap

**After publishing**
Send the new version DOI (the one ending in a different number from
19233968), and it goes into the paper's data availability section.

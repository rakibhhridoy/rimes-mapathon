# Eastern plains (Meghna–Feni): flood events, verified

Checked 2026-10-01. Region: Comilla, Feni, Noakhali and Lakshmipur, bounding
box 90.630, 22.023, 91.577, 23.796 (W, S, E, N).

Each event was checked twice: that a humanitarian report confirms flooding in
these districts with dates, and that Sentinel-1 imaged the region during the
flood and in that year's February–March dry-season baseline (Earth Engine,
`COPERNICUS/S1_GRD`, IW mode, VV). Since Sentinel-1B failed in December 2021
only Sentinel-1A was flying through 2024, so passes over a given place came
every twelve days on each orbit direction rather than every six. ReliefWeb's
API now requires a registered application name and its pages refuse automated
reading, so the dates come from the same reports in the copies their
publishers host (UN Bangladesh, ACAPS), and from national newspapers for the
local July 2024 flood, which ReliefWeb does not record.

## Results

| Event | What the reports say | Sentinel-1 passes (local time, UTC+6) | Verdict |
|---|---|---|---|
| **Eastern flash floods, August 2024** | Heavy rain from 20 August; Dumbur dam in Tripura released 22 August; Noakhali, Comilla, Lakshmipur and Feni the most affected; more than 100 unions still under water on 30 August, and waterlogging in Noakhali and Lakshmipur into late September | 21 Aug 18:04 (100 %), 24 Aug 05:56 (41 %), 28 Aug 17:56 (24 %), 31 Aug 05:48 (98 %) | **Strong.** Imaged at onset, near the peak and while still flooded |
| **Cyclone Remal, May 2024** | Landfall near Mongla and Khepupara about 20:00 on 26 May; surge of 3–5 ft forecast for Noakhali and Lakshmipur; embankment breaches and submerged villages along the coast | 27 May 05:48 (98 %), 29 May 18:04 (100 %), 1 Jun 05:56 (41 %) | **Strong** for the coastal south: a pass about ten hours after landfall |
| **Feni embankment breach, July 2024** | Muhuri embankment broke and flooded Fulgazi and Parshuram from 1 July; breached again on 2 August | 4 Jul 18:04 (100 %), 11 Jul 17:56 (24 %) | **Usable but small.** Two upazilas only, so few flooded pixels |
| **Cyclone Sitrang, October 2022** | Crossed the coast on the evening of 24 October; Noakhali and Lakshmipur among the districts hit | 26 Oct 17:56 (24 %), 29 Oct 05:48 (98 %) | **Weak.** The first full pass came about four and a half days after landfall, when surge water had likely drained |

Every window, and every year's February–March baseline (50 scenes), covers
100 % of the region when all its passes are combined.

## What this means for the region

1. **Two strong events, both in 2024.** August 2024 dominates, Remal adds the
   coast. Together with the small Feni flood that is three events, all in one
   year.
2. **The temporal test cannot run here.** The existing regions are also
   trained on floods up to 2022 and scored on 2024. With every strong event in
   2024 and Sitrang too weak to train on, this region can be validated on
   held-out spatial blocks only, and the paper-style comparison with the flood
   record does not apply. That should be stated wherever its results appear.
3. **The labelling rule needs a decision.** The two strong events are
   different regimes (flash flooding inland, surge on the coast) and overlap
   little, so the two-event rule of the riverine regions would mark almost
   nothing. The one-event rule of the south-west coast fits better: ground
   counts as flood-prone if any mapped event flooded it.
4. **Persistent waterlogging is a strength here.** Water stood for weeks in
   Noakhali and Lakshmipur, so the late-August passes see it clearly, where a
   flash flood elsewhere would already have drained.

## Events, ready for a region configuration

Windows bracket the flooding; baselines are each year's dry season, as in the
existing regions. Sitrang is left out for the reason above.

```yaml
sentinel1:
  events:
    - name: "aug2024_eastern"
      start: "2024-08-20"
      end: "2024-09-05"
      baseline_start: "2024-02-01"
      baseline_end: "2024-03-31"
      source: "https://bangladesh.un.org/sites/default/files/2024-08/SitRep%2002%20Eastern%20Flash%20Flood.pdf"
    - name: "may2024_remal"
      start: "2024-05-26"
      end: "2024-06-01"
      baseline_start: "2024-02-01"
      baseline_end: "2024-03-31"
      source: "https://reliefweb.int/report/bangladesh/bangladesh-cyclone-remal-2024-situation-report-no-03-29-may-2024"
    - name: "jul2024_feni"
      start: "2024-07-01"
      end: "2024-07-12"
      baseline_start: "2024-02-01"
      baseline_end: "2024-03-31"
      source: "https://www.dhakatribune.com/bangladesh/nation/350839/muhuri-river-embankment-collapse-triggers-severe"
data:
  labels:
    min_events: 1    # surge and flash floods overlap little; see above
```

The fields match the existing region files, which give one source URL per
event (`config.yaml`), and `min_events: 1` is the south-west coast's rule
(`configs/sw_coastal.yaml`).

## Sources

- UN Bangladesh, Eastern Flash Floods 2024 Situation Report No. 02, 30 August 2024: https://bangladesh.un.org/sites/default/files/2024-08/SitRep%2002%20Eastern%20Flash%20Flood.pdf
- ACAPS, Bangladesh: Flooding, 4 September 2024: https://www.acaps.org/fileadmin/Data_Product/Main_media/20240904_ACAPS_Bangladesh_Flooding.pdf
- ReliefWeb, Bangladesh: Eastern Flash Flood Situation Report No. 01, 25 August 2024: https://reliefweb.int/report/bangladesh/bangladesh-eastern-flash-flood-situation-report-no-01-25-august-2024
- ACAPS, Bangladesh: impact of Tropical Cyclone Remal, 12 June 2024: https://www.acaps.org/fileadmin/Data_Product/Main_media/20240612_ACAPS_Bangladesh_-_Impact_of_Tropical_Cyclone_Remal.pdf
- ReliefWeb, Cyclone Remal 2024 Situation Report No. 03, 29 May 2024: https://reliefweb.int/report/bangladesh/bangladesh-cyclone-remal-2024-situation-report-no-03-29-may-2024
- ReliefWeb, Cyclone SITRANG Situation Report #1, 24 October 2022: https://reliefweb.int/report/bangladesh/cyclone-sitrang-bangladesh-situation-report-1-24-october-2022
- Dhaka Tribune, Muhuri River embankment collapse triggers severe flooding in Feni, 2 July 2024: https://www.dhakatribune.com/bangladesh/nation/350839/muhuri-river-embankment-collapse-triggers-severe
- The Daily Star, 74 villages submerged, thousands affected, 4 August 2024: https://www.thedailystar.net/environment/climate-crisis/natural-disaster/news/74-villages-submerged-thousands-affected-3669056

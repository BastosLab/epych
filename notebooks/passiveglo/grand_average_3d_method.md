# Grand-average 3-D spectrograms across areas

How the grand-average-across-areas 3-D spectrograms in `oscillatory_contrasts.ipynb` are
built (cells: `grand_average_areas` / `area_probe_counts`, and the adapted- and
oddball-block plotting cells). Each contrast (DD, SSA, GO, CTL) gets one 3-D `heatmap3d`
that pools all six visual areas into a single depth × time × frequency volume, for both the
adapted (pre-oddball) and oddball analysis windows.

## Inputs

The average operates on the per-area **contrast** signal `contrasts[name].signals[area]`
— the cluster-masked difference between the two conditions of the contrast (e.g. DD =
`lo_rndctl − lonaive`), one `EvokedTfr` per area with axes `(channel, time, frequency)`.
Each per-area signal has already been grand-concatenated over probes and subjects within
that area (via `GrandConcatenation` and the laminar aligner), so it is itself a
probe/subject grand average for that area.

Because we average the already-masked `contrasts[name]` (not the raw `diffs[name]`), the
grand average inherits the notebook's existing cluster-size accounting without any extra
logic: the adapted block defines each area's significant `cluster_size`, and the oddball
block requires oddball clusters to be at least that large (`cluster_min_size`). Whatever
survived masking per area is what gets averaged.

## Step 1 — Align areas at layer 4

Different areas have different aligned channel counts (their laminar extents differ, e.g.
VISp ≈ 19, VISl ≈ 20, VISal ≈ 15 channels after downsampling), so their depth axes do not
line up channel-for-channel. We align them by cortical layer 4:

- For each area, the L4 centre channel is the median index of channels whose location label
  contains `"4"` (the same L4 definition the notebook's `LAMINAE` dict uses).
- Let `up_a` = number of channels above L4 in area `a` (i.e. its L4 index) and `down_a` =
  number below. Take the shared extent `up = min_a up_a`, `down = min_a down_a`.
- Crop every area to the window `[L4 − up, L4 + down]`. All areas then have the same
  `C = up + down + 1` channels, with L4 at the same cropped position for every area.

Because one aligned channel is a common depth step across areas (the probes and downsampling
are identical), cropped channel *i* corresponds to the same laminar position (the same
distance from L4) in every area. A `reldepth` channel column is stamped on the result, with
`reldepth = 0` at L4, negative superficial, positive deep; it is used as the 3-D z-axis.

## Step 2 — Probe-count-weighted average over areas

The cropped areas are averaged channel-by-channel (equivalently, layer-by-layer), weighting
each area by the number of probes that recorded it:

```
grand = ( Σ_a  w_a · data_a ) / ( Σ_a  w_a )
```

where `w_a` is the probe count for area `a`. Areas sampled by more probes therefore
contribute proportionally more; the under-sampled area (VISal) contributes less. Time is
cropped to the shared length across areas before summing; frequencies already match (all
areas share the analysis window's frequency grid).

### Where the probe counts come from

`w_a` is the number of probes contributing to each area in the grand data, counted from the
epoched condition pickles that built it (`area_probe_counts()`): for every subject, each
probe's `<condition>/<probe>/channels.csv` is read, visual channels are kept, and their
common location prefix gives the area; the tally counts one per probe, so a subject with two
probes in one area (a "doubled probe") counts twice.

For dataset 000253 (14 subjects) the counts are:

| VISp | VISl | VISrl | VISal | VISpm | VISam |
|------|------|-------|-------|-------|-------|
|  14  |  15  |  15   |  11   |  14   |  14   |

VISl and VISrl are 15 because two subjects each contribute a second probe there; VISal is 11
because several subjects have no VISal probe.

> **Note.** Do **not** take the probe count from the laminar aligner
> (`len(aligner.stats[area].data["center"])`). The saved `visual_alignment` aligner was fit
> on a larger cohort than this grand contrast uses (it reports 55–75 probes per area), and
> its global-k indexing only consumes the first *N*. The condition-pickle count above is the
> probe set the grandcat actually pooled.

## Output

The result is a single `EvokedTfr` with axes `(C channels, time, frequency)` and the
`reldepth` z-coordinate, plotted with `heatmap3d(depth_column="reldepth")`: X = time,
Y = frequency, Z = depth relative to L4, colour = the (masked) contrast power difference
(PiYG diverging). One such plot is written per contrast per analysis window, alongside the
existing per-area 3-D plots.

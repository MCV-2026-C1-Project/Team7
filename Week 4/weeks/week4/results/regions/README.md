# Region selection study (origin of `MIN_REGION_FRACTION`)

Read-only study run on commit `dc4924a` (before the region filter). Only the region selection was replaced; the rest
of the Week 4 method (threshold 10, global ratio 0.85, recovery ≥ 10 inliers) is unchanged. No matching score is
used to filter regions, so unknown paintings must still give `[-1]`.

| Option | Rule |
| --- | --- |
| (a) | `MAX_PAINTINGS = 3` (the method at `dc4924a`) |
| (b) | `MAX_PAINTINGS = 2` |
| (c) | `MAX_PAINTINGS = 3`; drop regions with area < X % of the photo's largest region, X = 5, 10, 20 |
| (d) | `MAX_PAINTINGS = 5` (safety net, also the recovery budget); keep a region only if area ≥ X % of the largest (X = 10, 20) and a convex 4-corner fit of its mask has aspect ratio ≤ 6 and IoU with the mask ≥ 0.6 |

Produced on commit `dc4924a` (before the region filter) by the study script now committed as
`weeks/week4/experiments/region_study.py` (same computation and outputs; the committed version was not rerun):

```bash
python weeks/week4/experiments/region_study.py
```

`--min-region-fraction 0` (the default) reproduces that setting on later commits; `0.10` runs the final
method. Every option is listed by `--help`.

| File | Content |
| --- | --- |
| [options.csv](options.csv) | options × sets (QSD1-W4, QSD2-W3, QSD1-W3, QSD2-W2): regions vs ground-truth paintings, top-1, mAP@1, mAP@5, mask F1, unknown F1 (QSD1-W4) |
| [rejected_regions.csv](rejected_regions.csv) | every region rejected by (c) and (d): size, area as % of the largest region, reason, ground truth of the photo |

Findings:

- The only change on any set is QSD2-W3 00002: (a) and (c5) keep a spurious 161×65 region (6.6 % of the largest)
  and answer an extra `[-1]` (40 regions, mAP@1 0.950); (b), (c10), (c20), (d10) and (d20) drop it (39 regions,
  mAP@1 0.974). Everything else is unchanged, including QSD1-W4 (38/39, 0.974, unknown F1 0.952).
- No real painting is rejected. The smallest real painting is 20.1 % of the largest region (QSD1-W4 00009), so
  X = 20 keeps it by 0.1 points; X = 10 has margin on both sides (6.6 % spurious, 20.1 % real).
- No development photo has three paintings, so (b) only works on these sets. (c10) was adopted
  (`MIN_REGION_FRACTION = 0.10`): same answers as (b) and (d10), with the three-painting limit kept.
- (d) requires the quadrilateral to fit the region's own mask (not the Week 3 Canny quadrilateral refinement).

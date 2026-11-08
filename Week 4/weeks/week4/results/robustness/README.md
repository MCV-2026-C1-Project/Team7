# Robustness of the unknown decision (threshold 10, global ratio 0.85)

Read-only study of the Week 4 method on QSD1-W4, QSD1-W3 and QSD2-W3. It was run on commit `dc4924a`, before the
region filter (`MIN_REGION_FRACTION`), so QSD2-W3 still contains the spurious third region of 00002 (40 regions).
The method ran once per query; the global candidate and its best / second ratio were stored for every final
region, and `src.unknowns.decide` was applied afterwards for each pair. The (10, 0.85) decisions were checked to be
identical to the pipeline's own answers on the three sets.

Produced on commit `dc4924a` (before the region filter) by the study script now committed as
`weeks/week4/experiments/robustness_study.py` (same computation and outputs; the committed version was not rerun):

```bash
python weeks/week4/experiments/robustness_study.py
```

`--min-region-fraction 0` (the default) reproduces that setting on later commits; `0.10` runs the final
method. Every option is listed by `--help`.

| File | Content |
| --- | --- |
| [grid.csv](grid.csv) | threshold {6, 8, 10, 12, 15} × ratio {0.80, 0.85, 0.90} per set: top-1, mAP@1, mAP@5, unknown precision / recall / F1 |
| [unknown_margins.csv](unknown_margins.csv) | QSD1-W4: the 10 unknown paintings and the known paintings below the threshold, with local score, global ratio, decision and margins |
| [wrong_rejections.csv](wrong_rejections.csv) | QSD1-W3 / QSD2-W3 (all paintings known): every region below the threshold, its decision, and whether a known painting got a wrong `[-1]` |

Findings:

- **Threshold:** 8–15 gives identical results on the three sets (QSD1-W4 38/39, mAP@1 0.974, unknown F1 0.952 at
  ratio 0.85; QSD1-W3 1.000; QSD2-W3 0.950). 6 is worse on QSD1-W4 and QSD2-W3
  (unchanged on QSD1-W3): the highest local score of a wrong painting is 6.
- **Ratio:** 0.80 loses QSD1-W4 00016 (accepted by the fallback with ratio 0.801); 0.90 gives the same results as
  0.85, but is closer to the nearest unknown painting (00019, ratio 0.913). The safe range is about 0.80–0.91.
- **Wrong rejections:** none on QSD1-W3; one on QSD2-W3 (00019, second painting, GT 128: local score 5, global ratio
  0.932). The only QSD1-W4 failure is 00008 (GT 128): local score 5 for a wrong painting against 4 for 128; the
  global fallback ranks 128 first, but its ratio 0.939 is above 0.85.

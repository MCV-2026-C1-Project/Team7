# Week 4 final method on QSD1-W4

Output of `python main.py evaluate --week 4 --set qsd1_w4` (submitted method1: region filter, DoG + RootSIFT,
ratio 0.75, cross-check, RANSAC, distinct inliers, unknown threshold 10, global ratio 0.85), with per-painting rows.

| File | Content |
| --- | --- |
| [evaluate_summary.json](evaluate_summary.json) | the JSON printed by `main.py evaluate` |
| [metrics.csv](metrics.csv) | all 39 paintings and the 29 known ones: top-1, mAP@1, mAP@5; unknown precision / recall / F1 |
| [answers.csv](answers.csv) | one row per painting slot: ground truth, top-10 answer, rank of the correct id, AP@1, AP@5, local score, decision (local match / global fallback / unknown rejection) |

All 39 paintings: 38/39, mAP@1 0.974, mAP@5 0.974. Known paintings only: 28/29, mAP@1 0.966. Unknown precision /
recall / F1: 0.909 / 1.000 / 0.952. The thresholds were chosen on this set, so these numbers are optimistic.

In `evaluate_summary.json`, `unknown_paintings` (11) is the number of `[-1]` answers the method gave, not the
number of unknown paintings in the ground truth (10): 10 of the 11 are correct, hence precision 10 / 11 = 0.909.

# Task 3: the Week 3 method on QSD1-W4

Output of `python main.py evaluate --week 3 --set qsd1_w4`: the submitted Week 3 method (colour + LBP + DCT + wavelet,
noise filtering, L1, up to two paintings, no `[-1]` answer), evaluated on all three painting slots.

| File | Content |
| --- | --- |
| [evaluate_summary.json](evaluate_summary.json) | the JSON printed by `main.py evaluate` |
| [metrics.csv](metrics.csv) | all 39 paintings and the 29 known ones: top-1, mAP@1, mAP@5; unknown precision / recall / F1 |
| [answers.csv](answers.csv) | one row per painting slot: ground truth, top-10 answer, rank of the correct id, AP@1, AP@5 |

All 39 paintings: 23/39, mAP@1 0.590, mAP@5 0.603. Known paintings only: 23/29, mAP@1 0.793, mAP@5 0.810. It never
answers `[-1]`, so unknown recall and F1 are 0. The Week 4 numbers on the same set are in [../final/](../final/).

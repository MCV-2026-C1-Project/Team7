# Week 4 — keypoints, local descriptors and unknown paintings

The code is the final Team 7 pipeline (Weeks 1–4 in one tree: `src/` and `main.py`). Run everything from this folder.
The Week 4 report, method, results and reproduction commands are in [weeks/week4/README.md](weeks/week4/README.md).

```bash
pip install -r requirements.txt
python main.py evaluate --week 4 --set qsd1_w4     # QSD1-W4: mAP@1 0.974, unknown F1 0.952
python main.py predict  --week 4                   # QST1-W4 -> weeks/week4/submission/QST1/method1/result.pkl
python main.py validate --week 4
```

Datasets go in `data/` (`BBDD/`, `qsd1_w4/`, `qst1_w4/`, ...). The figures of the notebooks and the inspection images
mentioned in the Week 4 report are not included here.

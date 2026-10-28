ROADGUARD — EXPERIMENT COMPARISON
=================================

Dataset
-------
Training images:   5,000
Validation images: 1,000

Overall Evaluation Metrics
--------------------------

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
|------------|-----------|--------|-------|----------|
| Baseline | 0.5447 | 0.1966 | 0.1963 | 0.1075 |
| Experiment 2 | 0.5364 | 0.2539 | 0.2596 | 0.1370 |
| Experiment 3 | 0.5386 | 0.2622 | 0.2576 | 0.1353 |
| Experiment 4 | 0.5057 | 0.2669 | 0.2556 | 0.1330 |

Comparison
----------
Experiment 2 produced a substantial improvement over the baseline,
especially in Recall, mAP50, and mAP50-95.

Experiment 3 achieved a slightly higher Recall than Experiment 2,
but its mAP50 and mAP50-95 were slightly lower.

Experiment 4 achieved the highest Recall of all four experiments
(0.2669), but it had the lowest Precision (0.5057). Its mAP50
(0.2556) and mAP50-95 (0.1330) were also lower than Experiment 2.

Based on the overall mAP metrics, Experiment 2 achieved the best
overall detection performance among the four experiments.

Precision
---------
Baseline:      0.5447
Experiment 2:  0.5364
Experiment 3:  0.5386
Experiment 4:  0.5057

Recall
------
Baseline:      0.1966
Experiment 2:  0.2539
Experiment 3:  0.2622
Experiment 4:  0.2669

mAP50
-----
Baseline:      0.1963
Experiment 2:  0.2596
Experiment 3:  0.2576
Experiment 4:  0.2556

mAP50-95
--------
Baseline:      0.1075
Experiment 2:  0.1370
Experiment 3:  0.1353
Experiment 4:  0.1330

Final Selection
---------------
Experiment 2 was selected as the best-performing experiment based
on the overall mAP50 and mAP50-95 results.

Experiment 4 achieved the highest Recall, but this came with a
reduction in Precision and lower mAP values compared with
Experiment 2.
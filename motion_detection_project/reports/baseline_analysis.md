# Baseline Analysis

A baseline motion detection system was evaluated using a fixed experimental protocol with data split by videos into training, validation, and test sets.
Motion detection was formulated as binary classification over fixed-length temporal windows using a threshold-based decision rule on a single scalar feature (`motion_score`).

The learned decision threshold obtained by training was **0.00174**.

## Validation Results

On the validation set, the baseline achieved the following metrics:
- Accuracy: **0.85**
- Precision: **0.91**
- Recall: **0.42**
- F1-score: **0.57**
- TP = **10**, TN = **78**, FP = **1**, FN = **14**

These results indicate conservative system behavior: false positives are rare, while a significant fraction of true motion events is missed.
The baseline prioritizes stability over sensitivity, leading to high precision at the cost of low recall.

## Test Results

On the test set, the following metrics were obtained:

- Accuracy: **0.23**
- Precision: **1.00**
- Recall: **0.23**
- F1-score: **0.38**
- TP = **21**, TN = **0**, FP = **0**, FN = **69**

The absence of negative samples on the test split (TN = 0) renders precision and accuracy weakly informative. In this setting, recall reflects the fraction of motion windows successfully detected by the baseline.

## Observations

The baseline reliably detects pronounced motion in scenes with stable backgrounds but systematically misses weak or gradual motion.
Metric interpretation is strongly affected by class imbalance in the evaluation data, highlighting the importance of the experimental protocol when assessing system performance.

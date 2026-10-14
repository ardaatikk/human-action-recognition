# Experimental Report

## Human Action Recognition — KTH Dataset

This document provides a detailed record of the experiments conducted in the Human Action Recognition project.

It complements the main [README](../README.md) by documenting training configurations, checkpoint selection, validation results, ablation studies, temporal sensitivity experiments, and model selection decisions.

All model comparisons in this report use the validation split unless explicitly stated otherwise.

The held-out test set is evaluated only for the final selected model.

---

## 1. Experimental Objectives

The experiments investigate five questions:

1. Can structured pose features outperform an initial raw-video baseline?
2. How much predictive information is contained in pose detection availability?
3. How does pose detection reliability vary across action classes?
4. Does the coordinate-based LSTM depend strongly on temporal order?
5. Does a simpler MLP provide more stable performance across random seeds?

The experiments were developed sequentially, with observations from earlier stages motivating later investigations.

---

## 2. Dataset and Evaluation Protocol

### Dataset

KTH Human Actions Dataset.

Six classes:

- Boxing
- Handclapping
- Handwaving
- Jogging
- Running
- Walking

### Data Preparation

The dataset preparation process excludes one corrupted video.

The resulting dataset contains 598 usable videos.

### Subject-Independent Split

| Split | Videos | Subjects |
|---|---:|---:|
| Train | 358 | 15 |
| Validation | 120 | 5 |
| Test | 120 | 5 |

The split contains no subject overlap.

Validation and test sets each contain 20 videos per class.

### Evaluation Metrics

The project records:

- Accuracy.
- Macro precision.
- Macro recall.
- Macro F1.
- Per-class precision, recall, and F1.
- Confusion matrices.
- Per-video predictions.

The best checkpoint is selected using minimum validation loss.

---

## 3. Shared Training Configuration

The main experiments use the following common settings:

| Parameter | Value |
|---|---|
| Learning rate | 0.001 |
| Batch size | 8 |
| Weight decay | 0.0001 |
| Early stopping patience | 7 |
| Minimum delta | 0.0 |
| Device | MPS |
| DataLoader workers | 0 |

Random seeds and maximum epoch limits vary by experiment.

### Important Note on Resumed Experiments

The original CNN-LSTM and Full Pose LSTM experiments were developed through multiple training stages.

Their saved configuration files do not fully describe the cumulative training duration.

For example:

- `cnn_lstm_baseline/config.json` records `epochs=5`, while its training history contains 30 epochs.
- `pose_lstm_baseline/config.json` records `epochs=1`, while its training history contains 30 epochs.

The complete training history is therefore used to determine the actual number of completed epochs.

This distinction is important when interpreting or reproducing the original experiments.

---

## 4. Experiment Registry

| Experiment | Model | Seed | Epochs Recorded | Best Epoch |
|---|---|---:|---:|---:|
| `cnn_lstm_baseline` | CNN-LSTM | 42 | 30 | 28 |
| `pose_lstm_baseline` | Full Pose LSTM | 42 | 30 | 24 |
| `pose_coordinates_only_ablation` | Coordinates LSTM | 42 | 28 | 21 |
| `pose_coordinates_seed123` | Coordinates LSTM | 123 | 25 | 18 |
| `pose_coordinates_seed456` | Coordinates LSTM | 456 | 26 | 19 |
| `pose_mask_only_ablation` | Mask LSTM | 42 | 30 | 30 |
| `pose_mean_mlp_baseline` | Pose Mean MLP | 42 | 43 | 36 |
| `pose_mean_mlp_seed123` | Pose Mean MLP | 123 | 50 | 44 |
| `pose_mean_mlp_seed456` | Pose Mean MLP | 456 | 50 | 46 |

The best epoch refers to the minimum recorded validation loss.

---

## 5. Experiment A — CNN-LSTM Baseline

### Objective

Establish an initial video-based benchmark using spatial feature extraction followed by recurrent sequence modeling.

### Input

- 16 sampled frames.
- Grayscale representation.
- 112 × 112 spatial resolution.

### Model

CNN feature extractor followed by an LSTM classifier.

### Training

| Property | Value |
|---|---:|
| Seed | 42 |
| Completed epochs | 30 |
| Best epoch | 28 |
| Best validation loss | 1.013765 |

### Validation Results

| Metric | Value |
|---|---:|
| Accuracy | 0.5000 |
| Macro precision | 0.5132 |
| Macro recall | 0.5000 |
| Macro F1 | 0.5012 |

### Interpretation

The CNN-LSTM establishes a moderate baseline on the subject-independent split.

Its performance motivates investigating a more structured representation of human movement.

The result does not imply that CNN-LSTM architectures are inherently unsuitable for action recognition.

It reflects this particular architecture, training setup, and dataset.

---

## 6. Experiment B — Full Pose LSTM

### Objective

Evaluate whether pose-based representations provide a more effective input than raw video frames.

### Input

A sequence of 16 pose feature vectors.

The full representation includes landmark-derived features and detection-related information.

### Training

| Property | Value |
|---|---:|
| Seed | 42 |
| Completed epochs | 30 |
| Best epoch | 24 |
| Best validation loss | 0.364672 |

### Validation Results

| Metric | Value |
|---|---:|
| Accuracy | 0.8750 |
| Macro precision | 0.8778 |
| Macro recall | 0.8750 |
| Macro F1 | 0.8709 |

### Per-Class Recall

| Class | Recall |
|---|---:|
| Boxing | 0.95 |
| Handclapping | 0.95 |
| Handwaving | 1.00 |
| Jogging | 0.55 |
| Running | 0.85 |
| Walking | 0.95 |

### Interpretation

The full pose representation substantially improves validation accuracy relative to the CNN-LSTM baseline.

However, the weaker jogging recall suggests that the pose-based representation may not perform equally well across all activities.

This motivates examining pose extraction reliability.

---

## 7. Experiment C — Coordinates-Only Ablation

### Objective

Investigate the contribution of landmark coordinates after removing the explicit detection-mask feature.

### Input

A sequence of 16 coordinate vectors, each containing 99 values.

### Training

| Property | Value |
|---|---:|
| Seed | 42 |
| Completed epochs | 28 |
| Best epoch | 21 |
| Best validation loss | 0.320149 |

### Validation Results

| Metric | Value |
|---|---:|
| Accuracy | 0.8583 |
| Macro precision | 0.8695 |
| Macro recall | 0.8583 |
| Macro F1 | 0.8601 |

### Interpretation

The coordinate-only model performs close to the full pose model.

This indicates that landmark coordinates retain substantial discriminative information.

Nevertheless, explicit mask removal does not guarantee the removal of all detection-related signals.

For example, the representation of missing landmarks may still reflect detection failures.

---

## 8. Experiment D — Detection-Mask-Only Ablation

### Objective

Determine whether pose detection availability alone can predict action labels.

### Input

A binary detection mask over 16 frames.

No landmark coordinates are supplied.

### Training

| Property | Value |
|---|---:|
| Seed | 42 |
| Completed epochs | 30 |
| Best epoch | 30 |
| Best validation loss | 1.059432 |

### Validation Results

| Metric | Value |
|---|---:|
| Accuracy | 0.4333 |
| Macro precision | 0.3375 |
| Macro recall | 0.4333 |
| Macro F1 | 0.3366 |

### Per-Class Recall

| Class | Recall |
|---|---:|
| Boxing | 0.25 |
| Handclapping | 0.90 |
| Handwaving | 0.00 |
| Jogging | 0.00 |
| Running | 0.65 |
| Walking | 0.80 |

### Interpretation

The mask-only model achieves 43.33% validation accuracy.

This is substantially above the expected 16.67% accuracy of uniform random guessing on six balanced classes.

Therefore, pose detection availability contains information associated with action labels.

The result raises a potential shortcut-learning concern.

However, it does not prove data leakage or establish that the full pose model primarily relies on the mask.

Additional controlled experiments would be needed to determine the exact contribution of detection-related signals.

---

## 9. Pose Detection Reliability Analysis

### Objective

Measure the relationship between pose extraction success and classification performance.

### Validation Detection Rates

| Class | Mean Detection Rate |
|---|---:|
| Boxing | 0.959375 |
| Handclapping | 0.993750 |
| Handwaving | 1.000000 |
| Jogging | 0.362500 |
| Running | 0.300000 |
| Walking | 0.412500 |

### Detection Groups

Videos are categorized into:

- Low: detection rate up to one-third.
- Medium: between one-third and two-thirds.
- High: above two-thirds.

### Full Pose LSTM Accuracy by Group

| Group | Videos | Correct | Accuracy |
|---|---:|---:|---:|
| Low | 25 | 16 | 0.6400 |
| Medium | 34 | 30 | 0.8824 |
| High | 61 | 59 | 0.9672 |

### Class-Conditional Observations

Detection groups have different action distributions.

For example:

- All validation boxing videos fall in the high-detection group.
- All handclapping videos fall in the high-detection group.
- All handwaving videos fall in the high-detection group.
- Jogging and running appear predominantly in low and medium groups.

For jogging specifically:

| Detection Group | Videos | Correct | Accuracy |
|---|---:|---:|---:|
| Low | 8 | 0 | 0.0000 |
| Medium | 12 | 11 | 0.9167 |

For running:

| Detection Group | Videos | Correct | Accuracy |
|---|---:|---:|---:|
| Low | 13 | 13 | 1.0000 |
| Medium | 7 | 4 | 0.5714 |

These examples show that the relationship between detection rate and classification accuracy is not uniform across classes.

### Interpretation

The overall association between detection rate and accuracy is strong, but the action distribution introduces confounding.

A causal interpretation is therefore not justified.

This analysis supports treating pose detection reliability as an important experimental variable.

---

## 10. Temporal Order Sensitivity Analysis

### Objective

Evaluate how much the coordinate-based LSTM depends on the temporal ordering of pose features.

### Model

`pose_coordinates_only_ablation`

Checkpoint: `best`

Split: `validation`

### Perturbation Conditions

**Original:** Preserve frame order.

**Reversed:** Reverse the 16-frame sequence.

**Shuffled:** Randomly permute the 16 frames independently for each video.

The shuffled condition is repeated 10 times using seeds 42–51.

### Results

| Condition | Seed | Accuracy | Prediction Agreement |
|---|---:|---:|---:|
| Original | 42 | 0.8583 | 1.0000 |
| Reversed | 42 | 0.8417 | 0.8833 |
| Shuffled | 42 | 0.8417 | 0.9000 |
| Shuffled | 43 | 0.8333 | 0.9250 |
| Shuffled | 44 | 0.8583 | 0.9417 |
| Shuffled | 45 | 0.8833 | 0.9250 |
| Shuffled | 46 | 0.8583 | 0.9167 |
| Shuffled | 47 | 0.8500 | 0.9000 |
| Shuffled | 48 | 0.8667 | 0.9000 |
| Shuffled | 49 | 0.8417 | 0.8750 |
| Shuffled | 50 | 0.8917 | 0.9083 |
| Shuffled | 51 | 0.8583 | 0.9083 |

### Interpretation

Reversing the sequence decreases accuracy by approximately 1.67 percentage points.

Randomly shuffling frames does not consistently reduce accuracy.

Some shuffled runs even produce slightly higher accuracy than the original ordering.

This suggests that the model can classify many videos using information that is relatively insensitive to temporal ordering.

However, prediction agreement decreases under both reversed and shuffled conditions.

Therefore, temporal order still affects individual predictions.

### Research Implication

These findings motivate evaluating an architecture that deliberately removes sequence-order modeling.

The Pose Mean MLP is introduced for this purpose.

---

## 11. Experiment E — Pose Mean MLP

### Objective

Evaluate whether temporally aggregated pose coordinates can provide competitive performance without recurrent sequence modeling.

### Representation

Pose coordinates are aggregated through temporal mean pooling.

The resulting feature representation is classified using a multilayer perceptron.

### Architecture

Approximately 30,086 trainable parameters.

### Training

| Property | Value |
|---|---:|
| Seed | 42 |
| Maximum configured epochs | 50 |
| Completed epochs | 43 |
| Best epoch | 36 |
| Best validation loss | 0.395961 |

### Validation Results

| Metric | Value |
|---|---:|
| Accuracy | 0.8333 |
| Macro precision | 0.8375 |
| Macro recall | 0.8333 |
| Macro F1 | 0.8242 |

### Interpretation

The Pose Mean MLP achieves competitive validation performance despite its simpler architecture.

This supports the hypothesis that order-insensitive pose information is useful for recognizing several KTH action classes.

However, the result does not imply that temporal modeling is unnecessary for action recognition in general.

---

## 12. Multi-Seed Stability Study

### Objective

Compare the sensitivity of the Coordinates-Only LSTM and Pose Mean MLP to random initialization and training variation.

### Protocol

Three random seeds:

```text
42
123
456
```

The subject-independent dataset split remains fixed.

### Coordinates-Only LSTM

| Seed | Best Epoch | Accuracy | Macro F1 |
|---|---:|---:|---:|
| 42 | 21 | 0.8583 | 0.8601 |
| 123 | 18 | 0.6750 | 0.6619 |
| 456 | 19 | 0.8333 | 0.8316 |

### Pose Mean MLP

| Seed | Best Epoch | Accuracy | Macro F1 |
|---|---:|---:|---:|
| 42 | 36 | 0.8333 | 0.8242 |
| 123 | 44 | 0.8250 | 0.8177 |
| 456 | 46 | 0.8250 | 0.8174 |

### Summary

| Statistic | Coordinates LSTM | Pose Mean MLP |
|---|---:|---:|
| Mean accuracy | 0.7889 | 0.8278 |
| Accuracy sample std. dev. | 0.0994 | 0.0048 |
| Mean macro F1 | 0.7846 | 0.8198 |

### Interpretation

The coordinate LSTM produces the highest individual result in this comparison but exhibits substantial variability.

The Pose Mean MLP achieves a higher average accuracy and much lower variability.

This supports selecting the MLP as a stable baseline for final evaluation.

### Statistical Limitations

The study includes only three random seeds.

It does not establish statistical significance.

Furthermore, the full pose LSTM was not included in this multi-seed comparison.

Consequently, the experiment should not be interpreted as a comprehensive stability ranking of all implemented architectures.

---

## 13. Final Model Selection

### Selection Criteria

The final model is chosen using:

1. Validation performance.
2. Multi-seed stability.
3. Architectural simplicity.
4. Findings from the detection-mask ablation.
5. Findings from temporal order perturbation.

### Selected Model

**Pose Mean MLP**

Experiment:

```text
pose_mean_mlp_baseline
```

Seed:

```text
42
```

Checkpoint:

```text
best
```

### Rationale

The selected model provides competitive validation accuracy, relatively stable performance across three random seeds, and a simpler architecture than the recurrent alternatives.

The decision is made without using test-set performance.

---

## 14. Held-Out Test Evaluation

### Protocol

The selected Pose Mean MLP checkpoint is evaluated on the independent test split.

The test split contains 120 videos from five subjects not used during training or validation.

### Results

| Metric | Value |
|---|---:|
| Accuracy | 0.8083 |
| Macro precision | 0.8114 |
| Macro recall | 0.8083 |
| Macro F1 | 0.8061 |

### Confusion Matrix

![Final Test Confusion Matrix](images/pose_mean_mlp_test_confusion_matrix.png)

### Per-Class Results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Boxing | 1.000 | 0.900 | 0.947 |
| Handclapping | 0.864 | 0.950 | 0.905 |
| Handwaving | 0.850 | 0.850 | 0.850 |
| Jogging | 0.667 | 0.500 | 0.571 |
| Running | 0.583 | 0.700 | 0.636 |
| Walking | 0.905 | 0.950 | 0.927 |

### Error Analysis

The main confusion occurs between jogging and running.

Specifically:

- 9 jogging videos are classified as running.
- 5 running videos are classified as jogging.

These errors are consistent with the difficulty of separating similar locomotion activities using temporally aggregated pose information.

However, the experiment does not establish the cause of the confusion.

### Generalization Gap

Validation accuracy:

```text
83.33%
```

Test accuracy:

```text
80.83%
```

Difference:

```text
2.50 percentage points
```

The result indicates a relatively small observed validation-to-test decrease for the selected checkpoint.

---

## 15. Reproducibility

### Training Records

Each experiment stores its training configuration and epoch history under:

```text
outputs/training/<experiment>/
```

### Model Checkpoints

Checkpoints are stored under:

```text
artifacts/models/<experiment>/
```

### Evaluation Reports

Evaluation artifacts are stored under:

```text
artifacts/reports/<experiment>/<split>/<checkpoint>/
```

### Analysis Outputs

Pose detection analysis:

```text
artifacts/reports/pose_lstm_baseline/validation/best/pose_detection_impact/
```

Temporal order analysis:

```text
artifacts/reports/pose_coordinates_only_ablation/validation/best/temporal_order_impact/
```

### Reproduction Caveats

The project uses deterministic seed configuration where supported, but exact numerical reproduction can still depend on hardware and library versions.

The original experiments were performed primarily using Apple MPS.

The exact MediaPipe pose model variant and preprocessing configuration should be matched when attempting to reproduce the reported metrics.

Some early experiments used resumed training, and their final saved configuration files do not represent the entire training history.

---

## 16. Threats to Validity

### Internal Validity

Pose detection reliability differs between classes.

This introduces potential shortcut-learning signals.

Temporal perturbations test sensitivity to frame order but do not isolate every form of temporal information.

### External Validity

KTH contains a limited set of actions and recording conditions.

Results may not generalize to complex real-world environments.

### Statistical Validity

The multi-seed study includes only three seeds.

Observed stability differences should be treated as descriptive.

### Construct Validity

Classification accuracy and macro F1 measure predictive performance but do not directly establish which features or temporal relationships a model uses internally.

---

## 17. Main Conclusions

The experimental investigation supports five conclusions.

**1. Pose-based representations are effective on KTH.**

The Full Pose LSTM substantially outperforms the initial CNN-LSTM baseline in the evaluated setup.

**2. Pose extraction behavior deserves explicit analysis.**

Detection rates vary strongly across action classes, and detection availability alone contains predictive information.

**3. Removing an explicit detection mask is not a complete guarantee against shortcut signals.**

Missing-landmark patterns may remain embedded in coordinate features.

**4. Temporal ordering is not the dominant observable factor for every prediction.**

The coordinate LSTM maintains similar accuracy after sequence reversal and shuffling, although individual predictions change.

**5. Simpler architectures can be competitive and stable.**

The Pose Mean MLP achieves competitive validation performance, low observed seed-to-seed variability, and 80.83% accuracy on held-out test subjects.

These conclusions are limited to the experimental configurations and dataset evaluated in this repository.

---

## 18. Future Experiments

Potential research directions include:

- Additional random seeds.
- Alternative pose estimators.
- Explicit velocity and acceleration features.
- Temporal convolutional architectures.
- Transformer-based sequence models.
- Controlled missing-landmark imputation.
- Class-conditional detection analysis.
- Additional action recognition datasets.
- Robustness analysis under camera variation and occlusion.

No results are claimed for these proposed experiments.

---

## 19. Experiment Summary

| Stage | Main Question | Principal Finding |
|---|---|---|
| CNN-LSTM | Can raw video features recognize KTH actions? | 50.00% validation accuracy |
| Full Pose LSTM | Are pose features more effective? | 87.50% validation accuracy |
| Coordinates ablation | Are coordinates sufficient without the explicit mask? | 85.83% validation accuracy |
| Mask-only ablation | Does detection availability carry label information? | 43.33% validation accuracy |
| Detection analysis | Does detection reliability differ across actions? | Strong class-dependent differences |
| Temporal analysis | Is the LSTM strongly order-sensitive? | Limited observed accuracy sensitivity |
| Pose Mean MLP | Can an order-insensitive model compete? | 83.33% seed-42 validation accuracy |
| Multi-seed study | Which evaluated model is more stable? | MLP shows lower observed variability |
| Final test | Does the selected model generalize to held-out subjects? | 80.83% accuracy; 0.8061 macro F1 |

---

**End of Experimental Report**
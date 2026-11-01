

Project Specification

Project Identification

Team: 18

Track: E – Wafer Maps

Card: E3

Dataset: WM-811K

Specification: Release Bundle RevD2

Experimental Requirements

1\. Task and Objective

This project investigates semi-supervised wafer defect classification using labelled and unlabelled wafer maps from WM-811K.



The objective is to evaluate whether pseudo-labelling improves classification performance when eligible unlabelled wafer maps are added to supervised training.



The task is single-label classification across nine categories: eight defect classes and none.



2\. Classical Baseline

Implement a Random Forest classifier using:



Radon features.

Geometric features.

Evaluate the baseline under the assigned built-in train/test split protocol.



The Random Forest default is 200 trees. Select max\_depth from {10, 20, unrestricted} using validation data. Classical tuning is limited to three declared settings, and all classical fitting (tuning and final fits) shares a 60-minute elapsed-time allowance.



3\. Required Deep Learning Models and Components

The project must demonstrate:



2-D CNN: Learns spatial features directly from 64×64 wafer-map images.

1-D radial-profile CNN: Uses radial/angular profile representations computed from the original valid wafer mask.

Spatial attention: Add spatial attention to the image model and compare it with the corresponding image model without attention.

Fusion: Combine predictions from the no-attention radial-profile CNN and the no-attention 2-D CNN.

Select the fusion weight using validation data only, from:



{0, 0.25, 0.5, 0.75, 1}



If multiple weights tie, choose the value closest to 0.5.



The fusion is a prediction-level combination and does not require an additional neural training fit.



4\. Core Experiment 1: Supervised Learning vs. Pseudo-Labelling

Compare the supervised-only image CNN with image CNNs trained after one round of pseudo-labelling.



The supervised image reference model must predict labels for the eligible unlabelled pool. Accept a prediction only when its confidence is at least 0.95. Add accepted pseudo-labelled examples to the supervised training data, then train a fresh image model.



5\. Core Experiment 2: Gain vs. CPU Cost

Evaluate the performance gain and computational cost of pseudo-labelling at two fixed, nested eligible-pool caps:



2,000 wafers.

10,000 wafers.

The 2,000-wafer pool must be a subset of the 10,000-wafer pool. Each cap requires a separate fresh image-model fit, using the same supervised data and training budget.



6\. Pseudo-Labelling Restrictions

Only one pseudo-labelling round is permitted.

The 2-D CNN supplies the pseudo-labels.

The confidence threshold is fixed at 0.95.

Use the issued eligible-pool row indices and the prescribed nested caps.

Both caps must use the same supervised data and training budget.

Give each training example equal weighting.

Do not use iterative self-training.

Zero accepted pseudo-labels is a valid result.

Do not lower the confidence threshold after examining test results.

7\. Required Reporting

Report:



Accepted pseudo-label count at each pool cap.

Accepted pseudo-label counts by predicted class.

Class distribution of accepted pseudo-labels.

Whether class collapse occurred.

Macro-F1 change compared with the supervised-only reference.

Per-class recall.

Elapsed wall-clock time and aggregate CPU time.

Gain in Macro-F1 per CPU time.

Whether pseudo-labelling produces no improvement or a negative improvement.

Report the accepted pseudo-label counts even if no examples meet the confidence threshold. Do not omit negative or unchanged results.



8\. Required Neural Training Fits

The execution contract identifies exactly five required neural fits:



Supervised image CNN.

Image CNN with the 2,000-wafer pseudo-label cap.

Image CNN with the 10,000-wafer pseudo-label cap.

Radial-profile CNN.

Image CNN with spatial attention.

The no-attention image model is the reference for the spatial-attention ablation. The fusion combines predictions from the no-attention radial-profile and image models and does not add another neural training fit.



9\. Execution and Resource Limits

Default random seed: 22.

Each neural fit is limited to 30 epochs or 30 minutes of elapsed wall-clock time, whichever occurs first.

Apply early stopping after five validation checks without improvement and retain the best validation checkpoint.

The overall elapsed-time budget for the required programme is 24 hours, including preprocessing, classical fits, permitted development, core training and repeated inference.

At most two additional development fits are permitted per group. They must use training/validation data only and together must take no more than 60 minutes. Freeze choices before the core comparisons.

Small from-scratch models must have no more than 250,000 trainable parameters.

The retained working data must not exceed 5 GB. Report peak temporary disk use separately.

Use streamed batches. Peak memory must be compatible with the 16 GB reference machine, sizing the pipeline for the 6-core low end.

Record actual hardware, thread/worker settings, elapsed time, aggregate CPU time and per-epoch time for every fit.

Aggregate CPU time must include main and worker process user/system time.

If a fit reaches its limit before learning adequately, report the result and request a scope correction rather than silently exceeding the limit.



Dataset and Split Protocol

1\. Dataset Information and Approved Source

Dataset: WM-811K wafer-map dataset.

Approved source: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

Access: A free Kaggle account is required.

Archive: archive.zip

Pickle file: LSWMD.pkl, contained in archive.zip.

The Kaggle copy is the approved acquisition route under Release Bundle RevD2. The RevD2 manifest identifies this copy as identical row-for-row to the original source.



Reference dataset inventory:



Total wafer maps: 811,457.

Total lots: 46,293.

Labelled wafers: 172,950.

Unlabelled wafers: 638,507.

Wafer maps with an actual defect pattern: 25,519.

2\. Dataset Fields

The pickle contains the following fields:



waferMap: Wafer-map representation.

dieSize: Die-count information.

lotName: Manufacturing lot identifier.

waferIndex: Wafer index.

trianTestLabel: Built-in train/test membership field. This spelling is intentional; it is how the field appears in the Kaggle pickle.

failureType: Failure or defect label.

3\. Label and Membership Normalisation

In the Kaggle pickle, labelled failureType values are stored as 1×1 arrays, for example:



array(\[\['none']])



Unlabelled rows contain empty 0×0 arrays.



Apply these normalisation rules:



A plain string remains unchanged.

An array containing exactly one string is converted to that string.

Any other value is treated as unlabelled.

Apply the same normalisation principle to the built-in membership field. Do not use a simple non-empty check to identify labelled wafers.



4\. Classification Labels

The task uses these nine single-label categories:



none

Edge-Ring

Edge-Loc

Center

Loc

Scratch

Random

Donut

Near-full

Retain the documented label mapping. Do not infer labels from folder order.



5\. Built-in Train/Test Split

Preserve the dataset's built-in train/test membership using the trianTestLabel field. Do not randomly regenerate the test split.



Reference labelled partition sizes:



Training: 54,355 wafers across 5,809 lots.

Test: 118,595 wafers across 4,953 lots.

The built-in training and test partitions share no lots.



6\. Validation Split

Create validation only from the built-in training partition.



Use this fixed procedure:



Identify all lots represented in the labelled training partition.

Sort the lot names by the SHA-256 hash of 22|lotName.

Select the first floor(20%) of the sorted lots for validation.

Use the remaining training lots for model training.

The default seed is 22. Do not redraw the validation split based on model results.



Instructor reference result:



Training-side lots before the split: 5,809.

Validation lots: 1,161.

Validation is lot-disjoint from the remaining training lots and the built-in test partition.

Every class has validation support in the instructor reference run.

Near-full is the thinnest class, with 10 validation wafers from 10 lots.

7\. Eligible Unlabelled Pool

The eligible unlabelled pool is mandatory for Card E3.



Exclude:



Every labelled wafer belonging to a validation or test lot.

Every unlabelled wafer belonging to any lot represented in validation.

Every unlabelled wafer belonging to any lot represented in test.

The instructor reference eligible pool contains:



620,038 unlabelled wafers.

39,496 lots.

Reproduce and report the actual count from the acquired dataset.



8\. Pseudo-Label Pool Selection

Create two nested caps from the eligible unlabelled pool:



2,000 wafers.

10,000 wafers.

Sort eligible unlabelled wafers by the SHA-256 hash of 22| followed by their original row index. Select the first 2,000 and first 10,000 entries, respectively.



Therefore, the 2,000-wafer pool must be a subset of the 10,000-wafer pool.



The issued row indices are provided in:



TrackE\_split\_lot\_lists.json



Use the issued lists and verify them against the dataset. Do not select new pool members based on model performance.



9\. Preprocessing Requirements

Build one dataset inventory using the pickle.

Use the original row index as the wafer ID.

Record lotName, normalised label, original wafer-map shape, die count and built-in membership.

Resize wafer maps to 64×64 using nearest-neighbour resizing.

Preserve categorical map values. Do not use interpolation that creates new categorical values.

Compute radial/angular profiles from the original valid wafer mask.

Do not treat the outside-wafer region as a good die.

Use streamed batches rather than loading all resized images into RAM.

10\. Data Integrity and Reproducibility

Before modelling:



Verify the downloaded archive SHA-256 against the RevD2 manifest.

Verify the extracted pickle SHA-256 against the RevD2 manifest.

Reproduce the dataset counts.

Verify the built-in split membership.

Reproduce the validation lot selection.

Verify the eligible unlabelled pool count.

Verify the issued pseudo-label pool row indices.

Investigate any mismatch with a selected-artifact check before proceeding with the affected modelling.



Evaluation Metrics

1\. Primary Classification Metrics

The primary evaluation metric is Macro-F1, supported by per-class recall.



Macro-F1: Calculate F1 over the fixed nine-class label universe and take the unweighted mean.

Per-class recall: Report recall separately for all nine classes.

Supported classes: Report how many classes have actual positive samples in the evaluation partition.

2\. Metric Calculation Rules

Use the same fixed nine-class label universe throughout evaluation.

If a class has undefined F1, assign it a zero contribution to Macro-F1.

If a class has no positive samples, report its recall as NA, not zero.

Mark unsupported classes explicitly; do not silently remove them from evaluation.

If balanced accuracy is reported, calculate it as the mean recall over supported classes and report the number of supported classes and the denominator used.

Do not use accuracy as the primary metric because of the severe class imbalance.

3\. Pseudo-Label Metrics

For both the 2,000-wafer and 10,000-wafer pool caps, report:



Number of accepted pseudo-labels.

Accepted pseudo-label count for each predicted class.

Class distribution of accepted pseudo-labels.

Whether class collapse occurred.

Macro-F1 change compared with the supervised-only reference.

Zero accepted pseudo-labels are a valid result. Do not lower the confidence threshold after examining test results.



4\. Computational Cost Metrics

Report computational cost alongside model performance:



Elapsed wall-clock time.

Aggregate CPU time, including main and worker processes.

CPU time for each relevant experiment.

Gain in Macro-F1 per CPU time.

Record hardware specifications, thread/worker settings and timing information for every fit. Report elapsed wall-clock time and aggregate CPU time separately.



5\. Comparison and Interpretation

Compare the supervised-only image CNN against the two pseudo-labelled image CNNs.



The second core experiment examines the relationship between performance gain and CPU cost at the additional pool cap.



Unchanged or negative Macro-F1 gain is a valid experimental outcome and must be reported honestly.



Use training and validation data for all model selection and tuning decisions. Keep the test partition untouched until final evaluation. At the readiness checkpoint, test-shaped inference may be timed, but test labels and test metrics must not be inspected or scored.

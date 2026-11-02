# Evaluation protocol and interpretation

This repository collects reusable code from exploratory SCORE-MIL development. It is not a claim that the method is clinically validated or ready for deployment.

## What the code supports

- Ordinal-attention multiple-instance learning for bag-level labels.
- Pseudo-unknown evaluation where a target label is withheld from downstream classifier training.
- Known-only calibration and fixed-false-alarm operating-point analysis.
- Paired case-level bootstrap comparisons and feature-integrity audits.
- A paired encoder/view factorial study with a two-GPU training option.

## Interpretation limits

- A pseudo-unknown class is withheld from the downstream classifier; this does not establish that its morphology was unseen during encoder pretraining.
- Reusing a development validation set to choose methods makes it unsuitable as an independent test set.
- A source-held-out stress protocol and a class-held-out protocol answer different questions and should be reported separately.
- Patient identity, diagnostic-label semantics, and acquisition metadata must be confirmed with the data owner before making clinical claims.
- The two-GPU scheduler parallelizes independent model tasks. Actual GPU utilization must be checked on the target machine during a real run.

Real-data metrics and patient-level outputs are not included in this public-facing code package. Run the scripts against authorized local data and report sample counts, split definitions, confidence intervals, and all protocol limitations alongside any result.

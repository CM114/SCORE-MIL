# SCORE-MIL: Cervical Cytology Open-World Research Code

Research code for **Severity-Calibrated Open-world Residual Evidence MIL (SCORE-MIL)**. The repository keeps the ordinal attention model, known-only calibration, pseudo-unknown evaluation, paired case bootstrap, tile/feature audits, and the scale-by-encoder experiment together with their focused tests.

This is an experimental research repository. Any real-data results produced by these scripts are development findings, not an independent clinical evaluation. No hospital images, patient-level manifests, feature arrays, checkpoints, validation prediction tables, or real-data metrics are included in this repository.

## Quick start: run the synthetic example

The example trains the same ordinal-attention MIL implementation on generated bags, then writes aggregate metrics plus synthetic validation predictions. These numbers only show that the code path runs; they are not cervical-cytology performance estimates.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python examples/synthetic_demo.py --out-dir outputs/synthetic-demo
```

On Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python examples/synthetic_demo.py --out-dir outputs/synthetic-demo
```

The command prints the unknown-detection AUROC, AUPR, FPR95, and known-class macro-F1. It also writes `result.json` and `validation_predictions.csv` under the requested output directory. `outputs/` and CSV files are ignored by Git.

## Core files

| File | Purpose |
|---|---|
| `score_mil_ordinal_control.py` | Shared and decoupled ordinal-attention MIL, ordinal targets, residual scores, and baseline training CLI. |
| `score_mil_true_pseudo_openworld.py` | True pseudo-unknown partitioning and downstream MIL training; pseudo-unknown labels are excluded from classifier training. |
| `score_mil_known_calibration_audit.py` | Class-stratified known-only calibration splits and fixed false-alarm evaluation. |
| `score_mil_paired_case_bootstrap.py` | Paired case-level AUROC bootstrap and tie diagnostics. |
| `score_mil_scale_factorial.py` | ResNet50/UniCAS × resized/native-crop feature experiment. |
| `score_mil_scale_factorial_multigpu.py` | Two-GPU model-level worker scheduler for completed development feature caches. |
| `score_mil_scale_factorial_summary.py` | Aggregates completed factorial results. |
| `score_mil_scale_factorial_bootstrap.py` | Predeclared paired bootstrap comparisons for the factorial experiment. |
| `score_mil_scale_factorial_audit.py`, `score_mil_exact_jpeg_duplicate_audit.py` | Feature and exact cross-split JPEG integrity checks. |

The `score_mil_*manifest.py`, feature extraction, and `ndpi_mil_baseline.py` files are included only as dependencies of these entry points. They read user-supplied local data; this repository does not include that data.

## Real-data experiment

The feature/scale experiment expects the protocol JSON, candidate CSV, prior development features, prepared JPEG tile root, and UniCAS checkpoint to be supplied locally. Example command (replace each path with your own data path):

```bash
python score_mil_scale_factorial.py \
  --protocol /data/protocol.json \
  --candidate-csv /data/candidates.csv \
  --previous /data/previous_feature_run \
  --wsl-root /data/prepared_tiles \
  --checkpoint /models/unicas.pth \
  --out-dir /results/score_mil_factorial \
  --device cuda:1 \
  --train-devices cuda:0 cuda:1
```

Feature extraction uses `--device`. When `--train-devices` is supplied, two independent workers train on the two listed CUDA devices. Each pseudo-unknown/seed pair stays on one GPU across all four feature conditions, preserving paired training order. The normal single-device behavior remains available by omitting `--train-devices`.

If completed features already exist, run only the two-GPU training stage into a new output directory:

```bash
python score_mil_scale_factorial_multigpu.py \
  --experiment /data/completed_factorial_features \
  --protocol /data/protocol.json \
  --out-dir /results/score_mil_dual_gpu_training \
  --devices cuda:0 cuda:1
```

To inspect the deterministic task assignment without GPUs, add `--plan-only`. The training runner checks the source run's development-only flags, selected-case hash, completed feature groups, and two visible CUDA devices before starting. It writes outputs to a new directory and preserves partial output on worker failure; diagnose failures before starting another run.

For encoder extraction, install `requirements-features.txt` and install a torchvision build that matches the server's existing PyTorch/CUDA build. Avoid letting pip replace a working CUDA PyTorch install. The synthetic demo and metric utilities do not need timm.

## Verification

Run the focused test suite:

```bash
python -m pytest -q
```

The tests exercise model outputs, pseudo-unknown exclusion, calibration isolation, feature-file validation, case pairing, duplicate detection, and the GPU task partition. Physical two-GPU utilization must still be confirmed on the target server with `nvidia-smi` during a real run.

## Data and interpretation limits

- The pseudo-unknown protocol withholds a class from downstream classifier training; it does not establish that the frozen encoder never saw that morphology during pretraining.
- The development validation cases have been reused for method selection and are not an independent test set.
- Patient identity, diagnosis semantics, and image-level acquisition differences have not been fully owner-verified.
- Do not put hospital images, identifiers, case manifests, feature arrays, checkpoints, or per-case predictions in GitHub. Aggregate metrics should retain their protocol and sample-size context.

See [`docs/evaluation_notes.md`](docs/evaluation_notes.md) for interpretation limits and [`docs/data_layout.md`](docs/data_layout.md) for the expected input structure.

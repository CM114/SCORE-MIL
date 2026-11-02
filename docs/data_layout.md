# Local data layout for the real-data runner

This repository deliberately contains no clinical inputs. Keep all case manifests, images, feature arrays, checkpoints, and predictions on approved local storage.

The real-data factorial entry point takes five explicit paths:

1. `--protocol`: development split JSON with `train_rows`, `validation_rows`, `test_rows`, and the predeclared outer unknown class.
2. `--candidate-csv`: candidate tile manifest with case IDs, role/label metadata, candidate index, coordinates, and image path.
3. `--previous`: completed prior feature run containing `features/summary.json` and the ResNet feature manifest used for cache-consistency checks.
4. `--wsl-root`: prepared local JPEG tiles referenced by the candidate manifest.
5. `--checkpoint`: local UniCAS checkpoint matching the SHA-256 fixed in `score_mil_cytology_scale.py`.

The command validates that selected cases are development train/validation only and that their candidate tiles match the prepared tile root. The two-GPU training-only command requires a completed factorial feature run with `config.json`, `selected_cases.json`, `features_complete.json`, and per-case cached feature files.

Do not commit any of these inputs or generated outputs. They can contain direct or indirect patient identifiers.

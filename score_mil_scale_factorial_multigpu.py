#!/usr/bin/env python3
"""Train the paired scale-factorial MIL models on two GPUs.

Each pseudo-unknown/seed pair is assigned intact to one GPU. The four feature
groups within that pair therefore share a device and training order. This
parallelizes independent models without splitting a MIL bag's attention
softmax across devices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


GROUPS = ("resnet_whole", "resnet_native_mean", "unicas_whole", "unicas_native_mean")
DEFAULT_PSEUDO_UNKNOWNS = ("ASC-US", "LISL")


def partition_task_pairs(pseudo_unknowns, seeds, devices):
    """Assign each pseudo-class/seed pair exactly once across two devices."""
    pseudo_unknowns = tuple(str(value) for value in pseudo_unknowns)
    seeds = tuple(int(value) for value in seeds)
    devices = tuple(str(value) for value in devices)
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("exactly two distinct devices are required")
    if not pseudo_unknowns or len(set(pseudo_unknowns)) != len(pseudo_unknowns):
        raise ValueError("pseudo-unknown classes must be nonempty and unique")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    assignments = [
        {"worker_id": index, "device": device, "tasks": []}
        for index, device in enumerate(devices)
    ]
    tasks = [(pseudo, seed) for pseudo in pseudo_unknowns for seed in seeds]
    for index, task in enumerate(tasks):
        assignments[index % len(assignments)]["tasks"].append(task)
    return assignments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_feature_source(experiment_root: Path):
    config_path = experiment_root / "config.json"
    selected_path = experiment_root / "selected_cases.json"
    complete_path = experiment_root / "features_complete.json"
    for path in (config_path, selected_path, complete_path):
        if not path.is_file():
            raise FileNotFoundError(f"required completed-feature input is missing: {path.name}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cases = json.loads(selected_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if config.get("outer_test_images_accessed") is not False or config.get("outer_unknown_images_accessed") is not False:
        raise ValueError("source run does not declare the required development-only boundary")
    if config.get("pseudo_unknown_features_used_for_classifier_training") is not False:
        raise ValueError("source run does not declare pseudo-unknown training exclusion")
    if complete.get("cases") != len(cases) or tuple(complete.get("groups", ())) != GROUPS:
        raise ValueError("feature completion metadata does not match selected cases and groups")
    if any(row.get("role") not in ("train", "validation") for row in cases):
        raise ValueError("selected cases include a role outside development train/validation")
    case_ids = [str(row["case_id"]) for row in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("selected cases contain duplicate identifiers")
    case_hash = hashlib.sha256("\n".join(sorted(case_ids)).encode("utf-8")).hexdigest()
    if case_hash != config.get("selected_case_set_sha256"):
        raise ValueError("selected-case hash differs from the source run configuration")
    seeds = tuple(int(value) for value in config.get("seeds", ()))
    epochs = int(config.get("epochs", 0))
    parents = int(config.get("parents_per_case", 0))
    if not seeds or epochs <= 0 or parents <= 0:
        raise ValueError("source run has invalid seed, epoch, or parent settings")
    return config, cases, seeds, epochs, parents


def _validate_two_cuda_devices(devices):
    import torch

    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("provide exactly two distinct CUDA devices")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA devices are not visible to this process")
    names = {}
    for name in devices:
        device = torch.device(name)
        if device.type != "cuda" or device.index is None:
            raise ValueError(f"device must be explicit CUDA index, got {name!r}")
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {name} is not visible")
        names[name] = torch.cuda.get_device_name(device.index)
    return names


def _build_splits(cases, outer, pseudo_unknowns):
    from score_mil_true_pseudo_openworld import build_inner_partition

    selected_train = {str(row["case_id"]) for row in cases if row["role"] == "train"}
    selected_validation = {str(row["case_id"]) for row in cases if row["role"] == "validation"}
    splits = {}
    for pseudo in pseudo_unknowns:
        partition = build_inner_partition(outer, pseudo)
        train_ids = [str(row["case_id"]) for row in partition["train_rows"]
                     if str(row["case_id"]) in selected_train]
        valid_rows = [row for row in partition["known_validation_rows"] + partition["unknown_validation_rows"]
                      if str(row["case_id"]) in selected_validation]
        if any(str(row["wsl_class"]) == pseudo for row in partition["train_rows"]
               if str(row["case_id"]) in selected_train):
            raise ValueError(f"pseudo-unknown class {pseudo} leaked into classifier training")
        known_order = tuple(partition["known_classes"])
        train_labels = {str(row["wsl_class"]) for row in cases
                        if str(row["case_id"]) in set(train_ids)}
        if set(known_order) != train_labels:
            raise ValueError(f"training support differs from known classes for {pseudo}")
        if not valid_rows or not any(str(row["wsl_class"]) == pseudo for row in valid_rows):
            raise ValueError(f"validation set lacks pseudo-unknown support for {pseudo}")
        splits[pseudo] = {
            "train_ids": train_ids,
            "valid_rows": valid_rows,
            "known_order": known_order,
        }
    return splits


def _worker_main(worker_id, device_name, task_pairs, experiment_root, protocol_path,
                 output_root, cases, pseudo_unknowns, seeds, epochs, parents):
    import torch

    from score_mil_known_calibration_audit import _load, evaluate_known_calibration, split_known_by_hash
    from score_mil_scale_factorial import GROUPS as SCALE_GROUPS, _feature_items
    from score_mil_true_pseudo_openworld import train_one

    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    outer = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    splits = _build_splits(cases, outer, pseudo_unknowns)
    feature_args = SimpleNamespace(out_dir=Path(experiment_root), parents=parents)
    summaries = []
    started = time.perf_counter()
    print(json.dumps({"event": "gpu_worker_started", "worker_id": worker_id,
                      "device": device_name, "tasks": task_pairs}), flush=True)

    for group in SCALE_GROUPS:
        data = _feature_items(cases, group, feature_args)
        if set(data) != {str(row["case_id"]) for row in cases}:
            raise ValueError(f"feature case set mismatch for {group}")
        for pseudo, seed in task_pairs:
            split = splits[pseudo]
            train_ids = split["train_ids"]
            valid_rows = split["valid_rows"]
            valid_ids = [str(row["case_id"]) for row in valid_rows]
            expected_val = {str(row["case_id"]): row for row in valid_rows}
            model_dir = Path(output_root) / pseudo / str(seed) / group
            result = train_one(
                [data[cid] for cid in train_ids],
                [data[cid] for cid in valid_ids],
                split["known_order"], pseudo, "unweighted", seed, epochs, 128, device, model_dir,
            )
            rows = _load(model_dir / "validation_predictions.csv", expected_val, pseudo)
            known_halves = split_known_by_hash(rows, split["known_order"], 20261031)
            unknown = [row for row in rows if int(row["is_pseudo_unknown"])]
            folds = []
            for index in (0, 1):
                folds.append({
                    "calibration_half": index,
                    "scores": {
                        name: evaluate_known_calibration(
                            known_halves[index], known_halves[1 - index], unknown, name, 0.05
                        )
                        for name in ("class_uncertainty", "ordinal_residual", "known_rank_fused")
                    },
                })
            summaries.append({
                "pseudo_unknown": pseudo,
                "seed": seed,
                "group": group,
                "train_n": len(train_ids),
                "known_validation_n": sum(not int(row["is_pseudo_unknown"]) for row in rows),
                "unknown_validation_n": len(unknown),
                "train_order_sha256": result["train_order_sha256"],
                "metrics": result["metrics"],
                "known_calibration_folds": folds,
                "elapsed_seconds": result["elapsed_seconds"],
                "worker_id": worker_id,
                "device": device_name,
            })
            print(json.dumps({"event": "model_complete", "worker_id": worker_id,
                              "device": device_name, "pseudo_unknown": pseudo,
                              "seed": seed, "group": group}), flush=True)
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected_count = len(task_pairs) * len(SCALE_GROUPS)
    if len(summaries) != expected_count:
        raise RuntimeError(f"worker produced {len(summaries)} summaries, expected {expected_count}")
    worker_summary = {
        "worker_id": worker_id,
        "device": device_name,
        "models": len(summaries),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "results": summaries,
    }
    _write_json(Path(output_root) / "workers" / f"worker_{worker_id}.json", worker_summary)
    print(json.dumps({"event": "gpu_worker_complete", "worker_id": worker_id,
                      "device": device_name, "models": len(summaries)}), flush=True)


def train_from_features(experiment_root: Path, protocol_path: Path, output_root: Path,
                        devices=("cuda:0", "cuda:1"), pseudo_unknowns=DEFAULT_PSEUDO_UNKNOWNS,
                        seeds=None, epochs=None):
    experiment_root = Path(experiment_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root == experiment_root or output_root in experiment_root.parents:
        raise ValueError("training output must not replace or contain the source feature run")
    if not protocol_path.is_file():
        raise FileNotFoundError("protocol file is missing")
    if output_root.exists():
        raise FileExistsError("training output exists; preserve it and choose a new output directory")

    source_config, cases, source_seeds, source_epochs, parents = _read_feature_source(experiment_root)
    if seeds is not None and tuple(int(value) for value in seeds) != source_seeds:
        raise ValueError("training seeds must match the predeclared source-run seeds")
    if epochs is not None and int(epochs) != source_epochs:
        raise ValueError("training epochs must match the source run")
    seeds = source_seeds
    epochs = source_epochs
    pseudo_unknowns = tuple(str(value) for value in pseudo_unknowns)
    devices = tuple(str(value) for value in devices)
    assignments = partition_task_pairs(pseudo_unknowns, seeds, devices)
    device_names = _validate_two_cuda_devices(devices)
    outer = json.loads(protocol_path.read_text(encoding="utf-8"))
    _build_splits(cases, outer, pseudo_unknowns)

    output_root.mkdir(parents=True, exist_ok=False)
    config = {
        "method": "independent paired-task workers; no within-bag DataParallel",
        "source_experiment": experiment_root.name,
        "source_config_sha256": _sha256(experiment_root / "config.json"),
        "selected_case_set_sha256": source_config["selected_case_set_sha256"],
        "groups": list(GROUPS),
        "pseudo_unknowns": list(pseudo_unknowns),
        "seeds": list(seeds),
        "epochs": epochs,
        "parents_per_case": parents,
        "devices": [{"device": name, "gpu_name": device_names[name],
                     "tasks": [[pseudo, seed] for pseudo, seed in assignment["tasks"]]}
                    for name, assignment in zip(devices, assignments)],
        "models_expected": len(pseudo_unknowns) * len(seeds) * len(GROUPS),
        "outer_test_images_accessed": False,
        "outer_unknown_images_accessed": False,
        "pseudo_unknown_features_used_for_classifier_training": False,
        "known_only_reference_split_seed": 20261031,
        "known_only_nominal_fpr": 0.05,
        "limitations": [
            "exploratory repeated development validation",
            "pseudo-unknown exclusion applies to downstream classifier training, not encoder pretraining",
            "patient identity and label semantics remain unverified",
        ],
    }
    _write_json(output_root / "config.json", config)
    print(json.dumps({"event": "multigpu_training_config", **config}, ensure_ascii=False), flush=True)

    context = mp.get_context("spawn")
    processes = []
    for assignment in assignments:
        process = context.Process(
            target=_worker_main,
            args=(assignment["worker_id"], assignment["device"], assignment["tasks"],
                  str(experiment_root), str(protocol_path), str(output_root), cases,
                  pseudo_unknowns, seeds, epochs, parents),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [{"pid": process.pid, "exitcode": process.exitcode}
                for process in processes if process.exitcode != 0]
    if failures:
        _write_json(output_root / "worker_failures.json", {"failures": failures,
                                                            "outputs_preserved": True,
                                                            "automatic_retry": False})
        raise RuntimeError(f"GPU worker failure(s); partial outputs preserved: {failures}")

    worker_summaries = [json.loads((output_root / "workers" / f"worker_{i}.json").read_text(encoding="utf-8"))
                        for i in range(len(assignments))]
    results = [row for worker in worker_summaries for row in worker["results"]]
    results.sort(key=lambda row: (row["pseudo_unknown"], row["seed"], row["group"]))
    expected_count = config["models_expected"]
    if len(results) != expected_count:
        raise RuntimeError(f"collected {len(results)} models, expected {expected_count}")
    order_hashes = defaultdict(set)
    for row in results:
        order_hashes[(row["pseudo_unknown"], row["seed"])].add(row["train_order_sha256"])
    if any(len(values) != 1 for values in order_hashes.values()):
        raise RuntimeError("paired training order differs across feature groups")
    summary = {
        "config": config,
        "worker_elapsed_seconds": {str(worker["worker_id"]): worker["elapsed_seconds"]
                                   for worker in worker_summaries},
        "results": results,
    }
    _write_json(output_root / "training_summary.json", summary)
    print(json.dumps({"event": "multigpu_training_complete", "models": len(results),
                      "output_dir": output_root.name}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True,
                        help="completed development-only factorial feature run")
    parser.add_argument("--protocol", type=Path, required=True,
                        help="development protocol JSON used to build the feature run")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new directory for two-GPU training outputs")
    parser.add_argument("--devices", nargs=2, default=("cuda:0", "cuda:1"),
                        help="two distinct CUDA device names")
    parser.add_argument("--pseudo-unknowns", nargs="+", default=DEFAULT_PSEUDO_UNKNOWNS)
    parser.add_argument("--plan-only", action="store_true",
                        help="show deterministic task assignment without requiring CUDA")
    args = parser.parse_args()
    source_config, _, source_seeds, _, _ = _read_feature_source(args.experiment)
    assignment = partition_task_pairs(args.pseudo_unknowns, source_seeds, args.devices)
    if args.plan_only:
        print(json.dumps({"source_experiment": Path(args.experiment).name,
                          "devices": assignment,
                          "models_per_device": [len(item["tasks"]) * len(GROUPS) for item in assignment],
                          "models_expected": len(args.pseudo_unknowns) * len(source_seeds) * len(GROUPS)},
                         ensure_ascii=False, indent=2))
        return
    if output_is_within(args.out_dir, args.experiment):
        raise ValueError("choose an output directory outside the source feature run")
    train_from_features(args.experiment, args.protocol, args.out_dir, args.devices,
                        args.pseudo_unknowns, source_seeds, source_config["epochs"])


def output_is_within(output_root, experiment_root):
    output_root = Path(output_root).resolve()
    experiment_root = Path(experiment_root).resolve()
    return output_root == experiment_root or experiment_root in output_root.parents


if __name__ == "__main__":
    mp.freeze_support()
    main()

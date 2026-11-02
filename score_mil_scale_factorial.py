#!/usr/bin/env python3
"""Paired exploratory encoder/view pilot, restricted to development JPEGs.

Two frozen encoders see identical parent tiles as a whole-resized image and
as the mean of 25 native-resolution crops.  Checkpoints are per case and can
be resumed only when the complete recorded configuration matches.  Pseudo
unknowns are excluded from classifier training, but their validation images
are embedded for evaluation.  Results are not independent clinical tests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from score_mil_candidate_manifest import _select_training_cases
from score_mil_cytology_scale import UNICAS_SHA256, _sha256, build_unicas_model, extract_native_crops
from score_mil_dev_candidate_manifest import development_cases
from score_mil_known_calibration_audit import _load, evaluate_known_calibration, split_known_by_hash
from score_mil_tile_feature_pilot import _build_encoder
from score_mil_true_pseudo_openworld import build_inner_partition, train_one


GROUPS = ("resnet_whole", "resnet_native_mean", "unicas_whole", "unicas_native_mean")


def select_pilot_cases(protocol: dict, train_count: int) -> list[dict]:
    if train_count <= 0:
        raise ValueError("training case count must be positive")
    development = development_cases(protocol)
    available = {str(r["case_id"]): r for r in development}
    selected = _select_training_cases(protocol, train_count)
    if len(selected) != min(train_count, len(protocol["train_rows"])):
        raise ValueError("unexpected selection count")
    ids = {str(r["case_id"]) for r in selected}
    if len(ids) != len(selected):
        raise ValueError("duplicate selection")
    chosen = [dict(available[str(r["case_id"])]) for r in selected]
    chosen += [dict(r) for r in development if r["role"] == "validation"]
    return sorted(chosen, key=lambda r: (r["role"], str(r["case_id"])))


def smoke_cases(cases: list[dict], per_class: int = 2) -> list[dict]:
    groups = defaultdict(list)
    for row in cases:
        groups[(row["role"], row["wsl_class"])].append(row)
    if any(len(v) < per_class for v in groups.values()):
        raise ValueError("smoke sample lacks class support")
    return sorted([r for key in sorted(groups) for r in groups[key][:per_class]],
                  key=lambda r: (r["role"], str(r["case_id"])))


def validate_subset_candidates(rows, cases, root: Path, budget: int) -> None:
    expected = {str(row["case_id"]): row for row in cases}
    if not expected or len(expected) != len(cases) or budget <= 0:
        raise ValueError("nonempty unique cases and positive budget required")
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        cid = str(row["case_id"])
        if cid not in expected:
            raise ValueError("candidate is not in the selected development cases")
        ref = expected[cid]
        if any(str(row.get(k, "")) != str(ref.get(k, "")) for k in ("role", "wsl_class", "hospital", "binary_label")):
            raise ValueError("candidate metadata mismatch")
        index = int(row["candidate_index"])
        if (cid, index) in seen:
            raise ValueError("duplicate candidate index")
        seen.add((cid, index))
        grouped[cid].append(index)
        filename = f"{int(row['grid_x'])}_{int(row['grid_y'])}.jpg"
        desired = root / str(ref["wsl_class"]) / str(ref["hospital"]) / cid / "img" / filename
        if Path(row["image_path"]).resolve() != desired.resolve():
            raise ValueError("candidate image path is not the expected case tile")
    if set(grouped) != set(expected) or any(sorted(v) != list(range(budget)) for v in grouped.values()):
        raise ValueError("candidate cases or indices do not match the fixed paired budget")


def collapse_views(features):
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] != 26 or array.shape[2] == 0:
        raise ValueError("each parent must have exactly 26 encoded views")
    if not np.isfinite(array).all():
        raise ValueError("features must be finite")
    return array[:, 0, :], array[:, 1:, :].mean(axis=1, dtype=np.float32)


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _case_file(out_dir: Path, backbone: str, cid: str) -> Path:
    return out_dir / "features" / backbone / (hashlib.sha256(cid.encode("utf-8")).hexdigest() + ".npz")


def _check_case_file(path: Path, budget: int, dim: int, verify_crops: bool = True) -> None:
    with np.load(path, allow_pickle=False) as item:
        if set(item.files) != {"whole", "native_mean", "native_crops"}:
            raise ValueError("unexpected per-case arrays")
        for key in ("whole", "native_mean"):
            if item[key].shape != (budget, dim) or not np.isfinite(item[key]).all():
                raise ValueError("invalid per-case feature shape or values")
        if verify_crops:
            crops = item["native_crops"]
            if crops.shape != (budget, 25, dim) or not np.isfinite(crops).all():
                raise ValueError("invalid native crop features")
            if not np.allclose(crops.mean(axis=1, dtype=np.float32), item["native_mean"], rtol=1e-6, atol=1e-6):
                raise ValueError("native crop mean does not match saved mean")


def _save_case_file(path: Path, whole, native, native_crops) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("refusing to overwrite a previous feature file")
    tmp = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as handle:
        np.savez(handle, whole=whole, native_mean=native, native_crops=native_crops)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError("feature file was created concurrently")
    os.replace(tmp, path)


class ParentDataset:
    def __init__(self, rows):
        import torch
        self.rows = rows
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        with Image.open(self.rows[index]["image_path"]) as handle:
            rgb = handle.convert("RGB")
        if rgb.size != (1024, 1024):
            raise ValueError("parent JPEG must be 1024x1024")
        whole = rgb.resize((224, 224), resample=Image.Resampling.BILINEAR)
        views = [whole, *extract_native_crops(rgb, 224, 5)]
        array = np.stack([np.asarray(view, dtype=np.uint8) for view in views])
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
        return (tensor - self.mean) / self.std, index


def _encode(model, views, device, batch_size):
    import torch
    out = []
    with torch.inference_mode():
        for offset in range(0, len(views), batch_size):
            chunk = views[offset:offset + batch_size]
            n = len(chunk)
            if n < batch_size:
                chunk = torch.cat([chunk, chunk[:1].expand(batch_size - n, -1, -1, -1)], dim=0)
            out.append(model(chunk.to(device, non_blocking=True)).detach().cpu().numpy()[:n])
    return np.concatenate(out, axis=0)


def _extract(backbone, rows, cases, args, config):
    import torch
    from torch.utils.data import DataLoader
    from ndpi_mil_baseline import group_feature_manifest, load_case_bag, read_csv

    dim = 2048 if backbone == "resnet" else 1024
    completed = set()
    for case in cases:
        path = _case_file(args.out_dir, backbone, str(case["case_id"]))
        if path.exists():
            _check_case_file(path, args.parents, dim)
            completed.add(str(case["case_id"]))
    pending_rows = [r for r in rows if str(r["case_id"]) not in completed]
    if not pending_rows:
        print(json.dumps({"event": "backbone_already_complete", "backbone": backbone, "cases": len(completed)}), flush=True)
        return
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < 8 * 1024 ** 3:
        raise RuntimeError("less than 8 GiB GPU memory available")
    if backbone == "resnet":
        model, checkpoint = _build_encoder(device)
        if _sha256(checkpoint) != config["resnet_checkpoint_sha256"]:
            raise ValueError("ResNet checkpoint differs from prior features")
        groups = group_feature_manifest(read_csv(args.previous / "features" / "spatial" / "feature_manifest.csv"))
    else:
        if _sha256(args.checkpoint) != UNICAS_SHA256:
            raise ValueError("UniCAS checksum mismatch")
        model = build_unicas_model()
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.to(device).eval()
        groups = None
    loader = DataLoader(ParentDataset(pending_rows), batch_size=2, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=(args.workers > 0))
    started = time.perf_counter()
    cid_current, whole_parts, native_parts, crop_parts = None, [], [], []
    maximum_relative_error, minimum_cosine = 0.0, 1.0
    num_done = len(completed)
    for views, indices in loader:
        batch_n = len(indices)
        batch_rows = [pending_rows[int(i)] for i in indices]
        cids = {str(r["case_id"]) for r in batch_rows}
        if len(cids) != 1:
            raise ValueError("loader batch crosses case boundary")
        cid = cids.pop()
        if cid_current is None:
            cid_current = cid
        if cid != cid_current:
            raise ValueError("case boundary before the expected full bag")
        values = _encode(model, views.reshape(batch_n * 26, 3, 224, 224), device, 64 if backbone == "resnet" else args.unicas_batch)
        if values.shape != (batch_n * 26, dim):
            raise ValueError("unexpected encoder shape")
        view_features = values.reshape(batch_n, 26, dim)
        whole, native = collapse_views(view_features)
        if backbone == "resnet" and not whole_parts:
            cached, _, _, meta = load_case_bag(groups[cid])
            original = next(r for r in cases if str(r["case_id"]) == cid)
            if cached.shape != (64, dim) or str(meta["split"]) != original["role"] or str(meta["wsl_class"]) != original["wsl_class"]:
                raise ValueError("old ResNet cached metadata or shape mismatch")
            a, b = whole[0].astype(np.float64), cached[0].astype(np.float64)
            rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if rel > 0.01 or cosine < 0.9999:
                raise ValueError("first parent feature disagrees with old cached feature")
            maximum_relative_error = max(maximum_relative_error, rel)
            minimum_cosine = min(minimum_cosine, cosine)
        whole_parts.append(whole)
        native_parts.append(native)
        crop_parts.append(view_features[:, 1:, :])
        count = sum(len(v) for v in whole_parts)
        if count == args.parents:
            target = _case_file(args.out_dir, backbone, cid)
            _save_case_file(target, np.concatenate(whole_parts), np.concatenate(native_parts), np.concatenate(crop_parts))
            _check_case_file(target, args.parents, dim)
            cid_current, whole_parts, native_parts, crop_parts = None, [], [], []
            num_done += 1
            if num_done % 10 == 0 or num_done == len(cases):
                print(json.dumps({"event": "feature_progress", "backbone": backbone, "cases_completed": num_done,
                                  "case_total": len(cases), "elapsed_seconds_this_run": round(time.perf_counter()-started, 1)}, ensure_ascii=False), flush=True)
        elif count > args.parents:
            raise ValueError("too many parents in a case")
    if cid_current is not None or num_done != len(cases):
        raise ValueError("incomplete final case")
    print(json.dumps({"event": "backbone_complete", "backbone": backbone, "cases": num_done,
                      "elapsed_seconds_this_run": round(time.perf_counter()-started, 1),
                      "max_resnet_cache_relative_l2_this_run": maximum_relative_error if backbone == "resnet" else None,
                      "min_resnet_cache_cosine_this_run": minimum_cosine if backbone == "resnet" else None}), flush=True)
    del model
    torch.cuda.empty_cache()


def _feature_items(cases, group, args):
    backbone, name = group.split("_", 1)
    dim = 2048 if backbone == "resnet" else 1024
    output = {}
    for row in cases:
        cid = str(row["case_id"])
        path = _case_file(args.out_dir, backbone, cid)
        _check_case_file(path, args.parents, dim, verify_crops=False)
        with np.load(path, allow_pickle=False) as item:
            output[cid] = {**row, "features": item[name].astype(np.float32, copy=True)}
    return output


def _train(cases, outer, args, config):
    import torch
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    selected_train = {str(r["case_id"]) for r in cases if r["role"] == "train"}
    selected_validation = {str(r["case_id"]) for r in cases if r["role"] == "validation"}
    summaries = []
    data = {g: _feature_items(cases, g, args) for g in GROUPS}
    for pseudo in ("ASC-US", "LISL"):
        partition = build_inner_partition(outer, pseudo)
        train_ids = [str(r["case_id"]) for r in partition["train_rows"] if str(r["case_id"]) in selected_train]
        valid_rows = [r for r in partition["known_validation_rows"] + partition["unknown_validation_rows"] if str(r["case_id"]) in selected_validation]
        val_ids = [str(r["case_id"]) for r in valid_rows]
        known_order = partition["known_classes"]
        if any(str(data[GROUPS[0]][cid]["wsl_class"]) == pseudo for cid in train_ids):
            raise ValueError("pseudo unknown included in classifier training")
        if set(known_order) != {str(data[GROUPS[0]][cid]["wsl_class"]) for cid in train_ids}:
            raise ValueError("selected training classes lack support")
        expected_val = {str(r["case_id"]): r for r in valid_rows}
        for seed in args.seeds:
            order_hashes = set()
            for group in GROUPS:
                out = args.out_dir / "training" / pseudo / str(seed) / group
                if out.exists():
                    required = [out / "result.json", out / "fixed_epoch.pt", out / "validation_predictions.csv"]
                    if not args.resume or not all(path.is_file() for path in required):
                        raise FileExistsError("incomplete training output: preserve it and diagnose before resuming")
                    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
                    expected_counts = dict(Counter(str(data[group][cid]["wsl_class"]) for cid in train_ids))
                    if (result["seed"] != seed or result["pseudo_unknown"] != pseudo or result["mode"] != "unweighted"
                            or result["epochs"] != args.epochs or result["known_order"] != known_order
                            or result["train_class_counts"] != expected_counts):
                        raise ValueError("existing training result does not match the current experiment")
                    checkpoint = torch.load(out / "fixed_epoch.pt", map_location="cpu", weights_only=True)
                    if checkpoint["input_dim"] != data[group][train_ids[0]]["features"].shape[1] or checkpoint["seed"] != seed:
                        raise ValueError("existing checkpoint metadata mismatch")
                else:
                    result = train_one([data[group][cid] for cid in train_ids], [data[group][cid] for cid in val_ids],
                                       known_order, pseudo, "unweighted", seed, args.epochs, 128, device, out)
                order_hashes.add(result["train_order_sha256"])
                rows = _load(out / "validation_predictions.csv", expected_val, pseudo)
                halves = split_known_by_hash(rows, known_order, 20261031)
                unknown = [r for r in rows if int(r["is_pseudo_unknown"])]
                folds = []
                for i in (0, 1):
                    folds.append({"calibration_half": i,
                                  "scores": {name: evaluate_known_calibration(halves[i], halves[1-i], unknown, name, 0.05)
                                             for name in ("class_uncertainty", "ordinal_residual", "known_rank_fused")}})
                entry = {"pseudo_unknown": pseudo, "seed": seed, "group": group, "train_n": len(train_ids),
                         "known_validation_n": sum(not int(r["is_pseudo_unknown"]) for r in rows),
                         "unknown_validation_n": len(unknown), "train_order_sha256": result["train_order_sha256"],
                         "metrics": result["metrics"], "known_calibration_folds": folds,
                         "elapsed_seconds": result["elapsed_seconds"]}
                summaries.append(entry)
                _json(args.out_dir / "training_partial_summary.json", {"results": summaries})
                print(json.dumps({"event": "model_complete", "pseudo_unknown": pseudo, "seed": seed, "group": group}, ensure_ascii=False), flush=True)
            if len(order_hashes) != 1:
                raise ValueError("paired training orders differ")
    _json(args.out_dir / "training_summary.json", {"config": config, "results": summaries})
    print(json.dumps({"event": "training_complete", "models": len(summaries)}), flush=True)


def dispatch_training(cases, outer, args, config):
    """Run the original single-device loop or the separate two-GPU workers."""
    devices = getattr(args, "train_devices", None)
    if not devices:
        return _train(cases, outer, args, config)
    from score_mil_scale_factorial_multigpu import train_from_features

    output_dir = args.out_dir / "training_two_gpu"
    return train_from_features(
        args.out_dir,
        args.protocol,
        output_dir,
        devices=devices,
        pseudo_unknowns=("ASC-US", "LISL"),
        seeds=args.seeds,
        epochs=args.epochs,
    )


def run(args):
    import torch
    import timm
    if args.train_devices:
        from score_mil_scale_factorial_multigpu import _validate_two_cuda_devices
        _validate_two_cuda_devices(args.train_devices)
    outer = json.loads(args.protocol.read_text(encoding="utf-8"))
    cases = select_pilot_cases(outer, args.train_count)
    if args.smoke:
        cases = smoke_cases(cases)
    by_case = {str(r["case_id"]): r for r in cases}
    with args.candidate_csv.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if str(row["case_id"]) in by_case and int(row["candidate_index"]) < args.parents]
    validate_subset_candidates(rows, cases, args.wsl_root, args.parents)
    rows.sort(key=lambda r: (str(r["case_id"]), int(r["candidate_index"])))
    old = json.loads((args.previous / "features" / "summary.json").read_text(encoding="utf-8"))
    if old["outer_test_images_accessed"] or old["outer_unknown_images_accessed"]:
        raise ValueError("prior features violate development-only boundary")
    if args.parents % 2 or args.parents <= 0 or args.parents > 64:
        raise ValueError("parent count must be even and between 2 and 64")
    if not args.smoke and (args.parents != 64 or args.train_count != 256 or args.epochs != 12 or args.seeds != [20261030, 20261031, 20261032]):
        raise ValueError("full pilot must use the predeclared settings")
    case_hash = hashlib.sha256("\n".join(sorted(by_case)).encode("utf-8")).hexdigest()
    config = {"scope": "selected outer known development train and validation only", "smoke": args.smoke,
              "outer_unknown_images_accessed": False, "outer_test_images_accessed": False,
              "pseudo_unknown_features_used_for_classifier_training": False,
              "train_cases": sum(r["role"] == "train" for r in cases), "validation_cases": sum(r["role"] == "validation" for r in cases),
              "train_class_counts": dict(Counter(r["wsl_class"] for r in cases if r["role"] == "train")),
              "validation_class_counts": dict(Counter(r["wsl_class"] for r in cases if r["role"] == "validation")),
              "groups": list(GROUPS), "parents_per_case": args.parents, "native_crops_per_parent": 25,
              "native_crop_features_saved_for_future_analysis": True,
              "case_selection": "fixed class/source round-robin in known outer training; all development validation",
              "resize": "RGB bilinear 1024 to 224", "native": "5x5 224 pixel overlapping crops, mean of embedding vectors",
              "normalization": "ImageNet mean/std", "pretraining_overlap_unknown": True,
              "resnet_checkpoint_sha256": old["encoder_checkpoint_sha256"], "unicas_checkpoint_sha256": UNICAS_SHA256,
              "protocol_sha256": _sha256(args.protocol), "candidate_sha256": _sha256(args.candidate_csv),
              "script_sha256": _sha256(Path(__file__).resolve()), "selected_case_set_sha256": case_hash,
              "seeds": args.seeds, "epochs": args.epochs, "device": args.device,
              "training_devices": args.train_devices if args.train_devices else [args.device],
              "training_parallel_mode": "independent_gpu_workers" if args.train_devices else "single_device_serial",
              "unicas_batch": args.unicas_batch, "workers": args.workers,
              "torch_version": torch.__version__, "timm_version": timm.__version__,
              "limitations": ["exploratory repeated development validation", "source/label case selection is not population random", "25-crop mean may dilute rare cells", "patient and label semantics unverified", "source resolution and pretrained overlap unverified"]}
    if args.resume:
        if json.loads((args.out_dir / "config.json").read_text(encoding="utf-8")) != config:
            raise ValueError("resume configuration does not match existing experiment")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        _json(args.out_dir / "config.json", config)
        _json(args.out_dir / "selected_cases.json", cases)
    print(json.dumps({"event": "pilot_config", **config}, ensure_ascii=False), flush=True)
    torch.set_num_threads(2)
    _extract("resnet", rows, cases, args, config)
    _extract("unicas", rows, cases, args, config)
    _json(args.out_dir / "features_complete.json", {"cases": len(cases), "parents_per_case": args.parents,
                                                     "backbones": ["resnet", "unicas"], "groups": list(GROUPS)})
    dispatch_training(cases, outer, args, config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--candidate-csv", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--wsl-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-count", type=int, default=256)
    p.add_argument("--parents", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[20261030, 20261031, 20261032])
    p.add_argument("--unicas-batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--train-devices", nargs=2, default=None, metavar=("CUDA0", "CUDA1"),
                   help="train independent paired model tasks on two GPUs")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.workers < 0 or args.unicas_batch <= 0:
        raise ValueError("invalid worker count or batch size")
    run(args)


if __name__ == "__main__":
    main()

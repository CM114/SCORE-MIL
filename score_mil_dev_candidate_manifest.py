#!/usr/bin/env python3
"""Build paired JPEG tile candidates for known outer development cases only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from score_mil_candidate_manifest import _case_seed, _parse_coordinate, select_candidates


def development_cases(protocol: dict) -> list[dict]:
    unknown = str(protocol["unknown_class"])
    known = set(str(x) for x in protocol["known_classes"])
    if unknown in known:
        raise ValueError("unknown label is present among known labels")
    result = []
    seen = set()
    test_ids = {str(cid) for cid in protocol.get("test_ids", [])}
    for role in ("train", "validation"):
        for original in protocol[f"{role}_rows"]:
            row = dict(original)
            cid = str(row["case_id"])
            label = str(row["wsl_class"])
            if cid in seen or cid in test_ids:
                raise ValueError("case ID overlap across development/test roles")
            if label == unknown or label not in known:
                raise ValueError("unknown or non-known label in development cases")
            row["role"] = role
            seen.add(cid)
            result.append(row)
    if not result or not any(x["role"] == "train" for x in result) or not any(x["role"] == "validation" for x in result):
        raise ValueError("training and validation cases are required")
    return sorted(result, key=lambda row: (str(row["role"]), str(row["case_id"])))


def build_development_candidates(protocol: dict, wsl_root: Path, budget: int, bins: int, seed: int):
    cases = development_cases(protocol)
    manifests = {"random": [], "spatial": []}
    audits = []
    errors = []
    for index, row in enumerate(cases, 1):
        cid = str(row["case_id"])
        image_dir = wsl_root / str(row["wsl_class"]) / str(row["hospital"]) / cid / "img"
        try:
            filenames = [p.name for p in image_dir.iterdir() if p.is_file()]
            selected = {
                policy: select_candidates(filenames, policy, budget, bins, _case_seed(seed, cid))
                for policy in ("random", "spatial")
            }
            for policy, records in selected.items():
                for candidate_index, item in enumerate(records):
                    manifests[policy].append({
                        "case_id": cid,
                        "role": row["role"],
                        "wsl_class": str(row["wsl_class"]),
                        "hospital": str(row["hospital"]),
                        "binary_label": str(row["binary_label"]),
                        "candidate_index": candidate_index,
                        "grid_x": item["grid_x"],
                        "grid_y": item["grid_y"],
                        "bin_x": item["bin_x"],
                        "bin_y": item["bin_y"],
                        "image_path": str(image_dir / item["filename"]),
                    })
            random_names = {r["filename"] for r in selected["random"]}
            spatial_names = {r["filename"] for r in selected["spatial"]}
            audits.append({
                "case_id": cid,
                "role": row["role"],
                "wsl_class": str(row["wsl_class"]),
                "hospital": str(row["hospital"]),
                "available_coordinate_tiles": sum(_parse_coordinate(name) is not None for name in filenames),
                "random_occupied_bins": len({(r["bin_x"], r["bin_y"]) for r in selected["random"]}),
                "spatial_occupied_bins": len({(r["bin_x"], r["bin_y"]) for r in selected["spatial"]}),
                "overlap_count": len(random_names & spatial_names),
            })
        except (OSError, ValueError) as exc:
            errors.append({"case_id": cid, "role": row["role"], "error": f"{type(exc).__name__}: {exc}"})
        if index % 100 == 0 or index == len(cases):
            print(json.dumps({"event": "candidate_progress", "processed": index, "total": len(cases), "errors": len(errors)}, ensure_ascii=False), flush=True)
    return manifests, audits, errors


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifests, audits, errors = build_development_candidates(protocol, args.wsl_root, args.budget, args.bins, args.seed)
    if not audits:
        raise RuntimeError("no development cases were sampled")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for policy, rows in manifests.items():
        _write_csv(args.out_dir / f"{policy}_candidates.csv", rows)
    _write_csv(args.out_dir / "case_audit.csv", audits)
    if errors:
        _write_csv(args.out_dir / "case_errors.csv", errors)
    summary = {
        "scope": "outer known train and validation cases only",
        "outer_unknown_class": str(protocol["unknown_class"]),
        "outer_unknown_images_accessed": False,
        "outer_test_images_accessed": False,
        "candidate_selection": "coordinates and case-seeded RNG only; no labels, detector outputs or annotations",
        "roles": dict(Counter(row["role"] for row in audits)),
        "classes": dict(Counter(row["wsl_class"] for row in audits)),
        "sources": dict(Counter(row["hospital"] for row in audits)),
        "cases_successful": len(audits),
        "case_errors": len(errors),
        "budget": args.budget,
        "bins": args.bins,
        "seed": args.seed,
        "random_rows": len(manifests["random"]),
        "spatial_rows": len(manifests["spatial"]),
        "mean_random_occupied_bins": float(mean(row["random_occupied_bins"] for row in audits)),
        "mean_spatial_occupied_bins": float(mean(row["spatial_occupied_bins"] for row in audits)),
        "median_available_tiles": float(median(row["available_coordinate_tiles"] for row in audits)),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "limitation": "grid coverage is not cell/lesion coverage; patient identity and label semantics remain unconfirmed",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "candidate_complete", **summary}, ensure_ascii=False), flush=True)
    if errors:
        raise RuntimeError("one or more development cases failed; inspect case_errors.csv before extracting features")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsl-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261030)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Native-scale crop geometry and a safe synthetic UniCAS compatibility probe.

The probe reads only a public, checksum-verified checkpoint and synthetic
pixels; it neither reads hospital data nor changes the active conda environment.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

UNICAS_SHA256 = "778e8da742d8b2775e70d934be9ac19536c6287d755ecdb429c097a78ee419e5"


def grid_starts(length: int, crop_size: int, grid_size: int) -> list[int]:
    if length <= 0 or crop_size <= 0 or grid_size <= 0 or length < crop_size:
        raise ValueError("parent must be at least as large as the crop")
    if grid_size == 1:
        if length != crop_size:
            raise ValueError("one crop cannot guarantee complete coverage")
        return [0]
    starts = [round(i * (length - crop_size) / (grid_size - 1)) for i in range(grid_size)]
    if len(set(starts)) != grid_size or any(b - a > crop_size for a, b in zip(starts, starts[1:])):
        raise ValueError("grid cannot guarantee coverage with unique crops")
    return starts


def crop_boxes(width: int, height: int, crop_size: int = 224, grid_size: int = 5) -> list[tuple[int, int, int, int]]:
    xs = grid_starts(width, crop_size, grid_size)
    ys = grid_starts(height, crop_size, grid_size)
    return [(x, y, x + crop_size, y + crop_size) for y in ys for x in xs]


def extract_native_crops(image: Image.Image, crop_size: int = 224, grid_size: int = 5) -> list[Image.Image]:
    rgb = image.convert("RGB")
    return [rgb.crop(box) for box in crop_boxes(rgb.width, rgb.height, crop_size, grid_size)]


def compare_synthetic_embeddings(first, second) -> dict:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != 1 or a.shape[1] == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("synthetic embeddings must have identical finite [1, dimension] shape")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("synthetic embedding norm must be non-zero")
    return {"max_abs_difference": float(np.max(np.abs(a - b))),
            "mean_abs_difference": float(np.mean(np.abs(a - b))),
            "cosine_similarity": float(np.clip(np.sum(a * b) / (na * nb), -1.0, 1.0)),
            "exact_match": bool(np.array_equal(a, b))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_unicas_model():
    import timm
    import torch

    return timm.models.VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5,
        mlp_ratio=2.671875 * 2, mlp_layer=functools.partial(timm.layers.mlp.GluMlp, gate_last=False),
        act_layer=torch.nn.SiLU, no_embed_class=False, img_size=224, num_classes=0, in_chans=3,
    )


def run_probe(args) -> dict:
    import timm
    import torch

    if _sha256(args.checkpoint) != UNICAS_SHA256:
        raise ValueError("public checkpoint SHA-256 does not match the published digest")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261031)
    torch.use_deterministic_algorithms(True)
    model = build_unicas_model()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    compatibility = model.load_state_dict(state, strict=True)
    if compatibility.missing_keys or compatibility.unexpected_keys:
        raise RuntimeError("checkpoint is not an exact architecture match")
    model.eval()
    x = torch.linspace(0.0, 1.0, steps=3 * 224 * 224, dtype=torch.float32).reshape(1, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1)
    with torch.inference_mode():
        output = model((x - mean) / std)
    array = output.detach().cpu().numpy()
    if array.shape != (1, 1024) or not np.isfinite(array).all():
        raise RuntimeError("unexpected synthetic feature shape or non-finite values")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.out_dir / "synthetic_feature.npy", array, allow_pickle=False)
    result = {
        "scope": "public UniCAS checkpoint and synthetic pixels only", "hospital_images_read": False,
        "checkpoint_sha256": UNICAS_SHA256, "weights_only_load": True, "strict_load": True,
        "timm_version": timm.__version__, "official_timm_version": "1.0.5", "torch_version": torch.__version__,
        "device": "cpu", "threads": args.threads, "output_shape": list(array.shape),
        "output_l2_norm": float(np.linalg.norm(array)), "output_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "output_first_8": [float(x) for x in array[0, :8]], "native_grid_starts_1024": grid_starts(1024, 224, 5),
        "limitations": ["synthetic compatibility only; not a performance or provenance validation", "reference timm version may differ"],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.threads <= 0:
        raise ValueError("thread count must be positive")
    run_probe(args)


if __name__ == "__main__":
    main()

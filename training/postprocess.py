"""Turn raw Blender bakes into training samples in the delighter app's input format.

    python postprocess.py --baked F:/DelighterData/baked --source F:/DelighterData/source/models \
        --out F:/DelighterData/dataset

Per asset writes 16-bit PNGs: albedo (sRGB), lit_XX (sRGB, exposure and white
balance normalised towards the albedo), normal_os, ao, plus an 8-bit UV mask and
a preview sheet. A report flags assets whose baked albedo drifts from the source
diffuse texture.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

PREVIEW = 256


def read_exr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Cannot read {path}")
    return cv2.cvtColor(image[:, :, :3].astype(np.float32), cv2.COLOR_BGR2RGB)


def to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def write_png16(path: Path, rgb: np.ndarray) -> None:
    data = (np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16)
    if data.ndim == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), data)


def rasterise_mask(triangles: np.ndarray, size: int) -> np.ndarray:
    shift = 4  # sub-pixel precision for fillPoly
    points = np.empty_like(triangles)
    points[..., 0] = triangles[..., 0] * size
    points[..., 1] = (1.0 - triangles[..., 1]) * size
    mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask, np.round(points * (1 << shift)).astype(np.int32), 255, lineType=cv2.LINE_8, shift=shift)
    # One pixel of padding so island borders are not cut off by the multiply in the app.
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def normalise_lit(lit: np.ndarray, albedo: np.ndarray, valid: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Scale exposure and partially remove the light's colour cast, like a photographer's auto WB."""
    lit_mean = lit[valid].mean(axis=0) + 1e-6
    albedo_mean = albedo[valid].mean(axis=0) + 1e-6
    # Skewed below 1: under-exposure is common in scans, blown highlights destroy information.
    exposure = float(albedo_mean.mean() / lit_mean.mean()) * rng.uniform(0.6, 1.1)
    peak = np.percentile(lit[valid].max(axis=1), 99.5) * exposure
    if peak > 1.0:
        exposure /= peak  # keep at most ~0.5% of texels clipped
    wb_strength = rng.uniform(0.5, 1.0)
    cast = (lit_mean / lit_mean.mean()) / (albedo_mean / albedo_mean.mean())
    gains = exposure / np.power(cast, wb_strength)
    return lit * gains, {"exposure": float(exposure),"wb_strength": wb_strength, "gains": gains.round(4).tolist()}


def contact_sheet(rows: list[list[np.ndarray]]) -> np.ndarray:
    tiles = [[cv2.resize(t, (PREVIEW, PREVIEW), interpolation=cv2.INTER_AREA) for t in row] for row in rows]
    width = max(len(r) for r in tiles)
    blank = np.zeros((PREVIEW, PREVIEW, 3), np.float32)
    return np.vstack([np.hstack(r + [blank] * (width - len(r))) for r in tiles])


def process(asset_dir: Path, source_dir: Path, out_dir: Path, seed: int) -> dict:
    rng = random.Random(f"{seed}:{asset_dir.name}")
    meta = json.loads((asset_dir / "meta.json").read_text())
    albedo = read_exr(asset_dir / "albedo.exr")
    normal = read_exr(asset_dir / "normal_os.exr")
    ao = read_exr(asset_dir / "ao.exr")[:, :, 0]
    size = albedo.shape[0]
    mask = rasterise_mask(np.load(asset_dir / "uv_tris.npy"), size)
    valid = mask > 0

    out_dir.mkdir(parents=True, exist_ok=True)
    albedo_srgb = to_srgb(albedo)
    write_png16(out_dir / "albedo.png", albedo_srgb)
    write_png16(out_dir / "normal_os.png", normal)
    write_png16(out_dir / "ao.png", ao)
    cv2.imwrite(str(out_dir / "mask.png"), mask)

    report = {"asset": asset_dir.name, "coverage": float(valid.mean()), "variations": []}
    diffuse = next(source_dir.glob(f"{asset_dir.name}/textures/*_diff_*"), None)
    if diffuse is not None:
        source = cv2.cvtColor(cv2.imread(str(diffuse), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        source = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
        core = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0  # ignore seam pixels
        report["albedo_vs_source_mae"] = float(np.abs(albedo_srgb[core] - source[core]).mean())

    previews = [albedo_srgb, normal, np.repeat(ao[:, :, None], 3, 2), np.repeat(mask[:, :, None] / 255.0, 3, 2)]
    lit_previews = []
    for variation in meta["variations"]:
        lit = read_exr(asset_dir / variation["file"])
        normalised, params = normalise_lit(lit, albedo, valid, rng)
        clipped = float((normalised[valid] > 1.0).any(axis=1).mean())
        lit_srgb = to_srgb(normalised)
        write_png16(out_dir / variation["file"].replace(".exr", ".png"), lit_srgb)
        lit_previews.append(lit_srgb)
        report["variations"].append({**variation, **params, "clipped_fraction": clipped})

    rows = [previews] + [lit_previews[i:i + 4] for i in range(0, len(lit_previews), 4)]
    sheet = contact_sheet(rows)
    cv2.imwrite(str(out_dir / "preview.jpg"), cv2.cvtColor((sheet * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    (out_dir / "sample.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baked", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="redo assets that already have sample.json")
    args = parser.parse_args()

    reports, pending = [], []
    for asset_dir in sorted(p for p in args.baked.iterdir() if (p / "meta.json").is_file()):
        done = args.out / asset_dir.name / "sample.json"
        if done.is_file() and not args.force:
            reports.append(json.loads(done.read_text()))
        else:
            pending.append(asset_dir)
    print(f"{len(reports)} assets already processed, {len(pending)} to do", flush=True)

    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(process, d, args.source, args.out / d.name, args.seed): d for d in pending}
        for future in as_completed(futures):
            try:
                report = future.result()
            except Exception as error:  # keep going; a broken bake should not cost the night
                print(f"{futures[future].name:<26} FAILED: {error}", flush=True)
                continue
            clipped = max((v["clipped_fraction"] for v in report["variations"]), default=0)
            print(f"{report['asset']:<26} coverage {report['coverage']:.2f}  albedo MAE {report.get('albedo_vs_source_mae', float('nan')):.4f}  "
                  f"lit {len(report['variations'])}  max clipped {clipped:.3f}", flush=True)
            reports.append(report)
    (args.out / "dataset.json").write_text(json.dumps(sorted(reports, key=lambda r: r["asset"]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

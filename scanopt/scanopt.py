"""Scan optimiser: high-res photogrammetry mesh + photo texture in, optimised PBR asset out.

    python scanopt.py scan.obj --texture photo.jpg --out result/ --faces 10000 --remesher decimate

Remeshers: decimate (triangles, Blender QEM), quadwild (QuadWild + Bi-MDF quads,
GPL-3.0 binary), autoremesher (AutoRemesher quads, MIT binary).

Stages: Blender reduces and unwraps the mesh and bakes the photo, normals and AO
from the high-res surface; the de-lighter turns the baked photo into albedo; a
heuristic derives roughness; Blender assembles the glTF PBR material and exports
<name>.glb plus the individual texture PNGs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))
from postprocess import rasterise_mask, read_exr, to_srgb, write_png16  # noqa: E402

DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
# Shown in the GUI dropdown, in this order. DJM (SIGGRAPH 2026) joins once its code is published.
REMESHERS = {
    "decimate": "Triangles - decimation (Blender QEM)",
    "autoremesher": "Quads - AutoRemesher (MIT)",
    "quadwild": "Quads - QuadWild + Bi-MDF (GPL)",
}
MODEL_CANDIDATES = [
    Path(r"F:\DelighterData\runs\overcast\delighter.onnx"),
    Path(r"F:\DelighterData\runs\pilot\delighter.onnx"),
    REPO / "weights" / "delighter_model_fp16.onnx",
]


def blender(executable: str, *arguments: str) -> None:
    command = [executable, "-b", "--factory-startup", "--disable-autoexec"]
    if arguments[0].endswith(".blend"):
        command.append(arguments[0])
        arguments = arguments[1:]
    command += ["-P", str(HERE / "blender_stage.py"), "--", *arguments]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or "Traceback" in result.stdout + result.stderr:
        log = (result.stdout + result.stderr)[-3000:]
        raise SystemExit(f"Blender stage failed:\n{log}")


def fill_gutter(image: np.ndarray, mask: np.ndarray, pixels: int = 32) -> np.ndarray:
    """Grow island colours into empty UV space so mipmaps and filtering do not bleed black."""
    image = image.astype(np.float32).copy()
    known = (mask > 0).astype(np.float32)
    for _ in range(pixels):
        weight = cv2.boxFilter(known, -1, (3, 3), normalize=False)
        total = cv2.boxFilter(image * known[..., None], -1, (3, 3), normalize=False)
        total = total.reshape(image.shape)  # OpenCV drops a single channel axis
        grow = (known == 0) & (weight > 0)
        image[grow] = total[grow] / weight[grow][:, None]
        known[grow] = 1.0
    return image


def roughness_map(normal_ts: np.ndarray, ao: np.ndarray, mask: np.ndarray, base: float, strength: float) -> np.ndarray:
    """Heuristic: fine surface relief and cavities read as rougher, smooth open areas as glossier."""
    tilt = 1.0 - np.clip(normal_ts[..., 2] * 2 - 1, 0, 1)        # deviation from the flat low-poly surface
    detail = cv2.GaussianBlur(tilt, (0, 0), 3)
    inside = mask > 0
    scale = np.percentile(detail[inside], 95) + 1e-6 if inside.any() else 1.0
    detail = np.clip(detail / scale, 0, 1)
    cavity = cv2.GaussianBlur(1.0 - ao, (0, 0), 2)
    roughness = base + strength * (0.6 * detail + 0.4 * cavity - 0.5)
    return np.clip(roughness, 0.2, 1.0)


def write_png8(path: Path, image: np.ndarray) -> None:
    data = (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)
    if data.ndim == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="high-res scan: OBJ, PLY, FBX, GLB/glTF")
    parser.add_argument("--texture", type=Path, help="photo texture (default: taken from the scan's material)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--faces", type=int, default=10000)
    parser.add_argument("--remesher", choices=list(REMESHERS), default="decimate")
    parser.add_argument("--topology", choices=["tri", "quad"], help=argparse.SUPPRESS)  # old name: tri=decimate, quad=quadwild
    parser.add_argument("--resolution", type=int, default=2048, choices=[512, 1024, 2048, 4096])
    parser.add_argument("--roughness", type=float, default=0.8, help="base roughness")
    parser.add_argument("--roughness-variation", type=float, default=0.25)
    parser.add_argument("--model", type=Path, help="de-lighter ONNX model (default: newest trained one)")
    parser.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    parser.add_argument("--quadwild", type=Path, help="QuadWild-BiMDF folder (default: QUADWILD_DIR or F:/DelighterData/tools/quadwild-bimdf)")
    parser.add_argument("--autoremesher", type=Path, help="AutoRemesher folder (default: AUTOREMESHER_DIR or F:/DelighterData/tools/autoremesher)")
    parser.add_argument("--unwrap", choices=["xatlas", "smart"], default="xatlas",
                        help="UV unwrapper: xatlas (MIT, fewer and larger charts) or Blender Smart UV Project")
    parser.add_argument("--margin", type=int, default=16, help="texels of padding around each UV chart")
    parser.add_argument("--ao-samples", type=int, default=64)
    parser.add_argument("--cage", type=float, default=0.01, help="bake ray start offset, fraction of the bounds diagonal")
    parser.add_argument("--max-ray", type=float, default=0.03, help="bake ray search distance, fraction of the bounds diagonal")
    parser.add_argument("--smart-angle", type=float, default=66.0, help="Smart UV Project angle limit, degrees")
    parser.add_argument("--tiled", action="store_true", help="de-light in overlapping tiles (less VRAM)")
    parser.add_argument("--sharp", action="store_true", help="quad mode: keep sharp feature edges (hard-surface scans)")
    parser.add_argument("--keep-work", action="store_true", help="keep intermediate EXRs in <out>/work")
    args = parser.parse_args()
    if args.topology:
        args.remesher = "quadwild" if args.topology == "quad" else "decimate"
    quads = args.remesher != "decimate"

    model = args.model or next((p for p in MODEL_CANDIDATES if p.is_file()), None)
    if model is None:
        raise SystemExit("No de-lighter model found; pass --model")
    name = args.input.stem
    work = args.out / "work"
    work.mkdir(parents=True, exist_ok=True)
    timings, started = {}, time.time()
    sys.path.insert(0, str(HERE))

    quad_report = None
    prep = ["prep", "--input", str(args.input), "--out", str(work), "--topology", "quad" if quads else "tri",
            "--faces", str(args.faces), "--resolution", str(args.resolution), "--unwrap", args.unwrap,
            "--margin", str(args.margin), "--ao-samples", str(args.ao_samples), "--cage", str(args.cage),
            "--max-ray", str(args.max_ray), "--smart-angle", str(args.smart_angle)]
    if quads:
        print(f"[0/4] quad remeshing with {REMESHERS[args.remesher]} towards {args.faces} faces", flush=True)
        blender(args.blender, "export-high", "--input", str(args.input), "--obj", str(work / "high.obj"))
        if args.remesher == "quadwild":
            import quadwild as backend
            tool_dir = args.quadwild
        else:
            import autoremesher as backend
            tool_dir = args.autoremesher
        quad_mesh, quad_report = backend.remesh(work / "high.obj", args.faces, work / args.remesher,
                                                tool_dir=tool_dir or backend.DEFAULT_DIR, sharp=args.sharp)
        prep += ["--low-mesh", str(quad_mesh)]
        timings["remesh"] = time.time() - started
    print(f"[1/4] {'unwrapping' if quads else f'reducing to {args.faces} triangles, unwrapping'}, baking high->low", flush=True)
    if args.texture:
        prep += ["--texture", str(args.texture)]
    blender(args.blender, *prep)
    timings["bake"] = time.time() - started

    print(f"[2/4] de-lighting with {model}", flush=True)
    mask = rasterise_mask(np.load(work / "uv_tris.npy"), args.resolution)
    lit, normal_os, normal_ts = (read_exr(work / f) for f in ("lit.exr", "normal_os.exr", "normal_ts.exr"))
    ao = read_exr(work / "ao.exr")[..., 0]
    write_png16(work / "lit.png", to_srgb(lit))
    write_png16(work / "normal_os.png", normal_os)
    write_png16(work / "ao.png", ao)
    cv2.imwrite(str(work / "mask.png"), mask)

    from engine import DelighterInferenceEngine  # the app's own inference path
    engine = DelighterInferenceEngine(str(model))
    stage = time.time()
    albedo_path, _, device = engine.process_texture(str(work / "lit.png"), str(work / "normal_os.png"), str(work / "ao.png"),
                                                    str(work / "mask.png"), str(work / "albedo_raw.png"), tiled_mode=args.tiled)
    timings["delight"] = time.time() - stage

    print("[3/4] roughness, gutter fill", flush=True)
    albedo = cv2.cvtColor(cv2.imread(albedo_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    write_png8(work / "albedo.png", fill_gutter(albedo, mask))
    write_png8(work / "lit8.png", fill_gutter(to_srgb(lit), mask))  # the photo before de-lighting, for comparison
    write_png8(work / "normal.png", normal_ts)  # Blender already extended these into the gutter
    write_png8(work / "ao8.png", np.clip(ao, 0, 1))
    shutil.copy(work / "ao8.png", work / "ao.png")
    write_png8(work / "roughness.png", fill_gutter(roughness_map(normal_ts, ao, mask, args.roughness, args.roughness_variation)[..., None], mask)[..., 0])

    print("[4/4] assembling glTF PBR material", flush=True)
    glb = args.out / f"{name}.glb"
    assemble = [str(work / "low.blend"), "assemble", "--out", str(work), "--export", str(glb)]
    blender(args.blender, *assemble, *(["--keep-quads"] if quads else []))

    textures = args.out / "textures"
    textures.mkdir(exist_ok=True)
    for source, target in [("albedo.png", "albedo"), ("normal.png", "normal_gl"), ("roughness.png", "roughness"), ("ao.png", "ao"),
                           ("lit8.png", "lit")]:
        shutil.copy(work / source, textures / f"{name}_{target}.png")
    info = json.loads((work / "prep.json").read_text())
    report = {**info, "remesher": args.remesher, "model": str(model), "delight_device": device, "quad_remesh": quad_report,
              "seconds": {k: round(v, 1) for k, v in {**timings, "total": time.time() - started}.items()}, "glb": str(glb)}
    (args.out / f"{name}_report.json").write_text(json.dumps(report, indent=2))
    if not args.keep_work:
        shutil.rmtree(work)
    print(f"done: {glb}  ({info['source_faces']} -> {info['faces']} faces, {report['seconds']['total']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Scan optimiser: high-res photogrammetry mesh + photo texture in, optimised PBR asset out.

    python scanopt.py scan.obj --texture photo.jpg --out result/ --faces 10000 --remesher decimate

Remeshers: decimate (triangles, QEM), quadwild (QuadWild + Bi-MDF quads,
GPL-3.0 binary), autoremesher (AutoRemesher quads, MIT binary).

Stages: the mesh is reduced and unwrapped and the photo, normals and AO are baked
from the high-res surface; the de-lighter turns the baked photo into albedo; a
heuristic derives roughness; the glTF PBR asset is assembled and exported as
<name>.glb plus <name>.obj and the individual texture PNGs.

Everything runs in plain Python (trimesh, fast-simplification, xatlas, Embree);
the de-lighter is the bundled model chosen with --model (models/<name>.onnx).

Sources (--source): scan (photo texture with the light in it, de-lit) or pbr (a
high-res model with PBR materials, glTF or OBJ + MTL: base colour, roughness,
metallic, normal and occlusion are baked as they are, without de-lighting).
--map-* replace the model's own maps.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
from bakedelight import models  # noqa: E402

sys.path.insert(0, str(HERE))
import native_stage  # noqa: E402
from native_stage import fill_gutter  # noqa: E402

STAGES = ["remesh", "uv", "bake", "delight", "export"]
STAGE_OUTPUT = {"remesh": "low.obj", "uv": "layout.npz", "bake": "lit.png", "delight": "albedo_raw.png"}  # checked by need()
# Shown in the GUI dropdown, in this order. DJM (SIGGRAPH 2026) joins once its code is published.
REMESHERS = {
    "decimate": "Triangles - decimation (QEM)",
    "autoremesher": "Quads - AutoRemesher (MIT)",
    "quadwild": "Quads - QuadWild + Bi-MDF (GPL)",
}


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
    parser.add_argument("input", type=Path, help="high-res scan or model: OBJ, PLY, GLB/glTF, STL")
    parser.add_argument("--texture", type=Path, help="photo texture, or base colour for --source pbr (default: taken from the material)")
    parser.add_argument("--source", choices=["scan", "pbr"], default="scan",
                        help="scan: photo scan, de-lit; pbr: model with PBR materials, baked as they are")
    for key, text in (("roughness", "roughness"), ("metallic", "metallic"), ("normal", "tangent-space normal (OpenGL, +Y)"),
                      ("ao", "ambient occlusion")):
        parser.add_argument(f"--map-{key}", type=Path, help=f"--source pbr: {text} map replacing the material's")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--faces", type=int, default=10000)
    parser.add_argument("--remesher", choices=list(REMESHERS), default="decimate")
    parser.add_argument("--topology", choices=["tri", "quad"], help=argparse.SUPPRESS)  # old name: tri=decimate, quad=quadwild
    parser.add_argument("--resolution", type=int, default=2048, choices=[512, 1024, 2048, 4096])
    parser.add_argument("--roughness", type=float, default=0.8, help="base roughness")
    parser.add_argument("--roughness-variation", type=float, default=0.25)
    parser.add_argument("--model", default=models.DEFAULT, help=f"de-lighter model: {', '.join(models.names())}")
    parser.add_argument("--quadwild", type=Path, help="QuadWild-BiMDF folder (default: QUADWILD_DIR or F:/DelighterData/tools/quadwild-bimdf)")
    parser.add_argument("--autoremesher", type=Path, help="AutoRemesher folder (default: AUTOREMESHER_DIR, else the copy shipped with the app)")
    parser.add_argument("--margin", type=int, default=16, help="texels of padding around each UV chart")
    parser.add_argument("--ao-samples", type=int, default=64)
    parser.add_argument("--cage", type=float, default=0.01, help="bake ray start offset, fraction of the bounds diagonal")
    parser.add_argument("--max-ray", type=float, default=0.03, help="bake ray search distance, fraction of the bounds diagonal")
    parser.add_argument("--tiled", action="store_true", help="de-light in overlapping tiles (less VRAM)")
    parser.add_argument("--tile-size", type=int, default=0, help="tile size in px with --tiled; 0 = half the image")
    parser.add_argument("--predecimate", type=float, default=2.0,
                        help="quad mode: decimate the scan to this many times --faces before quad remeshing (0 = off)")
    parser.add_argument("--sharp", action="store_true", help="quad mode: keep sharp feature edges (hard-surface scans)")
    parser.add_argument("--keep-work", action="store_true", help="keep intermediate EXRs in <out>/work")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all",
                        help="run one stage on the results of the earlier ones in <out>/work (kept), or all of them")
    args = parser.parse_args()
    if args.topology:
        args.remesher = "quadwild" if args.topology == "quad" else "decimate"
    quads = args.remesher != "decimate"
    args.input, args.out = args.input.resolve(), args.out.resolve()
    args.texture = args.texture.resolve() if args.texture else None
    if args.input.suffix.lower() == ".fbx":
        raise SystemExit("FBX is not supported; export the mesh as OBJ, PLY or glTF")
    if missing := native_stage.available():
        raise SystemExit(f"Missing packages: {missing}")

    name = args.input.stem
    work = args.out / "work"
    work.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stage == "all" else [args.stage]
    state_path = work / "stages.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() and args.stage != "all" else {}
    started = time.time()

    def need(path: Path, stage: str):
        """path is what `stage` wrote; it must exist and be newer than what the stages before it wrote."""
        if not path.is_file():
            raise SystemExit(f"{path.name} is missing: run the {stage} stage first")
        for earlier in STAGES[:STAGES.index(stage)]:
            source = work / STAGE_OUTPUT[earlier]
            if source.is_file() and source.stat().st_mtime > path.stat().st_mtime:
                raise SystemExit(f"{path.name} is older than {source.name} from the {earlier} stage: "
                                 f"run the {stage} stage again")

    def done(stage: str, since: float, **values):
        state[stage] = {**values, "seconds": round(time.time() - since, 1)}
        state_path.write_text(json.dumps(state, indent=2, default=str))

    for stage in stages:
        since = time.time()
        tag = f"[{STAGES.index(stage) + 1}/{len(STAGES)}]"
        if stage == "remesh":
            low = work / "low.obj"
            if quads:
                print(f"{tag} quad remeshing with {REMESHERS[args.remesher]} towards {args.faces} faces", flush=True)
                target = round(args.predecimate * args.faces) if args.predecimate > 0 else 0
                remesh_faces = native_stage.export_high(args.input, work / "high.obj", target)
                if target:
                    print(f"      pre-decimated to {remesh_faces} triangles for the remesher", flush=True)
                if args.remesher == "quadwild":
                    import quadwild as backend
                    tool_dir = args.quadwild
                else:
                    import autoremesher as backend
                    tool_dir = args.autoremesher
                native_stage.progress(0.35)  # the remesher itself reports nothing until it is done
                quad_mesh, quad_report = backend.remesh(work / "high.obj", args.faces, work / args.remesher,
                                                        tool_dir=tool_dir or backend.DEFAULT_DIR, sharp=args.sharp)
                shutil.copy(quad_mesh, low)
            else:
                print(f"{tag} reducing to {args.faces} triangles", flush=True)
                native_stage.reduce_scan(args.input, args.faces, low)
                quad_report = None
            native_stage.progress(0.9)
            native_stage.write_preview_obj(low, args.input, work / "low_preview.obj")
            native_stage.progress(1.0)
            print(f"      low mesh: {low}", flush=True)
            done("remesh", since, remesher=args.remesher, topology="quad" if quads else "tri", quad_remesh=quad_report)

        elif stage == "uv":
            need(work / "low.obj", "remesh")
            print(f"{tag} unwrapping (xatlas)", flush=True)
            info = native_stage.layout(work / "low.obj", args.input, work, args.resolution, args.margin,
                                       log=lambda text: print(text, flush=True))
            native_stage.progress(1.0)
            print(f"      uv mesh: {work / 'low_uv.obj'}", flush=True)
            done("uv", since, unwrap="xatlas", resolution=args.resolution, charts=info.get("charts"),
                 **{key: info.get(key) for key in ("margin", "stretched_triangles", "worst_stretch", "stray_origin_corners")})

        elif stage == "bake":
            print(f"{tag} baking high->low (Embree)", flush=True)
            need(work / "layout.npz", "uv")
            maps = {"roughness": args.map_roughness, "metallic": args.map_metallic, "normal": args.map_normal,
                    "occlusion": args.map_ao}
            native_stage.bake_layout(args.input, args.texture, work, args.cage, args.max_ray, args.ao_samples,
                                     topology="quad" if quads else "tri", log=lambda text: print(text, flush=True),
                                     source=args.source, maps={k: v.resolve() for k, v in maps.items() if v})
            write_delight_inputs(work)
            native_stage.progress(1.0)
            print(f"      maps: {work / 'lit.png'}", flush=True)
            done("bake", since)

        elif stage == "delight":
            need(work / "lit.png", "bake")
            if baked_source(work) == "pbr":
                print(f"{tag} PBR model: its base colour is the albedo, no de-lighting", flush=True)
                write_png8(work / "albedo_raw.png", to_srgb(read_exr(work / "lit.exr")))
                done("delight", since, model=None, device=None)
                continue
            try:
                model = models.resolve(args.model)
            except FileNotFoundError as error:
                raise SystemExit(str(error)) from None
            print(f"{tag} de-lighting with {model}", flush=True)
            from engine import DelighterInferenceEngine  # the app's own inference path
            engine = DelighterInferenceEngine(str(model))
            _, _, device = engine.process_texture(str(work / "lit.png"), str(work / "normal_os.png"), str(work / "ao.png"),
                                                  str(work / "mask.png"), str(work / "albedo_raw.png"), tiled_mode=args.tiled, tile_size=args.tile_size,
                                                  progress_callback=native_stage.progress)
            print(f"      albedo: {work / 'albedo_raw.png'}", flush=True)
            done("delight", since, model=str(model), device=device)

        elif stage == "export":
            need(work / "albedo_raw.png", "delight")
            print(f"{tag} roughness, gutter fill, assembling glTF PBR material", flush=True)
            glb = export(args, work, name)
            native_stage.progress(1.0)
            done("export", since, glb=str(glb))
            info = json.loads((work / "prep.json").read_text())
            report = {**info, "remesher": args.remesher,
                      "stages": state, "glb": str(glb)}
            (args.out / f"{name}_report.json").write_text(json.dumps(report, indent=2, default=str))
            if args.stage == "all" and not args.keep_work:
                shutil.rmtree(work)
            print(f"done: {glb}  ({info['source_faces']} -> {info['faces']} faces, {time.time() - started:.1f} s)", flush=True)
    if args.stage != "all":
        print(f"stage {args.stage} done ({time.time() - started:.1f} s)", flush=True)
    return 0


def baked_source(work: Path) -> str:
    """What the bake stage baked from: "scan" or "pbr" (its maps are then used as they are)."""
    try:
        return json.loads((work / "prep.json").read_text()).get("source", "scan")
    except (OSError, ValueError):
        return "scan"


def write_delight_inputs(work: Path) -> None:
    """The baked EXRs as the de-lighter's inputs (16-bit PNGs) plus the UV coverage mask at the bake's size."""
    lit = read_exr(work / "lit.exr")
    mask = rasterise_mask(np.load(work / "uv_tris.npy"), lit.shape[0])
    write_png16(work / "lit.png", to_srgb(lit))
    write_png16(work / "normal_os.png", read_exr(work / "normal_os.exr"))
    write_png16(work / "ao.png", read_exr(work / "ao.exr")[..., 0])
    cv2.imwrite(str(work / "mask.png"), mask)


def export(args, work: Path, name: str) -> Path:
    mask = cv2.imread(str(work / "mask.png"), cv2.IMREAD_GRAYSCALE)
    lit, normal_ts = read_exr(work / "lit.exr"), read_exr(work / "normal_ts.exr")
    ao = read_exr(work / "ao.exr")[..., 0]
    albedo = cv2.cvtColor(cv2.imread(str(work / "albedo_raw.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    write_png8(work / "albedo.png", fill_gutter(albedo, mask))
    native_stage.progress(0.25)
    write_png8(work / "lit8.png", fill_gutter(to_srgb(lit), mask))  # the photo before de-lighting, for comparison
    write_png8(work / "normal.png", normal_ts)  # already extended into the gutter by the bake
    write_png8(work / "ao8.png", np.clip(ao, 0, 1))
    pbr = baked_source(work) == "pbr"
    if pbr:  # the model's own roughness and metallic, already extended into the gutter by the bake
        write_png8(work / "roughness.png", read_exr(work / "roughness.exr")[..., 0])
        write_png8(work / "metallic.png", read_exr(work / "metallic.exr")[..., 0])
    else:
        write_png8(work / "roughness.png", fill_gutter(roughness_map(normal_ts, ao, mask, args.roughness, args.roughness_variation)[..., None], mask)[..., 0])
    native_stage.progress(0.6)
    glb = args.out / f"{name}.glb"
    native_stage.assemble(work, glb, name, metallic=pbr)  # the OBJ keeps quads
    native_stage.progress(0.9)
    textures = args.out / "textures"
    textures.mkdir(exist_ok=True)
    for source, target in [("albedo.png", "albedo"), ("normal.png", "normal_gl"), ("roughness.png", "roughness"),
                           ("ao8.png", "ao"), ("lit8.png", "lit")] + ([("metallic.png", "metallic")] if pbr else []):
        shutil.copy(work / source, textures / f"{name}_{target}.png")
    return glb


if __name__ == "__main__":
    sys.exit(main())

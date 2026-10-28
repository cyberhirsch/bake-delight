"""Blender side of scanopt: reduce a scan, unwrap it, bake high->low, and assemble the PBR asset.

Run headless with autoexec disabled (scans and their files are third-party data):

    blender -b --factory-startup --disable-autoexec -P blender_stage.py -- prep --input scan.obj --texture photo.png --out work/
    blender -b --factory-startup --disable-autoexec -P blender_stage.py -- assemble --out work/ --export asset.glb

prep writes into --out:
    low.blend        the optimised mesh with new UVs (no material)
    lit.exr          photo texture transferred to the new UVs (linear)
    normal_ts.exr    tangent-space normal, OpenGL (+Y), as glTF expects
    normal_os.exr    object-space normal, Blender axes, encoded n*0.5+0.5 (de-lighter input)
    ao.exr           ambient occlusion from the high-res surface
    uv_tris.npy      UV triangles of the low mesh, for the coverage mask
    prep.json        counts and settings
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

IMPORTERS = {
    ".obj": lambda p: bpy.ops.wm.obj_import(filepath=p),
    ".ply": lambda p: bpy.ops.wm.ply_import(filepath=p),
    ".stl": lambda p: bpy.ops.wm.stl_import(filepath=p),
    ".fbx": lambda p: bpy.ops.import_scene.fbx(filepath=p),
    ".glb": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".gltf": lambda p: bpy.ops.import_scene.gltf(filepath=p),
}

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prep")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--texture", type=Path, help="photo texture; default: the first image in the scan's material")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--topology", choices=["tri", "quad"], default="tri")
    prep.add_argument("--low-mesh", type=Path, help="use this already-optimised mesh (e.g. QuadWild output) instead of decimating")
    prep.add_argument("--faces", type=int, default=10000, help="target face count of the optimised mesh")
    prep.add_argument("--resolution", type=int, default=2048)
    prep.add_argument("--margin", type=int, default=16)
    prep.add_argument("--cage", type=float, default=0.01, help="ray start offset outside the low mesh, fraction of the bounds diagonal")
    prep.add_argument("--max-ray", type=float, default=0.03, help="ray search distance, fraction of the bounds diagonal")
    prep.add_argument("--smart-angle", type=float, default=66.0, help="Smart UV Project angle limit, degrees")
    prep.add_argument("--unwrap", choices=["xatlas", "smart"], default="xatlas")
    prep.add_argument("--ao-samples", type=int, default=64)
    export = sub.add_parser("export-high", help="write the joined scan as a triangle OBJ for external remeshers")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--obj", type=Path, required=True)
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--out", type=Path, required=True)
    assemble.add_argument("--export", type=Path, required=True)
    assemble.add_argument("--keep-quads", action="store_true", help="also write an FBX, which preserves quads")
    return parser.parse_args(argv)


# ---------------------------------------------------------------- helpers

def enable_gpu(scene):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "METAL"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == backend for d in prefs.devices):
            for device in prefs.devices:
                device.use = device.type == backend
            scene.cycles.device = "GPU"
            return backend
    scene.cycles.device = "CPU"
    return "CPU"


def select_only(objects, active):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active


def import_scan(path: Path):
    importer = IMPORTERS.get(path.suffix.lower())
    if importer is None:
        raise SystemExit(f"Unsupported mesh format: {path.suffix}")
    before = set(bpy.data.objects)
    importer(str(path))
    meshes = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
    if not meshes:
        raise SystemExit("The file contains no meshes")
    select_only(meshes, meshes[0])
    if len(meshes) > 1:
        bpy.ops.object.join()
    high = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    high.name = "scan_high"
    if not high.data.uv_layers:
        raise SystemExit("The scan has no UVs, so its photo texture cannot be transferred")
    return high


def find_texture(high, texture: Path | None):
    if texture:
        return bpy.data.images.load(str(texture))
    for slot in high.material_slots:
        if slot.material and slot.material.use_nodes:
            for node in slot.material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image and node.image.has_data:
                    return node.image
    raise SystemExit("No photo texture given and none found in the scan's material")


def emission_material(image):
    """Shows the photo texture unlit, so an EMIT bake transfers it exactly."""
    material = bpy.data.materials.new("scan_photo")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    image.colorspace_settings.name = "sRGB"
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(texture.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def bounds_diagonal(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    low = Vector(tuple(min(c[i] for c in corners) for i in range(3)))
    high = Vector(tuple(max(c[i] for c in corners) for i in range(3)))
    return (high - low).length


def load_low_mesh(path: Path):
    """Import an externally optimised mesh (same axis convention as export-high)."""
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(path))
    meshes = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
    select_only(meshes, meshes[0])
    if len(meshes) > 1:
        bpy.ops.object.join()
    low = bpy.context.view_layer.objects.active
    low.name = "scan_low"
    low.data.materials.clear()
    for layer in list(low.data.uv_layers):
        low.data.uv_layers.remove(layer)
    bpy.ops.object.shade_smooth()
    return low


def reduce_mesh(high, faces: int):
    """Triangle decimation (quadric edge collapse)."""
    select_only([high], high)
    bpy.ops.object.duplicate()
    low = bpy.context.view_layer.objects.active
    low.name = "scan_low"
    low.data.materials.clear()
    source_faces = len(low.data.polygons)
    if source_faces > faces:
        modifier = low.modifiers.new("decimate", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = faces / source_faces
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    for layer in list(low.data.uv_layers):
        low.data.uv_layers.remove(layer)
    bpy.ops.object.shade_smooth()
    return low, source_faces


def unwrap_xatlas(low, resolution: int, margin: int, work: Path) -> dict:
    """Chart and pack with xatlas, run by xatlas_unwrap.py under Blender's own Python but in a
    separate process. xatlas takes triangles only, so polygons go in as their loop triangles
    and each loop takes the UV of a triangle that uses it."""
    mesh = low.data
    mesh.calc_loop_triangles()
    positions = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", positions)
    tri_loops = np.empty(len(mesh.loop_triangles) * 3, np.int64)
    mesh.loop_triangles.foreach_get("loops", tri_loops)
    loop_verts = np.empty(len(mesh.loops), np.int64)
    mesh.loops.foreach_get("vertex_index", loop_verts)

    np.savez(work / "xatlas_in.npz", positions=positions.reshape(-1, 3), triangles=loop_verts[tri_loops].reshape(-1, 3))
    # Padding in texels between charts; the bake's EXTEND margin fills half of it from each side.
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("xatlas_unwrap.py")),
                             str(work / "xatlas_in.npz"), str(work / "xatlas_out.npz"),
                             "--resolution", str(resolution), "--padding", str(margin)],
                            capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"xatlas failed ({result.returncode:#x}):\n{(result.stdout + result.stderr)[-2000:]}")
    packed = np.load(work / "xatlas_out.npz")
    corner_uv = packed["corner_uv"].reshape(-1, 3, 2)  # per triangle corner, same order as tri_loops
    tri_loops = tri_loops.reshape(-1, 3)
    tri_poly = np.empty(len(mesh.loop_triangles), np.int64)
    mesh.loop_triangles.foreach_get("polygon_index", tri_poly)
    corner_pos = positions.reshape(-1, 3)[loop_verts[tri_loops]]
    loop_uv, split = join_split_polygons(tri_loops, corner_uv, corner_pos, tri_poly, len(mesh.loops))
    if not np.isfinite(loop_uv).all():
        raise RuntimeError("xatlas unwrap produced non-finite UVs")
    # xatlas mirrors about half of its charts; flip them back so textures read the right way round.
    tri_uv = loop_uv[tri_loops]
    winding = (tri_uv[:, 1, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 2, 1] - tri_uv[:, 0, 1]) \
        - (tri_uv[:, 1, 1] - tri_uv[:, 0, 1]) * (tri_uv[:, 2, 0] - tri_uv[:, 0, 0])
    loop_uv[np.unique(tri_loops[winding < 0]), 0] *= -1
    layer = mesh.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", loop_uv.reshape(-1))
    # xatlas charts well but packs loosely (about half of its bounding box), and rejoined
    # polygons stick out of their chart, so Blender's packer places the charts.
    pack(low, resolution, margin)
    return {"unwrap": "xatlas", "charts": int(packed["charts"]), "split_polygons": split}


def pack(low, resolution: int, margin: int):
    select_only([low], low)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, margin_method="FRACTION", margin=margin / resolution * 1.5, shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")


def signed_area(uv) -> float:
    (x0, y0), (x1, y1), (x2, y2) = uv
    return 0.5 * float((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))


def join_split_polygons(tri_loops, corner_uv, corner_pos, tri_poly, loop_count):
    """Per-loop UVs from per-triangle-corner UVs. Where xatlas put the triangles of one quad
    (or n-gon) into different charts, the stray triangles are carried into the chart of the
    largest one by the similarity that maps their shared edge, so every polygon stays whole."""
    loop_uv = np.zeros((loop_count, 2), np.float32)
    loop_uv[tri_loops] = corner_uv
    mismatch = np.abs(loop_uv[tri_loops] - corner_uv).max(axis=(1, 2)) > 1e-6
    split_polys = np.unique(tri_poly[mismatch])
    for poly in split_polys:
        tris = np.flatnonzero(tri_poly == poly)
        areas = [abs(signed_area(corner_uv[t])) for t in tris]
        reference = tris[int(np.argmax(areas))]
        placed = {int(l): corner_uv[reference, k] for k, l in enumerate(tri_loops[reference])}
        winding = np.sign(signed_area(corner_uv[reference]))
        pending = [t for t in tris if t != reference]
        while pending:
            for t in pending:
                shared = [k for k in range(3) if int(tri_loops[t, k]) in placed]
                if len(shared) >= 2:
                    break
            else:
                break  # disconnected fan; leave the rest as xatlas placed it
            pending.remove(t)
            # Rebuild the third corner from the 3D triangle on the already placed edge
            # (the stray chart's own UVs can be degenerate).
            k0, k1 = shared[:2]
            k2 = 3 - k0 - k1
            p0, p1, p2 = corner_pos[t, k0], corner_pos[t, k1], corner_pos[t, k2]
            edge = p1 - p0
            length2 = float(edge @ edge) or 1e-12
            along = float((p2 - p0) @ edge) / length2
            across = float(np.linalg.norm(np.cross(edge, p2 - p0))) / length2
            a, b = placed[int(tri_loops[t, k0])], placed[int(tri_loops[t, k1])]
            d = b - a
            for side in (1.0, -1.0):
                uv = np.empty((3, 2), np.float32)
                uv[k0], uv[k1] = a, b
                uv[k2] = a + along * d + side * across * np.array([-d[1], d[0]])
                if np.sign(signed_area(uv)) == winding:
                    break
            placed.setdefault(int(tri_loops[t, k2]), uv[k2])
        for loop, uv in placed.items():
            loop_uv[loop] = uv
    return loop_uv, int(split_polys.size)


def unwrap(low, resolution: int, margin: int, work: Path, method: str = "xatlas", smart_angle: float = 66.0) -> dict:
    if method == "xatlas":
        return unwrap_xatlas(low, resolution, margin, work)
    select_only([low], low)
    low.data.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(smart_angle), island_margin=0.0, area_weight=0.0, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    pack(low, resolution, margin)
    return {"unwrap": "smart"}


def save_exr(scene, image, path: Path):
    settings = scene.render.image_settings
    settings.file_format, settings.color_depth, settings.exr_codec, settings.color_mode = "OPEN_EXR", "16", "ZIP", "RGB"
    image.save_render(str(path), scene=scene)


# ---------------------------------------------------------------- modes

def prep(args):
    args.out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    device = enable_gpu(scene)
    scene.view_settings.view_transform, scene.view_settings.look = "Standard", "None"

    high = import_scan(args.input)
    photo = find_texture(high, args.texture)
    high.data.materials.clear()
    high.data.materials.append(emission_material(photo))
    if args.low_mesh:
        source_faces = len(high.data.polygons)
        low = load_low_mesh(args.low_mesh)
    else:
        low, source_faces = reduce_mesh(high, args.faces)
    unwrap_info = unwrap(low, args.resolution, args.margin, args.out, args.unwrap, args.smart_angle)
    print(f"UNWRAP {unwrap_info}", flush=True)

    target = bpy.data.images.new("bake_target", args.resolution, args.resolution, alpha=True, float_buffer=True)
    material = bpy.data.materials.new("scan_low")
    material.use_nodes = True
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = target
    material.node_tree.nodes.active = node
    low.data.materials.append(material)

    diagonal = bounds_diagonal(high)
    bake = scene.render.bake
    bake.use_selected_to_active = True
    bake.use_cage = False
    bake.cage_extrusion = diagonal * args.cage  # rays start just outside the low surface
    bake.max_ray_distance = diagonal * args.max_ray
    bake.margin, bake.margin_type = args.margin, "EXTEND"
    scene.world = scene.world or bpy.data.worlds.new("world")
    scene.world.light_settings.distance = diagonal * 0.1
    select_only([high, low], low)

    passes = [
        ("lit.exr", dict(type="EMIT"), 1),
        ("normal_ts.exr", dict(type="NORMAL", normal_space="TANGENT"), 1),
        ("normal_os.exr", dict(type="NORMAL", normal_space="OBJECT"), 1),
        ("ao.exr", dict(type="AO"), args.ao_samples),
    ]
    for filename, options, samples in passes:
        target.generated_color = (0, 0, 0, 1)
        target.source = "GENERATED"
        scene.cycles.samples = samples
        bpy.ops.object.bake(use_selected_to_active=True, cage_extrusion=bake.cage_extrusion,
                            max_ray_distance=bake.max_ray_distance, margin=args.margin, use_clear=True, **options)
        save_exr(scene, target, args.out / filename)
        print(f"BAKED {filename}", flush=True)

    mesh = low.data
    mesh.calc_loop_triangles()
    uv = mesh.uv_layers.active.data
    uv_tris = np.asarray([[tuple(uv[i].uv) for i in t.loops] for t in mesh.loop_triangles], np.float32)
    np.save(args.out / "uv_tris.npy", uv_tris)
    unwrap_info["uv_coverage"] = round(sum(abs(signed_area(t)) for t in uv_tris), 4)  # share of the texture in use

    bpy.data.objects.remove(high, do_unlink=True)
    low.data.materials.clear()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / "low.blend"))
    (args.out / "prep.json").write_text(json.dumps({
        "input": str(args.input), "topology": args.topology, "source_faces": source_faces,
        "faces": len(mesh.polygons), "quads": sum(len(p.vertices) == 4 for p in mesh.polygons),
        "triangles": len(mesh.loop_triangles), "resolution": args.resolution,
        "device": device, **unwrap_info,
    }, indent=2))
    print("PREP DONE", flush=True)


def assemble(args):
    low = bpy.data.objects["scan_low"]
    material = bpy.data.materials.new("scan_pbr")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.0

    def image_node(filename, colorspace):
        node = nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(args.out / filename))
        node.image.colorspace_settings.name = colorspace
        return node

    links.new(image_node("albedo.png", "sRGB").outputs["Color"], bsdf.inputs["Base Color"])
    links.new(image_node("roughness.png", "Non-Color").outputs["Color"], bsdf.inputs["Roughness"])
    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(image_node("normal.png", "Non-Color").outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    # glTF occlusion: the exporter reads a "glTF Material Output" group with an Occlusion input.
    group = bpy.data.node_groups.get("glTF Material Output") or bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
    if "Occlusion" not in [item.name for item in group.interface.items_tree]:
        group.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
    gltf = nodes.new("ShaderNodeGroup")
    gltf.node_tree = group
    links.new(image_node("ao.png", "Non-Color").outputs["Color"], gltf.inputs["Occlusion"])

    low.data.materials.append(material)
    select_only([low], low)
    bpy.ops.export_scene.gltf(filepath=str(args.export), export_format="GLB", use_selection=True,
                              export_image_format="AUTO", export_materials="EXPORT", export_apply=True)
    print("EXPORTED", args.export, flush=True)
    # OBJ always: the de-lighter app's 3D preview reads OBJ. It keeps quads as well.
    bpy.ops.wm.obj_export(filepath=str(args.export.with_suffix(".obj")), export_selected_objects=True,
                          export_triangulated_mesh=False, export_materials=True, path_mode="COPY")
    if args.keep_quads:
        # glTF only stores triangles; FBX keeps the quad topology for DCC tools.
        bpy.ops.export_scene.fbx(filepath=str(args.export.with_suffix(".fbx")), use_selection=True,
                                 path_mode="COPY", embed_textures=True, mesh_smooth_type="FACE")
        print("EXPORTED quads as FBX", flush=True)


def export_high(args):
    high = import_scan(args.input)
    select_only([high], high)
    args.obj.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.obj_export(filepath=str(args.obj), export_selected_objects=True, export_triangulated_mesh=True,
                          export_uv=False, export_normals=False, export_materials=False)
    print("EXPORTED", len(high.data.polygons), flush=True)


args = parse_args()
{"prep": prep, "assemble": assemble, "export-high": export_high}[args.mode](args)

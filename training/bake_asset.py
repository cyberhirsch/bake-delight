"""Blender script: bake one Poly Haven asset into UV-space training maps.

Run headless (autoexec disabled, the .blend is third-party content):

    blender -b --disable-autoexec <asset_2k.blend> -P bake_asset.py -- \
        --out F:/DelighterData/baked/<asset> --hdris F:/DelighterData/source/hdris \
        --variations 10 --seed 0

Writes linear half-float EXRs plus the UV triangles for mask rasterisation:
    albedo.exr      diffuse colour only (ground truth)
    normal_os.exr   object-space normal, Blender axes (Z up), encoded n*0.5+0.5
    ao.exr          ambient occlusion
    lit_XX.exr      albedo x irradiance (direct + indirect) under random lighting
    uv_tris.npy     float32 [N, 3, 2] UV triangles of the baked meshes
    meta.json       lighting parameters per variation
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

RESOLUTION = 2048
MARGIN = 16
# Poly Haven ships "Smooth by Angle" style node groups; they do not alter UVs or topology.
SHADING_ONLY_NODES = {
    "GROUP_INPUT", "GROUP_OUTPUT", "SET_SHADE_SMOOTH", "MESH_EDGE_ANGLE", "INPUT_EDGE_SMOOTH",
    "INPUT_SHADE_SMOOTH", "COMPARE", "BOOLEAN_MATH", "MATH", "REROUTE",
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hdris", type=Path, required=True)
    parser.add_argument("--variations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sun-probability", type=float, default=0.5, help="0 for overcast-only, 1 for always sunny")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true", help="validate the asset, bake nothing")
    return parser.parse_args(argv)


def enable_gpu(scene):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for device in prefs.devices:
                device.use = device.type == backend
            scene.cycles.device = "GPU"
            return backend
    scene.cycles.device = "CPU"
    return "CPU"


def textured_material(material):
    """Return the Principled BSDF of a plain base-colour-textured material, else None."""
    if not material or not material.use_nodes:
        return None
    nodes = material.node_tree.nodes
    principled = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled is None or not principled.inputs["Base Color"].is_linked:
        return None
    if principled.inputs["Alpha"].is_linked:
        return None  # alpha cutouts (foliage) do not bake into a usable atlas
    return principled


def prepare_meshes():
    """Keep LOD0 meshes with a textured material; delete everything else that would occlude them."""
    lod = re.compile(r"_LOD(\d+)$", re.IGNORECASE)
    keep, drop = [], []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            if obj.type in {"LIGHT", "CAMERA"}:
                drop.append(obj)
            continue
        match = lod.search(obj.name)
        if (match and match.group(1) != "0") or obj.hide_render:
            drop.append(obj)
            continue
        materials = [slot.material for slot in obj.material_slots]
        if not materials or not all(textured_material(m) for m in materials):
            raise SystemExit(f"SKIP unsupported material on {obj.name}")
        keep.append(obj)
    for obj in drop:
        bpy.data.objects.remove(obj, do_unlink=True)
    if not keep:
        raise SystemExit("SKIP no bakeable meshes")
    for obj in keep:
        if len(obj.data.uv_layers) == 0:
            raise SystemExit(f"SKIP {obj.name} has no UVs")
        for modifier in obj.modifiers:
            group = getattr(modifier, "node_group", None)
            if modifier.type == "NODES" and group and not {n.type for n in group.nodes} <= SHADING_ONLY_NODES:
                raise SystemExit(f"SKIP geometry nodes change topology on {obj.name}")
    return keep


def neutralise_materials(objects):
    """Remove metalness so the diffuse pass carries the whole base colour."""
    for material in {slot.material for obj in objects for slot in obj.material_slots}:
        principled = textured_material(material)
        metallic = principled.inputs["Metallic"]
        for link in list(metallic.links):
            material.node_tree.links.remove(link)
        metallic.default_value = 0.0


def bake_target(objects):
    image = bpy.data.images.new("bake_target", RESOLUTION, RESOLUTION, alpha=True, float_buffer=True)
    image.colorspace_settings.name = "Linear Rec.709" if "Linear Rec.709" in [c.name for c in bpy.types.ColorManagedInputColorspaceSettings.bl_rna.properties["name"].enum_items] else "Linear"
    for material in {slot.material for obj in objects for slot in obj.material_slots}:
        nodes = material.node_tree.nodes
        node = nodes.get("bake_target_node") or nodes.new("ShaderNodeTexImage")
        node.name = "bake_target_node"
        node.image = image
        nodes.active = node
    return image


def save_exr(scene, image, path):
    settings = scene.render.image_settings
    settings.file_format = "OPEN_EXR"
    settings.color_depth = "16"
    settings.exr_codec = "ZIP"
    settings.color_mode = "RGB"
    image.save_render(str(path), scene=scene)


def bake(scene, image, path, bake_type, pass_filter=None, samples=1, **kwargs):
    image.generated_color = (0, 0, 0, 1)
    image.source = "GENERATED"
    scene.cycles.samples = samples
    scene.render.bake.margin = MARGIN
    scene.render.bake.margin_type = "EXTEND"
    options = dict(type=bake_type, margin=MARGIN, use_clear=True, **kwargs)
    if pass_filter:
        options["pass_filter"] = pass_filter
    bpy.ops.object.bake(**options)
    save_exr(scene, image, path)


def bounds(objects):
    corners = [obj.matrix_world @ Vector(c) for obj in objects for c in obj.bound_box]
    low = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    high = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return low, high


def setup_world(scene):
    world = bpy.data.worlds.new("delight_world")
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    environment = nodes.new("ShaderNodeTexEnvironment")
    background = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    return world, mapping, environment, background


def add_ground(low, high):
    size = max(high.x - low.x, high.y - low.y, high.z - low.z) * 20.0
    bpy.ops.mesh.primitive_plane_add(size=size, location=((low.x + high.x) / 2, (low.y + high.y) / 2, low.z))
    plane = bpy.context.active_object
    plane.name = "delight_ground"
    material = bpy.data.materials.new("delight_ground")
    material.use_nodes = True
    plane.data.materials.append(material)
    return plane, material.node_tree.nodes["Principled BSDF"]


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    device = enable_gpu(scene)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    objects = prepare_meshes()
    if args.check_only:
        print(f"OK {len(objects)} meshes", flush=True)
        return
    neutralise_materials(objects)
    low, high = bounds(objects)
    diagonal = (high - low).length
    world, mapping, environment, background = setup_world(scene)
    world.light_settings.distance = diagonal * 0.1  # AO reach relative to asset size
    image = bake_target(objects)

    ground, ground_bsdf = add_ground(low, high)
    sun_data = bpy.data.lights.new("delight_sun", "SUN")
    sun = bpy.data.objects.new("delight_sun", sun_data)
    scene.collection.objects.link(sun)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]

    # Geometry-only maps: no ground or sun so AO reflects the asset alone.
    ground.hide_render = sun.hide_render = True
    bake(scene, image, args.out / "albedo.exr", "DIFFUSE", {"COLOR"})
    bake(scene, image, args.out / "normal_os.exr", "NORMAL", normal_space="OBJECT")
    bake(scene, image, args.out / "ao.exr", "AO", samples=64)

    triangles = []
    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        uv = mesh.uv_layers.active.data
        for triangle in mesh.loop_triangles:
            triangles.append([tuple(uv[i].uv) for i in triangle.loops])
    np.save(args.out / "uv_tris.npy", np.asarray(triangles, dtype=np.float32))

    hdris = sorted(args.hdris.glob("*.hdr"))
    order = hdris * (args.variations // len(hdris) + 1)
    rng.shuffle(order)
    variations = []
    for index in range(args.variations):
        hdri = order[index]
        environment.image = bpy.data.images.load(str(hdri), check_existing=True)
        rotation = rng.uniform(0, 360)
        mapping.inputs["Rotation"].default_value = (0, 0, math.radians(rotation))
        background.inputs["Strength"].default_value = rng.uniform(0.5, 2.0)

        use_ground = rng.random() < 0.7
        ground.hide_render = not use_ground
        grey = rng.uniform(0.08, 0.45)
        ground_bsdf.inputs["Base Color"].default_value = (grey, grey * rng.uniform(0.9, 1.05), grey * rng.uniform(0.8, 1.0), 1)

        use_sun = rng.random() < args.sun_probability
        sun.hide_render = not use_sun
        elevation, azimuth = rng.uniform(12, 75), rng.uniform(0, 360)
        sun.rotation_euler = (math.radians(90 - elevation), 0, math.radians(azimuth))
        sun_data.energy = rng.uniform(2.0, 8.0)
        sun_data.angle = math.radians(rng.uniform(0.5, 5.0))
        warmth = rng.uniform(0.0, 0.25)
        sun_data.color = (1.0, 1.0 - warmth * 0.4, 1.0 - warmth)

        bake(scene, image, args.out / f"lit_{index:02d}.exr", "DIFFUSE", {"DIRECT", "INDIRECT", "COLOR"}, samples=args.samples)
        variations.append({
            "file": f"lit_{index:02d}.exr", "hdri": hdri.stem, "hdri_rotation_deg": rotation,
            "hdri_strength": background.inputs["Strength"].default_value,
            "ground": use_ground, "ground_grey": grey,
            "sun": use_sun, "sun_elevation_deg": elevation, "sun_azimuth_deg": azimuth,
            "sun_energy": sun_data.energy, "sun_angle_deg": math.degrees(sun_data.angle),
        })
        print(f"BAKED {index + 1}/{args.variations} {hdri.stem}", flush=True)

    meta = {
        "asset": Path(bpy.data.filepath).stem, "objects": [o.name for o in objects], "device": device,
        "resolution": RESOLUTION, "margin": MARGIN, "samples": args.samples, "seed": args.seed,
        "normal_space": "object, Blender axes (Z up), OpenGL-style +Y", "variations": variations,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", args.out, flush=True)


main()

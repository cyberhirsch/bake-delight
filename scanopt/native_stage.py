"""Blender-free scanopt stages: reduce, unwrap, bake high->low and export, in plain Python.

Same job and same work files as blender_stage.py, built from permissively licensed parts:
    trimesh (MIT)               mesh import / glTF export
    fast-simplification (MIT)   quadric triangle decimation
    xatlas (MIT)                charting and packing
    Intel Embree via embreex    ray casting for the photo, normal and AO transfer
                                (Apache-2.0 / BSD)

prep writes into the work folder:
    lit.exr, normal_ts.exr, normal_os.exr, ao.exr, uv_tris.npy, prep.json   (as the Blender stage)
    low.npz      the optimised mesh: positions, polygons, per-corner UVs, axis convention

Normals in normal_os.exr use Blender's axes (Z up), like the training data. Formats that
are Y-up on disk (OBJ, glTF) are rotated the way Blender's importers rotate them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uvtools import join_split_polygons  # noqa: E402

Y_UP = {".obj", ".glb", ".gltf"}
# Blender axes from a Y-up file (what Blender's OBJ/glTF importers do), and glTF/OBJ axes from Blender's.
TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
BLENDER_TO_YUP = TO_BLENDER.T


def available() -> str | None:
    """None when every package the native backend needs imports, else what is missing."""
    missing = []
    for module in ("trimesh", "embreex", "xatlas", "fast_simplification"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module.replace("_", "-"))
    return None if not missing else "pip install " + " ".join(missing)


# ---------------------------------------------------------------- meshes

def weld(positions: np.ndarray, faces: np.ndarray):
    """Merge vertices that share a position (scans are split at UV seams). Returns the welded
    positions, the welded faces and, per input vertex, its welded index."""
    unique, inverse = np.unique(positions, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return unique, inverse[faces], inverse


def vertex_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted smooth normals."""
    corners = positions[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(positions)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)


def load_scan(path: Path, texture: Path | None):
    """The scan as welded triangles plus per-corner UVs, and its photo (linear RGB float32)."""
    import trimesh
    if path.suffix.lower() == ".fbx":
        raise SystemExit("FBX scans need the Blender backend; convert the scan to OBJ, PLY or GLB")
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not len(mesh.faces):
        raise SystemExit("The file contains no meshes")
    positions = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(positions):
        raise SystemExit("The scan has no UVs, so its photo texture cannot be transferred")
    corner_uv = np.asarray(uv, np.float64)[faces]
    if texture:
        photo = cv2.imread(str(texture), cv2.IMREAD_UNCHANGED)
        if photo is None:
            raise SystemExit(f"Cannot read the photo texture {texture}")
        photo = cv2.cvtColor(photo[..., :3] if photo.ndim == 3 else photo, cv2.COLOR_BGR2RGB if photo.ndim == 3 else cv2.COLOR_GRAY2RGB)
    else:
        material = getattr(mesh.visual, "material", None)
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        if image is None:
            raise SystemExit("No photo texture given and none found in the scan's material")
        photo = np.asarray(image.convert("RGB"))
    if photo.dtype == np.uint8:
        photo = photo.astype(np.float32) / 255
    elif photo.dtype == np.uint16:
        photo = photo.astype(np.float32) / 65535
    else:
        photo = photo.astype(np.float32)  # EXR / float TIFF: already linear
        return positions, faces, corner_uv, photo
    return positions, faces, corner_uv, srgb_to_linear(photo)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def export_high(scan: Path, obj: Path) -> int:
    """Welded triangle OBJ of the scan for the external quad remeshers (axes as on disk)."""
    import trimesh
    if scan.suffix.lower() == ".fbx":
        raise SystemExit("FBX scans need the Blender backend; convert the scan to OBJ, PLY or GLB")
    mesh = trimesh.load(str(scan), force="mesh", process=False)
    positions, faces, _ = weld(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64))
    obj.parent.mkdir(parents=True, exist_ok=True)
    with obj.open("w", encoding="ascii") as handle:
        handle.write("".join(f"v {x:.7g} {y:.7g} {z:.7g}\n" for x, y, z in positions))
        handle.write("".join(f"f {a} {b} {c}\n" for a, b, c in faces + 1))
    return len(faces)


def read_polygon_obj(path: Path):
    """Positions and polygons of an OBJ, keeping quads (trimesh would triangulate them)."""
    positions, sizes, corners = [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                positions.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [int(token.split("/")[0]) for token in line.split()[1:]]
                sizes.append(len(indices))
                corners.extend(i - 1 if i > 0 else len(positions) + i for i in indices)
    return np.asarray(positions, np.float64), np.asarray(sizes, np.int64), np.asarray(corners, np.int64)


def decimate(positions: np.ndarray, faces: np.ndarray, target: int):
    import fast_simplification
    if len(faces) <= target:
        return positions, faces
    reduced, reduced_faces = fast_simplification.simplify(positions.astype(np.float32), faces.astype(np.int32),
                                                          target_reduction=1 - target / len(faces))
    return reduced.astype(np.float64), reduced_faces.astype(np.int64)


def fan(sizes: np.ndarray):
    """Triangulate polygons as fans. Returns tri_loops (corner indices) and each triangle's polygon."""
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    count = sizes - 2
    tri_poly = np.repeat(np.arange(len(sizes)), count)
    step = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    base = starts[tri_poly]
    return np.stack([base, base + step + 1, base + step + 2], axis=1), tri_poly


# ---------------------------------------------------------------- UVs

def triangle_areas(uv: np.ndarray) -> np.ndarray:
    return 0.5 * ((uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
                  - (uv[:, 1, 1] - uv[:, 0, 1]) * (uv[:, 2, 0] - uv[:, 0, 0]))


def unwrap(positions, loop_verts, tri_loops, tri_poly, resolution: int, margin: int) -> tuple[np.ndarray, dict]:
    """xatlas charts the triangles, polygons it split across charts are rejoined, mirrored
    charts are flipped back, and xatlas packs the result as a UV mesh into the square."""
    import xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(positions.astype(np.float32), loop_verts[tri_loops].astype(np.uint32))
    atlas.generate()  # charts, loosely packed into one atlas; packed properly below
    _, indices, uvs = atlas[0]
    corner_uv = (uvs * np.array([atlas.width, atlas.height], np.float32) / max(atlas.width, atlas.height))[indices]
    charts = atlas.chart_count
    loop_uv, split = join_split_polygons(tri_loops, corner_uv, positions[loop_verts[tri_loops]], tri_poly, len(loop_verts))
    if not np.isfinite(loop_uv).all():
        raise RuntimeError("xatlas unwrap produced non-finite UVs")
    mirrored = triangle_areas(loop_uv[tri_loops]) < 0
    loop_uv[np.unique(tri_loops[mirrored]), 0] *= -1
    return pack(loop_uv, loop_verts, tri_loops, resolution, margin), {"unwrap": "xatlas", "charts": int(charts), "split_polygons": split}


def pack(loop_uv, loop_verts, tri_loops, resolution: int, margin: int, tries: int = 8) -> np.ndarray:
    """Pack the charts (connected UV triangles) into one square atlas, as large as fits."""
    import xatlas
    key = np.concatenate([loop_verts[:, None].astype(np.float64), np.round(loop_uv.astype(np.float64) * 1e6)], axis=1)
    _, first, uv_vertex = np.unique(key, axis=0, return_index=True, return_inverse=True)
    uv_vertex = uv_vertex.reshape(-1)
    uv_points = loop_uv[first].astype(np.float32)
    triangles = uv_vertex[tri_loops].astype(np.uint32)
    area = np.abs(triangle_areas(loop_uv[tri_loops])).sum()
    density = resolution * np.sqrt(0.45 / max(area, 1e-12))  # texels per UV unit, first guess
    best = None
    for _ in range(tries):
        atlas = xatlas.Atlas()
        atlas.add_uv_mesh(uv_points, triangles)
        options = xatlas.PackOptions()
        options.resolution, options.padding, options.texels_per_unit = resolution, margin, float(density)
        options.bilinear, options.rotate_charts = True, True
        atlas.generate(pack_options=options)
        if atlas.atlas_count == 1:
            _, indices, uvs = atlas[0]
            best = (uvs * np.array([atlas.width, atlas.height], np.float32) / resolution)[indices]
            fill = max(atlas.width, atlas.height) / resolution
            if fill > 0.97:
                break
            density *= min(1.15, 0.99 / fill)
        else:
            if best is not None:
                break
            density *= 0.9
    if best is None:
        raise RuntimeError("xatlas could not pack the charts into one texture")
    packed = loop_uv.copy()
    packed[tri_loops] = best
    return packed


# ---------------------------------------------------------------- bake

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



class Tracer:
    def __init__(self, positions: np.ndarray, faces: np.ndarray):
        from embreex import mesh_construction, rtcore_scene
        self.scene = rtcore_scene.EmbreeScene()
        self.mesh = mesh_construction.TriangleMesh(self.scene, positions[faces].astype(np.float32))

    def first(self, origins, directions, distances):
        result = self.scene.run(origins.astype(np.float32), directions.astype(np.float32),
                                dists=distances.astype(np.float32), output=1)
        return result["primID"], result["u"], result["v"], result["tfar"]

    def occluded(self, origins, directions, distances):
        return self.scene.run(origins.astype(np.float32), directions.astype(np.float32),
                              dists=distances.astype(np.float32), query="OCCLUDED") >= 0


def rasterise_ids(uv_tris: np.ndarray, size: int) -> np.ndarray:
    """Per texel, the index of the UV triangle covering it, -1 where none."""
    shift = 4
    points = np.empty_like(uv_tris, dtype=np.float64)
    points[..., 0] = uv_tris[..., 0] * size - 0.5
    points[..., 1] = (1.0 - uv_tris[..., 1]) * size - 0.5
    points = np.round(points * (1 << shift)).astype(np.int32)
    ids = np.zeros((size, size, 3), np.uint8)
    for index, triangle in enumerate(points, start=1):
        cv2.fillConvexPoly(ids, triangle, (index & 255, (index >> 8) & 255, index >> 16), lineType=cv2.LINE_8, shift=shift)
    ids = ids.astype(np.int64)
    return ids[..., 0] + (ids[..., 1] << 8) + (ids[..., 2] << 16) - 1


def loop_tangents(positions, loop_verts, loop_uv, tri_loops, loop_normals):
    """Per-corner tangents from the UV layout (accumulated per UV vertex), orthogonal to the
    smooth normal, plus the bitangent sign. Close to MikkTSpace, which glTF viewers use."""
    corner_pos = positions[loop_verts[tri_loops]]
    corner_uv = loop_uv[tri_loops].astype(np.float64)
    e1, e2 = corner_pos[:, 1] - corner_pos[:, 0], corner_pos[:, 2] - corner_pos[:, 0]
    d1, d2 = corner_uv[:, 1] - corner_uv[:, 0], corner_uv[:, 2] - corner_uv[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    det = np.where(np.abs(det) < 1e-20, 1e-20, det)[:, None]
    tangent = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det
    bitangent = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / det
    key = np.concatenate([loop_verts[:, None].astype(np.float64), np.round(loop_uv.astype(np.float64) * 1e6)], axis=1)
    _, uv_vertex = np.unique(key, axis=0, return_inverse=True)
    uv_vertex = uv_vertex.reshape(-1)
    t_sum = np.zeros((uv_vertex.max() + 1, 3))
    b_sum = np.zeros_like(t_sum)
    for k in range(3):
        np.add.at(t_sum, uv_vertex[tri_loops[:, k]], tangent)
        np.add.at(b_sum, uv_vertex[tri_loops[:, k]], bitangent)
    t, b = t_sum[uv_vertex], b_sum[uv_vertex]
    n = loop_normals
    t = t - n * np.sum(t * n, axis=1, keepdims=True)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-20)
    sign = np.where(np.sum(np.cross(n, t) * b, axis=1) < 0, -1.0, 1.0)
    return t, sign


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-20)


def sample_photo(photo: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = photo.shape[:2]
    columns = 4096
    rows = -(-len(uv) // columns)
    padded = np.zeros((rows * columns, 2), np.float32)
    padded[:len(uv)] = uv
    map_x = ((padded[:, 0] % 1.0) * width - 0.5).reshape(rows, columns)
    map_y = ((1.0 - padded[:, 1] % 1.0) * height - 0.5).reshape(rows, columns)
    values = cv2.remap(photo, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return values.reshape(-1, photo.shape[2])[:len(uv)]


def hemisphere(normals: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Cosine-weighted directions around each normal (Frisvad's orthonormal basis)."""
    radius, angle = np.sqrt(u1), 2 * np.pi * u2
    x, y, z = radius * np.cos(angle), radius * np.sin(angle), np.sqrt(np.maximum(0.0, 1 - u1))
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    sign = np.where(nz >= 0, 1.0, -1.0)
    a = -1.0 / (sign + nz)
    b = nx * ny * a
    tangent = np.stack([1 + sign * nx * nx * a, sign * b, -sign * nx], axis=1)
    bitangent = np.stack([b, sign + ny * ny * a, -ny], axis=1)
    return tangent * x[:, None] + bitangent * y[:, None] + normals * z[:, None]


def bake(high, low, photo, resolution: int, cage: float, max_ray: float, ao_samples: int, log=print) -> dict:
    high_pos, high_faces, high_uv = high
    positions, loop_verts, loop_uv, tri_loops = low
    tracer = Tracer(high_pos, high_faces)
    diagonal = float(np.linalg.norm(high_pos.max(axis=0) - high_pos.min(axis=0)))
    high_normals = vertex_normals(high_pos, high_faces)
    low_faces = loop_verts[tri_loops]
    loop_normals = vertex_normals(positions, low_faces)[loop_verts]
    loop_t, loop_sign = loop_tangents(positions, loop_verts, loop_uv, tri_loops, loop_normals)

    uv_tris = loop_uv[tri_loops].astype(np.float64)
    ids = rasterise_ids(uv_tris, resolution)
    rows, cols = np.nonzero(ids >= 0)
    tri = ids[rows, cols]
    # barycentrics of the texel centre in its UV triangle
    centre = np.stack([(cols + 0.5) / resolution, 1.0 - (rows + 0.5) / resolution], axis=1)
    a, b, c = uv_tris[tri, 0], uv_tris[tri, 1], uv_tris[tri, 2]
    v0, v1, v2 = b - a, c - a, centre - a
    det = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    det = np.where(np.abs(det) < 1e-20, 1e-20, det)
    w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / det
    w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / det
    weights = np.clip(np.stack([1 - w1 - w2, w1, w2], axis=1), 0, None)  # edge texels: clamp onto the triangle
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    def interpolate(per_loop):
        return np.einsum("nk,nkc->nc", weights, per_loop[tri_loops[tri]])

    point = interpolate(positions[loop_verts])
    normal = normalise(interpolate(loop_normals))
    tangent = interpolate(loop_t)
    tangent = normalise(tangent - normal * np.sum(tangent * normal, axis=1, keepdims=True))
    sign = np.where(interpolate(loop_sign[:, None])[:, 0] < 0, -1.0, 1.0)
    bitangent = np.cross(normal, tangent) * sign[:, None]
    log(f"  baking {len(tri):,} texels against {len(high_faces):,} triangles (Embree)")

    # Rays start just outside the low surface and look inward for the first high-res hit (as
    # Blender's selected-to-active bake without a cage); misses then look outward.
    start, reach = diagonal * cage, diagonal * (cage + max_ray)
    prim, hu, hv, _ = tracer.first(point + normal * start, -normal, np.full(len(point), reach))
    miss = prim < 0
    if miss.any():
        prim_out, hu_out, hv_out, _ = tracer.first(point[miss], normal[miss], np.full(miss.sum(), diagonal * max_ray))
        prim[miss], hu[miss], hv[miss] = prim_out, hu_out, hv_out
    miss = prim < 0
    if miss.any():  # still nothing: the nearest high-res vertex
        from scipy.spatial import cKDTree
        _, nearest = cKDTree(high_pos).query(point[miss])
        owner = np.full(len(high_pos), 0, np.int64)
        owner[high_faces.reshape(-1)] = np.repeat(np.arange(len(high_faces)), 3)
        prim[miss] = owner[nearest]
        corner = np.argmax(high_faces[prim[miss]] == nearest[:, None], axis=1)
        hu[miss], hv[miss] = (corner == 1).astype(np.float32), (corner == 2).astype(np.float32)
    hit_weights = np.stack([1 - hu - hv, hu, hv], axis=1).astype(np.float64)
    corners = high_faces[prim]
    hit_point = np.einsum("nk,nkc->nc", hit_weights, high_pos[corners])
    hit_normal = normalise(np.einsum("nk,nkc->nc", hit_weights, high_normals[corners]))
    hit_uv = np.einsum("nk,nkc->nc", hit_weights, high_uv[prim])

    lit = sample_photo(photo, hit_uv.astype(np.float32))
    normal_ts = np.stack([np.sum(hit_normal * tangent, axis=1), np.sum(hit_normal * bitangent, axis=1),
                          np.sum(hit_normal * normal, axis=1)], axis=1)
    normal_os = hit_normal

    # Ambient occlusion of the high-res surface; occluder distance as the Blender stage (10% of the diagonal).
    face = high_pos[corners]
    geometric = normalise(np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0]))
    geometric *= np.where(np.sum(geometric * hit_normal, axis=1) < 0, -1.0, 1.0)[:, None]
    origin = hit_point + geometric * diagonal * 2e-5
    reach = np.full(len(origin), diagonal * 0.1)
    rng = np.random.default_rng(0)
    open_sky = np.zeros(len(origin))
    for sample in range(ao_samples):
        u1 = (sample + rng.random(len(origin))) / ao_samples  # stratified in elevation
        direction = hemisphere(hit_normal, u1, rng.random(len(origin)))
        below = np.sum(direction * geometric, axis=1) < 0
        direction[below] -= 2 * np.sum(direction[below] * geometric[below], axis=1, keepdims=True) * geometric[below]
        open_sky += ~tracer.occluded(origin, direction, reach)
    ao = open_sky / max(ao_samples, 1)

    def image(values):
        out = np.zeros((resolution, resolution, values.shape[1]), np.float32)
        out[rows, cols] = values
        return out

    return {"lit": image(lit), "normal_ts": image(normal_ts * 0.5 + 0.5), "normal_os": image(normal_os),
            "ao": image(ao[:, None]), "coverage": (rows, cols), "misses": int(miss.sum())}


def write_exr(path: Path, rgb: np.ndarray) -> None:
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    cv2.imwrite(str(path), cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])


# ---------------------------------------------------------------- modes

def prep(scan: Path, texture: Path | None, out: Path, faces: int, resolution: int, margin: int, cage: float,
         max_ray: float, ao_samples: int, low_mesh: Path | None = None, topology: str = "tri", log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw_pos, raw_faces, corner_uv, photo = load_scan(scan, texture)
    high_pos, high_faces, _ = weld(raw_pos, raw_faces)
    source_faces = len(high_faces)
    if low_mesh:
        positions, sizes, loop_verts = read_polygon_obj(low_mesh)
    else:
        positions, triangles = decimate(high_pos, high_faces, faces)
        sizes, loop_verts = np.full(len(triangles), 3, np.int64), triangles.reshape(-1)
    tri_loops, tri_poly = fan(sizes)
    loop_uv, unwrap_info = unwrap(positions, loop_verts, tri_loops, tri_poly, resolution, margin)
    log(f"UNWRAP {unwrap_info}")

    axes = TO_BLENDER if scan.suffix.lower() in Y_UP else np.eye(3)
    maps = bake((high_pos, high_faces, corner_uv), (positions, loop_verts, loop_uv, tri_loops), photo,
                resolution, cage, max_ray, ao_samples, log=log)
    maps["normal_os"] = maps["normal_os"] @ axes.T * 0.5 + 0.5
    rows, cols = maps["coverage"]
    empty = np.ones((resolution, resolution), bool)
    empty[rows, cols] = False
    maps["normal_os"][empty] = 0
    for name in ("lit", "normal_ts", "normal_os", "ao"):  # extend into the gutter, like Blender's bake margin
        write_exr(out / f"{name}.exr", fill_gutter(maps[name], ~empty, margin))

    uv_tris = loop_uv[tri_loops].astype(np.float32)
    np.save(out / "uv_tris.npy", uv_tris)
    np.savez(out / "low.npz", positions=positions, sizes=sizes, loop_verts=loop_verts, loop_uv=loop_uv,
             axes=axes, y_up=scan.suffix.lower() in Y_UP)
    info = {"input": str(scan), "topology": topology, "source_faces": int(source_faces), "faces": int(len(sizes)),
            "quads": int((sizes == 4).sum()), "triangles": int(len(tri_loops)), "resolution": resolution,
            "device": "CPU (Embree)", "backend": "native", **unwrap_info,
            "uv_coverage": round(float(np.abs(triangle_areas(uv_tris)).sum()), 4), "bake_misses": maps["misses"]}
    (out / "prep.json").write_text(json.dumps(info, indent=2))
    return info


def assemble(work: Path, glb: Path, name: str) -> None:
    """glTF binary with the PBR material, and an OBJ + MTL that keeps quads (for DCC tools and
    the app's preview). Both in Y-up glTF/OBJ axes, as Blender exports them."""
    import trimesh
    from PIL import Image
    low = np.load(work / "low.npz")
    positions = low["positions"] @ (BLENDER_TO_YUP @ low["axes"]).T
    sizes, loop_verts, loop_uv = low["sizes"], low["loop_verts"], low["loop_uv"].astype(np.float64)
    tri_loops, _ = fan(sizes)
    loop_normals = vertex_normals(positions, loop_verts[tri_loops])[loop_verts]

    # glTF: one vertex per distinct (position, UV) corner
    key = np.concatenate([loop_verts[:, None].astype(np.float64), np.round(loop_uv * 1e6)], axis=1)
    _, first, vertex = np.unique(key, axis=0, return_index=True, return_inverse=True)
    vertex = vertex.reshape(-1)
    albedo = Image.open(work / "albedo.png").convert("RGB")
    roughness = np.asarray(Image.open(work / "roughness.png").convert("L"))
    ao = np.asarray(Image.open(work / "ao8.png").convert("L"))
    orm = Image.fromarray(np.stack([ao, roughness, np.zeros_like(ao)], axis=2))  # glTF: R occlusion, G roughness, B metallic
    material = trimesh.visual.material.PBRMaterial(
        name=f"{name}_pbr", baseColorTexture=albedo, metallicRoughnessTexture=orm, occlusionTexture=orm,
        normalTexture=Image.open(work / "normal.png").convert("RGB"), metallicFactor=0.0, roughnessFactor=1.0)
    mesh = trimesh.Trimesh(vertices=positions[loop_verts[first]], faces=vertex[tri_loops],
                           vertex_normals=loop_normals[first], process=False,
                           visual=trimesh.visual.TextureVisuals(uv=loop_uv[first], material=material))
    glb.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh), include_normals=True))

    obj = glb.with_suffix(".obj")
    uv_key, uv_index = np.unique(np.round(loop_uv * 1e6), axis=0, return_inverse=True)
    uv_index = uv_index.reshape(-1)
    vertex_normal = vertex_normals(positions, loop_verts[tri_loops])
    lines = [f"mtllib {obj.stem}.mtl\n", f"o {name}\n"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in positions]
    lines += [f"vt {u:.6f} {v:.6f}\n" for u, v in uv_key / 1e6]
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in vertex_normal]
    lines += ["s 1\n", f"usemtl {name}_pbr\n"]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    for begin, size in zip(starts, sizes):
        corners = range(begin, begin + size)
        lines.append("f " + " ".join(f"{loop_verts[c] + 1}/{uv_index[c] + 1}/{loop_verts[c] + 1}" for c in corners) + "\n")
    obj.write_text("".join(lines), encoding="ascii")
    textures = f"textures/{name}"
    obj.with_suffix(".mtl").write_text(
        f"newmtl {name}_pbr\nKd 1 1 1\nKs 0 0 0\nmap_Kd {textures}_albedo.png\nmap_Bump -bm 1 {textures}_normal_gl.png\n"
        f"Pr 1\nmap_Pr {textures}_roughness.png\n", encoding="ascii")


if __name__ == "__main__":
    print(available() or "native backend ready")
    sys.exit(0)

"""scanopt's stages: reduce, unwrap, bake high->low and export, in plain Python.

Built from permissively licensed parts:
    trimesh (MIT)               mesh import / glTF export
    fast-simplification (MIT)   quadric triangle decimation
    xatlas (MIT)                charting and packing
    Intel Embree via embreex    ray casting for the photo, normal and AO transfer
                                (Apache-2.0 / BSD)

The high mesh is a photo scan (one photo texture, to be de-lit) or a PBR model (glTF or OBJ + MTL,
any number of materials): then base colour, roughness, metallic, normal and occlusion maps are
sampled at each hit, normal maps decoded in the model's tangent frame and re-encoded in the low
mesh's.

The bake writes into the work folder:
    lit.exr, normal_ts.exr, normal_os.exr, ao.exr, uv_tris.npy, prep.json
    roughness.exr, metallic.exr                                             (PBR models; lit = base colour)
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


def _numbers(lines: list[bytes], columns: int, dtype) -> np.ndarray:
    """The first `columns` numbers of each line (prefix already stripped), parsed in one pass."""
    values = np.array(b" ".join(lines).split(), dtype=dtype)
    if len(values) % len(lines):  # mixed record lengths (optional w, vertex colours)
        return np.array([line.split()[:columns] for line in lines], dtype)
    return values.reshape(len(lines), -1)[:, :columns]


def read_obj(path: Path, details: bool = False):
    """Positions, triangles and per-corner UVs (or None) of an OBJ. trimesh's reader needs tens of
    minutes for a scan of several hundred MB; this one parses each record type with numpy in one go.
    details=True also returns per-corner normals (or None), each triangle's material index, the
    material names and the mtllib files."""
    v, vt, vn, f = [], [], [], []
    switches, libraries = [], []  # (face line, material name); MTL files
    with path.open("rb") as handle:
        for line in handle:
            if line[:2] == b"v ":
                v.append(line[2:])
            elif line[:3] == b"vt ":
                vt.append(line[3:])
            elif line[:2] == b"f ":
                f.append(line[2:])
            elif details and line[:3] == b"vn ":
                vn.append(line[3:])
            elif details and line[:7] == b"usemtl ":
                switches.append((len(f), line[7:].strip().decode("utf-8", "replace")))
            elif details and line[:7] == b"mtllib ":
                libraries.append(line[7:].strip().decode("utf-8", "replace"))
    if not f:
        empty = np.zeros((0, 3)), np.zeros((0, 3), np.int64), None
        return (*empty, None, np.zeros(0, np.int64), [], libraries) if details else empty
    positions = _numbers(v, 3, np.float64)
    texcoords = _numbers(vt, 2, np.float64) if vt else None
    normals = _numbers(vn, 3, np.float64) if vn else None
    del v, vt, vn
    fields = f[0].split()[0].count(b"/") + 1
    text = b" ".join(f).replace(b"//", b"/0/").replace(b"/", b" ")
    corners = np.array(text.split(), np.int64)
    if len(corners) == len(f) * 3 * fields:
        corners = corners.reshape(-1, 3, fields)
        per_line = np.ones(len(f), np.int64)
    else:  # polygons: fan-triangulate line by line
        triangles, per_line = [], []
        for line in f:
            polygon = np.array(line.replace(b"//", b"/0/").replace(b"/", b" ").split(), np.int64).reshape(-1, fields)
            triangles += [(polygon[0], polygon[k], polygon[k + 1]) for k in range(1, len(polygon) - 1)]
            per_line.append(len(polygon) - 2)
        corners = np.array(triangles, np.int64)
        per_line = np.array(per_line, np.int64)
    faces = corners[..., 0]
    faces = np.where(faces < 0, faces + len(positions), faces - 1)
    corner_uv = None
    if texcoords is not None and fields > 1:
        uv_index = corners[..., 1]
        corner_uv = texcoords[np.where(uv_index < 0, uv_index + len(texcoords), uv_index - 1)]
    if not details:
        return positions, faces, corner_uv
    corner_normals = None
    if normals is not None and fields > 2:
        normal_index = corners[..., 2]
        corner_normals = normals[np.where(normal_index < 0, normal_index + len(normals), normal_index - 1)]
    names = list(dict.fromkeys(name for _, name in switches)) or [""]
    line_material = np.zeros(len(f), np.int64)
    for (start, name), following in zip(switches, switches[1:] + [(len(f), None)]):
        line_material[start:following[0]] = names.index(name)
    return positions, faces, corner_uv, corner_normals, np.repeat(line_material, per_line), names, libraries


def load_scan(path: Path, texture: Path | None):
    """The scan as welded triangles plus per-corner UVs, and its photo (linear RGB float32)."""
    import trimesh
    if path.suffix.lower() == ".fbx":
        raise SystemExit("FBX is not supported; export the scan as OBJ, PLY or glTF")
    mesh = None
    if path.suffix.lower() == ".obj":
        positions, faces, corner_uv = read_obj(path)
    else:
        mesh = trimesh.load(str(path), force="mesh", process=False)
        positions, faces = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
        uv = getattr(mesh.visual, "uv", None)
        corner_uv = None if uv is None or len(uv) != len(positions) else np.asarray(uv, np.float64)[faces]
    if not len(faces):
        raise SystemExit("The file contains no meshes")
    if corner_uv is None:
        raise SystemExit("The scan has no UVs, so its photo texture cannot be transferred")
    if texture:
        photo = cv2.imread(str(texture), cv2.IMREAD_UNCHANGED)
        if photo is None:
            raise SystemExit(f"Cannot read the photo texture {texture}")
        photo = cv2.cvtColor(photo[..., :3] if photo.ndim == 3 else photo, cv2.COLOR_BGR2RGB if photo.ndim == 3 else cv2.COLOR_GRAY2RGB)
    else:
        if mesh is None:  # OBJ: the material's texture is resolved by trimesh
            mesh = trimesh.load(str(path), force="mesh", process=False)
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


def write_triangle_obj(obj: Path, positions: np.ndarray, faces: np.ndarray):
    obj.parent.mkdir(parents=True, exist_ok=True)
    with obj.open("w", encoding="ascii") as handle:
        handle.write("".join(f"v {x:.7g} {y:.7g} {z:.7g}\n" for x, y, z in positions))
        handle.write("".join(f"f {a} {b} {c}\n" for a, b, c in faces + 1))


def export_high(scan: Path, obj: Path, target: int = 0) -> int:
    """Welded triangle OBJ of the scan for the external quad remeshers (axes as on disk),
    QEM-decimated to `target` faces first when given: the remeshers then run in seconds instead of
    minutes on multi-million-face scans, and the bake still projects from the full scan."""
    import trimesh
    if scan.suffix.lower() == ".fbx":
        raise SystemExit("FBX is not supported; export the scan as OBJ, PLY or glTF")
    if scan.suffix.lower() == ".obj":
        positions, faces, _ = read_obj(scan)
    else:
        mesh = trimesh.load(str(scan), force="mesh", process=False)
        positions, faces = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
    positions, faces, _ = weld(positions, faces)
    if target:
        positions, faces = decimate(positions, faces, target)
    write_triangle_obj(obj, positions, faces)
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


# ---------------------------------------------------------------- PBR sources

PBR_MAPS = ("base_color", "roughness", "metallic", "occlusion")


def _float_image(image, srgb: bool) -> np.ndarray:
    """A PIL image or OpenCV array as float32 H x W x C; 8/16-bit sRGB colour is made linear."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255
    elif array.dtype == np.uint16:
        array = array.astype(np.float32) / 65535
    else:
        return array.astype(np.float32)  # EXR / float TIFF: already linear
    return srgb_to_linear(array) if srgb else array


def read_map(path: Path, srgb: bool) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Cannot read the texture {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
    return _float_image(image, srgb)


class Material:
    """One PBR material: per map an image (float32, H x W x C, linear) or None, times a constant factor."""

    def __init__(self, name: str = ""):
        self.name = name
        self.images: dict[str, np.ndarray | None] = {key: None for key in (*PBR_MAPS, "normal")}
        self.factors = {"base_color": np.ones(3, np.float32), "roughness": 1.0, "metallic": 0.0, "occlusion": 1.0}

    def set_map(self, key: str, image: np.ndarray, channel: int | None = None):
        """A map that is the value itself (MTL maps, overrides): its factor becomes 1."""
        self.images[key] = image if channel is None else image[..., channel:channel + 1]
        if key in self.factors:
            self.factors[key] = np.ones(3, np.float32) if key == "base_color" else 1.0

    def sample(self, uv: np.ndarray) -> dict:
        out = {}
        for key in PBR_MAPS:
            image, factor = self.images[key], self.factors[key]
            columns = 3 if key == "base_color" else 1
            value = sample_photo(image, uv)[:, :columns] if image is not None else np.ones((len(uv), columns), np.float32)
            if value.shape[1] < columns:  # grey base colour image
                value = np.repeat(value[:, :1], columns, axis=1)
            out[key] = value * np.asarray(factor, np.float32)
        if self.images["normal"] is not None:
            out["normal"] = sample_photo(self.images["normal"], uv)[:, :3] * 2 - 1
        return out


def _looks_like_normal_map(image: np.ndarray) -> bool:
    """OBJ's map_Bump holds either a height map or a tangent-space normal map: tell them apart."""
    if image.shape[2] < 3:
        return False
    mean = image[::8, ::8, :3].reshape(-1, 3).mean(axis=0)
    return abs(mean[0] - 0.5) < 0.15 and abs(mean[1] - 0.5) < 0.15 and mean[2] > 0.6


def _mtl_file(value: str, folder: Path) -> Path | None:
    """The file name of an MTL map statement, with options such as -bm 1 or -s 1 1 1 removed."""
    tokens, k = value.split(), 0
    while k < len(tokens) and tokens[k].startswith("-"):
        k += 1
        while k < len(tokens) - 1 and (tokens[k].replace(".", "", 1).lstrip("-").isdigit() or tokens[k] in ("on", "off")):
            k += 1
    name = " ".join(tokens[k:]).strip('"')
    if not name:
        return None
    for candidate in (folder / name, folder / Path(name).name, folder / "textures" / Path(name).name):
        if candidate.is_file():
            return candidate
    return None


def read_mtl(libraries: list[str], folder: Path, names: list[str], log=print) -> list[Material]:
    statements: dict[str, dict[str, str]] = {}
    current = None
    for library in libraries:
        path = folder / library
        if not path.is_file():
            log(f"  material library {library} not found")
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.strip().partition(" ")
            if key == "newmtl":
                current = statements.setdefault(value.strip(), {})
            elif current is not None and key:
                current[key.lower()] = value.strip()
    materials = []
    for name in names:
        material, values = Material(name), statements.get(name, {})
        if "kd" in values:
            material.factors["base_color"] = np.array([float(x) for x in values["kd"].split()[:3]], np.float32)
        for key, factor in (("roughness", "pr"), ("metallic", "pm")):
            if factor in values:
                material.factors[key] = float(values[factor].split()[0])
        for key, statement, srgb in (("base_color", "map_kd", True), ("roughness", "map_pr", False),
                                     ("metallic", "map_pm", False), ("occlusion", "map_ka", False)):
            path = _mtl_file(values[statement], folder) if statement in values else None
            if path:  # Kd next to map_Kd is usually a preview colour: the map wins
                material.set_map(key, read_map(path, srgb), channel=None if key == "base_color" else 0)
        for statement in ("norm", "map_bump", "bump"):
            path = _mtl_file(values[statement], folder) if statement in values else None
            if path:
                image = read_map(path, srgb=False)
                if _looks_like_normal_map(image):
                    material.images["normal"] = image
                else:
                    log(f"  {path.name} is a height map, not a normal map: skipped")
                break
        materials.append(material)
    return materials


def _gltf_material(source) -> Material:
    """trimesh's material of a glTF primitive (or of a textured PLY) as a Material."""
    from trimesh.visual.material import PBRMaterial
    material = Material(getattr(source, "name", "") or "")
    if source is None:
        return material
    if getattr(source, "baseColorFactor", None) is not None:  # linear in glTF; trimesh keeps it as 0-255
        factor = np.asarray(source.baseColorFactor, np.float32)[:3]
        material.factors["base_color"] = factor / 255 if np.asarray(source.baseColorFactor).dtype == np.uint8 else factor
    image = getattr(source, "baseColorTexture", None) or getattr(source, "image", None)
    if image is not None:
        material.images["base_color"] = _float_image(image.convert("RGB"), srgb=True)
    if not isinstance(source, PBRMaterial):
        return material
    # glTF defaults: metallic and roughness factors 1
    material.factors["roughness"] = 1.0 if source.roughnessFactor is None else float(source.roughnessFactor)
    material.factors["metallic"] = 1.0 if source.metallicFactor is None else float(source.metallicFactor)
    image = source.metallicRoughnessTexture
    if image is not None:  # glTF: G roughness, B metallic
        array = _float_image(image.convert("RGB"), srgb=False)
        material.images["roughness"], material.images["metallic"] = array[..., 1:2], array[..., 2:3]
    image = getattr(source, "occlusionTexture", None)
    if image is not None:
        material.images["occlusion"] = _float_image(image.convert("RGB"), srgb=False)[..., 0:1]
    image = getattr(source, "normalTexture", None)
    if image is not None:
        material.images["normal"] = _float_image(image.convert("RGB"), srgb=False)
    return material


class PbrSource:
    """A high-res model with PBR materials: triangles, per-corner UVs and normals, each triangle's
    material. Normal maps are decoded with the model's own tangent frame (as glTF viewers do)."""

    def __init__(self, positions, faces, corner_uv, corner_normals, face_material, materials: list[Material]):
        self.positions, self.faces, self.corner_uv = positions, faces, corner_uv
        if corner_normals is None:  # smooth normals across welded positions
            welded, welded_faces, _ = weld(positions, faces)
            corner_normals = vertex_normals(welded, welded_faces)[welded_faces]
        self.corner_normals = normalise(corner_normals)
        self.face_material, self.materials = face_material, materials
        self.tangents = None
        if any(m.images["normal"] is not None for m in materials):
            count = len(faces)
            tangent, sign = loop_tangents(positions, faces.reshape(-1), corner_uv.reshape(-1, 2),
                                          np.arange(3 * count).reshape(count, 3), self.corner_normals.reshape(-1, 3))
            self.tangents = tangent.reshape(count, 3, 3), sign.reshape(count, 3)

    def normal_at(self, prim, weights):
        return normalise(np.einsum("nk,nkc->nc", weights, self.corner_normals[prim]))

    def sample(self, prim, weights, normal) -> dict:
        """Every map at the hit points; 'normal' is the normal-mapped world normal."""
        uv = np.einsum("nk,nkc->nc", weights, self.corner_uv[prim]).astype(np.float32)
        out = {"base_color": np.zeros((len(prim), 3), np.float32), "normal": normal.copy(),
               **{key: np.zeros((len(prim), 1), np.float32) for key in PBR_MAPS[1:]}}
        material_of_hit = self.face_material[prim]
        for index, material in enumerate(self.materials):
            hits = np.nonzero(material_of_hit == index)[0]
            if not len(hits):
                continue
            values = material.sample(uv[hits])
            for key in PBR_MAPS:
                out[key][hits] = values[key]
            if "normal" in values and self.tangents is not None:
                n = normal[hits]
                t = np.einsum("nk,nkc->nc", weights[hits], self.tangents[0][prim[hits]])
                t = normalise(t - n * np.sum(t * n, axis=1, keepdims=True))
                s = np.where(np.einsum("nk,nk->n", weights[hits], self.tangents[1][prim[hits]]) < 0, -1.0, 1.0)
                b = np.cross(n, t) * s[:, None]
                m = values["normal"]
                out["normal"][hits] = normalise(t * m[:, 0:1] + b * m[:, 1:2] + n * m[:, 2:3])
        return out


def load_pbr(path: Path, overrides: dict[str, Path | None], log=print) -> PbrSource:
    """A high-res PBR model from glTF/GLB, OBJ + MTL, or PLY/STL with maps given in `overrides`
    (base_color, roughness, metallic, normal, occlusion), which also replace the file's own maps."""
    import trimesh
    suffix = path.suffix.lower()
    if suffix == ".fbx":
        raise SystemExit("FBX models need converting to glTF or OBJ for the PBR bake")
    if suffix == ".obj":
        positions, faces, corner_uv, corner_normals, face_material, names, libraries = read_obj(path, details=True)
        materials = read_mtl(libraries, path.parent, names, log)
    else:
        loaded = trimesh.load(str(path), process=False)
        meshes = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else [loaded]
        meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.faces)]
        parts, materials, known = [], [], {}
        offset = 0
        for mesh in meshes:
            mesh_faces = np.asarray(mesh.faces, np.int64)
            uv = getattr(mesh.visual, "uv", None)
            uv = np.zeros((len(mesh.vertices), 2)) if uv is None or len(uv) != len(mesh.vertices) else np.asarray(uv, np.float64)
            source = getattr(mesh.visual, "material", None)
            if id(source) not in known:
                known[id(source)] = len(materials)
                materials.append(_gltf_material(source))
            parts.append((np.asarray(mesh.vertices, np.float64), mesh_faces + offset, uv[mesh_faces],
                          np.asarray(mesh.vertex_normals, np.float64)[mesh_faces], np.full(len(mesh_faces), known[id(source)])))
            offset += len(mesh.vertices)
        if not parts:
            raise SystemExit("The file contains no meshes")
        positions, faces, corner_uv, corner_normals, face_material = (np.concatenate(p) for p in zip(*parts))
    if not len(faces):
        raise SystemExit("The file contains no meshes")
    if not materials:
        materials = [Material()]
    if corner_uv is None:
        if any(overrides.values()) or any(image is not None for m in materials for image in m.images.values()):
            raise SystemExit("The model has no UVs, so its textures cannot be transferred")
        corner_uv = np.zeros((len(faces), 3, 2))
    for key, file in overrides.items():
        if file:
            image = read_map(Path(file), srgb=key == "base_color")
            for material in materials:
                if key == "normal":
                    material.images["normal"] = image
                else:
                    material.set_map(key, image, channel=None if key == "base_color" else 0)
    for material in materials:
        present = [key for key, image in material.images.items() if image is not None]
        log(f"  material {material.name or '(unnamed)'}: {', '.join(present) or 'constants only'}")
    return PbrSource(positions, faces, corner_uv, corner_normals, face_material, materials)


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


def bake(high, low, photo, resolution: int, cage: float, max_ray: float, ao_samples: int, log=print,
         pbr: PbrSource | None = None) -> dict:
    """Scan bake: `high` is (positions, faces, corner UVs) and `photo` the scan's texture. PBR bake:
    `pbr` is the model; its maps are sampled at each hit instead of the photo (into 'lit', plus
    'roughness' and 'metallic'), its normal maps go into the normals and its occlusion into the AO."""
    if pbr is not None:
        high = pbr.positions, pbr.faces, pbr.corner_uv
    high_pos, high_faces, high_uv = high
    positions, loop_verts, loop_uv, tri_loops = low
    tracer = Tracer(high_pos, high_faces)
    diagonal = float(np.linalg.norm(high_pos.max(axis=0) - high_pos.min(axis=0)))
    high_normals = vertex_normals(high_pos, high_faces) if pbr is None else None
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
    extra = {}
    if pbr is None:
        hit_normal = normalise(np.einsum("nk,nkc->nc", hit_weights, high_normals[corners]))
        hit_uv = np.einsum("nk,nkc->nc", hit_weights, high_uv[prim])
        lit = sample_photo(photo, hit_uv.astype(np.float32))
        detail = hit_normal
    else:
        hit_normal = pbr.normal_at(prim, hit_weights)
        values = pbr.sample(prim, hit_weights, hit_normal)
        lit, detail = values["base_color"], values["normal"]
        extra = {key: values[key] for key in ("roughness", "metallic")}
    normal_ts = np.stack([np.sum(detail * tangent, axis=1), np.sum(detail * bitangent, axis=1),
                          np.sum(detail * normal, axis=1)], axis=1)
    normal_os = detail

    # Ambient occlusion of the high-res surface; occluders count within 10% of the diagonal.
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
    if pbr is not None:
        ao = ao * values["occlusion"][:, 0]  # the model's own occlusion map on top of the baked AO

    def image(values):
        out = np.zeros((resolution, resolution, values.shape[1]), np.float32)
        out[rows, cols] = values
        return out

    return {"lit": image(lit), "normal_ts": image(normal_ts * 0.5 + 0.5), "normal_os": image(normal_os),
            "ao": image(ao[:, None]), **{key: image(value) for key, value in extra.items()},
            "coverage": (rows, cols), "misses": int(miss.sum())}


def write_exr(path: Path, rgb: np.ndarray) -> None:
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    cv2.imwrite(str(path), cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])


# ---------------------------------------------------------------- modes

def reduce_scan(scan: Path, faces: int, obj: Path) -> int:
    """Triangle route of the remesh stage: the scan QEM-decimated to `faces`, as an OBJ in disk axes
    (the same convention as the quad remeshers' output)."""
    return export_high(scan, obj, faces)


def write_preview_obj(low_mesh: Path, scan: Path, obj: Path) -> None:
    """The remeshed low mesh for the app's viewer, which needs UVs: one placeholder UV, Y-up like the final OBJ."""
    positions, sizes, loop_verts = read_polygon_obj(low_mesh)
    axes = TO_BLENDER if scan.suffix.lower() in Y_UP else np.eye(3)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    with obj.open("w", encoding="ascii") as handle:
        handle.write("".join(f"v {x:.7g} {y:.7g} {z:.7g}\n" for x, y, z in positions @ (BLENDER_TO_YUP @ axes).T))
        handle.write("vt 0.5 0.5\n")
        handle.write("".join("f " + " ".join(f"{loop_verts[k] + 1}/1" for k in range(start, start + size)) + "\n"
                             for start, size in zip(starts, sizes)))


def layout(low_mesh: Path, scan: Path, out: Path, resolution: int, margin: int, log=print) -> dict:
    """UV stage: unwrap and pack the low mesh. Writes layout.npz for the bake stage and low_uv.obj
    (Y-up like the final OBJ) for the preview."""
    out.mkdir(parents=True, exist_ok=True)
    positions, sizes, loop_verts = read_polygon_obj(low_mesh)
    tri_loops, tri_poly = fan(sizes)
    loop_uv, unwrap_info = unwrap(positions, loop_verts, tri_loops, tri_poly, resolution, margin)
    log(f"UNWRAP {unwrap_info}")
    np.savez(out / "layout.npz", positions=positions, sizes=sizes, loop_verts=loop_verts, loop_uv=loop_uv,
             resolution=resolution, margin=margin, info=json.dumps(unwrap_info))
    axes = TO_BLENDER if scan.suffix.lower() in Y_UP else np.eye(3)
    shown = positions @ (BLENDER_TO_YUP @ axes).T
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    with (out / "low_uv.obj").open("w", encoding="ascii") as handle:
        handle.write("".join(f"v {x:.7g} {y:.7g} {z:.7g}\n" for x, y, z in shown))
        handle.write("".join(f"vt {u:.6f} {v:.6f}\n" for u, v in loop_uv))
        handle.write("".join("f " + " ".join(f"{loop_verts[k] + 1}/{k + 1}" for k in range(start, start + size)) + "\n"
                             for start, size in zip(starts, sizes)))
    return unwrap_info


def bake_layout(scan: Path, texture: Path | None, out: Path, cage: float, max_ray: float, ao_samples: int,
                topology: str = "tri", log=print, source: str = "scan", maps: dict | None = None) -> dict:
    """Bake stage: project the scan (source="scan") or a PBR model (source="pbr", `maps` overriding
    its material's maps) onto the unwrapped low mesh from layout.npz."""
    saved = np.load(out / "layout.npz")
    positions, sizes, loop_verts, loop_uv = saved["positions"], saved["sizes"], saved["loop_verts"], saved["loop_uv"]
    resolution, unwrap_info = int(saved["resolution"]), json.loads(str(saved["info"]))
    margin = int(saved["margin"])
    tri_loops, _ = fan(sizes)
    if source == "pbr":
        model = load_pbr(scan, {"base_color": texture, **(maps or {})}, log)
        return _bake_and_write(scan, out, model.positions, model.faces, model.corner_uv, None, positions, sizes, loop_verts,
                               loop_uv, tri_loops, unwrap_info, resolution, margin, cage, max_ray, ao_samples, topology,
                               log, pbr=model)
    raw_pos, raw_faces, corner_uv, photo = load_scan(scan, texture)
    high_pos, high_faces, _ = weld(raw_pos, raw_faces)
    return _bake_and_write(scan, out, high_pos, high_faces, corner_uv, photo, positions, sizes, loop_verts, loop_uv,
                           tri_loops, unwrap_info, resolution, margin, cage, max_ray, ao_samples, topology, log)


def _bake_and_write(scan, out, high_pos, high_faces, corner_uv, photo, positions, sizes, loop_verts, loop_uv, tri_loops,
                    unwrap_info, resolution, margin, cage, max_ray, ao_samples, topology, log, pbr=None) -> dict:
    source_faces = len(high_faces)
    axes = TO_BLENDER if scan.suffix.lower() in Y_UP else np.eye(3)
    maps = bake((high_pos, high_faces, corner_uv), (positions, loop_verts, loop_uv, tri_loops), photo,
                resolution, cage, max_ray, ao_samples, log=log, pbr=pbr)
    maps["normal_os"] = maps["normal_os"] @ axes.T * 0.5 + 0.5
    rows, cols = maps["coverage"]
    empty = np.ones((resolution, resolution), bool)
    empty[rows, cols] = False
    maps["normal_os"][empty] = 0
    names = ("lit", "normal_ts", "normal_os", "ao") + (("roughness", "metallic") if pbr is not None else ())
    for name in names:  # extend into the gutter, like Blender's bake margin
        write_exr(out / f"{name}.exr", fill_gutter(maps[name], ~empty, margin))

    uv_tris = loop_uv[tri_loops].astype(np.float32)
    np.save(out / "uv_tris.npy", uv_tris)
    np.savez(out / "low.npz", positions=positions, sizes=sizes, loop_verts=loop_verts, loop_uv=loop_uv,
             axes=axes, y_up=scan.suffix.lower() in Y_UP)
    info = {"input": str(scan), "source": "pbr" if pbr is not None else "scan", "topology": topology, "source_faces": int(source_faces), "faces": int(len(sizes)),
            "quads": int((sizes == 4).sum()), "triangles": int(len(tri_loops)), "resolution": resolution,
            "device": "CPU (Embree)", "backend": "native", **unwrap_info,
            "uv_coverage": round(float(np.abs(triangle_areas(uv_tris)).sum()), 4), "bake_misses": maps["misses"]}
    (out / "prep.json").write_text(json.dumps(info, indent=2))
    return info


def assemble(work: Path, glb: Path, name: str, metallic: bool = False) -> None:
    """glTF binary with the PBR material, and an OBJ + MTL that keeps quads (for DCC tools and
    the app's preview). Both in Y-up glTF/OBJ axes, as Blender exports them. metallic: use
    work/metallic.png (PBR bakes); scans are dielectric."""
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
    metal = np.asarray(Image.open(work / "metallic.png").convert("L")) if metallic else np.zeros_like(ao)
    orm = Image.fromarray(np.stack([ao, roughness, metal], axis=2))  # glTF: R occlusion, G roughness, B metallic
    material = trimesh.visual.material.PBRMaterial(
        name=f"{name}_pbr", baseColorTexture=albedo, metallicRoughnessTexture=orm, occlusionTexture=orm,
        normalTexture=Image.open(work / "normal.png").convert("RGB"), metallicFactor=1.0 if metallic else 0.0,
        roughnessFactor=1.0)
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
        f"Pr 1\nmap_Pr {textures}_roughness.png\n" + (f"Pm 1\nmap_Pm {textures}_metallic.png\n" if metallic else ""),
        encoding="ascii")


if __name__ == "__main__":
    print(available() or "native backend ready")
    sys.exit(0)

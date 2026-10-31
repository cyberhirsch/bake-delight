"""Bake & Delight project files (*.bad).

A .bad file is UTF-8 JSON, so it diffs and merges like source:

    {
      "format": "bake-delight",
      "version": 1,
      "name": "stone_01",
      "created": "2026-10-29T12:00:00", "modified": "...",
      "notes": "",
      "settings": {"delight": {...}, "scan": {...}},
      "results":  {"delight": {...}, "scan": {...}}
    }

Paths are stored relative to the .bad file (forward slashes) when they sit on the same
drive, absolute otherwise, so a project folder can be moved or shared as a whole. In
memory every path is absolute. Machine-specific settings (Blender, remesher folders, UI)
stay in the GUI's settings.json and never go into a project.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from pathlib import Path

FORMAT = "bake-delight"
VERSION = 1
SUFFIX = ".bad"

# section -> key -> default. The default's type is the setting's type.
SCHEMA = {
    "delight": {
        "model": "", "lit": "", "normal": "", "ao": "", "mask": "",
        "output_dir": "", "output_name": "delighted_albedo.png",
        "onnx_model": "", "force_cpu": False,
        "tiled": True, "tile_size": 0, "tile_overlap": 0.25, "bit_depth": 8, "apply_mask": True,
    },
    "scan": {
        "scan": "", "photo": "", "out": "", "source": "scan",
        "remesher": "decimate", "faces": 10000, "sharp": False,
        "unwrap": "xatlas", "smart_angle": 66.0, "margin": 16,
        "backend": "auto", "resolution": 2048, "ao_samples": 64, "cage": 0.01, "max_ray": 0.03,
        "roughness": 0.8, "roughness_variation": 0.25, "tiled": False, "keep_work": False,
    },
}
PATH_KEYS = {
    "delight": {"model", "lit", "normal", "ao", "mask", "output_dir", "onnx_model"},
    "scan": {"scan", "photo", "out"},
}
CHOICES = {
    ("delight", "bit_depth"): [8, 16],
    ("delight", "tile_size"): [0, 512, 1024, 2048, 4096],
    ("scan", "source"): ["scan", "pbr"],
    ("scan", "remesher"): ["decimate", "autoremesher", "quadwild"],
    ("scan", "unwrap"): ["xatlas", "smart"],
    ("scan", "backend"): ["auto", "blender", "native"],
    ("scan", "resolution"): [512, 1024, 2048, 4096],
}
DESCRIPTIONS = {
    ("delight", "model"): "OBJ mesh for the 3D preview (optional)",
    ("delight", "lit"): "lit photo texture (RGB) to de-light",
    ("delight", "normal"): "object-space normal map, same size as lit",
    ("delight", "ao"): "ambient occlusion map",
    ("delight", "mask"): "UV mask: white inside the UV islands",
    ("delight", "output_dir"): "output folder; empty = the lit map's folder",
    ("delight", "output_name"): "output file name",
    ("delight", "onnx_model"): "de-lighter ONNX model; empty = newest trained one",
    ("delight", "force_cpu"): "run inference on the CPU",
    ("delight", "tiled"): "tiled inference (less VRAM; 2048 px and up)",
    ("delight", "tile_size"): "tile size in px; 0 = half the image",
    ("delight", "tile_overlap"): "tile overlap, fraction of the tile (0.05-0.5)",
    ("delight", "bit_depth"): "output PNG bit depth",
    ("delight", "apply_mask"): "black out texels outside the UV islands",
    ("scan", "scan"): "high-res scan: OBJ, PLY, FBX, GLB/glTF, STL",
    ("scan", "photo"): "photo texture (base colour for pbr); empty = taken from the material",
    ("scan", "out"): "output folder; empty = <scan>_optimized next to the scan",
    ("scan", "source"): "scan: photo scan, de-lit; pbr: model with PBR materials, baked as they are",
    ("scan", "remesher"): "decimate (triangles), autoremesher or quadwild (quads)",
    ("scan", "faces"): "target face count",
    ("scan", "sharp"): "quads: keep sharp feature edges (hard-surface scans)",
    ("scan", "unwrap"): "UV unwrapper",
    ("scan", "smart_angle"): "Smart UV Project angle limit, degrees",
    ("scan", "margin"): "texels of padding around each UV chart",
    ("scan", "backend"): "bake backend: blender (Cycles) or native (Embree); auto = Blender if installed",
    ("scan", "resolution"): "texture size in px",
    ("scan", "ao_samples"): "AO samples per texel",
    ("scan", "cage"): "bake ray start offset, fraction of the bounds diagonal",
    ("scan", "max_ray"): "bake ray search distance, fraction of the bounds diagonal",
    ("scan", "roughness"): "base roughness",
    ("scan", "roughness_variation"): "roughness variation from surface relief and cavities",
    ("scan", "tiled"): "tiled de-light inference",
    ("scan", "keep_work"): "keep intermediate EXRs in <out>/work",
}
RESULT_PATH_KEYS = {"albedo", "glb", "obj", "fbx", "report", "textures", "out"}


class ProjectError(ValueError):
    pass


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def coerce(section: str, key: str, value):
    """`value` converted to the setting's type; strings as typed on a command line are accepted."""
    if section not in SCHEMA or key not in SCHEMA[section]:
        raise ProjectError(f"unknown setting {section}.{key}")
    default = SCHEMA[section][key]
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
                    raise ValueError(value)
                value = lowered in {"1", "true", "yes", "on"}
            value = bool(value)
        elif isinstance(default, int):
            if isinstance(value, str) and value.strip().lower() == "auto" and key == "tile_size":
                value = 0
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        else:
            value = "" if value is None else str(value)
    except (TypeError, ValueError):
        raise ProjectError(f"{section}.{key}: expected {type(default).__name__}, got {value!r}") from None
    choices = CHOICES.get((section, key))
    if choices and value not in choices:
        raise ProjectError(f"{section}.{key}: {value!r} is not one of {choices}")
    return value


def split_key(dotted: str) -> tuple[str, str]:
    """'scan.faces' -> ('scan', 'faces'). A bare key is accepted when only one section has it."""
    if "." in dotted:
        section, key = dotted.split(".", 1)
        if section in SCHEMA and key in SCHEMA[section]:
            return section, key
        raise ProjectError(f"unknown setting {dotted}")
    owners = [section for section, keys in SCHEMA.items() if dotted in keys]
    if len(owners) == 1:
        return owners[0], dotted
    if owners:
        raise ProjectError(f"{dotted} is ambiguous; use {' or '.join(f'{s}.{dotted}' for s in owners)}")
    raise ProjectError(f"unknown setting {dotted}")


def describe_settings() -> list[dict]:
    """Every setting with its type, default, choices and description (for --help style listings)."""
    return [{"key": f"{section}.{key}", "type": type(default).__name__, "default": default,
             "choices": CHOICES.get((section, key)), "path": key in PATH_KEYS[section],
             "description": DESCRIPTIONS.get((section, key), "")}
            for section, keys in SCHEMA.items() for key, default in keys.items()]


class Project:
    def __init__(self, name: str = "untitled", path: Path | None = None):
        self.path = Path(path).resolve() if path else None
        self.name = name
        self.notes = ""
        self.created = self.modified = _now()
        self.settings = copy.deepcopy(SCHEMA)
        self.results: dict = {}
        self.warnings: list[str] = []

    # ------------------------------------------------------------ access

    def get(self, dotted: str):
        section, key = split_key(dotted)
        return self.settings[section][key]

    def set(self, dotted: str, value) -> None:
        section, key = split_key(dotted)
        value = coerce(section, key, value)
        if key in PATH_KEYS[section] and value:
            value = str(Path(value).expanduser().resolve())
        self.settings[section][key] = value

    def update(self, values: dict) -> None:
        """Set several settings; nothing changes if any of them is invalid."""
        staged = Project()
        staged.settings = copy.deepcopy(self.settings)
        for dotted, value in values.items():
            staged.set(dotted, value)
        self.settings = staged.settings

    def autofill_maps(self) -> list[str]:
        """Fill empty normal/ao/mask from files next to the lit map whose names contain those words."""
        lit = self.settings["delight"]["lit"]
        if not lit or not os.path.isfile(lit):
            return []
        filled = []
        for file in sorted(os.listdir(os.path.dirname(lit))):
            full = os.path.join(os.path.dirname(lit), file)
            for key in ("normal", "ao", "mask"):
                if key in file.lower() and not self.settings["delight"][key] and full != lit:
                    self.settings["delight"][key] = full
                    filled.append(f"delight.{key}")
                    break
        return filled

    # ------------------------------------------------------------ files

    def _to_disk(self, value: str) -> str:
        if not value or self.path is None:
            return value
        try:
            relative = os.path.relpath(value, self.path.parent)
        except ValueError:  # another drive
            return value
        return Path(relative).as_posix()

    def _from_disk(self, value: str, base: Path) -> str:
        if not value:
            return value
        candidate = Path(value)
        return str(candidate if candidate.is_absolute() else (base / candidate).resolve())

    def _map_results(self, results: dict, convert) -> dict:
        mapped = {}
        for kind, entry in results.items():
            if isinstance(entry, dict):
                mapped[kind] = {k: convert(v) if k in RESULT_PATH_KEYS and isinstance(v, str) else v for k, v in entry.items()}
            else:
                mapped[kind] = entry
        return mapped

    def to_dict(self, relative: bool = True) -> dict:
        convert = self._to_disk if relative else (lambda value: value)
        settings = {section: {key: convert(value) if key in PATH_KEYS[section] else value for key, value in keys.items()}
                    for section, keys in self.settings.items()}
        return {"format": FORMAT, "version": VERSION, "name": self.name, "created": self.created,
                "modified": self.modified, "notes": self.notes, "settings": settings,
                "results": self._map_results(self.results, convert)}

    def save(self, path: Path | str | None = None) -> Path:
        if path is not None:
            path = Path(path).expanduser().resolve()
            if path.suffix.lower() != SUFFIX:
                path = path.with_name(path.name + SUFFIX)
            self.path = path
        if self.path is None:
            raise ProjectError("the project has no file yet; pass a path")
        self.modified = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)  # never leaves a half-written project behind
        return self.path

    @classmethod
    def load(cls, path: Path | str) -> "Project":
        path = Path(path).expanduser().resolve()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ProjectError(f"cannot read {path}: {error}") from None
        except ValueError as error:
            raise ProjectError(f"{path} is not a Bake & Delight project ({error})") from None
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ProjectError(f"{path} is not a Bake & Delight project")
        if int(data.get("version", 0)) > VERSION:
            raise ProjectError(f"{path} was written by a newer Bake & Delight (format {data['version']}, this is {VERSION})")
        project = cls(data.get("name") or path.stem, path)
        project.notes = data.get("notes", "")
        project.created = data.get("created", project.created)
        project.modified = data.get("modified", project.modified)
        for section, keys in (data.get("settings") or {}).items():
            for key, value in (keys or {}).items():
                try:
                    value = coerce(section, key, value)
                except ProjectError as error:
                    project.warnings.append(f"ignored: {error}")
                    continue
                project.settings[section][key] = project._from_disk(value, path.parent) if key in PATH_KEYS[section] else value
        project.results = project._map_results(data.get("results") or {}, lambda value: project._from_disk(value, path.parent))
        return project

    def summary(self) -> dict:
        """What a person or an agent needs at a glance: file, name, non-default settings, results."""
        changed = {f"{section}.{key}": value for section, keys in self.settings.items()
                   for key, value in keys.items() if value != SCHEMA[section][key]}
        return {"path": str(self.path) if self.path else None, "name": self.name, "notes": self.notes,
                "settings": changed, "results": self.results, "warnings": self.warnings}

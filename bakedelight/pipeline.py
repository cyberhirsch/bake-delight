"""Headless runs of a project's two jobs, shared by the CLI and the MCP server.

delight()  de-lights the lit map with the app's own engine (in this process).
scan()     runs scanopt/scanopt.py as a child process, as the GUI does.

Both return a result dict and store it in project.results; saving is the caller's job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from bakedelight import models
from bakedelight.project import Project, ProjectError

REPO = Path(__file__).resolve().parent.parent
SCANOPT_SCRIPT = REPO / "scanopt" / "scanopt.py"
# The GUI's settings file; the tool locations in it are shared with the CLI and the MCP server.
SETTINGS_PATH = Path(os.environ.get("APPDATA", os.path.expanduser("~"))) / "TextureDelighter" / "settings.json"
MACHINE_DEFAULTS = {
    "quadwild_dir": r"F:\DelighterData\tools\quadwild-bimdf",
    "autoremesher_dir": "",  # empty: scanopt's default (the copy shipped with the packaged app)
    "scanopt_python": "",
}


def machine_settings() -> dict:
    """Tool locations: the GUI's SYS tab values where set, else the defaults."""
    try:
        stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    return {key: stored.get(key) or default for key, default in MACHINE_DEFAULTS.items()}


def resolve_model(name: str = "") -> Path:
    """The bundled de-lighter model of that name (DELIGHTER_MODEL overrides)."""
    try:
        return models.resolve(name)
    except FileNotFoundError as error:
        raise ProjectError(str(error)) from None


def scanopt_python(machine: dict) -> str:
    """scanopt needs OpenCV with EXR support; prefer the training environment that has it."""
    candidate = REPO / "training" / ".venv" / "Scripts" / "python.exe"
    return machine["scanopt_python"] or os.environ.get("SCANOPT_PYTHON") or (str(candidate) if candidate.is_file() else sys.executable)


def _finished(started: float) -> dict:
    return {"seconds": round(time.time() - started, 1), "finished": datetime.now().isoformat(timespec="seconds")}


def delight(project: Project, log=print) -> dict:
    s = project.settings["delight"]
    maps = {key: s[key] for key in ("lit", "normal", "ao", "mask")}
    missing = [f"delight.{key}" for key, path in maps.items() if not path or not os.path.isfile(path)]
    if missing:
        raise ProjectError(f"missing input maps: {', '.join(missing)}")
    output = Path(s["output_dir"] or os.path.dirname(s["lit"])) / (s["output_name"] or "delighted_albedo.png")
    model = resolve_model(s["ai_model"])
    if s["force_cpu"]:
        os.environ["DELIGHTER_FORCE_CPU"] = "1"
    sys.path.insert(0, str(REPO))
    from engine import DelighterInferenceEngine  # the app's own inference path

    started = time.time()
    log(f"de-lighting {os.path.basename(s['lit'])} with {model}")
    engine = DelighterInferenceEngine(str(model))
    out_file, size, device = engine.process_texture(
        maps["lit"], maps["normal"], maps["ao"], maps["mask"], str(output),
        log_callback=None if log is print else log,  # the engine prints its progress itself

        tiled_mode=s["tiled"], tile_size=s["tile_size"], tile_overlap=s["tile_overlap"],
        bit_depth=s["bit_depth"], apply_mask=s["apply_mask"])
    result = {"albedo": out_file, "size": list(size), "device": device, "model": str(model), **_finished(started)}
    project.results["delight"] = result
    return result


def scan_command(project: Project, machine: dict | None = None) -> tuple[list[str], dict, Path]:
    """The scanopt command line for the project's scan settings, its environment and output folder."""
    machine = machine or machine_settings()
    s = project.settings["scan"]
    if not s["scan"] or not os.path.isfile(s["scan"]):
        raise ProjectError(f"scan.scan: no such file {s['scan']!r}" if s["scan"] else "scan.scan is not set")
    if s["photo"] and "<" not in s["photo"] and not os.path.isfile(s["photo"]):  # "<": a UDIM set, name.<UDIM>.png
        raise ProjectError(f"scan.photo: no such file {s['photo']!r}")
    scan = Path(s["scan"])
    out = Path(s["out"]) if s["out"] else scan.with_name(scan.stem + "_optimized")
    command = [scanopt_python(machine), "-u", str(SCANOPT_SCRIPT), str(scan), "--out", str(out),
               "--source", s["source"], "--faces", str(s["faces"]), "--remesher", s["remesher"], "--resolution", str(s["resolution"]),
               "--margin", str(s["margin"]), "--ao-samples", str(s["ao_samples"]),
               "--cage", f"{s['cage']:.4f}", "--max-ray", f"{s['max_ray']:.4f}",
               "--roughness", f"{s['roughness']:.3f}", "--roughness-variation", f"{s['roughness_variation']:.3f}",
               "--quadwild", machine["quadwild_dir"], "--model", project.settings["delight"]["ai_model"]]
    if machine["autoremesher_dir"]:
        command += ["--autoremesher", machine["autoremesher_dir"]]
    if s["photo"]:
        command += ["--texture", s["photo"]]
    for key, flag in (("sharp", "--sharp"), ("tiled", "--tiled"), ("keep_work", "--keep-work")):
        if s[key]:
            command.append(flag)
    if s["tiled"] and project.settings["delight"]["tile_size"]:
        command += ["--tile-size", str(project.settings["delight"]["tile_size"])]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    # The de-light step inside scanopt reads this, as the GUI's child does from its own process.
    if project.settings["delight"]["force_cpu"]:
        env["DELIGHTER_FORCE_CPU"] = "1"
    else:
        env.pop("DELIGHTER_FORCE_CPU", None)
    return command, env, out


def scan(project: Project, log=print) -> dict:
    command, env, out = scan_command(project)
    started, glb, tail = time.time(), None, []
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                               env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith("@progress "):
                continue
            tail = (tail + [line])[-20:]
            if line.startswith("done: "):
                glb = line[6:].split("  (")[0].strip()
            if line.startswith(("[", "backend:", "done:")) or "Error" in line:
                log(line)
        code = process.wait()
    except BaseException:
        process.kill()  # Ctrl+C or a cancelled job must not leave a remesher running
        raise
    if code != 0 or not glb:
        raise RuntimeError("scanopt failed:\n" + "\n".join(tail[-12:]))
    glb = Path(glb)
    stem = glb.with_suffix("")
    result = {"glb": str(glb), "out": str(out), "textures": str(out / "textures")}
    if stem.with_suffix(".obj").is_file():
        result["obj"] = str(stem.with_suffix(".obj"))
    report = out / f"{glb.stem}_report.json"
    if report.is_file():
        result["report"] = str(report)
        data = json.loads(report.read_text(encoding="utf-8"))
        result.update({key: data[key] for key in ("source", "source_faces", "faces", "quads", "remesher") if key in data})
    result.update(_finished(started))
    project.results["scan"] = result
    return result


def merge_results(path: Path, kind: str, result: dict) -> Project:
    """Store a result in the project on disk without overwriting settings changed meanwhile."""
    current = Project.load(path)
    current.results[kind] = result
    current.save()
    return current

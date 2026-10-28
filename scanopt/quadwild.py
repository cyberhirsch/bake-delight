"""Quad remeshing with QuadWild + Bi-MDF (Pietroni et al. 2021; Heistermann et al. 2023).

QuadWild is GPL-3.0 and is only ever run as a separate program, never linked.
Binaries: https://github.com/cgg-bern/quadwild-bimdf/releases (v0.0.2).

QuadWild has no face-count target; quad size follows `scaleFact`. We start from
an estimate and correct it with the measured count, since face count scales
with 1 / scaleFact^2.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from pathlib import Path

DEFAULT_DIR = Path(os.environ.get("QUADWILD_DIR", r"F:\DelighterData\tools\quadwild-bimdf"))


def count_faces(obj: Path) -> int:
    with obj.open("rb") as handle:
        return sum(1 for line in handle if line.startswith(b"f "))


def write_config(template: Path, destination: Path, overrides: dict) -> Path:
    """Copy a QuadWild config, replacing values in place (its parser depends on line order)."""
    lines = []
    for line in template.read_text().splitlines():
        key = line.split(maxsplit=1)[0] if line.strip() else ""
        lines.append(f"{key} {overrides[key]}" if key in overrides else line)
    destination.write_text("\n".join(lines) + "\n")
    return destination


def run(executable: Path, arguments: list[str], cwd: Path, log: Path, timeout: int, produces=None) -> None:
    """Run a QuadWild binary. `produces` returns its output path once written.

    The v0.0.2 Windows quad_from_patches.exe hits an assert (0x80000003) during
    teardown after writing a complete mesh, so a non-zero exit is accepted when
    the expected output was written by this run.
    """
    started = time.time()
    with log.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n===== {executable.name} {' '.join(arguments)}\n")
        handle.flush()
        result = subprocess.run([str(executable), *arguments], cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode != 0:
        output = produces() if produces else None
        if output is not None and output.stat().st_mtime >= started - 1:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n(exit {result.returncode:#x} after writing {output.name}; teardown crash, output kept)\n")
            return
        raise RuntimeError(f"{executable.name} exited with {result.returncode:#x}; see {log}")


def quad_output(folder: Path, stem: str) -> Path | None:
    """quad_from_patches names its result after the input; prefer the smoothed variant."""
    candidates = sorted(folder.glob(f"{stem}*quadrangulation*.obj"), key=lambda p: ("smooth" not in p.name, -p.stat().st_mtime))
    return candidates[0] if candidates else None


def remesh(high_obj: Path, target_faces: int, work: Path, tool_dir: Path = DEFAULT_DIR, sharp: bool = False,
           attempts: int = 3, timeout: int = 1800) -> tuple[Path, dict]:
    """Quad-remesh `high_obj` towards `target_faces`. Returns the quad OBJ and a report."""
    quadwild, from_patches = tool_dir / "quadwild.exe", tool_dir / "quad_from_patches.exe"
    if not quadwild.is_file() or not from_patches.is_file():
        raise SystemExit(f"QuadWild-BiMDF not found in {tool_dir}; set QUADWILD_DIR")
    work.mkdir(parents=True, exist_ok=True)
    log = work / "quadwild.log"
    prep_template = tool_dir / "config" / "prep_config" / ("basic_setup_Mechanical.txt" if sharp else "basic_setup_Organic.txt")
    flow_template = tool_dir / "config" / "main_config" / "flow_noalign.txt"

    scale, history = 1.0, []
    for attempt in range(attempts):
        # QuadWild writes next to its input, so each attempt gets its own folder.
        folder = work / f"attempt_{attempt}"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        mesh = folder / "scan.obj"
        shutil.copy(high_obj, mesh)
        prep_config = write_config(prep_template, folder / "prep.txt", {"scaleFact": f"{scale:.4f}"})
        flow_config = write_config(flow_template, folder / "flow.txt", {"scaleFact": f"{scale:.4f}"})
        # Configs reference satsuma/flow JSON files relative to the tool folder, so run from there.
        run(quadwild, [str(mesh), "2", str(prep_config)], tool_dir, log, timeout)
        patches = folder / "scan_rem_p0.obj"
        if not patches.is_file():
            raise RuntimeError(f"QuadWild produced no {patches.name}; see {log}")
        run(from_patches, [str(patches), "1", str(flow_config)], tool_dir, log, timeout,
            produces=lambda: quad_output(folder, "scan_rem_p0"))
        result = quad_output(folder, "scan_rem_p0")
        if result is None:
            raise RuntimeError(f"quad_from_patches produced no quad mesh; see {log}")
        faces = count_faces(result)
        history.append({"scaleFact": round(scale, 4), "faces": faces})
        if abs(faces - target_faces) / target_faces < 0.15:
            break
        if len(history) >= 2:
            # Fit faces ~ scale^k through the last two runs (k measured around -1.35, not -2).
            (s0, f0), (s1, f1) = [(h["scaleFact"], h["faces"]) for h in history[-2:]]
            exponent = math.log(f1 / f0) / math.log(s1 / s0) if s1 != s0 and f1 != f0 else -2.0
            scale = s1 * (target_faces / f1) ** (1.0 / min(exponent, -0.5))
        else:
            scale *= math.sqrt(faces / target_faces)  # bigger quads -> fewer faces
    return result, {"backend": "quadwild-bimdf v0.0.2 (GPL-3.0)", "attempts": history}

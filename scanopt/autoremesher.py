"""Quad remeshing with AutoRemesher (huxingyi/autoremesher, MIT).

Field-guided quad remeshing (isotropic remesh, cross field, mixed-integer
parametrisation, quad extraction). Unlike QuadWild it takes a quad-count target
directly, so no calibration loop is needed. Run as a separate program.
Binaries: https://github.com/huxingyi/autoremesher/releases (1.2.0).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# AUTOREMESHER_DIR, else the copy shipped in the packaged app's tools/ folder, else the development machine's.
DEFAULT_DIR = Path(os.environ.get("AUTOREMESHER_DIR") or (Path(sys.executable).parent / "tools" / "autoremesher"
                                                          if getattr(sys, "frozen", False)
                                                          else r"F:\DelighterData\tools\autoremesher"))


def count_faces(obj: Path) -> int:
    with obj.open("rb") as handle:
        return sum(1 for line in handle if line.startswith(b"f "))


def remesh(high_obj: Path, target_faces: int, work: Path, tool_dir: Path = DEFAULT_DIR, sharp: bool = False,
           adaptivity: float = 1.0, attempts: int = 3, timeout: int = 1800) -> tuple[Path, dict]:
    """Quad-remesh `high_obj` towards `target_faces` quads. Returns the quad OBJ and a report.

    `--target-quads` is approximate (measured: 3,556 faces for 6,000 on a rock scan),
    and a run takes seconds, so the request is rescaled by the measured ratio.
    """
    executable = tool_dir / "autoremesher.exe"
    if not executable.is_file():
        raise SystemExit(f"AutoRemesher not found in {tool_dir}; set AUTOREMESHER_DIR")
    work.mkdir(parents=True, exist_ok=True)
    output, report, log = work / "quads.obj", work / "report.txt", work / "autoremesher.log"
    log.write_text("")
    request, history = target_faces, []
    for _ in range(attempts):
        command = [str(executable), "--input", str(high_obj), "--output", str(output), "--report", str(report),
                   "--target-quads", str(request),
                   "--sharp-edge", "45.0" if sharp else "90.0",  # dihedral angle above which edges are kept sharp
                   "--adaptivity", f"{adaptivity:.2f}"]
        with log.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(" ".join(command) + "\n")
            handle.flush()
            result = subprocess.run(command, cwd=tool_dir, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError(f"AutoRemesher exited with {result.returncode:#x} and {'no' if not output.is_file() else 'an'} output; see {log}")
        faces = count_faces(output)
        history.append({"target_quads": request, "faces": faces})
        if abs(faces - target_faces) / target_faces < 0.1:
            break
        request = max(4, round(request * target_faces / faces))
    return output, {"backend": "autoremesher 1.2.0 (MIT)", "attempts": history,
                    "report": report.read_text(errors="replace")[:2000] if report.is_file() else None}

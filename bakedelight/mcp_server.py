"""Bake & Delight MCP server (stdio): lets Claude and other agents create projects and run jobs.

Register with Claude Code:
    claude mcp add bake-delight -- "<repo>\\training\\.venv\\Scripts\\python.exe" "<repo>\\bake_delight.py" mcp

Runs take from seconds (de-light) to minutes (QuadWild), so they are jobs: run_scan and
run_delight start one and return its id at once; job_status reports progress and, when
done, the result. Each job is a `bake-delight` child process, so it can be cancelled and
its output never mixes with the protocol on stdout.
"""

from __future__ import annotations

import collections
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # its message reaches the agent; other errors do not

from bakedelight.cli import RESULT_PREFIX
from bakedelight.project import CHOICES, Project, ProjectError, describe_settings

LAUNCHER = Path(__file__).resolve().parent.parent / "bake_delight.py"
MAX_WAIT = 300

server = MCPServer(
    name="bake-delight",
    title="Bake & Delight",
    instructions=(
        "Bake & Delight turns photogrammetry scans into game-ready PBR assets (remesh, UV unwrap, bake, "
        "de-light, roughness) and removes baked-in lighting from photo textures. Work happens in project "
        "files (*.bad). Typical flow: create_project with a scan and photo, update_settings, run_scan, then "
        "job_status with wait_seconds until it is done. list_settings shows every setting. Use absolute paths."),
)


class Job:
    _ids = itertools.count(1)

    def __init__(self, kind: str, project: Path):
        self.id = f"{kind}-{next(self._ids)}"
        self.kind, self.project = kind, project
        self.state, self.result, self.error = "running", None, None
        self.log = collections.deque(maxlen=200)
        self.started, self.ended = time.time(), None
        command = [sys.executable, str(LAUNCHER), kind, str(project), "--json"]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        text=True, errors="replace", env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            line = line.rstrip()
            if line.startswith(RESULT_PREFIX):
                self.result = json.loads(line[len(RESULT_PREFIX):])
            elif line:
                self.log.append(line)
        code = self.process.wait()
        self.ended = time.time()
        if self.state == "cancelled":
            return
        if code == 0 and self.result is not None:
            self.state = "done"
        else:
            self.state = "failed"
            self.error = "\n".join(list(self.log)[-12:]) or f"exit code {code}"

    def status(self, log_lines: int = 15) -> dict:
        return {"job_id": self.id, "kind": self.kind, "project": str(self.project), "state": self.state,
                "seconds": round((self.ended or time.time()) - self.started, 1),
                "log": list(self.log)[-log_lines:], "result": self.result, "error": self.error}


JOBS: dict[str, Job] = {}


def _load(path: str) -> Project:
    try:
        return Project.load(path)
    except ProjectError as error:
        raise ToolError(str(error)) from None


def _start(kind: str, project: str) -> dict:
    path = _load(project).path
    busy = [job.id for job in JOBS.values() if job.project == path and job.state == "running"]
    if busy:
        raise ToolError(f"{path.name} already has a running job: {busy[0]}")
    job = Job(kind, path)
    JOBS[job.id] = job
    return job.status()


@server.tool()
def list_settings() -> list[dict]:
    """Every project setting: key (section.key), type, default, allowed choices and description."""
    return describe_settings()


@server.tool()
def create_project(path: str, name: str = "", scan: str = "", photo: str = "", lit: str = "",
                   settings: dict[str, Any] | None = None, overwrite: bool = False) -> dict:
    """Create a Bake & Delight project file (.bad).

    scan/photo set up scan optimisation (high-res mesh + photo texture). lit sets up
    de-lighting of an existing texture; normal, ao and mask maps next to it are found by
    name. settings: optional {"scan.faces": 6000, ...}, see list_settings.
    """
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".bad":
        target = target.with_name(target.name + ".bad")
    if target.exists() and not overwrite:
        raise ToolError(f"{target} exists; pass overwrite=true to replace it")
    project = Project(name or target.stem, target)
    values = {**({"scan.scan": scan} if scan else {}), **({"scan.photo": photo} if photo else {}),
              **({"delight.lit": lit} if lit else {}), **(settings or {})}
    try:
        project.update(values)
    except ProjectError as error:
        raise ToolError(str(error)) from None
    project.autofill_maps()
    project.save()
    return project.summary()


@server.tool()
def read_project(path: str, full: bool = False) -> dict:
    """A project's name, notes, settings that differ from the defaults, and results of past runs.
    full=true returns every setting with absolute paths."""
    project = _load(path)
    return project.to_dict(relative=False) if full else project.summary()


@server.tool()
def update_settings(path: str, settings: dict[str, Any], notes: str | None = None) -> dict:
    """Change project settings, e.g. {"scan.remesher": "autoremesher", "scan.faces": 8000}.
    All values are validated first; on any error nothing is changed."""
    project = _load(path)
    try:
        project.update(settings)
    except ProjectError as error:
        raise ToolError(f"{error}. Choices: { {f'{s}.{k}': c for (s, k), c in CHOICES.items()} }") from None
    if notes is not None:
        project.notes = notes
    project.save()
    return project.summary()


@server.tool()
def run_scan(path: str) -> dict:
    """Start optimising the project's scan: remesh, unwrap, bake, de-light, export GLB/OBJ and
    PBR textures. Returns a job at once; poll job_status. Takes about 1-6 minutes."""
    return _start("scan", path)


@server.tool()
def run_delight(path: str) -> dict:
    """Start de-lighting the project's lit texture (needs lit, normal, ao and mask maps).
    Returns a job at once; poll job_status. Takes seconds to a minute."""
    return _start("delight", path)


@server.tool()
async def job_status(job_id: str, wait_seconds: float = 0, log_lines: int = 15) -> dict:
    """State of a job (running, done, failed, cancelled), its recent log and, when done, the
    result with output paths. wait_seconds (up to 300) blocks until the job ends or time runs out."""
    job = JOBS.get(job_id)
    if job is None:
        raise ToolError(f"unknown job {job_id}; known: {list(JOBS)}")
    deadline = time.time() + min(max(wait_seconds, 0), MAX_WAIT)
    while job.state == "running" and time.time() < deadline:
        await anyio.sleep(0.5)
    return job.status(log_lines)


@server.tool()
def list_jobs() -> list[dict]:
    """Every job started since the server started, newest first."""
    return [job.status(log_lines=0) for job in reversed(JOBS.values())]


@server.tool()
def cancel_job(job_id: str) -> dict:
    """Stop a running job and the tools it started (remeshers)."""
    job = JOBS.get(job_id)
    if job is None:
        raise ToolError(f"unknown job {job_id}")
    if job.state == "running":
        job.state = "cancelled"
        if sys.platform == "win32":  # /T ends the whole tree: scanopt, quadwild.exe
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(job.process.pid)], capture_output=True)
        else:
            job.process.kill()
    return job.status()


def main() -> None:
    try:
        server.run()
    finally:
        for job in JOBS.values():
            if job.state == "running":
                job.process.kill()

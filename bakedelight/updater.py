"""Updates of the packaged app from this repository's GitHub releases.

A running app cannot replace its own files, so the new version is unpacked into <app>/_update and a
small script swaps it in once the app has exited, puts the old files back if that fails, and starts
the app again. The zip is checked against the SHA-256 GitHub reports for it; it is not code-signed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import __version__

RELEASES_URL = "https://api.github.com/repos/cyberhirsch/bake-delight/releases?per_page=30"
ASSET = "BakeDelight-windows.zip"
STAGE = "_update"
APP_FILES = ("BakeDelight.exe", "BakeDelightJob.exe", "_internal")  # what makes a folder this app
TIMEOUT = 15

SWAP_SCRIPT = r"""@echo off
setlocal EnableExtensions
rem Full paths: another find.exe (Git's) or ping can come first on PATH.
set SYS=%SystemRoot%\System32
set tries=0
:wait
"%SYS%\tasklist.exe" /NH /FI "PID eq %BD_PID%" | "%SYS%\find.exe" "%BD_PID%" >nul || goto swap
set /a tries+=1
if %tries% geq 60 (
    >>"%BD_LOG%" echo FAILED: the app did not exit, nothing was changed
    goto start
)
"%SYS%\ping.exe" -n 2 127.0.0.1 >nul
goto wait

:swap
mkdir "%BD_OLD%" 2>nul
for /d %%I in ("%BD_NEW%\*") do (
    call :replace "%%~nxI" || goto rollback
)
for %%I in ("%BD_NEW%\*") do (
    call :replace "%%~nxI" || goto rollback
)
>>"%BD_LOG%" echo updated
rd /s /q "%BD_STAGE%"
goto start

:replace
if exist "%BD_TARGET%\%~1" (
    call :retire "%~1" || exit /b 1
)
for /l %%N in (1,1,10) do (
    move /y "%BD_NEW%\%~1" "%BD_TARGET%\%~1" >nul 2>&1 && exit /b 0
    "%SYS%\ping.exe" -n 2 127.0.0.1 >nul
)
exit /b 1

:retire
for /l %%N in (1,1,10) do (
    move /y "%BD_TARGET%\%~1" "%BD_OLD%\%~1" >nul 2>&1 && exit /b 0
    "%SYS%\ping.exe" -n 2 127.0.0.1 >nul
)
exit /b 1

:rollback
set restored=1
for /d %%I in ("%BD_OLD%\*") do (
    if exist "%BD_TARGET%\%%~nxI" rd /s /q "%BD_TARGET%\%%~nxI"
    move /y "%%I" "%BD_TARGET%\%%~nxI" >nul 2>&1 || set restored=0
)
for %%I in ("%BD_OLD%\*") do (
    move /y "%%I" "%BD_TARGET%\%%~nxI" >nul 2>&1 || set restored=0
)
if %restored%==1 (
    >>"%BD_LOG%" echo FAILED: files were in use, the old version was kept
    rd /s /q "%BD_STAGE%"
) else (
    >>"%BD_LOG%" echo FAILED: files were in use and some old files are still in %STAGE%\old
)

:start
start "" "%BD_TARGET%\%BD_EXE%"
(goto) 2>nul & del "%~f0"
""".replace("%STAGE%", STAGE)


@dataclass
class Release:
    version: str
    url: str
    size: int
    sha256: str
    page: str


def _version_key(tag: str) -> tuple[int, ...] | None:
    """(0, 5, 2) for v0.5.2; None for tags that are no app version, such as model-overcast or v0.3-beta."""
    if not re.fullmatch(r"v?\d+(\.\d+)*", tag):
        return None
    numbers = tuple(int(part) for part in tag.lstrip("v").split("."))
    return numbers + (0,) * (3 - len(numbers))


def _open(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": f"BakeDelight/{__version__}"}), timeout=TIMEOUT)


def check(current: str = __version__) -> Release | None:
    """The newest release with a finished zip if it is newer than `current`, else None. Scans the list
    because the 'latest' one can be a model release, or an app release whose build is still running."""
    with _open(RELEASES_URL) as response:
        releases = json.load(response)
    best, best_key = None, _version_key(current) or (0, 0, 0)
    for release in releases:
        key = _version_key(release.get("tag_name", ""))
        if key is None or key <= best_key or release.get("draft") or release.get("prerelease"):
            continue
        for asset in release.get("assets", []):
            digest = asset.get("digest") or ""
            if asset.get("name") == ASSET and asset.get("state", "uploaded") == "uploaded" and digest.startswith("sha256:"):
                best_key = key
                best = Release(release["tag_name"].lstrip("v"), asset["browser_download_url"], int(asset["size"]),
                               digest.split(":", 1)[1].lower(), release.get("html_url", ""))
    return best


def install_dir() -> Path | None:
    """The folder of the packaged app; None when running from source."""
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else None


def can_install(target: Path | None) -> bool:
    """Whether this app can replace its own files (not when in Program Files, for instance)."""
    if target is None:
        return False
    try:
        with tempfile.TemporaryFile(dir=target):
            return True
    except OSError:
        return False


def cleanup(target: Path) -> None:
    """Remove what an interrupted or failed update left behind, unless old files are still parked there."""
    if not (target / STAGE / "old").is_dir() or not any((target / STAGE / "old").iterdir()):
        shutil.rmtree(target / STAGE, ignore_errors=True)


def download(release: Release, target: Path, progress=lambda fraction: None) -> Path:
    """Download and check the zip and unpack it; returns the folder holding the new version."""
    stage = target / STAGE
    cleanup(target)
    stage.mkdir(exist_ok=True)
    needed = release.size * 3  # the zip, and the unpacked app beside it
    if shutil.disk_usage(stage).free < needed:
        raise RuntimeError(f"not enough disk space: the update needs {needed / 2 ** 30:.1f} GB")
    archive = stage / "download.zip"
    digest, received = hashlib.sha256(), 0
    with _open(release.url) as response, archive.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            progress(0.9 * received / max(release.size, 1))
    if received != release.size or digest.hexdigest() != release.sha256:
        raise RuntimeError("the download is damaged: its size or SHA-256 differs from the release")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(stage / "new")
    progress(1.0)
    archive.unlink()
    new = stage / "new" / "BakeDelight"
    if not all((new / name).exists() for name in APP_FILES):
        raise RuntimeError("the download does not hold the app")
    return new


def launch_swap(target: Path, new: Path, log: Path, pid: int | None = None) -> None:
    """Start the script that swaps `new` into `target` after process `pid` has exited; the caller exits next."""
    log.parent.mkdir(parents=True, exist_ok=True)
    handle, script = tempfile.mkstemp(prefix="bakedelight_update_", suffix=".cmd")
    with os.fdopen(handle, "w", encoding="ascii", newline="\r\n") as file:
        file.write(SWAP_SCRIPT)
    # The paths go in as variables: nothing in a folder's name can then be read as batch syntax.
    env = dict(os.environ, BD_PID=str(pid or os.getpid()), BD_TARGET=str(target), BD_NEW=str(new), BD_STAGE=str(target / STAGE),
               BD_OLD=str(target / STAGE / "old"), BD_EXE=Path(sys.executable).name, BD_LOG=str(log),
               PYINSTALLER_RESET_ENVIRONMENT="1")  # the restarted app is an independent one, not our child
    subprocess.Popen([os.environ.get("COMSPEC", "cmd"), "/c", script], env=env, cwd=tempfile.gettempdir(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)


def last_failure(log: Path) -> str | None:
    """What went wrong in the last update, once: the log is removed after reading."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        log.unlink()
    except OSError:
        return None
    return next((line for line in reversed(lines) if line.startswith("FAILED")), None)

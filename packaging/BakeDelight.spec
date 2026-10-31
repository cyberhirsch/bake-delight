# PyInstaller: a folder app, BakeDelight/ with BakeDelight.exe (the interface) and BakeDelightJob.exe
# (the scan job it starts). It runs from any folder without installing and without admin rights.
#     pyinstaller packaging/BakeDelight.spec --noconfirm      (models/*.onnx must be present)
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
paths = [str(ROOT / "scanopt"), str(ROOT / "training"), str(ROOT)]
models = sorted((ROOT / "models").glob("*.onnx"))
if not models:
    raise SystemExit("no de-lighter model in models/")

datas = [(str(model), "models") for model in models]
binaries, hidden = [], []
for package in ("customtkinter", "tkinterdnd2", "OpenGL", "pyopengltk", "embreex", "xatlas", "fast_simplification", "trimesh"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hidden += package_hidden
excludes = ["torch", "torchvision", "matplotlib", "IPython", "pytest", "mcp"]

app = Analysis([str(ROOT / "packaging" / "app.py")], pathex=paths, datas=datas, binaries=binaries,
               hiddenimports=hidden, excludes=excludes)
job = Analysis([str(ROOT / "packaging" / "job.py")], pathex=paths, hiddenimports=["autoremesher", "quadwild"],
               excludes=excludes)

app_exe = EXE(PYZ(app.pure), app.scripts, [], exclude_binaries=True, name="BakeDelight", console=False,
              uac_admin=False)
job_exe = EXE(PYZ(job.pure), job.scripts, [], exclude_binaries=True, name="BakeDelightJob", console=True,
              uac_admin=False)
COLLECT(app_exe, app.binaries, app.datas, job_exe, job.binaries, job.datas, name="BakeDelight")

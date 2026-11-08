# Bake & Delight

Photogrammetry scan or high-res PBR model in, game-ready asset out. Windows app.

Bake & Delight reduces a dense mesh, unwraps it, bakes the high-res detail onto the new mesh and, for scans, removes the lighting that was photographed into the texture. The result is a low-poly model with albedo, normal, ambient occlusion, roughness and metallic maps, as GLB and as OBJ with PNG textures.

## Download

1. Download `BakeDelight-windows.zip` from the [latest release](https://github.com/cyberhirsch/bake-delight/releases/latest).
2. Unzip it anywhere you can write to, such as your Documents folder rather than Program Files.
3. Start `BakeDelight.exe`.

No installation, no admin rights and no Python are needed; the de-lighting model is in the zip. It runs on any DirectX 12 graphics card and falls back to the CPU without one.

The app checks for a newer release at start and asks before updating itself. You can turn this off under SYS → updates.

## How it works

The tabs follow the pipeline. F5 runs the current tab's stage, F6 runs them all.

| Tab | What it does |
|---|---|
| 1 IMPORT | The high-res mesh (OBJ, PLY, GLB/glTF, STL) and its texture. Source **scan**: a photo texture to be de-lit. Source **pbr**: a model with PBR materials, baked as they are. |
| 2 REMESH | Reduce to a face count: decimate (triangles), AutoRemesher (quads, bundled) or QuadWild (quads, see below). |
| 3 UV | Unwrap and pack into one texture of 512 to 4096 px. |
| 4 BAKE | Ray-trace colour, normals and ambient occlusion from the high-res mesh onto the low one. PBR models also give roughness and metallic. |
| 5 DELIGHT | The AI model turns the baked photo colour into albedo. It can also de-light maps you made elsewhere. |
| 6 EXPORT | GLB, OBJ + MTL and the texture PNGs. |

The viewport compares before and after: 1 / 2 / 3 switch between single, side-by-side and wipe views, and M steps through the maps. The SYS tab lists every key.

Source textures can be a UDIM set (`name.<UDIM>.png`, `name.1001.png`, `name_u1_v1.png`); it is baked into the single texture of the new mesh.

### De-lighting your own maps

The model needs four maps of the same square size:

- **lit:** the colour texture with the lighting in it (sRGB)
- **normal:** object-space normals, Z up (Blender's axes), as the bake writes them
- **AO:** ambient occlusion
- **mask:** white where the texture is used

It was trained on [Poly Haven](https://polyhaven.com) scans relit under overcast skies. It removes soft shading and ambient occlusion well, and hard sun shadows less well.

### Graphics memory

DELIGHT → VRAM presets (CPU only, 4 GB, 6–8 GB, 12 GB +) set the tile size so large maps fit your card. Tiling is on by default.

## Projects, command line and agents

Settings and results are saved in `.bad` project files (1 IMPORT → project → save, or Ctrl+S).

`bake-delight` runs the same pipeline without the interface:

```bat
bake-delight new stone.bad --scan stone.obj --photo stone.jpg --set scan.faces=6000
bake-delight scan stone.bad
bake-delight delight --lit lit.png --out albedo.png
bake-delight keys
```

`bake-delight mcp` starts an MCP server on stdio, so Claude and other agents can create projects, change settings, run scans and de-light, and follow the jobs.

These two run from source (see below), not from the packaged app.

## When something goes wrong

- `BakeDelightDebug.exe` is the same app with a console showing everything it does.
- Every run writes `%APPDATA%\TextureDelighter\logs\bakedelight.log`, and keeps the previous one as `.log.1`. The DELIGHT tab has an "open log folder" button. Please attach the log to an issue.

## QuadWild

QuadWild + Bi-MDF (GPL-3.0) is not bundled. Download it from [cgg-bern/quadwild-bimdf](https://github.com/cgg-bern/quadwild-bimdf/releases) and set its folder under SYS → QUADWILD DIR. It runs as a separate program.

## Running from source

Python 3.13 on Windows:

```bat
python -m venv .venv
.venv\Scripts\pip install -r packaging/requirements.txt
gh release download model-overcast --repo cyberhirsch/bake-delight --pattern overcast.onnx --dir models
.venv\Scripts\python gui.py
```

For the command line and MCP server, also run `pip install mcp`.

`pyinstaller packaging/BakeDelight.spec --noconfirm` builds the folder app. GitHub Actions builds it for every `v*` tag and attaches the zip to that release.

`training/` holds the pipeline that made the model: fetching Poly Haven assets, baking lit variations in Blender, and training.

`run.bat`, `run.sh` and `download_weights.py` are left over from the original project and no longer work.

## Credits and licence

Bake & Delight grew out of [AI Texture De-Lighter](https://github.com/vdmrcll/ai-texture-delightier) by Vida Marcell. MIT licence, see [LICENSE](LICENSE).

The app bundles AutoRemesher (MIT) with its Qt libraries (LGPL-3.0), ONNX Runtime, OpenCV, trimesh, Embree, xatlas and others. [packaging/licenses/THIRD_PARTY_NOTICES.txt](packaging/licenses/THIRD_PARTY_NOTICES.txt) lists each one and its licence.

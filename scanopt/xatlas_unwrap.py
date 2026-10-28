"""Chart triangles with xatlas (MIT): positions + triangles in, per-corner UVs out.

    python xatlas_unwrap.py mesh.npz uvs.npz --resolution 2048 --padding 16

Runs as its own process: loaded inside Blender, the xatlas wheel crashes in
MSVCP140.dll (Blender ships an older C++ runtime than the wheel was built with).
"""

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("XATLAS_SITE", r"F:\DelighterData\tools\blender-site"))
import numpy as np  # noqa: E402
import xatlas  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("out")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--padding", type=int, default=16)
    args = parser.parse_args()

    data = np.load(args.mesh)
    atlas = xatlas.Atlas()
    atlas.add_mesh(data["positions"].astype(np.float32), data["triangles"].astype(np.uint32))
    packing = xatlas.PackOptions()
    packing.resolution = args.resolution
    packing.padding = args.padding
    packing.bilinear = True
    packing.rotate_charts = True
    atlas.generate(xatlas.ChartOptions(), packing)
    if atlas.atlas_count != 1:
        raise SystemExit(f"xatlas packed into {atlas.atlas_count} atlases")
    _, indices, uvs = atlas[0]
    # UVs come normalised per axis to a non-square atlas; restore one scale for both axes.
    # (Packing is redone in Blender, whose packer fills the square far more tightly.)
    uvs = uvs * np.array([atlas.width, atlas.height], np.float32) / max(atlas.width, atlas.height)
    np.savez(args.out, corner_uv=uvs[indices.reshape(-1)].astype(np.float32),
             charts=atlas.chart_count, size=[atlas.width, atlas.height])


if __name__ == "__main__":
    main()

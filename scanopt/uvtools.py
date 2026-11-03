"""UV helpers shared by the Blender and the native (Blender-free) stages. numpy only."""

import numpy as np


def signed_area(uv) -> float:
    (x0, y0), (x1, y1), (x2, y2) = uv
    return 0.5 * float((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))


def join_split_polygons(tri_loops, corner_uv, corner_pos, tri_poly, loop_count):
    """Per-loop UVs from per-triangle-corner UVs. Where xatlas put the triangles of one quad
    (or n-gon) into different charts, the stray triangles are carried into the chart of the
    largest one by the similarity that maps their shared edge, so every polygon stays whole."""
    loop_uv = np.zeros((loop_count, 2), np.float32)
    loop_uv[tri_loops] = corner_uv
    mismatch = np.abs(loop_uv[tri_loops] - corner_uv).max(axis=(1, 2)) > 1e-6
    split_polys = np.unique(tri_poly[mismatch])
    for poly in split_polys:
        tris = np.flatnonzero(tri_poly == poly)
        areas = [abs(signed_area(corner_uv[t])) for t in tris]
        reference = tris[int(np.argmax(areas))]
        placed = {int(l): corner_uv[reference, k] for k, l in enumerate(tri_loops[reference])}
        winding = np.sign(signed_area(corner_uv[reference]))
        pending = [t for t in tris if t != reference]
        while pending:
            for t in pending:
                shared = [k for k in range(3) if int(tri_loops[t, k]) in placed]
                if len(shared) >= 2:
                    break
            else:
                break  # disconnected fan; leave the rest as xatlas placed it
            pending.remove(t)
            # Rebuild the third corner from the 3D triangle on the already placed edge
            # (the stray chart's own UVs can be degenerate).
            k0, k1 = shared[:2]
            k2 = 3 - k0 - k1
            p0, p1, p2 = corner_pos[t, k0], corner_pos[t, k1], corner_pos[t, k2]
            edge = p1 - p0
            length2 = float(edge @ edge) or 1e-12
            along = float((p2 - p0) @ edge) / length2
            across = float(np.linalg.norm(np.cross(edge, p2 - p0))) / length2
            a, b = placed[int(tri_loops[t, k0])], placed[int(tri_loops[t, k1])]
            d = b - a
            if float(d @ d) < 1e-14:  # the placed edge has no length: nothing to rebuild on
                continue
            for side in (1.0, -1.0):
                uv = np.empty((3, 2), np.float32)
                uv[k0], uv[k1] = a, b
                uv[k2] = a + along * d + side * across * np.array([-d[1], d[0]])
                if np.sign(signed_area(uv)) == winding:
                    break
            placed.setdefault(int(tri_loops[t, k2]), uv[k2])
        for loop, uv in placed.items():
            loop_uv[loop] = uv
    return loop_uv, int(split_polys.size)

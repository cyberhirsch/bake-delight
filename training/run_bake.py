"""Bake every downloaded asset with Blender, one process per asset.

    python run_bake.py --source F:/DelighterData/source --out F:/DelighterData/baked --variations 10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
SCRIPT = Path(__file__).resolve().parent / "bake_asset.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variations", type=int, default=10)
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--hdris", type=Path, help="HDRI folder (default: <source>/hdris)")
    parser.add_argument("--sun-probability", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="bake only the first N assets (testing)")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per asset before giving up")
    parser.add_argument("--force", action="store_true", help="re-bake assets that already have meta.json")
    args = parser.parse_args()

    blends = sorted((args.source / "models").glob("*/*_2k.blend"))
    if args.limit:
        blends = blends[:args.limit]
    hdris = args.hdris or args.source / "hdris"
    done, skipped, failed = [], [], []
    for seed, blend in enumerate(blends):
        asset = blend.parent.name
        out = args.out / asset
        if not args.force and ((out / "meta.json").is_file() or (out / "skipped.txt").is_file()):
            print(f"[{seed + 1}/{len(blends)}] {asset}: already handled", flush=True)
            done.append(asset)
            continue
        start = time.time()
        try:
            result = subprocess.run(
                [args.blender, "-b", "--disable-autoexec", str(blend), "-P", str(SCRIPT), "--",
                 "--out", str(out), "--hdris", str(hdris), "--sun-probability", str(args.sun_probability),
                 "--variations", str(args.variations), "--seed", str(seed)],
                capture_output=True, text=True, errors="replace", timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            failed.append(asset)
            print(f"[{seed + 1}/{len(blends)}] {asset}: TIMEOUT after {args.timeout}s", flush=True)
            continue
        log = result.stdout + result.stderr
        skip = next((line for line in log.splitlines() if line.startswith("SKIP")), None)
        if skip:
            skipped.append(asset)
            out.mkdir(parents=True, exist_ok=True)
            (out / "skipped.txt").write_text(skip)
            print(f"[{seed + 1}/{len(blends)}] {asset}: {skip}", flush=True)
        elif result.returncode == 0 and (out / "meta.json").is_file():
            done.append(asset)
            print(f"[{seed + 1}/{len(blends)}] {asset}: ok in {time.time() - start:.0f}s", flush=True)
        else:
            failed.append(asset)
            out.mkdir(parents=True, exist_ok=True)
            (out / "blender.log").write_text(log)
            print(f"[{seed + 1}/{len(blends)}] {asset}: FAILED (exit {result.returncode}), see {out / 'blender.log'}", flush=True)

    print(f"baked {len(done)}, skipped {len(skipped)} {skipped}, failed {len(failed)} {failed}")
    return 0 if done else 1  # individual failures are logged; the pipeline continues with what baked


if __name__ == "__main__":
    sys.exit(main())

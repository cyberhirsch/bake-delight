"""Download CC0 models and HDRIs from Poly Haven for the de-lighting dataset.

Only models with exactly one 2K diffuse texture set are used, so every asset
bakes into a single UV atlas. Files are verified against Poly Haven's MD5s.

    python fetch_polyhaven.py --out F:/DelighterData/source --models 20 --hdris 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

API = "https://api.polyhaven.com"
USER_AGENT = "DelighterDataset/0.1 (research; CC0 assets)"

# Photoscan-heavy categories first; props add material variety.
MODEL_CATEGORIES = {"rocks": 8, "nature": 5, "ground cover": 3, "props": 4}
SKIP_CATEGORIES = {"plants", "trees", "rigged"}  # alpha cutouts / skinned meshes
HDRI_CATEGORIES = {"high contrast": 4, "overcast": 2, "partly cloudy": 2, "sunrise-sunset": 1, "indoor": 1}


def get_json(url: str):
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as response:
        return json.load(response)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path, expected_md5: str | None) -> None:
    if destination.is_file() and (expected_md5 is None or md5(destination) == expected_md5):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120) as response:
        with partial.open("wb") as handle:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                handle.write(chunk)
    if expected_md5 and md5(partial) != expected_md5:
        partial.unlink()
        raise RuntimeError(f"MD5 mismatch for {url}")
    partial.replace(destination)


def single_texture_set(files: dict) -> dict | None:
    """Return the 2K .blend entry if the model has exactly one diffuse map."""
    blend = files.get("blend", {}).get("2k", {}).get("blend")
    if not blend:
        return None
    diffuse = [name for name in blend.get("include", {}) if "_diff_" in name]
    return blend if len(diffuse) == 1 else None


def all_models() -> list[tuple[str, dict]]:
    """Every model with a single texture set, minus alpha-cutout and rigged categories."""
    catalog = get_json(f"{API}/assets?t=models")
    chosen = []
    for asset_id, info in sorted(catalog.items()):
        if SKIP_CATEGORIES.intersection(info.get("categories", [])):
            continue
        blend = single_texture_set(get_json(f"{API}/files/{asset_id}"))
        if blend:
            chosen.append((asset_id, blend))
        time.sleep(0.1)
    return chosen


def pick_models(count: int, rng: random.Random) -> list[tuple[str, dict]]:
    if count <= 0:
        return all_models()
    catalog = get_json(f"{API}/assets?t=models")
    quota = {category: round(share * count / sum(MODEL_CATEGORIES.values())) for category, share in MODEL_CATEGORIES.items()}
    chosen: list[tuple[str, dict]] = []
    for category, wanted in quota.items():
        candidates = [
            asset_id for asset_id, info in catalog.items()
            if category in info.get("categories", [])
            and not SKIP_CATEGORIES.intersection(info.get("categories", []))
            and asset_id not in {c[0] for c in chosen}
        ]
        rng.shuffle(candidates)
        taken = 0
        for asset_id in candidates:
            if taken >= wanted:
                break
            blend = single_texture_set(get_json(f"{API}/files/{asset_id}"))
            if blend:
                chosen.append((asset_id, blend))
                taken += 1
            time.sleep(0.1)
    return chosen[:count]


def pick_hdris(count: int, rng: random.Random, category: str | None = None) -> list[str]:
    catalog = get_json(f"{API}/assets?t=hdris")
    if category:  # single lighting regime, outdoor daylight only
        excluded = {"indoor", "night", "studio", "artificial light"}
        candidates = sorted(a for a, info in catalog.items()
                            if category in info.get("categories", []) and not excluded.intersection(info.get("categories", [])))
        rng.shuffle(candidates)
        return candidates[:count]
    quota = {category: round(share * count / sum(HDRI_CATEGORIES.values())) for category, share in HDRI_CATEGORIES.items()}
    chosen: list[str] = []
    for category, wanted in quota.items():
        candidates = sorted(a for a, info in catalog.items() if category in info.get("categories", []) and a not in chosen)
        rng.shuffle(candidates)
        chosen.extend(candidates[:wanted])
    return chosen[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=int, default=20, help="0 = every eligible model, -1 = none (HDRIs only)")
    parser.add_argument("--hdris", type=int, default=10)
    parser.add_argument("--hdri-category", help="e.g. 'overcast': only outdoor HDRIs of that category, into hdris_<category>")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    models_dir = args.out / "models"
    hdris_dir = args.out / (f"hdris_{args.hdri_category.replace(' ', '_')}" if args.hdri_category else "hdris")
    manifest = {"source": "polyhaven.com (CC0)", "models": [], "hdris": []}

    for asset_id, blend in (pick_models(args.models, rng) if args.models >= 0 else []):
        print(f"model {asset_id}", flush=True)
        asset_dir = models_dir / asset_id
        try:
            fetch(blend["url"], asset_dir / Path(blend["url"]).name, blend.get("md5"))
            for relative, info in blend.get("include", {}).items():
                fetch(info["url"], asset_dir / relative, info.get("md5"))
        except Exception as error:  # one broken asset must not stop a long download
            print(f"  failed: {error}", flush=True)
            continue
        manifest["models"].append(asset_id)

    for hdri_id in pick_hdris(args.hdris, rng, args.hdri_category):
        print(f"hdri  {hdri_id}", flush=True)
        info = get_json(f"{API}/files/{hdri_id}")["hdri"]["2k"]["hdr"]
        fetch(info["url"], hdris_dir / Path(info["url"]).name, info.get("md5"))
        manifest["hdris"].append(hdri_id)

    (hdris_dir.parent / f"manifest_{hdris_dir.name}.json" if args.models < 0 else args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest['models'])} models, {len(manifest['hdris'])} HDRIs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

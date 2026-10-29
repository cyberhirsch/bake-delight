"""Overnight pipeline: bake -> post-process -> train -> export, for one lighting profile.

    python run_pipeline.py --profile overcast

Every stage resumes where it stopped, so re-running after a crash or reboot is safe.
Windows is kept awake while the pipeline runs (no system setting is changed).
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
ROOT = Path(os.environ.get("DELIGHTER_DATA", r"F:\DelighterData"))

PROFILES = {
    # Soft sky light only: AO-like shading, no hard cast shadows.
    "overcast": {"hdris": "hdris_overcast", "sun_probability": 0.0, "variations": 12, "steps": 80000},
    # Direct sun on top of clear-sky HDRIs. Needs: fetch_polyhaven.py --models -1 --hdri-category clear
    "sunny": {"hdris": "hdris_clear", "sun_probability": 1.0, "variations": 16, "steps": 80000},
}


def keep_awake(enable: bool) -> None:
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0))


def stage(name: str, command: list[str], log_dir: Path) -> None:
    log_path = log_dir / f"{name}.log"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {name}: {' '.join(command)}  (log: {log_path})", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {stamp} {' '.join(command)}\n")
        log.flush()
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=HERE).returncode
    if code != 0:
        raise SystemExit(f"{name} failed with exit code {code}; see {log_path}")


def train_options(args) -> list[str]:
    options = []
    for flag, value in (("--batch", args.batch), ("--crop", args.crop), ("--lr", args.lr), ("--eval-every", args.eval_every)):
        if value is not None:
            options += [flag, str(value)]
    return options


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N assets only (testing)")
    parser.add_argument("--steps", type=int, help="override training steps")
    parser.add_argument("--variations", type=int, help="override lightings per asset")
    parser.add_argument("--tag", default="", help="suffix for output folders (testing)")
    parser.add_argument("--stop-after", choices=["bake", "postprocess", "train"], default="train")
    # passed through to train.py when given (the trainer UI sets these)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--crop", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--eval-every", type=int)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    name = args.profile + args.tag
    baked, dataset, run = ROOT / f"baked_{name}", ROOT / f"dataset_{name}", ROOT / "runs" / name
    logs = ROOT / "logs" / name
    logs.mkdir(parents=True, exist_ok=True)
    hdris = ROOT / "source" / profile["hdris"]
    if not any(hdris.glob("*.hdr")):
        raise SystemExit(f"No HDRIs in {hdris}")

    keep_awake(True)
    try:
        stage("1_bake", [PYTHON, "run_bake.py", "--source", str(ROOT / "source"), "--out", str(baked),
                         "--hdris", str(hdris), "--sun-probability", str(profile["sun_probability"]),
                         "--variations", str(args.variations or profile["variations"]), "--limit", str(args.limit)], logs)
        if args.stop_after == "bake":
            print(f"bake finished: {baked}", flush=True)
            return 0
        stage("2_postprocess", [PYTHON, "postprocess.py", "--baked", str(baked), "--source", str(ROOT / "source" / "models"),
                                "--out", str(dataset), "--workers", "8"], logs)
        if args.stop_after == "postprocess":
            print(f"dataset ready: {dataset}", flush=True)
            return 0
        stage("3_train", [PYTHON, "train.py", "--data", str(dataset), "--run", str(run),
                          "--steps", str(args.steps or profile["steps"]), "--resume", *train_options(args)], logs)
    finally:
        keep_awake(False)
    print(f"done: model at {run / 'delighter.onnx'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train a dual-encoder U-Net de-lighter on the baked dataset and export ONNX.

    python train.py --data F:/DelighterData/dataset --run F:/DelighterData/runs/pilot --steps 20000

The whole dataset is held on the GPU as float16, so data loading needs no
worker processes. The exported model matches the delighter app's interface:
    inputs  lit  [1,3,H,W] linear RGB, geometry [1,5,H,W] (object normal, AO, mask)
    output  albedo [1,3,H,W] linear RGB
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- colour

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-7, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-7, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, cin: int, widths: list[int]):
        super().__init__()
        self.stem = Block(cin, widths[0])
        self.down = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(widths[:-1], widths[1:]))
        self.blocks = nn.ModuleList(Block(w, w) for w in widths[1:])

    def forward(self, x):
        features = [self.stem(x)]
        for down, block in zip(self.down, self.blocks):
            features.append(block(down(features[-1])))
        return features


class DualUNet(nn.Module):
    """One encoder for the lit texture, one for geometry, fused at every scale."""

    def __init__(self, widths=(32, 64, 128, 256, 384)):
        super().__init__()
        widths = list(widths)
        self.lit_encoder = Encoder(3, widths)
        self.geo_encoder = Encoder(5, widths)
        self.fuse = nn.ModuleList(nn.Conv2d(2 * w, w, 1) for w in widths)
        self.bottleneck = Block(widths[-1], widths[-1])
        # Transposed convolutions instead of Resize: the original model's Resize
        # overflowed ONNX Runtime's CUDA index math at 8K.
        self.up = nn.ModuleList(nn.ConvTranspose2d(b, a, 2, stride=2) for a, b in zip(widths[:-1], widths[1:]))
        self.decode = nn.ModuleList(Block(2 * a, a) for a in widths[:-1])
        self.head = nn.Conv2d(widths[0], 3, 1)
        self.levels = len(widths) - 1

    def forward(self, lit_linear: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        lit = linear_to_srgb(lit_linear)  # perceptual encoding conditions the network better
        skips = [f(torch.cat([a, b], 1)) for f, a, b in zip(self.fuse, self.lit_encoder(lit), self.geo_encoder(geometry))]
        x = self.bottleneck(skips[-1])
        for level in reversed(range(self.levels)):
            x = self.decode[level](torch.cat([self.up[level](x), skips[level]], 1))
        albedo_srgb = torch.sigmoid(self.head(x))
        return srgb_to_linear(albedo_srgb)


# ---------------------------------------------------------------- data

def read_png(path: Path) -> np.ndarray:
    """Raw HWC uint8/uint16 RGB; conversion to float happens on the GPU."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Cannot read {path}")
    if image.ndim == 2:
        return image[:, :, None]
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def to_gpu(image: np.ndarray, device, channels: int | None = None) -> torch.Tensor:
    """HWC uint8/uint16 -> CHW float16 in 0..1 on the device."""
    image = image[:, :, :channels] if channels else image
    if image.dtype == np.uint16:  # torch has no uint16 transfers; reinterpret and unwrap on device
        tensor = torch.from_numpy(np.ascontiguousarray(image).view(np.int16)).to(device, non_blocking=True).int() & 0xFFFF
        scale = 65535.0
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device, non_blocking=True).int()
        scale = 255.0
    return (tensor.float() / scale).permute(2, 0, 1).half()


GEOMETRY_FILES = ("albedo.png", "normal_os.png", "ao.png", "mask.png")


def load_asset_maps(folder: Path, lit_paths: list[Path], pool=None) -> list[np.ndarray]:
    paths = [folder / name for name in GEOMETRY_FILES] + list(lit_paths)
    return list(pool.map(read_png, paths)) if pool else [read_png(p) for p in paths]


def asset_entry(name: str, maps: list[np.ndarray], device) -> dict:
    return {
        "name": name,
        "albedo": to_gpu(maps[0], device),  # sRGB
        "geometry": torch.cat([to_gpu(maps[1], device), to_gpu(maps[2], device, 1), to_gpu(maps[3], device, 1)], 0),
        "lit": [to_gpu(m, device) for m in maps[4:]],  # sRGB
    }


class GpuDataset:
    """All assets resident on the GPU as float16; crops are sliced on device."""

    def __init__(self, root: Path, assets: list[str], device: torch.device, max_lit: int | None = None):
        self.assets = []
        with ThreadPoolExecutor(8) as pool:
            for name in assets:
                lit_paths = sorted((root / name).glob("lit_*.png"))[:max_lit]
                self.assets.append(asset_entry(name, load_asset_maps(root / name, lit_paths, pool), device))
        self.size = self.assets[0]["albedo"].shape[-1]

    def __len__(self):
        return sum(len(a["lit"]) for a in self.assets)

    def crop(self, asset: dict, lit_index: int, y: int, x: int, size: int):
        window = (slice(None), slice(y, y + size), slice(x, x + size))
        return asset["lit"][lit_index][window], asset["geometry"][window], asset["albedo"][window]

    def sample(self, batch: int, size: int, rng: random.Random):
        lits, geos, targets = [], [], []
        while len(lits) < batch:
            asset = rng.choice(self.assets)
            y, x = rng.randrange(0, self.size - size + 1), rng.randrange(0, self.size - size + 1)
            lit, geo, albedo = self.crop(asset, rng.randrange(len(asset["lit"])), y, x, size)
            if geo[4].float().mean() < 0.25:  # mostly empty UV space teaches nothing
                continue
            lits.append(lit); geos.append(geo); targets.append(albedo)
        return torch.stack(lits).float(), torch.stack(geos).float(), torch.stack(targets).float()


class StreamingPool(GpuDataset):
    """A rotating pool of decoded pairs on the GPU, refilled from disk by background threads.

    For datasets too large for GPU memory: each slot holds one lit/geometry/albedo
    pair, and every step a few slots are swapped for freshly decoded random pairs.
    """

    def __init__(self, root: Path, assets: list[str], device: torch.device, slots: int = 64, threads: int = 6):
        self.items = [(root / name, lit) for name in assets for lit in sorted((root / name).glob("lit_*.png"))]
        self.device = device
        self.queue: queue.Queue = queue.Queue(maxsize=threads * 2)
        for seed in range(threads):
            threading.Thread(target=self._loader, args=(random.Random(seed),), daemon=True).start()
        self.assets = [self._next() for _ in range(slots)]
        self.size = self.assets[0]["albedo"].shape[-1]

    def __len__(self):
        return len(self.items)

    def _loader(self, rng: random.Random):
        while True:
            folder, lit = rng.choice(self.items)
            try:
                self.queue.put((folder.name, load_asset_maps(folder, [lit])))
            except Exception as error:  # a single unreadable file must not kill the run
                print(f"loader skipped {lit}: {error}", flush=True)

    def _next(self) -> dict:
        name, maps = self.queue.get()
        return asset_entry(name, maps, self.device)

    def refresh(self, rng: random.Random, limit: int = 1):
        """Swap up to `limit` slots for pairs that are already decoded; never blocks."""
        for _ in range(limit):
            if self.queue.empty():
                return
            self.assets[rng.randrange(len(self.assets))] = self._next()


def random_rotation(batch: int, device) -> torch.Tensor:
    """Random orthogonal 3x3 matrices (rotations and mirrors)."""
    q, r = torch.linalg.qr(torch.randn(batch, 3, 3, device=device))
    return q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)


def augment(lit_srgb, geometry, albedo_srgb):
    """Geometry-consistent augmentations. Returns linear lit, geometry, linear albedo."""
    b = lit_srgb.shape[0]
    # Spatial: flips and 90 degree turns keep per-texel object-space normals valid.
    k = random.randrange(4)
    lit_srgb, geometry, albedo_srgb = (torch.rot90(t, k, (2, 3)) for t in (lit_srgb, geometry, albedo_srgb))
    if random.random() < 0.5:
        lit_srgb, geometry, albedo_srgb = (t.flip(3) for t in (lit_srgb, geometry, albedo_srgb))

    # Rotating the object frame with its lighting leaves the lit texture unchanged, so any
    # orthogonal transform of the normals is valid. It also makes the model indifferent
    # to Y-up/Z-up and DirectX/OpenGL object-space conventions.
    normals = geometry[:, :3] * 2 - 1
    rotated = torch.einsum("bij,bjhw->bihw", random_rotation(b, normals.device), normals)
    geometry = torch.cat([rotated * 0.5 + 0.5, geometry[:, 3:]], 1)

    lit = srgb_to_linear(lit_srgb)
    # Exposure and white-balance jitter the network must undo.
    gain = torch.empty(b, 1, 1, 1, device=lit.device).uniform_(0.8, 1.2)
    tint = torch.empty(b, 3, 1, 1, device=lit.device).uniform_(0.95, 1.05)
    lit = (lit * gain * tint).clamp(0, 1)
    # Half the batch gets 8-bit quantisation, like JPG/PNG scan textures.
    quantise = (torch.rand(b, 1, 1, 1, device=lit.device) < 0.5).float()
    lit8 = srgb_to_linear(torch.round(linear_to_srgb(lit) * 255) / 255)
    lit = quantise * lit8 + (1 - quantise) * lit
    return lit, geometry, srgb_to_linear(albedo_srgb)


def masked_l1(a, b, mask):
    return ((a - b).abs() * mask).sum() / (mask.sum() * a.shape[1] + 1e-6)


def gradient_loss(a, b, mask):
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    mx, my = mask[..., :, 1:] * mask[..., :, :-1], mask[..., 1:, :] * mask[..., :-1, :]
    return masked_l1(dx(a), dx(b), mx) + masked_l1(dy(a), dy(b), my)


def loss_fn(prediction_linear, target_linear, mask):
    p, t = linear_to_srgb(prediction_linear), linear_to_srgb(target_linear)
    return masked_l1(p, t, mask) + 0.5 * masked_l1(prediction_linear, target_linear, mask) + 0.5 * gradient_loss(p, t, mask)


# ---------------------------------------------------------------- evaluation / export

@torch.no_grad()
def evaluate(model, dataset: GpuDataset, size: int = 1024):
    """Masked sRGB L1 on the central crop of every validation pair, plus the do-nothing baseline."""
    model.eval()
    total, baseline, count = 0.0, 0.0, 0
    offset = (dataset.size - size) // 2
    preview = None
    for asset in dataset.assets:
        for index in range(len(asset["lit"])):
            lit, geo, albedo = (t[None].float() for t in dataset.crop(asset, index, offset, offset, size))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(srgb_to_linear(lit), geo).float()
            mask = geo[:, 4:5]
            total += masked_l1(linear_to_srgb(prediction), albedo, mask).item()
            baseline += masked_l1(lit, albedo, mask).item()
            count += 1
            if preview is None:
                preview = torch.cat([lit[0], linear_to_srgb(prediction)[0], albedo[0]], 2)
    model.train()
    return total / count, baseline / count, preview


def save_preview(path: Path, tensor: torch.Tensor):
    image = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def export_onnx(model, path: Path):
    model = model.float().eval().cpu()
    lit, geo = torch.rand(1, 3, 512, 512), torch.rand(1, 5, 512, 512)
    torch.onnx.export(
        model, (lit, geo), str(path), input_names=["lit", "geometry"], output_names=["albedo"],
        dynamic_axes={"lit": {2: "h", 3: "w"}, "geometry": {2: "h", 3: "w"}, "albedo": {2: "h", 3: "w"}},
        opset_version=17, dynamo=False,
    )


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-assets", default="", help="comma list; default is a seeded 5%% split")
    parser.add_argument("--val-lit", type=int, default=3, help="lightings per validation asset")
    parser.add_argument("--stream", action="store_true", help="force disk streaming (automatic above ~400 pairs)")
    parser.add_argument("--pool", type=int, default=64, help="decoded pairs held on the GPU when streaming")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoint.pt in --run")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    args.run.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in args.data.iterdir() if (p / "albedo.png").is_file())
    if args.val_assets:
        val_names = [n for n in args.val_assets.split(",") if n in names]
    else:  # reproducible held-out split, ~5% of assets
        val_count = min(max(2, round(len(names) * 0.05)), len(names) // 2)
        val_names = sorted(random.Random(args.seed).sample(names, max(1, val_count)))
    train_names = [n for n in names if n not in val_names]
    pair_count = sum(len(list((args.data / n).glob("lit_*.png"))) for n in train_names)
    stream = args.stream or pair_count > 400  # ~400 pairs is where GPU residency stops fitting in 24 GB
    started = time.time()
    train_set = StreamingPool(args.data, train_names, device, slots=args.pool) if stream else GpuDataset(args.data, train_names, device)
    val_set = GpuDataset(args.data, val_names, device, max_lit=args.val_lit)
    print(f"{'streaming' if stream else 'loaded'} {len(train_set)} train / {len(val_set)} val pairs in {time.time() - started:.0f}s, "
          f"GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)

    model = DualUNet().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {parameters / 1e6:.1f}M", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    first_step, best = 1, float("inf")
    checkpoint_path = args.run / "checkpoint.pt"
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        schedule.load_state_dict(checkpoint["schedule"])
        first_step, best = checkpoint["step"] + 1, checkpoint["best"]
        print(f"resumed at step {first_step}, best val {best:.4f}", flush=True)
    else:
        (args.run / "config.json").write_text(json.dumps({**vars(args), "data": str(args.data), "run": str(args.run),
                                                          "train": train_names, "val": val_names, "parameters": parameters,
                                                          "streaming": stream}, indent=2))
    log_file = (args.run / "log.csv").open("a" if first_step > 1 else "w", newline="")
    log = csv.writer(log_file)
    if first_step == 1:
        log.writerow(["step", "train_loss", "val_l1", "baseline_l1", "lr", "seconds"])
    running, train_loss, started = 0.0, float("nan"), time.time()

    for step in range(first_step, args.steps + 1):
        if stream:
            train_set.refresh(rng)
        lit, geo, albedo = augment(*train_set.sample(args.batch, args.crop, rng))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(lit, geo)
        loss = loss_fn(prediction.float(), albedo, geo[:, 4:5])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); schedule.step()
        running += loss.item()

        if step % 100 == 0:
            rate = (step - first_step + 1) / (time.time() - started)
            print(f"step {step}/{args.steps}  loss {running / 100:.4f}  {rate:.1f} it/s  "
                  f"eta {(args.steps - step) / rate / 60:.0f} min", flush=True)
            train_loss, running = running / 100, 0.0
        if step % args.eval_every == 0 or step == args.steps:
            val, baseline, preview = evaluate(model, val_set)
            print(f"  val L1 {val:.4f} (do-nothing baseline {baseline:.4f})", flush=True)
            log.writerow([step, f"{train_loss:.5f}", f"{val:.5f}", f"{baseline:.5f}", f"{schedule.get_last_lr()[0]:.2e}", f"{time.time() - started:.0f}"])
            log_file.flush()
            save_preview(args.run / f"val_{step:06d}.jpg", preview)
            if val < best:
                best = val
                torch.save(model.state_dict(), args.run / "best.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                        "step": step, "best": best}, checkpoint_path)

    model.load_state_dict(torch.load(args.run / "best.pt"))
    export_onnx(model, args.run / "delighter.onnx")
    print(f"done; best val L1 {best:.4f}; exported {args.run / 'delighter.onnx'}", flush=True)


if __name__ == "__main__":
    main()

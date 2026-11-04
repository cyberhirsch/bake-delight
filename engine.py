import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import cv2
import numpy as np


# ONNX Runtime 1.19 does not automatically preload CUDA libraries from
# NVIDIA's Python packages. Configure both Windows DLL search and Linux
# shared-library search before importing ONNX Runtime.
_CUDA_DLL_HANDLES = []
if os.name == "nt":
    _site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for _relative_path in (
        "nvidia/cuda_runtime/bin",
        "nvidia/cuda_nvrtc/bin",
        "nvidia/cublas/bin",
        "nvidia/cufft/bin",
        "nvidia/cudnn/bin",
    ):
        _dll_dir = _site_packages / _relative_path
        if _dll_dir.is_dir():
            os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ.get("PATH", "")
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(_dll_dir)))
elif sys.platform.startswith("linux"):
    _site_packages = [Path(sysconfig.get_paths().get("purelib", ""))]
    try:
        _site_packages.extend(Path(path) for path in sysconfig.get_paths().get("platlib", "").split(os.pathsep) if path)
    except AttributeError:
        pass

    _cuda_roots = []
    for _cuda_env in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(_cuda_env):
            _cuda_roots.append(Path(os.environ[_cuda_env]))
    _cuda_roots.extend((Path("/usr/local/cuda"), Path("/usr/local/cuda-11.8")))
    _cuda_library_dirs = []
    for _root in _cuda_roots:
        _cuda_library_dirs.extend((_root / "lib64", _root / "lib"))
    for _site_package in _site_packages:
        for _relative_path in (
            "nvidia/cuda_runtime/lib",
            "nvidia/cuda_nvrtc/lib",
            "nvidia/cublas/lib",
            "nvidia/cufft/lib",
            "nvidia/cudnn/lib",
        ):
            _cuda_library_dirs.append(_site_package / _relative_path)
    _existing_cuda_dirs = [str(path) for path in _cuda_library_dirs if path.is_dir()]
    if _existing_cuda_dirs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            _existing_cuda_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]
        ).strip(os.pathsep)

import onnxruntime as ort


def _say(message: str) -> None:
    print(f"[Engine Init] {message}", flush=True)


def _gpu_names() -> list[str]:
    """The graphics adapters Windows knows, with driver versions: tells a missing driver from a missing GPU."""
    if os.name != "nt":
        return []
    command = ["powershell.exe", "-NoProfile", "-Command",
               "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' | driver ' + $_.DriverVersion }"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError) as error:
        return [f"could not ask Windows: {error}"]


def _free_ram_gb() -> float | None:
    if os.name != "nt":
        return None
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong), ("unused", ctypes.c_ulonglong * 5)]

    status = Status(length=ctypes.sizeof(Status))
    return status.available / 2 ** 30 if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None


def _system_report() -> None:
    _say(f"app: {'packaged' if getattr(sys, 'frozen', False) else 'source'}, {sys.executable}")
    _say(f"system: {platform.platform()}, Python {platform.python_version()}, working folder {os.getcwd()}")
    ram = _free_ram_gb()
    if ram is not None:
        _say(f"free RAM: {ram:.1f} GB")
    for name in _gpu_names() or ["none reported"]:
        _say(f"graphics adapter: {name}")
    _say(f"ONNX Runtime {ort.__version__} from {Path(ort.__file__).parent}")
    _say(f"providers available: {ort.get_available_providers()}")
    for key in ("DELIGHTER_FORCE_CPU", "DELIGHTER_MODEL"):
        if os.environ.get(key):
            _say(f"{key}={os.environ[key]}")


def _map_report(name: str, path: str, tensor: np.ndarray) -> str:
    size = os.path.getsize(path) / 2 ** 20
    return (f"{name}: {path} ({size:.1f} MB, {tensor.shape[2]}x{tensor.shape[1]}, {tensor.shape[0]} channels, "
            f"min {tensor.min():.3f} max {tensor.max():.3f} mean {tensor.mean():.3f})")


def _hardware_name_for_provider(provider: str) -> str:
    """Return a useful hardware name without making the app platform-specific."""
    if provider == "CPUExecutionProvider":
        return _cpu_model_name()

    if provider == "CUDAExecutionProvider":
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                result = subprocess.run(
                    [nvidia_smi, "--query-gpu=name", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                gpu_name = result.stdout.strip().splitlines()[0]
                if gpu_name:
                    return gpu_name
            except (OSError, IndexError, subprocess.SubprocessError):
                pass
        return "NVIDIA GPU"

    if provider == "DmlExecutionProvider":
        return "GPU (DirectX 12)"

    return provider.replace("ExecutionProvider", "") + " device"


def _cpu_model_name() -> str:
    """Get a friendly CPU model name across the platforms supported by Python."""
    commands = []
    if platform.system() == "Windows":
        commands = [
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-ItemProperty 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0' -Name ProcessorNameString).ProcessorNameString",
            ],
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ],
            ["wmic", "cpu", "get", "name", "/value"],
        ]
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False)
            for line in result.stdout.splitlines():
                name = line.strip()
                if name and not name.lower().startswith("name="):
                    return name
        except (OSError, subprocess.SubprocessError):
            pass

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.lower().startswith(("model name", "hardware")) and ":" in line:
                        name = line.split(":", 1)[1].strip()
                        if name:
                            return name
        except OSError:
            pass

    return platform.processor() or "CPU"

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-7, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-7, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

class DelighterInferenceEngine:
    TILE_SIZE = 2048

    def __init__(self, model_path: str = "weights/delighter_model_fp16.onnx"):
        started = time.time()
        print("\n" + "="*50, flush=True)
        _say("Initializing ONNX Runtime engine...")
        _system_report()
        
        available_providers = ort.get_available_providers()
        force_cpu = os.environ.get("DELIGHTER_FORCE_CPU", "").lower() in {"1", "true", "yes"}
        if force_cpu:
            providers = ["CPUExecutionProvider"]
            _say("CPU fallback forced by DELIGHTER_FORCE_CPU")
        elif "CUDAExecutionProvider" in available_providers:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                ),
                "CPUExecutionProvider",
            ]
        elif "DmlExecutionProvider" in available_providers:  # onnxruntime-directml: any DX12 GPU, no CUDA install
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        if providers[0] == "DmlExecutionProvider":  # DirectML needs these off
            session_options.enable_mem_pattern = False
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        resolved_model_path = Path(model_path)
        if not resolved_model_path.is_absolute():
            if getattr(sys, "frozen", False):
                app_dir = Path(sys.executable).resolve().parent
            else:
                app_dir = Path(__file__).resolve().parent
            resolved_model_path = app_dir / resolved_model_path
            if not resolved_model_path.is_file():
                resolved_model_path = app_dir / "weights" / Path(model_path).name
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

        _say(f"model: {resolved_model_path} ({resolved_model_path.stat().st_size / 2 ** 20:.1f} MB)")
        requested = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        _say(f"loading the model with {requested}...")
        try:
            self.model = ort.InferenceSession(str(resolved_model_path), sess_options=session_options, providers=providers)
        except Exception as provider_error:
            if providers and providers[0] != "CPUExecutionProvider" and not force_cpu:
                failed_provider = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
                _say(f"WARNING: {failed_provider} initialization failed; falling back to CPU: {provider_error}")
                providers = ["CPUExecutionProvider"]
                self.model = ort.InferenceSession(str(resolved_model_path), providers=providers)
            else:
                raise
        active_providers = self.model.get_providers()
        self.execution_provider = next(
            (provider for provider in active_providers if provider != "CPUExecutionProvider"),
            "CPUExecutionProvider",
        )
        self.hardware_device = _hardware_name_for_provider(self.execution_provider)
        self.runtime_name = "ONNX Runtime"
        self.provider_name = {
            "CUDAExecutionProvider": "CUDA",
            "DmlExecutionProvider": "DirectML",
        }.get(self.execution_provider, "CPU")
        self.is_accelerated = self.execution_provider != "CPUExecutionProvider"
        self.device_info = f"{self.hardware_device} via {self.runtime_name} / {self.provider_name}"
        self.device_display = f"{self.runtime_name} · {self.provider_name}\n{self.hardware_device}"
        _say(f"model loaded in {time.time() - started:.1f} s; session providers: {active_providers}")
        _say(f"Active Hardware Device: {self.device_info}")
        if requested != "CPUExecutionProvider" and not self.is_accelerated:
            _say(f"WARNING: {requested} was asked for but the model runs on the CPU, which is far slower")
        self.input_names = [item.name for item in self.model.get_inputs()]
        self.output_name = self.model.get_outputs()[0].name
        if len(self.input_names) != 2:
            raise ValueError(f"Expected 2 ONNX inputs, found {len(self.input_names)}: {self.input_names}")
        for item in self.model.get_inputs() + self.model.get_outputs():
            _say(f"model tensor {item.name}: {item.type} {item.shape}")
        self._self_test()
        _say(f"engine ready, {time.time() - started:.1f} s in all")
        print("="*50 + "\n", flush=True)

    def _self_test(self) -> None:
        """A small inference at load time, so a broken device shows up now rather than in the first job."""
        size, started = 256, time.time()
        lit = np.full((1, 3, size, size), 0.5, np.float32)
        geometry = np.ones((1, 5, size, size), np.float32)
        geometry[:, :2] = 0.5  # a flat normal, full AO, full mask
        try:
            output = self.model.run([self.output_name], {self.input_names[0]: lit, self.input_names[1]: geometry})[0]
        except Exception as error:
            _say(f"WARNING: self-test failed on {self.device_info}: {error}")
            return
        if np.isfinite(output).all():
            _say(f"self-test OK: {size} px in {time.time() - started:.2f} s, output {output.shape}, "
                 f"min {output.min():.3f} max {output.max():.3f}")
        else:
            _say(f"WARNING: self-test returned NaN or infinite values on {self.device_info}; try force CPU")

    def _load_image(self, path: str, channels: int = 3) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to load image at: {path}")

        if np.issubdtype(img.dtype, np.floating):
            scale = 1.0
        elif img.dtype == np.uint16:
            scale = 65535.0
        elif img.dtype == np.uint8:
            scale = 255.0
        else:
            raise ValueError(f"Unsupported image data type for '{path}': {img.dtype}")

        if channels == 1:
            if len(img.shape) == 3:
                img = img[:, :, 0:1]
            elif len(img.shape) == 2:
                img = img[:, :, np.newaxis]
        elif channels == 3:
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)

        img_np = img.astype(np.float32) / scale
        if not np.isfinite(img_np).all():
            raise ValueError(f"Image contains NaN or infinity values: {path}")
        img_np = np.clip(img_np, 0.0, 1.0)
        return np.transpose(img_np, (2, 0, 1)).astype(np.float32)

    def process_texture(
        self, 
        lit_path: str, 
        normal_path: str, 
        ao_path: str, 
        mask_path: str, 
        output_save_path: str,
        log_callback=None,
        tiled_mode: bool = False,
        tile_size: int = 0,
        tile_overlap: float = 0.25,
        bit_depth: int = 8,
        apply_mask: bool = True,
        progress_callback=None,
    ):
        """tile_size 0 = half the image; tile_overlap is a fraction of the tile;
        bit_depth 8 or 16 for the PNG written; apply_mask blacks out texels outside the UVs;
        progress_callback(fraction) hears how far it is, 0..1."""
        started = time.time()

        def log(msg):
            print(f"[Inference {time.time() - started:6.1f} s] {msg}", flush=True)
            if log_callback:
                log_callback(msg)

        def progress(fraction):
            if progress_callback:
                progress_callback(fraction)

        log(f"Starting on {self.device_info}: tiled {tiled_mode}, tile size {tile_size or 'auto'}, overlap {tile_overlap}, "
            f"{bit_depth}-bit, apply mask {apply_mask}")
        log("Loading texture maps onto GPU memory...")
        lit_t = self._load_image(lit_path, channels=3)
        normal_t = self._load_image(normal_path, channels=3)
        ao_t = self._load_image(ao_path, channels=1)
        mask_t = self._load_image(mask_path, channels=1)

        _, h, w = lit_t.shape
        inputs = {
            "normal map": normal_t,
            "ambient occlusion map": ao_t,
            "mask": mask_t,
        }
        for name, tensor in inputs.items():
            if tensor.shape[1:] != (h, w):
                raise ValueError(
                    f"{name.capitalize()} resolution {tensor.shape[2]}x{tensor.shape[1]} "
                    f"does not match lit map resolution {w}x{h}"
                )

        # Validate 1:1 aspect ratio, power of two, and max 8k limit
        if h != w:
            raise ValueError(f"Unsupported image aspect ratio: {w}x{h}. Only 1:1 square images are supported.")
        if h > 8192:
            raise ValueError(f"Image resolution {w}x{h} exceeds maximum supported resolution of 8192x8192.")
        if (h & (h - 1)) != 0 or h <= 0:
            raise ValueError(f"Image resolution {w}x{h} must be a power of two (e.g., 512, 1024, 2048, 4096, 8192).")

        log(_map_report("lit", lit_path, lit_t))
        log(_map_report("normal", normal_path, normal_t))
        log(_map_report("ao", ao_path, ao_t))
        log(_map_report("mask", mask_path, mask_t))
        inside = mask_t[0] > 0.5
        log(f"mask covers {inside.mean():.1%} of the map")
        if inside.mean() < 0.001:
            log("WARNING: the mask is practically empty, so the result will be black")
        log(f"Resolution: {w}x{h} px | Target Device: {self.device_info}")

        # 1. Color Space Conversion (sRGB -> Linear)
        log("Converting Lit RGB from sRGB to Linear space...")
        lit_linear = srgb_to_linear(lit_t)

        # 2. Geometry Stacking: Normals(3) + AO(1) + Mask(1) = 5 Channels
        progress(0.1)
        log("Stacking 5-Channel Geometry Maps...")
        geometry_t = np.concatenate([normal_t, ao_t, mask_t], axis=0)

        # 3. Dynamic Tiling Execution Strategy
        # Determine tile size: 1/2 of image size, or direct execution if < 2048
        tile_size = min(tile_size or h // 2, h)
        is_tiling_eligible = (h >= 2048) and tiled_mode and tile_size < h

        if is_tiling_eligible:
            overlap = max(1, int(tile_size * min(max(tile_overlap, 0.05), 0.5)))
            stride = tile_size - overlap

            # 2D cosine window for edge blending
            ramp_1d = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, overlap)))
            weight_1d = np.ones(tile_size, dtype=np.float32)
            weight_1d[:overlap] = ramp_1d
            weight_1d[-overlap:] = ramp_1d[::-1]
            tile_weights = np.outer(weight_1d, weight_1d)[np.newaxis, :, :]

            y_steps = list(range(0, max(1, h - tile_size + 1), stride))
            if y_steps[-1] + tile_size < h:
                y_steps.append(h - tile_size)

            x_steps = list(range(0, max(1, w - tile_size + 1), stride))
            if x_steps[-1] + tile_size < w:
                x_steps.append(w - tile_size)

            total_tiles = len(y_steps) * len(x_steps)
            log(f"Processing in tiled mode: {total_tiles} x {tile_size}x{tile_size} tiles, {overlap} px overlap...")

            accumulator = np.zeros((3, h, w), dtype=np.float32)
            weight_map = np.zeros((1, h, w), dtype=np.float32)
            tile_number = 0

            for y0 in y_steps:
                for x0 in x_steps:
                    tile_number += 1
                    y1, x1 = y0 + tile_size, x0 + tile_size

                    lit_tile = lit_linear[:, y0:y1, x0:x1]
                    geo_tile = geometry_t[:, y0:y1, x0:x1]

                    log(f"Executing tile {tile_number}/{total_tiles}...")
                    outputs = self.model.run(
                        [self.output_name],
                        {
                            self.input_names[0]: np.expand_dims(lit_tile, axis=0).astype(np.float32),
                            self.input_names[1]: np.expand_dims(geo_tile, axis=0).astype(np.float32),
                        },
                    )
                    tile_prediction = outputs[0][0]
                    log(f"Tile {tile_number}/{total_tiles} done, {tile_number / total_tiles:.0%}")

                    accumulator[:, y0:y1, x0:x1] += tile_prediction * tile_weights
                    weight_map[:, y0:y1, x0:x1] += tile_weights
                    progress(0.15 + 0.7 * tile_number / total_tiles)

            pred_albedo_linear = accumulator / np.maximum(weight_map, 1e-7)

        else:
            if h < 2048:
                log("Image resolution below 2048x2048: Processing full image without tiling...")
            else:
                log("Processing in full-resolution mode...")

            lit_tile = lit_linear
            geo_tile = geometry_t

            # Fallback padding if image is smaller than standard model patch target
            target_patch = min(h, self.TILE_SIZE)
            pad_h = max(0, target_patch - h)
            pad_w = max(0, target_patch - w)
            if pad_h or pad_w:
                lit_tile = np.pad(lit_tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
                geo_tile = np.pad(geo_tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

            log("Running the model on the whole map (no progress until it returns)...")
            outputs = self.model.run(
                [self.output_name],
                {
                    self.input_names[0]: np.expand_dims(lit_tile, axis=0).astype(np.float32),
                    self.input_names[1]: np.expand_dims(geo_tile, axis=0).astype(np.float32),
                },
            )
            pred_albedo_linear = outputs[0][0, :, :h, :w]
            log("Model finished")
            progress(0.85)

        # 4. Convert Linear -> sRGB & Apply UV Mask
        broken = int((~np.isfinite(pred_albedo_linear)).sum())
        if broken:
            raise RuntimeError(f"The model returned {broken} NaN or infinite values on {self.device_info}; try force CPU.")
        log("Converting predicted Albedo to sRGB space...")
        pred_srgb = linear_to_srgb(pred_albedo_linear)
        if inside.any():  # did it do anything? Compared with the lit map, inside the UVs
            result, change = pred_srgb[:, inside], float(np.abs(pred_srgb - lit_t)[:, inside].mean())
            log(f"result inside the mask: min {result.min():.3f} max {result.max():.3f} mean {result.mean():.3f}, "
                f"mean change from the lit map {change:.4f}")
            if result.mean() < 0.005:
                log("WARNING: the result is practically black")
            if change < 0.002:
                log("WARNING: the result is practically identical to the lit map: nothing was de-lit")
        if apply_mask:
            pred_srgb = pred_srgb * mask_t

        # 5. Save output map to disk
        progress(0.9)
        log(f"Saving final {bit_depth}-bit image to disk...")
        pred_np = np.transpose(pred_srgb, (1, 2, 0))
        if bit_depth == 16:
            pred_int = (np.clip(pred_np, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        else:
            pred_int = (np.clip(pred_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        output_bgr = cv2.cvtColor(pred_int, cv2.COLOR_RGB2BGR)

        output_path = Path(output_save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), output_bgr):
            raise IOError(f"Failed to write output image: {output_path}")
        progress(1.0)
        log(f"SUCCESS: Saved output to '{output_path}' ({output_path.stat().st_size / 2 ** 20:.1f} MB) "
            f"in {time.time() - started:.1f} s")

        return str(output_path), (w, h), self.device_info
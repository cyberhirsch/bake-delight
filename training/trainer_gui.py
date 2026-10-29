"""Trainer console: start, watch and stop the bake -> post-process -> train pipeline.

    training\\.venv\\Scripts\\python.exe training\\trainer_gui.py

Everything shown is read from the pipeline's own files (logs, log.csv, val_*.jpg) and
from nvidia-smi, so the window can be opened and closed at any time, also while a run
started elsewhere (start_overcast.bat, the scheduled task) is going. Closing the window
does not stop a run.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from run_pipeline import PROFILES  # noqa: E402
from tui import ACCENT, BG, BORDER, DIM, HINT, PANEL, TEXT, TEXT_HI, WARN, TuiWidgets  # noqa: E402

SETTINGS_PATH = Path(os.environ.get("APPDATA", Path.home())) / "TextureDelighter" / "trainer.json"
VENV_PYTHON = HERE / ".venv" / "Scripts" / "python.exe"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
STAGES = {"1_bake": "BAKE", "2_postprocess": "POST-PROCESS", "3_train": "TRAIN"}
DEFAULTS = {
    "data_root": os.environ.get("DELIGHTER_DATA", r"F:\DelighterData"), "python": str(VENV_PYTHON),
    "profile": next(iter(PROFILES)), "tag": "", "stop_after": "train", "steps": "", "variations": "", "limit": "0",
    "batch": "", "crop": "", "lr": "", "eval_every": "", "follow_latest": True, "chart_log": False,
}
TRAIN_LINE = re.compile(r"step (\d+)/(\d+)\s+loss ([\d.]+)\s+([\d.]+) it/s\s+eta (\d+) min")
CROSSHAIR = "#444444"
BAKE_LINE = re.compile(r"\[(\d+)/(\d+)\] (\S+): (ok|SKIP|FAILED)")


def tail(path: Path, lines: int = 60) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 64 * 1024))
            return handle.read().decode("utf-8", "replace").splitlines()[-lines:]
    except OSError:
        return []


def running_pipelines() -> list[dict]:
    """Every run_pipeline.py process on the machine: pid, profile name (with tag), command line."""
    script = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Where-Object { $_.CommandLine -like '*run_pipeline.py*' } | "
              "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress")
    try:
        output = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True,
                                timeout=20, creationflags=NO_WINDOW).stdout.strip()
        rows = json.loads(output) if output else []
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    found, seen = [], set()
    for row in rows if isinstance(rows, list) else [rows]:
        command = row.get("CommandLine") or ""
        profile = re.search(r"--profile\s+(\S+)", command)
        tag = re.search(r"--tag\s+(\S+)", command)
        if not profile:
            continue
        if command in seen:
            next(entry for entry in found if entry["command"] == command)["pids"].append(row["ProcessId"])
            continue
        seen.add(command)
        found.append({"pid": row["ProcessId"], "pids": [row["ProcessId"]],
                      "name": profile.group(1) + (tag.group(1) if tag else ""), "command": command})
    return found


def gpu_stats() -> dict | None:
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                                creationflags=NO_WINDOW).stdout.strip().splitlines()
        name, load, used, total, temp, power = [part.strip() for part in output[0].split(",")]
        return {"name": name, "load": float(load), "used": float(used) / 1024, "total": float(total) / 1024,
                "temp": float(temp), "power": float(power) if power.replace(".", "").isdigit() else None}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


class TrainerGUI(TuiWidgets):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        self.settings = self._load_settings()
        self.root = ctk.CTk(fg_color=BG)
        self.root.title("delighter trainer")
        self.root.geometry("1500x920")
        self.root.minsize(1150, 720)
        self.font_family = self._pick_font()
        self.var = {}
        for key, default in DEFAULTS.items():
            value = self.settings.get(key, default)
            self.var[key] = tk.BooleanVar(value=bool(value)) if isinstance(default, bool) else tk.StringVar(value=str(value))
        self._ui_queue = queue.Queue()
        self.state = {"pipelines": [], "gpu": None, "history": [], "previews": [], "progress": {}, "log": [], "data": {}}
        self.preview_index = None
        self._preview_cache = (None, None, None)
        self._hover = {}
        self._own_process = None

        self._build_ui()
        self._set_icon()
        # Pollers run in threads, which must not touch Tk variables; they read this copy instead.
        self.values = {key: variable.get() for key, variable in self.var.items()}
        for key, variable in self.var.items():
            variable.trace_add("write", lambda *_args, key=key: self._setting_changed(key))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._drain_ui_queue()
        self._start_pollers()

    # ================================================================ plumbing

    def _post(self, callback):
        self._ui_queue.put(callback)

    def _drain_ui_queue(self):
        while True:
            try:
                callback = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except Exception as error:  # one failing callback must not stop the pump
                print(f"ui callback: {error}", file=sys.stderr)
        self.root.after(40, self._drain_ui_queue)

    def _loop(self, seconds, work):
        """Run `work` in a background thread every `seconds`; it posts its own UI updates."""
        def run():
            while True:
                try:
                    work()
                except Exception as error:
                    print(f"poller: {error}", file=sys.stderr)
                time.sleep(seconds)
        threading.Thread(target=run, daemon=True).start()

    @staticmethod
    def _load_settings():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_settings(self):
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps({key: variable.get() for key, variable in self.var.items()}, indent=2),
                                     encoding="utf-8")
        except (OSError, tk.TclError):
            pass

    def _setting_changed(self, key):
        try:
            self.values[key] = self.var[key].get()
        except tk.TclError:  # half-typed entry
            return
        self._save_settings()

    def _on_close(self):
        self._save_settings()
        self.root.destroy()

    # ---------------------------------------------------------------- paths of the selected run

    @property
    def name(self) -> str:
        return self.values["profile"] + str(self.values["tag"]).strip()

    def _paths(self) -> dict:
        root, name = Path(self.values["data_root"]), self.name
        return {"root": root, "logs": root / "logs" / name, "pipeline_log": root / "logs" / f"pipeline_{name}.log",
                "baked": root / f"baked_{name}", "dataset": root / f"dataset_{name}", "run": root / "runs" / name,
                "source": root / "source"}

    # ================================================================ layout

    def _build_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self._build_top_bar()
        main = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)
        main.grid(row=1, column=0, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)
        self._build_monitor(main)
        self._build_sidebar(main)

    def _build_top_bar(self):
        bar = ctk.CTkFrame(self.root, fg_color=PANEL, corner_radius=0, height=46, border_width=1, border_color=BORDER)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(3, weight=1)
        self._logo = ctk.CTkImage(self._logo_image(64), size=(26, 26))
        ctk.CTkLabel(bar, image=self._logo, text="").grid(row=0, column=0, padx=(14, 8), pady=9)
        self._label(bar, "DE", color=TEXT_HI, size=17, weight="bold").grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(bar, text="LIGHTER · TRAINER", text_color=DIM, anchor="w",
                     font=ctk.CTkFont(family=self._light_family(), size=17)).grid(row=0, column=2, sticky="w")
        self.run_label = self._label(bar, "", color=DIM)
        self.run_label.grid(row=0, column=3, sticky="e", padx=10)
        self.status_label = self._label(bar, "○ IDLE", color=DIM, weight="bold")
        self.status_label.grid(row=0, column=4, sticky="e", padx=(0, 16))

    def _build_monitor(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG, corner_radius=0)
        frame.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=3)
        frame.grid_rowconfigure(3, weight=4)
        frame.grid_rowconfigure(4, weight=2)

        # readouts
        strip = ctk.CTkFrame(frame, fg_color=PANEL, corner_radius=0, border_width=1, border_color=BORDER)
        strip.grid(row=0, column=0, sticky="ew")
        self.readouts = {}
        for column, key in enumerate(["STAGE", "PROGRESS", "SPEED", "ETA", "LOSS", "BEST VAL", "GPU"]):
            strip.grid_columnconfigure(column, weight=1)
            self._label(strip, key, color=HINT, size=10).grid(row=0, column=column, sticky="w", padx=12, pady=(8, 0))
            value = self._label(strip, "—", color=TEXT_HI, size=15, weight="bold")
            value.grid(row=1, column=column, sticky="w", padx=12)
            detail = self._label(strip, "", color=DIM, size=10)
            detail.grid(row=2, column=column, sticky="w", padx=12, pady=(0, 8))
            self.readouts[key] = (value, detail)
        self.progress = ctk.CTkProgressBar(frame, height=4, corner_radius=0, fg_color=BORDER, progress_color=ACCENT)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        self.progress.set(0)

        # charts
        self.chart = tk.Canvas(frame, bg=BG, highlightthickness=1, highlightbackground=BORDER, height=220)
        self.chart.grid(row=2, column=0, sticky="nsew")
        self.chart.bind("<Configure>", lambda _event: self._draw_charts())
        self.chart.bind("<Motion>", self._chart_hover)
        self.chart.bind("<Leave>", lambda _event: (self._hover.clear(), self._draw_charts()))

        # validation preview
        box = ctk.CTkFrame(frame, fg_color=PANEL, corner_radius=0, border_width=1, border_color=BORDER)
        box.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        box.grid_columnconfigure(1, weight=1)
        box.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(box, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(6, 2))
        head.grid_columnconfigure(1, weight=1)
        self.preview_title = self._label(head, "┌─ VALIDATION  lit │ prediction │ target", color=ACCENT, weight="bold")
        self.preview_title.grid(row=0, column=0, sticky="w")
        self.preview_slider = ctk.CTkSlider(head, from_=0, to=1, number_of_steps=1, height=14, corner_radius=0,
                                            button_corner_radius=0, button_length=6, fg_color=BORDER, progress_color="#2e2e2e",
                                            button_color=ACCENT, button_hover_color="#e6e6e6", command=self._on_preview_slider)
        self.preview_slider.grid(row=0, column=1, sticky="ew", padx=12)
        self._check(head, "follow latest", self.var["follow_latest"], command=self._show_preview).grid(row=0, column=2)
        self.preview_label = tk.Label(box, bg=PANEL, fg=DIM, text="no validation images yet",
                                      font=(self.font_family, 11))
        self.preview_label.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=6, pady=(0, 6))
        self.preview_label.bind("<Configure>", lambda _event: self._show_preview())

        # stage log
        self.log_box = ctk.CTkTextbox(frame, fg_color=PANEL, corner_radius=0, border_width=1, border_color=BORDER,
                                      text_color=DIM, font=self._font(11), wrap="none")
        self.log_box.grid(row=4, column=0, sticky="nsew", pady=(6, 0))

    def _build_sidebar(self, parent):
        side = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0, width=430, border_width=1, border_color=BORDER)
        side.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(1, weight=1)
        self.tab_var = tk.StringVar(value="RUN")
        self._segmented(side, self.tab_var, ["RUN", "DATA", "SYS"], command=lambda _value: self._show_tab()).grid(
            row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        self.tabs = {}
        for name, build in (("RUN", self._build_run_tab), ("DATA", self._build_data_tab), ("SYS", self._build_sys_tab)):
            tab = ctk.CTkScrollableFrame(side, fg_color=PANEL, corner_radius=0)
            tab.grid_columnconfigure(1, weight=1)
            tab.next_row = 0
            build(tab)
            self.tabs[name] = tab
        actions = ctk.CTkFrame(side, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        actions.grid_columnconfigure((0, 1), weight=1)
        self.start_button = self._button(actions, "start / resume", self._start, primary=True)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_button = self._button(actions, "stop", self._stop)
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.action_hint = self._label(actions, "", color=HINT, size=10)
        self.action_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._show_tab()

    def _show_tab(self):
        for name, tab in self.tabs.items():
            if name == self.tab_var.get():
                tab.grid(row=1, column=0, sticky="nsew", padx=2)
            else:
                tab.grid_forget()

    def _hint(self, tab, text):
        self._label(tab, f"│   {text}", color=HINT, size=11).grid(row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1

    def _build_run_tab(self, tab):
        self._section(tab, "profile")
        self._row(tab, "PROFILE", lambda parent: self._segmented(parent, self.var["profile"], list(PROFILES),
                                                                 command=lambda _value: self._profile_changed()))
        self.profile_hint = self._label(tab, "", color=HINT, size=11)
        self.profile_hint.grid(row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1
        self._row(tab, "TAG", lambda parent: self._entry(parent, self.var["tag"]), hint="suffix for a separate test run")
        self._section(tab, "stages")
        self._row(tab, "STOP AFTER", lambda parent: self._segmented(parent, self.var["stop_after"], ["bake", "postprocess", "train"]))
        self._hint(tab, "every stage resumes; finished work is skipped")
        self._section(tab, "bake")
        self._row(tab, "VARIATIONS", lambda parent: self._entry(parent, self.var["variations"], width=90), hint="lightings per asset; empty = profile")
        self._row(tab, "LIMIT", lambda parent: self._entry(parent, self.var["limit"], width=90), hint="first N assets only; 0 = all")
        self._section(tab, "train")
        self._row(tab, "STEPS", lambda parent: self._entry(parent, self.var["steps"], width=90), hint="empty = profile")
        self._row(tab, "BATCH", lambda parent: self._entry(parent, self.var["batch"], width=90), hint="empty = 8")
        self._row(tab, "CROP PX", lambda parent: self._entry(parent, self.var["crop"], width=90), hint="empty = 512")
        self._row(tab, "LEARN RATE", lambda parent: self._entry(parent, self.var["lr"], width=90), hint="empty = 3e-4")
        self._row(tab, "EVAL EVERY", lambda parent: self._entry(parent, self.var["eval_every"], width=90),
                  hint="steps between checkpoints; empty = 1000")
        self._hint(tab, "a resumed run keeps its schedule; batch / crop / lr")
        self._hint(tab, "changes apply from the next start")
        self._profile_changed()

    def _build_data_tab(self, tab):
        self._section(tab, "dataset")
        self.data_labels = {}
        for key in ("source models", "hdris", "baked", "skipped", "failed", "post-processed", "lit pairs"):
            self.data_labels[key] = self._row(tab, key.upper(), lambda parent: self._label(parent, "—", color=TEXT))
        self.failed_label = self._label(tab, "", color=HINT, size=11, wraplength=380, justify="left")
        self.failed_label.grid(row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(18, 6))
        tab.next_row += 1
        self._section(tab, "open")
        for label, key in (("baked", "baked"), ("dataset", "dataset"), ("run", "run"), ("logs", "logs")):
            self._row(tab, label.upper(), lambda parent, key=key: self._button(parent, "open folder", lambda: self._open(key)))

    def _build_sys_tab(self, tab):
        self._section(tab, "paths")
        self._row(tab, "DATA ROOT", lambda parent: self._entry(parent, self.var["data_root"]))
        self._row(tab, "PYTHON", lambda parent: self._entry(parent, self.var["python"]), hint="the training venv")
        self._section(tab, "chart")
        self._row(tab, "Y AXIS", lambda parent: self._check(parent, "logarithmic", self.var["chart_log"], command=self._draw_charts))
        self._section(tab, "gpu")
        self.gpu_label = self._row(tab, "DEVICE", lambda parent: self._label(parent, "—", color=TEXT, wraplength=260))
        self._section(tab, "about")
        self._hint(tab, "reads logs, log.csv and val_*.jpg of the run;")
        self._hint(tab, "closing this window never stops a run")

    def _profile_changed(self):
        profile = PROFILES.get(self.var["profile"].get(), {})
        self.profile_hint.configure(text=f"│   hdris {profile.get('hdris')} · sun {profile.get('sun_probability')} · "
                                         f"{profile.get('variations')} lights · {profile.get('steps')} steps")
        self.state["history"], self.state["previews"], self.preview_index = [], [], None
        self._draw_charts()

    # ================================================================ polling

    def _start_pollers(self):
        self._loop(10, lambda: self._post_state("pipelines", running_pipelines()))
        self._loop(3, lambda: self._post_state("gpu", gpu_stats()))
        self._loop(2, self._poll_run)
        self._loop(30, self._poll_data)

    def _post_state(self, key, value):
        def apply():
            self.state[key] = value
            self._refresh()
        self._post(apply)

    def _poll_run(self):
        paths = self._paths()
        stage, stage_time = None, None
        for line in tail(paths["pipeline_log"], 200):
            match = re.match(r"\[([\d\- :]+)\] (\d_\w+):", line)
            if match:
                stage_time, stage = match.group(1), match.group(2)
        progress = {"stage": stage, "since": stage_time}
        log = tail(paths["logs"] / f"{stage}.log", 80) if stage else []
        for line in log:
            if (match := TRAIN_LINE.search(line)):
                progress.update(step=int(match[1]), steps=int(match[2]), loss=float(match[3]), rate=float(match[4]), eta=int(match[5]))
            elif (match := BAKE_LINE.search(line)):
                progress.update(asset=int(match[1]), assets=int(match[2]), last=match[3])
            elif line.startswith("done") or line.startswith("bake finished") or line.startswith("baked "):
                progress["finished_line"] = line
        if stage == "2_postprocess":
            progress["processed"] = sum(1 for line in tail(paths["logs"] / "2_postprocess.log", 5000) if " coverage " in line)
        history = []
        try:
            with (paths["run"] / "log.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    history.append({key: float(value) for key, value in row.items()})
        except (OSError, ValueError):
            pass
        previews = sorted(paths["run"].glob("val_*.jpg")) if paths["run"].is_dir() else []

        def apply():
            changed = len(history) != len(self.state["history"]) or self.state["previews"] != previews
            self.state.update(progress=progress, log=log, history=history, previews=previews)
            self._refresh()
            if changed:
                self._draw_charts()
                self._show_preview()
        self._post(apply)

    def _poll_data(self):
        paths = self._paths()
        data = {}
        models = paths["source"] / "models"
        data["source models"] = sum(1 for p in models.iterdir() if p.is_dir()) if models.is_dir() else 0
        hdris = paths["source"] / PROFILES.get(self.values["profile"], {}).get("hdris", "")
        data["hdris"] = len(list(hdris.glob("*.hdr"))) if hdris.is_dir() else 0
        baked = skipped = 0
        failed = []
        if paths["baked"].is_dir():
            for folder in paths["baked"].iterdir():
                if (folder / "meta.json").is_file():
                    baked += 1
                elif (folder / "skipped.txt").is_file():
                    skipped += 1
                elif (folder / "blender.log").is_file():
                    failed.append(folder.name)
        data.update(baked=baked, skipped=skipped, failed=len(failed), failed_names=failed)
        if paths["dataset"].is_dir():
            assets = [p for p in paths["dataset"].iterdir() if (p / "sample.json").is_file()]
            data["post-processed"] = len(assets)
            data["lit pairs"] = sum(len(list(p.glob("lit_*.png"))) for p in assets)
        self._post_state("data", data)

    # ================================================================ display

    def _mine(self):
        return [p for p in self.state["pipelines"] if p["name"] == self.name]

    def _refresh(self):
        running = self.state["pipelines"]
        mine = self._mine()
        progress = self.state["progress"]
        stage = progress.get("stage")
        set_value = lambda key, value, detail="": (self.readouts[key][0].configure(text=value),
                                                   self.readouts[key][1].configure(text=detail))
        self.run_label.configure(text=f"run {self.name}  ·  {self._paths()['run']}")
        if mine:
            self.status_label.configure(text=f"● RUNNING  pid {mine[0]['pid']}", text_color=TEXT_HI)
        elif running:
            self.status_label.configure(text=f"● OTHER RUN: {running[0]['name']}", text_color=WARN)
        else:
            self.status_label.configure(text="○ IDLE", text_color=DIM)
        # a disabled primary button keeps its white fill unless told otherwise
        self.start_button.configure(state="disabled" if running else "normal", fg_color="#1a1a1a" if running else ACCENT,
                                    border_color=BORDER if running else ACCENT)
        self.stop_button.configure(state="normal" if mine else "disabled")
        self.action_hint.configure(text="one pipeline at a time: the GPU is shared" if running and not mine else
                                   "stop loses at most one eval interval of training" if mine else "")

        label = STAGES.get(stage, "—")
        set_value("STAGE", label if mine or not stage else f"{label} (stopped)" if not progress.get("finished_line") else label,
                  f"since {progress['since'][11:16]}" if progress.get("since") else "")
        fraction = 0.0
        if stage == "3_train" and "step" in progress:
            fraction = progress["step"] / progress["steps"]
            set_value("PROGRESS", f"{progress['step']:,}", f"of {progress['steps']:,} steps")
            set_value("SPEED", f"{progress['rate']:.1f} it/s")
            finish = time.localtime(time.time() + progress["eta"] * 60)
            set_value("ETA", f"{progress['eta'] // 60}h {progress['eta'] % 60:02d}m" if mine else "—",
                      time.strftime("done ~%a %H:%M", finish) if mine else "")
            set_value("LOSS", f"{progress['loss']:.4f}", "train, last 100 steps")
        elif stage == "1_bake" and "asset" in progress:
            fraction = progress["asset"] / progress["assets"]
            set_value("PROGRESS", f"{progress['asset']}", f"of {progress['assets']} assets")
            for key in ("SPEED", "ETA", "LOSS"):
                set_value(key, "—")
        elif stage == "2_postprocess":
            total = self.state["data"].get("baked") or 0
            done = progress.get("processed", 0)
            fraction = done / total if total else 0
            set_value("PROGRESS", f"{done}", f"of {total} assets" if total else "")
            for key in ("SPEED", "ETA", "LOSS"):
                set_value(key, "—")
        self.progress.set(fraction)
        history = self.state["history"]
        if history:
            best = min(history, key=lambda row: row["val_l1"])
            gain = 1 - best["val_l1"] / best["baseline_l1"] if best["baseline_l1"] else 0
            set_value("BEST VAL", f"{best['val_l1']:.4f}", f"step {int(best['step']):,} · {gain:.0%} under baseline")
        else:
            set_value("BEST VAL", "—")
        gpu = self.state["gpu"]
        if gpu:
            set_value("GPU", f"{gpu['load']:.0f}%", f"{gpu['used']:.1f}/{gpu['total']:.0f} GB · {gpu['temp']:.0f}°C"
                      + (f" · {gpu['power']:.0f} W" if gpu["power"] else ""))
            self.gpu_label.configure(text=gpu["name"])
        else:
            set_value("GPU", "—", "nvidia-smi unavailable")
        text = "\n".join(self.state["log"][-40:])
        if text != self.log_box.get("1.0", "end").rstrip("\n"):
            self.log_box.delete("1.0", "end")
            self.log_box.insert("end", text)
            self.log_box.see("end")
        data = self.state["data"]
        for key, label in self.data_labels.items():
            value = data.get(key)
            label.configure(text="—" if value is None else f"{value:,}")
        names = data.get("failed_names") or []
        self.failed_label.configure(text=("failed: " + ", ".join(names)) if names else "")

    # ---------------------------------------------------------------- charts

    def _chart_panels(self):
        width, height = self.chart.winfo_width(), self.chart.winfo_height()
        gap = 16
        half = (width - gap) // 2
        return [(0, 0, half, height), (half + gap, 0, width - half - gap, height)]

    def _draw_charts(self):
        canvas = self.chart
        canvas.delete("all")
        history = self.state["history"]
        panels = self._chart_panels()
        series = [
            ("TRAIN LOSS", [("train loss", "train_loss", TEXT_HI, None)]),
            ("VALIDATION L1", [("prediction", "val_l1", TEXT_HI, None), ("do-nothing baseline", "baseline_l1", DIM, (4, 3))]),
        ]
        for (x0, y0, width, height), (title, lines) in zip(panels, series):
            self._draw_panel(canvas, x0, y0, width, height, title, lines, history)
        if not history:
            canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2, text="no checkpoints yet",
                               fill=HINT, font=(self.font_family, 11))

    def _draw_panel(self, canvas, x0, y0, width, height, title, lines, history):
        font, small = (self.font_family, 10, "bold"), (self.font_family, 9)
        canvas.create_text(x0 + 12, y0 + 12, text=f"┌─ {title}", fill=ACCENT, anchor="w", font=font)
        left, right, top, bottom = x0 + 56, x0 + width - 14, y0 + 54, y0 + height - 26
        if not history or right - left < 40 or bottom - top < 30:
            return
        steps = [row["step"] for row in history]
        values = [row[key] for _label, key, _color, _dash in lines for row in history]
        logarithmic = self.var["chart_log"].get()
        transform = (lambda v: math.log10(max(v, 1e-6))) if logarithmic else (lambda v: v)
        low, high = min(map(transform, values)), max(map(transform, values))
        if not logarithmic:
            low = 0.0
        if high - low < 1e-9:
            high = low + 1
        pad = (high - low) * 0.08
        high += pad
        low = low - pad if logarithmic else low
        last_step = max(steps[-1], 1)
        x_of = lambda step: left + (right - left) * step / last_step
        y_of = lambda value: bottom - (bottom - top) * (transform(value) - low) / (high - low)
        # recessive grid and axes
        for i in range(5):
            level = low + (high - low) * i / 4
            y = bottom - (bottom - top) * i / 4
            canvas.create_line(left, y, right, y, fill="#161616")
            label = 10 ** level if logarithmic else level
            canvas.create_text(left - 6, y, text=f"{label:.3f}", fill=HINT, anchor="e", font=small)
        for i in range(5):
            x = left + (right - left) * i / 4
            canvas.create_text(x, bottom + 12, text=f"{int(last_step * i / 4):,}", fill=HINT, font=small)
        canvas.create_line(left, bottom, right, bottom, fill=BORDER)
        # series
        for label, key, color, dash in lines:
            points = [coordinate for row in history for coordinate in (x_of(row["step"]), y_of(row[key]))]
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2, dash=dash)
            for row in history[-1:]:
                canvas.create_oval(x_of(row["step"]) - 3, y_of(row[key]) - 3, x_of(row["step"]) + 3, y_of(row[key]) + 3,
                                   fill=color, outline=BG, width=2)
        if len(lines) > 1:  # legend: line sample + name, identity never by brightness alone
            x = x0 + 200
            for label, _key, color, dash in lines:
                canvas.create_line(x, y0 + 12, x + 22, y0 + 12, fill=color, width=2, dash=dash)
                canvas.create_text(x + 28, y0 + 12, text=label, fill=TEXT, anchor="w", font=small)
                x += 40 + len(label) * 7
        if lines[0][1] == "val_l1":  # best checkpoint marker
            best = min(history, key=lambda row: row["val_l1"])
            bx, by = x_of(best["step"]), y_of(best["val_l1"])
            canvas.create_rectangle(bx - 4, by - 4, bx + 4, by + 4, outline=TEXT_HI, width=1)
        # hover: crosshair on the nearest checkpoint, values in the panel header
        hover = self._hover.get("x")
        if hover is not None and left <= hover <= right:
            row = min(history, key=lambda r: abs(x_of(r["step"]) - hover))
            x = x_of(row["step"])
            canvas.create_line(x, top, x, bottom, fill=CROSSHAIR)
            text = f"step {int(row['step']):,}   " + "   ".join(f"{label} {row[key]:.4f}" for label, key, _c, _d in lines)
            canvas.create_text(x0 + 12, y0 + 30, text=text, fill=TEXT_HI, anchor="w", font=small)
            for _label, key, color, _dash in lines:
                y = y_of(row[key])
                canvas.create_oval(x - 4, y - 4, x + 4, y + 4, outline=color, width=2)

    def _chart_hover(self, event):
        self._hover["x"] = event.x
        self._draw_charts()

    # ---------------------------------------------------------------- validation preview

    def _on_preview_slider(self, value):
        self.var["follow_latest"].set(False)
        self.preview_index = int(round(value))
        self._show_preview()

    def _show_preview(self):
        previews = self.state["previews"]
        if not previews:
            self.preview_label.configure(image="", text="no validation images yet")
            return
        if self.var["follow_latest"].get() or self.preview_index is None or self.preview_index >= len(previews):
            self.preview_index = len(previews) - 1
        self.preview_slider.configure(to=max(len(previews) - 1, 1), number_of_steps=max(len(previews) - 1, 1))
        self.preview_slider.set(self.preview_index)
        path = previews[self.preview_index]
        width, height = self.preview_label.winfo_width(), self.preview_label.winfo_height()
        if width < 20 or height < 20:
            return
        key = (path, width, height)
        if self._preview_cache[0] != key:
            image = Image.open(path)
            scale = min(width / image.width, height / image.height)
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
            self._preview_cache = (key, ImageTk.PhotoImage(image, master=self.root), None)
        self.preview_label.configure(image=self._preview_cache[1], text="")
        step = int(path.stem.split("_")[1])
        self.preview_title.configure(text=f"┌─ VALIDATION  step {step:,}  ·  lit │ prediction │ target")

    # ================================================================ actions

    def _command(self) -> list[str]:
        command = [self.var["python"].get(), "-u", str(HERE / "run_pipeline.py"), "--profile", self.var["profile"].get(),
                   "--stop-after", self.var["stop_after"].get()]
        tag = self.var["tag"].get().strip()
        if tag:
            command += ["--tag", tag]
        for key, flag, cast in (("steps", "--steps", int), ("variations", "--variations", int), ("limit", "--limit", int),
                                ("batch", "--batch", int), ("crop", "--crop", int), ("lr", "--lr", float),
                                ("eval_every", "--eval-every", int)):
            text = self.var[key].get().strip()
            if text:
                cast(text)  # raises ValueError on a typo
                command += [flag, text]
        return command

    def _start(self):
        if self.state["pipelines"]:
            return
        try:
            command = self._command()
        except ValueError:
            messagebox.showwarning("Trainer", "Steps, variations, limit, batch, crop and eval-every must be whole numbers; "
                                              "learn rate a number.", parent=self.root)
            return
        if not Path(command[0]).is_file():
            messagebox.showwarning("Trainer", f"Python not found: {command[0]}", parent=self.root)
            return
        log_path = self._paths()["pipeline_log"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        # Its own process group without a console, so it outlives this window.
        self._own_process = subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                             creationflags=NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        log.close()
        self.status_label.configure(text="● STARTING", text_color=TEXT_HI)
        self.start_button.configure(state="disabled")
        threading.Thread(target=lambda: self._post_state("pipelines", running_pipelines()), daemon=True).start()

    def _stop(self):
        mine = self._mine()
        if not mine:
            return
        if not messagebox.askyesno("Stop run", f"Stop the {self.name} pipeline (pid {mine[0]['pid']}) and its current stage?\n\n"
                                               "Training resumes from the last checkpoint on the next start.", parent=self.root):
            return
        for pid in (pid for process in mine for pid in process["pids"]):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
        threading.Thread(target=lambda: self._post_state("pipelines", running_pipelines()), daemon=True).start()

    def _open(self, key):
        path = self._paths()[key]
        if path.exists():
            os.startfile(path)
        else:
            messagebox.showinfo("Trainer", f"{path} does not exist yet.", parent=self.root)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    TrainerGUI().run()

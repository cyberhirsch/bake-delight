import json
import queue
import os
import subprocess
import sys
import threading
import time
import traceback
import webbrowser

import applog

LOG_PATH = applog.start()  # before the heavy imports, so a failing one is logged too

import cv2
import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image

from bakedelight import models
from bakedelight.project import SUFFIX as PROJECT_SUFFIX, Project, ProjectError
from engine import DelighterInferenceEngine
from tui import (ACCENT, ACCENT_DIM, BG, BORDER, BORDER_HI, DIM, ERROR, FIELD, HINT, PANEL, TEXT, TEXT_HI,  # noqa: F401
                 WARN, Button, ScrollPanel, TuiWidgets, background_of)
try:
    from viewport import OPENGL_AVAILABLE, Camera, TextureViewport, forget, load_obj
except Exception:
    OPENGL_AVAILABLE = False
    TextureViewport = None
    forget = lambda _path: None  # noqa: E731


APP_DIR = os.path.dirname(os.path.abspath(__file__))
SCANOPT_SCRIPT = os.path.join(APP_DIR, "scanopt", "scanopt.py")
SETTINGS_PATH = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "TextureDelighter", "settings.json")
# Mirrors scanopt.REMESHERS; kept here so the GUI starts even if scanopt's dependencies are missing.
REMESHERS = {
    "Triangles - decimation (QEM)": "decimate",
    "Quads - AutoRemesher (MIT)": "autoremesher",
    "Quads - QuadWild + Bi-MDF (GPL)": "quadwild",
}
VIEW_MODES = ["Single", "Side by side", "A/B wipe"]
# Sidebar tabs in workflow order; the numbered ones are pipeline stages with their own run button.
TABS = {"import": "1 IMPORT", "remesh": "2 REMESH", "uv": "3 UV", "bake": "4 BAKE", "delight": "5 DELIGHT",
        "export": "6 EXPORT", "view": "VIEW", "sys": "SYS"}
STAGES = ("remesh", "uv", "bake", "delight", "export")
# Graphics memory -> de-light settings. One pass of the model measured on the CPU provider:
# 0.26 GB at 512 px, 1.0 GB at 1024, 4.1 GB at 2048 (x4 per doubling); the presets leave about
# twice that plus 1 GB for the GPU runtime and the desktop. The tile size caps every pass,
# so 4096 px maps tile too.
VRAM_PRESETS = {
    "CPU only": {"force_cpu": True, "tiled": True, "scan_tiled": True, "tile_size": "2048"},
    "4 GB": {"force_cpu": False, "tiled": True, "scan_tiled": True, "tile_size": "512"},
    "6-8 GB": {"force_cpu": False, "tiled": True, "scan_tiled": True, "tile_size": "1024"},
    "12 GB +": {"force_cpu": False, "tiled": True, "scan_tiled": True, "tile_size": "2048"},
}
# View map -> (files in the stage work folder, exported texture suffixes, 1 IMPORT settings) that can hold it.
VIEW_MAPS = {
    "lit": (("lit.png", "lit8.png"), ("lit",), ("lit",)),
    "albedo": (("albedo.png", "albedo_raw.png"), ("albedo",), ()),
    "normal": (("normal.png",), ("normal_gl",), ()),
    "normal OS": (("normal_os.png",), (), ("normal",)),
    "AO": (("ao8.png", "ao.png"), ("ao",), ("ao",)),
    "roughness": (("roughness.png",), ("roughness",), ()),
    "metallic": (("metallic.png",), ("metallic",), ()),
    "mask": (("mask.png",), (), ("mask",)),
}  # scanopt's order: its "[n/5]" lines count these
STAGE_OF_TAB = {TABS[stage]: stage for stage in STAGES}
# Original scans above this size are not parsed for the before view; the baked photo is used instead.
MAX_BEFORE_SCAN_BYTES = 2 * 1024 * 1024 * 1024  # an 880 MB OBJ (8 M triangles) loads in about 35 s
BACKGROUNDS = {"Void": (0.0, 0.0, 0.0), "Charcoal": (0.07, 0.07, 0.07), "Graphite": (0.16, 0.16, 0.16),
               "Mid grey": (0.42, 0.42, 0.42), "Paper": (0.86, 0.86, 0.86)}

# Every persisted setting: key -> default. The type of the default picks the Tk variable.
DEFAULTS = {
    # input
    "model": "", "lit": "", "normal": "", "ao": "", "mask": "", "output_dir": "",
    # de-light
    "ai_model": models.DEFAULT, "force_cpu": False, "tiled": True, "tile_size": "auto", "tile_overlap": 0.25,
    "bit_depth": "8", "apply_mask": True, "output_name": "delighted_albedo.png", "show_after_run": True,
    # scan optimiser
    "scan": "", "photo": "", "scan_out": "", "source": "scan", "remesher": next(iter(REMESHERS)), "faces": "10000",
    "resolution": "2048", "margin": "16", "ao_samples": "64",
    "cage": 0.01, "max_ray": 0.03, "roughness": 0.8, "roughness_variation": 0.25,
    "predecimate": 2.0, "sharp": False, "scan_tiled": False, "keep_work": False,
    # view
    "view_mode": VIEW_MODES[0], "fov": 50.0, "wireframe": False, "solid": False, "show_cage": False, "view_map": "auto", "checker": False, "checker_cells": "32",
    "texture_filter": "mipmap", "background": "Charcoal", "orbit_speed": 0.5, "zoom_step": 1.14,
    "autorotate": False, "autorotate_speed": 20.0,
    # system
    "quadwild_dir": r"F:\DelighterData\tools\quadwild-bimdf", "autoremesher_dir": "",
    "scanopt_python": "", "ui_scale": 1.0, "log_lines": "2000",
}

# GUI setting -> project setting. These go into .bad files; the rest is per machine (settings.json).
PROJECT_KEYS = {
    **{key: f"delight.{key}" for key in ("model", "lit", "normal", "ao", "mask", "output_dir", "output_name", "ai_model",
                                         "force_cpu", "tiled", "tile_size", "tile_overlap", "bit_depth", "apply_mask")},
    **{key: f"scan.{key}" for key in ("scan", "photo", "source", "remesher", "faces", "resolution",
                                      "margin", "ao_samples", "cage", "max_ray", "roughness", "roughness_variation",
                                      "predecimate", "sharp", "keep_work")},
    "scan_out": "scan.out", "scan_tiled": "scan.tiled",
}
PROJECT_FILETYPES = [("Bake & Delight project", f"*{PROJECT_SUFFIX}")]

KEYMAP = [
    ("Ctrl+N / O / S", "new / open / save project"), ("Ctrl+Shift+S", "save project as"),
    ("F5", "run the current tab's stage"), ("F6", "run all stages"), ("1 / 2 / 3", "single / side by side / wipe"),
    ("Space", "single view: before <-> after"), ("W", "wireframe"), ("S", "solid + edges"), ("G", "bake cage"), ("M", "next map"),
    ("C", "UV checker"),
    ("R", "reset camera"), ("A", "auto-rotate"), ("Ctrl+L", "clear log"),
]


class CTkApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


class DelighterGUI(TuiWidgets):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.settings = self._load_settings()
        ctk.set_widget_scaling(float(self.settings.get("ui_scale", 1.0)))
        ctk.set_window_scaling(0.88)

        self.root = CTkApp(fg_color=BG)
        self.root.title("BAKE&DELIGHT")
        self.root.geometry("1440x880")
        self.root.minsize(1100, 700)
        self.font_family = self._pick_font()

        self.var = {}
        for key, default in DEFAULTS.items():
            value = self.settings.get(key, default)
            if (key == "remesher" and value not in REMESHERS) or (key == "ai_model" and value not in models.names()):
                value = default  # renamed option or a model no longer bundled
            if isinstance(default, bool):
                self.var[key] = tk.BooleanVar(value=bool(value))
            elif isinstance(default, float):
                self.var[key] = tk.DoubleVar(value=float(value))
            else:
                self.var[key] = tk.StringVar(value=str(value))
        self.preview_texture = tk.StringVar(value="lit")
        self.vram_preset = tk.StringVar(value="custom")  # follows the settings below; not saved itself

        self.engine = None
        self.last_output_path = None
        self._engine_load_id = 0
        self._busy = {"delight": False, "scan": False}
        self.viewport = None           # main view: single, wipe, or the "after" side
        self.viewport_before = None    # left view in side-by-side mode, same camera
        self.viewport_error = None
        self._scan_before = None       # (high-res scan OBJ, photo) once loaded: the HIGH side of the comparison
        self._uv_layout = None         # PIL image of the low mesh's UV layout, shown on the 3 UV tab
        self._uv_layout_key = None     # (path, mtime) it was drawn from
        self.project = None            # the open .bad project, if any
        self._project_saved = None     # its settings and results as last saved, to detect changes
        self._save_job = None
        self._rotate_last = None
        # Worker threads must not call Tk ("main thread is not in main loop"); they post here instead.
        self._ui_queue = queue.Queue()

        self._build_ui()
        self._set_icon()
        self._drain_ui_queue()
        self._bind_keys()
        for key, variable in self.var.items():
            variable.trace_add("write", lambda *_args: self._schedule_save())
        for key in ("fov", "wireframe", "solid", "show_cage", "cage", "checker", "checker_cells", "texture_filter", "background", "orbit_speed", "zoom_step"):
            self.var[key].trace_add("write", lambda *_args: self._apply_view_settings())
        self.var["autorotate"].trace_add("write", lambda *_args: self._start_autorotate())
        self.var["view_map"].trace_add("write", lambda *_args: self._on_view_map())
        for key in ("force_cpu", "tiled", "scan_tiled", "tile_size"):
            self.var[key].trace_add("write", lambda *_args: self._sync_vram_preset())
        self._sync_vram_preset()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._apply_view_settings()
        self._start_autorotate()
        self._load_engine_async()
        self._refresh_viewport()
        self._load_high()

    # ================================================================ threads

    def _post(self, callback):
        """Run `callback` on the UI thread; safe to call from any thread."""
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
                self._append_log(f"!! ui callback: {error}")
        self.root.after(40, self._drain_ui_queue)

    # ================================================================ logo

    # ================================================================ settings

    @staticmethod
    def _load_settings():
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    def _schedule_save(self):
        if self._save_job is not None:
            self.root.after_cancel(self._save_job)
        self._save_job = self.root.after(600, self._save_settings)

    def _save_settings(self):
        self._save_job = None
        values = {}
        for key, variable in self.var.items():
            try:
                values[key] = variable.get()
            except tk.TclError:  # half-typed number
                values[key] = DEFAULTS[key]
        try:
            os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
            with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
                json.dump(values, handle, indent=2)
        except OSError as error:
            self._append_log(f"!! settings not saved: {error}")

    def _reset_settings(self):
        if not messagebox.askyesno("Reset settings", "Restore every setting to its default?", parent=self.root):
            return
        for key, default in DEFAULTS.items():
            self.var[key].set(default)
        self._append_log("settings reset to defaults")

    def _on_close(self):
        if not self._confirm_discard():
            return
        self._save_settings()
        self.root.destroy()

    # ================================================================ projects (.bad)

    def _project_values(self):
        """The current GUI settings as project settings (section.key -> value)."""
        values = {}
        for key, dotted in PROJECT_KEYS.items():
            value = self._value(key)
            if key == "remesher":
                value = REMESHERS.get(value, "decimate")
            elif key == "tile_size" and value == "auto":
                value = 0
            values[dotted] = value
        return values

    def _project_state(self):
        try:
            return json.dumps([self._project_values(), self.project.results if self.project else {}], sort_keys=True, default=str)
        except tk.TclError:
            return None

    def _confirm_discard(self):
        """True when it is fine to drop the open project: unchanged, saved now, or the user says so."""
        if self.project is None or self._project_state() == self._project_saved:
            return True
        answer = messagebox.askyesnocancel("Project", f"Save changes to {self.project.path.name}?", parent=self.root)
        if answer is None:
            return False
        return self._save_project() if answer else True

    def _new_project(self):
        if not self._confirm_discard():
            return
        for key, dotted in PROJECT_KEYS.items():
            self.var[key].set(DEFAULTS[key])
        self.project, self._project_saved = None, None
        self.last_output_path, self._scan_before = None, None
        self.albedo_radio.configure(state="disabled")
        self.preview_texture.set("lit")
        self._show_project_name()
        self._append_log("project: new (unsaved)")
        self._refresh_viewport()

    def _open_project(self, path=None):
        if not self._confirm_discard():
            return
        path = path or filedialog.askopenfilename(parent=self.root, filetypes=PROJECT_FILETYPES)
        if not path:
            return
        try:
            project = Project.load(path)
        except ProjectError as error:
            messagebox.showerror("Project", str(error), parent=self.root)
            return
        labels = {key: label for label, key in REMESHERS.items()}
        for key, dotted in PROJECT_KEYS.items():
            value = project.get(dotted)
            if key == "remesher":
                value = labels.get(value, next(iter(REMESHERS)))
            elif key == "tile_size":
                value = str(value or "auto")
            elif key in ("faces", "resolution", "margin", "ao_samples", "bit_depth"):
                value = str(value)
            self.var[key].set(value)
        self.project = project
        for warning in project.warnings:
            self._append_log(f"!! {warning}")
        self._show_project_results()
        self._project_saved = self._project_state()
        self._show_project_name()
        self._append_log(f"project: {project.path}")

    def _show_project_results(self):
        """Put a project's last outputs back into the viewport."""
        self.last_output_path, self._scan_before = None, None
        self.albedo_radio.configure(state="disabled")
        self.preview_texture.set("lit")
        scan = self.project.results.get("scan", {})
        albedo = self.project.results.get("delight", {}).get("albedo")
        if scan.get("glb") and os.path.isfile(scan["glb"]):
            self._show_optimized_scan(scan["glb"], self.var["scan"].get() or None, self.var["photo"].get() or None)
        elif albedo and os.path.isfile(albedo):
            self.last_output_path = albedo
            self.albedo_radio.configure(state="normal")
            self.preview_texture.set("albedo")
            self._refresh_viewport()
        else:
            self._refresh_viewport()

    def _save_project(self, choose=False):
        """Save to the open project's file, or ask for one. Returns True when saved."""
        path = self.project.path if self.project and not choose else None
        if path is None:
            initial = os.path.splitext(os.path.basename(self.var["scan"].get() or self.var["lit"].get() or "untitled"))[0]
            path = filedialog.asksaveasfilename(parent=self.root, filetypes=PROJECT_FILETYPES, defaultextension=PROJECT_SUFFIX,
                                                initialfile=initial + PROJECT_SUFFIX)
            if not path:
                return False
        project = self.project or Project(os.path.splitext(os.path.basename(path))[0])
        try:
            project.update(self._project_values())
            project.save(path)
        except (ProjectError, OSError) as error:
            messagebox.showerror("Project", f"Not saved: {error}", parent=self.root)
            return False
        self.project = project
        self._project_saved = self._project_state()
        self._show_project_name()
        self._append_log(f"project saved: {project.path}")
        return True

    def _record_result(self, kind, result):
        """Note a run's outputs in the open project (saved with the project)."""
        if self.project is not None:
            self.project.results[kind] = {**result, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def _show_project_name(self):
        name = os.path.basename(self.project.path) if self.project and self.project.path else "unsaved project"
        self.project_label.configure(text=name)
        self.root.title(f"BAKE&DELIGHT · {name}")

    def _value(self, key, cast=None):
        """A setting's value; numbers typed into an entry fall back to the default when invalid."""
        try:
            value = self.var[key].get()
            return cast(value) if cast else value
        except (tk.TclError, ValueError):
            self._append_log(f"!! {key}: invalid value, using {DEFAULTS[key]}")
            return cast(DEFAULTS[key]) if cast else DEFAULTS[key]

    # ================================================================ styling helpers

    def _path_row(self, tab, label, key, kind="file", filetypes=None, on_change=None, extra=()):
        def build(parent):
            frame = tk.Frame(parent, bg=background_of(parent), bd=0, highlightthickness=0)
            frame.grid_columnconfigure(0, weight=1)
            entry = self._entry(frame, self.var[key])
            entry.grid(row=0, column=0, sticky="ew")
            entry.drop_target_register(DND_FILES)
            entry.dnd_bind("<<Drop>>", lambda event: self._drop_path(event, key, on_change))
            for column, (text, command) in enumerate((("…", lambda: self._browse(key, kind, filetypes, on_change)),
                                                      ("↻", lambda: self._reload_path(key, on_change)),
                                                      ("×", lambda: self._clear_path(key, on_change))), start=1):
                Button(frame, text=text, width=28, fg_color=FIELD, hover_color="#2a2a2a", font=self._tk_font(12),
                       command=command).grid(row=0, column=column, padx=(4, 0))
            for column, (text, command) in enumerate(extra, start=4):
                self._button(frame, text, command).grid(row=0, column=column, padx=(4, 0))
            return frame
        return self._row(tab, label, build)

    def _reload_path(self, key, on_change):
        """Read the file again from disk, e.g. after it was edited in another program."""
        path = self.var[key].get()
        if not path:
            return
        forget(path)
        for view in (self.viewport, self.viewport_before):
            for slot in (view.slots.values() if view else ()):
                slot.texture_path = None  # textures are re-read on the next refresh
        self._append_log(f"{key}: reloaded {os.path.basename(path)}")
        if key == "scan":
            self._scan_before = None
            self._load_high()
        elif on_change and key != "lit":
            on_change(key)
            return
        self._refresh_viewport()

    def _clear_path(self, key, on_change):
        """Unload: empty the field and let the owner react as to any other change."""
        if not self.var[key].get():
            return
        self.var[key].set("")
        if on_change:
            on_change(key)
        else:
            self._append_log(f"{key}: unloaded")

    def _drop_path(self, event, key, on_change):
        files = self.root.tk.splitlist(event.data)
        if files:
            self.var[key].set(files[0].strip("{}"))
            if on_change:
                on_change(key)

    def _browse(self, key, kind, filetypes, on_change):
        if kind == "dir":
            chosen = filedialog.askdirectory(parent=self.root)
        else:
            chosen = filedialog.askopenfilename(parent=self.root, filetypes=filetypes or [("All files", "*.*")])
        if chosen:
            self.var[key].set(chosen)
            if on_change:
                on_change(key)

    # ================================================================ layout

    def _build_ui(self):
        root = self.root
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=0)
        root.grid_rowconfigure(1, weight=1)

        # ---- top status line
        top = ctk.CTkFrame(root, fg_color=PANEL, corner_radius=0, height=30, border_width=0)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.grid_columnconfigure(1, weight=1)
        # The name and mark live in the title bar; the strip shows the workflow and the engine.
        self.top_info = self._label(top, "import → remesh → uv → bake → delight → export", color=HINT, size=12)
        self.top_info.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=4)
        self.engine_label = self._label(top, "engine: --", color=DIM, size=12)
        self.engine_label.grid(row=0, column=2, padx=12)
        self.status_label = self._label(top, "● INIT", color=WARN, size=12, weight="bold")
        self.status_label.grid(row=0, column=3, padx=(0, 12))

        # ---- left: viewport + log
        left = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
        left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        self._build_viewport(left)
        self._build_log(left)

        # ---- right: tabbed sidebar
        self._build_sidebar(root)

    def _build_viewport(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0, border_width=1, border_color=BORDER)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        header.grid_columnconfigure(1, weight=1)
        self._label(header, "┌─ VIEWPORT", color=ACCENT, size=12, weight="bold").grid(row=0, column=0, padx=(0, 10))
        # Fixed width: hint length differs per view mode and would otherwise resize the panels.
        self.view_hint = self._label(header, "", color=DIM, size=11, width=360)
        self.view_hint.grid(row=0, column=1, sticky="w")
        self._segmented(header, self.var["view_mode"], VIEW_MODES, command=lambda _value: self._apply_view_mode()).grid(
            row=0, column=2, padx=(0, 8))
        self._button(header, "reset", self._reset_view).grid(row=0, column=3)

        captions = ctk.CTkFrame(frame, fg_color="transparent", height=18)
        captions.grid(row=1, column=0, sticky="ew", padx=12)
        captions.grid_columnconfigure((0, 1), weight=1, uniform="views")
        self.caption_before = self._label(captions, "◀ BEFORE", color=DIM, size=11, height=18)
        self.caption_after = self._label(captions, "AFTER ▶", color=DIM, size=11, height=18, justify="right")
        self.caption_before.grid(row=0, column=0, sticky="w")
        self.caption_after.grid(row=0, column=1, sticky="e")

        self.views_frame = ctk.CTkFrame(frame, fg_color=BG, corner_radius=0)
        self.views_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(2, 10))
        self.views_frame.grid_rowconfigure(0, weight=1)
        self.views_frame.grid_propagate(False)  # switching between one and two views must not resize the panels
        if OPENGL_AVAILABLE and TextureViewport is not None:
            try:
                camera = Camera()
                self.viewport_before = TextureViewport(self.views_frame, camera=camera, highlightthickness=0, bd=0)
                self.viewport = TextureViewport(self.views_frame, camera=camera, highlightthickness=0, bd=0)
                self.viewport_before.set_mode("a")
                self.viewport_before.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
                self.uv_panel = tk.Label(self.views_frame, bg=BG, bd=0, highlightthickness=0)
                self.uv_panel.bind("<Configure>", lambda _event: self._draw_uv_panel())
                self.viewport.grid(row=0, column=1, sticky="nsew")
            except Exception as error:
                self.viewport_error = str(error)
                self.viewport = self.viewport_before = None
        if self.viewport is None:
            self._label(frame, "!! 3D viewport unavailable - see log", color=WARN).grid(row=2, column=0, sticky="nsew")

    def _build_log(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0, border_width=1, border_color=BORDER, height=170)
        frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        frame.grid_propagate(False)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        self._label(frame, "┌─ LOG", color=ACCENT, size=12, weight="bold").grid(row=0, column=0, sticky="w", padx=10, pady=(6, 2))
        self.console_textbox = ctk.CTkTextbox(frame, font=self._font(12), fg_color=BG, text_color=TEXT, corner_radius=0,
                                              border_width=0, wrap="word", scrollbar_button_color=BORDER)
        self.console_textbox.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.console_textbox.configure(state="disabled")
        if self.viewport_error:
            self._append_log(f"!! 3D viewport initialisation failed: {self.viewport_error}")
        elif self.viewport is not None:
            self._append_log("viewport ready [opengl / unlit]")
        self._append_log(f"settings: {SETTINGS_PATH}")

    def _build_sidebar(self, root):
        side = ctk.CTkFrame(root, fg_color=PANEL, corner_radius=0, border_width=1, border_color=BORDER, width=470)
        side.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(1, weight=1)

        self.tab_var = tk.StringVar(value=TABS["import"])
        tabs = list(TABS.values())
        self._segmented(side, self.tab_var, tabs, command=lambda _value: self._show_tab()).grid(
            row=0, column=0, sticky="ew", padx=8, pady=8)

        self.tabs, self._tab_panels = {}, {}
        for name in tabs:
            panel = ScrollPanel(side)
            panel.grid(row=1, column=0, sticky="nsew", padx=2)
            tab = panel.content
            tab.grid_columnconfigure(1, weight=1)
            tab.next_row = 0
            self.tabs[name], self._tab_panels[name] = tab, panel
        for key, build in (("import", self._build_input_tab), ("remesh", self._build_remesh_tab), ("uv", self._build_uv_tab),
                           ("bake", self._build_bake_tab), ("delight", self._build_delight_tab),
                           ("export", self._build_export_tab), ("view", self._build_view_tab), ("sys", self._build_sys_tab)):
            build(self.tabs[TABS[key]])
        self._show_tab()

        actions = ctk.CTkFrame(side, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=8, pady=8)
        actions.grid_columnconfigure((0, 1), weight=1, uniform="actions")
        self.stage_btn = self._button(actions, "run stage  F5", self._run_current_stage, primary=True)
        self.stage_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.all_btn = self._button(actions, "run all  F6", lambda: self._start_scan("all"), primary=True)
        self.all_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.progress_label = self._label(actions, "", color=DIM, size=11)
        self.progress_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.progress_bar = tk.Canvas(actions, height=3, bg=BORDER, highlightthickness=0, bd=0)
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.progress_bar.bind("<Configure>", lambda _event: self._draw_progress())
        self.progress_bar.grid_remove()
        self._progress = None  # (what, fraction) while a stage runs

    def _set_progress(self, what=None, fraction=0.0):
        """Show how far the running stage is (`what`, 0..1), or hide the bar when what is None."""
        self._progress = None if what is None else (what, fraction)
        if what is None:
            self.progress_label.configure(text="")
            self.progress_bar.grid_remove()
            return
        self.progress_label.configure(text=f"{what}  {round(fraction * 100):3d}%")
        self.progress_bar.grid()
        self._draw_progress()

    def _draw_progress(self):
        bar = self.progress_bar
        bar.delete("all")
        if self._progress:
            bar.create_rectangle(0, 0, round(bar.winfo_width() * self._progress[1]), 3, fill=ACCENT, width=0)

    def _goto_tab(self, name):
        self.tab_var.set(name)
        self._show_tab()

    def _show_tab(self):
        # Only the shown tab is mapped. Keeping all eight mapped and raising one made Windows
        # repaint about 900 widgets on every window resize.
        for name, panel in self._tab_panels.items():
            if name == self.tab_var.get():
                panel.grid()
            else:
                panel.grid_remove()
        self._refresh_run_buttons()
        if getattr(self, "viewport", None):  # the view follows the tab: UV layout on 3 UV, high | low elsewhere
            self._apply_view_settings()
            self._refresh_viewport()

    def _refresh_run_buttons(self):
        """The left button runs the current tab's stage; the right one the whole pipeline."""
        if not hasattr(self, "all_btn"):
            return
        busy = any(self._busy.values())
        stage = STAGE_OF_TAB.get(self.tab_var.get())
        label = f"[ RUN {stage.upper()}  F5 ]" if stage else "[ RUN STAGE  F5 ]"
        self.stage_btn.configure(state="normal" if stage and not busy else "disabled",
                                 text="[ RUNNING… ]" if busy else label)
        self.all_btn.configure(state="disabled" if busy else "normal", text="[ RUN ALL  F6 ]")

    # ---------------------------------------------------------------- tabs

    def _build_input_tab(self, tab):
        images = [("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.exr")]
        self._section(tab, "project")
        self.project_label = self._row(tab, "FILE", lambda parent: self._label(parent, "unsaved project", color=TEXT))

        def project_buttons(parent):
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            for column, (text, command) in enumerate((("new", self._new_project), ("open", self._open_project),
                                                      ("save", self._save_project),
                                                      ("save as", lambda: self._save_project(choose=True)))):
                self._button(frame, text, command).grid(row=0, column=column, padx=(0, 4))
            return frame
        self._row(tab, "", project_buttons, hint=f"{PROJECT_SUFFIX}: inputs, settings, results")
        self._section(tab, "scan")
        self._row(tab, "SOURCE", lambda parent: self._segmented(parent, self.var["source"], ["scan", "pbr"]),
                  hint="scan: de-lit photo · pbr: maps kept as they are")
        self._path_row(tab, "HIGH-RES SCAN", "scan", filetypes=[("Meshes", "*.obj *.ply *.glb *.gltf *.stl")],
                       on_change=self._on_scan_chosen)
        self._path_row(tab, "PHOTO TEXTURE", "photo", filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.exr")])
        self._label(tab, "│   empty = from the material · pbr: base colour", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1
        self._section(tab, "low-res mesh")
        self._path_row(tab, "OBJ", "model", filetypes=[("OBJ mesh", "*.obj")], on_change=self._on_input_changed,
                       extra=[("remesh", lambda: self._goto_tab(TABS["remesh"]))])
        self._section(tab, "maps")
        self._path_row(tab, "LIT RGB", "lit", filetypes=images, on_change=self._on_input_changed)
        self._path_row(tab, "NORMAL (OS)", "normal", filetypes=images)
        self._path_row(tab, "AO", "ao", filetypes=images)
        self._path_row(tab, "MASK", "mask", filetypes=images)
        def map_buttons(parent):
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            self._button(frame, "auto-fill from lit folder",
                         lambda: self._auto_fill_other_maps(self.var["lit"].get(), force=True)).grid(row=0, column=0, padx=(0, 4))
            self._button(frame, "bake", lambda: self._goto_tab(TABS["bake"])).grid(row=0, column=1)
            return frame
        self._row(tab, "", map_buttons, hint="no normal / AO yet? bake them from the scan")
        self._section(tab, "preview texture")

        def radios(parent):
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            for column, (text, value) in enumerate((("before (lit)", "lit"), ("after", "albedo"))):
                button = ctk.CTkRadioButton(frame, text=text, variable=self.preview_texture, value=value, radiobutton_width=14,
                                            radiobutton_height=14, border_width_unchecked=1, border_width_checked=4,
                                            fg_color=ACCENT, border_color=BORDER_HI, hover_color=ACCENT_DIM, text_color=TEXT,
                                            font=self._font(12), command=self._apply_view_mode)
                button.grid(row=0, column=column, padx=(0, 14))
                if value == "albedo":
                    self.albedo_radio = button
                    button.configure(state="disabled")
            return frame
        self._row(tab, "SINGLE VIEW", radios, hint="Space toggles · used by the Single view mode")
        self._row(tab, "", lambda parent: self._button(parent, "2d compare", self._open_preview_window))

    def _build_remesh_tab(self, tab):
        self._section(tab, "remesh")
        self._row(tab, "REMESHER", lambda parent: self._option(parent, self.var["remesher"], list(REMESHERS), width=250))
        self._row(tab, "TARGET FACES", lambda parent: self._entry(parent, self.var["faces"], width=90))
        self._row(tab, "FEATURES", lambda parent: self._check(parent, "keep sharp edges (quads)", self.var["sharp"]))
        self._slider_row(tab, "PRE-DECIMATE", "predecimate", 0, 8, 16, fmt="×{:.1f}")
        self._label(tab, "│   quads: scan reduced to × target faces before remeshing · 0 = off", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1

    def _build_uv_tab(self, tab):
        self._section(tab, "uv")
        self._row(tab, "UNWRAPPER", lambda parent: self._label(parent, "xatlas", color=TEXT, size=12))
        self._row(tab, "MARGIN PX", lambda parent: self._entry(parent, self.var["margin"], width=90))
        self._label(tab, "│   charts are packed for the texture size set in 4 BAKE", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1

    def _build_bake_tab(self, tab):
        self._section(tab, "bake")
        self._row(tab, "TEXTURE SIZE", lambda parent: self._segmented(parent, self.var["resolution"], ["512", "1024", "2048", "4096"]))
        self._row(tab, "AO SAMPLES", lambda parent: self._entry(parent, self.var["ao_samples"], width=90))
        self._slider_row(tab, "CAGE", "cage", 0.001, 0.05, 49, fmt="{:.3f}")
        self._row(tab, "", lambda parent: self._check(parent, "show cage in the view  [G]", self.var["show_cage"]))
        self._slider_row(tab, "MAX RAY", "max_ray", 0.005, 0.15, 29, fmt="{:.3f}")
        self._label(tab, "│   cage / ray: fraction of the scan's bounding diagonal", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1

    def _build_export_tab(self, tab):
        self._section(tab, "output")
        self._path_row(tab, "FOLDER", "scan_out", kind="dir")
        self._label(tab, "│   empty = <scan>_optimized next to the scan", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1
        self._row(tab, "FILES", lambda parent: self._label(parent, "glb (glTF PBR) · obj + mtl (quads) · texture PNGs",
                                                             color=TEXT, size=12))
        self._section(tab, "material")
        self._slider_row(tab, "ROUGHNESS", "roughness", 0.2, 1.0, 80)
        self._slider_row(tab, "VARIATION", "roughness_variation", 0.0, 0.6, 60)
        self._section(tab, "work files")
        self._row(tab, "RUN ALL", lambda parent: self._check(parent, "keep intermediate files", self.var["keep_work"]),
                  hint="single stages always keep <output>/work")

    def _build_delight_tab(self, tab):
        self._section(tab, "model")
        self._row(tab, "AI MODEL", lambda parent: self._option(parent, self.var["ai_model"], models.names(),
                                                               command=lambda _value: self._load_engine_async()),
                  hint="trained for overcast light; bundled with the app")
        self._row(tab, "DEVICE", lambda parent: self._check(parent, "force CPU", self.var["force_cpu"]))
        self._row(tab, "", lambda parent: self._button(parent, "reload engine", self._load_engine_async))
        self._row(tab, "", lambda parent: self._button(parent, "open log folder", lambda: os.startfile(applog.LOG_DIR)),
                  hint="BakeDelightDebug.exe also shows the log in a console")
        self._section(tab, "inference")
        self._row(tab, "VRAM", lambda parent: self._option(parent, self.vram_preset, [*VRAM_PRESETS, "custom"],
                                                           command=self._apply_vram_preset),
                  hint="sets device and tiling · 2048 px tiles need about 9 GB")
        self._row(tab, "TILING", lambda parent: self._check(parent, "tiled (less VRAM)", self.var["tiled"]),
                  hint="only images of 2048 px and up are tiled")
        self._row(tab, "TILE SIZE", lambda parent: self._option(parent, self.var["tile_size"], ["auto", "512", "1024", "2048", "4096"]),
                  hint="auto = half the image")
        self._slider_row(tab, "OVERLAP", "tile_overlap", 0.05, 0.5, 45, fmt="{:.0%}")
        self._row(tab, "SCAN ROUTE", lambda parent: self._check(parent, "tiled inference", self.var["scan_tiled"]))
        self._section(tab, "output (maps only)")
        self._path_row(tab, "DIRECTORY", "output_dir", kind="dir")
        self._row(tab, "FILENAME", lambda parent: self._entry(parent, self.var["output_name"]))
        self._row(tab, "BIT DEPTH", lambda parent: self._segmented(parent, self.var["bit_depth"], ["8", "16"]))
        self._row(tab, "MASK", lambda parent: self._check(parent, "black outside UV islands", self.var["apply_mask"]))
        self._row(tab, "AFTER RUN", lambda parent: self._check(parent, "switch preview to after", self.var["show_after_run"]))

    def _build_view_tab(self, tab):
        self._section(tab, "camera")
        self._row(tab, "MODE", lambda parent: self._segmented(parent, self.var["view_mode"], VIEW_MODES,
                                                               command=lambda _value: self._apply_view_mode()))
        self._slider_row(tab, "FOV", "fov", 20, 120, 100, fmt="{:.0f}°")
        self._slider_row(tab, "ORBIT SPEED", "orbit_speed", 0.1, 1.5, 28, fmt="{:.2f}°/px")
        self._slider_row(tab, "ZOOM STEP", "zoom_step", 1.02, 1.5, 48, fmt="×{:.2f}")
        self._row(tab, "AUTO-ROTATE", lambda parent: self._check(parent, "turntable", self.var["autorotate"]))
        self._slider_row(tab, "SPEED", "autorotate_speed", 2, 90, 88, fmt="{:.0f}°/s")
        self._row(tab, "", lambda parent: self._button(parent, "reset camera", self._reset_view))
        self._section(tab, "shading")
        self._row(tab, "OVERLAY", lambda parent: self._check(parent, "wireframe", self.var["wireframe"]))
        self._row(tab, "SOLID", lambda parent: self._check(parent, "grey with edges, both views", self.var["solid"]))
        self._row(tab, "CAGE", lambda parent: self._check(parent, "show the bake cage", self.var["show_cage"]),
                  hint="distance: CAGE on 4 BAKE")
        self._row(tab, "MAP", lambda parent: self._option(parent, self.var["view_map"], ["auto", *VIEW_MAPS]),
                  hint="auto: per tab · else that map on the low mesh")
        self._row(tab, "TEXTURE", lambda parent: self._check(parent, "UV checker", self.var["checker"]))
        self._row(tab, "CHECKER", lambda parent: self._option(parent, self.var["checker_cells"], ["8", "16", "32", "64", "128"]),
                  hint="cells per UV unit")
        self._row(tab, "FILTER", lambda parent: self._segmented(parent, self.var["texture_filter"], ["mipmap", "linear", "nearest"]))
        self._row(tab, "BACKGROUND", lambda parent: self._option(parent, self.var["background"], list(BACKGROUNDS)))
        self._section(tab, "stats")
        self.stats_label = self._label(tab, "│ --", color=TEXT, size=12, justify="left")
        self.stats_label.grid(row=tab.next_row, column=0, columnspan=2, sticky="w", padx=10)
        tab.next_row += 1

    def _build_sys_tab(self, tab):
        self._section(tab, "tools")
        self._path_row(tab, "QUADWILD DIR", "quadwild_dir", kind="dir")
        self._path_row(tab, "AUTOREMESHER", "autoremesher_dir", kind="dir")
        self._label(tab, "│   empty = the copy shipped with the app", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1
        self._path_row(tab, "SCANOPT PY", "scanopt_python", filetypes=[("python.exe", "*.exe")])
        self._label(tab, "│   empty = training/.venv, else this interpreter", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1
        self._section(tab, "interface")
        self._slider_row(tab, "UI SCALE", "ui_scale", 0.6, 1.4, 16, fmt="{:.2f}")
        self._row(tab, "", lambda parent: self._button(parent, "apply scale", lambda: self._apply_ui_scale(self._value("ui_scale", float))))
        self._row(tab, "LOG LINES", lambda parent: self._entry(parent, self.var["log_lines"], width=90))
        self._row(tab, "", lambda parent: self._button(parent, "clear log", self._clear_log))
        self._section(tab, "settings")
        self._row(tab, "", lambda parent: self._button(parent, "open settings folder",
                                                       lambda: os.startfile(os.path.dirname(SETTINGS_PATH))
                                                       if os.path.isdir(os.path.dirname(SETTINGS_PATH)) else None))
        self._row(tab, "", lambda parent: self._button(parent, "reset all settings", self._reset_settings))
        self._section(tab, "keymap")
        for keys, action in KEYMAP:
            self._label(tab, f"│ {keys:<10} {action}", color=TEXT, size=12).grid(row=tab.next_row, column=0, columnspan=2,
                                                                               sticky="w", padx=10)
            tab.next_row += 1
        self._section(tab, "about")
        for line in ("Texture Delighter - by Vida Marcell", "scan optimiser, retraining, TUI: local fork",
                     "QuadWild-BiMDF GPL-3.0 · AutoRemesher MIT · xatlas MIT"):
            self._label(tab, f"│ {line}", color=DIM, size=11).grid(row=tab.next_row, column=0, columnspan=2, sticky="w", padx=10)
            tab.next_row += 1
        link = self._label(tab, "│ github.com/vdmrcll/ai-texture-delighter", color=ACCENT, size=11, cursor="hand2")
        link.grid(row=tab.next_row, column=0, columnspan=2, sticky="w", padx=10)
        link.bind("<Button-1>", lambda _event: webbrowser.open_new("https://github.com/vdmrcll/ai-texture-delighter"))
        tab.next_row += 1

    # ================================================================ keys

    def _bind_keys(self):
        def guarded(action):
            def handler(event):
                if isinstance(event.widget, (tk.Entry, tk.Text)):
                    return None  # typing in a field
                action()
                return "break"
            return handler

        def set_mode(index):
            self.var["view_mode"].set(VIEW_MODES[index])
            self._apply_view_mode()

        def toggle(key):
            self.var[key].set(not self.var[key].get())

        def flip_single():
            if str(self.albedo_radio.cget("state")) != "disabled":
                self.preview_texture.set("albedo" if self.preview_texture.get() == "lit" else "lit")
                self._apply_view_mode()

        bindings = {"<F5>": self._run_current_stage, "<F6>": lambda: self._start_scan("all"), "1": lambda: set_mode(0),
                    "2": lambda: set_mode(1), "3": lambda: set_mode(2), "<space>": flip_single,
                    "w": lambda: toggle("wireframe"), "s": lambda: toggle("solid"),
                    "g": lambda: toggle("show_cage"), "m": self._next_view_map, "c": lambda: toggle("checker"), "r": self._reset_view,
                    "a": lambda: toggle("autorotate"), "<Control-l>": self._clear_log}
        for sequence, action in bindings.items():
            self.root.bind_all(sequence, guarded(action))
        # Project shortcuts work while typing in a field too.
        for sequence, action in {"<Control-n>": self._new_project, "<Control-o>": self._open_project,
                                 "<Control-s>": self._save_project,
                                 "<Control-S>": lambda: self._save_project(choose=True)}.items():
            self.root.bind_all(sequence, lambda _event, action=action: (action(), "break")[1])

    # ================================================================ engine

    def _load_engine_async(self):
        try:
            model_name = str(models.resolve(self.var["ai_model"].get()))
        except FileNotFoundError as error:
            self._set_status("NO MODEL", ERROR)
            self._append_log(f"!! engine: {error}")
            return
        if self.var["force_cpu"].get():
            os.environ["DELIGHTER_FORCE_CPU"] = "1"
        else:
            os.environ.pop("DELIGHTER_FORCE_CPU", None)
        self._engine_load_id += 1
        load_id = self._engine_load_id
        self.engine = None
        self._set_status("LOADING", WARN)
        self._append_log(f"engine: loading {model_name}")

        def _init():
            try:
                engine = DelighterInferenceEngine(model_name)
                self._post(lambda: self._on_engine_ready(engine, load_id, model_name))
            except Exception as e:
                traceback.print_exc()
                error = str(e)
                self._post(lambda error=error: self._on_engine_error(error, load_id))

        threading.Thread(target=_init, daemon=True).start()

    def _on_engine_ready(self, engine, load_id, model_name):
        if load_id != self._engine_load_id:
            return
        self.engine = engine
        self.engine_label.configure(text=f"engine: {engine.provider_name} · {os.path.basename(model_name)}",
                                    text_color=ACCENT if engine.is_accelerated else WARN)
        self._set_status("READY", ACCENT)
        self._append_log(f"engine ready [{engine.device_info}]")

    def _on_engine_error(self, error, load_id):
        if load_id != self._engine_load_id:
            return
        self._set_status("ENGINE FAILED", ERROR)
        self._append_log(f"!! engine: {error}")

    def _set_status(self, text, color):
        self.status_label.configure(text=f"● {text}", text_color=color)

    def _append_log(self, text, echo=True):
        if echo:  # the engine prints its own lines
            print(f"[App] {text}", flush=True)
        box = self.console_textbox
        box.configure(state="normal")
        box.insert("end", f"{time.strftime('%H:%M:%S')} > {text}\n")
        try:
            limit = max(100, int(self.var["log_lines"].get()))
        except (ValueError, tk.TclError, KeyError, AttributeError):
            limit = 2000
        lines = int(box.index("end-1c").split(".")[0])
        if lines > limit:
            box.delete("1.0", f"{lines - limit}.0")
        box.see("end")
        box.configure(state="disabled")

    def _clear_log(self):
        self.console_textbox.configure(state="normal")
        self.console_textbox.delete("1.0", "end")
        self.console_textbox.configure(state="disabled")

    # ================================================================ inputs

    def _on_input_changed(self, key):
        value = self.var[key].get()
        if key == "model" and value and not value.lower().endswith(".obj"):
            messagebox.showwarning("Model", "The 3D preview supports OBJ meshes only.", parent=self.root)
            return
        self._append_log(f"{key}: {os.path.basename(value) if value else 'unloaded'}")
        if key == "lit":
            self._validate_resolution(self.var["lit"].get())
            self._auto_fill_other_maps(self.var["lit"].get())
        self._refresh_viewport()

    def _on_scan_chosen(self, _key):
        scan = self.var["scan"].get()
        if not scan:  # unloaded: drop views that show it
            self._append_log("scan: unloaded")
            self._scan_before = None
            self._refresh_viewport()
            return
        self._scan_before = None
        self._load_high()
        if not self.var["scan_out"].get():
            self.var["scan_out"].set(os.path.join(os.path.dirname(scan), os.path.splitext(os.path.basename(scan))[0] + "_optimized"))

    def _validate_resolution(self, filepath):
        try:
            with Image.open(filepath) as img:
                w, h = img.size
                if w != 2048 or h != 2048:
                    self._append_log(f"warn: {os.path.basename(filepath)} is {w}x{h} (model trained at 2048x2048)")
        except Exception:
            pass

    def _auto_fill_other_maps(self, lit_path, force=False):
        if not lit_path or not os.path.isfile(lit_path):
            return
        folder = os.path.dirname(lit_path)
        if not self.var["output_dir"].get() or force:
            self.var["output_dir"].set(folder)
        for file in sorted(os.listdir(folder)):
            lower, full_path = file.lower(), os.path.join(folder, file)
            for key, token in (("normal", "normal"), ("ao", "ao"), ("mask", "mask")):
                if token in lower and (force or not self.var[key].get()) and full_path != lit_path:
                    self.var[key].set(full_path)
                    self._append_log(f"matched {key}: {file}")
                    break

    def _validate_inputs(self, paths):
        sizes = []
        for label, path in paths:
            if not os.path.isfile(path):
                raise ValueError(f"{label} does not exist: {path}")
            try:
                with Image.open(path) as img:
                    size = img.size
            except (OSError, ValueError):
                # Pillow may not have an EXR plugin; OpenCV can still read the dimensions.
                image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"Unable to read {label}: {path}")
                size = (image.shape[1], image.shape[0])
            sizes.append((label, size))
        expected = sizes[0][1]
        if any(size != expected for _, size in sizes):
            details = ", ".join(f"{label}={size[0]}x{size[1]}" for label, size in sizes)
            raise ValueError(f"All maps must have matching dimensions ({details})")

    # ================================================================ de-light

    def _apply_vram_preset(self, name):
        preset = VRAM_PRESETS.get(name)
        if not preset:  # "custom" only describes the settings; picking it changes nothing
            self._sync_vram_preset()
            return
        device_changed = bool(self.var["force_cpu"].get()) != preset["force_cpu"]
        for key, value in preset.items():
            self.var[key].set(value)
        self._append_log(f"vram preset {name}: tiles of {preset['tile_size']} px" + (", CPU" if preset["force_cpu"] else ""))
        if device_changed:
            self._load_engine_async()

    def _sync_vram_preset(self):
        current = {key: self.var[key].get() for key in ("force_cpu", "tiled", "scan_tiled", "tile_size")}
        self.vram_preset.set(next((name for name, preset in VRAM_PRESETS.items() if preset == current), "custom"))

    def _start_processing(self):
        if self._busy["delight"]:
            return
        lit, norm, ao, mask = (self.var[key].get() for key in ("lit", "normal", "ao", "mask"))
        if not all([lit, norm, ao, mask]):
            messagebox.showwarning("De-light", "All four input maps must be set (1 IMPORT tab).", parent=self.root)
            return
        try:
            self._validate_inputs([("Lit map", lit), ("Normal map", norm), ("AO map", ao), ("Mask", mask)])
        except (OSError, ValueError) as e:
            messagebox.showwarning("De-light", str(e), parent=self.root)
            return
        if not self.engine:
            messagebox.showerror("De-light", "Inference engine is not initialised.", parent=self.root)
            return

        out_dir = self.var["output_dir"].get() or os.path.dirname(lit)
        output_path = os.path.join(out_dir, self.var["output_name"].get().strip() or DEFAULTS["output_name"])
        tile = self.var["tile_size"].get()
        options = dict(tiled_mode=self.var["tiled"].get(), tile_size=0 if tile == "auto" else int(tile),
                       tile_overlap=self._value("tile_overlap", float), bit_depth=int(self.var["bit_depth"].get()),
                       apply_mask=self.var["apply_mask"].get())

        self._busy["delight"] = True
        self._refresh_run_buttons()
        self._set_status("DE-LIGHTING", TEXT_HI)
        self._append_log(f"de-light → {output_path}  {options}")
        self._set_progress(TABS["delight"], 0.0)

        def log_ui(msg):
            self._post(lambda: self._append_log(msg, echo=False))

        def progress_ui(fraction):
            self._post(lambda: self._set_progress(TABS["delight"], fraction))

        def _work():
            started = time.time()
            try:
                out_file, dims, dev = self.engine.process_texture(lit, norm, ao, mask, output_path, log_callback=log_ui,
                                                                   progress_callback=progress_ui, **options)
                result = {"albedo": out_file, "size": list(dims), "device": dev, "seconds": round(time.time() - started, 1)}
                self._post(lambda: (self._record_result("delight", result), self._on_success(out_file, time.time() - started)))
            except Exception as e:
                traceback.print_exc()
                message = str(e)
                self._post(lambda: self._on_error(message))

        threading.Thread(target=_work, daemon=True).start()

    def _on_success(self, out_file, seconds):
        self._busy["delight"] = False
        self._set_progress()
        self._refresh_run_buttons()
        self.albedo_radio.configure(state="normal")
        self.last_output_path = out_file
        self._set_status("DONE", ACCENT)
        self._append_log(f"done in {seconds:.1f} s: {out_file}")
        if self.var["show_after_run"].get():
            self.preview_texture.set("albedo")
        self._refresh_viewport()

    def _on_error(self, err_msg):
        self._busy["delight"] = False
        self._set_progress()
        self._refresh_run_buttons()
        self._set_status("FAILED", ERROR)
        self._append_log(f"!! {err_msg}")
        messagebox.showerror("De-light", f"Inference failed:\n{err_msg}", parent=self.root)

    # ================================================================ scan optimiser

    def _scanopt_command(self):
        """How to start scanopt: the packaged BakeDelightJob.exe next to the app, else a Python with
        OpenCV EXR support (the training environment when there is one)."""
        if getattr(sys, "frozen", False):
            return [os.path.join(os.path.dirname(sys.executable), "BakeDelightJob.exe")]
        chosen = self.var["scanopt_python"].get().strip()
        candidate = os.path.join(APP_DIR, "training", ".venv", "Scripts", "python.exe")
        python = chosen or os.environ.get("SCANOPT_PYTHON") or (candidate if os.path.isfile(candidate) else sys.executable)
        return [python, "-u", SCANOPT_SCRIPT]

    def _scan_out(self):
        scan = self.var["scan"].get().strip()
        return self.var["scan_out"].get().strip() or os.path.join(os.path.dirname(scan), os.path.splitext(os.path.basename(scan))[0] + "_optimized")

    def _run_current_stage(self):
        stage = STAGE_OF_TAB.get(self.tab_var.get())
        if stage is None or any(self._busy.values()):
            return
        scan_ready = os.path.isfile(self.var["scan"].get().strip()) and os.path.isfile(os.path.join(self._scan_out(), "work", "lit.png"))
        if stage == "delight" and not scan_ready:
            self._append_log("de-light: no baked scan here, using the maps set in 1 IMPORT")
            self._start_processing()  # no baked scan: de-light the maps set in 1 IMPORT
        else:
            self._start_scan(stage)

    def _start_scan(self, stage="all"):
        if any(self._busy.values()):
            return
        scan = self.var["scan"].get().strip()
        if not os.path.isfile(scan):
            self._goto_tab(TABS["import"])
            messagebox.showwarning("Scan optimiser", "Choose a high-res scan file (1 IMPORT tab).", parent=self.root)
            return
        try:
            faces = int(self.var["faces"].get())
            numbers = {key: int(self.var[key].get()) for key in ("margin", "ao_samples")}
        except ValueError:
            messagebox.showwarning("Scan optimiser", "Faces, margin and AO samples must be whole numbers.", parent=self.root)
            return
        out = self._scan_out()
        photo = self.var["photo"].get().strip()
        remesher = self.var["remesher"].get()
        command = [*self._scanopt_command(), scan, "--out", out, "--source", self.var["source"].get(), "--faces", str(faces),
                   "--remesher", REMESHERS[remesher], "--resolution", self.var["resolution"].get(),
                   "--margin", str(numbers["margin"]), "--ao-samples", str(numbers["ao_samples"]),
                   "--cage", f"{self._value('cage', float):.4f}", "--max-ray", f"{self._value('max_ray', float):.4f}",
                   "--roughness", f"{self._value('roughness', float):.3f}",
                   "--roughness-variation", f"{self._value('roughness_variation', float):.3f}",
                   "--predecimate", f"{self._value('predecimate', float):.1f}",
                   "--quadwild", self.var["quadwild_dir"].get(), "--model", self.var["ai_model"].get(), "--stage", stage]
        if self.var["autoremesher_dir"].get().strip():
            command += ["--autoremesher", self.var["autoremesher_dir"].get().strip()]
        if photo:
            command += ["--texture", photo]
        for key, flag in (("sharp", "--sharp"), ("scan_tiled", "--tiled"), ("keep_work", "--keep-work")):
            if self.var[key].get():
                command.append(flag)
        if self.var["scan_tiled"].get() and self.var["tile_size"].get() != "auto":
            command += ["--tile-size", self.var["tile_size"].get()]
        env = dict(os.environ)
        self._busy["scan"] = True
        self._refresh_run_buttons()
        self._set_status(stage.upper(), TEXT_HI)
        self._append_log(f"{stage}: {os.path.basename(scan)} → {faces} faces ({REMESHERS[remesher]})")
        threading.Thread(target=self._run_scan_optimizer, args=(command, env, (scan, photo or None), stage, out), daemon=True).start()

    def _run_scan_optimizer(self, command, env, source, stage="all", out=""):
        glb_path, tail, started = None, [], time.time()
        current = TABS.get(stage, "")  # the tab label of the stage now running
        decile = -1
        print(f"[App] job: {subprocess.list2cmdline(command)}", flush=True)
        print(f"[App] job DELIGHTER_FORCE_CPU={env.get('DELIGHTER_FORCE_CPU', '')!r}", flush=True)
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                       env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in process.stdout:
                line = applog.clean(line).rstrip()
                if line.startswith("@progress "):
                    try:
                        fraction = float(line.split()[1])
                    except (IndexError, ValueError):
                        continue
                    self._post(lambda what=current, fraction=fraction: self._set_progress(what, fraction))
                    if int(fraction * 10) != decile:  # the console gets every 10 %
                        decile = int(fraction * 10)
                        print(f"[Job] {current.strip()}  {fraction:.0%}", flush=True)
                    continue
                print(f"[Job] {line}", flush=True)
                tail = (tail + [line])[-12:]
                if line.startswith("[") and "/5]" in line[:6]:
                    number = int(line[1])
                    current = TABS[STAGES[number - 1]] + (f"  ·  {number}/5" if stage == "all" else "")
                    self._post(lambda text=line, what=current: (self._append_log(f"  {text}", echo=False), self._set_progress(what, 0.0)))
                elif line.startswith("done: "):
                    glb_path = line[6:].split("  (")[0].strip()
            code = process.wait()
            print(f"[App] job exit code {code} ({code & 0xFFFFFFFF:#x}) after {time.time() - started:.1f} s", flush=True)
        except OSError as error:
            traceback.print_exc()
            code, tail = -1, [str(error)]

        def finish():
            self._busy["scan"] = False
            self._refresh_run_buttons()
            self._set_progress()
            if code == 0 and stage not in ("all", "export"):
                self._set_status("DONE", ACCENT)
                self._append_log(f"{stage} done in {time.time() - started:.0f} s")
                self._show_stage_result(stage, os.path.join(out, "work"))
            elif code == 0 and glb_path:
                self._set_status("DONE", ACCENT)
                self._append_log(f"scanopt done in {time.time() - started:.0f} s: {glb_path}")
                stem = os.path.splitext(glb_path)[0]
                self._record_result("scan", {"glb": glb_path, "obj": stem + ".obj", "report": stem + "_report.json",
                                             "textures": os.path.join(os.path.dirname(glb_path), "textures"),
                                             "seconds": round(time.time() - started, 1)})
                self._show_optimized_scan(glb_path, *source)
            else:
                self._set_status("SCANOPT FAILED", ERROR)
                self._append_log("!! scanopt failed:\n" + "\n".join(tail[-6:]))

        self._post(finish)

    def _show_stage_result(self, stage, work):
        """Load what a single stage produced; the open tab decides how it is shown."""
        path = lambda name: os.path.join(work, name)
        if stage == "remesh":
            self.var["model"].set(path("low_preview.obj"))
        elif stage == "uv":
            self.var["model"].set(path("low_uv.obj"))
        elif stage == "bake":
            self.var["model"].set(path("low_uv.obj"))
            for key, name in (("lit", "lit.png"), ("normal", "normal_os.png"), ("ao", "ao.png"), ("mask", "mask.png")):
                self.var[key].set(path(name))
        else:
            self.last_output_path = path("albedo_raw.png")
            self.albedo_radio.configure(state="normal")
        self._refresh_viewport()
        if not self._scan_before:
            self._load_high()

    def _load_high(self):
        scan = self.var["scan"].get().strip()
        if scan and os.path.isfile(scan) and (self._scan_before or (None,))[0] != scan:
            self._load_scan_before(scan, self.var["photo"].get().strip())

    def _tab(self):
        return next((key for key, label in TABS.items() if label == self.tab_var.get()), "import")

    def _render_uv_layout(self, obj_path, size=1024):
        """The low mesh's UV charts on the unit square: filled triangles with light edges."""
        from PIL import ImageDraw
        uv = load_obj(obj_path).vertex_data[:, 3:5].reshape(-1, 3, 2)
        image = Image.new("RGB", (size, size), (24, 24, 24))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, size - 1, size - 1), outline=(60, 60, 60))
        points = np.column_stack([uv[..., 0].ravel() * (size - 1), (1 - uv[..., 1].ravel()) * (size - 1)]).reshape(-1, 3, 2)
        for triangle in points:
            draw.polygon([tuple(corner) for corner in triangle], fill=(46, 46, 46), outline=(150, 150, 150))
        return image

    def _draw_uv_panel(self):
        if self._uv_layout is None:
            return
        from PIL import ImageTk
        side = max(16, min(self.uv_panel.winfo_width(), self.uv_panel.winfo_height()) - 8)
        self._uv_photo = ImageTk.PhotoImage(self._uv_layout.resize((side, side), Image.Resampling.LANCZOS))
        self.uv_panel.configure(image=self._uv_photo)

    def _show_optimized_scan(self, glb_path, scan=None, photo=None):
        """Load the optimised OBJ into the 3D preview: the baked photo as before, the
        de-lighted albedo as after. The original scan replaces the before view once loaded."""
        stem = os.path.splitext(glb_path)[0]
        obj_path = stem + ".obj"
        textures = os.path.join(os.path.dirname(glb_path), "textures", os.path.basename(stem))
        if os.path.isfile(obj_path):
            self.var["model"].set(obj_path)
        if os.path.isfile(textures + "_lit.png"):
            self.var["lit"].set(textures + "_lit.png")
        if os.path.isfile(textures + "_albedo.png"):
            self.last_output_path = textures + "_albedo.png"
            self.albedo_radio.configure(state="normal")
            self.preview_texture.set("albedo")
        self._scan_before = None
        self._refresh_viewport()
        if scan:
            self._load_scan_before(scan, photo)

    def _load_scan_before(self, scan, photo):
        """Parse the original high-res scan off the UI thread, then use it as the before view."""
        if not (scan.lower().endswith(".obj") and photo and os.path.isfile(photo)):
            return
        if os.path.getsize(scan) > MAX_BEFORE_SCAN_BYTES:
            self._append_log("before view: original scan too large to preview; showing the baked photo")
            return
        self._append_log("before view: loading the original scan…")

        def work():
            try:
                load_obj(scan)
                self._post(lambda: self._use_scan_before(scan, photo))
            except Exception as error:
                message = str(error)
                self._post(lambda: self._append_log(f"before view: original scan not usable ({message}); showing the baked photo"))

        threading.Thread(target=work, daemon=True).start()

    def _use_scan_before(self, scan, photo):
        self._scan_before = (scan, photo)
        self._append_log("before view: original scan with its photo texture")
        self._refresh_viewport()

    # ================================================================ viewport

    def _apply_view_settings(self):
        if not self.viewport:
            return
        camera = self.viewport.camera
        try:
            camera.fov = float(self.var["fov"].get())
            camera.orbit_speed = float(self.var["orbit_speed"].get())
            camera.zoom_step = float(self.var["zoom_step"].get())
            camera.checker_cells = int(self.var["checker_cells"].get())
        except (tk.TclError, ValueError):
            pass
        uv_tab = self._tab() == "uv"  # the UV tab always shows the checker and the edges
        camera.wireframe = bool(self.var["wireframe"].get()) or uv_tab
        camera.solid = bool(self.var["solid"].get()) and not uv_tab
        camera.cage = self._value("cage", float) if self.var["show_cage"].get() else None
        camera.checker = bool(self.var["checker"].get()) or uv_tab
        camera.texture_filter = self.var["texture_filter"].get()
        camera.background = BACKGROUNDS.get(self.var["background"].get(), BACKGROUNDS["Charcoal"])
        camera.changed()

    def _start_autorotate(self):
        if self.var["autorotate"].get() and self._rotate_last is None:
            self._rotate_last = time.time()
            self.root.after(33, self._autorotate_tick)

    def _autorotate_tick(self):
        if not self.var["autorotate"].get() or not self.viewport:
            self._rotate_last = None
            return
        now = time.time()
        camera = self.viewport.camera
        camera.yaw += self._value("autorotate_speed", float) * (now - self._rotate_last)
        self._rotate_last = now
        camera.changed()
        self.root.after(33, self._autorotate_tick)

    def _reset_view(self):
        if self.viewport:
            self.viewport.reset_view()

    def _apply_view_mode(self):
        """Single: one view, before or after per the radio. Side by side: two views on one
        camera. A/B wipe: one view split by a draggable line, before on the left."""
        mode = VIEW_MODES[1] if self._tab() == "uv" else self.var["view_mode"].get()  # UV: layout | model
        if mode not in VIEW_MODES:
            mode = VIEW_MODES[0]
        side_by_side = mode == VIEW_MODES[1]
        hints = {VIEW_MODES[0]: "L orbit · M pan · R/wheel zoom · space: before/after",
                 VIEW_MODES[1]: "linked cameras · L orbit · M pan · R zoom",
                 VIEW_MODES[2]: "drag the line: compare · L orbit · M pan · R zoom"}
        self.view_hint.configure(text=hints[mode])
        if mode == VIEW_MODES[0]:
            self.caption_before.grid_remove()
            self.caption_after.grid_remove()
        else:
            self.caption_before.grid()
            self.caption_after.grid()
        if not self.viewport:
            return
        if side_by_side:
            self.views_frame.grid_columnconfigure(0, weight=1, uniform="views")
            self.views_frame.grid_columnconfigure(1, weight=1, uniform="views")
            if self._tab() == "uv" and self._uv_layout is not None:
                self.viewport_before.grid_remove()
                self.uv_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
                self.root.after_idle(self._draw_uv_panel)  # also when the panel keeps its size
            else:
                self.uv_panel.grid_remove()
                self.viewport_before.grid()
            self.viewport.set_mode("b")
        else:
            self.viewport_before.grid_remove()
            self.uv_panel.grid_remove()
            self.views_frame.grid_columnconfigure(0, weight=0, uniform="")
            self.views_frame.grid_columnconfigure(1, weight=1, uniform="")
            self.viewport.set_mode("wipe" if mode == VIEW_MODES[2] else "a" if self.preview_texture.get() == "lit" else "b")

    def _comparison(self):
        """(mesh, texture, shading) for the left and right slots. 3 UV: nothing left (the UV layout
        panel takes its place) and the model right. Every other tab: the high-res scan with its photo
        left, the low mesh right - grey with wireframe on 1 IMPORT and 2 REMESH, with the baked photo
        on 4 BAKE, with the albedo (or the photo) from 5 DELIGHT on."""
        model, lit = self.var["model"].get(), self.var["lit"].get()
        tab = self._tab()
        if tab == "uv":
            return (None, None, None), (model, None, None)
        before = (*self._scan_before, None) if self._scan_before else (None, None, None)
        if tab in ("import", "remesh"):
            texture = None
        elif tab == "bake":
            texture = lit
        else:
            texture = self.last_output_path if self.last_output_path and os.path.isfile(self.last_output_path) else lit
        texture = texture if texture and os.path.isfile(texture) else None
        chosen = self.var["view_map"].get()
        if chosen != "auto" and os.path.basename(model) != "low_preview.obj":  # the remesh preview has no UVs
            texture = self._map_path(chosen) or texture
        return before, (model, texture, None if texture else "flat")

    def _map_path(self, kind):
        """The newest file of one map kind: from the stage work folder, the exported textures, or the
        maps set on 1 IMPORT. None when there is none yet."""
        out, scan = self._scan_out(), self.var["scan"].get().strip()
        name = os.path.splitext(os.path.basename(scan))[0]
        work, textures = os.path.join(out, "work"), os.path.join(out, "textures")
        candidates = [os.path.join(work, file) for file in VIEW_MAPS[kind][0]]
        candidates += [os.path.join(textures, f"{name}_{suffix}.png") for suffix in VIEW_MAPS[kind][1]]
        candidates += [self.var[key].get() for key in VIEW_MAPS[kind][2]]
        if kind == "albedo" and self.last_output_path:
            candidates.append(self.last_output_path)
        found = [path for path in candidates if path and os.path.isfile(path)]
        return max(found, key=os.path.getmtime) if found else None

    def _on_view_map(self):
        chosen = self.var["view_map"].get()
        if chosen != "auto" and not self._map_path(chosen):
            self._append_log(f"view: no {chosen} map yet")
        elif chosen != "auto":
            self.preview_texture.set("albedo")  # single view: the low mesh, which wears the map
        self._refresh_viewport()

    def _next_view_map(self):
        names = ["auto", *VIEW_MAPS]
        current = self.var["view_map"].get()
        self.var["view_map"].set(names[(names.index(current) + 1) % len(names)] if current in names else "auto")

    def _update_uv_layout(self):
        model = self.var["model"].get()
        key = (model, os.path.getmtime(model)) if model and os.path.isfile(model) else None
        if key == self._uv_layout_key:
            return
        self._uv_layout_key, self._uv_layout = key, None
        if key:
            try:
                self._uv_layout = self._render_uv_layout(model)
            except Exception as error:
                self._append_log(f"uv layout: {error}")

    def _refresh_viewport(self):
        if not self.viewport:
            return
        before, after = self._comparison()
        if self._tab() == "uv":
            self._update_uv_layout()
        try:
            for view in (self.viewport, self.viewport_before):
                view.set_slot("a", *before)
                view.set_slot("b", *after)
        except Exception as error:
            self._append_log(f"!! 3D preview: {error}")
        self._apply_view_mode()
        self._update_stats(before, after)

    def _update_stats(self, before, after):
        slots = self.viewport.slots

        def describe(name, slot, pair):
            mesh = os.path.basename(pair[0]) if pair[0] else "-"
            texture = os.path.basename(pair[1]) if pair[1] else "-"
            return f"│ {name:<6} {slot.triangles:>9,} tris  {mesh}\n│        {texture}"

        self.stats_label.configure(text=describe("BEFORE", slots["a"], before) + "\n" + describe("AFTER", slots["b"], after))
        if self._tab() == "uv":
            self.caption_before.configure(text="◀ UV LAYOUT")
            self.caption_after.configure(text=f"MODEL · {slots['b'].triangles:,} tris ▶")
            return
        left = f"HIGH · {slots['a'].triangles:,} tris" if self._scan_before else "HIGH · no scan loaded"
        self.caption_before.configure(text=f"◀ {left}")
        right = "LOW"
        self.caption_after.configure(text=f"{right} · {slots['b'].triangles:,} tris ▶")

    # ================================================================ 2D compare

    def _open_preview_window(self):
        if not self.last_output_path or not os.path.exists(self.last_output_path):
            messagebox.showwarning("2D compare", "No generated output yet.", parent=self.root)
            return
        lit_path = self.var["lit"].get()
        if not lit_path or not os.path.exists(lit_path):
            messagebox.showwarning("2D compare", "The lit map is missing.", parent=self.root)
            return

        window = ctk.CTkToplevel(self.root, fg_color=BG)
        window.title("2d compare")
        window.geometry("700x600")
        window.minsize(700, 600)
        try:
            lit_img = Image.open(lit_path).convert("RGB")
            out_img = Image.open(self.last_output_path).convert("RGB").resize(lit_img.size)
        except Exception as e:
            messagebox.showerror("2D compare", f"Failed to load images:\n{e}", parent=window)
            window.destroy()
            return
        lit_img.thumbnail((460, 460), Image.Resampling.LANCZOS)
        out_img.thumbnail((460, 460), Image.Resampling.LANCZOS)

        image_label = ctk.CTkLabel(window, text="")
        image_label.pack(padx=20, pady=(20, 8), expand=True)
        info = self._label(window, "", color=DIM, size=12)
        info.pack(pady=(0, 4))

        def update_blend(value):
            ratio = float(value) / 100.0
            blended = Image.blend(lit_img, out_img, alpha=ratio)
            image = ctk.CTkImage(light_image=blended, dark_image=blended, size=blended.size)
            image_label.configure(image=image)
            image_label.image = image
            info.configure(text=f"lit {100 - ratio * 100:3.0f}%  ·  albedo {ratio * 100:3.0f}%")

        ctk.CTkSlider(window, from_=0, to=100, command=update_blend, width=500, corner_radius=0, button_corner_radius=0,
                      fg_color=BORDER, progress_color=ACCENT_DIM, button_color=ACCENT).pack(pady=(0, 20))
        update_blend(100)

    def run(self):
        self.root.mainloop()


def main():
    print(f"[App] log file: {LOG_PATH or 'none'}", flush=True)
    app = DelighterGUI()
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(PROJECT_SUFFIX):
        app.root.after(200, lambda: app._open_project(sys.argv[1]))  # double-clicked .bad file
    app.run()


if __name__ == "__main__":
    main()

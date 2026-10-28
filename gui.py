import json
import queue
import os
import subprocess
import sys
import threading
import time
import webbrowser
import cv2
import customtkinter as ctk
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image

from engine import DelighterInferenceEngine
try:
    from viewport import OPENGL_AVAILABLE, Camera, TextureViewport, load_obj
except Exception:
    OPENGL_AVAILABLE = False
    TextureViewport = None


MODEL_FILENAME = "delighter_model_fp16.onnx"
MODEL_LABEL = "FP16"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SCANOPT_SCRIPT = os.path.join(APP_DIR, "scanopt", "scanopt.py")
SETTINGS_PATH = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "TextureDelighter", "settings.json")
# Mirrors scanopt.REMESHERS; kept here so the GUI starts even if scanopt's dependencies are missing.
REMESHERS = {
    "Triangles - decimation (Blender QEM)": "decimate",
    "Quads - AutoRemesher (MIT)": "autoremesher",
    "Quads - QuadWild + Bi-MDF (GPL)": "quadwild",
}
VIEW_MODES = ["Single", "Side by side", "A/B wipe"]
# Original scans above this size are not parsed for the before view; the baked photo is used instead.
MAX_BEFORE_SCAN_BYTES = 400 * 1024 * 1024
BACKGROUNDS = {"Void": (0.0, 0.0, 0.0), "Charcoal": (0.04, 0.04, 0.04), "Graphite": (0.16, 0.16, 0.16),
               "Mid grey": (0.42, 0.42, 0.42), "Paper": (0.86, 0.86, 0.86)}

# TUI palette: greyscale only. State is shown by brightness and by markers (●, !!), never by hue.
BG = "#000000"
PANEL = "#060606"
FIELD = "#0d0d0d"
BORDER = "#262626"
BORDER_HI = "#555555"
HINT = "#4a4a4a"
TEXT = "#c8c8c8"
TEXT_HI = "#ffffff"
DIM = "#6e6e6e"
ACCENT = "#ffffff"       # primary buttons, section rules, active marks
ACCENT_DIM = "#2e2e2e"   # selected segment, slider fill
WARN = "#9a9a9a"
ERROR = "#ffffff"        # errors read by their "!!" / FAILED text, in full white

# Every persisted setting: key -> default. The type of the default picks the Tk variable.
DEFAULTS = {
    # input
    "model": "", "lit": "", "normal": "", "ao": "", "mask": "", "output_dir": "",
    # de-light
    "onnx_model": "", "force_cpu": False, "tiled": True, "tile_size": "auto", "tile_overlap": 0.25,
    "bit_depth": "8", "apply_mask": True, "output_name": "delighted_albedo.png", "show_after_run": True,
    # scan optimiser
    "scan": "", "photo": "", "scan_out": "", "remesher": next(iter(REMESHERS)), "faces": "10000",
    "resolution": "2048", "unwrap": "xatlas", "smart_angle": 66.0, "margin": "16", "ao_samples": "64",
    "cage": 0.01, "max_ray": 0.03, "roughness": 0.8, "roughness_variation": 0.25,
    "sharp": False, "scan_tiled": False, "keep_work": False,
    # view
    "view_mode": VIEW_MODES[0], "fov": 50.0, "wireframe": False, "checker": False, "checker_cells": "32",
    "texture_filter": "mipmap", "background": "Charcoal", "orbit_speed": 0.5, "zoom_step": 1.14,
    "autorotate": False, "autorotate_speed": 20.0,
    # system
    "blender": r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    "quadwild_dir": r"F:\DelighterData\tools\quadwild-bimdf", "autoremesher_dir": r"F:\DelighterData\tools\autoremesher",
    "xatlas_site": r"F:\DelighterData\tools\blender-site", "scanopt_python": "", "ui_scale": 1.0, "log_lines": "2000",
}

KEYMAP = [
    ("F5", "run de-light"), ("F6", "run scan optimiser"), ("1 / 2 / 3", "single / side by side / wipe"),
    ("Space", "single view: before <-> after"), ("W", "wireframe"), ("C", "UV checker"),
    ("R", "reset camera"), ("A", "auto-rotate"), ("Ctrl+L", "clear log"),
]


class CTkApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


class DelighterGUI:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.settings = self._load_settings()
        ctk.set_widget_scaling(float(self.settings.get("ui_scale", 1.0)))
        ctk.set_window_scaling(0.88)

        self.root = CTkApp(fg_color=BG)
        self.root.title("delighter")
        self.root.geometry("1440x880")
        self.root.minsize(1100, 700)
        self.font_family = self._pick_font()

        self.var = {}
        for key, default in DEFAULTS.items():
            value = self.settings.get(key, default)
            if isinstance(default, bool):
                self.var[key] = tk.BooleanVar(value=bool(value))
            elif isinstance(default, float):
                self.var[key] = tk.DoubleVar(value=float(value))
            else:
                self.var[key] = tk.StringVar(value=str(value))
        self.preview_texture = tk.StringVar(value="lit")

        self.engine = None
        self.last_output_path = None
        self._engine_load_id = 0
        self._busy = {"delight": False, "scan": False}
        self.viewport = None           # main view: single, wipe, or the "after" side
        self.viewport_before = None    # left view in side-by-side mode, same camera
        self.viewport_error = None
        self._scan_before = None       # (original scan OBJ, photo) once an optimised scan is shown
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
        for key in ("fov", "wireframe", "checker", "checker_cells", "texture_filter", "background", "orbit_speed", "zoom_step"):
            self.var[key].trace_add("write", lambda *_args: self._apply_view_settings())
        self.var["autorotate"].trace_add("write", lambda *_args: self._start_autorotate())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._apply_view_settings()
        self._start_autorotate()
        self._load_engine_async()
        self._refresh_viewport()

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

    @staticmethod
    def _logo_image(size=64):
        """The mark: a half-lit disc, the lit side white and the shadow side outlined only."""
        from PIL import ImageDraw
        scale = 4  # draw large, downsample for clean edges
        canvas = size * scale
        image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 255))
        draw = ImageDraw.Draw(image)
        inset, stroke = canvas * 0.14, max(2, canvas // 22)
        box = (inset, inset, canvas - inset, canvas - inset)
        draw.ellipse(box, outline=(255, 255, 255, 255), width=stroke)
        draw.pieslice(box, 90, 270, fill=(255, 255, 255, 255))
        return image.resize((size, size), Image.Resampling.LANCZOS)

    def _set_icon(self):
        from PIL import ImageTk
        self._icon = ImageTk.PhotoImage(self._logo_image(64), master=self.root)
        # CustomTkinter installs its own icon shortly after start; set ours after it.
        self.root.after(300, lambda: self.root.iconphoto(True, self._icon))

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
        self._save_settings()
        self.root.destroy()

    def _value(self, key, cast=None):
        """A setting's value; numbers typed into an entry fall back to the default when invalid."""
        try:
            value = self.var[key].get()
            return cast(value) if cast else value
        except (tk.TclError, ValueError):
            self._append_log(f"!! {key}: invalid value, using {DEFAULTS[key]}")
            return cast(DEFAULTS[key]) if cast else DEFAULTS[key]

    # ================================================================ styling helpers

    def _pick_font(self):
        families = set(tkfont.families(self.root))
        return next((name for name in ("JetBrains Mono", "Cascadia Mono", "Consolas") if name in families), "Courier")

    def _light_family(self):
        """The light cut of the UI font, for the quiet half of the wordmark."""
        light = f"{self.font_family} Light"
        return light if light in set(tkfont.families(self.root)) else self.font_family

    def _font(self, size=12, weight="normal"):
        return ctk.CTkFont(family=self.font_family, size=size, weight=weight)

    def _label(self, parent, text, color=DIM, size=12, weight="normal", **kwargs):
        return ctk.CTkLabel(parent, text=text, text_color=color, font=self._font(size, weight), anchor="w", **kwargs)

    def _button(self, parent, text, command, primary=False, width=0, **kwargs):
        return ctk.CTkButton(
            parent, text=f"[ {text.upper()} ]", command=command, corner_radius=0, border_width=1, height=28,
            width=width or 0, font=self._font(12, "bold" if primary else "normal"),
            fg_color=ACCENT if primary else "transparent", hover_color="#e6e6e6" if primary else "#141414",
            text_color=BG if primary else TEXT, border_color=ACCENT if primary else BORDER,
            text_color_disabled="#3a3a3a", **kwargs)

    def _entry(self, parent, variable, width=0, **kwargs):
        return ctk.CTkEntry(parent, textvariable=variable, corner_radius=0, border_width=1, height=26, width=width or 140,
                            fg_color=FIELD, border_color=BORDER, text_color=TEXT, font=self._font(12), **kwargs)

    def _option(self, parent, variable, values, width=0, command=None):
        return ctk.CTkOptionMenu(parent, variable=variable, values=values, corner_radius=0, height=26, width=width or 140,
                                 fg_color=FIELD, button_color="#1a1a1a", button_hover_color="#262626", text_color=TEXT,
                                 dropdown_fg_color=FIELD, dropdown_text_color=TEXT, dropdown_hover_color=ACCENT_DIM,
                                 font=self._font(12), dropdown_font=self._font(12), command=command)

    def _check(self, parent, text, variable, command=None):
        return ctk.CTkCheckBox(parent, text=text, variable=variable, corner_radius=0, border_width=1, checkbox_width=16,
                               checkbox_height=16, fg_color=ACCENT, hover_color=ACCENT_DIM, border_color=BORDER_HI,
                               checkmark_color=BG, text_color=TEXT, font=self._font(12), command=command)

    def _segmented(self, parent, variable, values, command=None):
        return ctk.CTkSegmentedButton(parent, variable=variable, values=values, corner_radius=0, height=26, border_width=1,
                                      fg_color=BORDER, selected_color=ACCENT_DIM, selected_hover_color="#333333",
                                      unselected_color=FIELD, unselected_hover_color="#161616", text_color=TEXT,
                                      font=self._font(12), command=command)

    # ---------------------------------------------------------------- sidebar rows

    def _section(self, tab, title):
        row = tab.next_row
        rule = "─" * max(2, 44 - len(title))
        self._label(tab, f"┌─ {title.upper()} {rule}", color=ACCENT, size=12, weight="bold").grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=(14 if row else 6, 4))
        tab.next_row += 1

    def _row(self, tab, label, widget_factory, hint=None):
        row = tab.next_row
        self._label(tab, f"│ {label}", color=DIM).grid(row=row, column=0, sticky="w", padx=(10, 6), pady=2)
        widget = widget_factory(tab)
        widget.grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=2)
        tab.next_row += 1
        if hint:
            self._label(tab, f"│   {hint}", color=HINT, size=11).grid(row=tab.next_row, column=0, columnspan=2,
                                                                         sticky="w", padx=(10, 6))
            tab.next_row += 1
        return widget

    def _path_row(self, tab, label, key, kind="file", filetypes=None, on_change=None):
        def build(parent):
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            frame.grid_columnconfigure(0, weight=1)
            entry = self._entry(frame, self.var[key])
            entry.grid(row=0, column=0, sticky="ew")
            entry.drop_target_register(DND_FILES)
            entry.dnd_bind("<<Drop>>", lambda event: self._drop_path(event, key, on_change))
            ctk.CTkButton(frame, text="…", width=28, height=26, corner_radius=0, border_width=1, border_color=BORDER,
                          fg_color=FIELD, hover_color="#1a1a1a", text_color=TEXT, font=self._font(12),
                          command=lambda: self._browse(key, kind, filetypes, on_change)).grid(row=0, column=1, padx=(4, 0))
            return frame
        return self._row(tab, label, build)

    def _slider_row(self, tab, label, key, low, high, steps, fmt="{:.2f}"):
        def build(parent):
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            frame.grid_columnconfigure(0, weight=1)
            readout = self._label(frame, "", color=TEXT, width=52)
            slider = ctk.CTkSlider(frame, from_=low, to=high, number_of_steps=steps, variable=self.var[key], height=14,
                                   corner_radius=0, button_corner_radius=0, button_length=6, fg_color=BORDER,
                                   progress_color=ACCENT_DIM, button_color=ACCENT, button_hover_color="#e6e6e6")
            slider.grid(row=0, column=0, sticky="ew")
            readout.grid(row=0, column=1, padx=(8, 0))

            def update(*_args):
                try:
                    readout.configure(text=fmt.format(self.var[key].get()))
                except tk.TclError:
                    pass
            self.var[key].trace_add("write", update)
            update()
            return frame
        return self._row(tab, label, build)

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
        # Wordmark: the half-lit disc, then DE in full white and LIGHTER dimmed - lit / unlit.
        mark = ctk.CTkFrame(top, fg_color="transparent")
        mark.grid(row=0, column=0, padx=(10, 14), pady=4)
        logo = self._logo_image(64)
        ctk.CTkLabel(mark, text="", image=ctk.CTkImage(light_image=logo, dark_image=logo, size=(20, 20))).pack(side="left", padx=(0, 8))
        self._label(mark, "DE", color=TEXT_HI, size=15, weight="bold").pack(side="left")
        ctk.CTkLabel(mark, text="LIGHTER", text_color=DIM, anchor="w",
                     font=ctk.CTkFont(family=self._light_family(), size=15)).pack(side="left")
        self.top_info = self._label(top, "scan → pbr  //  de-light  //  compare", color=HINT, size=12)
        self.top_info.grid(row=0, column=1, sticky="w")
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
        self.view_hint = self._label(header, "", color=DIM, size=11, width=320)
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

        self.tab_var = tk.StringVar(value="INPUT")
        tabs = ["INPUT", "DELIGHT", "SCAN", "VIEW", "SYS"]
        self._segmented(side, self.tab_var, tabs, command=lambda _value: self._show_tab()).grid(
            row=0, column=0, sticky="ew", padx=8, pady=8)

        self.tabs = {}
        for name in tabs:
            tab = ctk.CTkScrollableFrame(side, fg_color=PANEL, corner_radius=0, scrollbar_button_color=BORDER,
                                         scrollbar_button_hover_color=BORDER_HI)
            tab.grid_columnconfigure(1, weight=1)
            tab.next_row = 0
            tab.grid(row=1, column=0, sticky="nsew", padx=2)
            self.tabs[name] = tab
        self._build_input_tab(self.tabs["INPUT"])
        self._build_delight_tab(self.tabs["DELIGHT"])
        self._build_scan_tab(self.tabs["SCAN"])
        self._build_view_tab(self.tabs["VIEW"])
        self._build_sys_tab(self.tabs["SYS"])
        self._show_tab()

        actions = ctk.CTkFrame(side, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=8, pady=8)
        actions.grid_columnconfigure((0, 1), weight=1, uniform="actions")
        self.process_btn = self._button(actions, "run de-light  F5", self._start_processing, primary=True, state="disabled")
        self.process_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.scan_btn = self._button(actions, "run scanopt  F6", self._start_scan, primary=True)
        self.scan_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.progress_label = self._label(actions, "", color=DIM, size=11)
        self.progress_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _show_tab(self):
        for name, tab in self.tabs.items():
            if name == self.tab_var.get():
                tab.grid()
            else:
                tab.grid_remove()

    # ---------------------------------------------------------------- tabs

    def _build_input_tab(self, tab):
        images = [("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.exr")]
        self._section(tab, "model")
        self._path_row(tab, "OBJ", "model", filetypes=[("OBJ mesh", "*.obj")], on_change=self._on_input_changed)
        self._section(tab, "maps")
        self._path_row(tab, "LIT RGB", "lit", filetypes=images, on_change=self._on_input_changed)
        self._path_row(tab, "NORMAL (OS)", "normal", filetypes=images)
        self._path_row(tab, "AO", "ao", filetypes=images)
        self._path_row(tab, "MASK", "mask", filetypes=images)
        self._row(tab, "", lambda parent: self._button(parent, "auto-fill from lit folder",
                                                       lambda: self._auto_fill_other_maps(self.var["lit"].get(), force=True)))
        self._section(tab, "output")
        self._path_row(tab, "DIRECTORY", "output_dir", kind="dir")
        self._row(tab, "FILENAME", lambda parent: self._entry(parent, self.var["output_name"]))
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

    def _build_delight_tab(self, tab):
        self._section(tab, "model")
        self._path_row(tab, "ONNX", "onnx_model", filetypes=[("ONNX model", "*.onnx")])
        self._label(tab, "│   empty = newest trained model (overcast, pilot)", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1
        self._row(tab, "DEVICE", lambda parent: self._check(parent, "force CPU", self.var["force_cpu"]))
        self._row(tab, "", lambda parent: self._button(parent, "reload engine", self._load_engine_async))
        self._section(tab, "inference")
        self._row(tab, "TILING", lambda parent: self._check(parent, "tiled (less VRAM)", self.var["tiled"]),
                  hint="only images of 2048 px and up are tiled")
        self._row(tab, "TILE SIZE", lambda parent: self._option(parent, self.var["tile_size"], ["auto", "512", "1024", "2048", "4096"]),
                  hint="auto = half the image")
        self._slider_row(tab, "OVERLAP", "tile_overlap", 0.05, 0.5, 45, fmt="{:.0%}")
        self._section(tab, "output")
        self._row(tab, "BIT DEPTH", lambda parent: self._segmented(parent, self.var["bit_depth"], ["8", "16"]))
        self._row(tab, "MASK", lambda parent: self._check(parent, "black outside UV islands", self.var["apply_mask"]))
        self._row(tab, "AFTER RUN", lambda parent: self._check(parent, "switch preview to after", self.var["show_after_run"]))

    def _build_scan_tab(self, tab):
        self._section(tab, "source")
        self._path_row(tab, "HIGH-RES SCAN", "scan", filetypes=[("Meshes", "*.obj *.ply *.fbx *.glb *.gltf *.stl")],
                       on_change=self._on_scan_chosen)
        self._path_row(tab, "PHOTO TEXTURE", "photo", filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.exr")])
        self._label(tab, "│   empty = taken from the scan's material", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1
        self._path_row(tab, "OUTPUT FOLDER", "scan_out", kind="dir")
        self._section(tab, "remesh")
        self._row(tab, "REMESHER", lambda parent: self._option(parent, self.var["remesher"], list(REMESHERS), width=250))
        self._row(tab, "TARGET FACES", lambda parent: self._entry(parent, self.var["faces"], width=90))
        self._row(tab, "FEATURES", lambda parent: self._check(parent, "keep sharp edges (quads)", self.var["sharp"]))
        self._section(tab, "uv")
        self._row(tab, "UNWRAPPER", lambda parent: self._segmented(parent, self.var["unwrap"], ["xatlas", "smart"]))
        self._slider_row(tab, "SMART ANGLE", "smart_angle", 30, 89, 59, fmt="{:.0f}°")
        self._row(tab, "MARGIN PX", lambda parent: self._entry(parent, self.var["margin"], width=90))
        self._section(tab, "bake")
        self._row(tab, "TEXTURE SIZE", lambda parent: self._segmented(parent, self.var["resolution"], ["512", "1024", "2048", "4096"]))
        self._row(tab, "AO SAMPLES", lambda parent: self._entry(parent, self.var["ao_samples"], width=90))
        self._slider_row(tab, "CAGE", "cage", 0.001, 0.05, 49, fmt="{:.3f}")
        self._slider_row(tab, "MAX RAY", "max_ray", 0.005, 0.15, 29, fmt="{:.3f}")
        self._label(tab, "│   cage / ray: fraction of the scan's bounding diagonal", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1
        self._section(tab, "material")
        self._slider_row(tab, "ROUGHNESS", "roughness", 0.2, 1.0, 80)
        self._slider_row(tab, "VARIATION", "roughness_variation", 0.0, 0.6, 60)
        self._row(tab, "DE-LIGHT", lambda parent: self._check(parent, "tiled inference", self.var["scan_tiled"]))
        self._section(tab, "debug")
        self._row(tab, "WORK FILES", lambda parent: self._check(parent, "keep intermediate EXRs", self.var["keep_work"]))

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
        self._path_row(tab, "BLENDER", "blender", filetypes=[("blender.exe", "*.exe")])
        self._path_row(tab, "QUADWILD DIR", "quadwild_dir", kind="dir")
        self._path_row(tab, "AUTOREMESHER", "autoremesher_dir", kind="dir")
        self._path_row(tab, "XATLAS SITE", "xatlas_site", kind="dir")
        self._path_row(tab, "SCANOPT PY", "scanopt_python", filetypes=[("python.exe", "*.exe")])
        self._label(tab, "│   empty = training/.venv, else this interpreter", color=HINT, size=11).grid(
            row=tab.next_row, column=0, columnspan=2, sticky="w", padx=(10, 6))
        tab.next_row += 1
        self._section(tab, "interface")
        self._slider_row(tab, "UI SCALE", "ui_scale", 0.6, 1.4, 16, fmt="{:.2f}")
        self._row(tab, "", lambda parent: self._button(parent, "apply scale", lambda: ctk.set_widget_scaling(self._value("ui_scale", float))))
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

        bindings = {"<F5>": self._start_processing, "<F6>": self._start_scan, "1": lambda: set_mode(0),
                    "2": lambda: set_mode(1), "3": lambda: set_mode(2), "<space>": flip_single,
                    "w": lambda: toggle("wireframe"), "c": lambda: toggle("checker"), "r": self._reset_view,
                    "a": lambda: toggle("autorotate"), "<Control-l>": self._clear_log}
        for sequence, action in bindings.items():
            self.root.bind_all(sequence, guarded(action))

    # ================================================================ engine

    @staticmethod
    def _resolve_model(preferred=""):
        """The shipped weights were recalled; fall back to locally trained models (DELIGHTER_MODEL overrides)."""
        candidates = [preferred, os.environ.get("DELIGHTER_MODEL", ""),
                      os.path.join(APP_DIR, "weights", MODEL_FILENAME),
                      r"F:\DelighterData\runs\overcast\delighter.onnx",
                      r"F:\DelighterData\runs\pilot\delighter.onnx"]
        return next((path for path in candidates if path and os.path.isfile(path)), os.path.join("weights", MODEL_FILENAME))

    def _load_engine_async(self):
        model_name = self._resolve_model(self.var["onnx_model"].get().strip())
        if self.var["force_cpu"].get():
            os.environ["DELIGHTER_FORCE_CPU"] = "1"
        else:
            os.environ.pop("DELIGHTER_FORCE_CPU", None)
        self._engine_load_id += 1
        load_id = self._engine_load_id
        self.engine = None
        self.process_btn.configure(state="disabled")
        self._set_status("LOADING", WARN)
        self._append_log(f"engine: loading {model_name}")

        def _init():
            try:
                engine = DelighterInferenceEngine(model_name)
                self._post(lambda: self._on_engine_ready(engine, load_id, model_name))
            except Exception as e:
                error = str(e)
                self._post(lambda error=error: self._on_engine_error(error, load_id))

        threading.Thread(target=_init, daemon=True).start()

    def _on_engine_ready(self, engine, load_id, model_name):
        if load_id != self._engine_load_id:
            return
        self.engine = engine
        self.process_btn.configure(state="normal")
        self.engine_label.configure(text=f"engine: {engine.provider_name} · {os.path.basename(model_name)}",
                                    text_color=ACCENT if engine.is_accelerated else WARN)
        self._set_status("READY", ACCENT)
        self._append_log(f"engine ready [{engine.device_info}]")

    def _on_engine_error(self, error, load_id):
        if load_id != self._engine_load_id:
            return
        self.process_btn.configure(state="disabled")
        self._set_status("ENGINE FAILED", ERROR)
        self._append_log(f"!! engine: {error}")

    def _set_status(self, text, color):
        self.status_label.configure(text=f"● {text}", text_color=color)

    def _append_log(self, text):
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
        if key == "model" and not self.var["model"].get().lower().endswith(".obj"):
            messagebox.showwarning("Model", "The 3D preview supports OBJ meshes only.", parent=self.root)
            return
        self._scan_before = None
        self._append_log(f"{key}: {os.path.basename(self.var[key].get())}")
        if key == "lit":
            self._validate_resolution(self.var["lit"].get())
            self._auto_fill_other_maps(self.var["lit"].get())
        self._refresh_viewport()

    def _on_scan_chosen(self, _key):
        scan = self.var["scan"].get()
        if scan and not self.var["scan_out"].get():
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

    def _start_processing(self):
        if self._busy["delight"]:
            return
        lit, norm, ao, mask = (self.var[key].get() for key in ("lit", "normal", "ao", "mask"))
        if not all([lit, norm, ao, mask]):
            messagebox.showwarning("De-light", "All four input maps must be set (INPUT tab).", parent=self.root)
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
        self.process_btn.configure(state="disabled", text="[ RUNNING… ]")
        self._set_status("DE-LIGHTING", TEXT_HI)
        self._append_log(f"de-light → {output_path}  {options}")

        def log_ui(msg):
            self._post(lambda: self._append_log(msg))

        def _work():
            started = time.time()
            try:
                out_file, dims, dev = self.engine.process_texture(lit, norm, ao, mask, output_path, log_callback=log_ui, **options)
                self._post(lambda: self._on_success(out_file, time.time() - started))
            except Exception as e:
                message = str(e)
                self._post(lambda: self._on_error(message))

        threading.Thread(target=_work, daemon=True).start()

    def _on_success(self, out_file, seconds):
        self._busy["delight"] = False
        self.process_btn.configure(state="normal", text="[ RUN DE-LIGHT  F5 ]")
        self.albedo_radio.configure(state="normal")
        self.last_output_path = out_file
        self._set_status("DONE", ACCENT)
        self._append_log(f"done in {seconds:.1f} s: {out_file}")
        if self.var["show_after_run"].get():
            self.preview_texture.set("albedo")
        self._refresh_viewport()

    def _on_error(self, err_msg):
        self._busy["delight"] = False
        self.process_btn.configure(state="normal", text="[ RUN DE-LIGHT  F5 ]")
        self._set_status("FAILED", ERROR)
        self._append_log(f"!! {err_msg}")
        messagebox.showerror("De-light", f"Inference failed:\n{err_msg}", parent=self.root)

    # ================================================================ scan optimiser

    def _scanopt_python(self):
        """scanopt needs OpenCV with EXR support; prefer the training environment that has it."""
        chosen = self.var["scanopt_python"].get().strip()
        candidate = os.path.join(APP_DIR, "training", ".venv", "Scripts", "python.exe")
        return chosen or os.environ.get("SCANOPT_PYTHON") or (candidate if os.path.isfile(candidate) else sys.executable)

    def _start_scan(self):
        if self._busy["scan"]:
            return
        scan = self.var["scan"].get().strip()
        if not os.path.isfile(scan):
            self.tab_var.set("SCAN")
            self._show_tab()
            messagebox.showwarning("Scan optimiser", "Choose a high-res scan file (SCAN tab).", parent=self.root)
            return
        try:
            faces = int(self.var["faces"].get())
            numbers = {key: int(self.var[key].get()) for key in ("margin", "ao_samples")}
        except ValueError:
            messagebox.showwarning("Scan optimiser", "Faces, margin and AO samples must be whole numbers.", parent=self.root)
            return
        out = self.var["scan_out"].get().strip() or os.path.join(os.path.dirname(scan), os.path.splitext(os.path.basename(scan))[0] + "_optimized")
        photo = self.var["photo"].get().strip()
        remesher = self.var["remesher"].get()
        command = [self._scanopt_python(), "-u", SCANOPT_SCRIPT, scan, "--out", out, "--faces", str(faces),
                   "--remesher", REMESHERS[remesher], "--resolution", self.var["resolution"].get(),
                   "--unwrap", self.var["unwrap"].get(), "--smart-angle", f"{self._value('smart_angle', float):.1f}",
                   "--margin", str(numbers["margin"]), "--ao-samples", str(numbers["ao_samples"]),
                   "--cage", f"{self._value('cage', float):.4f}", "--max-ray", f"{self._value('max_ray', float):.4f}",
                   "--roughness", f"{self._value('roughness', float):.3f}",
                   "--roughness-variation", f"{self._value('roughness_variation', float):.3f}",
                   "--blender", self.var["blender"].get(), "--quadwild", self.var["quadwild_dir"].get(),
                   "--autoremesher", self.var["autoremesher_dir"].get()]
        model = self.var["onnx_model"].get().strip()
        if model:
            command += ["--model", model]
        if photo:
            command += ["--texture", photo]
        for key, flag in (("sharp", "--sharp"), ("scan_tiled", "--tiled"), ("keep_work", "--keep-work")):
            if self.var[key].get():
                command.append(flag)
        env = dict(os.environ, XATLAS_SITE=self.var["xatlas_site"].get())
        self._busy["scan"] = True
        self.scan_btn.configure(state="disabled", text="[ RUNNING… ]")
        self._set_status("SCANOPT", TEXT_HI)
        self._append_log(f"scanopt: {os.path.basename(scan)} → {faces} faces ({REMESHERS[remesher]}, {self.var['unwrap'].get()})")
        threading.Thread(target=self._run_scan_optimizer, args=(command, env, (scan, photo or None)), daemon=True).start()

    def _run_scan_optimizer(self, command, env, source):
        glb_path, tail, started = None, [], time.time()
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                       env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in process.stdout:
                line = line.rstrip()
                tail = (tail + [line])[-12:]
                if line.startswith("[") and "/4]" in line[:6]:
                    self._post(lambda text=line: (self._append_log(f"  {text}"),
                                                          self.progress_label.configure(text=f"scanopt {text[:5]}")))
                elif line.startswith("done: "):
                    glb_path = line[6:].split("  (")[0].strip()
            code = process.wait()
        except OSError as error:
            code, tail = -1, [str(error)]

        def finish():
            self._busy["scan"] = False
            self.scan_btn.configure(state="normal", text="[ RUN SCANOPT  F6 ]")
            self.progress_label.configure(text="")
            if code == 0 and glb_path:
                self._set_status("DONE", ACCENT)
                self._append_log(f"scanopt done in {time.time() - started:.0f} s: {glb_path}")
                self._show_optimized_scan(glb_path, *source)
            else:
                self._set_status("SCANOPT FAILED", ERROR)
                self._append_log("!! scanopt failed:\n" + "\n".join(tail[-6:]))

        self._post(finish)

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
        camera.wireframe = bool(self.var["wireframe"].get())
        camera.checker = bool(self.var["checker"].get())
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
        mode = self.var["view_mode"].get()
        if mode not in VIEW_MODES:
            mode = VIEW_MODES[0]
        side_by_side = mode == VIEW_MODES[1]
        hints = {VIEW_MODES[0]: "drag: orbit · wheel: zoom · space: before/after",
                 VIEW_MODES[1]: "linked cameras · drag either view",
                 VIEW_MODES[2]: "drag the line: compare · elsewhere: orbit"}
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
            self.viewport_before.grid()
            self.viewport.set_mode("b")
        else:
            self.viewport_before.grid_remove()
            self.views_frame.grid_columnconfigure(0, weight=0, uniform="")
            self.views_frame.grid_columnconfigure(1, weight=1, uniform="")
            self.viewport.set_mode("wipe" if mode == VIEW_MODES[2] else "a" if self.preview_texture.get() == "lit" else "b")

    def _comparison(self):
        """(mesh, texture) for before and after. Before is the original scan once an optimised
        scan is shown and it could be loaded, else the lit texture on the same mesh."""
        model = self.var["model"].get()
        before = self._scan_before or (model, self.var["lit"].get())
        return before, (model, self.last_output_path)

    def _refresh_viewport(self):
        if not self.viewport:
            return
        before, after = self._comparison()
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
        self.caption_before.configure(text=f"◀ BEFORE · {slots['a'].triangles:,} tris")
        self.caption_after.configure(text=f"AFTER · {slots['b'].triangles:,} tris ▶")

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


if __name__ == "__main__":
    app = DelighterGUI()
    app.run()

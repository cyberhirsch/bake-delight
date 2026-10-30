"""Shared greyscale TUI look for the delighter apps: palette, wordmark logo and widget helpers.

A host class mixes in TuiWidgets and provides `self.root`, `self.font_family` (from
_pick_font) and, for _slider_row, `self.var` (name -> Tk variable).
"""

import tkinter as tk
import tkinter.font as tkfont

import customtkinter as ctk
from PIL import Image

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


class TuiWidgets:
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
        import os
        import sys
        import tempfile
        from PIL import ImageTk
        self._icon = ImageTk.PhotoImage(self._logo_image(64), master=self.root)
        # On Windows the title bar and taskbar use the .ico, and CustomTkinter sets its own blue one
        # about 200 ms after start; write ours and set it after that.
        ico = None
        if sys.platform == "win32":
            ico = os.path.join(tempfile.gettempdir(), "bake_delight.ico")
            try:
                self._logo_image(256).save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)])
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("BakeDelight")  # own taskbar icon, not python's
            except (OSError, AttributeError):
                ico = None

        def apply():
            if ico:
                self.root.iconbitmap(ico)  # also tells CustomTkinter not to set its own
                self.root.iconbitmap(default=ico)  # dialogs and other toplevels
            self.root.iconphoto(True, self._icon)
        apply()
        self.root.after(300, apply)

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

"""Shared greyscale TUI look for the delighter apps: palette, wordmark logo and widget helpers.

A host class mixes in TuiWidgets and provides `self.root`, `self.font_family` (from
_pick_font) and, for _slider_row, `self.var` (name -> Tk variable).
"""

import tkinter as tk
import tkinter.font as tkfont

import customtkinter as ctk
from PIL import Image

# TUI palette: greyscale only. State is shown by brightness and by markers (●, !!), never by hue.
BG = "#121212"
PANEL = "#171717"
FIELD = "#1e1e1e"
BORDER = "#303030"
BORDER_HI = "#555555"
HINT = "#4a4a4a"
TEXT = "#c8c8c8"
TEXT_HI = "#ffffff"
DIM = "#6e6e6e"
ACCENT = "#ffffff"       # primary buttons, section rules, active marks
ACCENT_DIM = "#3a3a3a"   # selected segment, slider fill
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

    # Labels and buttons are plain Tk widgets in the CustomTkinter look: one native window each
    # instead of two or three, which Windows repaints on every resize (a resize took 0.8 s).

    def _tk_font(self, size=12, weight="normal"):
        """A font sized in pixels like CustomTkinter's, shared and rescaled by _apply_ui_scale."""
        fonts = self.__dict__.setdefault("_tk_fonts", {})
        if (size, weight) not in fonts:
            fonts[(size, weight)] = tkfont.Font(root=self.root, family=self.font_family, weight=weight,
                                                size=-round(size * self._ui_scale()))
        return fonts[(size, weight)]

    def _ui_scale(self):
        return ctk.ScalingTracker.get_widget_scaling(self.root)

    def _apply_ui_scale(self, scale):
        ctk.set_widget_scaling(scale)
        self.root.update_idletasks()  # CustomTkinter applies the new scaling here
        for (size, _weight), font in getattr(self, "_tk_fonts", {}).items():
            font.configure(size=-round(size * self._ui_scale()))
        for widget in getattr(self, "_sized_labels", []):
            if widget.winfo_exists():
                widget.rescale(self._ui_scale())

    def _label(self, parent, text, color=DIM, size=12, weight="normal", **kwargs):
        label = Label(parent, text=text, text_color=color, font=self._tk_font(size, weight), scale=self._ui_scale(), **kwargs)
        self.__dict__.setdefault("_sized_labels", []).append(label)
        return label

    def _button(self, parent, text, command, primary=False, width=0, **kwargs):
        return Button(parent, text=f"[ {text.upper()} ]", command=command, width=width,
                      font=self._tk_font(12, "bold" if primary else "normal"),
                      fg_color=ACCENT if primary else None, hover_color="#e6e6e6" if primary else "#141414",
                      text_color=BG if primary else TEXT, border_color=ACCENT if primary else BORDER, **kwargs)

    def _entry(self, parent, variable, width=0, **kwargs):
        return ctk.CTkEntry(parent, textvariable=variable, corner_radius=0, border_width=1, height=26, width=width or 140,
                            fg_color=FIELD, border_color=BORDER, text_color=TEXT, font=self._font(12), **kwargs)

    def _option(self, parent, variable, values, width=0, command=None):
        return ctk.CTkOptionMenu(parent, variable=variable, values=values, corner_radius=0, height=26, width=width or 140,
                                 fg_color=FIELD, button_color="#2a2a2a", button_hover_color="#363636", text_color=TEXT,
                                 dropdown_fg_color=FIELD, dropdown_text_color=TEXT, dropdown_hover_color=ACCENT_DIM,
                                 font=self._font(12), dropdown_font=self._font(12), command=command)

    def _check(self, parent, text, variable, command=None):
        return ctk.CTkCheckBox(parent, text=text, variable=variable, corner_radius=0, border_width=1, checkbox_width=16,
                               checkbox_height=16, fg_color=ACCENT, hover_color=ACCENT_DIM, border_color=BORDER_HI,
                               checkmark_color=BG, text_color=TEXT, font=self._font(12), command=command)

    def _segmented(self, parent, variable, values, command=None):
        return ctk.CTkSegmentedButton(parent, variable=variable, values=values, corner_radius=0, height=26, border_width=1,
                                      fg_color=BORDER, selected_color=ACCENT_DIM, selected_hover_color="#333333",
                                      unselected_color=FIELD, unselected_hover_color="#2a2a2a", text_color=TEXT,
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


def background_of(widget):
    """The colour a widget placed in `widget` should paint behind itself."""
    while widget is not None:
        widget = getattr(widget, "_parent_frame", widget)  # a CTkScrollableFrame paints with its outer frame
        if isinstance(widget, (ctk.CTkBaseClass, ctk.CTk, ctk.CTkToplevel)):
            color = getattr(widget, "_fg_color", "transparent")
            if color == "transparent":
                color = getattr(widget, "_bg_color", "transparent")
            if color != "transparent":
                return widget._apply_appearance_mode(color)
        else:
            return widget.cget("bg")
        widget = widget.master
    return BG


class Label(tk.Label):
    """CTkLabel's job for flat text: `text_color`, and `width`/`height` in pixels (height 28 by default)."""

    def __init__(self, master, text="", text_color=DIM, font=None, width=0, height=28, anchor="w", scale=1.0, **kwargs):
        self._size = (width, height)
        self._blank = tk.PhotoImage(master=master, width=1, height=1)  # makes width and height count in pixels
        super().__init__(master, text=text, fg=text_color, bg=background_of(master), font=font, anchor=anchor,
                         image=self._blank, compound="center", bd=0, padx=0, pady=0, highlightthickness=0, **kwargs)
        self.rescale(scale)

    def rescale(self, scale):
        width, height = self._size
        super().configure(width=round(width * scale), height=round(height * scale))

    def configure(self, cnf=None, **kwargs):
        if "text_color" in kwargs:
            kwargs["fg"] = kwargs.pop("text_color")
        return super().configure(cnf, **kwargs)

    config = configure


class Button(tk.Label):
    """CTkButton's job for a flat, square TUI button: 1 px border, hover colour, CTk option names.

    A label, because Tk on Windows draws no border around a native button.
    """

    def __init__(self, master, text="", command=None, font=None, width=0, fg_color=None, hover_color="#141414",
                 text_color=TEXT, border_color=BORDER, text_color_disabled="#3a3a3a", **kwargs):
        self._fg, self._hover, self._command = fg_color or background_of(master), hover_color, command
        super().__init__(master, text=text, font=font, bd=0, padx=4, pady=5, bg=self._fg, fg=text_color,
                         activebackground=hover_color, activeforeground=text_color, disabledforeground=text_color_disabled,
                         highlightthickness=1, highlightbackground=border_color, highlightcolor=border_color, **kwargs)
        if width:
            # Tk counts a label's width in characters; leave room for the padding and the border.
            super().configure(width=max(1, int((width - 10) / tkfont.Font(font=font).measure("0"))))
        self.bind("<Enter>", lambda _event: self._enabled() and super(Button, self).configure(bg=self._hover))
        self.bind("<Leave>", lambda _event: super(Button, self).configure(bg=self._fg))
        self.bind("<ButtonRelease-1>", self._release)

    def _enabled(self):
        return str(self["state"]) != "disabled"

    def _release(self, event):
        inside = 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()
        if inside and self._enabled() and self._command:
            self._command()

    def invoke(self):
        if self._enabled() and self._command:
            self._command()

    def configure(self, cnf=None, **kwargs):
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        if "fg_color" in kwargs:
            self._fg = kwargs.pop("fg_color") or background_of(self.master)
            kwargs["bg"] = self._fg
        if "hover_color" in kwargs:
            self._hover = kwargs["activebackground"] = kwargs.pop("hover_color")
        if "text_color" in kwargs:
            kwargs["fg"] = kwargs["activeforeground"] = kwargs.pop("text_color")
        if "border_color" in kwargs:
            kwargs["highlightbackground"] = kwargs["highlightcolor"] = kwargs.pop("border_color")
        if "text_color_disabled" in kwargs:
            kwargs["disabledforeground"] = kwargs.pop("text_color_disabled")
        return super().configure(cnf, **kwargs)

    config = configure


class ScrollPanel(tk.Frame):
    """A vertically scrolling panel; put widgets in `.content`.

    Replaces CTkScrollableFrame, whose scrollbar runs a full layout pass on every redraw:
    with eight tabs laid out at once, one window resize took 3-21 s.
    """

    def __init__(self, master, bg=PANEL, thumb=BORDER, thumb_hover=BORDER_HI, bar_width=8):
        super().__init__(master, bg=bg, bd=0, highlightthickness=0)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._colors = (thumb, thumb_hover)
        self._canvas = tk.Canvas(self, bg=bg, bd=0, highlightthickness=0, yscrollincrement=20)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._bar = tk.Canvas(self, bg=bg, width=bar_width, bd=0, highlightthickness=0)
        self._bar.grid(row=0, column=1, sticky="ns", padx=(2, 3), pady=3)
        self._thumb = self._bar.create_rectangle(0, 0, 0, 0, fill=thumb, outline="", state="hidden")
        self._view = (0.0, 1.0)
        self._drag = None
        self.content = ctk.CTkFrame(self._canvas, fg_color=bg, corner_radius=0)
        self._window = self._canvas.create_window(0, 0, window=self.content, anchor="nw")
        self._canvas.configure(yscrollcommand=self._set_view)
        self._canvas.bind("<Configure>", self._fit_width)
        self.content.bind("<Configure>", lambda _event: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._bar.bind("<Configure>", lambda _event: self._draw_thumb())
        self._bar.bind("<Enter>", lambda _event: self._bar.itemconfigure(self._thumb, fill=self._colors[1]))
        self._bar.bind("<Leave>", lambda _event: self._bar.itemconfigure(self._thumb, fill=self._colors[0]))
        self._bar.bind("<ButtonPress-1>", self._press)
        self._bar.bind("<B1-Motion>", self._move)
        self.bind_all("<MouseWheel>", self._wheel, add="+")

    def _fit_width(self, event):
        # Only on a real change: re-setting the same width re-lays out and repaints the whole tab.
        if int(self._canvas.itemcget(self._window, "width") or 0) != event.width:
            self._canvas.itemconfigure(self._window, width=event.width)

    def _set_view(self, first, last):
        self._view = (float(first), float(last))
        self._draw_thumb()

    def _draw_thumb(self):
        first, last = self._view
        height = self._bar.winfo_height()
        if last - first >= 1.0 or height <= 1:
            self._bar.itemconfigure(self._thumb, state="hidden")
            return
        self._bar.coords(self._thumb, 0, first * height, self._bar.winfo_width(), last * height)
        self._bar.itemconfigure(self._thumb, state="normal")

    def _press(self, event):
        first, last = self._view
        height = max(1, self._bar.winfo_height())
        if not first * height <= event.y <= last * height:  # click beside the thumb: centre it there
            self._canvas.yview_moveto(event.y / height - (last - first) / 2)
        self._drag = (event.y, self._view[0])

    def _move(self, event):
        if self._drag:
            y, first = self._drag
            self._canvas.yview_moveto(first + (event.y - y) / max(1, self._bar.winfo_height()))

    def _wheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self:
            widget = widget.master
        if widget is self and self._view != (0.0, 1.0):
            self._canvas.yview_scroll(-2 if event.delta > 0 else 2, "units")

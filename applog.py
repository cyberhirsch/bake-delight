"""The app's diagnostics: everything printed goes to the console (when there is one, as in
BakeDelightDebug.exe) and to a log file that can be sent along when a machine misbehaves."""

import faulthandler
import os
import re
import sys
import threading
import time

LOG_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "TextureDelighter", "logs")
LOG_PATH = os.path.join(LOG_DIR, "bakedelight.log")


def clean(line: str) -> str:
    """A native library's line as text: ONNX Runtime writes UTF-16 with colour codes."""
    return re.sub(r"\x1b\[[0-9;]*m", "", line.replace("\0", ""))


class _Tee:
    """A text stream that stamps each line and writes it to the console and the log file."""

    def __init__(self, console, handle, lock):
        self.console, self.handle, self.lock = console, handle, lock
        self.line_start = True

    def write(self, text):
        with self.lock:
            stamped = text
            if text:
                stamp = time.strftime("%H:%M:%S ")
                body = text[:-1].replace("\n", "\n" + stamp) + text[-1]
                stamped = (stamp if self.line_start else "") + body
                self.line_start = text.endswith("\n")
            for stream in (self.console, self.handle):
                if stream is not None:
                    try:
                        stream.write(stamped)
                        stream.flush()
                    except (OSError, ValueError):
                        pass  # a closed console must not stop the app
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


def start() -> str:
    """Send stdout and stderr to the console and the log (the previous run's becomes .1); returns its path."""
    if isinstance(sys.stdout, _Tee):
        return LOG_PATH
    handle = None
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        if os.path.isfile(LOG_PATH):
            os.replace(LOG_PATH, LOG_PATH + ".1")
        handle = open(LOG_PATH, "w", encoding="utf-8", errors="replace", buffering=1)
        faulthandler.enable(file=handle)  # native crashes (a GPU driver, say) leave a trace too
    except OSError as error:
        print(f"[Log] no log file: {error}", file=sys.stderr)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # a console codepage must not break a print
    lock = threading.Lock()
    sys.stdout = _Tee(sys.stdout, handle, lock)
    sys.stderr = _Tee(_capture_native_stderr(), handle, lock)
    return LOG_PATH if handle else ""


def _capture_native_stderr():
    """Native libraries write to file descriptor 2, past Python: ONNX Runtime says there why a GPU
    provider failed. Lead it through a pipe into sys.stderr; returns a stream to the real console."""
    try:
        console = open(os.dup(2), "w", encoding="utf-8", errors="replace")
    except OSError:
        console = None  # the windowed app has none
    try:
        read_end, write_end = os.pipe()
        os.dup2(write_end, 2)
    except OSError:
        return console if console is not None else sys.stderr

    def pump():
        with open(read_end, encoding="utf-8", errors="replace") as lines:
            for line in lines:
                sys.stderr.write(clean(line))

    threading.Thread(target=pump, daemon=True).start()
    return console

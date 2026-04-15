import sys
import os
import time
import ctypes
from pathlib import Path
import numpy as np

# ANSI escape helpers
CSI = "\033["
CURSOR_HOME = f"{CSI}H"
CLEAR_SCREEN = f"{CSI}2J"
HIDE_CURSOR = f"{CSI}?25l"
SHOW_CURSOR = f"{CSI}?25h"
RESET = f"{CSI}0m"
REVERSE = f"{CSI}7m"

# Load C renderer
_LIB_DIR = Path(__file__).parent
_so_path = _LIB_DIR / "_render.so"
if not _so_path.exists():
    raise RuntimeError(
        f"C renderer not built. Run:\n"
        f"  gcc -O2 -shared -fPIC -o {_so_path} {_LIB_DIR / '_render.c'}"
    )
_lib = ctypes.CDLL(str(_so_path))
_lib.render_halfblock.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # top
    ctypes.POINTER(ctypes.c_uint8),  # bot
    ctypes.c_int,                    # n_rows
    ctypes.c_int,                    # width
    ctypes.c_int,                    # pad
    ctypes.c_char_p,                 # out
]
_lib.render_halfblock.restype = ctypes.c_int


def _build_heat_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.2:
            s = t / 0.2
            r, g, b = 0, 0, int(128 * s)
        elif t < 0.4:
            s = (t - 0.2) / 0.2
            r, g, b = 0, int(200 * s), int(128 + 127 * s)
        elif t < 0.6:
            s = (t - 0.4) / 0.2
            r, g, b = int(255 * s), 200 + int(55 * s), int(255 * (1 - s))
        elif t < 0.8:
            s = (t - 0.6) / 0.2
            r, g, b = 255, int(255 * (1 - 0.6 * s)), 0
        else:
            s = (t - 0.8) / 0.2
            r, g, b = 255, int(102 + 153 * s), int(255 * s)
        lut[i] = [r, g, b]
    return lut


def _build_gray_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lut[i] = [i, i, i]
    return lut


HEAT_LUT = _build_heat_lut()
GRAY_LUT = _build_gray_lut()


class LiveViewer:

    def __init__(self, width, height, max_depth=4096, enabled=True):
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.colormap = "heat"
        self.enabled = enabled

        self._frame_count = 0
        self._fps = 0.0
        self._fps_t0 = 0.0
        # Rendering cache
        self._prev_cols = 0
        self._prev_rows = 0
        self._y_idx = None
        self._x_idx = None
        self._out_w = 0
        self._out_h = 0
        # Reusable output buffer (512 KB)
        self._buf = ctypes.create_string_buffer(512 * 1024)

    def open(self):
        """Enter TUI mode: hidden cursor, cleared screen."""
        sys.stdout.write(HIDE_CURSOR + CLEAR_SCREEN)
        sys.stdout.flush()
        self._fps_t0 = time.monotonic()

    def close(self):
        """Restore terminal to normal."""
        sys.stdout.write(SHOW_CURSOR + RESET + "\n")
        sys.stdout.flush()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _update_downsample_indices(self, cols, rows):
        if cols == self._prev_cols and rows == self._prev_rows:
            return
        self._prev_cols = cols
        self._prev_rows = rows

        avail_rows = rows - 2
        pixel_rows = avail_rows * 2

        scale_x = self.width / cols
        scale_y = self.height / pixel_rows
        scale = max(scale_x, scale_y)

        self._out_w = min(cols, int(self.width / scale))
        self._out_h = min(pixel_rows, int(self.height / scale)) & ~1

        self._y_idx = np.linspace(0, self.height - 1, self._out_h).astype(int)
        self._x_idx = np.linspace(0, self.width - 1, self._out_w).astype(int)

    def _depth_to_rgb(self, frame):
        valid = frame > 0
        normalized = np.zeros(frame.shape, dtype=np.uint8)
        if valid.any():
            d = frame.astype(np.float32)
            d[valid] = (self.max_depth - np.clip(d[valid], 0, self.max_depth)) / self.max_depth * 255
            normalized = d.astype(np.uint8)

        lut = HEAT_LUT if self.colormap == "heat" else GRAY_LUT
        rgb = lut[normalized]
        bg = [30, 0, 0] if self.colormap == "heat" else [20, 20, 20]
        rgb[~valid] = bg
        return rgb

    def _stamp_markers(self, small, points, color, out_w, out_h):
        """Draw cross markers on the downsampled image at given pixel positions."""
        for cx, cy in points:
            px = int(cx / self.width * out_w)
            py = int(cy / self.height * out_h)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dy) + abs(dx) <= 2:
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < out_h and 0 <= nx < out_w:
                            small[ny, nx] = color

    def _render_frame(self, frame, points=None):
        cols, rows = os.get_terminal_size()
        self._update_downsample_indices(cols, rows)
        ow = self._out_w
        oh = self._out_h

        rgb = self._depth_to_rgb(frame)
        small = rgb[np.ix_(self._y_idx, self._x_idx)]

        if points:
            self._stamp_markers(small, points, [255, 105, 180], ow, oh)

        top = np.ascontiguousarray(small[0::2])  # (n_rows, ow, 3)
        bot = np.ascontiguousarray(small[1::2])
        n_rows = top.shape[0]
        pad = max(0, (cols - ow) // 2)

        nbytes = _lib.render_halfblock(
            top.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            bot.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n_rows, ow, pad,
            self._buf,
        )
        image = self._buf.raw[:nbytes].decode("utf-8")

        # Blank leftover rows
        avail_lines = rows - 2
        blank_count = avail_lines - n_rows
        blanks = (" " * cols + "\n") * blank_count if blank_count > 0 else ""

        # Status bar
        cmap = self.colormap.upper()
        status = f" LIVE {self.width}x{self.height} | {self._fps:.0f} fps | {cmap}"
        if points:
            status += f" | {len(points)} det"

        return image + blanks + f"{REVERSE}{status[:cols].ljust(cols)}{RESET}"

    def draw(self, frame, points=None):
        """
        Render a depth frame to the terminal.

        Args:
            frame: 2D numpy array (height x width) of depth values.
                   Zero means invalid/no-data.
            points: optional list of (x, y) tuples in pixel coordinates
                    to draw as pink markers on the frame.
        """
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_t0 = now

        if self.enabled:
            sys.stdout.write(self._render_frame(frame, points=points))
        else:
            cols, _ = os.get_terminal_size()
            cmap = self.colormap.upper()
            status = f" LIVE {self.width}x{self.height} | {self._fps:.0f} fps | {cmap}"
            sys.stdout.write(CURSOR_HOME + f"{REVERSE}{status[:cols].ljust(cols)}{RESET}")
        sys.stdout.flush()

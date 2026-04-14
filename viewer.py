"""
TUI depth rendering API.

Usage:
    from arducam_tui import LiveViewer

    viewer = LiveViewer(width=240, height=180)
    with viewer:
        while running:
            frame = cam.get_frame()
            viewer.draw(frame)
"""

import sys
import os
import time
import numpy as np

# ANSI escape helpers
CSI = "\033["
CURSOR_HOME = f"{CSI}H"
CLEAR_SCREEN = f"{CSI}2J"
HIDE_CURSOR = f"{CSI}?25l"
SHOW_CURSOR = f"{CSI}?25h"
RESET = f"{CSI}0m"
REVERSE = f"{CSI}7m"


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
    """
    TUI depth frame renderer.

    Args:
        width:     Frame width in pixels.
        height:    Frame height in pixels.
        max_depth: Maximum depth value for colormap normalization.
    """

    def __init__(self, width, height, max_depth=4096):
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.colormap = "heat"

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

    # ---- lifecycle (context manager) ----

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

    # ---- rendering ----

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

    def _render_frame(self, frame):
        cols, rows = os.get_terminal_size()
        self._update_downsample_indices(cols, rows)
        ow, oh = self._out_w, self._out_h

        rgb = self._depth_to_rgb(frame)
        small = rgb[np.ix_(self._y_idx, self._x_idx)]

        pad = max(0, (cols - ow) // 2)
        pad_str = " " * pad

        buf = [CURSOR_HOME]
        for r in range(0, oh, 2):
            top_row = small[r]
            bot_row = small[r + 1] if r + 1 < oh else np.zeros((ow, 3), dtype=np.uint8)
            parts = [pad_str]
            for c in range(ow):
                tr, tg, tb = int(top_row[c, 0]), int(top_row[c, 1]), int(top_row[c, 2])
                br, bg, bb = int(bot_row[c, 0]), int(bot_row[c, 1]), int(bot_row[c, 2])
                parts.append(f"\033[38;2;{tr};{tg};{tb};48;2;{br};{bg};{bb}m▀")
            parts.append(RESET)
            buf.append("".join(parts))

        rendered_lines = oh // 2
        avail_lines = rows - 2
        for _ in range(avail_lines - rendered_lines):
            buf.append(" " * cols)

        # Status bar
        cmap = self.colormap.upper()
        status = f" LIVE {self.width}x{self.height} | {self._fps:.0f} fps | {cmap}"
        buf.append(f"{REVERSE}{status[:cols].ljust(cols)}{RESET}")

        return "\n".join(buf)

    # ---- public API ----

    def draw(self, frame):
        """
        Render a depth frame to the terminal.

        Args:
            frame: 2D numpy array (height x width) of depth values.
                   Zero means invalid/no-data.
        """
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_t0 = now

        sys.stdout.write(self._render_frame(frame))
        sys.stdout.flush()

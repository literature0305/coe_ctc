"""Logging + box-drawing helpers.

Reproduces the visual style of
``tsm-trainer008_aed/errlog019-2_output_example`` so training health reports
look identical across the two frameworks.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Box geometry — width 74 matches the reference output exactly.
# ---------------------------------------------------------------------------
BOX_W: int = 74

_OUTER_TL = "╔"
_OUTER_TR = "╗"
_OUTER_BL = "╚"
_OUTER_BR = "╝"
_OUTER_H = "═"
_OUTER_V = "║"
_OUTER_LSEP = "╠"
_OUTER_RSEP = "╣"

_INNER_TL = "┌"
_INNER_TR = "┐"
_INNER_BL = "└"
_INNER_BR = "┘"
_INNER_H = "─"
_INNER_V = "│"


def _strip_pad(text: str, width: int) -> str:
    """Right-pad / clip a string to *width* visible characters.

    We rely on the caller to use single-cell characters; LibriSpeech
    transcripts (Latin-1) are safe. For mixed-width characters we fall back
    to ``len()`` which over-pads but never under-pads.
    """
    if len(text) > width:
        return text[: width - 1] + "…"
    return text + " " * (width - len(text))


# ─────────────────────────────────────────────────────────────────────────
# Outer box helpers — ╔══╗ style (used for the top-level TRAINING HEALTH
# REPORT envelope and the BENCHMARK EVALUATION block).
# ─────────────────────────────────────────────────────────────────────────


def box_top(title: str | None = None, width: int = BOX_W) -> str:
    """Render ``╔════ Title ════╗`` style top line."""
    inner = width - 2
    if title:
        # Pad title on both sides with the horizontal char so total inner width matches.
        padded = f"  {title}  "
        if len(padded) >= inner:
            return _OUTER_TL + padded[:inner] + _OUTER_TR
        left = (inner - len(padded)) // 2
        right = inner - len(padded) - left
        return _OUTER_TL + (_OUTER_H * left) + padded + (_OUTER_H * right) + _OUTER_TR
    return _OUTER_TL + (_OUTER_H * inner) + _OUTER_TR


def box_bottom(width: int = BOX_W) -> str:
    inner = width - 2
    return _OUTER_BL + (_OUTER_H * inner) + _OUTER_BR


def box_separator(width: int = BOX_W) -> str:
    inner = width - 2
    return _OUTER_LSEP + (_OUTER_H * inner) + _OUTER_RSEP


# ─────────────────────────────────────────────────────────────────────────
# Inner section helpers — ┌── Section ──┐ style.
# ─────────────────────────────────────────────────────────────────────────


def box_inner_top(section: str, width: int = BOX_W) -> str:
    inner = width - 2  # space inside outer ║ … ║ (or no outer)
    # Format: ║  ┌─ Section ─────┐ ║
    label = f" {section} "
    payload_width = inner - 4  # account for outer "║  " + " ║"
    label_with_lines = f"{_INNER_TL}{_INNER_H}{label}"
    pad = payload_width - len(label_with_lines) - 1  # -1 for trailing ┐
    if pad < 0:
        pad = 0
    return f"{_OUTER_V}  {label_with_lines}{_INNER_H * pad}{_INNER_TR} {_OUTER_V}"


def box_inner_bottom(width: int = BOX_W) -> str:
    inner = width - 2
    payload_width = inner - 4
    # payload_width chars between │ … │: 1 (└) + N (─) + 1 (┘) = payload_width
    return f"{_OUTER_V}  {_INNER_BL}{_INNER_H * (payload_width - 2)}{_INNER_BR} {_OUTER_V}"


def box_line(text: str, width: int = BOX_W) -> str:
    """A line nested inside an inner-box section: ``║  │  content │ ║``."""
    inner = width - 2
    payload_width = inner - 4  # account for outer "║  " + " ║"
    # Inside inner box: │  text  │ (text width = payload_width - 4)
    text_width = payload_width - 4
    content = _strip_pad(text, text_width)
    return f"{_OUTER_V}  {_INNER_V} {content} {_INNER_V} {_OUTER_V}"


def box_section_title(title: str, width: int = BOX_W) -> str:
    """A non-nested separator inside the outer envelope (no inner box)."""
    inner = width - 2
    pad_l = 2
    text = f" {title} "
    avail = inner - pad_l - 2
    return _OUTER_V + (" " * pad_l) + _strip_pad(text, avail) + "  " + _OUTER_V


def plain_box_top(title: str | None = None, width: int = BOX_W) -> str:
    """Single-bordered ┌── Title ──┐ box (used by BENCHMARK EVALUATION)."""
    inner = width - 2
    if title:
        padded = f" {title} "
        if len(padded) >= inner:
            return _INNER_TL + padded[:inner] + _INNER_TR
        left = (inner - len(padded)) // 2
        right = inner - len(padded) - left
        return _INNER_TL + (_INNER_H * left) + padded + (_INNER_H * right) + _INNER_TR
    return _INNER_TL + (_INNER_H * inner) + _INNER_TR


def plain_box_bottom(width: int = BOX_W) -> str:
    inner = width - 2
    return _INNER_BL + (_INNER_H * inner) + _INNER_BR


def plain_box_separator(width: int = BOX_W) -> str:
    inner = width - 2
    return "├" + (_INNER_H * inner) + "┤"


def plain_box_line(text: str, width: int = BOX_W) -> str:
    inner = width - 2
    return _INNER_V + " " + _strip_pad(text, inner - 2) + " " + _INNER_V


# ─────────────────────────────────────────────────────────────────────────
# Progress bar / sparkline helpers
# ─────────────────────────────────────────────────────────────────────────


def make_progress_bar(fraction: float, width: int = 40) -> str:
    """Render ``[████░░░░]`` style bar.

    Uses 3 fill levels (█▓░) to match the reference output's appearance.
    """
    fraction = max(0.0, min(1.0, fraction))
    cells = int(round(fraction * width))
    return "[" + ("█" * cells) + ("░" * (width - cells)) + "]"


def make_sparkline(values: Sequence[float]) -> str:
    """Compact unicode sparkline for short numeric series."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    arr = list(values)
    lo, hi = min(arr), max(arr)
    if hi == lo:
        return blocks[0] * len(arr)
    out_chars = []
    for v in arr:
        idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
        out_chars.append(blocks[idx])
    return "".join(out_chars)


# ─────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────


def format_duration(seconds: float) -> str:
    """Pretty-print a duration. Examples: ``"3.2s"``, ``"26m 40s"``, ``"44d 13h"``."""
    if seconds is None or (isinstance(seconds, float) and (math.isnan(seconds) or math.isinf(seconds))):
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def bytes_to_human(n: int | float) -> str:
    if n is None:
        return "—"
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


# ─────────────────────────────────────────────────────────────────────────
# Logger factory
# ─────────────────────────────────────────────────────────────────────────


_LOGGER_INITIALIZED = False


def setup_logger(
    name: str = "coe_ctc",
    level: int = logging.INFO,
    *,
    log_file: str | None = None,
    rank: int = 0,
) -> logging.Logger:
    """Idempotent logger setup.

    On non-zero ranks the logger is silenced (only rank-0 prints) unless the
    user explicitly raised ``level`` for debugging.
    """
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)
    if rank != 0 and level <= logging.INFO:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(level)

    if _LOGGER_INITIALIZED:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    _LOGGER_INITIALIZED = True
    return logger


# ─────────────────────────────────────────────────────────────────────────
# Tiny convenience: print a list of (label, value) inside an inner box
# ─────────────────────────────────────────────────────────────────────────


def render_kv_box(section: str, items: Iterable[tuple[str, str]], width: int = BOX_W) -> list[str]:
    """Render an inner-box section with ``Label: Value`` rows.

    Returns a list of pre-formatted lines (without trailing newlines).
    """
    out: list[str] = [box_inner_top(section, width)]
    items = list(items)
    if items:
        max_label = max(len(label) for label, _ in items)
    else:
        max_label = 0
    for label, value in items:
        out.append(box_line(f"{label.ljust(max_label)}  {value}", width))
    out.append(box_inner_bottom(width))
    return out

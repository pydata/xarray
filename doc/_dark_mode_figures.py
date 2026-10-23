"""Show every matplotlib figure in the docs together with a dark mode version.

This file runs at the start of each notebook kernel (through ``PYTHONSTARTUP``, set in
``conf.py``). Each figure is displayed twice: as is, and with the default black and
white colors (text, axes, ticks, legend frame, background) temporarily swapped for
dark mode ones. Nothing is executed twice, and the data colors are kept. The outputs
are tagged with the pydata-sphinx-theme ``only-light`` / ``only-dark`` classes, which
``conf.py`` passes on to the image nodes.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import matplotlib.colors as mcolors
from IPython import get_ipython
from IPython.display import display
from matplotlib.collections import Collection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

# the pydata-sphinx-theme dark mode text and background colors
DARK_MODE_FOREGROUND = "#ced6dd"
DARK_MODE_BACKGROUND = "#14181e"
TRANSPARENT = (0, 0, 0, 0)

IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/svg+xml")


def _dark_mode_color(
    color: Any, white: Any
) -> tuple[float, float, float, float] | None:
    """Dark mode replacement of a black or white color, None to keep the color.

    Black becomes the dark mode text color, white becomes ``white`` (None keeps it).
    """
    if isinstance(color, str) and color in ("none", "auto", "face"):
        return None
    try:
        r, g, b, a = mcolors.to_rgba(color)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    if (r, g, b) == (0, 0, 0):
        return mcolors.to_rgba(DARK_MODE_FOREGROUND, a)
    if (r, g, b) == (1, 1, 1) and white is not None:
        return mcolors.to_rgba(white, a if white != TRANSPARENT else None)
    return None


@contextmanager
def dark_mode(fig: Figure) -> Iterator[None]:
    """Temporarily swap the black and white colors of ``fig`` for dark mode colors.

    The white figure background becomes transparent, other white areas, lines and
    edges (e.g. legend frames or the edges separating scatter markers) the page
    background color. White text and the colors of colormapped data are kept.
    """
    restore: list[tuple[Callable[[Any], Any], Any]] = []

    def recolor(original: Any, setter: Callable[[Any], Any], white: Any) -> None:
        if (color := _dark_mode_color(original, white)) is not None:
            setter(color)
            restore.append((setter, original))

    try:
        artists = fig.findobj()
        # text boxes and annotation arrows are not children of their text
        for text in [a for a in artists if isinstance(a, Text)]:
            patches = (text.get_bbox_patch(), getattr(text, "arrow_patch", None))
            artists += [patch for patch in patches if patch is not None]
        for obj in artists:
            if isinstance(obj, Text):
                recolor(obj.get_color(), obj.set_color, None)
            elif isinstance(obj, Line2D):
                line = DARK_MODE_BACKGROUND
                recolor(obj.get_color(), obj.set_color, line)
                recolor(obj.get_markeredgecolor(), obj.set_markeredgecolor, line)
                recolor(obj.get_markerfacecolor(), obj.set_markerfacecolor, line)
            elif isinstance(obj, Patch):
                # the figure background shows the page, others (axes, legend frame)
                # must keep covering what is behind them
                face = TRANSPARENT if obj is fig.patch else DARK_MODE_BACKGROUND
                recolor(obj.get_facecolor(), obj.set_facecolor, face)
                recolor(obj.get_edgecolor(), obj.set_edgecolor, DARK_MODE_BACKGROUND)
            elif isinstance(obj, Collection):
                # only single colors: per element colors are data
                edgecolors = obj.get_edgecolor()
                if len(edgecolors) == 1:
                    recolor(edgecolors[0], obj.set_edgecolor, DARK_MODE_BACKGROUND)
                # and so are the face colors of colormapped collections
                facecolors = obj.get_facecolor()
                if obj.get_array() is None and len(facecolors) == 1:
                    recolor(facecolors[0], obj.set_facecolor, DARK_MODE_BACKGROUND)
        yield
    finally:
        for setter, original in reversed(restore):
            setter(original)


def _with_classes(classes: str) -> dict[str, dict[str, str]]:
    return {mime_type: {"classes": classes} for mime_type in IMAGE_MIME_TYPES}


_formatter = get_ipython().display_formatter.ipython_display_formatter


def display_with_dark_mode(fig: Figure) -> None:
    """IPython display formatter for figures that also shows the dark mode version.

    Covers figures shown by the inline backend (at the end of a cell or by
    ``plt.show()``) and figures returned as the last expression of a cell.
    """
    # IPython treats every figure as handled by this formatter, so switch it off
    # to render the figure itself
    _formatter.enabled = False
    try:
        display(fig, metadata=_with_classes("only-light"))
        with dark_mode(fig):
            display(fig, metadata=_with_classes("only-dark"))
    finally:
        _formatter.enabled = True


_formatter.for_type(Figure, display_with_dark_mode)

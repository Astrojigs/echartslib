"""Style presets and colour palettes for echartsy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

# ── Reusable palettes ────────────────────────────────────────────────────

PALETTE_DEFAULT = [
    "#5470C6", "#91CC75", "#FAC858", "#EE6666", "#73C0DE",
    "#3BA272", "#FC8452", "#9A60B4", "#EA7CCC",
]
PALETTE_RUSTY = ["#893448", "#d95850", "#eb8146", "#ffb248", "#f2d643", "#ebdba4"]
PALETTE_DARK = [
    "#dd6b66", "#759aa0", "#e69d87", "#8dc1a9", "#ea7e53",
    "#eedd78", "#73a373", "#73b9bc", "#7289ab", "#91ca8c",
]
PALETTE_OCEAN = [
    "#0077B6", "#00B4D8", "#48CAE4", "#90E0EF", "#ADE8F4",
    "#023E8A", "#0096C7", "#CAF0F8",
]
PALETTE_SUNSET = [
    "#F94144", "#F3722C", "#F8961E", "#F9C74F", "#90BE6D",
    "#43AA8B", "#577590",
]
PALETTE_EARTH = [
    "#606C38", "#283618", "#FEFAE0", "#DDA15E", "#BC6C25",
    "#6B705C", "#A5A58D", "#B7B7A4",
]
PALETTE_PASTEL = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
    "#E8BAFF", "#FFB3DE", "#B3FFE0",
]
PALETTE_NEON = [
    "#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#3A86FF",
    "#06D6A0", "#118AB2", "#EF476F",
]
PALETTE_MONOCHROME = [
    "#1A1A2E", "#16213E", "#0F3460", "#533483", "#E94560",
    "#3D5A80", "#98C1D9", "#293241",
]

# Backward compatibility alias
PALETTE_CLINICAL = PALETTE_DEFAULT


@dataclass(frozen=True)
class StylePreset:
    """Immutable bundle of visual defaults that can be applied to a Figure.

    Use one of the class-level constants (``DEFAULT``, ``DASHBOARD_DARK``,
    etc.) or build your own.

    Example
    -------
    >>> fig = Figure(style=StylePreset.DEFAULT)
    >>> fig = Figure(style=StylePreset(palette=("#ff0000", "#00ff00"), bg="#fafafa"))
    """

    palette: Tuple[str, ...] = tuple(PALETTE_DEFAULT)
    bg: Optional[str] = None
    font_family: str = "sans-serif"
    title_font_size: int = 16
    subtitle_font_size: int = 12
    axis_label_font_size: int = 12
    axis_label_color: str = "#666"
    grid_line_color: str = "#eee"
    tooltip_pointer: str = "cross"
    legend_orient: str = "horizontal"

    # Pre-built presets (populated after class definition)
    DEFAULT: ClassVar["StylePreset"]
    CLINICAL: ClassVar["StylePreset"]  # backward compat alias
    DASHBOARD_DARK: ClassVar["StylePreset"]
    KPI_REPORT: ClassVar["StylePreset"]
    MINIMAL: ClassVar["StylePreset"]
    OCEAN: ClassVar["StylePreset"]
    SUNSET: ClassVar["StylePreset"]
    EARTH: ClassVar["StylePreset"]
    PASTEL: ClassVar["StylePreset"]
    NEON: ClassVar["StylePreset"]


# Populate class-level presets after the class is defined
StylePreset.DEFAULT = StylePreset()  # type: ignore[misc]
StylePreset.CLINICAL = StylePreset.DEFAULT  # type: ignore[misc]  # backward compat
StylePreset.DASHBOARD_DARK = StylePreset(  # type: ignore[misc]
    palette=tuple(PALETTE_DARK),
    bg="#333",
    axis_label_color="#ccc",
    grid_line_color="#555",
)
StylePreset.KPI_REPORT = StylePreset(  # type: ignore[misc]
    palette=tuple(PALETTE_RUSTY),
    title_font_size=18,
    grid_line_color="#f0f0f0",
)
StylePreset.MINIMAL = StylePreset(  # type: ignore[misc]
    palette=("#5470C6", "#91CC75", "#FAC858", "#EE6666"),
    title_font_size=14,
    axis_label_font_size=11,
)
StylePreset.OCEAN = StylePreset(  # type: ignore[misc]
    palette=tuple(PALETTE_OCEAN),
    grid_line_color="#d0e8f2",
)
StylePreset.SUNSET = StylePreset(  # type: ignore[misc]
    palette=tuple(PALETTE_SUNSET),
)
StylePreset.EARTH = StylePreset(  # type: ignore[misc]
    palette=tuple(PALETTE_EARTH),
    grid_line_color="#e0ddd5",
)
StylePreset.PASTEL = StylePreset(  # type: ignore[misc]
    palette=tuple(PALETTE_PASTEL),
    grid_line_color="#f0f0f0",
)
StylePreset.NEON = StylePreset(  # type: ignore[misc]
    palette=tuple(PALETTE_NEON),
    bg="#1a1a2e",
    axis_label_color="#ccc",
    grid_line_color="#333",
)

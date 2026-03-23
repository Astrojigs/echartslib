"""
Figure — the top-level chart canvas for echartsy.

Equivalent to ``plt.figure()`` + ``plt.subplots()`` in matplotlib.
Collects series incrementally via ``.plot()``, ``.bar()``, etc.,
stores axis / legend / tooltip configuration, and renders everything
in a single call to ``.show()``.
"""
from __future__ import annotations

import copy
import itertools
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from echartsy._helpers import (
    _coerce_numeric,
    _resolve_agg,
    _resolve_layout,
    _sort_categories,
    _validate_columns,
    _validate_df,
)
from echartsy.exceptions import BuilderConfigError, DataValidationError
from echartsy.renderers import render
from echartsy.styles import StylePreset
from echartsy.emphasis import (
    Emphasis, LineEmphasis, ScatterEmphasis, PieEmphasis,
    RadarEmphasis, SankeyEmphasis, FunnelEmphasis, TreemapEmphasis,
    GraphEmphasis,
    ItemStyle, LabelStyle, LineStyle, AreaStyle, EndLabelStyle,
    LabelLineStyle, Blur, Select, TooltipStyle, AnimationConfig,
)


def _merge_style_params(
    base: dict, *,
    line_style=None, area_style=None, label_style=None,
    end_label=None, item_style=None,
    show_symbol=None, color=None,
    blur=None, select=None, selected_mode=None,
    animation=None, tooltip=None,
) -> None:
    """Merge dataclass-based style overrides into a series dict (in-place).

    Applied after scalar params and before ``series_kw`` overrides.
    Scalar shortcuts (color, show_symbol) are applied first so that
    dataclass params (item_style, etc.) take precedence.
    """
    if show_symbol is not None:
        base["showSymbol"] = show_symbol
    if color is not None:
        base.setdefault("itemStyle", {})["color"] = color
    if selected_mode is not None:
        base["selectedMode"] = selected_mode
    if line_style is not None:
        base.setdefault("lineStyle", {}).update(line_style.to_dict())
    if area_style is not None:
        base.setdefault("areaStyle", {}).update(area_style.to_dict())
    if label_style is not None:
        base.setdefault("label", {}).update(label_style.to_dict())
    if end_label is not None:
        base["endLabel"] = end_label.to_dict()
    if item_style is not None:
        base.setdefault("itemStyle", {}).update(item_style.to_dict())
    if blur is not None:
        base["blur"] = blur.to_dict()
    if select is not None:
        base["select"] = select.to_dict()
    if animation is not None:
        base.update(animation.to_dict())
    if tooltip is not None:
        base["tooltip"] = tooltip.to_dict()


# ═══════════════════════════════════════════════════════════════════════════
# Internal series-builder dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _SeriesMeta:
    """Book-keeping for one logical series."""
    chart_type: str
    name: str = ""
    y_axis_index: int = 0
    is_category_series: bool = True
    raw_config: dict = field(default_factory=dict)
    grid_index: int = 0


_FIGURE_COUNTER = itertools.count()


def _build_axis_pointer_cfg(
    type_val=None, snap=None, pointer_label=None,
    label_precision=None, label_bg=None, label_color=None,
    line_color=None, line_width=None, line_type=None,
    cross_color=None, cross_width=None, cross_type=None,
    shadow_color=None, shadow_opacity=None,
):
    """Build an ECharts axisPointer config dict from keyword arguments."""
    ap = {}
    if type_val is not None:
        ap["type"] = type_val
    if snap is not None:
        ap["snap"] = snap
    label_cfg = {}
    if pointer_label is not None:
        label_cfg["show"] = pointer_label
    if label_precision is not None:
        label_cfg["precision"] = label_precision
    if label_bg is not None:
        label_cfg["backgroundColor"] = label_bg
    if label_color is not None:
        label_cfg["color"] = label_color
    if label_cfg:
        ap["label"] = label_cfg
    line_style = {}
    if line_color is not None: line_style["color"] = line_color
    if line_width is not None: line_style["width"] = line_width
    if line_type is not None: line_style["type"] = line_type
    if line_style:
        ap["lineStyle"] = line_style
    cross_style = {}
    if cross_color is not None: cross_style["color"] = cross_color
    if cross_width is not None: cross_style["width"] = cross_width
    if cross_type is not None: cross_style["type"] = cross_type
    if cross_style:
        ap["crossStyle"] = cross_style
    shadow_style = {}
    if shadow_color is not None: shadow_style["color"] = shadow_color
    if shadow_opacity is not None: shadow_style["opacity"] = shadow_opacity
    if shadow_style:
        ap["shadowStyle"] = shadow_style
    return ap

# ═══════════════════════════════════════════════════════════════════════════
# ██  FIGURE  ██
# ═══════════════════════════════════════════════════════════════════════════

class Figure:
    """The top-level chart canvas — equivalent to ``plt.figure()``.

    Parameters
    ----------
    height : str
        CSS height of the rendered chart container (e.g. ``"500px"``).
    width : str or None
        CSS width; ``None`` means container width.
    renderer : ``{"canvas", "svg"}``
        ECharts renderer.
    theme : str or None
        ECharts built-in theme name (e.g. ``"dark"``).
    style : StylePreset or None
        Apply a :class:`StylePreset` bundle for palette, font sizes, etc.
    key : str or None
        Explicit Streamlit widget key. Auto-generated when ``None``.

    Example
    -------
    >>> import echartsy as ec
    >>> ec.config(engine="jupyter")
    >>> fig = ec.Figure(height="600px", style=ec.StylePreset.DEFAULT)
    >>> fig.bar(df, x="Month", y="Revenue").title("Monthly Revenue").show()
    """

    def __init__(
        self,
        height: str = "400px",
        width: Optional[str] = None,
        renderer: Literal["canvas", "svg"] = "svg",
        theme: Optional[str] = None,
        style: Optional[StylePreset] = None,
        key: Optional[str] = None,
        rows: int = 1,
        row_heights: Optional[List[str]] = None,
        _subplot_mode: bool = False,
    ) -> None:
        self._height = height
        self._width = width
        self._renderer = renderer
        self._theme = theme
        self._key = key
        self._style = style or StylePreset.DEFAULT
        self._subplot_mode = _subplot_mode

        # Multi-grid support
        self._n_grids = max(1, rows)
        self._row_heights = row_heights

        # Internal state
        self._series: List[dict] = []
        self._series_meta: List[_SeriesMeta] = []
        self._legend_items: List[str] = []
        self._user_set_rotate: bool = False  # True if user explicitly set label rotation

        # Axis storage
        self._x_axis: dict = {
            "type": "category",
            "data": [],
            "axisLabel": {
                "rotate": 0,
                "fontSize": self._style.axis_label_font_size,
                "color": self._style.axis_label_color,
            },
            "axisTick": {"show": False},
        }
        self._y_axes: List[dict] = [
            {
                "type": "value",
                "splitLine": {"show": True, "lineStyle": {"color": self._style.grid_line_color}},
                "axisLabel": {
                    "fontSize": self._style.axis_label_font_size,
                    "color": self._style.axis_label_color,
                },
            }
        ]
        self._x_categories: List[str] = []

        # Per-grid category tracking (grid 0 aliases self._x_categories)
        self._grid_categories: Dict[int, List[str]] = {0: self._x_categories}
        for gi in range(1, self._n_grids):
            self._grid_categories[gi] = []

        # Layout / chrome
        self._title_cfg: Optional[dict] = None
        self._legend_cfg: Optional[dict] = None
        self._tooltip_cfg: dict = {
            "trigger": "axis",
            "axisPointer": {"type": self._style.tooltip_pointer},
            "confine": True,
            "appendTo": "body",
        }
        self._axis_pointer_cfg: Optional[dict] = None
        self._grid_cfg: dict = {
            "left": 70, "right": 70,
            "top": 60, "bottom": 50,
            "containLabel": True,
        }
        self._toolbox_cfg: Optional[dict] = None
        self._datazoom_cfg: Optional[list] = None
        self._palette: Optional[Sequence[str]] = (
            list(self._style.palette) if self._style.palette else None
        )
        self._extra: dict = {}

        # Track chart mode for conflict detection
        self._chart_mode: Optional[str] = None  # "cartesian" | "pie" | "radar" | ...

    # ─── Repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        parts = [f"series={len(self._series)}", f"height={self._height!r}"]
        if self._chart_mode:
            parts.append(f"mode={self._chart_mode!r}")
        if self._renderer != "svg":
            parts.append(f"renderer={self._renderer!r}")
        if self._title_cfg:
            parts.append(f"title={self._title_cfg.get('text', '')!r}")
        return f"<Figure {', '.join(parts)}>"

    def summary(self) -> str:
        """Print a human-readable summary of the current figure state."""
        lines = [f"Figure(height={self._height!r}, renderer={self._renderer!r})"]
        lines.append(f"  Series: {len(self._series)}")
        for i, (s, m) in enumerate(zip(self._series, self._series_meta)):
            name_part = f"name={s.get('name', '')!r}" if s.get("name") else ""
            axis_part = f", axis={m.y_axis_index}" if m.y_axis_index > 0 else ""
            lines.append(f"    [{i}] {m.chart_type:10s} — {name_part}{axis_part}")
        if self._title_cfg:
            lines.append(f"  Title: {self._title_cfg.get('text', '')!r}")
        axes_parts = []
        if self._y_axes and self._y_axes[0].get("name"):
            axes_parts.append(f"left={self._y_axes[0]['name']!r}")
        if len(self._y_axes) > 1 and self._y_axes[1].get("name"):
            axes_parts.append(f"right={self._y_axes[1]['name']!r}")
        if axes_parts:
            lines.append(f"  Axes: {', '.join(axes_parts)}")
        result = "\n".join(lines)
        print(result)
        return result

    # ─── Private helpers ──────────────────────────────────────────────────

    def _ensure_mode(self, mode: str, caller: str) -> None:
        if self._chart_mode is None:
            self._chart_mode = mode
        elif self._chart_mode != mode:
            raise BuilderConfigError(
                f"{caller}() requires mode '{mode}' but figure is already in "
                f"'{self._chart_mode}' mode. Create a new Figure for a different chart type."
            )

    def _ensure_cartesian(self, caller: str) -> None:
        self._ensure_mode("cartesian", caller)

    def _merge_x_categories(self, new_cats: List[str], grid: int = 0) -> None:
        cats = self._grid_categories.get(grid, self._x_categories)
        existing = set(cats)
        for c in new_cats:
            if c not in existing:
                cats.append(c)
                existing.add(c)

    def _auto_key(self) -> str:
        return f"ecb_{next(_FIGURE_COUNTER)}"

    def _align_to_categories(
        self, df: pd.DataFrame, x: str, y: str, agg: str = "mean", grid: int = 0,
    ) -> List[Optional[float]]:
        agg_fn = _resolve_agg(agg)
        grouped = df.groupby(x)[y].agg(agg_fn)
        cats = self._grid_categories.get(grid, self._x_categories)
        return [
            None if cat not in grouped.index or pd.isna(grouped.get(cat))
            else round(float(grouped.get(cat)), 4)
            for cat in cats
        ]

    # ═══════════════════════════════════════════════════════════════════════
    # Chrome configuration (title, axes, legend, tooltip, grid, …)
    # ═══════════════════════════════════════════════════════════════════════

    def title(
        self, text: str, subtitle: Optional[str] = None,
        left: str = "center", top: Optional[Union[str, int]] = None,
        text_color: Optional[str] = None,
        text_font_weight: Optional[str] = None,
        link: Optional[str] = None,
        sublink: Optional[str] = None,
        background_color: Optional[str] = None,
        border_color: Optional[str] = None,
        border_width: Optional[int] = None,
        padding: Optional[Union[int, list]] = None,
    ) -> "Figure":
        """Set the chart title (and optional subtitle).

        Returns self for chaining.
        """
        self._title_cfg = {
            "text": text, "left": left,
            "textStyle": {
                "fontSize": self._style.title_font_size,
                "fontFamily": self._style.font_family,
            },
        }
        if subtitle:
            self._title_cfg["subtext"] = subtitle
            self._title_cfg["subtextStyle"] = {"fontSize": self._style.subtitle_font_size}
        if top is not None:
            self._title_cfg["top"] = top
        if text_color is not None:
            self._title_cfg["textStyle"]["color"] = text_color
        if text_font_weight is not None:
            self._title_cfg["textStyle"]["fontWeight"] = text_font_weight
        if link is not None:
            self._title_cfg["link"] = link
        if sublink is not None:
            self._title_cfg["sublink"] = sublink
        if background_color is not None:
            self._title_cfg["backgroundColor"] = background_color
        if border_color is not None:
            self._title_cfg["borderColor"] = border_color
        if border_width is not None:
            self._title_cfg["borderWidth"] = border_width
        if padding is not None:
            self._title_cfg["padding"] = padding
        return self

    def xlabel(
        self, name: str, rotate: Optional[int] = None,
        font_size: Optional[int] = None, color: Optional[str] = None,
    ) -> "Figure":
        """Configure the x-axis label and tick styling."""
        self._x_axis["name"] = name
        lbl = self._x_axis.setdefault("axisLabel", {})
        if rotate is not None:
            lbl["rotate"] = rotate
            self._user_set_rotate = True
        if font_size is not None:
            lbl["fontSize"] = font_size
        if color is not None:
            lbl["color"] = color
        return self

    def xticks(
        self, rotate: Optional[int] = None,
        interval: Optional[Union[int, str]] = None,
        formatter: Optional[str] = None,
    ) -> "Figure":
        """Fine-tune x-axis tick marks."""
        lbl = self._x_axis.setdefault("axisLabel", {})
        if rotate is not None:
            lbl["rotate"] = rotate
            self._user_set_rotate = True
        if interval is not None:
            lbl["interval"] = interval
        if formatter is not None:
            lbl["formatter"] = formatter
        return self

    def xlim(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None,
        scale: Optional[Literal["value", "log"]] = None,
    ) -> "Figure":
        """Set explicit x-axis bounds and/or scale type."""
        if self._chart_mode == "pie":
            raise BuilderConfigError("xlim() is not applicable to pie charts.")
        if min_val is not None:
            self._x_axis["min"] = min_val
        if max_val is not None:
            self._x_axis["max"] = max_val
        if scale is not None:
            self._x_axis["type"] = scale
        return self

    def xscale(self, scale: Literal["value", "log"]) -> "Figure":
        """Set x-axis scale type (shortcut for ``xlim(scale=...)``).

        >>> fig.xscale("log")
        """
        return self.xlim(scale=scale)

    def ylabel(
        self, name: str, font_size: Optional[int] = None,
        color: Optional[str] = None,
    ) -> "Figure":
        """Set the primary (left) y-axis title."""
        self._y_axes[0]["name"] = name
        if font_size is not None:
            self._y_axes[0].setdefault("axisLabel", {})["fontSize"] = font_size
        if color is not None:
            self._y_axes[0].setdefault("axisLabel", {})["color"] = color
        return self

    def ylabel_right(
        self, name: str, font_size: Optional[int] = None,
        color: Optional[str] = None,
    ) -> "Figure":
        """Set the secondary (right) y-axis title, creating the axis if needed."""
        if len(self._y_axes) < 2:
            self._y_axes.append({
                "type": "value",
                "splitLine": {"show": False},
                "axisLabel": {
                    "fontSize": self._style.axis_label_font_size,
                    "color": self._style.axis_label_color,
                },
            })
        self._y_axes[1]["name"] = name
        if font_size is not None:
            self._y_axes[1].setdefault("axisLabel", {})["fontSize"] = font_size
        if color is not None:
            self._y_axes[1].setdefault("axisLabel", {})["color"] = color
        return self

    def ylim(
        self, min_val: Optional[float] = None,
        max_val: Optional[float] = None, axis: int = 0,
        scale: Optional[Literal["value", "log"]] = None,
    ) -> "Figure":
        """Set y-axis bounds and/or scale type."""
        if axis < 0:
            raise ValueError("axis index must be non-negative")
        if axis >= len(self._y_axes):
            raise BuilderConfigError(
                f"ylim(axis={axis}) — only {len(self._y_axes)} y-axis(es) exist. "
                "Call ylabel_right() first to create a second axis."
            )
        if min_val is not None:
            self._y_axes[axis]["min"] = min_val
        if max_val is not None:
            self._y_axes[axis]["max"] = max_val
        if scale is not None:
            self._y_axes[axis]["type"] = scale
        return self

    def yscale(self, scale: Literal["value", "log"], axis: int = 0) -> "Figure":
        """Set y-axis scale type (shortcut for ``ylim(scale=...)``).

        >>> fig.yscale("log")
        """
        return self.ylim(scale=scale, axis=axis)

    def yticks(
        self, rotate: Optional[int] = None,
        interval: Optional[Union[int, str]] = None,
        formatter: Optional[str] = None, axis: int = 0,
    ) -> "Figure":
        """Configure y-axis tick display."""
        if axis >= len(self._y_axes):
            raise BuilderConfigError(f"yticks(axis={axis}) — axis does not exist yet.")
        lbl = self._y_axes[axis].setdefault("axisLabel", {})
        if rotate is not None:
            lbl["rotate"] = rotate
        if interval is not None:
            lbl["interval"] = interval
        if formatter is not None:
            lbl["formatter"] = formatter
        return self

    def grid(
        self, show: Optional[bool] = None,
        axis: Literal["x", "y", "both"] = "y",
        style: str = "solid", color: Optional[str] = None,
    ) -> "Figure":
        """Toggle grid-line visibility and styling."""
        targets = []
        if axis in ("y", "both"):
            targets.append(self._y_axes[0])
        if axis in ("x", "both"):
            targets.append(self._x_axis)
        for t in targets:
            sl = t.setdefault("splitLine", {})
            if show is not None:
                sl["show"] = show
            ls = sl.setdefault("lineStyle", {})
            ls["type"] = style
            if color:
                ls["color"] = color
        return self

    def margins(
        self, left: Optional[Union[int, str]] = None,
        right: Optional[Union[int, str]] = None,
        top: Optional[Union[int, str]] = None,
        bottom: Optional[Union[int, str]] = None,
    ) -> "Figure":
        """Set explicit grid margins."""
        if left is not None:
            self._grid_cfg["left"] = left
        if right is not None:
            self._grid_cfg["right"] = right
        if top is not None:
            self._grid_cfg["top"] = top
        if bottom is not None:
            self._grid_cfg["bottom"] = bottom
        return self

    def legend(
        self, show: bool = True,
        orient: Optional[Literal["horizontal", "vertical"]] = None,
        left: Optional[str] = None,
        top: Optional[Union[str, int]] = None,
        bottom: Optional[Union[str, int]] = None,
        text_style_color: Optional[str] = None,
        text_style_font_size: Optional[int] = None,
        icon: Optional[str] = None,
        item_gap: Optional[int] = None,
        item_width: Optional[int] = None,
        item_height: Optional[int] = None,
        background_color: Optional[str] = None,
        border_color: Optional[str] = None,
        border_width: Optional[int] = None,
        padding: Optional[Union[int, list]] = None,
        scroll: Optional[bool] = None,
    ) -> "Figure":
        """Configure legend placement and visibility."""
        cfg: dict = {"show": show}
        cfg["orient"] = orient or self._style.legend_orient
        if left is not None:
            cfg["left"] = left
        if top is not None:
            cfg["top"] = top
        if bottom is not None:
            cfg["bottom"] = bottom
        if text_style_color is not None or text_style_font_size is not None:
            ts: dict = {}
            if text_style_color is not None:
                ts["color"] = text_style_color
            if text_style_font_size is not None:
                ts["fontSize"] = text_style_font_size
            cfg["textStyle"] = ts
        if icon is not None:
            cfg["icon"] = icon
        if item_gap is not None:
            cfg["itemGap"] = item_gap
        if item_width is not None:
            cfg["itemWidth"] = item_width
        if item_height is not None:
            cfg["itemHeight"] = item_height
        if background_color is not None:
            cfg["backgroundColor"] = background_color
        if border_color is not None:
            cfg["borderColor"] = border_color
        if border_width is not None:
            cfg["borderWidth"] = border_width
        if padding is not None:
            cfg["padding"] = padding
        if scroll:
            cfg["type"] = "scroll"
        self._legend_cfg = cfg
        return self

    def tooltip(
        self, trigger: Literal["item", "axis", "none"] = "axis",
        pointer: Literal["cross", "shadow", "line", "none"] = "cross",
        formatter: Optional[str] = None,
        snap: Optional[bool] = None,
        pointer_label: Optional[bool] = None,
        line_color: Optional[str] = None,
        line_width: Optional[int] = None,
        line_type: Optional[Literal["solid", "dashed", "dotted"]] = None,
        cross_color: Optional[str] = None,
        cross_width: Optional[int] = None,
        cross_type: Optional[Literal["solid", "dashed", "dotted"]] = None,
        shadow_color: Optional[str] = None,
        shadow_opacity: Optional[float] = None,
        background_color: Optional[str] = None,
        border_color: Optional[str] = None,
        border_width: Optional[int] = None,
        text_color: Optional[str] = None,
        text_size: Optional[int] = None,
        padding: Optional[Union[int, list]] = None,
        enterable: Optional[bool] = None,
        order: Optional[Literal["seriesAsc", "seriesDesc", "valueAsc", "valueDesc"]] = None,
        render_mode: Optional[Literal["html", "richText"]] = None,
    ) -> "Figure":
        """Configure tooltip behaviour and axis-pointer styling."""
        self._tooltip_cfg["trigger"] = trigger
        ap = _build_axis_pointer_cfg(
            type_val=pointer, snap=snap, pointer_label=pointer_label,
            line_color=line_color, line_width=line_width, line_type=line_type,
            cross_color=cross_color, cross_width=cross_width, cross_type=cross_type,
            shadow_color=shadow_color, shadow_opacity=shadow_opacity,
        )
        self._tooltip_cfg.setdefault("axisPointer", {}).update(ap)
        if formatter:
            self._tooltip_cfg["formatter"] = formatter
        if background_color is not None:
            self._tooltip_cfg["backgroundColor"] = background_color
        if border_color is not None:
            self._tooltip_cfg["borderColor"] = border_color
        if border_width is not None:
            self._tooltip_cfg["borderWidth"] = border_width
        if text_color is not None:
            self._tooltip_cfg.setdefault("textStyle", {})["color"] = text_color
        if text_size is not None:
            self._tooltip_cfg.setdefault("textStyle", {})["fontSize"] = text_size
        if padding is not None:
            self._tooltip_cfg["padding"] = padding
        if enterable is not None:
            self._tooltip_cfg["enterable"] = enterable
        if order is not None:
            self._tooltip_cfg["order"] = order
        if render_mode is not None:
            self._tooltip_cfg["renderMode"] = render_mode
        return self

    def axis_pointer(
        self,
        type: Literal["line", "shadow", "cross", "none"] = "line",
        snap: Optional[bool] = None,
        label: Optional[bool] = None,
        label_precision: Optional[int] = None,
        label_bg: Optional[str] = None,
        label_color: Optional[str] = None,
        line_color: Optional[str] = None,
        line_width: Optional[int] = None,
        line_type: Optional[Literal["solid", "dashed", "dotted"]] = None,
        cross_color: Optional[str] = None,
        cross_width: Optional[int] = None,
        cross_type: Optional[Literal["solid", "dashed", "dotted"]] = None,
        shadow_color: Optional[str] = None,
        shadow_opacity: Optional[float] = None,
    ) -> "Figure":
        """Configure the global axis pointer (top-level ``option.axisPointer``).

        Also syncs type and styles into ``tooltip.axisPointer``.
        If :meth:`tooltip` is called afterwards, it overwrites the
        tooltip-level pointer (last call wins).
        """
        cfg = _build_axis_pointer_cfg(
            type_val=type, snap=snap, pointer_label=label,
            label_precision=label_precision, label_bg=label_bg,
            label_color=label_color,
            line_color=line_color, line_width=line_width, line_type=line_type,
            cross_color=cross_color, cross_width=cross_width, cross_type=cross_type,
            shadow_color=shadow_color, shadow_opacity=shadow_opacity,
        )
        self._axis_pointer_cfg = cfg
        self._tooltip_cfg["axisPointer"] = copy.deepcopy(cfg)
        return self

    def save(
        self, name: str = "chart",
        fmt: Literal["png", "svg"] = "png",
        dpi: int = 3, bg: Optional[str] = "#ffffff",
    ) -> "Figure":
        """Enable the download button in the ECharts toolbox."""
        self._toolbox_cfg = {
            "show": True, "right": 10, "top": 10,
            "feature": {
                "saveAsImage": {
                    "show": True, "type": fmt, "name": name,
                    "pixelRatio": dpi, "backgroundColor": bg,
                },
                "restore": {"show": True},
            },
        }
        return self

    def toolbox(
        self, download: bool = True, zoom: bool = False,
        restore: bool = True,
        data_view: Optional[bool] = None,
        magic_type: Optional[List[str]] = None,
        orient: Optional[Literal["horizontal", "vertical"]] = None,
        item_size: Optional[int] = None,
        item_gap: Optional[int] = None,
    ) -> "Figure":
        """Configure which toolbox features are visible."""
        features: dict = {}
        if download:
            features["saveAsImage"] = {"show": True, "pixelRatio": 3}
        if zoom:
            features["dataZoom"] = {"show": True}
        if restore:
            features["restore"] = {"show": True}
        if data_view:
            features["dataView"] = {"show": True}
        if magic_type is not None:
            features["magicType"] = {"show": True, "type": magic_type}
        cfg: dict = {"show": True, "right": 10, "top": 10, "feature": features}
        if orient is not None:
            cfg["orient"] = orient
        if item_size is not None:
            cfg["itemSize"] = item_size
        if item_gap is not None:
            cfg["itemGap"] = item_gap
        self._toolbox_cfg = cfg
        return self

    def datazoom(
        self, start: int = 0, end: int = 100,
        orient: Literal["horizontal", "vertical"] = "horizontal",
        show_slider: bool = True,
        grids: Optional[List[int]] = None,
        zoom_lock: Optional[bool] = None,
        throttle: Optional[int] = None,
        range_mode: Optional[list] = None,
        fill_color: Optional[str] = None,
        border_color: Optional[str] = None,
        handle_color: Optional[str] = None,
    ) -> "Figure":
        """Add a data-zoom slider for long time-series or many categories.

        Parameters
        ----------
        grids : list[int] or None
            When using multi-grid subplots, link zoom across these grid
            indices (e.g. ``grids=[0, 1]``).
        """
        if not (0 <= start <= 100) or not (0 <= end <= 100):
            raise ValueError("datazoom start/end must be between 0 and 100")
        inside: dict = {"type": "inside", "start": start, "end": end, "orient": orient}
        slider: dict = {"type": "slider", "start": start, "end": end, "orient": orient}
        if grids and self._n_grids > 1:
            inside["xAxisIndex"] = grids
            slider["xAxisIndex"] = grids
        if zoom_lock is not None:
            inside["zoomLock"] = zoom_lock
            slider["zoomLock"] = zoom_lock
        if range_mode is not None:
            inside["rangeMode"] = range_mode
            slider["rangeMode"] = range_mode
        if throttle is not None:
            slider["throttle"] = throttle
        if fill_color is not None:
            slider["fillerColor"] = fill_color
        if border_color is not None:
            slider["borderColor"] = border_color
        if handle_color is not None:
            slider["handleStyle"] = {"color": handle_color}
        zoom: list = [inside]
        if show_slider:
            zoom.append(slider)
        self._datazoom_cfg = zoom
        return self

    def palette(self, colors: Sequence[str]) -> "Figure":
        """Set the global colour cycle for all series."""
        self._palette = list(colors)
        return self

    def visual_map(
        self, min_val: float = 0, max_val: float = 100,
        colors: Optional[List[str]] = None,
        orient: Literal["horizontal", "vertical"] = "vertical",
        position: Optional[str] = None,
        calculable: bool = True,
        piecewise: bool = False,
        pieces: Optional[List[dict]] = None,
        dimension: Optional[int] = None,
        series_index: Optional[Union[int, list]] = None,
        out_of_range: Optional[dict] = None,
        realtime: Optional[bool] = None,
        text: Optional[list] = None,
        inverse: Optional[bool] = None,
    ) -> "Figure":
        """Expose the ECharts ``visualMap`` component for colour mapping.

        Parameters
        ----------
        min_val, max_val : float
            Data range for continuous mapping.
        colors : list[str]
            Colour gradient stops (low → high).
        orient : ``"horizontal"`` | ``"vertical"``
            Legend orientation.
        position : ``"left"`` | ``"right"`` | ``None``
            Shorthand positioning.
        calculable : bool
            Allow interactive range dragging.
        piecewise : bool
            Use piecewise (discrete) mapping instead of continuous.
        pieces : list[dict]
            Explicit piece definitions when *piecewise* is True.
        """
        cfg: dict = {}
        if piecewise:
            cfg["type"] = "piecewise"
            if pieces:
                cfg["pieces"] = pieces
        else:
            cfg["type"] = "continuous"
            cfg["min"] = min_val
            cfg["max"] = max_val
        cfg["calculable"] = calculable
        cfg["orient"] = orient
        if position == "right":
            cfg["right"] = 10
        elif position == "left":
            cfg["left"] = 10
        else:
            cfg["left"] = "center"
        if colors:
            cfg["inRange"] = {"color": colors}
        if dimension is not None:
            cfg["dimension"] = dimension
        if series_index is not None:
            cfg["seriesIndex"] = series_index
        if out_of_range is not None:
            cfg["outOfRange"] = out_of_range
        if realtime is not None:
            cfg["realtime"] = realtime
        if text is not None:
            cfg["text"] = text
        if inverse is not None:
            cfg["inverse"] = inverse
        self._extra["visualMap"] = cfg
        return self

    def extra(self, **kwargs: Any) -> "Figure":
        """Merge arbitrary top-level ECharts options (escape hatch)."""
        self._extra.update(kwargs)
        return self

    def raw_series(self, config: dict) -> "Figure":
        """Inject a raw ECharts series dict (escape hatch for composite charts).

        Use this to overlay chart types that would normally conflict —
        for example, a small pie chart on top of a bar chart.

        The ``config`` dict is appended directly to the series list with
        no mode-locking or validation.

        Parameters
        ----------
        config : dict
            A raw ECharts series configuration, e.g.
            ``{"type": "pie", "radius": "30%", "center": ["75%", "25%"],
            "data": [{"name": "A", "value": 10}, ...]}``

        Returns
        -------
        Figure
            Self, for chaining.

        Example
        -------
        >>> (ec.Figure()
        ...  .bar(df, x="Month", y="Revenue")
        ...  .raw_series({
        ...      "type": "pie", "radius": "25%",
        ...      "center": ["80%", "25%"],
        ...      "data": [{"name": "A", "value": 40}, {"name": "B", "value": 60}],
        ...      "label": {"fontSize": 10},
        ...  })
        ...  .show())
        """
        self._series.append(config)
        self._series_meta.append(_SeriesMeta(
            chart_type=config.get("type", "custom"),
            name=config.get("name", ""),
            raw_config=config,
        ))
        # Extend legend if the raw series has named data items
        if "name" in config and config["name"]:
            self._legend_items.append(config["name"])
        return self

    # ═══════════════════════════════════════════════════════════════════════
    # Series methods
    # ═══════════════════════════════════════════════════════════════════════

    # ─── LINE ─────────────────────────────────────────────────────────────

    def plot(
        self, df: pd.DataFrame, x: str, y: str, *,
        hue: Optional[str] = None, smooth: bool = False,
        area: bool = False, area_opacity: float = 0.15,
        connect_nulls: bool = False, line_width: int = 2,
        symbol_size: int = 6, symbol: str = "circle",
        labels: bool = False, label_position: str = "top",
        label_prefix: str = "", label_suffix: str = "",
        agg: str = "mean", axis: int = 0, grid: int = 0,
        emphasis: Optional[LineEmphasis] = None,
        line_style: Optional[LineStyle] = None,
        area_style: Optional[AreaStyle] = None,
        label_style: Optional[LabelStyle] = None,
        end_label: Optional[EndLabelStyle] = None,
        show_symbol: Optional[bool] = None,
        color: Optional[str] = None,
        blur: Optional[Blur] = None,
        select: Optional[Select] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        animation: Optional[AnimationConfig] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add one or more line series — equivalent to ``plt.plot()``.

        Parameters
        ----------
        df : pd.DataFrame
            Source data.
        x : str
            Category-axis column.
        y : str
            Value-axis column.
        hue : str or None
            Grouping column — one line per unique value.
        smooth : bool
            Bezier-smooth the line.
        area : bool
            Fill the area under the line.
        """
        self._ensure_cartesian("plot")
        df = _validate_df(df, "plot")
        _validate_columns(df, [x, y, hue], "plot")

        dff = df.copy()
        dff[x] = dff[x].astype(str).str.strip()
        dff[y] = _coerce_numeric(dff, y, "plot")
        if hue:
            dff[hue] = dff[hue].astype(str).str.strip()
            dff = dff.dropna(subset=[hue])
        n_before = len(dff)
        dff = dff.dropna(subset=[x, y])
        n_dropped = n_before - len(dff)
        if n_dropped > 0:
            warnings.warn(f"plot(): {n_dropped} rows dropped due to missing values", stacklevel=2)
        if dff.empty:
            warnings.warn("plot(): all rows dropped after removing missing values; chart will be empty", stacklevel=2)
            return self

        cats = _sort_categories(dff[x])
        self._merge_x_categories(cats, grid=grid)

        if axis == 1 and len(self._y_axes) < 2:
            self.ylabel_right("")

        y_idx = axis + grid * 2 if self._n_grids > 1 else axis
        x_idx = grid if self._n_grids > 1 else 0
        base: dict = {
            "type": "line", "smooth": smooth,
            "connectNulls": connect_nulls, "showSymbol": True,
            "symbol": symbol, "symbolSize": symbol_size,
            "lineStyle": {"width": line_width}, "yAxisIndex": y_idx,
        }
        if self._n_grids > 1:
            base["xAxisIndex"] = x_idx
        if area:
            base["areaStyle"] = {"opacity": area_opacity}
        if labels:
            base["label"] = {
                "show": True, "position": label_position,
                "formatter": f"{label_prefix}{{c}}{label_suffix}",
            }
        if emphasis is not None:
            base["emphasis"] = emphasis.to_dict()
        _merge_style_params(
            base, line_style=line_style, area_style=area_style,
            label_style=label_style, end_label=end_label,
            show_symbol=show_symbol, color=color, blur=blur,
            select=select, selected_mode=selected_mode,
            animation=animation, tooltip=tooltip,
        )
        base.update(series_kw)

        groups = dff.groupby(hue) if hue else [(y, dff)]
        for name, grp in groups:
            name_str = str(name)
            values = self._align_to_categories(grp, x, y, agg, grid=grid)
            entry = {**base, "name": name_str, "data": values}
            self._series.append(entry)
            self._series_meta.append(_SeriesMeta("line", name_str, axis, grid_index=grid))
            if name_str not in self._legend_items:
                self._legend_items.append(name_str)

        return self

    # ─── BAR ──────────────────────────────────────────────────────────────

    def bar(
        self, df: pd.DataFrame, x: str, y: str, *,
        hue: Optional[str] = None, stack: bool = False,
        orient: Literal["v", "h"] = "v",
        bar_width: Optional[Union[int, str]] = None,
        bar_gap: Optional[str] = None,
        border_radius: int = 4, labels: bool = False,
        label_formatter: str = "{c}", label_font_size: int = 12,
        label_color: str = "#333",
        gradient: bool = False,
        gradient_colors: Tuple[str, str] = ("#83bff6", "#188df0"),
        agg: str = "sum", axis: int = 0, grid: int = 0,
        emphasis: Optional[Emphasis] = None,
        item_style: Optional[ItemStyle] = None,
        label_style: Optional[LabelStyle] = None,
        bar_min_width: Optional[Union[int, str]] = None,
        bar_category_gap: Optional[str] = None,
        color: Optional[str] = None,
        blur: Optional[Blur] = None,
        select: Optional[Select] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        animation: Optional[AnimationConfig] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add one or more bar series — equivalent to ``plt.bar()``.

        Parameters
        ----------
        df : pd.DataFrame
        x : str
            Category column.
        y : str
            Value column.
        hue : str or None
            Grouping column.
        stack : bool
            Stack bars of different hue values.
        orient : ``"v"`` or ``"h"``
            ``"v"`` = vertical bars, ``"h"`` = horizontal bars.
        """
        self._ensure_cartesian("bar")
        df = _validate_df(df, "bar")
        _validate_columns(df, [x, y, hue], "bar")

        dff = df.copy()
        dff[x] = dff[x].astype(str).str.strip()
        dff[y] = _coerce_numeric(dff, y, "bar")
        if hue:
            dff[hue] = dff[hue].astype(str).str.strip()
            dff = dff.dropna(subset=[hue])
        n_before = len(dff)
        dff = dff.dropna(subset=[x, y])
        n_dropped = n_before - len(dff)
        if n_dropped > 0:
            warnings.warn(f"bar(): {n_dropped} rows dropped due to missing values", stacklevel=2)
        if dff.empty:
            warnings.warn("bar(): all rows dropped after removing missing values; chart will be empty", stacklevel=2)
            return self

        if orient == "h":
            self._x_axis["type"] = "value"
            cats = _sort_categories(dff[x])
            self._merge_x_categories(cats, grid=grid)
        else:
            cats = _sort_categories(dff[x])
            self._merge_x_categories(cats, grid=grid)

        if axis == 1 and len(self._y_axes) < 2:
            self.ylabel_right("")

        label_pos = "top" if orient == "v" else "right"
        bar_item_style: dict = {"borderRadius": border_radius}
        if gradient:
            if len(gradient_colors) != 2:
                raise ValueError("gradient_colors must be a tuple of exactly 2 color strings")
            bar_item_style["color"] = {
                "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                "colorStops": [
                    {"offset": 0, "color": gradient_colors[0]},
                    {"offset": 1, "color": gradient_colors[1]},
                ],
            }

        y_idx = axis + grid * 2 if self._n_grids > 1 else axis
        x_idx = grid if self._n_grids > 1 else 0
        base: dict = {
            "type": "bar",
            "label": {
                "show": labels, "position": label_pos,
                "formatter": label_formatter, "fontSize": label_font_size,
                "color": label_color,
            },
            "itemStyle": bar_item_style, "yAxisIndex": y_idx,
        }
        if self._n_grids > 1:
            base["xAxisIndex"] = x_idx
        if stack:
            base["stack"] = "total"
        if bar_width is not None:
            base["barWidth"] = bar_width
        if bar_gap is not None:
            base["barGap"] = bar_gap
        if bar_min_width is not None:
            base["barMinWidth"] = bar_min_width
        if bar_category_gap is not None:
            base["barCategoryGap"] = bar_category_gap
        if emphasis is not None:
            base["emphasis"] = emphasis.to_dict()
        _merge_style_params(
            base, item_style=item_style, label_style=label_style,
            color=color, blur=blur, select=select,
            selected_mode=selected_mode, animation=animation,
            tooltip=tooltip,
        )
        base.update(series_kw)

        groups = dff.groupby(hue) if hue else [(y, dff)]
        for name, grp in groups:
            name_str = str(name)
            values = self._align_to_categories(grp, x, y, agg, grid=grid)
            entry = {**base, "name": name_str, "data": values}
            self._series.append(entry)
            self._series_meta.append(_SeriesMeta("bar", name_str, axis, grid_index=grid))
            if name_str not in self._legend_items:
                self._legend_items.append(name_str)

        if orient == "h":
            self._extra["_bar_orient_h"] = True

        return self

    # ─── WATERFALL ─────────────────────────────────────────────────────────

    def waterfall(
        self, df: pd.DataFrame, x: str, y: str, *,
        positive_color: str = "#4ade80",
        negative_color: str = "#f87171",
        total: bool = False,
        total_label: str = "Total",
        total_color: str = "#60a5fa",
        connector: bool = True,
        connector_color: str = "#999",
        connector_width: float = 1,
        connector_dash: str = "dashed",
        border_radius: int = 4,
        labels: bool = False,
        label_formatter: str = "{c}",
        label_font_size: int = 12,
        label_color: Optional[str] = None,
        agg: str = "sum",
        axis: int = 0,
        grid: int = 0,
        emphasis: Optional[Emphasis] = None,
        label_style: Optional[LabelStyle] = None,
        item_style: Optional[ItemStyle] = None,
        animation: Optional[AnimationConfig] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a waterfall chart — cumulative positive/negative deltas.

        Uses stacked transparent bars under the hood: an invisible base bar
        plus colored positive/negative delta bars on top.

        Parameters
        ----------
        df : pd.DataFrame
        x : str
            Category column.
        y : str
            Value column (deltas).
        positive_color : str
            Color for positive delta bars.
        negative_color : str
            Color for negative delta bars.
        total : bool
            Append a final total bar.
        total_label : str
            Label for the total bar.
        total_color : str
            Color for the total bar.
        connector : bool
            Draw dashed connector lines between bar tops.
        labels : bool
            Show value labels on bars.
        """
        self._ensure_cartesian("waterfall")
        df = _validate_df(df, "waterfall")
        _validate_columns(df, [x, y], "waterfall")

        dff = df.copy()
        dff[x] = dff[x].astype(str).str.strip()
        dff[y] = _coerce_numeric(dff, y, "waterfall")
        n_before = len(dff)
        dff = dff.dropna(subset=[x, y])
        n_dropped = n_before - len(dff)
        if n_dropped > 0:
            warnings.warn(f"waterfall(): {n_dropped} rows dropped due to missing values", stacklevel=2)
        if dff.empty:
            warnings.warn("waterfall(): all rows dropped after removing missing values; chart will be empty", stacklevel=2)
            return self

        cats = _sort_categories(dff[x])
        self._merge_x_categories(cats, grid=grid)

        # Align values to the merged category order
        values = self._align_to_categories(dff, x, y, agg, grid=grid)

        # ── Compute waterfall base / positive / negative arrays ──
        all_cats = self._grid_categories.get(grid, self._x_categories)
        base_data: List[float] = []
        pos_data: List[float] = []
        neg_data: List[float] = []
        running = 0.0
        for v in values:
            val = v if v is not None else 0.0
            if val >= 0:
                base_data.append(round(running, 4))
                pos_data.append(round(val, 4))
                neg_data.append(0)
            else:
                base_data.append(round(running + val, 4))
                pos_data.append(0)
                neg_data.append(round(abs(val), 4))
            running += val

        # Append total bar
        if total:
            if total_label not in set(all_cats):
                all_cats.append(total_label)
            base_data.append(0)
            pos_data.append(round(running, 4) if running >= 0 else 0)
            neg_data.append(round(abs(running), 4) if running < 0 else 0)

        if axis == 1 and len(self._y_axes) < 2:
            self.ylabel_right("")

        y_idx = axis + grid * 2 if self._n_grids > 1 else axis
        x_idx = grid if self._n_grids > 1 else 0
        stack_name = f"waterfall_{len(self._series)}"

        # ── Base series (invisible) ──
        base_series: dict = {
            "type": "bar",
            "name": "",
            "stack": stack_name,
            "data": base_data,
            "itemStyle": {"color": "transparent", "borderColor": "transparent"},
            "emphasis": {"disabled": True},
            "tooltip": {"show": False},
            "yAxisIndex": y_idx,
        }
        if self._n_grids > 1:
            base_series["xAxisIndex"] = x_idx

        # ── Connector markLines on the base series ──
        if connector:
            mark_lines = []
            n_bars = len(base_data)
            for i in range(n_bars - 1):
                # Running total after bar i = base + positive (neg bars: base already = running_after)
                top_i = base_data[i] + pos_data[i]
                mark_lines.append([
                    {"xAxis": i, "yAxis": round(top_i, 4)},
                    {"xAxis": i + 1, "yAxis": round(top_i, 4)},
                ])
            if mark_lines:
                base_series["markLine"] = {
                    "symbol": "none",
                    "lineStyle": {
                        "color": connector_color,
                        "width": connector_width,
                        "type": connector_dash,
                    },
                    "label": {"show": False},
                    "data": mark_lines,
                    "silent": True,
                    "animation": False,
                }

        self._series.append(base_series)
        self._series_meta.append(_SeriesMeta("bar", "", axis, grid_index=grid))

        # ── Positive series ──
        pos_item_style: dict = {"color": positive_color, "borderRadius": border_radius}
        # If total=True, color the last bar with total_color
        pos_series_data: list = list(pos_data)
        if total and pos_series_data:
            last_val = pos_series_data[-1]
            if last_val > 0:
                pos_series_data[-1] = {
                    "value": last_val,
                    "itemStyle": {"color": total_color},
                }

        pos_label_cfg: dict = {
            "show": labels, "position": "top",
            "formatter": label_formatter, "fontSize": label_font_size,
            "color": label_color if label_color else positive_color,
        }
        pos_series: dict = {
            "type": "bar",
            "name": "Positive",
            "stack": stack_name,
            "data": pos_series_data,
            "itemStyle": pos_item_style,
            "label": pos_label_cfg,
            "yAxisIndex": y_idx,
        }
        if self._n_grids > 1:
            pos_series["xAxisIndex"] = x_idx
        if emphasis is not None:
            pos_series["emphasis"] = emphasis.to_dict()
        if label_style is not None:
            pos_series.setdefault("label", {}).update(label_style.to_dict())
        if item_style is not None:
            pos_series.setdefault("itemStyle", {}).update(item_style.to_dict())
        if animation is not None:
            pos_series.update(animation.to_dict())
        if selected_mode is not None:
            pos_series["selectedMode"] = selected_mode
        if tooltip is not None:
            pos_series["tooltip"] = tooltip.to_dict()
        pos_series.update({k: v for k, v in series_kw.items() if k not in ("name", "data", "stack", "itemStyle", "label")})
        self._series.append(pos_series)
        self._series_meta.append(_SeriesMeta("bar", "Positive", axis, grid_index=grid))

        # ── Negative series ──
        neg_item_style: dict = {"color": negative_color, "borderRadius": border_radius}
        neg_series_data: list = list(neg_data)
        if total and neg_series_data:
            last_val = neg_series_data[-1]
            if last_val > 0:
                neg_series_data[-1] = {
                    "value": last_val,
                    "itemStyle": {"color": total_color},
                }

        neg_label_cfg: dict = {
            "show": labels, "position": "bottom",
            "formatter": label_formatter, "fontSize": label_font_size,
            "color": label_color if label_color else negative_color,
        }
        neg_series: dict = {
            "type": "bar",
            "name": "Negative",
            "stack": stack_name,
            "data": neg_series_data,
            "itemStyle": neg_item_style,
            "label": neg_label_cfg,
            "yAxisIndex": y_idx,
        }
        if self._n_grids > 1:
            neg_series["xAxisIndex"] = x_idx
        if emphasis is not None:
            neg_series["emphasis"] = emphasis.to_dict()
        if label_style is not None:
            neg_series.setdefault("label", {}).update(label_style.to_dict())
        if item_style is not None:
            neg_series.setdefault("itemStyle", {}).update(item_style.to_dict())
        if animation is not None:
            neg_series.update(animation.to_dict())
        if selected_mode is not None:
            neg_series["selectedMode"] = selected_mode
        if tooltip is not None:
            neg_series["tooltip"] = tooltip.to_dict()
        neg_series.update({k: v for k, v in series_kw.items() if k not in ("name", "data", "stack", "itemStyle", "label")})
        self._series.append(neg_series)
        self._series_meta.append(_SeriesMeta("bar", "Negative", axis, grid_index=grid))

        # Add legend items (skip the invisible base)
        for lbl in ("Positive", "Negative"):
            if lbl not in self._legend_items:
                self._legend_items.append(lbl)
        if total and total_label not in self._legend_items:
            self._legend_items.append(total_label)

        return self

    def barh(self, df: pd.DataFrame, x: str, y: str, **kwargs: Any) -> "Figure":
        """Add horizontal bars — shortcut for ``bar(..., orient="h")``.

        Mirrors the matplotlib ``barh()`` convention.
        All keyword arguments are forwarded to :meth:`bar`.
        """
        kwargs.pop("orient", None)
        return self.bar(df, x=x, y=y, orient="h", **kwargs)

    # ─── SCATTER ──────────────────────────────────────────────────────────

    def scatter(
        self, df: pd.DataFrame, x: str, y: str, *,
        color: Optional[str] = None, size: Optional[str] = None,
        size_range: Tuple[int, int] = (5, 30), symbol: str = "circle",
        opacity: float = 0.7, labels: bool = False, grid: int = 0,
        emphasis: Optional[ScatterEmphasis] = None,
        item_style: Optional[ItemStyle] = None,
        label_style: Optional[LabelStyle] = None,
        symbol_rotate: Optional[int] = None,
        show_symbol: Optional[bool] = None,
        blur: Optional[Blur] = None,
        select: Optional[Select] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        animation: Optional[AnimationConfig] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a scatter series — equivalent to ``plt.scatter()``."""
        self._ensure_cartesian("scatter")
        df = _validate_df(df, "scatter")
        _validate_columns(df, [x, y, color, size], "scatter")

        dff = df.copy()
        dff[x] = _coerce_numeric(dff, x, "scatter")
        dff[y] = _coerce_numeric(dff, y, "scatter")
        n_before = len(dff)
        dff = dff.dropna(subset=[x, y])
        n_dropped = n_before - len(dff)
        if n_dropped > 0:
            warnings.warn(f"scatter(): {n_dropped} rows dropped due to missing values", stacklevel=2)
        if dff.empty:
            warnings.warn("scatter(): all rows dropped after removing missing values; chart will be empty", stacklevel=2)
            return self

        if not self._x_categories:
            self._x_axis["type"] = "value"
            self._x_axis.pop("data", None)
        self._tooltip_cfg["trigger"] = "item"

        def _build_scatter_data(sub: pd.DataFrame) -> List[list]:
            return [[float(r[x]), float(r[y])] for _, r in sub.iterrows()]

        def _calc_symbol_sizes(sub: pd.DataFrame) -> Optional[List[int]]:
            if size is None:
                return None
            vals = pd.to_numeric(sub[size], errors="coerce").fillna(0)
            if vals.max() == vals.min():
                return [int((size_range[0] + size_range[1]) / 2)] * len(vals)
            normed = (vals - vals.min()) / (vals.max() - vals.min())
            return [int(size_range[0] + n * (size_range[1] - size_range[0])) for n in normed]

        groups = dff.groupby(color) if color else [("scatter", dff)]
        for name, grp in groups:
            name_str = str(name)
            data = _build_scatter_data(grp)
            entry: dict = {
                "type": "scatter", "name": name_str, "data": data,
                "symbol": symbol, "itemStyle": {"opacity": opacity},
            }
            sizes = _calc_symbol_sizes(grp)
            if sizes is not None:
                entry["symbolSize"] = int(np.mean(sizes)) if len(set(sizes)) == 1 else sizes
            else:
                entry["symbolSize"] = 8
            if labels:
                entry["label"] = {"show": True, "position": "top", "formatter": "{@[1]}"}
            if symbol_rotate is not None:
                entry["symbolRotate"] = symbol_rotate
            if emphasis is not None:
                entry["emphasis"] = emphasis.to_dict()
            _merge_style_params(
                entry, item_style=item_style, label_style=label_style,
                show_symbol=show_symbol, blur=blur, select=select,
                selected_mode=selected_mode, animation=animation,
                tooltip=tooltip,
            )
            entry.update(series_kw)
            self._series.append(entry)
            self._series_meta.append(_SeriesMeta("scatter", name_str))
            if color and name_str not in self._legend_items:
                self._legend_items.append(name_str)

        return self

    # ─── PIE ──────────────────────────────────────────────────────────────

    def pie(
        self, df: pd.DataFrame, names: str, values: str, *,
        inner_radius: Optional[str] = None, outer_radius: str = "60%",
        center: Optional[List[str]] = None,
        radius: Optional[Union[str, List[str]]] = None,
        link_legend: Optional[bool] = None,
        border_radius: int = 0, start_angle: int = 45,
        label_inside: bool = False, label_outside: bool = True,
        label_formatter: str = "{b}: {c} ({d}%)",
        label_font_size: Optional[int] = None,
        center_on_hover: bool = False,
        center_formatter: str = "{b}\n{c}",
        center_font_size: int = 18,
        rose_type: Optional[Literal["radius", "area"]] = None,
        emphasis: Optional[PieEmphasis] = None,
        min_angle: Optional[int] = None,
        min_show_label_angle: Optional[int] = None,
        selected_offset: Optional[int] = None,
        clockwise: Optional[bool] = None,
        avoid_label_overlap: Optional[bool] = None,
        item_style: Optional[ItemStyle] = None,
        animation_type: Optional[Literal["expansion", "scale"]] = None,
        blur: Optional[Blur] = None,
        select: Optional[Select] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        animation: Optional[AnimationConfig] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a pie (or donut) chart — equivalent to ``plt.pie()``.

        When called on a figure that already has cartesian series (bar, line,
        etc.), the pie is automatically rendered as an **overlay**.  If
        *center* and *radius* are not provided, sensible defaults are used
        (top-right mini pie at ``["82%", "25%"]`` with radius
        ``["15%", "28%"]``).  Explicit values always take precedence.

        Parameters
        ----------
        center : list[str], optional
            ``["x%", "y%"]`` position of the pie centre inside the chart
            area (e.g. ``["80%", "25%"]``).  Defaults to ``["82%", "25%"]``
            when overlaying on a cartesian figure.
        radius : str | list[str], optional
            Shorthand for ``[inner_radius, outer_radius]``.  Accepts a
            single string (``"30%"``) or a two-element list
            (``["15%", "25%"]``).  Overrides *inner_radius* / *outer_radius*
            when provided.  Defaults to ``["15%", "28%"]`` for overlays.
        link_legend : bool, optional
            Controls whether the pie shares legend colors with existing
            series.  ``True`` forces shared colors, ``False`` forces
            distinct (offset) colors.  Default (``None``) auto-detects:
            shared when slice names overlap with existing legend entries,
            offset otherwise.
        label_font_size : int, optional
            Override the label font size (handy for small overlay pies).
        """
        # ── Overlay mode: allow pie on top of cartesian charts ─────────
        _OVERLAY_COMPATIBLE = ("cartesian",)
        _on_cartesian = (
            self._chart_mode is not None
            and self._chart_mode != "pie"
        )

        if _on_cartesian:
            if self._chart_mode not in _OVERLAY_COMPATIBLE:
                raise BuilderConfigError(
                    f"pie() overlay is only supported on cartesian figures "
                    f"(bar/line/scatter), but the current mode is "
                    f"'{self._chart_mode}'. Use a separate Figure."
                )
            if center is None:
                center = ["82%", "25%"]
            if radius is None and inner_radius is None:
                radius = ["15%", "28%"]
            is_overlay = True
        else:
            is_overlay = False
            self._ensure_mode("pie", "pie")

        df = _validate_df(df, "pie")
        _validate_columns(df, [names, values], "pie")

        dff = df.copy()
        dff[values] = _coerce_numeric(dff, values, "pie")
        dff = dff.dropna(subset=[names, values])

        data = [
            {"name": str(n), "value": round(float(v), 4)}
            for n, v in zip(dff[names], dff[values])
        ]

        # Resolve radius: explicit `radius` param > inner/outer pair > default
        if radius is not None:
            resolved_radius = radius
        elif inner_radius:
            resolved_radius = [inner_radius, outer_radius]
        else:
            resolved_radius = outer_radius

        if label_inside:
            lbl = {"show": True, "position": "inside", "formatter": label_formatter}
            lbl_line = {"show": False}
        elif label_outside:
            lbl = {"show": True, "position": "outside", "formatter": label_formatter}
            lbl_line = {"show": True, "length": 15, "length2": 10}
        else:
            lbl = {"show": False}
            lbl_line = {"show": False}

        if label_font_size is not None:
            lbl["fontSize"] = label_font_size

        emphasis_dict: dict = {
            "itemStyle": {
                "shadowBlur": 10, "shadowOffsetX": 0,
                "shadowColor": "rgba(0, 0, 0, 0.5)",
            }
        }
        if center_on_hover:
            emphasis_dict["label"] = {
                "show": True, "position": "center",
                "formatter": center_formatter,
                "fontSize": center_font_size, "fontWeight": "bold",
            }

        # Allow user-provided emphasis to override the default, but keep overlay focus
        if emphasis is not None:
            emphasis_dict = emphasis.to_dict()
            if is_overlay and "focus" not in emphasis_dict:
                emphasis_dict["focus"] = "self"
        elif is_overlay:
            # For overlays with default emphasis, also add "focus self" so hovering the pie doesn't
            # dim the underlying cartesian series.
            emphasis_dict["focus"] = "self"

        entry: dict = {
            "type": "pie", "radius": resolved_radius, "startAngle": start_angle,
            "data": data, "avoidLabelOverlap": True,
            "label": lbl, "labelLine": lbl_line, "emphasis": emphasis_dict,
        }
        if min_angle is not None:
            entry["minAngle"] = min_angle
        if min_show_label_angle is not None:
            entry["minShowLabelAngle"] = min_show_label_angle
        if selected_offset is not None:
            entry["selectedOffset"] = selected_offset
        if clockwise is not None:
            entry["clockwise"] = clockwise
        if avoid_label_overlap is not None:
            entry["avoidLabelOverlap"] = avoid_label_overlap
        if animation_type is not None:
            entry["animationType"] = animation_type
        if item_style is not None:
            entry.setdefault("itemStyle", {}).update(item_style.to_dict())
        if blur is not None:
            entry["blur"] = blur.to_dict()
        if select is not None:
            entry["select"] = select.to_dict()
        if selected_mode is not None:
            entry["selectedMode"] = selected_mode
        if animation is not None:
            entry.update(animation.to_dict())
        if tooltip is not None:
            entry["tooltip"] = tooltip.to_dict()
        if center is not None:
            entry["center"] = center
        if border_radius:
            entry.setdefault("itemStyle", {})["borderRadius"] = border_radius
            entry["itemStyle"].setdefault("borderColor", "#fff")
            entry["itemStyle"].setdefault("borderWidth", 2)
        if rose_type:
            entry["roseType"] = rose_type
        entry.update(series_kw)

        # ── Overlay color offset ──────────────────────────────────────────
        # When the pie's categories differ from the existing legend items,
        # offset slice colours so they don't collide with the bar/line palette.
        # `link_legend` lets the user override the auto-detection.
        if is_overlay:
            pie_names_set = {str(n) for n in dff[names]}
            existing_names_set = set(self._legend_items)
            names_overlap = bool(pie_names_set & existing_names_set)

            if link_legend is None:
                effectively_linked = names_overlap
            else:
                effectively_linked = link_legend

            if not effectively_linked:
                pal = list(self._palette or (
                    "#5470C6", "#91CC75", "#FAC858", "#EE6666",
                    "#73C0DE", "#3BA272", "#FC8452", "#9A60B4", "#EA7CCC",
                ))
                offset = len(existing_names_set)
                for i, item in enumerate(data):
                    item["itemStyle"] = item.get("itemStyle", {})
                    item["itemStyle"]["color"] = pal[(offset + i) % len(pal)]

        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("pie", names))

        pie_names = [str(n) for n in dff[names]]
        if is_overlay:
            for n in pie_names:
                if n not in self._legend_items:
                    self._legend_items.append(n)
        else:
            for n in pie_names:
                if n not in self._legend_items:
                    self._legend_items.append(n)

        return self

    # ─── HISTOGRAM ────────────────────────────────────────────────────────

    def hist(
        self, df: pd.DataFrame, column: str, *,
        bins: int = 10, density: bool = False,
        bar_color: Optional[str] = None,
        border_radius: int = 2, labels: bool = False, grid: int = 0,
        emphasis: Optional[Emphasis] = None,
        item_style: Optional[ItemStyle] = None,
        label_style: Optional[LabelStyle] = None,
        animation: Optional[AnimationConfig] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a histogram — equivalent to ``plt.hist()``."""
        self._ensure_cartesian("hist")
        df = _validate_df(df, "hist")
        _validate_columns(df, [column], "hist")

        if bins is not None and bins <= 0:
            raise ValueError("bins must be a positive integer")

        vals = df[column].dropna().astype(float).values
        if len(vals) == 0:
            raise DataValidationError(f"hist(): Column '{column}' has no non-null numeric values.")

        counts, edges = np.histogram(vals, bins=bins, density=density)
        bin_labels = [f"{edges[i]:.1f}\u2013{edges[i + 1]:.1f}" for i in range(len(counts))]
        self._merge_x_categories(bin_labels)

        entry: dict = {
            "type": "bar",
            "data": [round(float(c), 6) for c in counts],
            "name": column,
            "label": {"show": labels, "position": "top"},
            "itemStyle": {"borderRadius": border_radius},
        }
        if bar_color:
            entry["itemStyle"]["color"] = bar_color
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        _merge_style_params(
            entry, item_style=item_style, label_style=label_style,
            selected_mode=selected_mode, animation=animation,
            tooltip=tooltip,
        )
        entry.update(series_kw)

        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("bar", column))
        self._y_axes[0].setdefault("name", "Density" if density else "Count")
        self._x_axis["name"] = column

        # Histogram bin labels (e.g. "34.5–45.2") are inherently long;
        # pre-rotate to avoid OverlapWarning from the layout resolver.
        self._x_axis.setdefault("axisLabel", {}).setdefault("rotate", 30)
        self._user_set_rotate = True

        return self

    # ─── RADAR ────────────────────────────────────────────────────────────

    def radar(
        self, indicators: Sequence[dict],
        data: Sequence[Sequence[float]], *,
        series_names: Optional[Sequence[str]] = None,
        show_labels: bool = True, area_opacity: float = 0.15,
        radius: Union[int, str] = 150,
        center: Optional[List[str]] = None,
        emphasis: Optional[RadarEmphasis] = None,
        shape: Optional[Literal["polygon", "circle"]] = None,
        split_number: Optional[int] = None,
        axis_name_font_size: Optional[int] = None,
        axis_name_color: Optional[str] = None,
        split_area_show: Optional[bool] = None,
        axis_line_show: Optional[bool] = None,
        line_style: Optional[LineStyle] = None,
        color: Optional[str] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a radar chart."""
        self._ensure_mode("radar", "radar")

        if not indicators:
            raise DataValidationError("radar(): indicators list must not be empty.")
        if not data:
            raise DataValidationError("radar(): data list must not be empty.")
        for i, row in enumerate(data):
            if len(row) != len(indicators):
                raise DataValidationError(
                    f"radar(): data[{i}] has {len(row)} values but there are "
                    f"{len(indicators)} indicators."
                )

        series_data = []
        for i, vals in enumerate(data):
            name = series_names[i] if series_names and i < len(series_names) else f"Series {i + 1}"
            series_data.append({"value": list(vals), "name": name, "label": {"show": show_labels}})
            if name not in self._legend_items:
                self._legend_items.append(name)

        entry: dict = {
            "type": "radar", "data": series_data,
            "areaStyle": {"opacity": area_opacity},
        }
        if line_style is not None:
            entry["lineStyle"] = line_style.to_dict()
        if color is not None:
            entry.setdefault("itemStyle", {})["color"] = color
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        if tooltip is not None:
            entry["tooltip"] = tooltip.to_dict()
        entry.update(series_kw)
        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("radar", "radar"))

        radar_cfg: dict = {
            "indicator": list(indicators),
            "radius": radius,
            "center": center or ["50%", "55%"],
        }
        if shape is not None:
            radar_cfg["shape"] = shape
        if split_number is not None:
            radar_cfg["splitNumber"] = split_number
        if axis_name_font_size is not None or axis_name_color is not None:
            ax_name: dict = {}
            if axis_name_font_size is not None:
                ax_name["fontSize"] = axis_name_font_size
            if axis_name_color is not None:
                ax_name["color"] = axis_name_color
            radar_cfg["axisName"] = ax_name
        if split_area_show is not None:
            radar_cfg["splitArea"] = {"show": split_area_show}
        if axis_line_show is not None:
            radar_cfg["axisLine"] = {"show": axis_line_show}
        self._extra["_radar_cfg"] = radar_cfg
        return self

    # ─── KDE ──────────────────────────────────────────────────────────────

    def kde(
        self, df: pd.DataFrame, column: str, *,
        hue: Optional[str] = None, bandwidth: Optional[float] = None,
        grid_size: int = 200, show_median: bool = True,
        emphasis: Optional[LineEmphasis] = None,
        line_style: Optional[LineStyle] = None,
        area: bool = False,
        area_style: Optional[AreaStyle] = None,
        label_style: Optional[LabelStyle] = None,
        labels: bool = False,
        connect_nulls: Optional[bool] = None,
        color: Optional[str] = None,
        animation: Optional[AnimationConfig] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a KDE (kernel density estimate) curve — like ``sns.kdeplot()``.

        Requires ``scipy``. Install with ``pip install echartsy[scipy]``.
        """
        try:
            from scipy.stats import gaussian_kde as _gaussian_kde
        except ImportError:
            raise ImportError(
                "kde() requires scipy. Install it with: pip install echartsy[scipy]"
            ) from None

        self._ensure_cartesian("kde")
        df = _validate_df(df, "kde")
        _validate_columns(df, [column, hue], "kde")

        vals_all = df[column].dropna().astype(float).values
        if len(vals_all) < 2:
            raise DataValidationError(f"kde(): Column '{column}' needs at least 2 non-null values.")

        xmin, xmax = float(vals_all.min()), float(vals_all.max())
        xs = np.linspace(xmin, xmax, grid_size)

        self._x_axis["type"] = "value"
        self._x_axis.pop("data", None)
        self._x_axis.setdefault("name", column)

        groups = df.groupby(hue) if hue else [(column, df)]
        for lvl, sub in groups:
            arr = sub[column].dropna().astype(float).values
            if len(arr) < 2:
                continue
            kde_fn = _gaussian_kde(arr, bw_method=bandwidth)
            ys = kde_fn(xs)
            median = float(np.median(arr))
            name = f"{lvl} (Median: {median:.1f})" if show_median else str(lvl)

            entry: dict = {
                "type": "line", "name": name, "smooth": True,
                "data": list(zip(xs.tolist(), ys.tolist())),
                "showSymbol": False,
            }
            if area:
                entry["areaStyle"] = {"opacity": 0.15}
            if labels:
                entry["label"] = {"show": True}
            if connect_nulls is not None:
                entry["connectNulls"] = connect_nulls
            if emphasis is not None:
                entry["emphasis"] = emphasis.to_dict()
            _merge_style_params(
                entry, line_style=line_style, area_style=area_style,
                label_style=label_style, color=color,
                animation=animation, tooltip=tooltip,
            )
            entry.update(series_kw)
            self._series.append(entry)
            self._series_meta.append(_SeriesMeta("line", name))
            if name not in self._legend_items:
                self._legend_items.append(name)

        self._y_axes[0].setdefault("name", "Density")
        return self

    # ─── HEATMAP ──────────────────────────────────────────────────────────

    def heatmap(
        self, df: pd.DataFrame, x: str, y: str, value: str, *,
        label_show: bool = True, label_formatter: str = "{c}",
        visual_min: Optional[float] = None,
        visual_max: Optional[float] = None,
        in_range_colors: Optional[List[str]] = None,
        visual_map_type: Literal["continuous", "piecewise"] = "continuous",
        pieces: Optional[List[dict]] = None,
        progressive: Optional[int] = None,
        progressive_threshold: Optional[int] = None,
        emphasis: Optional[Emphasis] = None,
        label_style: Optional[LabelStyle] = None,
        item_style: Optional[ItemStyle] = None,
        animation: Optional[AnimationConfig] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a heatmap — similar to ``sns.heatmap()``."""
        self._ensure_mode("heatmap", "heatmap")
        df = _validate_df(df, "heatmap")
        _validate_columns(df, [x, y, value], "heatmap")

        dff = df.copy()
        dff[value] = _coerce_numeric(dff, value, "heatmap")
        n_before = len(dff)
        dff = dff.dropna(subset=[x, y, value])
        n_dropped = n_before - len(dff)
        if n_dropped > 0:
            warnings.warn(f"heatmap(): {n_dropped} rows dropped due to missing values", stacklevel=2)
        if dff.empty:
            warnings.warn("heatmap(): all rows dropped after removing missing values; chart will be empty", stacklevel=2)
            return self

        x_cats = list(dict.fromkeys(dff[x].astype(str).tolist()))
        y_cats = list(dict.fromkeys(dff[y].astype(str).tolist()))

        x_lookup = {cat: i for i, cat in enumerate(x_cats)}
        y_lookup = {cat: i for i, cat in enumerate(y_cats)}
        data = []
        for _, r in dff.iterrows():
            xk, yk = str(r[x]), str(r[y])
            if xk not in x_lookup or yk not in y_lookup:
                continue
            data.append([x_lookup[xk], y_lookup[yk], round(float(r[value]), 4)])

        self._x_axis.update({"type": "category", "data": x_cats, "splitArea": {"show": True}})
        self._y_axes[0].update({"type": "category", "data": y_cats, "splitArea": {"show": True}})

        vmin = visual_min if visual_min is not None else float(dff[value].min())
        vmax = visual_max if visual_max is not None else float(dff[value].max())
        colors = in_range_colors or [
            "#313695", "#4575b4", "#74add1", "#abd9e9",
            "#e0f3f8", "#ffffbf", "#fee090", "#fdae61",
            "#f46d43", "#d73027", "#a50026",
        ]

        if visual_map_type == "piecewise":
            vm: dict = {
                "type": "piecewise",
                "pieces": pieces or [],
                "orient": "horizontal",
                "left": "center", "bottom": "5%",
            }
            if in_range_colors:
                vm["inRange"] = {"color": in_range_colors}
            self._extra["visualMap"] = vm
        else:
            self._extra["visualMap"] = {
                "min": vmin, "max": vmax, "calculable": True,
                "orient": "horizontal", "left": "center", "bottom": "5%",
                "inRange": {"color": colors},
            }

        emphasis_dict = {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}}
        if emphasis is not None:
            emphasis_dict = emphasis.to_dict()

        entry: dict = {
            "type": "heatmap", "data": data,
            "label": {"show": label_show, "formatter": label_formatter},
            "emphasis": emphasis_dict,
        }
        if progressive is not None:
            entry["progressive"] = progressive
        if progressive_threshold is not None:
            entry["progressiveThreshold"] = progressive_threshold
        _merge_style_params(
            entry, label_style=label_style, item_style=item_style,
            selected_mode=selected_mode, animation=animation,
            tooltip=tooltip,
        )
        entry.update(series_kw)
        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("heatmap", value))
        self._tooltip_cfg["trigger"] = "item"
        return self

    # ─── SANKEY ───────────────────────────────────────────────────────────

    def sankey(
        self, df: pd.DataFrame, levels: Sequence[str], value: str, *,
        node_width: Union[int, str] = 20, node_gap: int = 8,
        layout: Literal["none", "orthogonal"] = "none",
        orient: Literal["horizontal", "vertical"] = "horizontal",
        emphasis: Optional[Union[SankeyEmphasis, dict]] = None,
        node_align: Optional[Literal["justify", "left", "right"]] = None,
        draggable: Optional[bool] = None,
        focus_node_adjacency: Optional[bool] = None,
        layout_iterations: Optional[int] = None,
        item_style: Optional[ItemStyle] = None,
        label_style: Optional[LabelStyle] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add an N-stage Sankey diagram."""
        self._ensure_mode("sankey", "sankey")
        df = _validate_df(df, "sankey")
        _validate_columns(df, list(levels) + [value], "sankey")

        if len(levels) < 2:
            raise BuilderConfigError("sankey() requires at least 2 levels.")

        links = []
        for src_col, tgt_col in zip(levels, levels[1:]):
            grouped = df.groupby([src_col, tgt_col], as_index=False)[value].sum()
            for _, row in grouped.iterrows():
                links.append({
                    "source": str(row[src_col]),
                    "target": str(row[tgt_col]),
                    "value": float(row[value]),
                })

        unique_nodes = pd.unique(df[list(levels)].values.ravel())
        nodes = [{"name": str(n)} for n in unique_nodes]

        entry: dict = {
            "type": "sankey", "layout": layout, "orient": orient,
            "data": nodes, "links": links,
            "nodeWidth": node_width, "nodeGap": node_gap,
        }
        if node_align is not None:
            entry["nodeAlign"] = node_align
        if draggable is not None:
            entry["draggable"] = draggable
        if focus_node_adjacency is not None:
            entry["focusNodeAdjacency"] = focus_node_adjacency
        if layout_iterations is not None:
            entry["layoutIterations"] = layout_iterations
        if item_style is not None:
            entry.setdefault("itemStyle", {}).update(item_style.to_dict())
        if label_style is not None:
            entry.setdefault("label", {}).update(label_style.to_dict())
        if tooltip is not None:
            entry["tooltip"] = tooltip.to_dict()
        if emphasis is not None:
            if isinstance(emphasis, dict):
                import warnings as _w
                _w.warn(
                    "Passing emphasis as a dict is deprecated. "
                    "Use SankeyEmphasis(...) instead.",
                    DeprecationWarning, stacklevel=2,
                )
                entry["emphasis"] = emphasis
            else:
                entry["emphasis"] = emphasis.to_dict()
        entry.update(series_kw)

        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("sankey", "sankey"))
        self._tooltip_cfg = {"trigger": "item", "triggerOn": "mousemove", "confine": True}
        return self

    # ─── TREEMAP ──────────────────────────────────────────────────────────

    def treemap(
        self, df: pd.DataFrame, path: List[str],
        value: Optional[str] = None, *,
        leaf_depth: Optional[int] = 2, roam: bool = True,
        gap_width: int = 2, border_width: int = 1,
        # --- Labels ---
        label_formatter: str = "{b}\n{c}",
        label_style: Optional[LabelStyle] = None,
        upper_label: bool = False,
        upper_label_height: int = 30,
        upper_label_style: Optional[LabelStyle] = None,
        # --- Layout ---
        sort: Optional[Literal["desc", "asc"]] = None,
        square_ratio: Optional[float] = None,
        # --- Visibility ---
        children_visible_min: Optional[int] = None,
        visible_min: Optional[int] = None,
        visual_min: Optional[float] = None,
        visual_max: Optional[float] = None,
        # --- Interaction ---
        zoom_to_node_ratio: Optional[float] = None,
        drill_down_icon: Optional[str] = None,
        node_click: Optional[Literal["zoomToNode", "link", False]] = None,
        breadcrumb: Optional[bool] = None,
        # --- Color ---
        color_alpha: Optional[list] = None,
        color_saturation: Optional[list] = None,
        color_mapping_by: Optional[Literal["index", "value", "id"]] = None,
        # --- Style objects ---
        emphasis: Optional[TreemapEmphasis] = None,
        item_style: Optional[ItemStyle] = None,
        tooltip: Optional[TooltipStyle] = None,
        levels: Optional[list] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a treemap chart."""
        self._ensure_mode("treemap", "treemap")
        df = _validate_df(df, "treemap")
        _validate_columns(df, path + ([value] if value else []), "treemap")

        val_col = value
        work = df.copy()
        if val_col is None:
            work["__count__"] = 1
            val_col = "__count__"

        for lvl in path:
            work[lvl] = work[lvl].astype(object).where(pd.notna(work[lvl]), "<NA>")

        g = work.groupby(path, dropna=False)[val_col].sum().reset_index()

        root: dict = {}
        for _, row in g.iterrows():
            node = root
            for lvl in path:
                key = str(row[lvl])
                node = node.setdefault(key, {"__val": 0, "__ch": {}})
                node["__val"] += float(row[val_col])
                node = node["__ch"]

        def _build(subtree: dict) -> list:
            out = []
            for name, meta in subtree.items():
                item: dict = {"name": name, "value": round(meta["__val"], 2)}
                children = _build(meta["__ch"])
                if children:
                    item["children"] = children
                out.append(item)
            return out

        data = _build(root)

        if levels is not None:
            resolved_levels = levels
        elif upper_label:
            resolved_levels = [
                {
                    "itemStyle": {"borderColor": "#777", "borderWidth": 0, "gapWidth": 1},
                    "upperLabel": {"show": False},
                },
                {
                    "itemStyle": {"borderColor": "#555", "borderWidth": 5, "gapWidth": 1},
                    "emphasis": {"itemStyle": {"borderColor": "#ddd"}},
                },
                {
                    "colorSaturation": [0.35, 0.5],
                    "itemStyle": {"borderWidth": 5, "gapWidth": 1, "borderColorSaturation": 0.6},
                },
            ]
        else:
            resolved_levels = [
                {"itemStyle": {"borderColor": "#fff", "borderWidth": border_width, "gapWidth": gap_width}},
                {"itemStyle": {"borderColor": "#eee", "borderWidth": border_width, "gapWidth": gap_width}},
                {"itemStyle": {"borderColor": "#ddd", "borderWidth": border_width, "gapWidth": gap_width}},
            ]

        label_dict: dict = {"show": True, "formatter": label_formatter}
        if label_style is not None:
            label_dict.update(label_style.to_dict())

        entry: dict = {
            "type": "treemap", "data": data,
            "leafDepth": leaf_depth, "roam": roam,
            "label": label_dict,
            "breadcrumb": {"show": True}, "levels": resolved_levels,
        }
        if upper_label:
            upper_dict: dict = {"show": True, "height": upper_label_height}
            if upper_label_style is not None:
                upper_dict.update(upper_label_style.to_dict())
            entry["upperLabel"] = upper_dict
        if drill_down_icon is not None:
            entry["drillDownIcon"] = drill_down_icon
        if node_click is not None:
            entry["nodeClick"] = node_click
        if color_alpha is not None:
            entry["colorAlpha"] = color_alpha
        if color_saturation is not None:
            entry["colorSaturation"] = color_saturation
        if visible_min is not None:
            entry["visibleMin"] = visible_min
        if sort is not None:
            entry["sort"] = sort
        if square_ratio is not None:
            entry["squareRatio"] = square_ratio
        if children_visible_min is not None:
            entry["childrenVisibleMin"] = children_visible_min
        if visual_min is not None:
            entry["visualMin"] = visual_min
        if visual_max is not None:
            entry["visualMax"] = visual_max
        if zoom_to_node_ratio is not None:
            entry["zoomToNodeRatio"] = zoom_to_node_ratio
        if color_mapping_by is not None:
            entry["colorMappingBy"] = color_mapping_by
        if breadcrumb is not None:
            entry["breadcrumb"] = {"show": breadcrumb}
        if item_style is not None:
            entry.setdefault("itemStyle", {}).update(item_style.to_dict())
        if tooltip is not None:
            entry["tooltip"] = tooltip.to_dict()
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        entry.update(series_kw)
        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("treemap", "treemap"))
        self._tooltip_cfg = {"trigger": "item", "confine": True}
        return self

    # ─── FUNNEL ───────────────────────────────────────────────────────────

    def funnel(
        self, df: pd.DataFrame, names: str, values: str, *,
        sort_order: Literal["descending", "ascending", "none"] = "descending",
        gap: int = 2, label_position: str = "inside",
        label_formatter: str = "{b}: {c}", emphasis: Optional[FunnelEmphasis] = None,
        orient: Optional[Literal["vertical", "horizontal"]] = None,
        funnel_align: Optional[Literal["center", "left", "right"]] = None,
        min_size: Optional[str] = None,
        max_size: Optional[str] = None,
        item_style: Optional[ItemStyle] = None,
        label_line: Optional[LabelLineStyle] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a funnel chart."""
        self._ensure_mode("funnel", "funnel")
        df = _validate_df(df, "funnel")
        _validate_columns(df, [names, values], "funnel")

        dff = df.copy()
        dff[values] = _coerce_numeric(dff, values, "funnel")
        dff = dff.dropna(subset=[names, values])

        data = [
            {"name": str(n), "value": round(float(v), 4)}
            for n, v in zip(dff[names], dff[values])
        ]

        entry: dict = {
            "type": "funnel", "data": data, "sort": sort_order,
            "gap": gap,
            "label": {"show": True, "position": label_position, "formatter": label_formatter},
        }
        if orient is not None:
            entry["orient"] = orient
        if funnel_align is not None:
            entry["funnelAlign"] = funnel_align
        if min_size is not None:
            entry["minSize"] = min_size
        if max_size is not None:
            entry["maxSize"] = max_size
        if item_style is not None:
            entry.setdefault("itemStyle", {}).update(item_style.to_dict())
        if label_line is not None:
            entry["labelLine"] = label_line.to_dict()
        if tooltip is not None:
            entry["tooltip"] = tooltip.to_dict()
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        entry.update(series_kw)
        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("funnel", names))
        for n in dff[names]:
            if str(n) not in self._legend_items:
                self._legend_items.append(str(n))
        self._tooltip_cfg = {"trigger": "item", "confine": True}
        return self

    # ─── BOXPLOT ──────────────────────────────────────────────────────────

    def boxplot(
        self, df: pd.DataFrame, x: str, y: str, *,
        orient: Literal["v", "h"] = "v", grid: int = 0,
        emphasis: Optional[Emphasis] = None,
        item_style: Optional[ItemStyle] = None,
        label_style: Optional[LabelStyle] = None,
        labels: bool = False,
        color: Optional[str] = None,
        blur: Optional[Blur] = None,
        select: Optional[Select] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        animation: Optional[AnimationConfig] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a box-and-whisker plot — equivalent to ``plt.boxplot()``."""
        if orient != "v":
            warnings.warn(
                "boxplot(): horizontal orientation (orient='h') is not yet supported; "
                "rendering as vertical.",
                stacklevel=2,
            )
        self._ensure_cartesian("boxplot")
        df = _validate_df(df, "boxplot")
        _validate_columns(df, [x, y], "boxplot")

        dff = df.copy()
        dff[y] = _coerce_numeric(dff, y, "boxplot")
        dff = dff.dropna(subset=[x, y])

        categories = list(dict.fromkeys(dff[x].astype(str).tolist()))
        self._merge_x_categories(categories)

        box_data = []
        for cat in categories:
            vals = dff.loc[dff[x].astype(str) == cat, y].dropna().values
            if len(vals) == 0:
                box_data.append([0, 0, 0, 0, 0])
            else:
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                box_data.append([
                    round(float(vals.min()), 4), round(float(q1), 4),
                    round(float(med), 4), round(float(q3), 4),
                    round(float(vals.max()), 4),
                ])

        entry: dict = {"type": "boxplot", "data": box_data, "name": y}
        if labels:
            entry["label"] = {"show": True}
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        _merge_style_params(
            entry, item_style=item_style, label_style=label_style,
            color=color, blur=blur, select=select,
            selected_mode=selected_mode, animation=animation,
            tooltip=tooltip,
        )
        entry.update(series_kw)
        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("boxplot", y))
        self._tooltip_cfg["trigger"] = "item"
        return self

    # ─── CANDLESTICK ─────────────────────────────────────────────────────

    def candlestick(
        self, df: pd.DataFrame, date: str, open: str, close: str,
        low: str, high: str, *,
        up_color: str = "#EE6666",
        down_color: str = "#73C0DE",
        up_border: Optional[str] = None,
        down_border: Optional[str] = None,
        axis: int = 0, grid: int = 0,
        emphasis: Optional[Emphasis] = None,
        border_width: Optional[int] = None,
        opacity: Optional[float] = None,
        label_style: Optional[LabelStyle] = None,
        labels: bool = False,
        animation: Optional[AnimationConfig] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a candlestick (OHLC) series — used for financial / stock charts.

        Each row of *df* provides one candle. The four numeric columns map to
        the ECharts data order ``[open, close, low, high]``.

        Parameters
        ----------
        df : DataFrame
            Source data.
        date : str
            Column used as category axis (typically a date string).
        open, close, low, high : str
            Column names for the four price components.
        up_color / down_color : str
            Body fill colour for bullish / bearish candles.
        up_border / down_border : str | None
            Border colour (defaults to body colour).
        axis : int
            ``0`` = left Y-axis, ``1`` = right Y-axis.
        emphasis : Emphasis | None
            Hover highlight — base ``Emphasis`` with ``ItemStyle`` works.
        **series_kw
            Forwarded verbatim into the ECharts series dict.
        """
        self._ensure_cartesian("candlestick")
        df = _validate_df(df, "candlestick")
        _validate_columns(df, [date, open, close, low, high], "candlestick")

        dff = df.copy()
        dff[date] = dff[date].astype(str).str.strip()
        for col in (open, close, low, high):
            dff[col] = _coerce_numeric(dff, col, "candlestick")
        dff = dff.dropna(subset=[date, open, close, low, high])

        cats = _sort_categories(dff[date])
        self._merge_x_categories(cats, grid=grid)

        if axis == 1 and len(self._y_axes) < 2:
            self.ylabel_right("")

        # Build data aligned to merged categories (ECharts order: open, close, low, high)
        n_unique_dates = dff[date].nunique()
        if n_unique_dates < len(dff):
            warnings.warn(
                f"candlestick(): {len(dff) - n_unique_dates} duplicate date(s) found; "
                "only the last row per date will be used.",
                stacklevel=2,
            )
        grouped = {str(r[date]): r for _, r in dff.iterrows()}
        grid_cats = self._grid_categories.get(grid, self._x_categories)
        candlestick_data: list = []
        for cat in grid_cats:
            row = grouped.get(cat)
            if row is not None:
                candlestick_data.append([
                    round(float(row[open]), 4), round(float(row[close]), 4),
                    round(float(row[low]), 4), round(float(row[high]), 4),
                ])
            else:
                candlestick_data.append(None)

        y_idx = axis + grid * 2 if self._n_grids > 1 else axis
        x_idx = grid if self._n_grids > 1 else 0
        entry: dict = {
            "type": "candlestick",
            "name": series_kw.pop("name", ""),
            "data": candlestick_data,
            "yAxisIndex": y_idx,
            "itemStyle": {
                "color": up_color,
                "color0": down_color,
                "borderColor": up_border or up_color,
                "borderColor0": down_border or down_color,
            },
        }
        if self._n_grids > 1:
            entry["xAxisIndex"] = x_idx
        if border_width is not None:
            entry["itemStyle"]["borderWidth"] = border_width
        if opacity is not None:
            entry["itemStyle"]["opacity"] = opacity
        if labels:
            entry["label"] = {"show": True}
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        _merge_style_params(
            entry, label_style=label_style,
            selected_mode=selected_mode, animation=animation,
            tooltip=tooltip,
        )
        entry.update(series_kw)
        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("candlestick", entry["name"], axis, grid_index=grid))
        self._tooltip_cfg = {"trigger": "axis", "confine": True}
        return self

    # ─── GAUGE ───────────────────────────────────────────────────────────

    def gauge(
        self, value: float, name: str = "", *,
        min_val: float = 0, max_val: float = 100,
        start_angle: int = 225, end_angle: int = -45,
        split_number: int = 10, pointer: bool = True,
        axis_line_colors: Optional[List[Tuple[float, str]]] = None,
        radius: str = "75%",
        emphasis: Optional[Emphasis] = None,
        detail_formatter: Optional[str] = None,
        detail_font_size: Optional[int] = None,
        detail_color: Optional[str] = None,
        axis_tick_show: Optional[bool] = None,
        axis_label_show: Optional[bool] = None,
        progress: Optional[bool] = None,
        progress_color: Optional[str] = None,
        item_style: Optional[ItemStyle] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a gauge / meter chart (speedometer style).

        Parameters
        ----------
        value : float
            The gauge value.
        name : str
            Label shown below the pointer.
        min_val, max_val : float
            Data range.
        start_angle, end_angle : int
            Arc angles in degrees.
        split_number : int
            Number of tick divisions.
        pointer : bool
            Show the pointer needle.
        axis_line_colors : list of (float, str) tuples
            Colour stops for the arc, e.g.
            ``[(0.3, "#67e0e3"), (0.7, "#37a2da"), (1, "#fd666d")]``.
        """
        self._ensure_mode("gauge", "gauge")

        entry: dict = {
            "type": "gauge",
            "min": min_val,
            "max": max_val,
            "startAngle": start_angle,
            "endAngle": end_angle,
            "splitNumber": split_number,
            "radius": radius,
            "pointer": {"show": pointer},
            "data": [{"value": value, "name": name}],
            "detail": {"formatter": "{value}"},
        }
        if axis_line_colors:
            entry["axisLine"] = {
                "lineStyle": {
                    "width": 30,
                    "color": [[stop, clr] for stop, clr in axis_line_colors],
                }
            }
        if detail_formatter is not None:
            entry["detail"]["formatter"] = detail_formatter
        if detail_font_size is not None:
            entry["detail"]["fontSize"] = detail_font_size
        if detail_color is not None:
            entry["detail"]["color"] = detail_color
        if axis_tick_show is not None:
            entry.setdefault("axisTick", {})["show"] = axis_tick_show
        if axis_label_show is not None:
            entry.setdefault("axisLabel", {})["show"] = axis_label_show
        if progress is not None:
            entry.setdefault("progress", {})["show"] = progress
        if progress_color is not None:
            entry.setdefault("progress", {}).setdefault("itemStyle", {})["color"] = progress_color
        if item_style is not None:
            entry.setdefault("itemStyle", {}).update(item_style.to_dict())
        if tooltip is not None:
            entry["tooltip"] = tooltip.to_dict()
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        entry.update(series_kw)

        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("gauge", name))
        self._tooltip_cfg = {"trigger": "item", "confine": True}
        return self

    # ─── SUNBURST ────────────────────────────────────────────────────────

    def sunburst(
        self, df: pd.DataFrame, path: List[str],
        value: Optional[str] = None, *,
        inner_radius: str = "15%",
        sort: Optional[Literal["desc", "asc"]] = "desc",
        emphasis: Optional[Emphasis] = None,
        outer_radius: Optional[str] = None,
        node_click: Optional[Literal["rootToNode", "link", False]] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        label_rotate: Optional[Union[str, int]] = None,
        animation_type: Optional[Literal["expansion", "scale"]] = None,
        item_style: Optional[ItemStyle] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a sunburst (hierarchical pie) chart.

        Uses the same *path* + *value* API as :meth:`treemap`.

        Parameters
        ----------
        df : pd.DataFrame
            Source data.
        path : list[str]
            Hierarchy columns from root → leaf.
        value : str or None
            Numeric size column (counts if ``None``).
        inner_radius : str
            Inner ring radius.
        sort : ``"desc"`` | ``"asc"`` | ``None``
            Slice sort order.
        """
        self._ensure_mode("sunburst", "sunburst")
        df = _validate_df(df, "sunburst")
        _validate_columns(df, path + ([value] if value else []), "sunburst")

        val_col = value
        work = df.copy()
        if val_col is None:
            work["__count__"] = 1
            val_col = "__count__"

        for lvl in path:
            work[lvl] = work[lvl].astype(object).where(pd.notna(work[lvl]), "<NA>")

        g = work.groupby(path, dropna=False)[val_col].sum().reset_index()

        root: dict = {}
        for _, row in g.iterrows():
            node = root
            for lvl in path:
                key = str(row[lvl])
                node = node.setdefault(key, {"__val": 0, "__ch": {}})
                node["__val"] += float(row[val_col])
                node = node["__ch"]

        def _build(subtree: dict) -> list:
            out = []
            for nm, meta in subtree.items():
                item: dict = {"name": nm, "value": round(meta["__val"], 2)}
                children = _build(meta["__ch"])
                if children:
                    item["children"] = children
                out.append(item)
            return out

        data = _build(root)

        entry: dict = {
            "type": "sunburst",
            "data": data,
            "radius": [inner_radius, outer_radius or "90%"],
            "label": {"rotate": "radial"},
        }
        if sort is not None:
            entry["sort"] = sort
        if node_click is not None:
            entry["nodeClick"] = node_click
        if selected_mode is not None:
            entry["selectedMode"] = selected_mode
        if label_rotate is not None:
            entry["label"]["rotate"] = label_rotate
        if animation_type is not None:
            entry["animationType"] = animation_type
        if item_style is not None:
            entry.setdefault("itemStyle", {}).update(item_style.to_dict())
        if tooltip is not None:
            entry["tooltip"] = tooltip.to_dict()
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        entry.update(series_kw)

        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("sunburst", "sunburst"))
        self._tooltip_cfg = {"trigger": "item", "confine": True}
        return self

    # ─── GRAPH / NETWORK ─────────────────────────────────────────────────

    def graph(
        self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, *,
        source: str = "source", target: str = "target",
        value: Optional[str] = None,
        node_name: str = "name", node_value: Optional[str] = None,
        node_category: Optional[str] = None,
        layout: Literal["force", "circular", "none"] = "force",
        roam: bool = True, symbol_size: int = 20,
        emphasis: Optional[Union[GraphEmphasis, Emphasis]] = None,
        repulsion: Optional[int] = None,
        gravity: Optional[float] = None,
        edge_length: Optional[Union[int, list]] = None,
        friction: Optional[float] = None,
        edge_label: Optional[dict] = None,
        edge_symbol: Optional[list] = None,
        node_scale_ratio: Optional[float] = None,
        label_style: Optional[LabelStyle] = None,
        item_style: Optional[ItemStyle] = None,
        line_style: Optional[LineStyle] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a graph / network chart.

        Parameters
        ----------
        nodes_df : pd.DataFrame
            DataFrame of nodes with at least a *node_name* column.
        edges_df : pd.DataFrame
            DataFrame of edges with *source* and *target* columns.
        source, target : str
            Column names in *edges_df*.
        value : str or None
            Edge weight column in *edges_df*.
        node_name : str
            Column in *nodes_df* for node labels.
        node_value : str or None
            Column in *nodes_df* for node size.
        node_category : str or None
            Column in *nodes_df* for grouping (legend categories).
        layout : ``"force"`` | ``"circular"`` | ``"none"``
            Graph layout algorithm.
        roam : bool
            Allow pan/zoom.
        """
        self._ensure_mode("graph", "graph")
        _validate_df(nodes_df, "graph (nodes)")
        _validate_df(edges_df, "graph (edges)")
        _validate_columns(edges_df, [source, target], "graph (edges)")
        _validate_columns(nodes_df, [node_name], "graph (nodes)")

        nodes: list = []
        for _, r in nodes_df.iterrows():
            node: dict = {"name": str(r[node_name]), "symbolSize": symbol_size}
            if node_value and node_value in nodes_df.columns:
                node["value"] = float(r[node_value])
            if node_category and node_category in nodes_df.columns:
                node["category"] = str(r[node_category])
            nodes.append(node)

        edges: list = []
        for _, r in edges_df.iterrows():
            edge: dict = {"source": str(r[source]), "target": str(r[target])}
            if value and value in edges_df.columns:
                edge["value"] = float(r[value])
            edges.append(edge)

        categories: list = []
        if node_category and node_category in nodes_df.columns:
            cats = nodes_df[node_category].dropna().unique()
            categories = [{"name": str(c)} for c in cats]

        entry: dict = {
            "type": "graph",
            "layout": layout,
            "data": nodes,
            "links": edges,
            "roam": roam,
            "label": {"show": True, "position": "right"},
        }
        if layout == "force":
            force_cfg: dict = {"repulsion": 100, "edgeLength": [50, 200]}
            if repulsion is not None:
                force_cfg["repulsion"] = repulsion
            if gravity is not None:
                force_cfg["gravity"] = gravity
            if edge_length is not None:
                force_cfg["edgeLength"] = edge_length
            if friction is not None:
                force_cfg["friction"] = friction
            entry["force"] = force_cfg
        if categories:
            entry["categories"] = categories
        if edge_label is not None:
            entry["edgeLabel"] = edge_label
        if edge_symbol is not None:
            entry["edgeSymbol"] = edge_symbol
        if node_scale_ratio is not None:
            entry["nodeScaleRatio"] = node_scale_ratio
        if label_style is not None:
            entry.setdefault("label", {}).update(label_style.to_dict())
        if item_style is not None:
            entry.setdefault("itemStyle", {}).update(item_style.to_dict())
        if line_style is not None:
            entry.setdefault("lineStyle", {}).update(line_style.to_dict())
        if tooltip is not None:
            entry["tooltip"] = tooltip.to_dict()
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        entry.update(series_kw)

        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("graph", "graph"))
        self._tooltip_cfg = {"trigger": "item", "confine": True}
        if categories:
            self._legend_items = [c["name"] for c in categories]
        return self

    # ─── CALENDAR HEATMAP ────────────────────────────────────────────────

    def calendar_heatmap(
        self, df: pd.DataFrame, date: str, value: str, *,
        year: Optional[int] = None,
        cell_size: Optional[List] = None,
        orient: Literal["horizontal", "vertical"] = "horizontal",
        visual_min: Optional[float] = None,
        visual_max: Optional[float] = None,
        in_range_colors: Optional[List[str]] = None,
        split_line_show: Optional[bool] = None,
        split_line_color: Optional[str] = None,
        day_label_show: Optional[bool] = None,
        month_label_show: Optional[bool] = None,
        item_style: Optional[ItemStyle] = None,
        emphasis: Optional[Emphasis] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a calendar heatmap (GitHub-style contribution grid).

        Parameters
        ----------
        df : pd.DataFrame
            Source data.
        date : str
            Date column (parseable by ``pd.to_datetime``).
        value : str
            Numeric value column.
        year : int or None
            Calendar year to display (auto-detected from data if ``None``).
        cell_size : list or None
            ``["auto", 14]`` etc.
        orient : ``"horizontal"`` | ``"vertical"``
        visual_min, visual_max : float or None
            Colour scale range.
        in_range_colors : list[str] or None
            Colour stops for the gradient.
        """
        self._ensure_mode("calendar", "calendar_heatmap")
        df = _validate_df(df, "calendar_heatmap")
        _validate_columns(df, [date, value], "calendar_heatmap")

        dff = df.copy()
        dff[value] = _coerce_numeric(dff, value, "calendar_heatmap")
        dff[date] = pd.to_datetime(dff[date]).dt.strftime("%Y-%m-%d")
        dff = dff.dropna(subset=[date, value])

        if year is None:
            year = int(pd.to_datetime(dff[date]).dt.year.mode().iloc[0])

        data = [[str(d), round(float(v), 4)] for d, v in zip(dff[date], dff[value])]

        vmin = visual_min if visual_min is not None else float(dff[value].min())
        vmax = visual_max if visual_max is not None else float(dff[value].max())
        colors = in_range_colors or ["#ebedf0", "#9be9a8", "#40c463", "#30a14e", "#216e39"]

        cal_cfg: dict = {
            "range": str(year),
            "cellSize": cell_size or ["auto", 14],
            "orient": orient,
            "left": 30, "right": 30, "top": 60, "bottom": 30,
            "itemStyle": {"borderWidth": 0.5, "borderColor": "#fff"},
        }
        if split_line_show is not None or split_line_color is not None:
            sl: dict = {}
            if split_line_show is not None:
                sl["show"] = split_line_show
            if split_line_color is not None:
                sl["lineStyle"] = {"color": split_line_color}
            cal_cfg["splitLine"] = sl
        if day_label_show is not None:
            cal_cfg["dayLabel"] = {"show": day_label_show}
        if month_label_show is not None:
            cal_cfg["monthLabel"] = {"show": month_label_show}
        self._extra["calendar"] = cal_cfg
        self._extra["visualMap"] = {
            "min": vmin, "max": vmax, "calculable": True,
            "orient": "horizontal", "left": "center", "bottom": "2%",
            "inRange": {"color": colors},
        }

        entry: dict = {
            "type": "heatmap",
            "coordinateSystem": "calendar",
            "data": data,
        }
        if item_style is not None:
            entry.setdefault("itemStyle", {}).update(item_style.to_dict())
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        if tooltip is not None:
            entry["tooltip"] = tooltip.to_dict()
        entry.update(series_kw)

        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("heatmap", value))
        self._tooltip_cfg = {"trigger": "item", "confine": True}
        return self

    # ─── CALENDAR PIE ─────────────────────────────────────────────────────

    def calendar_pie(
        self, df: pd.DataFrame,
        date: str, names: str, values: str, *,
        year: Optional[int] = None,
        pie_radius: int = 15,
        cell_size: Optional[List] = None,
        orient: Literal["horizontal", "vertical"] = "horizontal",
        **series_kw: Any,
    ) -> "Figure":
        """Add mini pie charts on a calendar grid.

        Each unique date gets its own pie chart positioned on the calendar.

        Parameters
        ----------
        df : DataFrame
            Source data.
        date : str
            Date column (parseable by ``pd.to_datetime``).
        names : str
            Column for pie slice names.
        values : str
            Column for pie slice values.
        year : int, optional
            Calendar year to display. Auto-detected if None.
        pie_radius : int
            Radius of each mini pie in pixels.
        cell_size : list, optional
            Calendar cell size (e.g. ``[50, 50]``).
        orient : {"horizontal", "vertical"}
            Calendar orientation.

        Returns
        -------
        Figure
            Self, for method chaining.
        """
        self._ensure_mode("calendar", "calendar_pie")
        df = _validate_df(df, "calendar_pie")
        _validate_columns(df, [date, names, values], "calendar_pie")

        dff = df.copy()
        dff[values] = _coerce_numeric(dff, values, "calendar_pie")
        dff[date] = pd.to_datetime(dff[date]).dt.strftime("%Y-%m-%d")
        dff = dff.dropna(subset=[date, values])

        if year is None:
            year = int(pd.to_datetime(dff[date]).dt.year.mode().iloc[0])

        resolved_cell_size = cell_size or [50, 50]

        # Calendar component
        cal_cfg: dict = {
            "range": str(year),
            "cellSize": resolved_cell_size,
            "orient": orient,
            "left": 30, "right": 30, "top": 60, "bottom": 30,
            "itemStyle": {"borderWidth": 0.5, "borderColor": "#fff"},
        }
        self._extra["calendar"] = cal_cfg

        # One pie series per unique date
        for date_val, group in dff.groupby(date):
            pie_data = [
                {"name": str(r[names]), "value": round(float(r[values]), 4)}
                for _, r in group.iterrows()
            ]
            entry: dict = {
                "type": "pie",
                "coordinateSystem": "calendar",
                "calendarIndex": 0,
                "center": date_val,
                "radius": pie_radius,
                "data": pie_data,
                "label": {"show": False},
            }
            entry.update(series_kw)
            self._series.append(entry)
            self._series_meta.append(_SeriesMeta("pie", str(date_val)))

        self._tooltip_cfg = {"trigger": "item", "confine": True}
        return self

    # ─── PARALLEL COORDINATES ─────────────────────────────────────────────

    def parallel(
        self, df: pd.DataFrame,
        dimensions: Sequence[str], *,
        color_dim: Optional[str] = None,
        color_range: Optional[List[str]] = None,
        smooth: bool = False,
        line_opacity: float = 0.3,
        active_opacity: float = 0.8,
        inactive_opacity: float = 0.05,
        realtime: bool = True,
        axis_expand: bool = False,
        axis_expand_count: int = 5,
        emphasis: Optional[Emphasis] = None,
        item_style: Optional[ItemStyle] = None,
        line_style: Optional[LineStyle] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "Figure":
        """Add a parallel coordinates chart.

        Parameters
        ----------
        df : DataFrame
            Source data.
        dimensions : Sequence[str]
            Column names to use as parallel axes (ordered left → right).
        color_dim : str, optional
            Column to map to colour via a continuous visualMap.
        color_range : list[str], optional
            Colour gradient endpoints for the visualMap (e.g. ``["#3b8686", "#e74c3c"]``).
        smooth : bool
            Whether to draw smooth (curved) lines.
        line_opacity : float
            Opacity of each polyline (0–1).
        active_opacity : float
            Opacity when an axis brush is active.
        inactive_opacity : float
            Opacity for lines outside the brush selection.
        realtime : bool
            Whether brushing updates in real-time.
        axis_expand : bool
            Enable axis expand mode (collapse axes to save space).
        axis_expand_count : int
            Number of axes visible when axis expand is active.
        emphasis : Emphasis, optional
            Emphasis style.
        item_style : ItemStyle, optional
            Item style override.
        line_style : LineStyle, optional
            Line style override.
        tooltip : TooltipStyle, optional
            Per-series tooltip.

        Returns
        -------
        Figure
            Self, for method chaining.
        """
        self._ensure_mode("parallel", "parallel")
        df = _validate_df(df, "parallel")
        _validate_columns(df, list(dimensions), "parallel")

        # ── Build parallelAxis array ──
        parallel_axes: List[dict] = []
        for i, dim in enumerate(dimensions):
            col = df[dim]
            if pd.api.types.is_numeric_dtype(col):
                parallel_axes.append({"dim": i, "name": dim, "type": "value"})
            else:
                unique_vals = sorted(col.dropna().unique(), key=str)
                parallel_axes.append({
                    "dim": i, "name": dim,
                    "type": "category", "data": [str(v) for v in unique_vals],
                })

        # ── Build parallel component ──
        parallel_cfg: dict = {
            "left": "5%", "right": "13%", "bottom": "10%", "top": "15%",
            "parallelAxisDefault": {
                "type": "value", "nameLocation": "end", "nameGap": 20,
                "realtime": realtime,
            },
        }
        if axis_expand:
            parallel_cfg["axisExpandable"] = True
            parallel_cfg["axisExpandCount"] = axis_expand_count

        # ── Build series data ──
        data_list: List[list] = []
        for _, row in df.iterrows():
            item: list = []
            for dim in dimensions:
                val = row[dim]
                item.append(val if pd.api.types.is_numeric_dtype(df[dim]) else str(val))
            data_list.append(item)

        # ── Series entry ──
        entry: dict = {
            "type": "parallel",
            "coordinateSystem": "parallel",
            "lineStyle": {"opacity": line_opacity},
            "smooth": smooth,
            "inactiveOpacity": inactive_opacity,
            "activeOpacity": active_opacity,
            "data": data_list,
        }
        if line_style is not None:
            entry.setdefault("lineStyle", {}).update(line_style.to_dict())
        if item_style is not None:
            entry.setdefault("itemStyle", {}).update(item_style.to_dict())
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        if tooltip is not None:
            entry["tooltip"] = tooltip.to_dict()
        entry.update(series_kw)

        # ── visualMap for colour dimension ──
        if color_dim is not None:
            _validate_columns(df, [color_dim], "parallel (color_dim)")
            dim_idx = list(dimensions).index(color_dim) if color_dim in dimensions else None
            col_vals = _coerce_numeric(df, color_dim, "parallel")
            colors = color_range or ["#3b8686", "#e74c3c"]
            vm: dict = {
                "min": float(col_vals.min()),
                "max": float(col_vals.max()),
                "calculable": True,
                "orient": "vertical",
                "right": 0, "top": "15%", "bottom": "10%",
                "inRange": {"color": colors},
            }
            if dim_idx is not None:
                vm["dimension"] = dim_idx
            self._extra["visualMap"] = vm

        self._extra["_parallel_axes"] = parallel_axes
        self._extra["_parallel_cfg"] = parallel_cfg

        self._series.append(entry)
        self._series_meta.append(_SeriesMeta("parallel", "parallel"))
        self._tooltip_cfg = {"trigger": "item", "confine": True}
        return self

    # ═══════════════════════════════════════════════════════════════════════
    # Annotations — markLine / markPoint / markArea
    # ═══════════════════════════════════════════════════════════════════════

    def mark_line(
        self, y: Optional[float] = None, x: Optional[str] = None,
        label: Optional[str] = None, color: Optional[str] = None,
        line_dash: Optional[Literal["solid", "dashed", "dotted"]] = None,
        series_index: int = -1,
    ) -> "Figure":
        """Add a reference line to a series.

        Parameters
        ----------
        y : float or None
            Horizontal line at this y-value.
        x : str or None
            Vertical line at this x-value.
        label : str or None
            Annotation text.
        color : str or None
            Line colour.
        line_dash : ``"solid"`` | ``"dashed"`` | ``"dotted"``
            Line dash style.
        series_index : int
            Which series to attach to (default: last added).
        """
        if not self._series:
            raise BuilderConfigError("mark_line(): no series to annotate.")
        idx = series_index if series_index >= 0 else len(self._series) + series_index
        if idx < 0 or idx >= len(self._series):
            raise BuilderConfigError(
                f"mark_line(): series_index {series_index} out of range "
                f"(have {len(self._series)} series)."
            )

        mark_data: dict = {}
        if y is not None:
            mark_data["yAxis"] = y
        elif x is not None:
            mark_data["xAxis"] = x

        if label is not None:
            mark_data["name"] = label
            mark_data["label"] = {"formatter": label, "show": True}

        line_style: dict = {}
        if color is not None:
            line_style["color"] = color
        if line_dash is not None:
            line_style["type"] = line_dash
        if line_style:
            mark_data["lineStyle"] = line_style

        ml = self._series[idx].setdefault("markLine", {"data": [], "silent": True})
        ml["data"].append(mark_data)
        return self

    def mark_point(
        self, type: Optional[Literal["max", "min"]] = None,
        label: Optional[str] = None,
        coord: Optional[List] = None,
        symbol_size: int = 50,
        series_index: int = -1,
    ) -> "Figure":
        """Add a marker point to a series.

        Parameters
        ----------
        type : ``"max"`` | ``"min"`` or None
            Auto-detect the max/min point.
        label : str or None
            Annotation text.
        coord : list or None
            Explicit ``[x, y]`` coordinate.
        series_index : int
            Which series to attach to (default: last added).
        """
        if not self._series:
            raise BuilderConfigError("mark_point(): no series to annotate.")
        idx = series_index if series_index >= 0 else len(self._series) + series_index
        if idx < 0 or idx >= len(self._series):
            raise BuilderConfigError(
                f"mark_point(): series_index {series_index} out of range "
                f"(have {len(self._series)} series)."
            )

        mark_data: dict = {}
        if type is not None:
            mark_data["type"] = type
        if label is not None:
            mark_data["name"] = label
        if coord is not None:
            mark_data["coord"] = coord

        mp = self._series[idx].setdefault(
            "markPoint", {"data": [], "symbolSize": symbol_size}
        )
        mp["data"].append(mark_data)
        return self

    def mark_area(
        self, y_range: Optional[List[float]] = None,
        x_range: Optional[List[str]] = None,
        color: Optional[str] = None, opacity: float = 0.1,
        series_index: int = -1,
    ) -> "Figure":
        """Add a shaded region to a series.

        Parameters
        ----------
        y_range : [low, high] or None
            Horizontal band between two y-values.
        x_range : [start, end] or None
            Vertical band between two x-values.
        color : str or None
            Fill colour.
        opacity : float
            Fill opacity.
        series_index : int
            Which series to attach to (default: last added).
        """
        if not self._series:
            raise BuilderConfigError("mark_area(): no series to annotate.")
        idx = series_index if series_index >= 0 else len(self._series) + series_index
        if idx < 0 or idx >= len(self._series):
            raise BuilderConfigError(
                f"mark_area(): series_index {series_index} out of range "
                f"(have {len(self._series)} series)."
            )

        item_style: dict = {"opacity": opacity}
        if color is not None:
            item_style["color"] = color

        area_data: list = []
        if y_range is not None and len(y_range) == 2:
            area_data = [
                {"yAxis": y_range[0], "itemStyle": item_style},
                {"yAxis": y_range[1]},
            ]
        elif x_range is not None and len(x_range) == 2:
            area_data = [
                {"xAxis": x_range[0], "itemStyle": item_style},
                {"xAxis": x_range[1]},
            ]

        if area_data:
            ma = self._series[idx].setdefault("markArea", {"data": []})
            ma["data"].append(area_data)
        return self

    # ═══════════════════════════════════════════════════════════════════════
    # Build + render
    # ═══════════════════════════════════════════════════════════════════════

    def to_option(self) -> dict:
        """Assemble and return the raw ECharts option dict without rendering.

        Useful for debugging, testing, or feeding into a custom renderer.
        """
        if not self._series:
            if self._subplot_mode:
                return {"_empty": True}
            raise BuilderConfigError("Cannot build option — no series have been added.")

        mode = self._chart_mode or "cartesian"

        # ── Non-cartesian modes ───────────────────────────────────────────
        if mode == "pie":
            option: dict = {}
            if self._title_cfg:
                option["title"] = copy.deepcopy(self._title_cfg)
            option["tooltip"] = {"trigger": "item", "formatter": "{b}: {c} ({d}%)", "confine": True}
            legend_cfg = copy.deepcopy(self._legend_cfg) or {"orient": self._style.legend_orient, "left": "left"}
            legend_cfg["data"] = list(self._legend_items)
            option["legend"] = legend_cfg
            if self._palette:
                option["color"] = list(self._palette)
            option["series"] = copy.deepcopy(self._series)
            if self._toolbox_cfg:
                option["toolbox"] = copy.deepcopy(self._toolbox_cfg)
            option.update({k: copy.deepcopy(v) for k, v in self._extra.items() if not k.startswith("_")})
            return option

        if mode == "radar":
            option = {}
            if self._title_cfg:
                option["title"] = copy.deepcopy(self._title_cfg)
            option["tooltip"] = {"trigger": "item", "confine": True, "appendTo": "body"}
            legend_cfg = copy.deepcopy(self._legend_cfg) or {"left": "center", "show": True, "padding": 40}
            legend_cfg["data"] = list(self._legend_items)
            option["legend"] = legend_cfg
            radar_cfg = copy.deepcopy(self._extra.get("_radar_cfg", {"indicator": [], "radius": 150, "center": ["50%", "55%"]}))
            option["radar"] = radar_cfg
            if self._palette:
                option["color"] = list(self._palette)
            option["series"] = copy.deepcopy(self._series)
            if self._toolbox_cfg:
                option["toolbox"] = copy.deepcopy(self._toolbox_cfg)
            option.update({k: copy.deepcopy(v) for k, v in self._extra.items() if not k.startswith("_")})
            return option

        if mode == "parallel":
            option = {}
            if self._title_cfg:
                option["title"] = copy.deepcopy(self._title_cfg)
            option["tooltip"] = copy.deepcopy(self._tooltip_cfg)
            option["parallelAxis"] = copy.deepcopy(self._extra.get("_parallel_axes", []))
            option["parallel"] = copy.deepcopy(self._extra.get("_parallel_cfg", {}))
            if self._palette:
                option["color"] = list(self._palette)
            option["series"] = copy.deepcopy(self._series)
            if self._toolbox_cfg:
                option["toolbox"] = copy.deepcopy(self._toolbox_cfg)
            if "visualMap" in self._extra:
                option["visualMap"] = copy.deepcopy(self._extra["visualMap"])
            option.update({k: copy.deepcopy(v) for k, v in self._extra.items() if not k.startswith("_")})
            return option

        if mode in ("sankey", "treemap", "funnel", "heatmap",
                    "sunburst", "graph", "gauge", "calendar"):
            option = {}
            if self._title_cfg:
                option["title"] = copy.deepcopy(self._title_cfg)
            option["tooltip"] = copy.deepcopy(self._tooltip_cfg)
            if self._palette:
                option["color"] = list(self._palette)
            if mode == "heatmap":
                option["xAxis"] = copy.deepcopy(self._x_axis)
                option["yAxis"] = copy.deepcopy(self._y_axes[0] if len(self._y_axes) == 1 else self._y_axes)
                option["grid"] = copy.deepcopy(self._grid_cfg)
                if self._axis_pointer_cfg:
                    option["axisPointer"] = copy.deepcopy(self._axis_pointer_cfg)
            if self._legend_items and mode in ("funnel", "graph"):
                legend_cfg = copy.deepcopy(self._legend_cfg) or {"orient": "vertical", "left": "left"}
                legend_cfg["data"] = list(self._legend_items)
                option["legend"] = legend_cfg
            option["series"] = copy.deepcopy(self._series)
            if self._toolbox_cfg:
                option["toolbox"] = copy.deepcopy(self._toolbox_cfg)
            if "visualMap" in self._extra:
                option["visualMap"] = copy.deepcopy(self._extra["visualMap"])
            option.update({k: copy.deepcopy(v) for k, v in self._extra.items() if not k.startswith("_")})
            return option

        # ── Cartesian mode ────────────────────────────────────────────────
        is_h_bar = self._extra.get("_bar_orient_h", False)

        # Multi-grid mode
        if self._n_grids > 1:
            return self._build_multi_grid_option(is_h_bar)

        x_axis_cfg = copy.deepcopy(self._x_axis)
        y_axis_cfg = copy.deepcopy(self._y_axes)

        if not is_h_bar:
            x_axis_cfg["data"] = self._x_categories

        if is_h_bar:
            x_axis_cfg = {"type": "value"}
            y_base = copy.deepcopy(self._y_axes[0]) if self._y_axes else {}
            y_base.update({"type": "category", "data": self._x_categories})
            y_axis_cfg = [y_base]

        option = {}
        if self._title_cfg:
            option["title"] = copy.deepcopy(self._title_cfg)
        if self._palette:
            option["color"] = list(self._palette)
        option["tooltip"] = copy.deepcopy(self._tooltip_cfg)
        if self._axis_pointer_cfg:
            option["axisPointer"] = copy.deepcopy(self._axis_pointer_cfg)

        if self._legend_items:
            legend_cfg = copy.deepcopy(self._legend_cfg) or {}
            legend_cfg["data"] = list(dict.fromkeys(self._legend_items))
            option["legend"] = legend_cfg

        option["grid"] = copy.deepcopy(self._grid_cfg)
        option["xAxis"] = x_axis_cfg
        option["yAxis"] = y_axis_cfg if len(y_axis_cfg) > 1 else y_axis_cfg[0]
        option["series"] = copy.deepcopy(self._series)

        # Overlay pies need per-series tooltip (global "axis" trigger won't fire)
        for s in option["series"]:
            if s.get("type") == "pie" and "tooltip" not in s:
                s["tooltip"] = {"trigger": "item", "formatter": "{b}: {c} ({d}%)"}

        if self._toolbox_cfg:
            option["toolbox"] = copy.deepcopy(self._toolbox_cfg)
        if self._datazoom_cfg:
            option["dataZoom"] = copy.deepcopy(self._datazoom_cfg)

        if self._style.bg:
            option.setdefault("backgroundColor", self._style.bg)
        option.update({k: copy.deepcopy(v) for k, v in self._extra.items() if not k.startswith("_")})

        # Pass internal metadata for the layout resolver (stripped before output)
        option["_meta"] = {"user_set_rotate": self._user_set_rotate}

        # Anti-overlap pass (skip in subplot mode — container handles layout)
        if not self._subplot_mode:
            option = _resolve_layout(option, self._series_meta)

        return option

    def _build_multi_grid_option(self, is_h_bar: bool) -> dict:
        """Build the ECharts option dict for multi-grid (subplot) layouts."""
        n = self._n_grids
        heights = self._row_heights

        # Compute grid positions
        if heights:
            nums = [float(h.rstrip("%")) for h in heights]
            total = sum(nums)
            ratios = [v / total for v in nums]
        else:
            ratios = [1.0 / n] * n

        avail_pct = 75  # percentage of total height available
        start_pct = 10  # top offset percentage
        gap_pct = 5

        grids: list = []
        x_axes: list = []
        y_axes: list = []
        cur_top = start_pct

        base_grid = copy.deepcopy(self._grid_cfg)
        for gi in range(n):
            h_pct = avail_pct * ratios[gi] - (gap_pct if gi < n - 1 else 0)
            g = {
                "left": base_grid.get("left", 70),
                "right": base_grid.get("right", 70),
                "top": f"{cur_top:.0f}%",
                "height": f"{h_pct:.0f}%",
                "containLabel": True,
            }
            grids.append(g)
            cur_top += avail_pct * ratios[gi]

            cats = self._grid_categories.get(gi, [])
            x_cfg = copy.deepcopy(self._x_axis)
            x_cfg["gridIndex"] = gi
            if not is_h_bar:
                x_cfg["data"] = cats
            else:
                x_cfg = {"type": "value", "gridIndex": gi}
            x_axes.append(x_cfg)

            # Primary y-axis (left) for this grid
            y_cfg = copy.deepcopy(self._y_axes[0])
            y_cfg["gridIndex"] = gi
            y_axes.append(y_cfg)

            # Secondary y-axis (right) for this grid — keeps indices
            # consistent with the formula: y_idx = axis + grid * 2
            if len(self._y_axes) > 1:
                y2_cfg = copy.deepcopy(self._y_axes[1])
            else:
                y2_cfg = {"type": "value", "show": False}
            y2_cfg["gridIndex"] = gi
            y_axes.append(y2_cfg)

        option: dict = {}
        if self._title_cfg:
            option["title"] = copy.deepcopy(self._title_cfg)
        if self._palette:
            option["color"] = list(self._palette)
        option["tooltip"] = copy.deepcopy(self._tooltip_cfg)
        if self._axis_pointer_cfg:
            option["axisPointer"] = copy.deepcopy(self._axis_pointer_cfg)

        if self._legend_items:
            legend_cfg = copy.deepcopy(self._legend_cfg) or {}
            legend_cfg["data"] = list(dict.fromkeys(self._legend_items))
            option["legend"] = legend_cfg

        option["grid"] = grids
        option["xAxis"] = x_axes
        option["yAxis"] = y_axes
        option["series"] = copy.deepcopy(self._series)

        # Overlay pies need per-series tooltip (global "axis" trigger won't fire)
        for s in option["series"]:
            if s.get("type") == "pie" and "tooltip" not in s:
                s["tooltip"] = {"trigger": "item", "formatter": "{b}: {c} ({d}%)"}

        if self._toolbox_cfg:
            option["toolbox"] = copy.deepcopy(self._toolbox_cfg)
        if self._datazoom_cfg:
            option["dataZoom"] = copy.deepcopy(self._datazoom_cfg)
        if self._style.bg:
            option.setdefault("backgroundColor", self._style.bg)

        option.update({k: copy.deepcopy(v) for k, v in self._extra.items() if not k.startswith("_")})
        return option

    def show(self, **render_kw: Any) -> None:
        """Render the chart using the currently configured engine.

        Calls :meth:`to_option` to assemble the full config, runs the
        anti-overlap engine, and dispatches to the active renderer.

        Returns ``None`` so that Jupyter does not auto-display the option
        dict below the chart.  Use :meth:`to_option` if you need the raw
        dict.
        """
        option = self.to_option()

        render(
            option,
            height=self._height,
            width=self._width,
            theme=self._theme,
            renderer=self._renderer,
            key=self._key,
            **render_kw,
        )

    def to_html(self, filepath: str = "chart.html") -> str:
        """Export the chart to a standalone HTML file.

        Parameters
        ----------
        filepath : str
            Output file path.

        Returns
        -------
        str
            The filepath written to.
        """
        import os
        from echartsy._config import get_adaptive
        from echartsy.renderers._html_template import build_html

        # Validate parent directory exists
        parent = os.path.dirname(os.path.abspath(filepath))
        if not os.path.isdir(parent):
            raise FileNotFoundError(
                f"to_html(): directory '{parent}' does not exist. "
                "Please create it first or use a different path."
            )

        option = self.to_option()
        html = build_html(
            option, height=self._height,
            width=self._width or "100%",
            theme=self._theme, renderer=self._renderer,
            adaptive=get_adaptive(),
        )
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
        except OSError as e:
            raise OSError(f"Cannot write to '{filepath}': {e}") from e
        abs_path = os.path.abspath(filepath)
        print(f"Chart saved to {abs_path}")
        return filepath


# ═══════════════════════════════════════════════════════════════════════════
# Convenience alias — feels like matplotlib
# ═══════════════════════════════════════════════════════════════════════════

def figure(
    height: str = "400px",
    width: Optional[str] = None,
    renderer: Literal["canvas", "svg"] = "svg",
    style: Optional[StylePreset] = None,
    **kwargs: Any,
) -> Figure:
    """Module-level factory — equivalent to ``plt.figure()``.

    Example
    -------
    >>> import echartsy as ec
    >>> fig = ec.figure(height="600px", style=ec.StylePreset.KPI_REPORT)
    >>> fig.bar(df, "x", "y").show()
    """
    return Figure(height=height, width=width, renderer=renderer, style=style, **kwargs)

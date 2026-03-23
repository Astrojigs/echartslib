"""
TimelineFigure — animated companion to Figure.

Every series method accepts an extra ``time_col`` argument that partitions
the DataFrame into frames. The resulting ECharts option uses the
``{ baseOption, options }`` timeline structure.
"""
from __future__ import annotations

import copy
import itertools
import re
import warnings
from typing import (
    Any, Dict, List, Literal, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd

from echartsy._helpers import (
    _coerce_numeric, _resolve_agg, _resolve_layout,
    _sort_categories, _validate_columns, _validate_df,
)
from echartsy.exceptions import (
    BuilderConfigError, DataValidationError, TimelineConfigError,
)
from echartsy.emphasis import (
    ItemStyle, LabelStyle, LineStyle, AreaStyle, EndLabelStyle,
    LabelLineStyle, Blur, Select, TooltipStyle, AnimationConfig,
    LineEmphasis, ScatterEmphasis, PieEmphasis, Emphasis,
)
from echartsy.figure import _SeriesMeta, _build_axis_pointer_cfg, _merge_style_params
from echartsy.renderers import render
from echartsy.styles import StylePreset

# ═══════════════════════════════════════════════════════════════════════════
# Temporal-parser patterns (for auto-sorting timeline frames)
# ═══════════════════════════════════════════════════════════════════════════

_TL_RE_YEAR        = re.compile(r"^(\d{4})$")
_TL_RE_Q_PREFIX    = re.compile(r"^[Qq]([1-4])[\s/\-](\d{4})$")
_TL_RE_Q_SUFFIX    = re.compile(r"^(\d{4})[\s/\-][Qq]([1-4])$")
_TL_RE_HALF_PREFIX = re.compile(r"^[HhSs]([12])[\s/\-](\d{4})$")
_TL_RE_MONTH_ISO   = re.compile(r"^(\d{4})[\-/](\d{1,2})$")
_TL_RE_WEEK_PREFIX = re.compile(r"^[Ww](\d{1,2})[\s/\-](\d{4})$")
_TL_RE_FY_LONG     = re.compile(r"^FY[\s]?(\d{4})(?:[/\-]\d{2,4})?$", re.IGNORECASE)
_TL_RE_DAY_ISO     = re.compile(r"^(\d{4})[\-/](\d{1,2})[\-/](\d{1,2})$")

_TL_MONTH_MAP: Dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9, "sept": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_TL_Q_MONTH = {1: 1, 2: 4, 3: 7, 4: 10}
_TL_H_MONTH = {1: 1, 2: 7}


def _parse_temporal_label(v: str) -> Optional[pd.Timestamp]:
    """Parse common temporal period labels into a Timestamp (start of period)."""
    v = str(v).strip()
    m = _TL_RE_YEAR.match(v)
    if m:
        return pd.Timestamp(int(m.group(1)), 1, 1)
    m = _TL_RE_Q_PREFIX.match(v)
    if m:
        return pd.Timestamp(int(m.group(2)), _TL_Q_MONTH[int(m.group(1))], 1)
    m = _TL_RE_Q_SUFFIX.match(v)
    if m:
        return pd.Timestamp(int(m.group(1)), _TL_Q_MONTH[int(m.group(2))], 1)
    m = _TL_RE_HALF_PREFIX.match(v)
    if m:
        return pd.Timestamp(int(m.group(2)), _TL_H_MONTH[int(m.group(1))], 1)
    m = _TL_RE_MONTH_ISO.match(v)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return pd.Timestamp(y, mo, 1)
    # Text month: "Jan 2024"
    parts = re.split(r"[\s\-/,]+", v, maxsplit=1)
    if len(parts) == 2:
        first, second = parts[0].lower(), parts[1].strip()
        if first in _TL_MONTH_MAP:
            try:
                return pd.Timestamp(int(second), _TL_MONTH_MAP[first], 1)
            except (ValueError, TypeError):
                pass
    m = _TL_RE_FY_LONG.match(v)
    if m:
        return pd.Timestamp(int(m.group(1)), 4, 1)
    m = _TL_RE_DAY_ISO.match(v)
    if m:
        try:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    # Last resort
    if not _TL_RE_YEAR.match(v):
        try:
            return pd.to_datetime(v)
        except (ValueError, TypeError, OverflowError):
            pass
    return None


def detect_time_format(series: pd.Series) -> str:
    """Inspect a column and describe the temporal format detected.

    Call this on the column you intend to pass as ``time_col`` to verify
    that TimelineFigure will sort the frames correctly.
    """
    uniq = series.dropna().astype(str).str.strip().unique().tolist()
    if not uniq:
        return "empty series — no frames detected"
    parsed = [(v, _parse_temporal_label(v)) for v in uniq]
    n_ok = sum(1 for _, ts in parsed if ts is not None)
    sample = str(uniq[0])
    status = "sorted correctly" if n_ok == len(parsed) else f"{len(parsed) - n_ok} label(s) unrecognised"
    return f"{n_ok}/{len(uniq)} unique frames parsed, {status}\n  Sample: {sample!r}"


_TL_COUNTER = itertools.count()

# ═══════════════════════════════════════════════════════════════════════════
# ██  TimelineFigure  ██
# ═══════════════════════════════════════════════════════════════════════════

class TimelineFigure:
    """Fluent builder for ECharts charts with an animated timeline slider.

    Mirrors the Figure API. Every series method gains one required extra
    parameter — ``time_col`` — which names the DataFrame column whose
    unique values become the animation frames.

    Example
    -------
    >>> import echartsy as ec
    >>> ec.config(engine="jupyter")
    >>> (
    ...     ec.TimelineFigure(height="500px", interval=1.5)
    ...     .bar(df, x="Country", y="GDP", time_col="Year")
    ...     .title("GDP Race")
    ...     .ylabel("GDP (USD bn)")
    ...     .show()
    ... )
    """

    def __init__(
        self,
        height: str = "450px",
        width: Optional[str] = None,
        renderer: Literal["canvas", "svg"] = "svg",
        theme: Optional[str] = None,
        style: Optional[StylePreset] = None,
        key: Optional[str] = None,
        interval: float = 2.0,
        autoplay: bool = True,
        loop: bool = True,
    ) -> None:
        self._height = height
        self._width = width
        self._renderer = renderer
        self._theme = theme
        self._key = key
        self._style = style or StylePreset.DEFAULT
        self._default_interval = interval
        self._autoplay = autoplay
        self._loop = loop

        self._frames: Dict[str, Dict[str, Any]] = {}
        self._frame_order: List[str] = []

        self._title_cfg: Optional[dict] = None
        self._legend_cfg: Optional[dict] = None
        self._toolbox_cfg: Optional[dict] = None
        self._datazoom_cfg: Optional[list] = None
        self._palette: Optional[List[str]] = (
            list(self._style.palette) if self._style.palette else None
        )
        self._extra: dict = {}
        self._timeline_style: dict = {}
        self._user_set_rotate: bool = False

        self._grid_cfg: dict = {
            "left": 70, "right": 70, "top": 60, "bottom": 80,
            "containLabel": True,
        }
        self._x_axis_template: dict = {
            "type": "category", "data": [],
            "axisLabel": {
                "rotate": 0,
                "fontSize": self._style.axis_label_font_size,
                "color": self._style.axis_label_color,
            },
            "axisTick": {"show": False},
        }
        self._y_axes_template: List[dict] = [{
            "type": "value",
            "splitLine": {"show": True, "lineStyle": {"color": self._style.grid_line_color}},
            "axisLabel": {
                "fontSize": self._style.axis_label_font_size,
                "color": self._style.axis_label_color,
            },
        }]
        self._global_x_cats: List[str] = []
        self._chart_mode: Optional[str] = None
        self._tooltip_cfg: dict = {
            "trigger": "axis",
            "axisPointer": {"type": self._style.tooltip_pointer},
            "confine": True, "appendTo": "body",
        }
        self._axis_pointer_cfg: Optional[dict] = None

    def __repr__(self) -> str:
        return (
            f"<TimelineFigure frames={len(self._frame_order)} "
            f"mode={self._chart_mode!r} height={self._height!r}>"
        )

    # ─── Private helpers ──────────────────────────────────────────────────

    def _ensure_mode(self, mode: str, caller: str) -> None:
        if self._chart_mode is None:
            self._chart_mode = mode
        elif self._chart_mode != mode:
            raise BuilderConfigError(
                f"{caller}() requires mode '{mode}' but figure is already in "
                f"'{self._chart_mode}' mode."
            )

    def _ensure_cartesian(self, caller: str) -> None:
        self._ensure_mode("cartesian", caller)

    def _auto_key(self) -> str:
        return f"ecb_tl_{next(_TL_COUNTER)}"

    def _get_frame(self, label: str) -> Dict[str, Any]:
        if label not in self._frames:
            self._frames[label] = {
                "series": [], "x_cats": [], "meta": [],
                "legend_items": [], "extra": {},
                "tooltip": copy.deepcopy(self._tooltip_cfg),
            }
            self._frame_order.append(label)
        return self._frames[label]

    def _sort_frame_order(self) -> None:
        parsed = {lbl: _parse_temporal_label(lbl) for lbl in self._frame_order}
        if all(v is not None for v in parsed.values()):
            self._frame_order = sorted(self._frame_order, key=lambda l: parsed[l])
        else:
            failed = [l for l, v in parsed.items() if v is None]
            warnings.warn(
                f"TimelineFigure: {len(failed)} frame label(s) could not be parsed — "
                f"preserving insertion order. Tip: call detect_time_format() to diagnose.",
                UserWarning, stacklevel=4,
            )

    def _merge_global_x_cats(self, new_cats: List[str]) -> None:
        existing = set(self._global_x_cats)
        for c in new_cats:
            if c not in existing:
                self._global_x_cats.append(c)
                existing.add(c)

    def _align_to_cats(
        self, df: pd.DataFrame, x: str, y: str,
        cats: List[str], agg: str = "mean",
    ) -> List[Optional[float]]:
        agg_fn = _resolve_agg(agg)
        grouped = df.groupby(x)[y].agg(agg_fn)
        return [
            None if cat not in grouped.index or pd.isna(grouped.get(cat))
            else round(float(grouped.get(cat)), 4)
            for cat in cats
        ]

    def _resolve_interval(self, interval: Optional[float]) -> int:
        secs = interval if interval is not None else self._default_interval
        if secs <= 0:
            raise TimelineConfigError(f"interval must be > 0 seconds (got {secs}).")
        return int(secs * 1000)

    # ═══════════════════════════════════════════════════════════════════════
    # Chrome — identical surface to Figure
    # ═══════════════════════════════════════════════════════════════════════

    def title(self, text: str, subtitle: Optional[str] = None,
              left: str = "center", top: Optional[Union[str, int]] = None,
              text_color: Optional[str] = None,
              text_font_weight: Optional[str] = None,
              link: Optional[str] = None,
              sublink: Optional[str] = None,
              background_color: Optional[str] = None,
              border_color: Optional[str] = None,
              border_width: Optional[int] = None,
              padding: Optional[Union[int, list]] = None) -> "TimelineFigure":
        self._title_cfg = {
            "text": text, "left": left,
            "textStyle": {"fontSize": self._style.title_font_size, "fontFamily": self._style.font_family},
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

    def xlabel(self, name: str, rotate: Optional[int] = None,
               font_size: Optional[int] = None, color: Optional[str] = None) -> "TimelineFigure":
        self._x_axis_template["name"] = name
        lbl = self._x_axis_template.setdefault("axisLabel", {})
        if rotate is not None:
            lbl["rotate"] = rotate
            self._user_set_rotate = True
        if font_size is not None: lbl["fontSize"] = font_size
        if color is not None: lbl["color"] = color
        return self

    def ylabel(self, name: str, font_size: Optional[int] = None,
               color: Optional[str] = None) -> "TimelineFigure":
        self._y_axes_template[0]["name"] = name
        lbl = self._y_axes_template[0].setdefault("axisLabel", {})
        if font_size is not None: lbl["fontSize"] = font_size
        if color is not None: lbl["color"] = color
        return self

    def ylabel_right(self, name: str, font_size: Optional[int] = None,
                     color: Optional[str] = None) -> "TimelineFigure":
        if len(self._y_axes_template) < 2:
            self._y_axes_template.append({
                "type": "value", "splitLine": {"show": False},
                "axisLabel": {"fontSize": self._style.axis_label_font_size,
                              "color": self._style.axis_label_color},
            })
        self._y_axes_template[1]["name"] = name
        lbl = self._y_axes_template[1].setdefault("axisLabel", {})
        if font_size is not None: lbl["fontSize"] = font_size
        if color is not None: lbl["color"] = color
        return self

    def legend(self, show: bool = True, orient: Optional[str] = None,
               left: Optional[str] = None, top: Optional[Union[str, int]] = None,
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
               scroll: Optional[bool] = None) -> "TimelineFigure":
        cfg: dict = {"show": show, "orient": orient or self._style.legend_orient}
        if left is not None: cfg["left"] = left
        if top is not None: cfg["top"] = top
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
    ) -> "TimelineFigure":
        """Configure tooltip behaviour and axis-pointer styling."""
        self._tooltip_cfg["trigger"] = trigger
        ap = _build_axis_pointer_cfg(
            type_val=pointer, snap=snap, pointer_label=pointer_label,
            line_color=line_color, line_width=line_width, line_type=line_type,
            cross_color=cross_color, cross_width=cross_width, cross_type=cross_type,
            shadow_color=shadow_color, shadow_opacity=shadow_opacity,
        )
        self._tooltip_cfg["axisPointer"] = ap
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
        for fd in self._frames.values():
            fd["tooltip"] = copy.deepcopy(self._tooltip_cfg)
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
    ) -> "TimelineFigure":
        """Configure the global axis pointer (top-level ``baseOption.axisPointer``).

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
        for fd in self._frames.values():
            fd["tooltip"]["axisPointer"] = copy.deepcopy(cfg)
        return self

    def palette(self, colors: Sequence[str]) -> "TimelineFigure":
        self._palette = list(colors)
        return self

    def margins(self, left=None, right=None, top=None, bottom=None) -> "TimelineFigure":
        if left is not None: self._grid_cfg["left"] = left
        if right is not None: self._grid_cfg["right"] = right
        if top is not None: self._grid_cfg["top"] = top
        if bottom is not None: self._grid_cfg["bottom"] = bottom
        return self

    def playback(self, interval: Optional[float] = None, autoplay: Optional[bool] = None,
                 loop: Optional[bool] = None, rewind: bool = False) -> "TimelineFigure":
        if interval is not None: self._default_interval = interval
        if autoplay is not None: self._autoplay = autoplay
        if loop is not None: self._loop = loop
        if rewind: self._timeline_style["rewind"] = True
        return self

    def save(self, name: str = "chart", fmt: Literal["png", "svg"] = "png",
             dpi: int = 3, bg: Optional[str] = "#ffffff") -> "TimelineFigure":
        self._toolbox_cfg = {
            "show": True, "right": 10, "top": 10,
            "feature": {
                "saveAsImage": {"show": True, "type": fmt, "name": name,
                                "pixelRatio": dpi, "backgroundColor": bg},
                "restore": {"show": True},
            },
        }
        return self

    def xlim(self, min_val: Optional[float] = None, max_val: Optional[float] = None,
             scale: Optional[Literal["value", "log"]] = None) -> "TimelineFigure":
        """Set explicit x-axis bounds and/or scale type."""
        if min_val is not None:
            self._x_axis_template["min"] = min_val
        if max_val is not None:
            self._x_axis_template["max"] = max_val
        if scale is not None:
            self._x_axis_template["type"] = scale
        return self

    def ylim(self, min_val: Optional[float] = None,
             max_val: Optional[float] = None, axis: int = 0,
             scale: Optional[Literal["value", "log"]] = None) -> "TimelineFigure":
        """Set y-axis bounds and/or scale type."""
        if axis < 0:
            raise ValueError("axis index must be non-negative")
        if axis >= len(self._y_axes_template):
            raise BuilderConfigError(
                f"ylim(axis={axis}) — only {len(self._y_axes_template)} y-axis(es) exist. "
                "Call ylabel_right() first to create a second axis."
            )
        if min_val is not None:
            self._y_axes_template[axis]["min"] = min_val
        if max_val is not None:
            self._y_axes_template[axis]["max"] = max_val
        if scale is not None:
            self._y_axes_template[axis]["type"] = scale
        return self

    def extra(self, **kwargs: Any) -> "TimelineFigure":
        self._extra.update(kwargs)
        return self

    # ═══════════════════════════════════════════════════════════════════════
    # Series methods
    # ═══════════════════════════════════════════════════════════════════════

    def plot(self, df: pd.DataFrame, x: str, y: str, *, time_col: str,
             hue: Optional[str] = None, smooth: bool = False,
             area: bool = False, area_opacity: float = 0.15,
             connect_nulls: bool = False, line_width: int = 2,
             symbol_size: int = 6, symbol: str = "circle",
             labels: bool = False, label_position: str = "top",
             label_prefix: str = "", label_suffix: str = "",
             agg: str = "mean", axis: int = 0,
             interval: Optional[float] = None,
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
             **series_kw: Any) -> "TimelineFigure":
        """Add timeline-animated line series."""
        self._ensure_cartesian("plot")
        df = _validate_df(df, "plot")
        _validate_columns(df, [x, y, hue, time_col], "plot")
        self._resolve_interval(interval)

        dff = df.copy()
        dff[x] = dff[x].astype(str).str.strip()
        dff[y] = _coerce_numeric(dff, y, "plot")
        dff[time_col] = dff[time_col].astype(str).str.strip()
        if hue:
            dff[hue] = dff[hue].astype(str).str.strip()
            dff = dff.dropna(subset=[hue])
        dff = dff.dropna(subset=[x, y, time_col])

        self._merge_global_x_cats(_sort_categories(dff[x]))
        if axis == 1 and len(self._y_axes_template) < 2:
            self.ylabel_right("")

        base: dict = {
            "type": "line", "smooth": smooth, "connectNulls": connect_nulls,
            "showSymbol": True, "symbol": symbol, "symbolSize": symbol_size,
            "lineStyle": {"width": line_width}, "yAxisIndex": axis,
        }
        if area: base["areaStyle"] = {"opacity": area_opacity}
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

        for time_val, tgrp in dff.groupby(time_col, sort=False):
            frame = self._get_frame(str(time_val))
            cats = _sort_categories(tgrp[x])
            for c in cats:
                if c not in frame["x_cats"]: frame["x_cats"].append(c)
            groups = tgrp.groupby(hue) if hue else [(y, tgrp)]
            for name, grp in groups:
                ns = str(name)
                frame["series"].append({
                    **base, "name": ns,
                    "data": self._align_to_cats(grp, x, y, frame["x_cats"], agg),
                })
                frame["meta"].append(_SeriesMeta("line", ns, axis))
                if ns not in frame["legend_items"]: frame["legend_items"].append(ns)

        self._sort_frame_order()
        return self

    def bar(self, df: pd.DataFrame, x: str, y: str, *, time_col: str,
            hue: Optional[str] = None, stack: bool = False,
            orient: Literal["v", "h"] = "v",
            bar_width: Optional[Union[int, str]] = None,
            bar_gap: Optional[str] = None,
            border_radius: int = 4, labels: bool = False,
            label_formatter: str = "{c}", label_font_size: int = 12,
            label_color: str = "#333",
            gradient: bool = False,
            gradient_colors: Tuple[str, str] = ("#83bff6", "#188df0"),
            agg: str = "sum", axis: int = 0,
            interval: Optional[float] = None,
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
            **series_kw: Any) -> "TimelineFigure":
        """Add timeline-animated bar series."""
        self._ensure_cartesian("bar")
        df = _validate_df(df, "bar")
        _validate_columns(df, [x, y, hue, time_col], "bar")
        self._resolve_interval(interval)

        dff = df.copy()
        dff[x] = dff[x].astype(str).str.strip()
        dff[y] = _coerce_numeric(dff, y, "bar")
        dff[time_col] = dff[time_col].astype(str).str.strip()
        if hue:
            dff[hue] = dff[hue].astype(str).str.strip()
            dff = dff.dropna(subset=[hue])
        dff = dff.dropna(subset=[x, y, time_col])

        self._merge_global_x_cats(_sort_categories(dff[x]))
        if axis == 1 and len(self._y_axes_template) < 2:
            self.ylabel_right("")

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

        base: dict = {
            "type": "bar",
            "label": {
                "show": labels,
                "position": "top" if orient == "v" else "right",
                "formatter": label_formatter,
                "fontSize": label_font_size, "color": label_color,
            },
            "itemStyle": bar_item_style, "yAxisIndex": axis,
        }
        if stack: base["stack"] = "total"
        if bar_width is not None: base["barWidth"] = bar_width
        if bar_gap is not None: base["barGap"] = bar_gap
        if bar_min_width is not None: base["barMinWidth"] = bar_min_width
        if bar_category_gap is not None: base["barCategoryGap"] = bar_category_gap
        if emphasis is not None:
            base["emphasis"] = emphasis.to_dict()
        _merge_style_params(
            base, item_style=item_style, label_style=label_style,
            color=color, blur=blur, select=select,
            selected_mode=selected_mode, animation=animation,
            tooltip=tooltip,
        )
        base.update(series_kw)

        for time_val, tgrp in dff.groupby(time_col, sort=False):
            frame = self._get_frame(str(time_val))
            cats = _sort_categories(tgrp[x])
            for c in cats:
                if c not in frame["x_cats"]: frame["x_cats"].append(c)
            groups = tgrp.groupby(hue) if hue else [(y, tgrp)]
            for name, grp in groups:
                ns = str(name)
                frame["series"].append({
                    **base, "name": ns,
                    "data": self._align_to_cats(grp, x, y, frame["x_cats"], agg),
                })
                frame["meta"].append(_SeriesMeta("bar", ns, axis))
                if ns not in frame["legend_items"]: frame["legend_items"].append(ns)
            if orient == "h":
                frame["extra"]["_bar_orient_h"] = True

        self._sort_frame_order()
        return self

    def pie(self, df: pd.DataFrame, names: str, values: str, *,
            time_col: str, radius: Union[str, List[str]] = ["40%", "70%"],
            rose_type: Optional[Literal["radius", "area"]] = None,
            start_angle: int = 90, border_radius: int = 6,
            show_labels: bool = True, label_formatter: str = "{b}: {d}%",
            label_font_size: Optional[int] = None,
            center: Optional[List[str]] = None,
            interval: Optional[float] = None,
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
            **series_kw: Any) -> "TimelineFigure":
        """Add timeline-animated pie / donut chart."""
        self._ensure_mode("pie", "pie")
        df = _validate_df(df, "pie")
        _validate_columns(df, [names, values, time_col], "pie")
        self._resolve_interval(interval)

        dff = df.copy()
        dff[values] = _coerce_numeric(dff, values, "pie")
        dff[time_col] = dff[time_col].astype(str).str.strip()
        dff = dff.dropna(subset=[names, values, time_col])

        for time_val, tgrp in dff.groupby(time_col, sort=False):
            frame = self._get_frame(str(time_val))
            data = [{"name": str(n), "value": round(float(v), 4)}
                    for n, v in zip(tgrp[names], tgrp[values])]
            entry: dict = {
                "type": "pie", "radius": radius,
                "center": center or ["50%", "55%"],
                "startAngle": start_angle, "data": data,
                "avoidLabelOverlap": True,
                "label": {"show": show_labels, "formatter": label_formatter},
                "labelLine": {"show": show_labels},
            }
            if border_radius:
                entry.setdefault("itemStyle", {})["borderRadius"] = border_radius
            if rose_type:
                entry["roseType"] = rose_type
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
            if label_font_size is not None:
                entry["label"]["fontSize"] = label_font_size
            if animation_type is not None:
                entry["animationType"] = animation_type
            if item_style is not None:
                entry.setdefault("itemStyle", {}).update(item_style.to_dict())
            if emphasis is not None:
                entry["emphasis"] = emphasis.to_dict()
            if blur is not None:
                entry["blur"] = blur.to_dict()
            if select is not None:
                entry["select"] = select.to_dict()
            if selected_mode is not None:
                entry["selectedMode"] = selected_mode
            if animation is not None:
                anim_cfg = animation.to_dict()
                for k, v in anim_cfg.items():
                    entry[k] = v
            if tooltip is not None:
                entry["tooltip"] = tooltip.to_dict()
            entry.update(series_kw)
            frame["series"].append(entry)
            frame["meta"].append(_SeriesMeta("pie", names))
            frame["legend_items"] = [str(n) for n in tgrp[names]]

        self._sort_frame_order()
        return self

    def scatter(self, df: pd.DataFrame, x: str, y: str, *,
                time_col: str, color: Optional[str] = None,
                size: Optional[str] = None, size_range: Tuple[int, int] = (5, 30),
                symbol: str = "circle", opacity: float = 0.7,
                labels: bool = False,
                interval: Optional[float] = None,
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
                **series_kw: Any) -> "TimelineFigure":
        """Add timeline-animated scatter series."""
        self._ensure_cartesian("scatter")
        df = _validate_df(df, "scatter")
        _validate_columns(df, [x, y, color, size, time_col], "scatter")
        self._resolve_interval(interval)

        dff = df.copy()
        dff[x] = _coerce_numeric(dff, x, "scatter")
        dff[y] = _coerce_numeric(dff, y, "scatter")
        dff[time_col] = dff[time_col].astype(str).str.strip()
        dff = dff.dropna(subset=[x, y, time_col])

        self._x_axis_template["type"] = "value"
        self._x_axis_template.pop("data", None)
        self._tooltip_cfg["trigger"] = "item"

        def _build_data(sub):
            rows = []
            sizes_list = None
            if size is not None:
                vals = pd.to_numeric(sub[size], errors="coerce").fillna(0)
                if vals.max() != vals.min():
                    normed = (vals - vals.min()) / (vals.max() - vals.min())
                    sizes_list = [int(size_range[0] + n * (size_range[1] - size_range[0])) for n in normed]
                else:
                    sizes_list = [int((size_range[0] + size_range[1]) / 2)] * len(vals)
            for i, (_, r) in enumerate(sub.iterrows()):
                pt = [float(r[x]), float(r[y])]
                if sizes_list is not None: pt.append(sizes_list[i])
                rows.append(pt)
            return rows

        for time_val, tgrp in dff.groupby(time_col, sort=False):
            frame = self._get_frame(str(time_val))
            groups = tgrp.groupby(color) if color else [("scatter", tgrp)]
            for name, grp in groups:
                ns = str(name)
                entry = {
                    "type": "scatter", "name": ns,
                    "data": _build_data(grp), "symbol": symbol,
                    "itemStyle": {"opacity": opacity},
                }
                if labels:
                    entry["label"] = {"show": True, "position": "top"}
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
                frame["series"].append(entry)
                frame["meta"].append(_SeriesMeta("scatter", ns))
                if ns not in frame["legend_items"]:
                    frame["legend_items"].append(ns)

        self._sort_frame_order()
        return self

    def hist(self, df: pd.DataFrame, column: str, *, time_col: str,
             bins: int = 10, density: bool = False,
             bar_color: Optional[str] = None,
             border_radius: int = 2, labels: bool = False,
             interval: Optional[float] = None,
             emphasis: Optional[Emphasis] = None,
             item_style: Optional[ItemStyle] = None,
             label_style: Optional[LabelStyle] = None,
             animation: Optional[AnimationConfig] = None,
             selected_mode: Optional[Union[bool, str]] = None,
             tooltip: Optional[TooltipStyle] = None,
             **series_kw: Any) -> "TimelineFigure":
        """Add a timeline-animated histogram.

        Computes bin edges from the **full dataset** (across all frames) so
        that bins are directly comparable across timeline frames.

        Parameters
        ----------
        df : pd.DataFrame
            Source data.
        column : str
            Numeric column to bin.
        time_col : str
            Column whose unique values become animation frames.
        bins : int
            Number of histogram bins (default 10).
        density : bool
            If ``True``, normalize to a probability density.
        bar_color : str or None
            Override bar colour for all frames.
        border_radius : int
            Corner rounding on bars.
        labels : bool
            Show value labels on bars.
        interval : float or None
            Override playback interval (seconds) for this series.
        """
        self._ensure_cartesian("hist")
        df = _validate_df(df, "hist")
        _validate_columns(df, [column, time_col], "hist")
        self._resolve_interval(interval)

        if bins is not None and bins <= 0:
            raise DataValidationError("hist(): bins must be a positive integer.")

        dff = df.copy()
        dff[column] = _coerce_numeric(dff, column, "hist")
        dff[time_col] = dff[time_col].astype(str).str.strip()
        n_before = len(dff)
        dff = dff.dropna(subset=[column, time_col])
        n_dropped = n_before - len(dff)
        if n_dropped > 0:
            warnings.warn(f"hist(): {n_dropped} rows dropped due to missing values", stacklevel=2)

        all_vals = dff[column].astype(float).values
        if len(all_vals) == 0:
            raise DataValidationError(
                f"hist(): Column '{column}' has no non-null numeric values."
            )

        # Compute global bin edges so bins are comparable across frames
        _, edges = np.histogram(all_vals, bins=bins)
        bin_labels = [f"{edges[i]:.1f}\u2013{edges[i + 1]:.1f}" for i in range(len(edges) - 1)]
        self._merge_global_x_cats(bin_labels)

        self._x_axis_template["name"] = column
        self._y_axes_template[0].setdefault("name", "Density" if density else "Count")

        # Histogram bin labels (e.g. "34.5–45.2") are inherently long;
        # pre-rotate to avoid OverlapWarning from the layout resolver.
        self._x_axis_template.setdefault("axisLabel", {}).setdefault("rotate", 30)
        self._user_set_rotate = True
        self._tooltip_cfg["trigger"] = "axis"

        for time_val, tgrp in dff.groupby(time_col, sort=False):
            frame = self._get_frame(str(time_val))

            frame_vals = tgrp[column].astype(float).values
            if len(frame_vals) == 0:
                counts = np.zeros(len(edges) - 1)
            else:
                counts, _ = np.histogram(frame_vals, bins=edges, density=density)

            for bl in bin_labels:
                if bl not in frame["x_cats"]:
                    frame["x_cats"].append(bl)

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
            frame["series"].append(entry)
            frame["meta"].append(_SeriesMeta("bar", column))
            if column not in frame["legend_items"]:
                frame["legend_items"].append(column)

        self._sort_frame_order()
        return self

    # ═══════════════════════════════════════════════════════════════════════
    # Build + render
    # ═══════════════════════════════════════════════════════════════════════

    def to_option(self) -> dict:
        """Assemble the full ECharts timeline option dict."""
        if not self._frames:
            raise BuilderConfigError("Cannot build option — no series added.")

        self._sort_frame_order()
        interval_ms = self._resolve_interval(None)
        mode = self._chart_mode or "cartesian"

        tl: dict = {
            "axisType": "category", "autoPlay": self._autoplay,
            "playInterval": interval_ms, "loop": self._loop,
            "data": self._frame_order,
            "left": "10%", "right": "10%", "bottom": 5,
            "padding": 5, "symbolSize": 10, "controlPosition": "left",
            "label": {
                "fontSize": self._style.axis_label_font_size,
                "color": self._style.axis_label_color,
            },
        }
        tl.update(self._timeline_style)

        base: dict = {"timeline": tl}
        if self._title_cfg: base["title"] = self._title_cfg
        if self._palette: base["color"] = list(self._palette)
        if self._toolbox_cfg: base["toolbox"] = self._toolbox_cfg
        if self._datazoom_cfg: base["dataZoom"] = self._datazoom_cfg
        if self._axis_pointer_cfg: base["axisPointer"] = copy.deepcopy(self._axis_pointer_cfg)

        frame_options: List[dict] = []
        all_legend_items: List[str] = []

        for lbl in self._frame_order:
            fd = self._frames[lbl]
            f_opt: dict = {}
            is_h_bar = fd["extra"].get("_bar_orient_h", False)

            if mode == "pie":
                f_opt["tooltip"] = {"trigger": "item", "formatter": "{b}: {c} ({d}%)", "confine": True}
                f_opt["series"] = fd["series"]
            elif mode == "radar":
                f_opt["tooltip"] = {"trigger": "item", "confine": True, "appendTo": "body"}
                f_opt["series"] = fd["series"]
            else:
                # Cartesian
                frame_cats = fd["x_cats"] or self._global_x_cats
                x_axis_cfg = copy.deepcopy(self._x_axis_template)
                y_axis_cfg = copy.deepcopy(self._y_axes_template)

                if is_h_bar:
                    x_axis_cfg = {"type": "value"}
                    y_axis_cfg = [{"type": "category", "data": frame_cats}]
                elif x_axis_cfg.get("type") != "value":
                    x_axis_cfg["data"] = frame_cats

                f_opt["xAxis"] = x_axis_cfg
                f_opt["yAxis"] = y_axis_cfg if len(y_axis_cfg) > 1 else y_axis_cfg[0]
                f_opt["grid"] = copy.deepcopy(self._grid_cfg)
                f_opt["tooltip"] = fd["tooltip"]
                f_opt["series"] = fd["series"]

            for item in fd["legend_items"]:
                if item not in all_legend_items:
                    all_legend_items.append(item)

            if mode == "cartesian" and "xAxis" in f_opt:
                f_opt["_meta"] = {"user_set_rotate": self._user_set_rotate}
                f_opt = _resolve_layout(f_opt, fd["meta"])

            frame_options.append(f_opt)

        if all_legend_items:
            legend_cfg = copy.deepcopy(self._legend_cfg) or {}
            legend_cfg["data"] = all_legend_items
            if len(all_legend_items) > 8:
                legend_cfg["type"] = "scroll"
            base["legend"] = legend_cfg

        if "visualMap" in self._extra:
            base["visualMap"] = self._extra["visualMap"]
        if mode == "radar" and "_radar_cfg" in self._extra:
            base["radar"] = self._extra["_radar_cfg"]

        _skip = {"visualMap", "_radar_cfg", "_heatmap_x_cats", "_heatmap_y_cats"}
        for k, v in self._extra.items():
            if k not in _skip and not k.startswith("_"):
                base[k] = v

        if self._style.bg:
            base.setdefault("backgroundColor", self._style.bg)

        return {"baseOption": base, "options": frame_options}

    def show(self, **render_kw: Any) -> None:
        """Render the animated chart using the currently configured engine."""
        option = self.to_option()
        render(
            option, height=self._height, width=self._width,
            theme=self._theme, renderer=self._renderer,
            key=self._key, **render_kw,
        )

    def to_html(self, filepath: str = "timeline_chart.html") -> str:
        """Export the timeline chart to a standalone HTML file."""
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


def timeline_figure(
    height: str = "450px", width: Optional[str] = None,
    renderer: Literal["canvas", "svg"] = "svg",
    style: Optional[StylePreset] = None,
    interval: float = 2.0, autoplay: bool = True, loop: bool = True,
    **kwargs: Any,
) -> TimelineFigure:
    """Module-level factory for TimelineFigure."""
    return TimelineFigure(
        height=height, width=width, renderer=renderer,
        style=style, interval=interval, autoplay=autoplay,
        loop=loop, **kwargs,
    )

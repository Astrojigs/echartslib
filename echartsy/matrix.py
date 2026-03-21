"""MatrixFigure — table-like grid of independent charts.

Each cell in the matrix can contain an independent chart (bar, line,
scatter, pie, heatmap, etc.) arranged in a grid defined by row and
column labels.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from echartsy._chart_methods import (
    build_bar_series, build_line_series, build_pie_series, build_scatter_series,
)
from echartsy._helpers import _coerce_numeric, _validate_columns, _validate_df
from echartsy.emphasis import (
    Emphasis, ItemStyle, LabelStyle, LineStyle, PieEmphasis,
    ScatterEmphasis, TooltipStyle, Blur, Select, AnimationConfig,
    AreaStyle, EndLabelStyle, LineEmphasis,
)
from echartsy.figure import _merge_style_params
from echartsy.renderers import render
from echartsy.styles import StylePreset


# ── Internal dataclass for cell config ────────────────────────────────────


@dataclass
class _CellConfig:
    """Configuration for one matrix cell."""

    chart_type: Optional[str] = None
    series_configs: list = field(default_factory=list)
    x_axis_config: Optional[dict] = None
    y_axis_config: Optional[dict] = None
    show_axis: bool = True
    legend_names: list = field(default_factory=list)


# ── CellBuilder ──────────────────────────────────────────────────────────


class _CellBuilder:
    """Fluent builder for a single matrix cell."""

    def __init__(self, matrix: "MatrixFigure", row: int, col: int) -> None:
        self._matrix = matrix
        self._row = row
        self._col = col
        self._cell = matrix._cells.setdefault((row, col), _CellConfig())

    def bar(
        self, df: pd.DataFrame, x: str, y: str, *,
        hue: Optional[str] = None, stack: bool = False,
        bar_width: Optional[Union[int, str]] = None,
        bar_gap: Optional[str] = None,
        border_radius: int = 4, labels: bool = False,
        label_formatter: str = "{c}", label_font_size: int = 12,
        label_color: str = "#333",
        gradient: bool = False,
        gradient_colors: Tuple[str, str] = ("#83bff6", "#188df0"),
        agg: str = "sum",
        emphasis: Optional[Emphasis] = None,
        item_style: Optional[ItemStyle] = None,
        label_style: Optional[LabelStyle] = None,
        bar_min_width: Optional[int] = None,
        bar_category_gap: Optional[str] = None,
        color: Optional[str] = None,
        blur: Optional[Blur] = None,
        select: Optional[Select] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        animation: Optional[AnimationConfig] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "MatrixFigure":
        """Add a bar chart to this cell with full Figure.bar() parity."""
        series_list, legend, cats, _ = build_bar_series(
            df, x, y, hue=hue, stack=stack, orient="v",
            bar_width=bar_width, bar_gap=bar_gap,
            border_radius=border_radius, labels=labels,
            label_formatter=label_formatter, label_font_size=label_font_size,
            label_color=label_color, gradient=gradient,
            gradient_colors=gradient_colors, agg=agg,
            emphasis=emphasis, **series_kw,
        )
        self._cell.chart_type = "bar"
        self._cell.x_axis_config = {"type": "category", "data": cats}
        self._cell.y_axis_config = {"type": "value"}
        for entry in series_list:
            entry.pop("yAxisIndex", None)
            _merge_style_params(
                entry, item_style=item_style, label_style=label_style,
                color=color, blur=blur, select=select,
                selected_mode=selected_mode, animation=animation,
                tooltip=tooltip,
            )
            if bar_min_width is not None:
                entry["barMinWidth"] = bar_min_width
            if bar_category_gap is not None:
                entry["barCategoryGap"] = bar_category_gap
            self._cell.series_configs.append(entry)
        if hue:
            self._cell.legend_names.extend(legend)
        return self._matrix

    def plot(
        self, df: pd.DataFrame, x: str, y: str, *,
        hue: Optional[str] = None, smooth: bool = False,
        area: bool = False, area_opacity: float = 0.15,
        connect_nulls: bool = False, line_width: int = 2,
        symbol_size: int = 6, symbol: str = "circle",
        labels: bool = False, label_position: str = "top",
        label_prefix: str = "", label_suffix: str = "",
        agg: str = "mean",
        emphasis: Optional[LineEmphasis] = None,
        line_style: Optional[LineStyle] = None,
        area_style: Optional[AreaStyle] = None,
        label_style: Optional[LabelStyle] = None,
        end_label: Optional[EndLabelStyle] = None,
        show_symbol: Optional[bool] = None,
        color: Optional[str] = None,
        item_style: Optional[ItemStyle] = None,
        blur: Optional[Blur] = None,
        select: Optional[Select] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        animation: Optional[AnimationConfig] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "MatrixFigure":
        """Add a line chart to this cell with full Figure.plot() parity."""
        series_list, legend, cats = build_line_series(
            df, x, y, hue=hue, smooth=smooth,
            area=area, area_opacity=area_opacity,
            connect_nulls=connect_nulls, line_width=line_width,
            symbol_size=symbol_size, symbol=symbol,
            labels=labels, label_position=label_position,
            label_prefix=label_prefix, label_suffix=label_suffix,
            agg=agg, emphasis=emphasis, **series_kw,
        )
        self._cell.chart_type = "line"
        self._cell.x_axis_config = {"type": "category", "data": cats}
        self._cell.y_axis_config = {"type": "value"}
        for entry in series_list:
            entry.pop("yAxisIndex", None)
            _merge_style_params(
                entry, line_style=line_style, area_style=area_style,
                label_style=label_style, end_label=end_label,
                item_style=item_style, show_symbol=show_symbol,
                color=color, blur=blur, select=select,
                selected_mode=selected_mode, animation=animation,
                tooltip=tooltip,
            )
            self._cell.series_configs.append(entry)
        if hue:
            self._cell.legend_names.extend(legend)
        return self._matrix

    def scatter(
        self, df: pd.DataFrame, x: str, y: str, *,
        color: Optional[str] = None,
        size: Optional[str] = None,
        size_range: Tuple[int, int] = (5, 30),
        symbol: str = "circle", opacity: float = 0.7,
        labels: bool = False,
        symbol_rotate: Optional[int] = None,
        emphasis: Optional[ScatterEmphasis] = None,
        item_style: Optional[ItemStyle] = None,
        label_style: Optional[LabelStyle] = None,
        show_symbol: Optional[bool] = None,
        blur: Optional[Blur] = None,
        select: Optional[Select] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        animation: Optional[AnimationConfig] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "MatrixFigure":
        """Add a scatter chart to this cell with full Figure.scatter() parity.

        Note: ``color`` is a **column name** for grouping (matching Figure API),
        not a literal colour string. Use ``item_style=ec.ItemStyle(color=...)``
        for literal colours.
        """
        s_list, legend = build_scatter_series(
            df, x, y, color=color, size=size, size_range=size_range,
            symbol=symbol, opacity=opacity, labels=labels,
            symbol_rotate=symbol_rotate, emphasis=emphasis,
            **series_kw,
        )
        self._cell.chart_type = "scatter"
        self._cell.x_axis_config = {"type": "value"}
        self._cell.y_axis_config = {"type": "value"}
        for entry in s_list:
            _merge_style_params(
                entry, item_style=item_style, label_style=label_style,
                show_symbol=show_symbol, blur=blur, select=select,
                selected_mode=selected_mode, animation=animation,
                tooltip=tooltip,
            )
            self._cell.series_configs.append(entry)
        if color:
            self._cell.legend_names.extend(legend)
        return self._matrix

    def pie(
        self, df: pd.DataFrame, names: str, values: str, *,
        radius: Optional[Union[str, List[str]]] = None,
        center: Optional[List[str]] = None,
        border_radius: int = 0, start_angle: int = 45,
        label_inside: bool = False, labels: bool = False,
        label_formatter: str = "{b}: {c} ({d}%)",
        label_font_size: Optional[int] = None,
        rose_type: Optional[Literal["radius", "area"]] = None,
        min_angle: Optional[int] = None,
        min_show_label_angle: Optional[int] = None,
        selected_offset: Optional[int] = None,
        clockwise: Optional[bool] = None,
        avoid_label_overlap: Optional[bool] = None,
        animation_type: Optional[Literal["expansion", "scale"]] = None,
        emphasis: Optional[PieEmphasis] = None,
        item_style: Optional[ItemStyle] = None,
        label_style: Optional[LabelStyle] = None,
        agg: str = "sum",
        blur: Optional[Blur] = None,
        select: Optional[Select] = None,
        selected_mode: Optional[Union[bool, str]] = None,
        animation: Optional[AnimationConfig] = None,
        tooltip: Optional[TooltipStyle] = None,
        **series_kw: Any,
    ) -> "MatrixFigure":
        """Add a pie chart to this cell with full Figure.pie() parity."""
        entry, legend = build_pie_series(
            df, names, values,
            radius=radius, border_radius=border_radius,
            start_angle=start_angle, label_inside=label_inside,
            label_outside=labels, label_formatter=label_formatter,
            label_font_size=label_font_size, rose_type=rose_type,
            min_angle=min_angle, min_show_label_angle=min_show_label_angle,
            selected_offset=selected_offset, clockwise=clockwise,
            avoid_label_overlap=avoid_label_overlap,
            animation_type=animation_type, emphasis=emphasis,
            item_style=item_style, agg=agg, **series_kw,
        )
        # Apply remaining style params
        if label_style is not None:
            entry.setdefault("label", {}).update(label_style.to_dict())
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

        self._cell.chart_type = "pie"
        self._cell.show_axis = False
        self._cell.series_configs.append(entry)
        self._cell.legend_names.extend(legend)
        return self._matrix

    def heatmap(
        self, df: pd.DataFrame, x: str, y: str, value: str, *,
        label_show: bool = True,
        emphasis: Optional[Emphasis] = None,
        item_style: Optional[ItemStyle] = None,
        label_style: Optional[LabelStyle] = None,
        tooltip: Optional[TooltipStyle] = None,
        **kw: Any,
    ) -> "MatrixFigure":
        """Add a heatmap to this cell.

        Parameters
        ----------
        df : DataFrame
            Source data.
        x, y, value : str
            Category and value columns.
        label_show : bool
            Show value labels in cells.
        emphasis : Emphasis, optional
            Hover emphasis style.
        item_style : ItemStyle, optional
            Visual style.
        label_style : LabelStyle, optional
            Label text style.
        tooltip : TooltipStyle, optional
            Per-series tooltip.
        """
        df = _validate_df(df, "matrix.cell.heatmap")
        _validate_columns(df, [x, y, value], "matrix.cell.heatmap")
        dff = df.copy()
        dff[value] = _coerce_numeric(dff, value, "matrix.cell.heatmap")
        x_cats = list(dict.fromkeys(dff[x].astype(str).tolist()))
        y_cats = list(dict.fromkeys(dff[y].astype(str).tolist()))
        x_lookup = {c: i for i, c in enumerate(x_cats)}
        y_lookup = {c: i for i, c in enumerate(y_cats)}
        data = []
        for _, r in dff.iterrows():
            data.append([x_lookup[str(r[x])], y_lookup[str(r[y])], round(float(r[value]), 4)])

        self._cell.chart_type = "heatmap"
        self._cell.x_axis_config = {"type": "category", "data": x_cats}
        self._cell.y_axis_config = {"type": "category", "data": y_cats}
        entry: dict = {
            "type": "heatmap", "data": data,
            "label": {"show": label_show},
        }
        _merge_style_params(entry, item_style=item_style,
                           label_style=label_style, tooltip=tooltip)
        if emphasis is not None:
            entry["emphasis"] = emphasis.to_dict()
        entry.update(kw)
        self._cell.series_configs.append(entry)
        return self._matrix

    def text(self, content: str, **kw: Any) -> "MatrixFigure":
        """Add a text annotation to this cell."""
        self._cell.chart_type = "text"
        self._cell.show_axis = False
        self._cell.series_configs.append({
            "type": "scatter", "data": [],
            "markPoint": {
                "data": [{
                    "coord": [0, 0],
                    "value": content,
                    "label": {
                        "show": True, "formatter": content,
                        "fontSize": kw.pop("font_size", 14),
                        "color": kw.pop("color", "#333"),
                    },
                    "symbolSize": 0,
                }],
            },
            **kw,
        })
        self._cell.x_axis_config = {"type": "value", "show": False, "min": -1, "max": 1}
        self._cell.y_axis_config = {"type": "value", "show": False, "min": -1, "max": 1}
        return self._matrix


# ── MatrixFigure ─────────────────────────────────────────────────────────


class MatrixFigure:
    """Matrix coordinate system layout — table-like grid of charts.

    Each cell in the matrix can contain an independent chart (bar, line,
    scatter, pie, heatmap, etc.).

    Parameters
    ----------
    rows : list[str]
        Row labels (y-axis of matrix).
    cols : list[str]
        Column labels (x-axis of matrix).
    height : str
        CSS height.
    width : str or None
        CSS width.
    renderer : str
        ``"canvas"`` or ``"svg"``.
    theme : str or None
        ECharts built-in theme.
    style : StylePreset or None
        Style preset.
    key : str or None
        Widget key (Streamlit only).
    cell_gap : int
        Gap between cells in pixels.
    """

    def __init__(
        self,
        rows: List[str],
        cols: List[str],
        height: str = "600px",
        width: Optional[str] = None,
        renderer: str = "svg",
        theme: Optional[str] = None,
        style: Optional[StylePreset] = None,
        key: Optional[str] = None,
        cell_gap: int = 5,
    ) -> None:
        self._rows = list(rows)
        self._cols = list(cols)
        self._height = height
        self._width = width
        self._renderer = renderer
        self._theme = theme
        self._key = key
        self._style = style or StylePreset.CLINICAL
        self._cell_gap = cell_gap

        self._cells: Dict[Tuple[int, int], _CellConfig] = {}
        self._title_cfg: Optional[dict] = None
        self._palette: Optional[Sequence[str]] = (
            list(self._style.palette) if self._style.palette else None
        )

    def __repr__(self) -> str:
        return (
            f"<MatrixFigure rows={len(self._rows)}, cols={len(self._cols)}, "
            f"cells={len(self._cells)}, height={self._height!r}>"
        )

    # ── Cell accessor ────────────────────────────────────────────────────

    def cell(self, row: int, col: int) -> _CellBuilder:
        """Get a builder for a specific cell.

        Parameters
        ----------
        row : int
            Row index (0-based).
        col : int
            Column index (0-based).

        Returns
        -------
        _CellBuilder
            Fluent builder for the cell.
        """
        if row < 0 or row >= len(self._rows):
            raise IndexError(f"row index {row} out of range [0, {len(self._rows) - 1}]")
        if col < 0 or col >= len(self._cols):
            raise IndexError(f"col index {col} out of range [0, {len(self._cols) - 1}]")
        return _CellBuilder(self, row, col)

    # ── Auto-populating chart methods ────────────────────────────────────

    def bar(
        self, df: pd.DataFrame, x: str, y: str, *,
        group_row: str, group_col: Optional[str] = None, **kw: Any,
    ) -> "MatrixFigure":
        """Auto-populate cells with bar charts grouped by row/col.

        Parameters
        ----------
        df : DataFrame
            Source data.
        x, y : str
            Columns for x-axis categories and y-axis values.
        group_row : str
            Column whose unique values map to matrix rows.
        group_col : str, optional
            Column whose unique values map to matrix columns.
            If None, every cell in each row gets the same chart.
        """
        self._auto_fill(df, x, y, "bar", group_row, group_col, **kw)
        return self

    def plot(
        self, df: pd.DataFrame, x: str, y: str, *,
        group_row: str, group_col: Optional[str] = None, **kw: Any,
    ) -> "MatrixFigure":
        """Auto-populate cells with line charts grouped by row/col."""
        self._auto_fill(df, x, y, "line", group_row, group_col, **kw)
        return self

    def scatter(
        self, df: pd.DataFrame, x: str, y: str, *,
        group_row: str, group_col: Optional[str] = None, **kw: Any,
    ) -> "MatrixFigure":
        """Auto-populate cells with scatter charts grouped by row/col."""
        self._auto_fill(df, x, y, "scatter", group_row, group_col, **kw)
        return self

    def pie(
        self, df: pd.DataFrame, names: str, values: str, *,
        group_row: str, group_col: Optional[str] = None, **kw: Any,
    ) -> "MatrixFigure":
        """Auto-populate cells with pie charts grouped by row/col."""
        df = _validate_df(df, "matrix.pie")
        cols_needed = [names, values, group_row]
        if group_col:
            cols_needed.append(group_col)
        _validate_columns(df, cols_needed, "matrix.pie")

        for ri, row_label in enumerate(self._rows):
            for ci, col_label in enumerate(self._cols):
                mask = df[group_row].astype(str) == row_label
                if group_col:
                    mask &= df[group_col].astype(str) == col_label
                subset = df[mask]
                if subset.empty:
                    continue
                self.cell(ri, ci).pie(subset, names, values, **kw)
        return self

    def sparkline(
        self, df: pd.DataFrame, x: str, y: str, *,
        group_row: str, group_col: Optional[str] = None,
        smooth: bool = True, **kw: Any,
    ) -> "MatrixFigure":
        """Fill cells with tiny line charts (no axes, minimal chrome).

        Parameters
        ----------
        df : DataFrame
            Source data.
        x, y : str
            Columns for x-axis and y-axis.
        group_row : str
            Column whose unique values map to matrix rows.
        group_col : str, optional
            Column whose unique values map to matrix columns.
        smooth : bool
            Whether to smooth the sparklines.
        """
        df = _validate_df(df, "matrix.sparkline")
        cols_needed = [x, y, group_row]
        if group_col:
            cols_needed.append(group_col)
        _validate_columns(df, cols_needed, "matrix.sparkline")

        for ri, row_label in enumerate(self._rows):
            for ci, col_label in enumerate(self._cols):
                mask = df[group_row].astype(str) == row_label
                if group_col:
                    mask &= df[group_col].astype(str) == col_label
                subset = df[mask]
                if subset.empty:
                    continue

                dff = subset.copy()
                dff[y] = _coerce_numeric(dff, y, "matrix.sparkline")
                cats = list(dict.fromkeys(dff[x].astype(str).tolist()))
                vals = dff.groupby(x, sort=False)[y].sum().reindex(cats).fillna(0).tolist()

                cell = self._cells.setdefault((ri, ci), _CellConfig())
                cell.chart_type = "line"
                cell.show_axis = False
                cell.x_axis_config = {"type": "category", "data": cats, "show": False}
                cell.y_axis_config = {"type": "value", "show": False}
                cell.series_configs.append({
                    "type": "line", "data": [round(float(v), 4) for v in vals],
                    "smooth": smooth, "showSymbol": False,
                    "areaStyle": {"opacity": 0.15},
                    **kw,
                })
        return self

    def _auto_fill(
        self, df: pd.DataFrame, x: str, y: str,
        chart_type: str, group_row: str, group_col: Optional[str],
        **kw: Any,
    ) -> None:
        """Shared logic for auto-populating cells."""
        df = _validate_df(df, f"matrix.{chart_type}")
        cols_needed = [x, y, group_row]
        if group_col:
            cols_needed.append(group_col)
        _validate_columns(df, cols_needed, f"matrix.{chart_type}")

        for ri, row_label in enumerate(self._rows):
            for ci, col_label in enumerate(self._cols):
                mask = df[group_row].astype(str) == row_label
                if group_col:
                    mask &= df[group_col].astype(str) == col_label
                subset = df[mask]
                if subset.empty:
                    continue

                if chart_type == "scatter":
                    self.cell(ri, ci).scatter(subset, x, y, **kw)
                elif chart_type == "line":
                    self.cell(ri, ci).plot(subset, x, y, **kw)
                else:
                    self.cell(ri, ci).bar(subset, x, y, **kw)

    # ── Pairplot ─────────────────────────────────────────────────────────

    @classmethod
    def pairplot(
        cls, df: pd.DataFrame, columns: Sequence[str], *,
        hue: Optional[str] = None,
        diag: Literal["hist", "label"] = "hist",
        upper: Literal["scatter"] = "scatter",
        lower: Literal["scatter"] = "scatter",
        height: str = "600px",
        **kw: Any,
    ) -> "MatrixFigure":
        """Create a seaborn-style pairplot / scatterplot matrix.

        Parameters
        ----------
        df : DataFrame
            Source data.
        columns : Sequence[str]
            Numeric columns to pair.
        hue : str, optional
            Column to group / colour code scatter points.
        diag : {"hist", "label"}
            What to show on the diagonal.
        upper, lower : {"scatter"}
            What to show off-diagonal.
        height : str
            CSS height.

        Returns
        -------
        MatrixFigure
            Configured matrix.
        """
        df = _validate_df(df, "pairplot")
        cols = list(columns)
        _validate_columns(df, cols, "pairplot")

        mat = cls(rows=cols, cols=cols, height=height, **kw)

        for ri, row_col in enumerate(cols):
            for ci, col_col in enumerate(cols):
                if ri == ci:
                    # Diagonal
                    if diag == "hist":
                        vals = _coerce_numeric(df, row_col, "pairplot")
                        vals = vals.dropna()
                        counts, edges = np.histogram(vals, bins=15)
                        labels = [f"{edges[i]:.1f}" for i in range(len(counts))]
                        cell = mat._cells.setdefault((ri, ci), _CellConfig())
                        cell.chart_type = "bar"
                        cell.x_axis_config = {
                            "type": "category", "data": labels,
                            "axisLabel": {"show": False},
                            "axisTick": {"show": False},
                        }
                        cell.y_axis_config = {"type": "value", "axisLabel": {"show": False}}
                        cell.series_configs.append({
                            "type": "bar",
                            "data": [int(c) for c in counts],
                            "barWidth": "90%",
                        })
                    else:
                        mat.cell(ri, ci).text(row_col, font_size=12)
                else:
                    # Off-diagonal scatter
                    scatter_data: list = []
                    if hue:
                        groups = df.groupby(hue)
                        for gname, gdf in groups:
                            sdata = [
                                [round(float(r[col_col]), 4), round(float(r[row_col]), 4)]
                                for _, r in gdf.iterrows()
                                if pd.notna(r[col_col]) and pd.notna(r[row_col])
                            ]
                            cell = mat._cells.setdefault((ri, ci), _CellConfig())
                            cell.chart_type = "scatter"
                            cell.x_axis_config = {"type": "value"}
                            cell.y_axis_config = {"type": "value"}
                            cell.series_configs.append({
                                "type": "scatter", "data": sdata,
                                "name": str(gname),
                                "symbolSize": 4,
                            })
                    else:
                        scatter_data = [
                            [round(float(r[col_col]), 4), round(float(r[row_col]), 4)]
                            for _, r in df.iterrows()
                            if pd.notna(r[col_col]) and pd.notna(r[row_col])
                        ]
                        mat.cell(ri, ci).scatter(
                            pd.DataFrame({col_col: [d[0] for d in scatter_data],
                                          row_col: [d[1] for d in scatter_data]}),
                            col_col, row_col, symbolSize=4,
                        )
        return mat

    # ── Chrome methods ───────────────────────────────────────────────────

    def title(self, text: str, **kw: Any) -> "MatrixFigure":
        """Set the chart title."""
        self._title_cfg = {"text": text, **kw}
        return self

    def palette(self, colors: Sequence[str]) -> "MatrixFigure":
        """Set the color palette."""
        self._palette = list(colors)
        return self

    # ── Option assembly ──────────────────────────────────────────────────

    def to_option(self) -> dict:
        """Assemble and return the raw ECharts option dict.

        Returns
        -------
        dict
            Complete ECharts option configuration.
        """
        n_rows = len(self._rows)
        n_cols = len(self._cols)
        gap = self._cell_gap

        # Collect legend names first — needed for layout math
        all_legend: list = []
        seen: set = set()
        for cell in self._cells.values():
            for name in cell.legend_names:
                if name not in seen:
                    all_legend.append(name)
                    seen.add(name)
        has_legend = bool(all_legend)

        # Reserve vertical space for title and legend
        top_margin = 8                           # % for title area
        bottom_margin = 2 + (5 if has_legend else 0)  # legend
        usable_h = 100 - top_margin - bottom_margin

        # Horizontal space
        left_margin = 2
        right_margin = 2
        usable_w = 100 - left_margin - right_margin

        # Cell sizes with inter-cell gap
        gap_pct = gap * 0.6
        cell_w = round((usable_w - gap_pct * (n_cols - 1)) / n_cols, 2)
        cell_h = round((usable_h - gap_pct * (n_rows - 1)) / n_rows, 2)

        grids: list = []
        x_axes: list = []
        y_axes: list = []
        all_series: list = []

        grid_idx = 0
        for ri in range(n_rows):
            for ci in range(n_cols):
                cell = self._cells.get((ri, ci))

                left = round(left_margin + ci * (cell_w + gap_pct), 2)
                top = round(top_margin + ri * (cell_h + gap_pct), 2)
                w = cell_w
                h = cell_h

                grid = {
                    "left": f"{left}%", "top": f"{top}%",
                    "width": f"{w}%", "height": f"{h}%",
                    "containLabel": True,
                }
                grids.append(grid)

                # X axis
                x_cfg: dict = {"gridIndex": grid_idx, "type": "category", "data": []}
                if cell and cell.x_axis_config:
                    x_cfg.update(cell.x_axis_config)
                    x_cfg["gridIndex"] = grid_idx
                if cell and not cell.show_axis:
                    x_cfg["show"] = False
                # Row/column name labels: bottom row gets x-axis name (col label),
                # first column gets y-axis name (row label) — avoids mid-grid overlap
                is_bottom_row = (ri == n_rows - 1)
                if is_bottom_row:
                    x_cfg.setdefault("name", self._cols[ci])
                    x_cfg.setdefault("nameLocation", "center")
                    x_cfg.setdefault("nameGap", 25)
                x_axes.append(x_cfg)

                # Y axis
                y_cfg: dict = {"gridIndex": grid_idx, "type": "value"}
                if cell and cell.y_axis_config:
                    y_cfg.update(cell.y_axis_config)
                    y_cfg["gridIndex"] = grid_idx
                if cell and not cell.show_axis:
                    y_cfg["show"] = False
                is_first_col = (ci == 0)
                if is_first_col:
                    y_cfg.setdefault("name", self._rows[ri])
                    y_cfg.setdefault("nameLocation", "center")
                    y_cfg.setdefault("nameGap", 35)
                y_axes.append(y_cfg)

                # Series
                if cell:
                    for sc in cell.series_configs:
                        s = copy.deepcopy(sc)
                        if s["type"] == "pie":
                            # Pie charts position inside the grid
                            cx = round(left + w / 2, 2)
                            cy = round(top + h / 2, 2)
                            if "center" not in s:
                                s["center"] = [f"{cx}%", f"{cy}%"]
                            if "radius" not in s:
                                s["radius"] = f"{min(w, h) * 0.30}%"
                        else:
                            s["xAxisIndex"] = grid_idx
                            s["yAxisIndex"] = grid_idx
                        all_series.append(s)

                grid_idx += 1

        option: dict = {}
        if self._title_cfg:
            option["title"] = copy.deepcopy(self._title_cfg)
        option["tooltip"] = {"trigger": "item", "confine": True}
        if has_legend:
            option["legend"] = {
                "data": all_legend,
                "bottom": 0,
                "type": "scroll",
            }
        if self._palette:
            option["color"] = list(self._palette)
        option["grid"] = grids
        option["xAxis"] = x_axes
        option["yAxis"] = y_axes
        option["series"] = all_series
        return option

    # ── Rendering ────────────────────────────────────────────────────────

    def show(self, **render_kw: Any) -> None:
        """Render the matrix chart using the currently configured engine."""
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
        """Export the matrix chart to a standalone HTML file.

        Parameters
        ----------
        filepath : str
            Output file path.

        Returns
        -------
        str
            The filepath written to.
        """
        from echartsy._config import get_adaptive
        from echartsy.renderers._html_template import build_html

        parent = os.path.dirname(os.path.abspath(filepath))
        if not os.path.isdir(parent):
            raise FileNotFoundError(
                f"to_html(): directory '{parent}' does not exist."
            )

        option = self.to_option()
        html = build_html(
            option, height=self._height,
            width=self._width or "100%",
            theme=self._theme, renderer=self._renderer,
            adaptive=get_adaptive(),
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        abs_path = os.path.abspath(filepath)
        print(f"Chart saved to {abs_path}")
        return filepath

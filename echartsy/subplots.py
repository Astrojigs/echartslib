"""
Subplot layout — matplotlib-style ``fig, ax = ec.subplots(nrows, ncols)``.

Each *ax* element is a full :class:`~echartsy.figure.Figure` instance,
so every chart method (``.bar()``, ``.plot()``, ``.scatter()``, …) works
exactly the same as on a standalone Figure.
"""
from __future__ import annotations

import copy
import os
from typing import (
    Any,
    Literal,
    List,
    Optional,
    Sequence,
    Tuple,
)

from echartsy.figure import Figure
from echartsy.renderers import render
from echartsy.styles import StylePreset


# ─── Axes grid ────────────────────────────────────────────────────────────────

class AxesGrid:
    """Lightweight 2-D container whose ``__getitem__`` returns :class:`Figure`.

    Jupyter's *jedi* engine reads the return-type annotation on
    ``__getitem__`` to enable **Shift-Tab** docstring popups and
    tab-completion on ``ax[i, j].bar(...)``.
    """

    def __init__(self, grid: List[List["Figure"]]) -> None:
        self._grid = grid
        self._nrows = len(grid)
        self._ncols = len(grid[0]) if grid else 0

    # ── 2-D indexing (ax[r, c]) ──────────────────────────────────────
    def __getitem__(self, key) -> "Figure":
        if isinstance(key, tuple):
            r, c = key
            return self._grid[r][c]
        return self._grid[key]  # type: ignore[return-value]

    def __len__(self) -> int:
        return self._nrows

    def __repr__(self) -> str:
        return f"AxesGrid({self._nrows}x{self._ncols})"


class AxesRow:
    """1-D row of :class:`Figure` cells — returned when ``nrows == 1`` or ``ncols == 1``."""

    def __init__(self, cells: List["Figure"]) -> None:
        self._cells = cells

    def __getitem__(self, key: int) -> "Figure":
        return self._cells[key]

    def __len__(self) -> int:
        return len(self._cells)

    def __repr__(self) -> str:
        return f"AxesRow({len(self._cells)})"


# ─── Factory ──────────────────────────────────────────────────────────────────

def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    height: str = "600px",
    width: Optional[str] = None,
    renderer: Literal["canvas", "svg"] = "svg",
    theme: Optional[str] = None,
    style: Optional[StylePreset] = None,
    key: Optional[str] = None,
    cell_gap: int = 5,
) -> Tuple[SubplotFigure, Any]:
    """Create a grid of subplots — matplotlib style.

    Parameters
    ----------
    nrows, ncols : int
        Grid dimensions (default 1×1).
    height : str
        CSS height for the overall chart container.
    width : str or None
        CSS width; ``None`` = container width.
    renderer : ``{"canvas", "svg"}``
        ECharts renderer.
    theme : str or None
        ECharts built-in theme name.
    style : StylePreset or None
        Apply a style bundle for palettes, fonts, etc.
    key : str or None
        Streamlit widget key.
    cell_gap : int
        Pixel-like gap between cells (translated to percentage internally).

    Returns
    -------
    (SubplotFigure, ax)
        *ax* shape follows matplotlib conventions:

        * ``(1, 1)`` → scalar :class:`Figure`
        * ``(1, N)`` or ``(N, 1)`` → 1-D :class:`AxesRow` of Figures
        * ``(M, N)`` → 2-D :class:`AxesGrid` of shape ``(M, N)``
    """
    if nrows < 1 or ncols < 1:
        raise ValueError(
            f"subplots() requires nrows >= 1 and ncols >= 1, got ({nrows}, {ncols})"
        )

    fig = SubplotFigure(
        nrows=nrows, ncols=ncols,
        height=height, width=width,
        renderer=renderer, theme=theme,
        style=style, key=key,
        cell_gap=cell_gap,
    )

    # Build ax — shape follows matplotlib conventions
    cells = fig._cells
    if nrows == 1 and ncols == 1:
        ax: Any = cells[0][0]                           # scalar Figure
    elif nrows == 1:
        ax = AxesRow(cells[0])                          # 1-D row
    elif ncols == 1:
        ax = AxesRow([cells[r][0] for r in range(nrows)])  # 1-D col
    else:
        ax = AxesGrid(cells)                            # 2-D grid

    return fig, ax


# ─── Container ────────────────────────────────────────────────────────────────

class SubplotFigure:
    """Container that merges multiple :class:`Figure` cells into one ECharts config.

    Typically created via :func:`subplots`, not instantiated directly.
    """

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        *,
        height: str = "600px",
        width: Optional[str] = None,
        renderer: Literal["canvas", "svg"] = "svg",
        theme: Optional[str] = None,
        style: Optional[StylePreset] = None,
        key: Optional[str] = None,
        cell_gap: int = 5,
    ) -> None:
        self._nrows = nrows
        self._ncols = ncols
        self._height = height
        self._width = width
        self._renderer = renderer
        self._theme = theme
        self._key = key
        self._style = style or StylePreset.DEFAULT
        self._cell_gap = cell_gap

        # 2-D grid of Figure instances
        self._cells: List[List[Figure]] = [
            [
                Figure(
                    style=self._style,
                    theme=theme,
                    _subplot_mode=True,
                )
                for _ in range(ncols)
            ]
            for _ in range(nrows)
        ]

        # Global chrome
        self._suptitle_cfg: Optional[dict] = None
        self._global_palette: Optional[List[str]] = (
            list(self._style.palette) if self._style.palette else None
        )
        self._global_legend_cfg: Optional[dict] = None
        self._margins: dict = {}  # user overrides for top/bottom/left/right

    def __repr__(self) -> str:
        return f"SubplotFigure({self._nrows}×{self._ncols})"

    # ── Chrome ────────────────────────────────────────────────────────────

    def suptitle(
        self,
        text: str,
        subtitle: Optional[str] = None,
        **kw: Any,
    ) -> SubplotFigure:
        """Set the global title above the subplot grid.

        Parameters
        ----------
        text : str
            Main title text.
        subtitle : str or None
            Subtitle text.
        **kw
            Extra ECharts title properties (``left``, ``textStyle``, …).
        """
        cfg: dict = {"text": text, "left": "center", **kw}
        if subtitle:
            cfg["subtext"] = subtitle
        self._suptitle_cfg = cfg
        return self

    def palette(self, colors: Sequence[str]) -> SubplotFigure:
        """Set a global color palette.

        Cells that have *not* called ``ax.palette()`` inherit this palette.
        Cells with their own palette are unaffected.
        """
        self._global_palette = list(colors)
        return self

    def legend(
        self,
        position: str = "bottom",
        **kw: Any,
    ) -> SubplotFigure:
        """Configure the global (merged) legend.

        Parameters
        ----------
        position : str
            Convenience shortcut — ``"bottom"``, ``"top"``, ``"left"``,
            ``"right"``.  Overridden by explicit kwargs.
        **kw
            Extra ECharts legend properties.
        """
        pos_map = {
            "bottom": {"bottom": 0, "type": "scroll"},
            "top":    {"top": 0},
            "left":   {"orient": "vertical", "left": 0},
            "right":  {"orient": "vertical", "right": 0},
        }
        cfg = {**pos_map.get(position, {}), **kw}
        self._global_legend_cfg = cfg
        return self

    def margins(
        self,
        top: Optional[int] = None,
        bottom: Optional[int] = None,
        left: Optional[int] = None,
        right: Optional[int] = None,
    ) -> SubplotFigure:
        """Set outer margins (in percentage) around the entire subplot grid.

        Parameters
        ----------
        top, bottom, left, right : int or None
            Margin as a percentage of the container. Only provided values
            are changed; ``None`` keeps the auto-computed default.
        """
        if top is not None:
            self._margins["top"] = top
        if bottom is not None:
            self._margins["bottom"] = bottom
        if left is not None:
            self._margins["left"] = left
        if right is not None:
            self._margins["right"] = right
        return self

    # ── Option assembly ───────────────────────────────────────────────────

    @staticmethod
    def _parse_px(css_val: Optional[str], fallback: int) -> int:
        """Extract numeric pixel value from a CSS string like '600px'."""
        if css_val and css_val.strip().endswith("px"):
            try:
                return int(css_val.strip()[:-2])
            except ValueError:
                pass
        return fallback

    def to_option(self) -> dict:
        """Merge all cell Figures into a single ECharts option dict."""
        n_rows = self._nrows
        n_cols = self._ncols
        gap = self._cell_gap

        # Container dimensions (px) — used for margin conversion
        container_h = self._parse_px(self._height, 600)
        container_w = self._parse_px(self._width, 1200)

        # ── Check if any cell has legend items (for layout math) ─────
        has_legend = any(
            bool(self._cells[ri][ci]._legend_items)
            for ri in range(n_rows)
            for ci in range(n_cols)
        )

        # ── Layout math (percentage-based grid positions) ─────────────
        top_margin = self._margins.get(
            "top", 8 if self._suptitle_cfg else 2
        )
        bottom_margin = self._margins.get(
            "bottom", 2 + (5 if has_legend else 0)
        )
        left_margin = self._margins.get("left", 2)
        right_margin = self._margins.get("right", 2)

        usable_h = 100 - top_margin - bottom_margin
        usable_w = 100 - left_margin - right_margin

        gap_pct = gap * 0.6
        cell_w = round((usable_w - gap_pct * (n_cols - 1)) / n_cols, 2)
        cell_h = round((usable_h - gap_pct * (n_rows - 1)) / n_rows, 2)

        grids: List[dict] = []
        x_axes: List[dict] = []
        y_axes: List[dict] = []
        all_series: List[dict] = []
        all_legend: List[str] = []
        legend_seen: set = set()
        cell_titles: List[dict] = []
        all_legend_cfgs: List[dict] = []
        all_tooltips: List[dict] = []
        all_toolbox: Optional[dict] = None
        all_datazoom: List[dict] = []
        all_visual_maps: List[dict] = []
        option_extra: dict = {}   # for radar, parallel, calendar components

        grid_idx = 0
        for ri in range(n_rows):
            for ci in range(n_cols):
                cell = self._cells[ri][ci]
                cell_opt = cell.to_option()
                is_empty = cell_opt.get("_empty", False)

                left = round(left_margin + ci * (cell_w + gap_pct), 2)
                top = round(top_margin + ri * (cell_h + gap_pct), 2)

                grid = {
                    "left": f"{left}%",
                    "top": f"{top}%",
                    "width": f"{cell_w}%",
                    "height": f"{cell_h}%",
                    "containLabel": True,
                }

                # ── Per-cell margins ─────────────────────────────────
                # Figure._grid_cfg defaults: left=70, right=70, top=60, bottom=50
                # If user called ax.margins(), the changed values shrink the
                # cell's grid inward.  We convert px values to percentage
                # offsets relative to the container dimensions.
                if not is_empty:
                    gc = cell._grid_cfg
                    _DEFAULTS = {"left": 70, "right": 70, "top": 60, "bottom": 50}
                    for side, default in _DEFAULTS.items():
                        val = gc.get(side)
                        if val is not None and val != default:
                            if isinstance(val, (int, float)):
                                # Convert px → % of container dimension
                                dim = container_h if side in ("top", "bottom") else container_w
                                pct_offset = round(val / dim * 100, 2)
                                if side == "left":
                                    grid["left"] = f"{left + pct_offset}%"
                                    grid["width"] = f"{cell_w - pct_offset}%"
                                elif side == "right":
                                    grid["width"] = f"{cell_w - pct_offset}%"
                                elif side == "top":
                                    grid["top"] = f"{top + pct_offset}%"
                                    grid["height"] = f"{cell_h - pct_offset}%"
                                elif side == "bottom":
                                    grid["height"] = f"{cell_h - pct_offset}%"
                            else:
                                # String value (e.g. "10%") — pass through
                                grid[side] = val

                grids.append(grid)

                # ── Per-cell title ────────────────────────────────────
                if not is_empty and cell._title_cfg:
                    t = copy.deepcopy(cell._title_cfg)
                    # Position inside this cell's grid area
                    cell_center_x = round(left + cell_w / 2, 2)
                    user_left = t.get("left", "center")
                    if user_left == "center":
                        t["left"] = f"{cell_center_x}%"
                        t["textAlign"] = "center"
                    elif user_left == "left":
                        t["left"] = f"{left}%"
                    elif user_left == "right":
                        t["left"] = f"{round(left + cell_w, 2)}%"
                        t["textAlign"] = "right"
                    elif isinstance(user_left, (int, float)):
                        # User specified px offset — convert relative to cell
                        t["left"] = round(left + user_left / container_w * 100, 2)
                        t["left"] = f"{t['left']}%"
                    t.setdefault("top", f"{top}%")
                    t.setdefault("textStyle", {})
                    t["textStyle"].setdefault("fontSize", 13)
                    cell_titles.append(t)

                # ── Per-cell tooltip ──────────────────────────────────
                if not is_empty:
                    cell_tt = cell_opt.get("tooltip")
                    if cell_tt:
                        all_tooltips.append(copy.deepcopy(cell_tt))

                # ── Per-cell toolbox (use first non-empty one) ────────
                if not is_empty and all_toolbox is None:
                    cell_tb = cell_opt.get("toolbox")
                    if cell_tb:
                        all_toolbox = copy.deepcopy(cell_tb)

                # Build axes for this cell
                mode = (cell._chart_mode or "cartesian") if not is_empty else None
                # Modes that use cartesian axes (xAxis + yAxis + grid)
                _CARTESIAN = {"cartesian", "heatmap"}
                uses_axes = mode in _CARTESIAN

                x_cfg: dict = {
                    "gridIndex": grid_idx,
                    "type": "category",
                    "data": [],
                }
                y_cfg: dict = {
                    "gridIndex": grid_idx,
                    "type": "value",
                }
                extra_y_axes: List[dict] = []  # for dual y-axis

                if is_empty or not uses_axes:
                    x_cfg["show"] = False
                    y_cfg["show"] = False
                else:
                    # Merge cell's x-axis config
                    opt_x = cell_opt.get("xAxis")
                    if opt_x:
                        src = opt_x[0] if isinstance(opt_x, list) else opt_x
                        for k, v in src.items():
                            if k not in ("gridIndex",):
                                x_cfg[k] = copy.deepcopy(v)

                    # Merge cell's y-axis config (support dual y-axis)
                    opt_y = cell_opt.get("yAxis")
                    if opt_y:
                        y_list = opt_y if isinstance(opt_y, list) else [opt_y]
                        # Primary y-axis
                        for k, v in y_list[0].items():
                            if k not in ("gridIndex",):
                                y_cfg[k] = copy.deepcopy(v)
                        # Additional y-axes (dual axis)
                        for extra in y_list[1:]:
                            ey = {"gridIndex": grid_idx, "type": "value"}
                            for k, v in extra.items():
                                if k not in ("gridIndex",):
                                    ey[k] = copy.deepcopy(v)
                            extra_y_axes.append(ey)

                x_axes.append(x_cfg)
                y_base_idx = len(y_axes)  # base index for this cell's y-axes
                y_axes.append(y_cfg)
                for ey in extra_y_axes:
                    y_axes.append(ey)

                # ── Series ────────────────────────────────────────────
                # Namespace prefix: invisible zero-width spaces unique per
                # cell so that identically-named series in different cells
                # don't interfere when toggling legends.
                ns = "\u200B" * grid_idx  # cell 0 = "", cell 1 = "\u200B", …

                # Non-cartesian types that need center positioning (like pie)
                _CENTERED = {"pie", "sunburst", "radar"}
                _POSITIONED = {"treemap", "funnel", "gauge", "sankey", "graph"}

                if not is_empty:
                    # Per-cell palette: inject colors into series
                    cell_palette = cell._palette
                    cell_series = cell_opt.get("series", [])

                    # Forward radar/parallel/calendar components
                    for comp_key in ("radar", "parallelAxis", "parallel",
                                     "calendar"):
                        comp = cell_opt.get(comp_key)
                        if comp:
                            option_extra.setdefault(comp_key, [])
                            if isinstance(comp, list):
                                option_extra[comp_key].extend(
                                    copy.deepcopy(comp)
                                )
                            else:
                                option_extra[comp_key].append(
                                    copy.deepcopy(comp)
                                )

                    for si, sc in enumerate(cell_series):
                        s = copy.deepcopy(sc)
                        stype = s.get("type", "")
                        # Namespace the series name
                        if ns and "name" in s:
                            s["name"] = ns + s["name"]

                        if stype in _CENTERED:
                            # Position inside the grid area
                            cx = round(left + cell_w / 2, 2)
                            cy = round(top + cell_h / 2, 2)
                            if "center" not in s:
                                s["center"] = [f"{cx}%", f"{cy}%"]
                            if stype == "pie" and "radius" not in s:
                                s["radius"] = f"{min(cell_w, cell_h) * 0.30}%"
                            # Namespace pie/sunburst data item names
                            if ns and stype == "pie":
                                for item in s.get("data", []):
                                    if isinstance(item, dict) and "name" in item:
                                        item["name"] = ns + item["name"]
                        elif stype in _POSITIONED:
                            # Position within cell bounds
                            s.setdefault("left", f"{left}%")
                            s.setdefault("top", f"{top}%")
                            s.setdefault("width", f"{cell_w}%")
                            s.setdefault("height", f"{cell_h}%")
                        else:
                            # Cartesian — bind to this cell's axes
                            s["xAxisIndex"] = grid_idx
                            # Handle dual y-axis: offset local yAxisIndex
                            local_yi = s.get("yAxisIndex", 0)
                            s["yAxisIndex"] = y_base_idx + local_yi
                        # Apply per-cell palette
                        if cell_palette:
                            if s.get("type") == "pie":
                                # Pie: inject color per data item
                                for di, item in enumerate(s.get("data", [])):
                                    if isinstance(item, dict):
                                        item.setdefault("itemStyle", {})
                                        item["itemStyle"].setdefault(
                                            "color",
                                            cell_palette[di % len(cell_palette)],
                                        )
                            elif "color" not in s.get("itemStyle", {}):
                                s.setdefault("itemStyle", {})
                                s["itemStyle"]["color"] = cell_palette[
                                    si % len(cell_palette)
                                ]
                        # Strip internal metadata
                        s.pop("_meta", None)
                        for key in list(s):
                            if key.startswith("_"):
                                s.pop(key)
                        all_series.append(s)

                    # Collect namespaced legend names
                    for name in cell._legend_items:
                        ns_name = ns + name
                        if ns_name not in legend_seen:
                            all_legend.append(ns_name)
                            legend_seen.add(ns_name)

                # Track per-cell legend config (from ax.legend() calls)
                if cell._legend_items and cell._legend_cfg:
                    lcfg = copy.deepcopy(cell._legend_cfg)
                    # Namespace legend data to match series names
                    lcfg["data"] = [
                        ns + n for n in dict.fromkeys(cell._legend_items)
                    ]
                    # Position within the cell's grid area
                    if "left" not in lcfg and "right" not in lcfg:
                        cell_cx = round(left + cell_w / 2, 2)
                        lcfg["left"] = f"{cell_cx}%"
                        lcfg.setdefault("textAlign", "center")
                    elif "left" in lcfg:
                        v = lcfg["left"]
                        if isinstance(v, (int, float)):
                            lcfg["left"] = round(left + v / container_w * 100, 2)
                            lcfg["left"] = f"{lcfg['left']}%"
                        elif v == "left":
                            lcfg["left"] = f"{left}%"
                        elif v == "right":
                            lcfg["left"] = f"{round(left + cell_w, 2)}%"
                            lcfg["textAlign"] = "right"
                        elif v == "center":
                            cell_cx = round(left + cell_w / 2, 2)
                            lcfg["left"] = f"{cell_cx}%"
                            lcfg.setdefault("textAlign", "center")
                    if "right" in lcfg:
                        v = lcfg["right"]
                        if isinstance(v, (int, float)):
                            cr = round(100 - left - cell_w + v / container_w * 100, 2)
                            lcfg["right"] = f"{cr}%"
                    if "top" not in lcfg and "bottom" not in lcfg:
                        lcfg["top"] = f"{top}%"
                    elif "top" in lcfg:
                        v = lcfg["top"]
                        if isinstance(v, (int, float)):
                            lcfg["top"] = round(top + v / container_h * 100, 2)
                            lcfg["top"] = f"{lcfg['top']}%"
                    if "bottom" in lcfg:
                        v = lcfg["bottom"]
                        if isinstance(v, (int, float)):
                            cb = round(100 - top - cell_h + v / container_h * 100, 2)
                            lcfg["bottom"] = f"{cb}%"
                    all_legend_cfgs.append(lcfg)

                # ── Per-cell datazoom (after axes so y_base_idx is set) ─
                if not is_empty:
                    cell_dz = cell_opt.get("dataZoom")
                    if cell_dz:
                        for dz in cell_dz:
                            d = copy.deepcopy(dz)
                            d["xAxisIndex"] = grid_idx
                            d["yAxisIndex"] = y_base_idx
                            all_datazoom.append(d)

                # ── Per-cell visualMap (after series so offset is correct)
                if not is_empty:
                    cell_vm = cell_opt.get("visualMap")
                    if cell_vm:
                        vm = copy.deepcopy(cell_vm)
                        # series_offset: how many series existed before this cell
                        series_offset = len(all_series) - len(
                            cell_opt.get("series", [])
                        )
                        if "seriesIndex" not in vm:
                            vm["seriesIndex"] = series_offset
                        elif isinstance(vm["seriesIndex"], list):
                            vm["seriesIndex"] = [
                                i + series_offset for i in vm["seriesIndex"]
                            ]
                        elif isinstance(vm["seriesIndex"], int):
                            vm["seriesIndex"] += series_offset
                        all_visual_maps.append(vm)

                grid_idx += 1

        # ── Assemble final option ─────────────────────────────────────
        option: dict = {}

        # ── Titles (suptitle + per-cell titles) ──────────────────────
        titles: List[dict] = []
        if self._suptitle_cfg:
            titles.append(copy.deepcopy(self._suptitle_cfg))
        titles.extend(cell_titles)
        if titles:
            option["title"] = titles if len(titles) > 1 else titles[0]

        # ── Tooltip (merge per-cell configs, fallback to default) ─────
        if all_tooltips:
            # Use the richest tooltip config from any cell
            merged_tt: dict = {"confine": True}
            for tt in all_tooltips:
                merged_tt.update(tt)
            merged_tt["confine"] = True          # always confine in subplots
            option["tooltip"] = merged_tt
        else:
            option["tooltip"] = {"trigger": "item", "confine": True}

        # ── Legend ────────────────────────────────────────────────────
        if all_legend_cfgs:
            # Per-cell legends — each cell that called ax.legend()
            # gets its own positioned legend; remaining names go global
            per_cell_names: set = set()
            for lc in all_legend_cfgs:
                per_cell_names.update(lc.get("data", []))
            remaining = [n for n in all_legend if n not in per_cell_names]

            legends: List[dict] = list(all_legend_cfgs)
            if remaining:
                # Global legend for cells that didn't call ax.legend()
                global_cfg = (
                    copy.deepcopy(self._global_legend_cfg)
                    if self._global_legend_cfg
                    else {"bottom": 0, "type": "scroll"}
                )
                global_cfg["data"] = remaining
                legends.append(global_cfg)
            option["legend"] = legends if len(legends) > 1 else legends[0]
        elif has_legend:
            legend_cfg = (
                copy.deepcopy(self._global_legend_cfg)
                if self._global_legend_cfg
                else {"bottom": 0, "type": "scroll"}
            )
            legend_cfg["data"] = all_legend
            option["legend"] = legend_cfg

        # ── Palette ───────────────────────────────────────────────────
        if self._global_palette:
            option["color"] = list(self._global_palette)

        option["grid"] = grids
        option["xAxis"] = x_axes
        option["yAxis"] = y_axes
        option["series"] = all_series

        # ── Toolbox (first cell that defines one wins) ────────────────
        if all_toolbox:
            option["toolbox"] = all_toolbox

        # ── DataZoom (collected from all cells with axis binding) ─────
        if all_datazoom:
            option["dataZoom"] = all_datazoom

        # ── VisualMap (collected from all cells) ──────────────────────
        if all_visual_maps:
            option["visualMap"] = (
                all_visual_maps[0]
                if len(all_visual_maps) == 1
                else all_visual_maps
            )

        # ── Extra components (radar, parallel, calendar) ────────────
        for comp_key, comp_list in option_extra.items():
            if len(comp_list) == 1:
                option[comp_key] = comp_list[0]
            else:
                option[comp_key] = comp_list

        return option

    # ── Rendering ─────────────────────────────────────────────────────

    def show(self, **render_kw: Any) -> None:
        """Render the subplot grid using the currently configured engine."""
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
        """Export the subplot grid to a standalone HTML file.

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
            option,
            height=self._height,
            width=self._width or "100%",
            theme=self._theme,
            renderer=self._renderer,
            adaptive=get_adaptive(),
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        abs_path = os.path.abspath(filepath)
        print(f"Chart saved to {abs_path}")
        return filepath

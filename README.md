<p align="center">
  <img src="https://echarts.apache.org/en/images/logo.png" alt="Apache ECharts" width="70" />
</p>

<h1 align="center">echartsy</h1>

<p align="center">
  <em>Interactive charts in Python — the matplotlib workflow, the ECharts experience.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/echartsy/"><img src="https://img.shields.io/pypi/v/echartsy?color=%2334d058&label=PyPI" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/echartsy/"><img src="https://img.shields.io/pypi/pyversions/echartsy?color=%2334d058" alt="Python versions" /></a>
  <a href="https://github.com/astrojigs/echartsy/blob/main/LICENSE"><img src="https://img.shields.io/github/license/astrojigs/echartsy?color=blue" alt="License" /></a>
  <a href="https://github.com/astrojigs/echartsy/stargazers"><img src="https://img.shields.io/github/stars/astrojigs/echartsy?style=flat&color=yellow" alt="Stars" /></a>
  <a href="https://github.com/astrojigs/echartsy/issues"><img src="https://img.shields.io/github/issues/astrojigs/echartsy?color=orange" alt="Issues" /></a>
</p>

<p align="center">
  Build publication-quality interactive charts with a familiar<br/>
  <code>fig = figure()</code> &rarr; <code>fig.bar()</code> &rarr; <code>fig.show()</code> workflow.<br/>
  Works everywhere: <b>Jupyter</b> &middot; <b>Streamlit</b> &middot; <b>standalone scripts</b>
</p>

---

## Why echartsy?

| | |
|:---|:---|
| **Familiar API** | If you know `plt.figure()` / `plt.show()`, you already know 90% of the API. No JSON, no JavaScript. |
| **Interactive by default** | Every chart ships with tooltips, legend toggling, zoom, and an export toolbox. Zero config needed. |
| **Three render engines** | Write once, render in Jupyter notebooks, Streamlit apps, or standalone browser windows. |
| **19 chart types** | From bar charts and waterfall charts to sankey diagrams, sunbursts, gauges, and network graphs. |
| **Composable & animated** | Layer pies on bar charts, build dual-axis dashboards, use multi-grid subplots, or animate across time with `TimelineFigure`. |
| **Dark mode** | Adaptive dark/light theming out of the box, including automatic Streamlit theme detection. |

---

## Installation

```bash
pip install echartsy
```

Optional extras:

```bash
pip install echartsy[jupyter]     # Jupyter Notebook / JupyterLab
pip install echartsy[streamlit]   # Streamlit apps
pip install echartsy[scipy]       # KDE density plots
pip install echartsy[all]         # Everything
```

> **Requirements:** Python 3.9+ &middot; pandas &ge; 1.5 &middot; numpy &ge; 1.23

---

## Quick Start

```python
import pandas as pd
import echartsy as ec

ec.config(engine="jupyter")          # or "python" / "streamlit"

df = pd.DataFrame({
    "Fruit": ["Apples", "Bananas", "Cherries", "Dates", "Elderberries"],
    "Sales": [120, 95, 78, 42, 63],
})

fig = ec.figure()
fig.bar(df, x="Fruit", y="Sales", gradient=True, labels=True)
fig.title("Fruit Sales")
fig.show()
```

Three lines from DataFrame to interactive chart.

<p align="center">
  <img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_bar.png" alt="Quick Start — Bar Chart" width="600" />
</p>

---

## Chart Types

echartsy v0.7.0 supports **19 chart types** covering cartesian, standalone, hierarchical, relational, and statistical visualizations.

### Cartesian Charts

| Method | Description | Key options |
|:---|:---|:---|
| `fig.plot()` | Line chart | `smooth`, `area`, `hue`, `line_style`, `area_style`, `end_label`, `blur`, `select`, `animation`, `tooltip` |
| `fig.bar()` | Vertical bar | `hue`, `stack`, `gradient`, `item_style`, `label_style`, `blur`, `select`, `animation`, `tooltip` |
| `fig.barh()` | Horizontal bar | Same as `bar()`, horizontal orientation |
| `fig.scatter()` | Scatter plot | `color`, `size`, `item_style`, `symbol_rotate`, `blur`, `select`, `animation`, `tooltip` |
| `fig.hist()` | Histogram | `bins`, `item_style`, `label_style`, `animation`, `tooltip` |
| `fig.boxplot()` | Box plot | `item_style`, `label_style`, `labels`, `color`, `blur`, `select`, `animation`, `tooltip` |
| `fig.kde()` | KDE density | `line_style`, `area_style`, `labels`, `connect_nulls`, `color`, `animation`, `tooltip` |
| `fig.waterfall()` | Waterfall chart | `total`, `connector`, `label_style`, `item_style`, `animation`, `tooltip` |
| `fig.candlestick()` | Candlestick / OHLC | `border_width`, `opacity`, `label_style`, `labels`, `animation`, `tooltip` |
| `fig.heatmap()` | Matrix heatmap | `label_style`, `item_style`, `animation`, `tooltip` |

### Standalone & Hierarchical Charts

| Method | Description | Key options |
|:---|:---|:---|
| `fig.pie()` | Pie / donut | `inner_radius`, `link_legend`, `min_angle`, `clockwise`, `item_style`, `blur`, `select`, `tooltip` |
| `fig.radar()` | Radar / spider | `shape`, `split_number`, `line_style`, `color`, `tooltip` |
| `fig.funnel()` | Funnel | `orient`, `funnel_align`, `min_size`, `max_size`, `item_style`, `tooltip` |
| `fig.gauge()` | Gauge / meter | `detail_formatter`, `progress`, `item_style`, `tooltip` |
| `fig.treemap()` | Treemap | `drill_down_icon`, `node_click`, `breadcrumb`, `item_style`, `tooltip` |
| `fig.sunburst()` | Sunburst | `node_click`, `label_rotate`, `item_style`, `tooltip` |

### Relational & Calendar Charts

| Method | Description | Key options |
|:---|:---|:---|
| `fig.sankey()` | Sankey diagram | `node_align`, `draggable`, `item_style`, `label_style`, `tooltip` |
| `fig.graph()` | Network graph | `repulsion`, `gravity`, `edge_label`, `item_style`, `line_style`, `tooltip` |
| `fig.calendar_heatmap()` | Calendar heatmap | `split_line_show`, `day_label_show`, `item_style`, `emphasis`, `tooltip` |

---

## Gallery

### Cartesian Charts

<table>
<tr>
<td width="50%">
<p align="center"><strong>Bar + Pie Overlay</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_bar_pie.png" alt="Bar + Pie Overlay" width="100%" /></p>

```python
fig = ec.figure(height="500px")
fig.bar(df, x="Dept", y="Budget",
        gradient=True, labels=True)
fig.pie(df, names="Dept", values="Budget")  # auto-overlay
fig.show()
```
</td>
<td width="50%">
<p align="center"><strong>Smooth Line</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_line.png" alt="Line Chart" width="100%" /></p>

```python
fig = ec.figure()
fig.plot(df, x="Month", y="Sales",
         smooth=True, area=True)
fig.show()
```
</td>
</tr>
<tr>
<td width="50%">
<p align="center"><strong>Scatter Plot</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_scatter.png" alt="Scatter Plot" width="100%" /></p>

```python
fig = ec.figure()
fig.scatter(df, x="Height", y="Weight",
            color="Gender", size="Age")
fig.show()
```
</td>
<td width="50%">
<p align="center"><strong>Grouped Bar</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_grouped.png" alt="Grouped Bar" width="100%" /></p>

```python
fig = ec.figure()
fig.bar(df, x="Quarter", y="Revenue",
        hue="Region")
fig.show()
```
</td>
</tr>
<tr>
<td width="50%">
<p align="center"><strong>Stacked Bar</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_stacked.png" alt="Stacked Bar" width="100%" /></p>

```python
fig = ec.figure()
fig.bar(df, x="Month", y="Revenue",
        hue="Product", stack=True)
fig.show()
```
</td>
<td width="50%">
<p align="center"><strong>Histogram</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_histogram.png" alt="Histogram" width="100%" /></p>

```python
fig = ec.figure()
fig.hist(df, column="Score", bins=20)
fig.show()
```
</td>
</tr>
<tr>
<td width="50%">
<p align="center"><strong>Dual Axis: Bar + Line</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_dual.png" alt="Dual Axis" width="100%" /></p>

```python
fig = ec.figure()
fig.bar(df, x="Month", y="Revenue")
fig.plot(df, x="Month", y="Growth",
         smooth=True, axis=1)
fig.ylabel("Revenue ($K)")
fig.ylabel_right("Growth %")
fig.show()
```
</td>
<td width="50%">
<p align="center"><strong>Boxplot</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_boxplot.png" alt="Boxplot" width="100%" /></p>

```python
fig = ec.figure()
fig.boxplot(df, x="Department", y="Salary")
fig.show()
```
</td>
</tr>
</table>

### Standalone Charts

<table>
<tr>
<td width="50%">
<p align="center"><strong>Donut / Pie</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_pie.png" alt="Donut Chart" width="100%" /></p>

```python
fig = ec.figure()
fig.pie(df, names="Browser", values="Share",
        inner_radius="40%")
fig.show()
```
</td>
<td width="50%">
<p align="center"><strong>Radar</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_radar.png" alt="Radar Chart" width="100%" /></p>

```python
fig = ec.figure()
fig.radar(indicators, data,
          series_names=["Warrior","Mage"])
fig.show()
```
</td>
</tr>
<tr>
<td width="50%">
<p align="center"><strong>Heatmap</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_heatmap.png" alt="Heatmap" width="100%" /></p>

```python
fig = ec.figure()
fig.heatmap(df, x="Day", y="Hour",
            value="Count")
fig.show()
```
</td>
<td width="50%">
<p align="center"><strong>Funnel</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_funnel.png" alt="Funnel" width="100%" /></p>

```python
fig = ec.figure()
fig.funnel(df, names="Stage", values="Count")
fig.show()
```
</td>
</tr>
<tr>
<td width="50%">
<p align="center"><strong>Treemap</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_treemap.png" alt="Treemap" width="100%" /></p>

```python
fig = ec.figure()
fig.treemap(df,
    path=["Category","SubCat"],
    value="Sales")
fig.show()
```
</td>
<td width="50%">
<p align="center"><strong>Sankey Diagram</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_sankey.png" alt="Sankey Diagram" width="100%" /></p>

```python
fig = ec.figure()
fig.sankey(df,
    levels=["Source","Channel","Outcome"],
    value="Users")
fig.show()
```
</td>
</tr>
</table>

### Composite & Dashboard Charts

<table>
<tr>
<td width="50%">
<p align="center"><strong>Bar + Pie (Dark)</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_composite_dark.png" alt="Composite Dark" width="100%" /></p>
</td>
<td width="50%">
<p align="center"><strong>Triple Composite</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_composite.png" alt="Triple Composite" width="100%" /></p>
</td>
</tr>
<tr>
<td width="50%">
<p align="center"><strong>KPI Dashboard</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_dashboard.png" alt="KPI Dashboard" width="100%" /></p>
</td>
<td width="50%">
<p align="center"><strong>Stacked + Trend + Pie</strong></p>
<p align="center"><img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_full_dashboard.png" alt="Full Dashboard" width="100%" /></p>
</td>
</tr>
</table>

> Every chart is fully interactive -- hover for tooltips, click legend items to toggle series, use the toolbox to export.
> Open the [HTML demos in `assets/`](assets/) for the live experience, or run `python generate_demos.py` yourself.

---

## Rendering Engines

Write your chart once; `ec.config()` controls where it renders.

| Engine | Use case | Install |
|:---|:---|:---|
| `"python"` | Standalone scripts -- opens the default browser | No extra deps |
| `"jupyter"` | Jupyter Notebook / JupyterLab inline widgets | `pip install echartsy[jupyter]` |
| `"streamlit"` | Streamlit applications | `pip install echartsy[streamlit]` |

```python
ec.config(engine="jupyter")          # or "python" / "streamlit"
```

---

## Advanced Features

### Multi-Grid Subplots

Create vertically stacked chart panels sharing independent axes:

```python
fig = ec.figure(rows=2, height="700px", row_heights=["60%", "40%"])
fig.bar(df, x="Month", y="Revenue", grid=0)
fig.plot(df, x="Month", y="Growth", grid=1, smooth=True)
fig.show()
```

### Timeline Animations

Animate any chart across a time dimension with `TimelineFigure`:

```python
fig = ec.TimelineFigure(height="500px", interval=1.5)
fig.bar(df, x="Country", y="GDP", time_col="Year", labels=True)
fig.title("GDP by Country")
fig.show()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/astrojigs/echartsy/main/assets/demo_timeline.png" alt="Timeline Animation" width="700" />
</p>

| Feature | API |
|:---|:---|
| Playback control | `TimelineFigure(interval=2.0, autoplay=True, loop=True)` |
| Adjust after creation | `fig.playback(interval=1.0, rewind=True)` |
| Fixed axis ranges | `fig.xlim()`, `fig.ylim()` -- consistent scales across frames |
| Smart frame sorting | Parses years, quarters (`Q1 2024`), months, ISO dates, fiscal years |
| Supported series | `bar()`, `plot()`, `scatter()`, `pie()`, `hist()` |
| Diagnose format | `ec.detect_time_format(df["Year"])` |

### Annotations

Add reference lines, points, and shaded regions to any series:

```python
fig.plot(df, x="Month", y="Sales", smooth=True)
fig.mark_line(y=500, label="Target", color="red", line_dash="dashed")
fig.mark_point(type="max")
fig.mark_area(y_range=[200, 400], color="#ccc", opacity=0.15)
fig.show()
```

### Visual Map

Attach a colour-mapping control for continuous or piecewise data ranges:

```python
fig.heatmap(df, x="Day", y="Hour", value="Count")
fig.visual_map(min_val=0, max_val=100,
               colors=["#313695", "#ffffbf", "#a50026"],
               calculable=True)
fig.show()
```

### Log Scale

Switch any y-axis to logarithmic scale:

```python
fig.yscale("log")          # shorthand
fig.ylim(scale="log")      # equivalent
```

### Emphasis (Hover Highlighting)

Control what happens when users hover over chart elements using typed Python dataclasses:

```python
from echartsy import Emphasis, ItemStyle

fig.bar(df, x="Month", y="Revenue", hue="Region",
        emphasis=Emphasis(
            focus="series",
            item_style=ItemStyle(shadow_blur=10),
        ))
```

Every chart method accepts an optional `emphasis` parameter with a chart-specific type:

| Chart method | Emphasis class |
|:---|:---|
| `bar()`, `waterfall()`, `hist()`, `boxplot()`, `heatmap()`, `candlestick()` | `Emphasis` |
| `plot()`, `kde()` | `LineEmphasis` |
| `scatter()` | `ScatterEmphasis` |
| `pie()` | `PieEmphasis` |
| `radar()` | `RadarEmphasis` |
| `sankey()` | `SankeyEmphasis` |
| `funnel()` | `FunnelEmphasis` |
| `treemap()` | `TreemapEmphasis` |
| `graph()` | `GraphEmphasis` |

### Per-Series Style Control

Beyond emphasis, every chart method now accepts typed dataclass parameters for fine-grained visual control:

```python
from echartsy import ItemStyle, LabelStyle, Blur, Select, AnimationConfig, TooltipStyle

fig.bar(df, x="Month", y="Revenue",
        item_style=ItemStyle(border_type="dashed", opacity=0.9),
        label_style=LabelStyle(show=True, rotate=45, font_family="monospace"),
        blur=Blur(item_style=ItemStyle(opacity=0.2)),
        select=Select(item_style=ItemStyle(border_width=3)),
        selected_mode="multiple",
        animation=AnimationConfig(animation_duration=1500, animation_easing="elasticOut"),
        tooltip=TooltipStyle(formatter="{b}: {c}"))
```

Three-tier control: scalar params (e.g. `color="red"`) → dataclass overrides → `**series_kw` raw dict.

### Adaptive Dark Mode

Charts automatically respond to the user's OS or browser `prefers-color-scheme` setting:

```python
ec.config(engine="jupyter", adaptive="auto")     # auto-detect (default)
ec.config(engine="jupyter", adaptive="dark")     # force dark
ec.config(engine="streamlit")                    # auto-adapts to Streamlit theme
```

### Style Presets and Palettes

Apply a pre-built visual theme or set custom colour palettes:

```python
fig = ec.figure(style=ec.StylePreset.CLINICAL)
fig = ec.figure(style=ec.StylePreset.DASHBOARD_DARK)
fig = ec.figure(style=ec.StylePreset.KPI_REPORT)
fig = ec.figure(style=ec.StylePreset.MINIMAL)

fig.palette(["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe"])
fig.palette(ec.PALETTE_RUSTY)
```

Build custom presets for full control over fonts, grid lines, tooltip style, and more:

```python
my_style = ec.StylePreset(
    palette=("#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"),
    bg="#fefae0",
    font_family="Georgia",
    title_font_size=20,
)
fig = ec.figure(style=my_style)
```

---

## Configuration Reference

Every `Figure` and `TimelineFigure` supports these configuration methods:

```python
# Titles
fig.title("Main Title", subtitle="Sub-title")

# Axes
fig.xlabel("X Label", rotate=30)
fig.ylabel("Y Label")
fig.ylabel_right("Secondary Y")
fig.xlim(0, 100)
fig.ylim(0, 500)
fig.yscale("log")

# Layout
fig.legend(orient="vertical", left="right", top=40)
fig.margins(left=100, right=120, top=40)
fig.grid(show=True)

# Interactivity
fig.datazoom(start=0, end=80)
fig.toolbox(download=True, zoom=True)
fig.tooltip(trigger="axis", pointer="cross")
fig.axis_pointer(type="shadow", snap=True)
fig.visual_map(min_val=0, max_val=100)

# Export
fig.save(name="my_chart", fmt="png", dpi=3)
fig.to_html("my_chart.html")
option = fig.to_option()       # raw ECharts option dict

# Palette
fig.palette(["#5470C6", "#91CC75", "#FAC858"])
```

---

## API at a Glance

### `ec.config(engine, adaptive="auto")`

Set the global rendering engine (`"python"`, `"jupyter"`, `"streamlit"`) and theme adaptation mode (`"auto"`, `"light"`, `"dark"`).

### `ec.figure(**kwargs)` / `ec.Figure(**kwargs)`

Create a chart canvas.

| Parameter | Default | Description |
|:---|:---|:---|
| `height` | `"400px"` | CSS height of the chart container |
| `width` | `None` | CSS width (defaults to full container) |
| `renderer` | `"svg"` | `"canvas"` or `"svg"` |
| `style` | `StylePreset.CLINICAL` | A `StylePreset` instance |
| `rows` | `1` | Number of vertical grid panels (subplots) |
| `row_heights` | `None` | List of CSS heights per grid panel |

### `ec.TimelineFigure(**kwargs)` / `ec.timeline_figure(**kwargs)`

Same as `Figure` but adds timeline animation. Extra parameters:

| Parameter | Default | Description |
|:---|:---|:---|
| `interval` | `2.0` | Seconds between animation frames |
| `autoplay` | `True` | Start playing automatically |
| `loop` | `True` | Loop back to the first frame |

### Sub-style Dataclasses

| Class | Key fields |
|:---|:---|
| `ec.ItemStyle` | `color`, `border_color`, `border_width`, `border_radius`, `border_type`, `shadow_blur`, `shadow_color`, `shadow_offset_x`, `shadow_offset_y`, `opacity`, `decal` |
| `ec.LabelStyle` | `show`, `position`, `formatter`, `font_size`, `font_weight`, `font_family`, `color`, `rotate`, `offset`, `align` |
| `ec.LineStyle` | `color`, `width`, `type`, `shadow_blur`, `shadow_color`, `shadow_offset_x`, `shadow_offset_y`, `opacity`, `cap`, `join` |
| `ec.AreaStyle` | `color`, `opacity`, `origin`, `shadow_blur`, `shadow_color` |
| `ec.LabelLineStyle` | `show`, `length`, `length2` |
| `ec.EndLabelStyle` | `show`, `formatter`, `font_size`, `font_weight`, `color` |
| `ec.Blur` | `item_style`, `label`, `line_style`, `area_style` |
| `ec.Select` | `disabled`, `item_style`, `label`, `line_style`, `area_style` |
| `ec.TooltipStyle` | `show`, `formatter`, `value_formatter`, `background_color`, `border_color`, `border_width`, `text_color`, `text_size` |
| `ec.AnimationConfig` | `animation`, `animation_duration`, `animation_easing`, `animation_delay` |

---

## Changelog (Recent)

### v0.7.0
- **Added:** Full Figure API parity for MatrixFigure cell builders — `hue`, `stack`, `gradient`, `agg`, `blur`, `select`, `animation` on `.cell().bar()`, `.cell().plot()`, `.cell().scatter()`, `.cell().pie()`
- **Added:** `build_scatter_series()` and `build_pie_series()` shared builder functions in `_chart_methods.py`
- **Added:** Automatic legend emission from cell hue/color grouping with scrollable bottom legend
- **Changed:** `_CellBuilder.scatter(color=...)` now takes a column name for grouping (matching `Figure.scatter()`) — use `item_style=ec.ItemStyle(color=...)` for literal colours
- **Fixed:** MatrixFigure grid layout — `containLabel: True`, proper margins, no cell/label/legend overlap
- **Fixed:** Row/column name labels positioned on y-axis (first column) and x-axis (bottom row) to prevent mid-grid collisions

### v0.6.5
- **Added:** `label_style` and `upper_label_style` on `treemap()` — full `LabelStyle` control for treemap labels (font, color, position, overflow)
- **Added:** `sort`, `square_ratio`, `children_visible_min`, `visual_min`, `visual_max`, `zoom_to_node_ratio`, `color_mapping_by`, and `levels` parameters on `treemap()`
- **Added:** `vertical_align`, `padding`, `overflow`, `ellipsis` fields on `LabelStyle`

### v0.6.4
- **Added:** `upper_label` and `upper_label_height` parameters on `treemap()` — enables parent category labels as header bars on treemap nodes (mirrors ECharts `upperLabel` feature)

### v0.6.3
- **Extended:** TimelineFigure API parity — added 34 missing parameters across `plot()`, `bar()`, `pie()`, `title()`, `legend()`, `tooltip()`, `xlim()`, `ylim()`
- **Added:** Documentation for gauge, sunburst, graph, and calendar_heatmap chart types
- **Fixed:** 10 undocumented parameters added to existing Documentation.md tables

### v0.6.2

- **Added:** Auto-overlay for `pie()` on cartesian figures — calling `fig.bar(); fig.pie()` now works without specifying `center` or `radius` (defaults to top-right mini pie at `["82%", "25%"]`).
- **Added:** `link_legend` parameter on `pie()` — controls whether overlay pie shares colors with existing series (`True`), uses distinct colors (`False`), or auto-detects based on name overlap (`None`, default).
- **Fixed:** Overlay pies now automatically get per-series `trigger: "item"` tooltips so hovering works correctly (the global `"axis"` trigger doesn't fire for pie series).
- **Fixed:** Multi-grid (subplot) figures with overlay pies now correctly inject per-series tooltips.

### v0.6.1

- **Fixed:** Waterfall `connector=True` rendered blank charts — markLine data used invalid nested format instead of `[start, end]` pairs.
- **Fixed:** `bar_width` parameter mapped to `barMaxWidth` instead of `barWidth` in ECharts options.
- **Fixed:** `scatter()` mutated the global x-axis type to `"value"`, breaking any previously added category-based series (bar/line).
- **Fixed:** Horizontal bar charts (`barh()`) discarded all y-axis customizations (labels, fonts, colors) during option building.
- **Fixed:** `heatmap()` overwrote axis config entirely, discarding `StylePreset` axis formatting.
- **Fixed:** `tooltip()` replaced the entire `axisPointer` sub-config instead of merging, losing prior `axis_pointer()` settings.
- **Fixed:** `sankey()` and `graph()` used direct assignment for `item_style`/`line_style`, overwriting prior config instead of merging.
- **Fixed:** `datazoom` and `extra` configs were not deep-copied in `to_option()`, causing mutable reference leaks.
- **Fixed:** `radar()`, `funnel()`, and standalone `pie()` legend deduplication — duplicate or overwritten legend entries no longer occur.
- **Fixed:** `candlestick()` now warns when duplicate date rows are found (only the last row per date is used).
- **Fixed:** `boxplot(orient="h")` now warns that horizontal orientation is not yet supported instead of silently ignoring it.
- **Fixed:** `_sort_categories` 60% parse threshold could silently drop non-date categories — now requires all unique categories to parse as dates before sorting chronologically.
- **Fixed:** `LineEmphasis.end_label` was typed as `LabelStyle` instead of `EndLabelStyle`, allowing invalid fields.
- **Fixed:** `StylePreset` preset constants (`CLINICAL`, `DASHBOARD_DARK`, etc.) were dataclass fields instead of `ClassVar`, polluting constructor, `repr`, and `fields()`.
- **Fixed:** Python/Jupyter HTML renderers did not escape `</` sequences in JSON, allowing potential `</script>` injection (XSS).
- **Fixed:** HTML template placeholder substitution order could corrupt user data containing `__CHART_ID__` or similar strings.
- **Fixed:** Malformed `file://` URL on Windows in the Python standalone renderer.
- **Added:** `animation` parameter to `pie()` (was missing while all other chart methods had it).

### v0.6.0

- **Added:** Full parameter expansion across all 19 chart methods and 6 config methods — typed `ItemStyle`, `LabelStyle`, `LineStyle`, `AreaStyle`, `EndLabelStyle`, `Blur`, `Select`, `TooltipStyle`, and `AnimationConfig` dataclass parameters on every series method.
- **Added:** Per-series `tooltip`, `blur`, `select`, `selected_mode`, and `animation` support for bar, line, scatter, pie, histogram, boxplot, KDE, candlestick, heatmap, waterfall, radar, sankey, treemap, funnel, gauge, sunburst, graph, and calendar heatmap.
- **Added:** Extended config methods — `title()` (+8 params), `legend()` (+11 params), `tooltip()` (+9 params), `toolbox()` (+5 params), `datazoom()` (+6 params), `visual_map()` (+6 params).
- **Added:** Timeline parity — `TimelineFigure.plot()`, `.bar()`, `.scatter()`, `.pie()`, `.hist()` now accept all new style parameters.
- **Added:** 5 new dataclasses: `EndLabelStyle`, `Blur`, `Select`, `TooltipStyle`, `AnimationConfig`.
- **Extended:** `ItemStyle` (+`border_type`, `decal`), `LabelStyle` (+`font_family`, `rotate`, `offset`, `align`), `LineStyle` (+`shadow_color`, `shadow_offset_x/y`, `cap`, `join`), `AreaStyle` (+`origin`, `shadow_blur`, `shadow_color`).
- **Added:** Three-tier control model: scalar params → dataclass overrides → `**series_kw` raw dict.

### v0.5.3

- **Added:** `fig.waterfall()` -- finance/accounting waterfall charts with cumulative positive/negative deltas, optional total bar, connector lines, and value labels.

### v0.5.1

See [Releases](https://github.com/astrojigs/echartsy/releases) for full details.

### v0.4.8

- **Changed:** Default renderer switched from `"canvas"` to `"svg"` -- fixes the broken toolbox download button under canvas/iframe sandbox restrictions.

### v0.4.7

- **Added:** `fig.candlestick()` -- native OHLC candlestick charts with dual-axis support and full composability.

### v0.4.6

- **Added:** `Emphasis` support for `boxplot()`, `heatmap()`, and `funnel()`.

### v0.4.5

- **Fixed:** Charts vanishing across `st.tabs()` in Streamlit 1.48+.

---

## Support the Project

If echartsy saves you time or you enjoy using it, consider buying me a coffee. It helps keep the project maintained and growing.

<p align="center">
  <a href="https://www.buymeacoffee.com/astrojigs2a" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50" width="200" />
  </a>
</p>

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an [issue](https://github.com/astrojigs/echartsy/issues) or submit a pull request on [GitHub](https://github.com/astrojigs/echartsy).

## License

[MIT](LICENSE) &mdash; Jigar, 2026

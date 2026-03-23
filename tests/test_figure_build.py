"""Tests for Figure chart methods and option building."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import echartsy as ec
from echartsy.exceptions import BuilderConfigError


# ═══════════════════════════════════════════════════════════════════════════
# Bar chart tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBar:
    def test_bar_basic(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        opt = fig.to_option()
        assert any(s["type"] == "bar" for s in opt["series"])

    def test_bar_hue(self, hue_df):
        fig = ec.Figure()
        fig.bar(hue_df, x="X", y="Y", hue="Group")
        opt = fig.to_option()
        assert len(opt["series"]) == 2
        names = {s["name"] for s in opt["series"]}
        assert "G1" in names and "G2" in names

    def test_bar_stack(self, hue_df):
        fig = ec.Figure()
        fig.bar(hue_df, x="X", y="Y", hue="Group", stack=True)
        opt = fig.to_option()
        for s in opt["series"]:
            assert "stack" in s

    def test_bar_horizontal(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y", orient="h")
        opt = fig.to_option()
        # Horizontal bar swaps axes: xAxis becomes value, yAxis becomes category
        assert opt["xAxis"]["type"] == "value"
        assert opt["yAxis"][0]["type"] == "category" if isinstance(opt["yAxis"], list) else opt["yAxis"]["type"] == "category"

    def test_barh_shortcut(self, simple_df):
        fig = ec.Figure()
        fig.barh(simple_df, x="X", y="Y")
        opt = fig.to_option()
        assert opt["xAxis"]["type"] == "value"


# ═══════════════════════════════════════════════════════════════════════════
# Line/plot chart tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPlot:
    def test_plot_basic(self, simple_df):
        fig = ec.Figure()
        fig.plot(simple_df, x="X", y="Y")
        opt = fig.to_option()
        assert any(s["type"] == "line" for s in opt["series"])

    def test_plot_smooth(self, simple_df):
        fig = ec.Figure()
        fig.plot(simple_df, x="X", y="Y", smooth=True)
        opt = fig.to_option()
        line = [s for s in opt["series"] if s["type"] == "line"][0]
        assert line["smooth"] is True

    def test_plot_area(self, simple_df):
        fig = ec.Figure()
        fig.plot(simple_df, x="X", y="Y", area=True)
        opt = fig.to_option()
        line = [s for s in opt["series"] if s["type"] == "line"][0]
        assert "areaStyle" in line


# ═══════════════════════════════════════════════════════════════════════════
# Scatter chart tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScatter:
    def test_scatter_basic(self):
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0]})
        fig = ec.Figure()
        fig.scatter(df, x="X", y="Y")
        opt = fig.to_option()
        assert any(s["type"] == "scatter" for s in opt["series"])
        # Scatter sets x-axis to value type
        assert opt["xAxis"]["type"] == "value"


# ═══════════════════════════════════════════════════════════════════════════
# Pie chart tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPie:
    def test_pie_basic(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y")
        opt = fig.to_option()
        assert any(s["type"] == "pie" for s in opt["series"])

    def test_pie_donut(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y", inner_radius="30%")
        opt = fig.to_option()
        pie_series = [s for s in opt["series"] if s["type"] == "pie"][0]
        # When inner_radius is set, radius is a list [inner, outer]
        assert isinstance(pie_series["radius"], list)
        assert len(pie_series["radius"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Histogram tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHist:
    def test_hist_basic(self):
        df = pd.DataFrame({"Values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        fig = ec.Figure()
        fig.hist(df, column="Values", bins=5)
        opt = fig.to_option()
        # Histogram produces bar-type series with binned data
        assert any(s["type"] == "bar" for s in opt["series"])


# ═══════════════════════════════════════════════════════════════════════════
# Radar chart tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRadar:
    def test_radar_basic(self):
        indicators = [{"name": "A", "max": 100}, {"name": "B", "max": 100}, {"name": "C", "max": 100}]
        data = [[80, 90, 70]]
        fig = ec.Figure()
        fig.radar(indicators, data, series_names=["Team"])
        opt = fig.to_option()
        assert "radar" in opt
        assert any(s["type"] == "radar" for s in opt["series"])


# ═══════════════════════════════════════════════════════════════════════════
# Heatmap tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHeatmap:
    def test_heatmap_basic(self):
        df = pd.DataFrame({
            "X": ["Mon", "Mon", "Tue", "Tue"],
            "Y": ["AM", "PM", "AM", "PM"],
            "Val": [10, 20, 30, 40],
        })
        fig = ec.Figure()
        fig.heatmap(df, x="X", y="Y", value="Val")
        opt = fig.to_option()
        assert "visualMap" in opt
        assert any(s["type"] == "heatmap" for s in opt["series"])


# ═══════════════════════════════════════════════════════════════════════════
# Sankey tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSankey:
    def test_sankey_basic(self, sankey_df):
        fig = ec.Figure()
        fig.sankey(sankey_df, levels=["Level1", "Level2"], value="Value")
        opt = fig.to_option()
        sankey_s = [s for s in opt["series"] if s["type"] == "sankey"][0]
        assert "data" in sankey_s  # nodes
        assert "links" in sankey_s  # links


# ═══════════════════════════════════════════════════════════════════════════
# Treemap tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTreemap:
    def test_treemap_basic(self, sankey_df):
        fig = ec.Figure()
        fig.treemap(sankey_df, path=["Level1", "Level2"], value="Value")
        opt = fig.to_option()
        treemap_s = [s for s in opt["series"] if s["type"] == "treemap"][0]
        assert "data" in treemap_s
        # Hierarchy: Level1 items have children
        assert any("children" in item for item in treemap_s["data"])


# ═══════════════════════════════════════════════════════════════════════════
# Funnel tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFunnel:
    def test_funnel_basic(self, simple_df):
        fig = ec.Figure()
        fig.funnel(simple_df, names="X", values="Y")
        opt = fig.to_option()
        assert any(s["type"] == "funnel" for s in opt["series"])


# ═══════════════════════════════════════════════════════════════════════════
# Boxplot tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBoxplot:
    def test_boxplot_basic(self):
        df = pd.DataFrame({
            "Group": ["A"] * 5 + ["B"] * 5,
            "Val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        fig = ec.Figure()
        fig.boxplot(df, x="Group", y="Val")
        opt = fig.to_option()
        box_s = [s for s in opt["series"] if s["type"] == "boxplot"][0]
        # Each category produces a 5-number summary [min, Q1, median, Q3, max]
        assert len(box_s["data"]) == 2
        assert len(box_s["data"][0]) == 5


# ═══════════════════════════════════════════════════════════════════════════
# Candlestick tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCandlestick:
    def test_candlestick_basic(self, ohlc_df):
        fig = ec.Figure()
        fig.candlestick(ohlc_df, date="Date", open="Open", close="Close",
                        low="Low", high="High")
        opt = fig.to_option()
        assert any(s["type"] == "candlestick" for s in opt["series"])


# ═══════════════════════════════════════════════════════════════════════════
# Gauge tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGauge:
    def test_gauge_basic(self):
        fig = ec.Figure()
        fig.gauge(75, name="Score")
        opt = fig.to_option()
        gauge_s = [s for s in opt["series"] if s["type"] == "gauge"][0]
        assert gauge_s["data"][0]["value"] == 75

    def test_gauge_colors(self):
        colors = [(0.3, "#67e0e3"), (0.7, "#37a2da"), (1, "#fd666d")]
        fig = ec.Figure()
        fig.gauge(50, axis_line_colors=colors)
        opt = fig.to_option()
        gauge_s = [s for s in opt["series"] if s["type"] == "gauge"][0]
        assert "axisLine" in gauge_s
        assert len(gauge_s["axisLine"]["lineStyle"]["color"]) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Sunburst tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSunburst:
    def test_sunburst_basic(self, sankey_df):
        fig = ec.Figure()
        fig.sunburst(sankey_df, path=["Level1", "Level2"], value="Value")
        opt = fig.to_option()
        sun_s = [s for s in opt["series"] if s["type"] == "sunburst"][0]
        assert "data" in sun_s
        # Hierarchical structure
        assert any("children" in item for item in sun_s["data"])


# ═══════════════════════════════════════════════════════════════════════════
# Graph tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGraph:
    def test_graph_basic(self, nodes_df, edges_df):
        fig = ec.Figure()
        fig.graph(nodes_df, edges_df)
        opt = fig.to_option()
        graph_s = [s for s in opt["series"] if s["type"] == "graph"][0]
        assert len(graph_s["data"]) == 3
        assert len(graph_s["links"]) == 3

    def test_graph_with_categories(self, nodes_df, edges_df):
        fig = ec.Figure()
        fig.graph(nodes_df, edges_df, node_category="group")
        opt = fig.to_option()
        graph_s = [s for s in opt["series"] if s["type"] == "graph"][0]
        assert "categories" in graph_s
        assert len(graph_s["categories"]) == 2
        # Categories create a legend
        assert "legend" in opt


# ═══════════════════════════════════════════════════════════════════════════
# Calendar heatmap tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCalendarHeatmap:
    def test_calendar_heatmap_basic(self, calendar_df):
        fig = ec.Figure()
        fig.calendar_heatmap(calendar_df, date="date", value="value")
        opt = fig.to_option()
        assert "calendar" in opt
        assert "visualMap" in opt
        assert any(s.get("coordinateSystem") == "calendar" for s in opt["series"])


# ═══════════════════════════════════════════════════════════════════════════
# Error / conflict tests
# ═══════════════════════════════════════════════════════════════════════════

class TestErrors:
    def test_empty_series_raises(self):
        fig = ec.Figure()
        with pytest.raises(BuilderConfigError, match="no series"):
            fig.to_option()

    def test_mode_conflict(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y")
        with pytest.raises(BuilderConfigError, match="mode"):
            fig.bar(simple_df, x="X", y="Y")


# ═══════════════════════════════════════════════════════════════════════════
# Chrome configuration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestChrome:
    def test_title_config(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").title("My Chart")
        opt = fig.to_option()
        assert opt["title"]["text"] == "My Chart"

    def test_xlabel_ylabel(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").xlabel("Category").ylabel("Amount")
        opt = fig.to_option()
        assert opt["xAxis"]["name"] == "Category"
        y_axis = opt["yAxis"] if isinstance(opt["yAxis"], dict) else opt["yAxis"][0]
        assert y_axis["name"] == "Amount"

    def test_log_scale_ylim(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").ylim(scale="log")
        opt = fig.to_option()
        y_axis = opt["yAxis"] if isinstance(opt["yAxis"], dict) else opt["yAxis"][0]
        assert y_axis["type"] == "log"

    def test_log_scale_yscale(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").yscale("log")
        opt = fig.to_option()
        y_axis = opt["yAxis"] if isinstance(opt["yAxis"], dict) else opt["yAxis"][0]
        assert y_axis["type"] == "log"

    def test_visual_map(self):
        df = pd.DataFrame({
            "X": ["Mon", "Tue"],
            "Y": ["AM", "PM"],
            "Val": [10, 20],
        })
        fig = ec.Figure()
        fig.heatmap(df, x="X", y="Y", value="Val")
        fig.visual_map(min_val=0, max_val=50, colors=["#blue", "#red"])
        opt = fig.to_option()
        assert "visualMap" in opt

    def test_summary(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").title("Test")
        result = fig.summary()
        assert isinstance(result, str)
        assert "Series" in result

    def test_datazoom(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").datazoom(start=0, end=80)
        opt = fig.to_option()
        assert "dataZoom" in opt
        assert len(opt["dataZoom"]) == 2  # inside + slider

    def test_palette(self, simple_df):
        custom_colors = ["#ff0000", "#00ff00", "#0000ff"]
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").palette(custom_colors)
        opt = fig.to_option()
        assert opt["color"] == custom_colors

    def test_dual_axis(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").ylabel_right("Secondary")
        opt = fig.to_option()
        # yAxis should be a list with 2 elements
        assert isinstance(opt["yAxis"], list)
        assert len(opt["yAxis"]) == 2
        assert opt["yAxis"][1]["name"] == "Secondary"


# ═══════════════════════════════════════════════════════════════════════════
# Annotation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAnnotations:
    def test_mark_line_horizontal(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").mark_line(y=15)
        opt = fig.to_option()
        ml = opt["series"][0]["markLine"]
        assert any("yAxis" in d for d in ml["data"])

    def test_mark_line_vertical(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").mark_line(x="B")
        opt = fig.to_option()
        ml = opt["series"][0]["markLine"]
        assert any("xAxis" in d for d in ml["data"])

    def test_mark_point_max(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").mark_point(type="max")
        opt = fig.to_option()
        mp = opt["series"][0]["markPoint"]
        assert any(d.get("type") == "max" for d in mp["data"])

    def test_mark_area_y_range(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").mark_area(y_range=[10, 25])
        opt = fig.to_option()
        ma = opt["series"][0]["markArea"]
        assert len(ma["data"]) == 1
        pair = ma["data"][0]
        assert pair[0]["yAxis"] == 10
        assert pair[1]["yAxis"] == 25


# ═══════════════════════════════════════════════════════════════════════════
# HTML export test
# ═══════════════════════════════════════════════════════════════════════════

class TestExport:
    def test_to_html(self, simple_df, tmp_path):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        filepath = str(tmp_path / "test_chart.html")
        result = fig.to_html(filepath)
        assert result == filepath
        assert os.path.exists(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        assert "<!DOCTYPE html>" in html
        assert "echarts" in html


# ═══════════════════════════════════════════════════════════════════════════
# Subplot / multi-grid tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSubplots:
    def test_multi_grid_basic(self, simple_df):
        fig = ec.Figure(rows=2, row_heights=["70%", "30%"])
        fig.plot(simple_df, x="X", y="Y", grid=0)
        fig.bar(simple_df, x="X", y="Y", grid=1)
        opt = fig.to_option()
        assert isinstance(opt["grid"], list)
        assert len(opt["grid"]) == 2
        assert len(opt["xAxis"]) == 2
        assert len(opt["yAxis"]) == 4  # 2 per grid (left + right)

    def test_multi_grid_series_indices(self, simple_df):
        fig = ec.Figure(rows=2)
        fig.plot(simple_df, x="X", y="Y", grid=0)
        fig.bar(simple_df, x="X", y="Y", grid=1)
        opt = fig.to_option()
        assert opt["series"][0]["xAxisIndex"] == 0
        assert opt["series"][1]["xAxisIndex"] == 1

    def test_multi_grid_per_grid_categories(self):
        df1 = pd.DataFrame({"X": ["A", "B"], "Y": [1, 2]})
        df2 = pd.DataFrame({"X": ["C", "D"], "Y": [3, 4]})
        fig = ec.Figure(rows=2)
        fig.bar(df1, x="X", y="Y", grid=0)
        fig.bar(df2, x="X", y="Y", grid=1)
        opt = fig.to_option()
        assert opt["xAxis"][0]["data"] == ["A", "B"]
        assert opt["xAxis"][1]["data"] == ["C", "D"]

    def test_linked_datazoom(self, simple_df):
        fig = ec.Figure(rows=2)
        fig.plot(simple_df, x="X", y="Y", grid=0)
        fig.bar(simple_df, x="X", y="Y", grid=1)
        fig.datazoom(grids=[0, 1])
        opt = fig.to_option()
        assert opt["dataZoom"][0]["xAxisIndex"] == [0, 1]

    def test_single_grid_backward_compat(self, simple_df):
        """rows=1 (default) should produce the same single-grid output."""
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        opt = fig.to_option()
        assert isinstance(opt["grid"], dict)
        assert isinstance(opt["xAxis"], dict)

    def test_candlestick_multi_grid(self, ohlc_df, simple_df):
        fig = ec.Figure(rows=2, row_heights=["70%", "30%"])
        fig.candlestick(ohlc_df, date="Date", open="Open",
                        close="Close", low="Low", high="High", grid=0)
        fig.bar(simple_df, x="X", y="Y", grid=1)
        opt = fig.to_option()
        assert len(opt["grid"]) == 2
        assert opt["series"][0].get("xAxisIndex") == 0

    def test_multi_grid_with_title(self, simple_df):
        fig = ec.Figure(rows=2)
        fig.bar(simple_df, x="X", y="Y", grid=0)
        fig.bar(simple_df, x="X", y="Y", grid=1)
        fig.title("My Subplots")
        opt = fig.to_option()
        assert opt["title"]["text"] == "My Subplots"


# ═══════════════════════════════════════════════════════════════════════════
# Additional coverage tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAdditionalCoverage:
    def test_log_scale_xlim(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.xlim(scale="log")
        opt = fig.to_option()
        assert opt["xAxis"]["type"] == "log"

    def test_xscale(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.xscale("log")
        opt = fig.to_option()
        assert opt["xAxis"]["type"] == "log"

    def test_mark_line_no_series_raises(self):
        fig = ec.Figure()
        fig._chart_mode = "cartesian"
        with pytest.raises(BuilderConfigError):
            fig.mark_line(y=5)

    def test_mark_point_no_series_raises(self):
        fig = ec.Figure()
        fig._chart_mode = "cartesian"
        with pytest.raises(BuilderConfigError):
            fig.mark_point(type="max")

    def test_mark_area_no_series_raises(self):
        fig = ec.Figure()
        fig._chart_mode = "cartesian"
        with pytest.raises(BuilderConfigError):
            fig.mark_area(y_range=[0, 10])

    def test_mark_line_with_color_dash(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.mark_line(y=15, label="Target", color="red", line_dash="dashed")
        opt = fig.to_option()
        ml = opt["series"][0]["markLine"]["data"][0]
        assert ml["lineStyle"]["color"] == "red"
        assert ml["lineStyle"]["type"] == "dashed"
        assert ml["label"]["formatter"] == "Target"

    def test_mark_point_with_coord(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.mark_point(coord=["A", 10], label="Special")
        opt = fig.to_option()
        mp = opt["series"][0]["markPoint"]["data"][0]
        assert mp["coord"] == ["A", 10]
        assert mp["name"] == "Special"

    def test_mark_area_x_range(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.mark_area(x_range=["A", "B"], color="blue", opacity=0.2)
        opt = fig.to_option()
        ma = opt["series"][0]["markArea"]["data"][0]
        assert ma[0]["xAxis"] == "A"
        assert ma[0]["itemStyle"]["color"] == "blue"

    def test_visual_map_piecewise(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.visual_map(piecewise=True, pieces=[{"min": 0, "max": 10}])
        opt = fig.to_option()
        vm = opt["visualMap"]
        assert vm["type"] == "piecewise"
        assert len(vm["pieces"]) == 1

    def test_repr(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.title("Test")
        r = repr(fig)
        assert "series=1" in r
        assert "title='Test'" in r

    def test_barh_passes_orient(self, simple_df):
        fig = ec.Figure()
        fig.barh(simple_df, x="X", y="Y")
        opt = fig.to_option()
        assert opt["xAxis"]["type"] == "value"

    def test_gauge_no_colors(self):
        fig = ec.Figure()
        fig.gauge(50, name="Test")
        opt = fig.to_option()
        assert opt["series"][0]["data"][0]["value"] == 50
        assert "axisLine" not in opt["series"][0]

    def test_graph_no_categories(self, nodes_df, edges_df):
        fig = ec.Figure()
        fig.graph(nodes_df, edges_df)
        opt = fig.to_option()
        assert "legend" not in opt

    def test_calendar_auto_year(self, calendar_df):
        fig = ec.Figure()
        fig.calendar_heatmap(calendar_df, date="date", value="value")
        opt = fig.to_option()
        assert "calendar" in opt


# ═══════════════════════════════════════════════════════════════════════════
# Extended coverage tests — chrome methods, edge cases, axis_pointer
# ═══════════════════════════════════════════════════════════════════════════

class TestChromeExtended:
    def test_title_with_subtitle(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.title("Main", subtitle="Sub", top=20)
        opt = fig.to_option()
        assert opt["title"]["subtext"] == "Sub"
        assert opt["title"]["top"] == 20

    def test_xlabel_rotate_fontsize_color(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.xlabel("Cat", rotate=45, font_size=14, color="#ff0000")
        opt = fig.to_option()
        assert opt["xAxis"]["axisLabel"]["rotate"] == 45
        assert opt["xAxis"]["axisLabel"]["fontSize"] == 14
        assert opt["xAxis"]["axisLabel"]["color"] == "#ff0000"

    def test_xticks(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.xticks(rotate=30, interval=2, formatter="{value}%")
        opt = fig.to_option()
        assert opt["xAxis"]["axisLabel"]["interval"] == 2

    def test_xlim_min_max(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.xlim(min_val=0, max_val=100)
        opt = fig.to_option()
        assert opt["xAxis"]["min"] == 0
        assert opt["xAxis"]["max"] == 100

    def test_ylim_min_max(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.ylim(min_val=0, max_val=50)
        opt = fig.to_option()
        y = opt["yAxis"] if isinstance(opt["yAxis"], dict) else opt["yAxis"][0]
        assert y["min"] == 0
        assert y["max"] == 50

    def test_ylim_scale_on_second_axis(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.ylabel_right("Right")
        fig.ylim(scale="log", axis=1)
        opt = fig.to_option()
        assert opt["yAxis"][1]["type"] == "log"

    def test_grid_config(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.grid(show=True, axis="both", style="dashed", color="#ccc")
        opt = fig.to_option()
        # Grid lines affect split lines on axes (not directly in option["grid"])
        # Just check it doesn't error
        assert "grid" in opt

    def test_margins(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.margins(left=100, right=100, top=80, bottom=60)
        opt = fig.to_option()
        assert opt["grid"]["left"] == 100
        assert opt["grid"]["right"] == 100

    def test_legend_config(self, hue_df):
        fig = ec.Figure()
        fig.bar(hue_df, x="X", y="Y", hue="Group")
        fig.legend(show=True, orient="vertical", left="right", top=50, bottom=10)
        opt = fig.to_option()
        assert opt["legend"]["orient"] == "vertical"
        assert opt["legend"]["left"] == "right"

    def test_tooltip_with_formatter(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.tooltip(trigger="item", pointer="shadow", formatter="{b}: {c}")
        opt = fig.to_option()
        assert opt["tooltip"]["formatter"] == "{b}: {c}"
        assert opt["tooltip"]["trigger"] == "item"

    def test_axis_pointer(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.axis_pointer(type="cross", snap=True, label=True,
                         label_precision=2, label_bg="#333", label_color="#fff",
                         line_color="#000", line_width=2, line_type="dashed",
                         cross_color="#red", cross_width=1, cross_type="dotted",
                         shadow_color="rgba(0,0,0,0.2)", shadow_opacity=0.5)
        opt = fig.to_option()
        assert "axisPointer" in opt

    def test_save_toolbox(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").save(name="test", fmt="svg", dpi=2)
        opt = fig.to_option()
        assert opt["toolbox"]["feature"]["saveAsImage"]["type"] == "svg"

    def test_toolbox_config(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").toolbox(download=True, zoom=True)
        opt = fig.to_option()
        assert "dataZoom" in opt["toolbox"]["feature"]

    def test_extra(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y").extra(animation=False)
        opt = fig.to_option()
        assert opt["animation"] is False

    def test_raw_series(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.raw_series({"type": "pie", "name": "overlay",
                        "data": [{"name": "A", "value": 10}],
                        "center": ["80%", "25%"], "radius": "20%"})
        opt = fig.to_option()
        assert len(opt["series"]) == 2
        assert opt["series"][1]["type"] == "pie"

    def test_visual_map_position_right(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.visual_map(position="right", colors=["#blue", "#red"])
        opt = fig.to_option()
        assert opt["visualMap"]["right"] == 10

    def test_visual_map_position_left(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.visual_map(position="left")
        opt = fig.to_option()
        assert opt["visualMap"]["left"] == 10


class TestPlotExtended:
    def test_plot_with_hue(self, hue_df):
        fig = ec.Figure()
        fig.plot(hue_df, x="X", y="Y", hue="Group")
        opt = fig.to_option()
        assert len(opt["series"]) == 2

    def test_plot_labels(self, simple_df):
        fig = ec.Figure()
        fig.plot(simple_df, x="X", y="Y", labels=True,
                 label_prefix="$", label_suffix="k")
        opt = fig.to_option()
        line = opt["series"][0]
        assert line["label"]["show"] is True

    def test_plot_axis_1(self, simple_df):
        fig = ec.Figure()
        fig.plot(simple_df, x="X", y="Y", axis=1)
        opt = fig.to_option()
        # axis=1 auto-creates second y-axis
        assert isinstance(opt["yAxis"], list)
        assert len(opt["yAxis"]) == 2

    def test_plot_dropped_rows_warns(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10.0, float("nan")]})
        fig = ec.Figure()
        with pytest.warns(UserWarning, match="1 rows dropped"):
            fig.plot(df, x="X", y="Y")


class TestBarExtended:
    def test_bar_gradient(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y", gradient=True)
        opt = fig.to_option()
        item = opt["series"][0]["itemStyle"]
        assert "colorStops" in item["color"]

    def test_bar_gradient_bad_colors(self, simple_df):
        fig = ec.Figure()
        with pytest.raises(ValueError, match="exactly 2"):
            fig.bar(simple_df, x="X", y="Y", gradient=True,
                    gradient_colors=("#a", "#b", "#c"))

    def test_bar_dropped_rows_warns(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10.0, float("nan")]})
        fig = ec.Figure()
        with pytest.warns(UserWarning, match="1 rows dropped"):
            fig.bar(df, x="X", y="Y")

    def test_bar_width_gap(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y", bar_width=40, bar_gap="20%")
        opt = fig.to_option()
        assert opt["series"][0]["barWidth"] == 40
        assert opt["series"][0]["barGap"] == "20%"

    def test_bar_emphasis(self, simple_df):
        from echartsy.emphasis import Emphasis
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y", emphasis=Emphasis(focus="series"))
        opt = fig.to_option()
        assert "emphasis" in opt["series"][0]


class TestScatterExtended:
    def test_scatter_with_color(self):
        df = pd.DataFrame({
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [10.0, 20.0, 30.0, 40.0],
            "G": ["a", "b", "a", "b"],
        })
        fig = ec.Figure()
        fig.scatter(df, x="X", y="Y", color="G")
        opt = fig.to_option()
        assert len(opt["series"]) == 2

    def test_scatter_with_size(self):
        df = pd.DataFrame({
            "X": [1.0, 2.0, 3.0],
            "Y": [10.0, 20.0, 30.0],
            "S": [5, 10, 15],
        })
        fig = ec.Figure()
        fig.scatter(df, x="X", y="Y", size="S")
        opt = fig.to_option()
        # symbolSize should be a list (varying sizes)
        assert isinstance(opt["series"][0]["symbolSize"], list)

    def test_scatter_with_labels(self):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
        fig = ec.Figure()
        fig.scatter(df, x="X", y="Y", labels=True)
        opt = fig.to_option()
        assert opt["series"][0]["label"]["show"] is True

    def test_scatter_dropped_warns(self):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [2.0, float("nan")]})
        fig = ec.Figure()
        with pytest.warns(UserWarning, match="1 rows dropped"):
            fig.scatter(df, x="X", y="Y")


class TestPieExtended:
    def test_pie_label_inside(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y", label_inside=True)
        opt = fig.to_option()
        assert opt["series"][0]["label"]["position"] == "inside"

    def test_pie_label_outside(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y", label_outside=True)
        opt = fig.to_option()
        assert opt["series"][0]["label"]["position"] == "outside"

    def test_pie_label_font_size(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y", label_inside=True,
                label_font_size=16)
        opt = fig.to_option()
        assert opt["series"][0]["label"]["fontSize"] == 16

    def test_pie_rose_type(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y", rose_type="radius")
        opt = fig.to_option()
        assert opt["series"][0]["roseType"] == "radius"

    def test_pie_border_radius(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y", border_radius=10)
        opt = fig.to_option()
        assert opt["series"][0]["itemStyle"]["borderRadius"] == 10

    def test_pie_center_on_hover(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y", center_on_hover=True)
        opt = fig.to_option()
        assert opt["series"][0]["emphasis"]["label"]["show"] is True

    def test_pie_overlay_on_bar(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.pie(simple_df, names="X", values="Y", center=["75%", "25%"])
        opt = fig.to_option()
        # Overlay pie doesn't change mode from cartesian
        assert any(s["type"] == "bar" for s in opt["series"])
        assert any(s["type"] == "pie" for s in opt["series"])

    def test_pie_explicit_radius(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y", radius="50%")
        opt = fig.to_option()
        assert opt["series"][0]["radius"] == "50%"

    def test_pie_auto_overlay(self, simple_df):
        """bar() then pie() without center auto-defaults to overlay."""
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.pie(simple_df, names="X", values="Y")
        opt = fig.to_option()
        assert any(s["type"] == "bar" for s in opt["series"])
        pie = [s for s in opt["series"] if s["type"] == "pie"][0]
        assert pie["center"] == ["82%", "25%"]
        assert pie["radius"] == ["15%", "28%"]

    def test_pie_auto_overlay_explicit_center(self, simple_df):
        """Explicit center overrides auto-default."""
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.pie(simple_df, names="X", values="Y", center=["50%", "50%"])
        opt = fig.to_option()
        pie = [s for s in opt["series"] if s["type"] == "pie"][0]
        assert pie["center"] == ["50%", "50%"]

    def test_pie_auto_overlay_explicit_radius(self, simple_df):
        """Explicit radius is used; center still gets default."""
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.pie(simple_df, names="X", values="Y", radius="40%")
        opt = fig.to_option()
        pie = [s for s in opt["series"] if s["type"] == "pie"][0]
        assert pie["radius"] == "40%"
        assert pie["center"] == ["82%", "25%"]

    def test_pie_link_legend_false(self, simple_df):
        """link_legend=False forces offset colors even with disjoint names."""
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        df2 = pd.DataFrame({"N": ["P", "Q"], "V": [5, 10]})
        fig.pie(df2, names="N", values="V", link_legend=False)
        opt = fig.to_option()
        pie = [s for s in opt["series"] if s["type"] == "pie"][0]
        # Offset applied → first slice gets color at index len(bar_names)
        assert "itemStyle" in pie["data"][0]
        assert "color" in pie["data"][0]["itemStyle"]

    def test_pie_link_legend_true(self, simple_df):
        """link_legend=True forces shared colors (no offset) even with overlap."""
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.pie(simple_df, names="X", values="Y", link_legend=True)
        opt = fig.to_option()
        pie = [s for s in opt["series"] if s["type"] == "pie"][0]
        # No color offset applied → data items should not have itemStyle.color
        for item in pie["data"]:
            assert "color" not in item.get("itemStyle", {})

    def test_pie_overlay_tooltip(self, simple_df):
        """Overlay pie gets a per-series item tooltip."""
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.pie(simple_df, names="X", values="Y")
        opt = fig.to_option()
        pie = [s for s in opt["series"] if s["type"] == "pie"][0]
        assert pie["tooltip"]["trigger"] == "item"

    def test_pie_overlay_non_cartesian_error(self):
        """pie() on a radar figure raises a clear error."""
        fig = ec.Figure()
        fig.radar(
            indicators=[{"name": "A", "max": 10}, {"name": "B", "max": 10}],
            data=[[5, 8]],
        )
        with pytest.raises(BuilderConfigError, match="only supported on cartesian"):
            fig.pie(
                pd.DataFrame({"N": ["X"], "V": [1]}),
                names="N", values="V",
            )


class TestHistExtended:
    def test_hist_with_color(self):
        df = pd.DataFrame({"V": list(range(20))})
        fig = ec.Figure()
        fig.hist(df, column="V", bar_color="#ff0000")
        opt = fig.to_option()
        assert opt["series"][0]["itemStyle"]["color"] == "#ff0000"

    def test_hist_density(self):
        df = pd.DataFrame({"V": list(range(20))})
        fig = ec.Figure()
        fig.hist(df, column="V", density=True)
        opt = fig.to_option()
        y = opt["yAxis"] if isinstance(opt["yAxis"], dict) else opt["yAxis"][0]
        assert y["name"] == "Density"

    def test_hist_bad_bins(self):
        df = pd.DataFrame({"V": [1, 2, 3]})
        fig = ec.Figure()
        with pytest.raises(ValueError, match="positive"):
            fig.hist(df, column="V", bins=0)

    def test_hist_emphasis(self):
        from echartsy.emphasis import Emphasis
        df = pd.DataFrame({"V": list(range(10))})
        fig = ec.Figure()
        fig.hist(df, column="V", emphasis=Emphasis(focus="series"))
        opt = fig.to_option()
        assert "emphasis" in opt["series"][0]


class TestHeatmapExtended:
    def test_heatmap_dropped_warns(self):
        df = pd.DataFrame({
            "X": ["A", "B"], "Y": ["C", "D"], "V": [10, float("nan")]
        })
        fig = ec.Figure()
        with pytest.warns(UserWarning, match="1 rows dropped"):
            fig.heatmap(df, x="X", y="Y", value="V")

    def test_heatmap_all_nan_value_raises(self):
        from echartsy.exceptions import DataValidationError
        df = pd.DataFrame({
            "X": ["A"], "Y": ["B"], "V": ["abc"]
        })
        fig = ec.Figure()
        with pytest.raises(DataValidationError, match="no numeric"):
            fig.heatmap(df, x="X", y="Y", value="V")


class TestHeatmapPiecewise:
    def test_piecewise_visual_map(self):
        df = pd.DataFrame({
            "X": ["A", "B", "C"], "Y": ["D", "E", "F"], "V": [1, 5, 10]
        })
        pieces = [
            {"min": 0, "max": 3, "label": "Low", "color": "#00ff00"},
            {"min": 3, "max": 7, "label": "Mid", "color": "#ffff00"},
            {"min": 7, "max": 10, "label": "High", "color": "#ff0000"},
        ]
        fig = ec.Figure()
        fig.heatmap(df, x="X", y="Y", value="V",
                    visual_map_type="piecewise", pieces=pieces)
        opt = fig.to_option()
        vm = opt["visualMap"]
        assert vm["type"] == "piecewise"
        assert len(vm["pieces"]) == 3

    def test_continuous_default_unchanged(self):
        df = pd.DataFrame({
            "X": ["A", "B"], "Y": ["C", "D"], "V": [10, 20]
        })
        fig = ec.Figure()
        fig.heatmap(df, x="X", y="Y", value="V")
        opt = fig.to_option()
        vm = opt["visualMap"]
        assert vm.get("type", "continuous") != "piecewise"
        assert "min" in vm and "max" in vm

    def test_progressive_rendering(self):
        df = pd.DataFrame({
            "X": ["A", "B"], "Y": ["C", "D"], "V": [10, 20]
        })
        fig = ec.Figure()
        fig.heatmap(df, x="X", y="Y", value="V",
                    progressive=500, progressive_threshold=2000)
        opt = fig.to_option()
        s = opt["series"][0]
        assert s["progressive"] == 500
        assert s["progressiveThreshold"] == 2000


class TestFunnelExtended:
    def test_funnel_emphasis(self, simple_df):
        from echartsy.emphasis import FunnelEmphasis
        fig = ec.Figure()
        fig.funnel(simple_df, names="X", values="Y",
                   emphasis=FunnelEmphasis(focus="series"))
        opt = fig.to_option()
        assert "emphasis" in opt["series"][0]


class TestSankeyExtended:
    def test_sankey_emphasis_dict_deprecation(self, sankey_df):
        fig = ec.Figure()
        with pytest.warns(DeprecationWarning, match="deprecated"):
            fig.sankey(sankey_df, levels=["Level1", "Level2"], value="Value",
                       emphasis={"focus": "adjacency"})

    def test_sankey_too_few_levels(self, sankey_df):
        fig = ec.Figure()
        with pytest.raises(BuilderConfigError, match="at least 2"):
            fig.sankey(sankey_df, levels=["Level1"], value="Value")


class TestTreemapExtended:
    def test_treemap_no_value(self, sankey_df):
        fig = ec.Figure()
        fig.treemap(sankey_df, path=["Level1", "Level2"])
        opt = fig.to_option()
        assert any(s["type"] == "treemap" for s in opt["series"])


class TestSummaryExtended:
    def test_summary_with_dual_axis(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.ylabel("Left Axis")
        fig.ylabel_right("Right Axis")
        fig.plot(simple_df, x="X", y="Y", axis=1)
        fig.title("Test")
        result = fig.summary()
        assert "Left Axis" in result or "Right Axis" in result

    def test_summary_with_named_series(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        result = fig.summary()
        assert "bar" in result


class TestYticks:
    def test_yticks_basic(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        fig.yticks(rotate=15, interval=5, formatter="{value}K")
        opt = fig.to_option()
        y = opt["yAxis"] if isinstance(opt["yAxis"], dict) else opt["yAxis"][0]
        assert y["axisLabel"]["rotate"] == 15
        assert y["axisLabel"]["interval"] == 5
        assert y["axisLabel"]["formatter"] == "{value}K"

    def test_yticks_invalid_axis(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        with pytest.raises(BuilderConfigError, match="axis does not exist"):
            fig.yticks(axis=5)


_has_scipy = True
try:
    import scipy  # noqa: F401
except ImportError:
    _has_scipy = False


@pytest.mark.skipif(not _has_scipy, reason="scipy not installed")
class TestKde:
    def test_kde_basic(self):
        np.random.seed(42)
        df = pd.DataFrame({"V": np.random.normal(0, 1, 100)})
        fig = ec.Figure()
        fig.kde(df, column="V")
        opt = fig.to_option()
        assert any(s["type"] == "line" for s in opt["series"])
        assert opt["xAxis"]["type"] == "value"

    def test_kde_with_hue(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "V": np.random.normal(0, 1, 60),
            "G": ["A"] * 30 + ["B"] * 30,
        })
        fig = ec.Figure()
        fig.kde(df, column="V", hue="G")
        opt = fig.to_option()
        assert len(opt["series"]) == 2

    def test_kde_too_few_values(self):
        from echartsy.exceptions import DataValidationError
        df = pd.DataFrame({"V": [1.0]})
        fig = ec.Figure()
        with pytest.raises(DataValidationError, match="at least 2"):
            fig.kde(df, column="V")


class TestErrorPaths:
    def test_xlim_on_pie_raises(self, simple_df):
        fig = ec.Figure()
        fig.pie(simple_df, names="X", values="Y")
        with pytest.raises(BuilderConfigError, match="not applicable"):
            fig.xlim(min_val=0)

    def test_ylim_negative_axis(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        with pytest.raises(ValueError, match="non-negative"):
            fig.ylim(axis=-1)

    def test_ylim_nonexistent_axis(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        with pytest.raises(BuilderConfigError, match="only 1 y-axis"):
            fig.ylim(axis=2)

    def test_datazoom_bad_range(self, simple_df):
        fig = ec.Figure()
        fig.bar(simple_df, x="X", y="Y")
        with pytest.raises(ValueError, match="between 0 and 100"):
            fig.datazoom(start=-10, end=200)

    def test_radar_empty_indicators(self):
        fig = ec.Figure()
        with pytest.raises(ec.DataValidationError, match="must not be empty"):
            fig.radar([], [[1, 2]])

    def test_radar_empty_data(self):
        fig = ec.Figure()
        with pytest.raises(ec.DataValidationError, match="must not be empty"):
            fig.radar([{"name": "A", "max": 100}], [])

    def test_radar_data_length_mismatch(self):
        fig = ec.Figure()
        with pytest.raises(ec.DataValidationError, match="values but there are"):
            fig.radar([{"name": "A", "max": 100}], [[1, 2, 3]])


class TestCalendarPie:
    @pytest.fixture
    def cal_pie_df(self):
        return pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
            "Category": ["A", "B", "A", "B", "A"],
            "Value": [30, 70, 40, 60, 100],
        })

    def test_calendar_pie_basic(self, cal_pie_df):
        fig = ec.Figure()
        fig.calendar_pie(cal_pie_df, date="Date", names="Category", values="Value")
        opt = fig.to_option()
        assert "calendar" in opt
        pie_series = [s for s in opt["series"] if s["type"] == "pie"]
        assert len(pie_series) == 3  # 3 unique dates

    def test_calendar_pie_coordinate_system(self, cal_pie_df):
        fig = ec.Figure()
        fig.calendar_pie(cal_pie_df, date="Date", names="Category", values="Value")
        opt = fig.to_option()
        for s in opt["series"]:
            assert s["coordinateSystem"] == "calendar"
            assert s["calendarIndex"] == 0

    def test_calendar_pie_has_calendar_component(self, cal_pie_df):
        fig = ec.Figure()
        fig.calendar_pie(cal_pie_df, date="Date", names="Category", values="Value")
        opt = fig.to_option()
        assert "calendar" in opt
        assert opt["calendar"]["range"] == "2024"

    def test_calendar_pie_mode(self, cal_pie_df):
        fig = ec.Figure()
        fig.calendar_pie(cal_pie_df, date="Date", names="Category", values="Value")
        assert fig._chart_mode == "calendar"

    def test_calendar_pie_radius(self, cal_pie_df):
        fig = ec.Figure()
        fig.calendar_pie(cal_pie_df, date="Date", names="Category", values="Value",
                         pie_radius=20)
        opt = fig.to_option()
        assert opt["series"][0]["radius"] == 20


class TestParallel:
    @pytest.fixture
    def parallel_df(self):
        return pd.DataFrame({
            "Speed": [120, 130, 140, 150],
            "Weight": [2.5, 3.0, 2.8, 3.2],
            "Safety": [8, 7, 9, 6],
            "Type": ["SUV", "Sedan", "SUV", "Truck"],
        })

    def test_parallel_basic(self, parallel_df):
        fig = ec.Figure()
        fig.parallel(parallel_df, dimensions=["Speed", "Weight", "Safety"])
        opt = fig.to_option()
        assert opt["series"][0]["type"] == "parallel"
        assert len(opt["parallelAxis"]) == 3
        assert "parallel" in opt
        assert len(opt["series"][0]["data"]) == 4

    def test_parallel_mixed_dims(self, parallel_df):
        fig = ec.Figure()
        fig.parallel(parallel_df, dimensions=["Speed", "Type", "Safety"])
        opt = fig.to_option()
        axes = opt["parallelAxis"]
        assert axes[0]["type"] == "value"
        assert axes[1]["type"] == "category"
        assert "data" in axes[1]
        assert axes[2]["type"] == "value"

    def test_parallel_color_dim(self, parallel_df):
        fig = ec.Figure()
        fig.parallel(parallel_df, dimensions=["Speed", "Weight", "Safety"],
                     color_dim="Safety")
        opt = fig.to_option()
        assert "visualMap" in opt
        assert opt["visualMap"]["dimension"] == 2

    def test_parallel_mode_set(self, parallel_df):
        fig = ec.Figure()
        fig.parallel(parallel_df, dimensions=["Speed", "Weight"])
        assert fig._chart_mode == "parallel"

    def test_parallel_mode_conflict(self, parallel_df, simple_df):
        fig = ec.Figure()
        fig.parallel(parallel_df, dimensions=["Speed", "Weight"])
        with pytest.raises(BuilderConfigError, match="mode"):
            fig.bar(simple_df, x="X", y="Y")

    def test_parallel_to_option_structure(self, parallel_df):
        fig = ec.Figure()
        fig.parallel(parallel_df, dimensions=["Speed", "Weight", "Safety"])
        fig.title("Test")
        opt = fig.to_option()
        assert "title" in opt
        assert "parallelAxis" in opt
        assert "parallel" in opt
        assert "series" in opt
        assert "tooltip" in opt

    def test_parallel_axis_expand(self, parallel_df):
        fig = ec.Figure()
        fig.parallel(parallel_df, dimensions=["Speed", "Weight", "Safety"],
                     axis_expand=True, axis_expand_count=2)
        opt = fig.to_option()
        assert opt["parallel"]["axisExpandable"] is True
        assert opt["parallel"]["axisExpandCount"] == 2

    def test_parallel_smooth(self, parallel_df):
        fig = ec.Figure()
        fig.parallel(parallel_df, dimensions=["Speed", "Weight"],
                     smooth=True, line_opacity=0.5)
        opt = fig.to_option()
        s = opt["series"][0]
        assert s["smooth"] is True
        assert s["lineStyle"]["opacity"] == 0.5

    def test_parallel_emphasis(self, parallel_df):
        fig = ec.Figure()
        fig.parallel(parallel_df, dimensions=["Speed", "Weight"],
                     emphasis=ec.Emphasis(focus="self"))
        opt = fig.to_option()
        assert opt["series"][0]["emphasis"]["focus"] == "self"


class TestFigureFactory:
    def test_figure_factory(self):
        fig = ec.figure(height="600px", renderer="canvas")
        assert isinstance(fig, ec.Figure)

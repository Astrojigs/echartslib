"""Tests for _chart_methods.py shared builder functions."""
from __future__ import annotations

import pandas as pd
import pytest

from echartsy._chart_methods import (
    build_bar_series, build_line_series, build_pie_series, build_scatter_series,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "X": ["A", "B", "C", "A", "B", "C"],
        "Y": [10, 20, 30, 15, 25, 35],
    })


@pytest.fixture
def hue_df():
    return pd.DataFrame({
        "X": ["A", "A", "B", "B"],
        "Y": [10, 20, 30, 40],
        "G": ["g1", "g2", "g1", "g2"],
    })


class TestBuildLineSeries:
    def test_basic(self, df):
        series, legend, cats = build_line_series(df, "X", "Y")
        assert len(series) == 1
        assert series[0]["type"] == "line"
        assert set(cats) == {"A", "B", "C"}
        assert len(legend) == 1

    def test_smooth(self, df):
        series, _, _ = build_line_series(df, "X", "Y", smooth=True)
        assert series[0]["smooth"] is True

    def test_area(self, df):
        series, _, _ = build_line_series(df, "X", "Y", area=True, area_opacity=0.3)
        assert "areaStyle" in series[0]
        assert series[0]["areaStyle"]["opacity"] == 0.3

    def test_labels(self, df):
        series, _, _ = build_line_series(df, "X", "Y", labels=True,
                                         label_prefix="$", label_suffix="k")
        assert series[0]["label"]["show"] is True
        assert "$" in series[0]["label"]["formatter"]
        assert "k" in series[0]["label"]["formatter"]

    def test_hue(self, hue_df):
        series, legend, cats = build_line_series(hue_df, "X", "Y", hue="G")
        assert len(series) == 2
        assert set(legend) == {"g1", "g2"}

    def test_connect_nulls(self, df):
        series, _, _ = build_line_series(df, "X", "Y", connect_nulls=True)
        assert series[0]["connectNulls"] is True

    def test_all_nan_raises(self):
        from echartsy.exceptions import DataValidationError
        df = pd.DataFrame({"X": ["A"], "Y": ["abc"]})
        with pytest.raises(DataValidationError, match="no numeric"):
            build_line_series(df, "X", "Y")

    def test_drops_warning(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10.0, float("nan")]})
        with pytest.warns(UserWarning, match="1 rows dropped"):
            build_line_series(df, "X", "Y")

    def test_existing_categories(self, df):
        series, _, cats = build_line_series(df, "X", "Y", categories=["Z"])
        assert cats[0] == "Z"
        assert "A" in cats

    def test_emphasis(self, df):
        from echartsy.emphasis import LineEmphasis
        emp = LineEmphasis(focus="series")
        series, _, _ = build_line_series(df, "X", "Y", emphasis=emp)
        assert "emphasis" in series[0]

    def test_series_kw(self, df):
        series, _, _ = build_line_series(df, "X", "Y", markLine={"data": []})
        assert "markLine" in series[0]


class TestBuildBarSeries:
    def test_basic(self, df):
        series, legend, cats, is_h = build_bar_series(df, "X", "Y")
        assert len(series) == 1
        assert series[0]["type"] == "bar"
        assert is_h is False

    def test_horizontal(self, df):
        series, _, _, is_h = build_bar_series(df, "X", "Y", orient="h")
        assert is_h is True

    def test_stack(self, hue_df):
        series, _, _, _ = build_bar_series(hue_df, "X", "Y", hue="G", stack=True)
        for s in series:
            assert s["stack"] == "total"

    def test_gradient(self, df):
        series, _, _, _ = build_bar_series(df, "X", "Y", gradient=True,
                                            gradient_colors=("#aaa", "#bbb"))
        assert "colorStops" in series[0]["itemStyle"]["color"]

    def test_gradient_bad_colors(self, df):
        with pytest.raises(ValueError, match="exactly 2"):
            build_bar_series(df, "X", "Y", gradient=True,
                             gradient_colors=("#a", "#b", "#c"))

    def test_labels(self, df):
        series, _, _, _ = build_bar_series(df, "X", "Y", labels=True)
        assert series[0]["label"]["show"] is True

    def test_bar_width_and_gap(self, df):
        series, _, _, _ = build_bar_series(df, "X", "Y", bar_width=40, bar_gap="20%")
        assert series[0]["barWidth"] == 40
        assert series[0]["barGap"] == "20%"

    def test_hue(self, hue_df):
        series, legend, _, _ = build_bar_series(hue_df, "X", "Y", hue="G")
        assert len(series) == 2
        assert set(legend) == {"g1", "g2"}

    def test_all_nan_raises(self):
        from echartsy.exceptions import DataValidationError
        df = pd.DataFrame({"X": ["A"], "Y": ["abc"]})
        with pytest.raises(DataValidationError, match="no numeric"):
            build_bar_series(df, "X", "Y")

    def test_emphasis(self, df):
        from echartsy.emphasis import Emphasis
        emp = Emphasis(focus="series")
        series, _, _, _ = build_bar_series(df, "X", "Y", emphasis=emp)
        assert "emphasis" in series[0]


# ── Scatter builder ──────────────────────────────────────────────────────


@pytest.fixture
def scatter_df():
    return pd.DataFrame({
        "X": [1.0, 2.0, 3.0, 4.0],
        "Y": [10.0, 20.0, 30.0, 40.0],
        "G": ["a", "a", "b", "b"],
        "S": [5.0, 10.0, 15.0, 20.0],
    })


class TestBuildScatterSeries:
    def test_basic(self, scatter_df):
        series, legend = build_scatter_series(scatter_df, "X", "Y")
        assert len(series) == 1
        assert series[0]["type"] == "scatter"
        assert len(series[0]["data"]) == 4
        assert legend == ["scatter"]

    def test_color_grouping(self, scatter_df):
        series, legend = build_scatter_series(scatter_df, "X", "Y", color="G")
        assert len(series) == 2
        assert set(legend) == {"a", "b"}

    def test_bubble_size(self, scatter_df):
        series, _ = build_scatter_series(scatter_df, "X", "Y", size="S")
        assert "symbolSize" in series[0]

    def test_labels(self, scatter_df):
        series, _ = build_scatter_series(scatter_df, "X", "Y", labels=True)
        assert series[0]["label"]["show"] is True

    def test_emphasis(self, scatter_df):
        from echartsy.emphasis import ScatterEmphasis
        emp = ScatterEmphasis(focus="series", scale=True)
        series, _ = build_scatter_series(scatter_df, "X", "Y", emphasis=emp)
        assert "emphasis" in series[0]

    def test_symbol_rotate(self, scatter_df):
        series, _ = build_scatter_series(scatter_df, "X", "Y", symbol_rotate=45)
        assert series[0]["symbolRotate"] == 45

    def test_all_nan_raises(self):
        from echartsy.exceptions import DataValidationError
        df = pd.DataFrame({"X": ["abc"], "Y": [1.0]})
        with pytest.raises(DataValidationError, match="no numeric"):
            build_scatter_series(df, "X", "Y")


# ── Pie builder ──────────────────────────────────────────────────────────


@pytest.fixture
def pie_df():
    return pd.DataFrame({
        "Name": ["A", "B", "C", "A"],
        "Val": [10, 20, 30, 5],
    })


class TestBuildPieSeries:
    def test_basic(self, pie_df):
        entry, legend = build_pie_series(pie_df, "Name", "Val")
        assert entry["type"] == "pie"
        assert len(entry["data"]) == 3  # A aggregated
        assert set(legend) == {"A", "B", "C"}

    def test_aggregation(self, pie_df):
        entry, _ = build_pie_series(pie_df, "Name", "Val", agg="sum")
        vals = {d["name"]: d["value"] for d in entry["data"]}
        assert vals["A"] == 15.0  # 10 + 5

    def test_labels_inside(self, pie_df):
        entry, _ = build_pie_series(pie_df, "Name", "Val", label_inside=True)
        assert entry["label"]["position"] == "inside"

    def test_labels_outside(self, pie_df):
        entry, _ = build_pie_series(pie_df, "Name", "Val", label_outside=True)
        assert entry["label"]["show"] is True
        assert entry["labelLine"]["show"] is True

    def test_rose_type(self, pie_df):
        entry, _ = build_pie_series(pie_df, "Name", "Val", rose_type="radius")
        assert entry["roseType"] == "radius"

    def test_border_radius(self, pie_df):
        entry, _ = build_pie_series(pie_df, "Name", "Val", border_radius=10)
        assert entry["itemStyle"]["borderRadius"] == 10

    def test_emphasis(self, pie_df):
        from echartsy.emphasis import PieEmphasis
        emp = PieEmphasis(scale=True, scale_size=10)
        entry, _ = build_pie_series(pie_df, "Name", "Val", emphasis=emp)
        assert "emphasis" in entry

    def test_label_font_size(self, pie_df):
        entry, _ = build_pie_series(pie_df, "Name", "Val", label_font_size=20)
        assert entry["label"]["fontSize"] == 20

    def test_empty_df(self):
        df = pd.DataFrame({"Name": [None], "Val": [10.0]})
        with pytest.warns(UserWarning, match="all rows dropped"):
            entry, legend = build_pie_series(df, "Name", "Val")
        assert entry["data"] == []
        assert legend == []


class TestBuildScatterSeriesEmpty:
    def test_drops_nan_rows(self):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [10.0, float("nan")]})
        with pytest.warns(UserWarning, match="1 rows dropped"):
            series, legend = build_scatter_series(df, "X", "Y")
        assert len(series) == 1
        assert len(series[0]["data"]) == 1  # only valid row kept

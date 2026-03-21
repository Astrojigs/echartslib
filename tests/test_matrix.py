"""Tests for MatrixFigure chart grid."""
from __future__ import annotations

import pandas as pd
import pytest

import echartsy as ec
from echartsy.matrix import MatrixFigure, _CellConfig


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def grid_df():
    """Sample data with row/col grouping columns."""
    return pd.DataFrame({
        "Region": ["East", "East", "West", "West", "East", "West"],
        "Quarter": ["Q1", "Q2", "Q1", "Q2", "Q1", "Q2"],
        "Product": ["A", "A", "A", "A", "B", "B"],
        "Sales": [100, 150, 200, 250, 80, 120],
    })


@pytest.fixture
def numeric_df():
    """DataFrame with multiple numeric columns for pairplot."""
    return pd.DataFrame({
        "Height": [170, 165, 180, 175, 160],
        "Weight": [70, 60, 85, 75, 55],
        "Age": [30, 25, 35, 28, 22],
        "Group": ["A", "B", "A", "B", "A"],
    })


# ── Basic creation ────────────────────────────────────────────────────────


class TestMatrixCreation:
    def test_basic_creation(self):
        mat = MatrixFigure(rows=["R1", "R2"], cols=["C1", "C2"])
        assert len(mat._rows) == 2
        assert len(mat._cols) == 2
        assert len(mat._cells) == 0

    def test_repr(self):
        mat = MatrixFigure(rows=["R1"], cols=["C1", "C2"])
        r = repr(mat)
        assert "rows=1" in r
        assert "cols=2" in r

    def test_title_chaining(self):
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        result = mat.title("Test Title")
        assert result is mat
        assert mat._title_cfg["text"] == "Test Title"

    def test_palette_chaining(self):
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        result = mat.palette(["#f00", "#0f0"])
        assert result is mat
        assert mat._palette == ["#f00", "#0f0"]


# ── Cell builder ──────────────────────────────────────────────────────────


class TestCellBuilder:
    def test_cell_bar(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        result = mat.cell(0, 0).bar(df, "X", "Y")
        assert result is mat
        cell = mat._cells[(0, 0)]
        assert cell.chart_type == "bar"
        assert len(cell.series_configs) == 1
        assert cell.series_configs[0]["type"] == "bar"

    def test_cell_plot(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).plot(df, "X", "Y", smooth=True)
        cell = mat._cells[(0, 0)]
        assert cell.chart_type == "line"
        assert cell.series_configs[0]["smooth"] is True

    def test_cell_scatter(self):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).scatter(df, "X", "Y")
        cell = mat._cells[(0, 0)]
        assert cell.chart_type == "scatter"
        assert len(cell.series_configs[0]["data"]) == 2

    def test_cell_pie(self):
        df = pd.DataFrame({"Name": ["A", "B"], "Val": [30, 70]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).pie(df, "Name", "Val")
        cell = mat._cells[(0, 0)]
        assert cell.chart_type == "pie"
        assert cell.show_axis is False

    def test_cell_heatmap(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": ["C", "D"], "V": [1, 2]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).heatmap(df, "X", "Y", "V")
        cell = mat._cells[(0, 0)]
        assert cell.chart_type == "heatmap"

    def test_cell_bar_styled(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).bar(df, "X", "Y", labels=True, color="#ff0000",
                           emphasis=ec.Emphasis(focus="self"),
                           item_style=ec.ItemStyle(border_width=2),
                           label_style=ec.LabelStyle(font_size=14),
                           tooltip=ec.TooltipStyle(formatter="{b}"))
        s = mat._cells[(0, 0)].series_configs[0]
        assert s["label"]["show"] is True
        assert s["label"]["fontSize"] == 14
        assert s["itemStyle"]["color"] == "#ff0000"
        assert s["itemStyle"]["borderWidth"] == 2
        assert s["emphasis"]["focus"] == "self"
        assert s["tooltip"]["formatter"] == "{b}"

    def test_cell_plot_area(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).plot(df, "X", "Y", area=True, area_opacity=0.5,
                            line_style=ec.LineStyle(width=3))
        s = mat._cells[(0, 0)].series_configs[0]
        assert s["areaStyle"]["opacity"] == 0.5
        assert s["lineStyle"]["width"] == 3

    def test_cell_scatter_styled(self):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).scatter(df, "X", "Y",
                               item_style=ec.ItemStyle(color="#00ff00"),
                               emphasis=ec.ScatterEmphasis(focus="series"))
        s = mat._cells[(0, 0)].series_configs[0]
        assert s["itemStyle"]["color"] == "#00ff00"
        assert s["emphasis"]["focus"] == "series"

    def test_cell_pie_labels(self):
        df = pd.DataFrame({"Name": ["A", "B"], "Val": [30, 70]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).pie(df, "Name", "Val", labels=True,
                           label_style=ec.LabelStyle(font_size=10))
        s = mat._cells[(0, 0)].series_configs[0]
        assert s["label"]["show"] is True
        assert s["label"]["fontSize"] == 10

    def test_cell_text(self):
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).text("Hello")
        cell = mat._cells[(0, 0)]
        assert cell.chart_type == "text"
        assert cell.show_axis is False

    def test_cell_out_of_range(self):
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        with pytest.raises(IndexError):
            mat.cell(1, 0)
        with pytest.raises(IndexError):
            mat.cell(0, 1)


# ── Auto-populate ─────────────────────────────────────────────────────────


class TestAutoPopulate:
    def test_bar_group_row(self, grid_df):
        mat = MatrixFigure(rows=["East", "West"], cols=["Q1", "Q2"])
        mat.bar(grid_df, x="Product", y="Sales",
                group_row="Region", group_col="Quarter")
        assert len(mat._cells) > 0
        opt = mat.to_option()
        bar_series = [s for s in opt["series"] if s["type"] == "bar"]
        assert len(bar_series) > 0

    def test_plot_group(self, grid_df):
        mat = MatrixFigure(rows=["East", "West"], cols=["Q1"])
        result = mat.plot(grid_df, x="Product", y="Sales", group_row="Region")
        assert result is mat

    def test_scatter_group(self, grid_df):
        mat = MatrixFigure(rows=["East", "West"], cols=["all"])
        mat.scatter(grid_df, x="Sales", y="Sales", group_row="Region")
        assert len(mat._cells) > 0

    def test_pie_group(self, grid_df):
        mat = MatrixFigure(rows=["East", "West"], cols=["Q1", "Q2"])
        mat.pie(grid_df, names="Product", values="Sales",
                group_row="Region", group_col="Quarter")
        opt = mat.to_option()
        pie_series = [s for s in opt["series"] if s["type"] == "pie"]
        assert len(pie_series) > 0

    def test_sparkline(self, grid_df):
        mat = MatrixFigure(rows=["East", "West"], cols=["all"])
        mat.sparkline(grid_df, x="Quarter", y="Sales", group_row="Region")
        for key, cell in mat._cells.items():
            assert cell.show_axis is False
            assert cell.series_configs[0]["showSymbol"] is False


# ── Pairplot ──────────────────────────────────────────────────────────────


class TestPairplot:
    def test_pairplot_basic(self, numeric_df):
        mat = MatrixFigure.pairplot(numeric_df, columns=["Height", "Weight", "Age"])
        assert len(mat._rows) == 3
        assert len(mat._cols) == 3
        opt = mat.to_option()
        assert len(opt["grid"]) == 9
        assert len(opt["xAxis"]) == 9
        assert len(opt["yAxis"]) == 9

    def test_pairplot_with_hue(self, numeric_df):
        mat = MatrixFigure.pairplot(
            numeric_df, columns=["Height", "Weight"], hue="Group"
        )
        opt = mat.to_option()
        # Off-diagonal should have multiple scatter series (one per group)
        scatter_series = [s for s in opt["series"] if s["type"] == "scatter"]
        assert len(scatter_series) >= 2

    def test_pairplot_diag_label(self, numeric_df):
        mat = MatrixFigure.pairplot(
            numeric_df, columns=["Height", "Weight"], diag="label"
        )
        opt = mat.to_option()
        assert len(opt["series"]) > 0


# ── to_option structure ───────────────────────────────────────────────────


class TestToOption:
    def test_option_has_grids_axes_series(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1", "R2"], cols=["C1"])
        mat.cell(0, 0).bar(df, "X", "Y")
        mat.cell(1, 0).bar(df, "X", "Y")
        opt = mat.to_option()
        assert "grid" in opt
        assert "xAxis" in opt
        assert "yAxis" in opt
        assert "series" in opt
        assert len(opt["grid"]) == 2  # 2 rows x 1 col
        assert len(opt["xAxis"]) == 2
        assert len(opt["yAxis"]) == 2

    def test_pie_gets_center_not_axis(self):
        df = pd.DataFrame({"Name": ["A", "B"], "Val": [30, 70]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).pie(df, "Name", "Val")
        opt = mat.to_option()
        pie = [s for s in opt["series"] if s["type"] == "pie"][0]
        assert "center" in pie
        assert "xAxisIndex" not in pie

    def test_title_in_option(self):
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.title("My Matrix")
        opt = mat.to_option()
        assert opt["title"]["text"] == "My Matrix"

    def test_palette_in_option(self):
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.palette(["#abc", "#def"])
        opt = mat.to_option()
        assert opt["color"] == ["#abc", "#def"]

    def test_empty_cells_no_series(self):
        mat = MatrixFigure(rows=["R1"], cols=["C1", "C2"])
        df = pd.DataFrame({"X": ["A"], "Y": [10]})
        mat.cell(0, 0).bar(df, "X", "Y")
        opt = mat.to_option()
        # Only 1 series even though 2 grid cells
        assert len(opt["series"]) == 1
        assert len(opt["grid"]) == 2


# ── Cell builder parity with Figure ─────────────────────────────────────


class TestCellBuilderParity:
    def test_bar_hue_stack(self):
        df = pd.DataFrame({
            "X": ["A", "A", "B", "B"],
            "Y": [10, 20, 30, 40],
            "G": ["g1", "g2", "g1", "g2"],
        })
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).bar(df, "X", "Y", hue="G", stack=True)
        cell = mat._cells[(0, 0)]
        assert len(cell.series_configs) == 2
        assert all(s["stack"] == "total" for s in cell.series_configs)
        assert set(cell.legend_names) == {"g1", "g2"}

    def test_bar_gradient(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).bar(df, "X", "Y", gradient=True,
                           gradient_colors=("#aaa", "#bbb"))
        s = mat._cells[(0, 0)].series_configs[0]
        assert "colorStops" in s["itemStyle"]["color"]

    def test_bar_agg(self):
        df = pd.DataFrame({"X": ["A", "A"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).bar(df, "X", "Y", agg="mean")
        s = mat._cells[(0, 0)].series_configs[0]
        assert s["data"][0] == 15.0

    def test_plot_hue(self):
        df = pd.DataFrame({
            "X": ["A", "A", "B", "B"],
            "Y": [10, 20, 30, 40],
            "G": ["g1", "g2", "g1", "g2"],
        })
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).plot(df, "X", "Y", hue="G")
        cell = mat._cells[(0, 0)]
        assert len(cell.series_configs) == 2
        assert set(cell.legend_names) == {"g1", "g2"}

    def test_scatter_color_grouping(self):
        df = pd.DataFrame({
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [10.0, 20.0, 30.0, 40.0],
            "G": ["a", "a", "b", "b"],
        })
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).scatter(df, "X", "Y", color="G")
        cell = mat._cells[(0, 0)]
        assert len(cell.series_configs) == 2
        assert set(cell.legend_names) == {"a", "b"}

    def test_pie_rose_type_agg(self):
        df = pd.DataFrame({"Name": ["A", "A", "B"], "Val": [10, 5, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).pie(df, "Name", "Val", rose_type="radius", agg="sum")
        s = mat._cells[(0, 0)].series_configs[0]
        assert s["roseType"] == "radius"
        vals = {d["name"]: d["value"] for d in s["data"]}
        assert vals["A"] == 15.0

    def test_legend_emission_from_hue(self):
        df = pd.DataFrame({
            "X": ["A", "A"], "Y": [10, 20], "G": ["g1", "g2"],
        })
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).bar(df, "X", "Y", hue="G")
        opt = mat.to_option()
        assert "legend" in opt
        assert set(opt["legend"]["data"]) == {"g1", "g2"}

    def test_no_legend_without_hue(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).bar(df, "X", "Y")
        opt = mat.to_option()
        # No legend when there's no hue grouping
        assert "legend" not in opt

    def test_backward_compat_simple_bar(self):
        """Simple bar call still works the same as before."""
        df = pd.DataFrame({"X": ["A", "B", "C"], "Y": [10, 20, 30]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).bar(df, "X", "Y")
        cell = mat._cells[(0, 0)]
        assert cell.chart_type == "bar"
        assert len(cell.series_configs) == 1
        assert cell.series_configs[0]["type"] == "bar"
        assert cell.x_axis_config["data"] == ["A", "B", "C"]

    def test_backward_compat_simple_plot(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).plot(df, "X", "Y", smooth=True)
        cell = mat._cells[(0, 0)]
        assert cell.chart_type == "line"
        assert cell.series_configs[0]["smooth"] is True

    def test_backward_compat_simple_scatter(self):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).scatter(df, "X", "Y")
        cell = mat._cells[(0, 0)]
        assert cell.chart_type == "scatter"
        assert len(cell.series_configs[0]["data"]) == 2

    def test_backward_compat_simple_pie(self):
        df = pd.DataFrame({"Name": ["A", "B"], "Val": [30, 70]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).pie(df, "Name", "Val")
        cell = mat._cells[(0, 0)]
        assert cell.chart_type == "pie"
        assert cell.show_axis is False

    def test_pie_user_center_radius_preserved(self):
        df = pd.DataFrame({"Name": ["A", "B"], "Val": [30, 70]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).pie(df, "Name", "Val",
                           center=["50%", "50%"], radius=["20%", "40%"])
        opt = mat.to_option()
        pie = [s for s in opt["series"] if s["type"] == "pie"][0]
        assert pie["center"] == ["50%", "50%"]
        assert pie["radius"] == ["20%", "40%"]

    def test_bar_with_blur_select(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).bar(df, "X", "Y",
                           blur=ec.Blur(item_style=ec.ItemStyle(opacity=0.2)),
                           select=ec.Select(item_style=ec.ItemStyle(border_width=3)),
                           selected_mode="multiple")
        s = mat._cells[(0, 0)].series_configs[0]
        assert "blur" in s
        assert "select" in s
        assert s["selectedMode"] == "multiple"

    def test_plot_with_animation(self):
        df = pd.DataFrame({"X": ["A", "B"], "Y": [10, 20]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).plot(df, "X", "Y",
                            animation=ec.AnimationConfig(
                                animation_duration=1000))
        s = mat._cells[(0, 0)].series_configs[0]
        assert s["animationDuration"] == 1000

    def test_pie_legend_without_hue(self):
        """Pie always emits legend (names are meaningful categories)."""
        df = pd.DataFrame({"Name": ["A", "B"], "Val": [30, 70]})
        mat = MatrixFigure(rows=["R1"], cols=["C1"])
        mat.cell(0, 0).pie(df, "Name", "Val")
        opt = mat.to_option()
        assert "legend" in opt
        assert set(opt["legend"]["data"]) == {"A", "B"}

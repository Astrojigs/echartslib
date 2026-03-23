"""Tests for ec.subplots() — matplotlib-style subplot grid."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

import echartsy as ec
from echartsy.figure import Figure
from echartsy.subplots import subplots, SubplotFigure, AxesGrid, AxesRow


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_df():
    return pd.DataFrame({
        "X": ["A", "B", "C", "D"],
        "Y": [10, 20, 30, 40],
    })


@pytest.fixture()
def hue_df():
    return pd.DataFrame({
        "X": ["A", "A", "B", "B"],
        "Y": [10, 20, 30, 40],
        "G": ["g1", "g2", "g1", "g2"],
    })


@pytest.fixture()
def pie_df():
    return pd.DataFrame({
        "Name": ["Apple", "Banana", "Cherry"],
        "Val": [30, 50, 20],
    })


# ═════════════════════════════════════════════════════════════════════════════
#  Factory shape tests
# ═════════════════════════════════════════════════════════════════════════════

class TestSubplotsFactory:
    """Test that subplots() returns correct shapes like matplotlib."""

    def test_scalar_return(self):
        fig, ax = subplots()
        assert isinstance(fig, SubplotFigure)
        assert isinstance(ax, Figure)

    def test_scalar_explicit_1x1(self):
        fig, ax = subplots(1, 1)
        assert isinstance(ax, Figure)

    def test_1d_row(self):
        fig, ax = subplots(1, 3)
        assert len(ax) == 3
        for i in range(3):
            assert isinstance(ax[i], Figure)

    def test_1d_col(self):
        fig, ax = subplots(3, 1)
        assert len(ax) == 3
        for i in range(3):
            assert isinstance(ax[i], Figure)

    def test_2d(self):
        fig, ax = subplots(2, 3)
        assert len(ax) == 2
        for ri in range(2):
            for ci in range(3):
                assert isinstance(ax[ri, ci], Figure)

    def test_each_ax_is_figure(self):
        fig, ax = subplots(2, 2)
        assert all(isinstance(ax[r, c], Figure) for r in range(2) for c in range(2))

    def test_default_params(self):
        fig, _ = subplots()
        assert fig._height == "600px"
        assert fig._renderer == "svg"

    def test_ec_dot_subplots(self):
        """ec.subplots works as a top-level convenience."""
        fig, ax = ec.subplots(2, 2)
        assert isinstance(fig, SubplotFigure)


# ═════════════════════════════════════════════════════════════════════════════
#  Chrome
# ═════════════════════════════════════════════════════════════════════════════

class TestSubplotFigureChrome:
    def test_suptitle(self):
        fig, _ = subplots()
        ret = fig.suptitle("Hello")
        assert ret is fig
        assert fig._suptitle_cfg["text"] == "Hello"

    def test_suptitle_with_subtitle(self):
        fig, _ = subplots()
        fig.suptitle("Main", subtitle="Sub")
        assert fig._suptitle_cfg["subtext"] == "Sub"

    def test_palette(self):
        fig, _ = subplots()
        fig.palette(["#f00", "#0f0", "#00f"])
        assert fig._global_palette == ["#f00", "#0f0", "#00f"]

    def test_legend_bottom(self):
        fig, _ = subplots()
        fig.legend(position="bottom")
        assert fig._global_legend_cfg["bottom"] == 0

    def test_legend_right(self):
        fig, _ = subplots()
        fig.legend(position="right")
        assert fig._global_legend_cfg["orient"] == "vertical"
        assert fig._global_legend_cfg["right"] == 0

    def test_legend_custom_kwargs(self):
        fig, _ = subplots()
        fig.legend(position="bottom", itemGap=20)
        assert fig._global_legend_cfg["itemGap"] == 20

    def test_repr(self):
        fig, _ = subplots(3, 4)
        assert "3×4" in repr(fig)


# ═════════════════════════════════════════════════════════════════════════════
#  to_option — core merge logic
# ═════════════════════════════════════════════════════════════════════════════

class TestPerCellChrome:
    """Test that per-cell Figure methods work in subplots."""

    def test_cell_title(self, simple_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y").title("Left Chart")
        ax[1].bar(simple_df, "X", "Y").title("Right Chart")
        opt = fig.to_option()
        # title should be a list of title objects
        assert isinstance(opt["title"], list)
        texts = [t["text"] for t in opt["title"]]
        assert "Left Chart" in texts
        assert "Right Chart" in texts

    def test_cell_title_with_suptitle(self, simple_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y").title("Cell Title")
        fig.suptitle("Global Title")
        opt = fig.to_option()
        assert isinstance(opt["title"], list)
        texts = [t["text"] for t in opt["title"]]
        assert "Global Title" in texts
        assert "Cell Title" in texts

    def test_cell_xlabel_ylabel(self, simple_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y").xlabel("Categories").ylabel("Values")
        opt = fig.to_option()
        assert opt["xAxis"][0]["name"] == "Categories"
        assert opt["yAxis"][0]["name"] == "Values"

    def test_cell_legend_styling(self, hue_df):
        """ax.legend() styling is preserved in subplot option."""
        fig, ax = subplots(1, 2)
        ax[0].bar(hue_df, "X", "Y", hue="G").legend(
            orient="horizontal", icon="circle"
        )
        opt = fig.to_option()
        legend = opt["legend"] if isinstance(opt["legend"], dict) else opt["legend"][0]
        assert legend["orient"] == "horizontal"
        assert legend["icon"] == "circle"


class TestToOption:
    def test_basic_2x2_bar(self, simple_df):
        fig, ax = subplots(2, 2)
        for r in range(2):
            for c in range(2):
                ax[r, c].bar(simple_df, "X", "Y")
        opt = fig.to_option()

        assert len(opt["grid"]) == 4
        assert len(opt["xAxis"]) == 4
        assert len(opt["yAxis"]) == 4
        assert len(opt["series"]) == 4
        # Verify axis indices
        for i, s in enumerate(opt["series"]):
            assert s["xAxisIndex"] == i
            assert s["yAxisIndex"] == i

    def test_pie_positioning(self, pie_df):
        fig, ax = subplots(1, 2)
        ax[0].pie(pie_df, "Name", "Val")
        ax[1].bar(pd.DataFrame({"X": ["A"], "Y": [10]}), "X", "Y")
        opt = fig.to_option()

        pie_series = [s for s in opt["series"] if s["type"] == "pie"]
        assert len(pie_series) == 1
        assert "center" in pie_series[0]
        assert "xAxisIndex" not in pie_series[0]

        bar_series = [s for s in opt["series"] if s["type"] == "bar"]
        assert "xAxisIndex" in bar_series[0]

    def test_mixed_types(self, simple_df, pie_df):
        scatter_df = pd.DataFrame({"X": [1, 2, 3], "Y": [10, 20, 30]})
        fig, ax = subplots(2, 2)
        ax[0, 0].bar(simple_df, "X", "Y")
        ax[0, 1].plot(simple_df, "X", "Y")
        ax[1, 0].pie(pie_df, "Name", "Val")
        ax[1, 1].scatter(scatter_df, "X", "Y")
        opt = fig.to_option()

        types = [s["type"] for s in opt["series"]]
        assert "bar" in types
        assert "line" in types
        assert "pie" in types
        assert "scatter" in types

    def test_empty_cell(self, simple_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        # ax[1] left empty
        opt = fig.to_option()

        assert len(opt["grid"]) == 2
        assert len(opt["xAxis"]) == 2
        assert len(opt["yAxis"]) == 2
        # Only 1 series (from the bar cell)
        assert len(opt["series"]) == 1
        # Empty cell's axes should be hidden
        assert opt["xAxis"][1].get("show") is False
        assert opt["yAxis"][1].get("show") is False

    def test_suptitle_in_option(self, simple_df):
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        fig.suptitle("Test Title", subtitle="Sub")
        opt = fig.to_option()
        assert opt["title"]["text"] == "Test Title"
        assert opt["title"]["subtext"] == "Sub"

    def test_global_legend(self, hue_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(hue_df, "X", "Y", hue="G")
        ax[1].bar(hue_df, "X", "Y")
        opt = fig.to_option()

        assert isinstance(opt["legend"], dict)
        assert "g1" in opt["legend"]["data"]
        assert "g2" in opt["legend"]["data"]

    def test_global_legend_custom_position(self, hue_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(hue_df, "X", "Y", hue="G")
        fig.legend(position="right")
        opt = fig.to_option()

        assert opt["legend"]["orient"] == "vertical"
        assert opt["legend"]["right"] == 0

    def test_palette_in_option(self, simple_df):
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        fig.palette(["#e74c3c", "#3498db"])
        opt = fig.to_option()
        assert opt["color"] == ["#e74c3c", "#3498db"]

    def test_grid_positioning(self, simple_df):
        fig, ax = subplots(2, 2)
        for r in range(2):
            for c in range(2):
                ax[r, c].bar(simple_df, "X", "Y")
        opt = fig.to_option()

        # All grids should have percentage-based positioning
        for g in opt["grid"]:
            assert "%" in g["left"]
            assert "%" in g["top"]
            assert "%" in g["width"]
            assert "%" in g["height"]

    def test_hue_legend_items(self, hue_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(hue_df, "X", "Y", hue="G")
        ax[1].bar(hue_df, "X", "Y", hue="G")
        opt = fig.to_option()

        # Each cell's legend names are namespaced so toggling one
        # doesn't affect identically-named series in another cell
        legend_data = opt["legend"]["data"]
        assert len(legend_data) == 4  # g1, g2 + namespaced g1, g2
        # Visible text is the same (ZWS is invisible)
        visible = [n.replace("\u200B", "") for n in legend_data]
        assert visible == ["g1", "g2", "g1", "g2"]


# ═════════════════════════════════════════════════════════════════════════════
#  Per-cell legend via ax.legend()
# ═════════════════════════════════════════════════════════════════════════════

class TestPerCellLegend:
    def test_ax_legend_creates_per_cell_legend(self, hue_df):
        """ax.legend() on a cell creates a positioned legend for that cell."""
        fig, ax = subplots(1, 2)
        ax[0].bar(hue_df, "X", "Y", hue="G")
        ax[0].legend(orient="vertical")
        df2 = pd.DataFrame({
            "X": ["A", "A"], "Y": [5, 15], "G": ["g3", "g4"],
        })
        ax[1].bar(df2, "X", "Y", hue="G")
        ax[1].legend(orient="horizontal")
        opt = fig.to_option()
        assert isinstance(opt["legend"], list)
        assert len(opt["legend"]) == 2

    def test_no_ax_legend_gives_global(self, hue_df):
        """Without ax.legend(), all items merge into one global legend."""
        fig, ax = subplots(1, 2)
        ax[0].bar(hue_df, "X", "Y", hue="G")
        opt = fig.to_option()
        assert isinstance(opt["legend"], dict)

    def test_mixed_per_cell_and_global(self, hue_df):
        """One cell with ax.legend(), another without → both appear."""
        fig, ax = subplots(1, 2)
        ax[0].bar(hue_df, "X", "Y", hue="G")
        ax[0].legend(orient="vertical")
        df2 = pd.DataFrame({
            "X": ["A", "A"], "Y": [5, 15], "G": ["g3", "g4"],
        })
        ax[1].bar(df2, "X", "Y", hue="G")
        # ax[1] has NO ax.legend() call → its items go to global
        opt = fig.to_option()
        assert isinstance(opt["legend"], list)
        # One per-cell + one global for the remaining
        assert len(opt["legend"]) == 2

    def test_legend_show_false_hides(self, hue_df):
        """ax.legend(show=False) suppresses that cell's legend."""
        fig, ax = subplots(1, 2)
        ax[0].bar(hue_df, "X", "Y", hue="G")
        ax[0].legend(show=False)
        opt = fig.to_option()
        # The legend should have show=False
        if isinstance(opt["legend"], list):
            assert opt["legend"][0].get("show") is False
        else:
            assert opt["legend"].get("show") is False


# ═════════════════════════════════════════════════════════════════════════════
#  Rendering
# ═════════════════════════════════════════════════════════════════════════════

class TestRendering:
    def test_show_does_not_crash(self, simple_df):
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        with patch("echartsy.subplots.render") as mock_render:
            fig.show()
            mock_render.assert_called_once()

    def test_to_html_produces_file(self, simple_df):
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.html")
            result = fig.to_html(path)
            assert os.path.isfile(path)
            assert result == path

    def test_to_option_returns_dict(self, simple_df):
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        opt = fig.to_option()
        assert isinstance(opt, dict)
        assert "series" in opt


# ═════════════════════════════════════════════════════════════════════════════
#  Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_scalar_indexing(self, simple_df):
        """ax.bar() works on a scalar (1×1 layout)."""
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        opt = fig.to_option()
        assert len(opt["series"]) == 1

    def test_all_cells_empty(self):
        """All cells empty → grids exist but no series."""
        fig, ax = subplots(2, 2)
        opt = fig.to_option()
        assert len(opt["grid"]) == 4
        assert len(opt["series"]) == 0

    def test_mode_independence(self, simple_df, pie_df):
        """Different chart types in different cells don't conflict."""
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        ax[1].pie(pie_df, "Name", "Val")
        opt = fig.to_option()

        bar = [s for s in opt["series"] if s["type"] == "bar"]
        pie = [s for s in opt["series"] if s["type"] == "pie"]
        assert len(bar) == 1
        assert len(pie) == 1

    def test_large_grid_4x4(self, simple_df):
        fig, ax = subplots(4, 4)
        for r in range(4):
            for c in range(4):
                ax[r, c].bar(simple_df, "X", "Y")
        opt = fig.to_option()
        assert len(opt["grid"]) == 16
        assert len(opt["series"]) == 16

    def test_1d_col_indexing(self, simple_df):
        """Single-column layout uses 1D indexing."""
        scatter_df = pd.DataFrame({"X": [1, 2, 3], "Y": [10, 20, 30]})
        fig, ax = subplots(3, 1)
        ax[0].bar(simple_df, "X", "Y")
        ax[1].plot(simple_df, "X", "Y")
        ax[2].scatter(scatter_df, "X", "Y")
        opt = fig.to_option()
        assert len(opt["series"]) == 3

    def test_no_suptitle_reduces_top_margin(self, simple_df):
        """Without suptitle, top margin is smaller."""
        fig1, ax1 = subplots()
        ax1.bar(simple_df, "X", "Y")
        fig1.suptitle("Title")
        opt1 = fig1.to_option()

        fig2, ax2 = subplots()
        ax2.bar(simple_df, "X", "Y")
        opt2 = fig2.to_option()

        # Grid with title should have larger top
        top1 = float(opt1["grid"][0]["top"].rstrip("%"))
        top2 = float(opt2["grid"][0]["top"].rstrip("%"))
        assert top1 > top2


# ── Per-cell Figure method propagation ────────────────────────────────────

class TestFigureMethodPropagation:
    """Verify that Figure chrome methods work in subplot context."""

    def test_margins_top(self, simple_df):
        """ax.margins(top=...) shrinks cell grid from the top."""
        fig, ax = subplots(1, 2, height="600px")
        ax[0].bar(simple_df, "X", "Y")
        ax[1].bar(simple_df, "X", "Y")
        ax[0].margins(top=100)
        opt = fig.to_option()
        # Cell 0 should have a larger top % than cell 1
        top0 = float(opt["grid"][0]["top"].rstrip("%"))
        top1 = float(opt["grid"][1]["top"].rstrip("%"))
        assert top0 > top1

    def test_margins_left(self, simple_df):
        """ax.margins(left=...) shifts cell grid right from its default."""
        # Compare same cell with and without margin
        fig1, ax1 = subplots(height="600px")
        ax1.bar(simple_df, "X", "Y")
        opt1 = fig1.to_option()

        fig2, ax2 = subplots(height="600px")
        ax2.bar(simple_df, "X", "Y")
        ax2.margins(left=120)
        opt2 = fig2.to_option()

        left_default = float(opt1["grid"][0]["left"].rstrip("%"))
        left_margin = float(opt2["grid"][0]["left"].rstrip("%"))
        assert left_margin > left_default

    def test_margins_bottom(self, simple_df):
        """ax.margins(bottom=...) shrinks cell grid height."""
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        ax.margins(bottom=80)
        opt = fig.to_option()
        h = float(opt["grid"][0]["height"].rstrip("%"))
        # Default cell height without margin is ~91% — with margin it should shrink
        assert h < 90

    def test_margins_unchanged_without_call(self, simple_df):
        """Grid keeps default positioning when margins() is not called."""
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        opt = fig.to_option()
        grid = opt["grid"][0]
        # Should use standard percentage positioning
        assert "%" in str(grid["left"])
        assert "%" in str(grid["top"])

    def test_tooltip_propagation(self, simple_df):
        """ax.tooltip() config is reflected in the merged option."""
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        ax[0].tooltip(trigger="axis", pointer="shadow")
        ax[1].bar(simple_df, "X", "Y")
        opt = fig.to_option()
        assert opt["tooltip"]["trigger"] == "axis"
        assert opt["tooltip"]["confine"] is True  # always forced

    def test_toolbox_propagation(self, simple_df):
        """ax.save() / ax.toolbox() is included in subplot option."""
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        ax.save("my_chart")
        opt = fig.to_option()
        assert "toolbox" in opt
        assert opt["toolbox"]["show"] is True

    def test_datazoom_propagation(self, simple_df):
        """ax.datazoom() is included with correct axis binding."""
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        ax[0].datazoom(start=20, end=80)
        ax[1].bar(simple_df, "X", "Y")
        opt = fig.to_option()
        assert "dataZoom" in opt
        # Should be bound to grid index 0's x-axis
        assert opt["dataZoom"][0]["xAxisIndex"] == 0

    def test_visual_map_propagation(self):
        """ax.visual_map() is included in subplot option."""
        df = pd.DataFrame({
            "X": ["A", "B", "C"],
            "Y": [10, 20, 30],
        })
        fig, ax = subplots()
        ax.bar(df, "X", "Y")
        ax.visual_map(min_val=0, max_val=50, colors=["#blue", "#red"])
        opt = fig.to_option()
        assert "visualMap" in opt

    def test_per_cell_palette(self, simple_df):
        """ax.palette() applies colors to that cell's series."""
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        ax[0].palette(["#ff0000", "#00ff00", "#0000ff"])
        ax[1].bar(simple_df, "X", "Y")
        opt = fig.to_option()
        # Cell 0's series should have itemStyle.color from the palette
        s0 = opt["series"][0]
        assert s0.get("itemStyle", {}).get("color") == "#ff0000"
        # Cell 1's series should NOT have itemStyle.color from custom palette
        s1 = opt["series"][1]
        assert s1.get("itemStyle", {}).get("color") != "#ff0000"

    def test_grid_splitline(self, simple_df):
        """ax.grid() splitLine is carried through axis merge."""
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        ax[0].grid(show=True, style="dashed", color="#ccc")
        ax[1].bar(simple_df, "X", "Y")
        opt = fig.to_option()
        y0 = opt["yAxis"][0]
        assert y0.get("splitLine", {}).get("show") is True

    def test_xlim_ylim(self, simple_df):
        """ax.xlim() / ax.ylim() are carried through axis merge."""
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        ax.ylim(0, 100)
        opt = fig.to_option()
        assert opt["yAxis"][0]["min"] == 0
        assert opt["yAxis"][0]["max"] == 100

    def test_xticks_yticks(self, simple_df):
        """ax.xticks() / ax.yticks() are carried through axis merge."""
        fig, ax = subplots()
        ax.bar(simple_df, "X", "Y")
        ax.xticks(rotate=45)
        opt = fig.to_option()
        assert opt["xAxis"][0]["axisLabel"]["rotate"] == 45


# ═════════════════════════════════════════════════════════════════════════════
#  Input validation
# ═════════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    def test_zero_nrows_raises(self):
        with pytest.raises(ValueError):
            subplots(0, 2)

    def test_negative_ncols_raises(self):
        with pytest.raises(ValueError):
            subplots(2, -1)


# ═════════════════════════════════════════════════════════════════════════════
#  Heatmap in subplot
# ═════════════════════════════════════════════════════════════════════════════

class TestHeatmapInSubplot:
    @pytest.fixture()
    def heatmap_df(self):
        return pd.DataFrame({
            "X": ["A", "A", "B", "B"],
            "Y": ["r1", "r2", "r1", "r2"],
            "Val": [1, 2, 3, 4],
        })

    def test_heatmap_axes_visible(self, simple_df, heatmap_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        ax[1].heatmap(heatmap_df, "X", "Y", "Val")
        opt = fig.to_option()
        # Heatmap cell (index 1) needs visible axes — show should NOT be False
        assert opt["xAxis"][1].get("show") is not False
        assert opt["yAxis"][1].get("show") is not False

    def test_heatmap_visual_map_series_index(self, simple_df, heatmap_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        ax[1].heatmap(heatmap_df, "X", "Y", "Val")
        opt = fig.to_option()
        # The bar is series 0; heatmap is series 1 → visualMap should target 1
        vm = opt["visualMap"]
        if isinstance(vm, list):
            vm = vm[0]
        assert vm["seriesIndex"] == 1


# ═════════════════════════════════════════════════════════════════════════════
#  Horizontal bar in subplot
# ═════════════════════════════════════════════════════════════════════════════

class TestHorizontalBarInSubplot:
    def test_barh_axis_types(self, simple_df):
        fig, ax = subplots(1, 2)
        ax[0].barh(simple_df, "X", "Y")
        ax[1].bar(simple_df, "X", "Y")
        opt = fig.to_option()
        # barh cell (index 0): xAxis should be "value", yAxis should be "category"
        assert opt["xAxis"][0]["type"] == "value"
        assert opt["yAxis"][0]["type"] == "category"
        # Normal bar cell (index 1): xAxis = "category", yAxis = "value"
        assert opt["xAxis"][1]["type"] == "category"
        assert opt["yAxis"][1]["type"] == "value"


# ═════════════════════════════════════════════════════════════════════════════
#  Dual Y-axis
# ═════════════════════════════════════════════════════════════════════════════

class TestDualYAxis:
    def test_dual_yaxis_creates_extra_y(self):
        df = pd.DataFrame({"X": ["A", "B", "C"], "Y1": [10, 20, 30], "Y2": [100, 200, 300]})
        fig, ax = subplots(1, 1)
        ax.plot(df, "X", "Y1")
        ax.plot(df, "X", "Y2", axis=1)
        opt = fig.to_option()
        assert len(opt["yAxis"]) >= 2


# ═════════════════════════════════════════════════════════════════════════════
#  Non-cartesian positioning (funnel)
# ═════════════════════════════════════════════════════════════════════════════

class TestNonCartesianPositioning:
    def test_funnel_positioned_in_cell(self, simple_df):
        funnel_df = pd.DataFrame({
            "Stage": ["Visit", "Signup", "Purchase"],
            "Count": [100, 60, 20],
        })
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        ax[1].funnel(funnel_df, "Stage", "Count")
        opt = fig.to_option()
        funnel_series = [s for s in opt["series"] if s["type"] == "funnel"]
        assert len(funnel_series) == 1
        fs = funnel_series[0]
        assert "left" in fs
        assert "top" in fs
        assert "width" in fs
        assert "height" in fs


# ═════════════════════════════════════════════════════════════════════════════
#  Title positioning
# ═════════════════════════════════════════════════════════════════════════════

class TestTitlePositioning:
    def test_title_left_keyword(self, simple_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y").title("T", left="left")
        ax[1].bar(simple_df, "X", "Y")
        opt = fig.to_option()
        titles = opt["title"] if isinstance(opt["title"], list) else [opt["title"]]
        cell_title = [t for t in titles if t["text"] == "T"][0]
        # "left" keyword should resolve to a percentage string, not literal "left"
        assert isinstance(cell_title["left"], str)
        assert "%" in cell_title["left"]

    def test_title_right_keyword(self, simple_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y").title("T", left="right")
        ax[1].bar(simple_df, "X", "Y")
        opt = fig.to_option()
        titles = opt["title"] if isinstance(opt["title"], list) else [opt["title"]]
        cell_title = [t for t in titles if t["text"] == "T"][0]
        assert cell_title["textAlign"] == "right"


# ═════════════════════════════════════════════════════════════════════════════
#  Pie palette in subplot
# ═════════════════════════════════════════════════════════════════════════════

class TestPiePaletteInSubplot:
    def test_pie_per_cell_palette(self, pie_df):
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        fig, ax = subplots(1, 2)
        ax[0].pie(pie_df, "Name", "Val")
        ax[0].palette(colors)
        ax[1].bar(
            pd.DataFrame({"X": ["A"], "Y": [10]}), "X", "Y"
        )
        opt = fig.to_option()
        pie_series = [s for s in opt["series"] if s["type"] == "pie"][0]
        for i, item in enumerate(pie_series["data"]):
            assert item["itemStyle"]["color"] == colors[i % len(colors)]


# ═════════════════════════════════════════════════════════════════════════════
#  Multiple series in one cell
# ═════════════════════════════════════════════════════════════════════════════

class TestMultipleSeriesOneCell:
    def test_overlay_bar_and_line(self, simple_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        ax[0].plot(simple_df, "X", "Y")
        ax[1].bar(simple_df, "X", "Y")
        opt = fig.to_option()
        # The bar and line in cell 0 should share the same xAxisIndex
        cell0_series = [s for s in opt["series"] if s.get("xAxisIndex") == 0]
        assert len(cell0_series) == 2
        types = {s["type"] for s in cell0_series}
        assert "bar" in types
        assert "line" in types


# ═════════════════════════════════════════════════════════════════════════════
#  AxesGrid / AxesRow repr and len
# ═════════════════════════════════════════════════════════════════════════════

class TestAxesGridRow:
    def test_axes_grid_repr(self):
        _, ax = subplots(2, 3)
        assert repr(ax) == "AxesGrid(2x3)"

    def test_axes_row_repr(self):
        _, ax = subplots(1, 3)
        assert repr(ax) == "AxesRow(3)"

    def test_axes_row_len(self):
        _, ax = subplots(1, 3)
        assert len(ax) == 3


# ═════════════════════════════════════════════════════════════════════════════
#  SubplotFigure.margins()
# ═════════════════════════════════════════════════════════════════════════════

class TestFigMargins:
    def test_fig_margins_top(self, simple_df):
        fig, ax = subplots(1, 2)
        ax[0].bar(simple_df, "X", "Y")
        ax[1].bar(simple_df, "X", "Y")
        fig.margins(top=15)
        opt = fig.to_option()
        top_val = float(opt["grid"][0]["top"].rstrip("%"))
        assert top_val >= 15

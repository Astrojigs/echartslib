"""
echartsy — A matplotlib-style fluent builder API for Apache ECharts.

Quick-start
-----------
>>> import echartsy as ec
>>> ec.config(engine="jupyter")  # or "python" or "streamlit"
>>>
>>> (
...     ec.Figure(height="500px")
...     .bar(df, x="Month", y="Revenue", hue="Region", stack=True)
...     .plot(df_totals, x="Month", y="Target", smooth=True, axis=1)
...     .title("Revenue vs Target")
...     .ylabel("Revenue")
...     .ylabel_right("Target (%)")
...     .show()
... )
"""
from echartsy._config import config, get_adaptive, get_engine, get_overlap_warnings
from echartsy.emphasis import (
    AnimationConfig,
    AreaStyle,
    Blur,
    Emphasis,
    EndLabelStyle,
    FunnelEmphasis,
    GraphEmphasis,
    ItemStyle,
    LabelLineStyle,
    LabelStyle,
    LineEmphasis,
    LineStyle,
    PieEmphasis,
    RadarEmphasis,
    SankeyEmphasis,
    ScatterEmphasis,
    Select,
    TooltipStyle,
    TreemapEmphasis,
)
from echartsy.exceptions import (
    BuilderConfigError,
    BuilderError,
    DataValidationError,
    OverlapWarning,
    TimelineConfigError,
)
from echartsy.figure import Figure, figure
from echartsy.subplots import SubplotFigure, subplots
from echartsy.styles import (
    PALETTE_CLINICAL,
    PALETTE_DARK,
    PALETTE_RUSTY,
    StylePreset,
)
from echartsy.timeline import (
    TimelineFigure,
    detect_time_format,
    timeline_figure,
)

__version__ = "0.7.2"

__all__ = [
    # Config
    "config",
    "get_engine",
    "get_adaptive",
    "get_overlap_warnings",
    # Figure
    "Figure",
    "figure",
    # Subplots
    "SubplotFigure",
    "subplots",
    # Timeline
    "TimelineFigure",
    "timeline_figure",
    "detect_time_format",
    # Styles
    "StylePreset",
    "PALETTE_CLINICAL",
    "PALETTE_DARK",
    "PALETTE_RUSTY",
    # Emphasis
    "Emphasis",
    "LineEmphasis",
    "ScatterEmphasis",
    "PieEmphasis",
    "RadarEmphasis",
    "SankeyEmphasis",
    "FunnelEmphasis",
    "TreemapEmphasis",
    "GraphEmphasis",
    "ItemStyle",
    "LabelStyle",
    "LineStyle",
    "AreaStyle",
    "LabelLineStyle",
    "EndLabelStyle",
    "Blur",
    "Select",
    "TooltipStyle",
    "AnimationConfig",
    # Exceptions
    "BuilderError",
    "BuilderConfigError",
    "DataValidationError",
    "TimelineConfigError",
    "OverlapWarning",
]

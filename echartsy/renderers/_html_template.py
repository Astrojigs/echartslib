"""Shared HTML template for rendering ECharts in a browser or Jupyter."""
from __future__ import annotations

import datetime
import json
import re
from typing import Optional


def _sanitize_chart_id(chart_id: str) -> str:
    """Strip chart_id to alphanumeric, underscore, and hyphen only."""
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "", chart_id)
    if not sanitized:
        raise ValueError("chart_id must contain at least one alphanumeric character")
    return sanitized


_VALID_RENDERERS = {"canvas", "svg"}
_VALID_ADAPTIVE = {"auto", "light", "dark"}
# Whitelist of allowed ECharts built-in themes.
_VALID_THEMES = {"dark", "light", "vintage", "westeros", "essos", "wonderland", "walden",
                 "chalk", "infographic", "macarons", "roma", "shine", "purple-passion",
                 "halloween"}


def _validate_css_dimension(value: str, name: str) -> None:
    """Validate that a CSS dimension string is safe (e.g. '100%', '400px')."""
    if not re.match(r'^[\d.]+(px|%|em|rem|vw|vh)?$', value):
        raise ValueError(f"{name} must be a valid CSS dimension (e.g. '100%', '400px'), got '{value}'")


def _json_default(obj: object) -> object:
    """JSON serializer for types not handled by default."""
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# CDN URL for Apache ECharts
ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@6/dist/echarts.min.js"

# ── Adaptive colour constants ────────────────────────────────────────────
# Hardcoded light-mode colours that the StylePreset defaults ship with.
# When dark mode is detected these are swapped to the dark equivalents.
_LIGHT_TEXT_COLORS = {"#666", "#333", "#666666", "#333333"}
_DARK_TEXT_REPLACEMENT = "#ccc"
_DARK_GRID_REPLACEMENT = "#555"
_LIGHT_GRID_COLORS = {"#eee", "#eeeeee", "#f0f0f0"}

# ── JavaScript snippet for adaptive theme ────────────────────────────────
# Injected when adaptive="auto".  Walks the raw option object and replaces
# light-mode hardcoded colours with dark-friendly ones *before* handing it
# to ECharts.  Also picks the built-in 'dark' theme when appropriate.

_ADAPTIVE_JS = """
(function() {
  // ---------- detect dark mode ----------
  var FORCE = "__FORCE_MODE__";          // "auto" | "light" | "dark"
  var isDark;
  if (FORCE === "dark")       isDark = true;
  else if (FORCE === "light") isDark = false;
  else                        isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;

  // ---------- colour maps ----------
  var LIGHT_TEXT = {"#666":1, "#333":1, "#666666":1, "#333333":1};
  var LIGHT_GRID = {"#eee":1, "#eeeeee":1, "#f0f0f0":1};
  var DARK_TEXT  = "#ccc";
  var DARK_GRID  = "#555";
  var DARK_TITLE = "#eee";
  var DARK_SUBTITLE = "#bbb";

  // ---------- recursive patcher ----------
  function patchColors(obj) {
    if (!obj || typeof obj !== "object") return;
    if (Array.isArray(obj)) { obj.forEach(patchColors); return; }
    for (var k in obj) {
      if (!obj.hasOwnProperty(k)) continue;
      var v = obj[k];
      if (typeof v === "string") {
        var lv = v.toLowerCase();
        if (k === "color" && LIGHT_TEXT[lv])         obj[k] = DARK_TEXT;
        else if (k === "color" && LIGHT_GRID[lv])    obj[k] = DARK_GRID;
        else if (k === "borderColor" && LIGHT_GRID[lv]) obj[k] = DARK_GRID;
      } else if (typeof v === "object") {
        patchColors(v);
      }
    }
  }

  // ---------- patch title specifically ----------
  function patchTitle(t) {
    if (!t) return;
    if (t.textStyle)    t.textStyle.color    = t.textStyle.color    || DARK_TITLE;
    else                t.textStyle = {color: DARK_TITLE};
    if (t.subtextStyle) t.subtextStyle.color = t.subtextStyle.color || DARK_SUBTITLE;
  }

  // ---------- patch splitLine / grid line colours ----------
  function patchGridLines(axes) {
    if (!axes) return;
    var arr = Array.isArray(axes) ? axes : [axes];
    arr.forEach(function(ax) {
      if (ax.splitLine && ax.splitLine.lineStyle) {
        var c = (ax.splitLine.lineStyle.color || "").toLowerCase();
        if (LIGHT_GRID[c]) ax.splitLine.lineStyle.color = DARK_GRID;
      }
    });
  }

  // ---------- apply ----------
  var option  = __OPTION_JSON__;
  var themeToUse = __EXPLICIT_THEME__;   // null or "dark" etc

  if (isDark && !themeToUse) {
    themeToUse = "dark";
    patchColors(option);

    // Patch title
    if (option.title) patchTitle(option.title);
    // Patch axes
    patchGridLines(option.xAxis);
    patchGridLines(option.yAxis);

    // Remove hardcoded light backgrounds so the dark theme shines through
    if (option.backgroundColor) {
      var bg = option.backgroundColor.toLowerCase();
      if (bg === "#fff" || bg === "#ffffff" || bg === "#fafafa" || bg === "white")
        delete option.backgroundColor;
    }

    // Timeline: patch baseOption if present
    if (option.baseOption) {
      patchColors(option.baseOption);
      if (option.baseOption.title) patchTitle(option.baseOption.title);
      if (option.baseOption.backgroundColor) {
        var bbg = option.baseOption.backgroundColor.toLowerCase();
        if (bbg === "#fff" || bbg === "#ffffff" || bbg === "#fafafa" || bbg === "white")
          delete option.baseOption.backgroundColor;
      }
    }
    if (option.options) {
      option.options.forEach(function(fo) {
        patchColors(fo);
        patchGridLines(fo.xAxis);
        patchGridLines(fo.yAxis);
      });
    }
  }

  // ---------- init chart ----------
  var chart = echarts.init(
    document.getElementById("__CHART_ID__"),
    themeToUse,
    { renderer: "__RENDERER__" }
  );
  chart.setOption(option);
  window.addEventListener("resize", function() { chart.resize(); });

  // ---------- live theme switch (OS toggle) ----------
  if (FORCE === "auto" && window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function() {
      location.reload();
    });
  }
})();
"""


def _build_adaptive_script(
    option_json: str,
    chart_id: str,
    theme: Optional[str],
    renderer: str,
    adaptive: str,
) -> str:
    """Return the <script> body with adaptive dark-mode support."""
    chart_id = _sanitize_chart_id(chart_id)
    if renderer not in _VALID_RENDERERS:
        raise ValueError(f"renderer must be one of {_VALID_RENDERERS}, got '{renderer}'")
    if adaptive not in _VALID_ADAPTIVE:
        raise ValueError(f"adaptive must be one of {_VALID_ADAPTIVE}, got '{adaptive}'")
    if theme is not None and theme not in _VALID_THEMES:
        raise ValueError(f"theme must be one of {sorted(_VALID_THEMES)} or None, got '{theme}'")
    explicit_theme = f"'{theme}'" if theme else "null"
    return (
        _ADAPTIVE_JS
        .replace("__CHART_ID__", chart_id)
        .replace("__EXPLICIT_THEME__", explicit_theme)
        .replace('"__RENDERER__"', f'"{renderer}"')
        .replace('"__FORCE_MODE__"', f'"{adaptive}"')
        .replace("__OPTION_JSON__", option_json)
    )


def build_html(
    option: dict,
    height: str = "400px",
    width: str = "100%",
    theme: Optional[str] = None,
    renderer: str = "svg",
    chart_id: str = "ec_chart",
    adaptive: str = "auto",
) -> str:
    """Build a self-contained HTML page that renders an ECharts option dict.

    Parameters
    ----------
    option : dict
        The ECharts option configuration.
    height : str
        CSS height for the chart container.
    width : str
        CSS width for the chart container.
    theme : str or None
        ECharts built-in theme name (e.g. ``"dark"``).
    renderer : str
        ``"canvas"`` or ``"svg"``.
    chart_id : str
        DOM element id for the chart div.
    adaptive : str
        ``"auto"`` | ``"light"`` | ``"dark"`` — controls dark-mode adaptation.

    Returns
    -------
    str
        Complete HTML document string.
    """
    chart_id = _sanitize_chart_id(chart_id)
    _validate_css_dimension(width, "width")
    _validate_css_dimension(height, "height")
    option_json = json.dumps(option, indent=2, default=_json_default).replace("</", "<\\/")
    script_body = _build_adaptive_script(
        option_json, chart_id, theme, renderer, adaptive,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EChartsLib Chart</title>
<script src="{ECHARTS_CDN}"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1e1e1e; }}
    #{chart_id} {{ background: #2a2a2a; box-shadow: 0 2px 12px rgba(0,0,0,0.3); }}
  }}
  @media (prefers-color-scheme: light) {{
    body {{ background: #fafafa; }}
    #{chart_id} {{ background: #fff; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
  }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         display: flex; justify-content: center;
         align-items: center; min-height: 100vh; padding: 20px; }}
  #{chart_id} {{ width: {width}; height: {height}; border-radius: 8px; }}
</style>
</head>
<body>
<div id="{chart_id}"></div>
<script>
{script_body}
</script>
</body>
</html>"""


def build_jupyter_html(
    option: dict,
    height: str = "400px",
    width: str = "100%",
    theme: Optional[str] = None,
    renderer: str = "svg",
    chart_id: str = "ec_chart",
    adaptive: str = "auto",
) -> str:
    """Build an HTML fragment suitable for embedding inside a Jupyter IFrame.

    Similar to :func:`build_html` but as a minimal fragment without the
    full page chrome, optimised for ``IPython.display.IFrame``.
    """
    chart_id = _sanitize_chart_id(chart_id)
    _validate_css_dimension(width, "width")
    _validate_css_dimension(height, "height")
    option_json = json.dumps(option, indent=2, default=_json_default).replace("</", "<\\/")
    script_body = _build_adaptive_script(
        option_json, chart_id, theme, renderer, adaptive,
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script src="{ECHARTS_CDN}"></script>
<style>
  html, body {{ margin: 0; padding: 0; width: 100%; height: 100%;
               overflow: hidden; }}
  @media (prefers-color-scheme: dark) {{
    html, body {{ background: transparent; }}
  }}
  @media (prefers-color-scheme: light) {{
    html, body {{ background: transparent; }}
  }}
  #{chart_id} {{ width: {width}; height: {height}; }}
</style>
</head>
<body>
<div id="{chart_id}"></div>
<script>
{script_body}
</script>
</body>
</html>"""

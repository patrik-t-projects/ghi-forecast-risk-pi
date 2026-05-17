import pandas as pd
import numpy as np
import os
import json
from datetime import date
import requests
from PV_forecast_uncertainty_functions import choose_f0, limits_for_f0
from GHI_forecast_risk_functions import fill_model_gaps_with_backup_model, compute_future_quantile_error_band
from GHI_forecast_risk_Fit_Performance import build_risk_feature_dataframe, fit_risk_score_from_features
from config import DATA_DIR, DOWNLOADS_DIR, REPORTS_DIR, maybe_open_browser


# =========================
# USER SELECTION
# =========================
START_DATE = globals().get("START_DATE", "2026-05-15")
END_DATE   = globals().get("END_DATE", "2026-05-21")
MODEL = "ICON1"   # "ICON1" or "ICON2"
FIGSIZE = (10, 5)
INCLUDE_SCATTERPLOTS = globals().get("INCLUDE_SCATTERPLOTS", True)
OUTPUT_SUFFIX = globals().get("OUTPUT_SUFFIX", "")
OPEN_BROWSER = globals().get("OPEN_BROWSER", True)

GHI_threshold = 1
risk_window=0.1
quantile=0.95
min_samples=5

TRACKING_DIR = REPORTS_DIR
TRACKING_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DATA_PATH = TRACKING_DIR / f"GHI_forecast_risk_report_data_{MODEL}.csv"

# =========================

# Colors
elcom_colors = {
    "dark_blue": (75/255, 96/255, 165/255),
    "light_blue": (146/255, 162/255, 206/255),
    "green": (145/255, 195/255, 74/255),
    "yellow": (255/255, 217/255, 102/255),
    "orange": (249/255, 170/255, 0/255),
    "grey": (191/255, 191/255, 191/255),
    "lila": (150/255, 120/255, 200/255),
    "red": (192/255, 0/255, 0/255),
}

elcom_color_text = (55/255, 55/255, 55/255)



def rgb_to_css(rgb, alpha=1.0):
    r, g, b = [int(round(v * 255)) for v in rgb]
    return f"rgba({r}, {g}, {b}, {alpha})"


def clean_float(value):
    if pd.isna(value):
        return None
    return float(value)


def english_day_label(value):
    ts = pd.Timestamp(value)
    return f"{ts.strftime('%B')} {ts.day}, {ts.year}"


def time_series_trace(df, x_col, y_col, name, color, mode="line", dash=False, marker=False, y_axis="y"):
    plot_df = df[[x_col, y_col]].dropna().copy()
    return {
        "type": mode,
        "name": name,
        "color": color,
        "dash": dash,
        "marker": marker,
        "yAxis": y_axis,
        "x": plot_df[x_col].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        "y": [clean_float(v) for v in plot_df[y_col]],
    }


def band_trace(df, x_col, lower_col, upper_col, name, color):
    plot_df = df[[x_col, lower_col, upper_col]].dropna().copy()
    return {
        "type": "band",
        "name": name,
        "color": color,
        "x": plot_df[x_col].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        "lower": [clean_float(v) for v in plot_df[lower_col]],
        "upper": [clean_float(v) for v in plot_df[upper_col]],
    }


def bar_trace(df, x_col, y_col, name, color, hover_columns=None, y_axis="y", width_minutes=12, alpha=0.7):
    hover_columns = hover_columns or []
    plot_df = df[[x_col, y_col] + hover_columns].dropna(subset=[x_col, y_col]).copy()
    return {
        "type": "bar",
        "name": name,
        "color": color,
        "alpha": alpha,
        "yAxis": y_axis,
        "widthMs": width_minutes * 60 * 1000,
        "x": plot_df[x_col].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        "y": [clean_float(v) for v in plot_df[y_col]],
        "hover": plot_df[hover_columns].to_dict("records") if hover_columns else [],
    }


def scatter_trace(df, x_col, y_col, name, color, hover_columns, size=5):
    plot_df = df[[x_col, y_col] + hover_columns].dropna(subset=[x_col, y_col]).copy()
    return {
        "type": "scatter",
        "name": name,
        "color": color,
        "size": size,
        "x": [clean_float(v) for v in plot_df[x_col]],
        "y": [clean_float(v) for v in plot_df[y_col]],
        "hover": plot_df[hover_columns].to_dict("records"),
    }


def error_scatter_trace(df, x_col, y_col, x_left_col, x_right_col, name, color, hover_columns, size=7):
    plot_df = df[[x_col, y_col, x_left_col, x_right_col] + hover_columns].dropna(subset=[x_col, y_col]).copy()
    return {
        "type": "error_scatter",
        "name": name,
        "color": color,
        "size": size,
        "x": [clean_float(v) for v in plot_df[x_col]],
        "y": [clean_float(v) for v in plot_df[y_col]],
        "xLeft": [clean_float(v) for v in plot_df[x_left_col]],
        "xRight": [clean_float(v) for v in plot_df[x_right_col]],
        "hover": plot_df[hover_columns].to_dict("records"),
    }


def write_interactive_html_report(figures, output_path, page_title):
    payload = json.dumps(figures, ensure_ascii=False, allow_nan=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title}</title>
<style>
  :root {{
    --text: rgb(55, 55, 55);
    --grid: rgba(55, 55, 55, 0.14);
    --muted: rgba(55, 55, 55, 0.68);
    --panel: #ffffff;
    --border: rgba(55, 55, 55, 0.16);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    color: var(--text);
    background: #3a3a3a;
  }}
  main {{
    width: min(1280px, calc(100vw - 32px));
    margin: 24px auto 36px;
  }}
  h1 {{
    font-size: 24px;
    font-weight: 700;
    margin: 0 0 18px;
    color: #ffffff;
  }}
  .chart {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 18px 12px;
    margin-bottom: 22px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
  }}
  .report-row {{
    margin-bottom: 24px;
  }}
  .row-grid {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 18px;
  }}
  .row-grid .chart {{
    margin-bottom: 0;
  }}
  .chart-title {{
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
    white-space: pre-line;
  }}
  .chart-body {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 220px;
    gap: 18px;
    align-items: start;
  }}
  svg {{
    width: 100%;
    height: 520px;
    display: block;
  }}
  .axis text {{
    fill: var(--muted);
    font-size: 14px;
  }}
  .axis path,
  .axis line {{
    stroke: rgba(55, 55, 55, 0.35);
    shape-rendering: crispEdges;
  }}
  .grid line {{
    stroke: var(--grid);
    shape-rendering: crispEdges;
  }}
  .axis-label {{
    fill: var(--text);
    font-size: 15px;
    font-weight: 600;
  }}
  .legend {{
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 2px;
  }}
  .legend button {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-height: 28px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: #fff;
    color: var(--text);
    cursor: pointer;
    font-size: 12px;
    padding: 4px 8px;
    text-align: left;
    width: 100%;
  }}
  .legend button.hidden {{
    opacity: 0.38;
    text-decoration: line-through;
  }}
  .swatch {{
    width: 22px;
    min-width: 22px;
    border-radius: 2px;
    display: inline-block;
  }}
  .tooltip {{
    position: fixed;
    z-index: 10;
    pointer-events: none;
    min-width: 190px;
    max-width: 280px;
    background: rgba(255, 255, 255, 0.98);
    border: 1px solid var(--border);
    border-radius: 7px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.14);
    padding: 9px 10px;
    color: var(--text);
    font-size: 12px;
    line-height: 1.4;
    display: none;
  }}
  .tooltip b {{
    display: block;
    font-size: 13px;
    margin-bottom: 4px;
  }}
  @media (max-width: 900px) {{
    .chart-body {{
      grid-template-columns: 1fr;
    }}
    .legend {{
      display: grid;
      grid-template-columns: 1fr;
    }}
  }}
</style>
</head>
<body>
<main>
  <h1>{page_title}</h1>
  <div id="charts"></div>
</main>
<div id="tooltip" class="tooltip"></div>
<script>
const reportRows = {payload};
const chartsEl = document.getElementById("charts");
const tooltip = document.getElementById("tooltip");

const fmtNumber = value => Number.isFinite(value) ? value.toLocaleString("en-US", {{ maximumFractionDigits: 3 }}) : "n/a";
const parseX = (figure, value) => figure.xType === "date" ? new Date(value).getTime() : Number(value);
const niceTicks = (min, max, count = 5) => {{
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {{
    const base = Number.isFinite(min) ? min : 0;
    return [base - 1, base, base + 1];
  }}
  const span = max - min;
  const step0 = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const err = step0 / mag;
  const step = err >= 7.5 ? 10 * mag : err >= 3.5 ? 5 * mag : err >= 1.5 ? 2 * mag : mag;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step * 0.5; v += step) ticks.push(v);
  return ticks;
}};
const dateTicks = (min, max, count = 6, hourStep = null) => {{
  const ticks = [];
  if (hourStep) {{
    const step = hourStep * 60 * 60 * 1000;
    let t = Math.ceil(min / step) * step;
    for (; t <= max; t += step) ticks.push(t);
    return ticks;
  }}
  const step = (max - min) / Math.max(1, count - 1);
  for (let i = 0; i < count; i++) ticks.push(min + step * i);
  return ticks;
}};

const formatXTick = (figure, value) => {{
  if (figure.xType !== "date") return fmtNumber(value);

  const d = new Date(value);
  const hour = d.getUTCHours();
  const minute = d.getUTCMinutes();

  // Second plot: show only hours
  if (figure.hoursOnlyTicks) {{
    return d.toLocaleString("en-GB", {{
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "UTC"
    }});
  }}

  // First plot: show day only at midnight, otherwise only hours
  if (figure.dayOnlyAtMidnight) {{
    if (hour === 0 && minute === 0) {{
      return d.toLocaleString("en-GB", {{
        month: "short",
        day: "2-digit",
        timeZone: "UTC"
      }});
    }}

    return d.toLocaleString("en-GB", {{
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "UTC"
    }});
  }}

  // Default: date + time
  return d.toLocaleString("en-GB", {{
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC"
  }});
}};

const linePath = points => points.map((p, i) => `${{i ? "L" : "M"}}${{p[0].toFixed(2)}} ${{p[1].toFixed(2)}}`).join(" ");
const renderPriority = trace => trace.type === "bar" ? 0 : trace.type === "band" ? 1 : trace.type === "line" ? 2 : 3;
const htmlEscape = value => String(value ?? "").replace(/[&<>"']/g, c => {{
  const escapes = {{ "&": "&amp;", "<": "&lt;", ">": "&gt;" }};
  if (c === '"') return "&quot;";
  if (c === "'") return "&#39;";
  return escapes[c];
}});

function traceDomain(figure, trace) {{
  const xs = [];
  const ys = [];
  if (trace.type === "band") {{
    trace.x.forEach((x, i) => {{
      if (trace.lower[i] !== null && trace.upper[i] !== null) {{
        xs.push(parseX(figure, x));
        ys.push(trace.lower[i], trace.upper[i]);
      }}
    }});
  }} else if (trace.type === "error_scatter") {{
    trace.x.forEach((x, i) => {{
      if (trace.y[i] !== null) {{
        xs.push(trace.xLeft[i] ?? x, trace.xRight[i] ?? x);
        ys.push(trace.y[i]);
      }}
    }});
  }} else if (trace.type === "bar") {{
    trace.x.forEach((x, i) => {{
      if (trace.y[i] !== null) {{
        xs.push(parseX(figure, x));
        ys.push(trace.y[i], 0);
      }}
    }});
  }} else {{
    trace.x.forEach((x, i) => {{
      if (trace.y[i] !== null) {{
        xs.push(parseX(figure, x));
        ys.push(trace.y[i]);
      }}
    }});
  }}
  return {{ xs, ys }};
}}

function renderFigure(figure, parent = chartsEl) {{
  const wrapper = document.createElement("section");
  wrapper.className = "chart";
  wrapper.innerHTML = `<div class="chart-title">${{htmlEscape(figure.title)}}</div><div class="chart-body"><svg></svg><div class="legend"></div></div>`;
  parent.appendChild(wrapper);

  const svg = wrapper.querySelector("svg");
  const legend = wrapper.querySelector(".legend");
  const hidden = new Set();

  figure.traces
    .map((trace, traceIndex) => ({{ trace, traceIndex }}))
    .sort((a, b) => (a.trace.legendOrder ?? a.traceIndex) - (b.trace.legendOrder ?? b.traceIndex))
    .forEach(({{ trace, traceIndex }}) => {{
    const button = document.createElement("button");
    button.type = "button";

    let swatchStyle = "";

    if (trace.type === "band") {{
      swatchStyle = `
        background:${{trace.color}};
        height:10px;
        opacity:0.28;
        border:1px solid ${{trace.color}};
      `;
    }} else if (trace.type === "bar") {{
      swatchStyle = `
        background:${{trace.color}};
        height:10px;
        opacity:${{trace.alpha ?? 0.7}};
      `;
    }} else if (trace.dash) {{
      swatchStyle = `
        background:repeating-linear-gradient(
          to right,
          ${{trace.color}} 0px,
          ${{trace.color}} 8px,
          transparent 8px,
          transparent 14px
        );
        height:3px;
      `;
    }} else {{
      swatchStyle = `
        background:${{trace.color}};
        height:3px;
      `;
    }}

    button.innerHTML = `
      <span class="swatch" style="${{swatchStyle}}"></span>
      <span>${{htmlEscape(trace.name)}}</span>
    `;

    button.addEventListener("click", () => {{
      if (hidden.has(traceIndex)) hidden.delete(traceIndex);
      else hidden.add(traceIndex);
      button.classList.toggle("hidden", hidden.has(traceIndex));
      draw();
    }});

    legend.appendChild(button);
  }});

  function draw() {{
    const rect = svg.getBoundingClientRect();
    const width = Math.max(760, rect.width || 1100);
    const height = 520;
    const margin = {{ top: 22, right: 78, bottom: figure.xTickHours ? 94 : 70, left: 78 }};
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
    svg.innerHTML = "";

    let xs = [];
    let ysLeft = [];
    let ysRight = [];
    figure.traces.forEach((trace, traceIndex) => {{
      if (hidden.has(traceIndex)) return;
      const domain = traceDomain(figure, trace);
      xs = xs.concat(domain.xs);
      if (trace.yAxis === "y2") ysRight = ysRight.concat(domain.ys);
      else ysLeft = ysLeft.concat(domain.ys);
    }});
    xs = xs.filter(Number.isFinite);
    ysLeft = ysLeft.filter(Number.isFinite);
    ysRight = ysRight.filter(Number.isFinite);
    if (!xs.length || (!ysLeft.length && !ysRight.length)) return;

    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    if (!ysLeft.length) ysLeft = ysRight.slice();
    let yMin = figure.yRange ? figure.yRange[0] : Math.min(...ysLeft);
    let yMax = figure.yRange ? figure.yRange[1] : Math.max(...ysLeft);
    if (!figure.yRange) {{
      const pad = (yMax - yMin || 1) * 0.08;
      yMin -= pad;
      yMax += pad;
    }}
    let y2Min = null;
    let y2Max = null;
    const hasY2 = ysRight.length > 0;
    if (hasY2) {{
      y2Min = figure.y2Range ? figure.y2Range[0] : Math.min(...ysRight);
      y2Max = figure.y2Range ? figure.y2Range[1] : Math.max(...ysRight);
      if (!figure.y2Range) {{
        const y2Pad = (y2Max - y2Min || 1) * 0.08;
        y2Min -= y2Pad;
        y2Max += y2Pad;
      }}
    }}
    const xPad = (xMax - xMin || 1) * 0.025;
    const x0 = xMin - xPad;
    const x1 = xMax + xPad;

    const sx = x => margin.left + ((x - x0) / (x1 - x0 || 1)) * innerW;
    const sy = y => margin.top + innerH - ((y - yMin) / (yMax - yMin || 1)) * innerH;
    const sy2 = y => margin.top + innerH - ((y - y2Min) / (y2Max - y2Min || 1)) * innerH;
    const syFor = trace => trace.yAxis === "y2" && hasY2 ? sy2 : sy;
    const make = (tag, attrs, parent = svg) => {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      parent.appendChild(el);
      return el;
    }};
    const tracePointCount = trace => {{
      if (!trace.x) return 0;
      return trace.x.length;
    }};

    const traceXValue = (trace, i) => {{
      if (!trace.x || i < 0 || i >= trace.x.length) return NaN;
      return parseX(figure, trace.x[i]);
    }};

    const traceYValue = trace => trace.yAxis === "y2" && hasY2 ? sy2 : sy;

    const traceHasValueAt = (trace, i) => {{
      if (i < 0 || i >= tracePointCount(trace)) return false;

      if (trace.type === "band") {{
        return trace.lower?.[i] !== null && trace.upper?.[i] !== null;
      }}

      if (trace.type === "error_scatter") {{
        return trace.y?.[i] !== null;
      }}

      return trace.y?.[i] !== null;
    }};

    const minTraceSpacing = trace => {{
      const values = [];

      for (let i = 0; i < tracePointCount(trace); i++) {{
        const x = traceXValue(trace, i);
        if (Number.isFinite(x)) values.push(x);
      }}

      values.sort((a, b) => a - b);

      let minSpacing = Infinity;

      for (let i = 1; i < values.length; i++) {{
        const spacing = values[i] - values[i - 1];
        if (spacing > 0 && spacing < minSpacing) minSpacing = spacing;
      }}

      return Number.isFinite(minSpacing) ? minSpacing : Infinity;
    }};

    const nearestIndexForTrace = (trace, targetX) => {{
      let bestIndex = -1;
      let bestDistance = Infinity;

      for (let i = 0; i < tracePointCount(trace); i++) {{
        if (!traceHasValueAt(trace, i)) continue;

        const x = traceXValue(trace, i);
        const distance = Math.abs(x - targetX);

        if (Number.isFinite(distance) && distance < bestDistance) {{
          bestDistance = distance;
          bestIndex = i;
        }}
      }}

      const spacing = minTraceSpacing(trace);
      const tolerance = Number.isFinite(spacing)
        ? Math.max(spacing * 0.51, 1)
        : Math.max((x1 - x0) * 0.02, 1);

      return bestDistance <= tolerance ? bestIndex : -1;
    }};

    const nearestSharedX = mouseSvgX => {{
      let bestX = NaN;
      let bestDistance = Infinity;

      figure.traces.forEach((trace, traceIndex) => {{
        if (hidden.has(traceIndex)) return;

        for (let i = 0; i < tracePointCount(trace); i++) {{
          if (!traceHasValueAt(trace, i)) continue;

          const xValue = traceXValue(trace, i);
          const px = sx(xValue);
          const distance = Math.abs(px - mouseSvgX);

          if (Number.isFinite(distance) && distance < bestDistance) {{
            bestDistance = distance;
            bestX = xValue;
          }}
        }}
      }});

      return bestX;
    }};

    const sharedHoverInfo = targetX => {{
      const info = {{}};

      if (figure.xType === "date") {{
        info.Day = new Date(targetX).toLocaleString("en-GB", {{
          year: "numeric",
          month: "short",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          timeZone: "UTC",
          timeZoneName: "short"
        }});
      }} else {{
        info.Day = fmtNumber(targetX);
      }}

      figure.traces.forEach((trace, traceIndex) => {{
        if (hidden.has(traceIndex)) return;

        const i = nearestIndexForTrace(trace, targetX);

        if (i < 0) return;

        if (trace.type === "band") {{
          info[`${{trace.name}} lower`] = fmtNumber(trace.lower[i]);
          info[`${{trace.name}} upper`] = fmtNumber(trace.upper[i]);
        }} else {{
          info[trace.name] = fmtNumber(trace.y[i]);
        }}
      }});

      return info;
    }};

    const xTicksRaw = figure.xType === "date"
      ? dateTicks(x0, x1, 6, figure.xTickHours)
      : niceTicks(x0, x1, 6);

    const xTicks = xTicksRaw.filter(t => t >= x0 && t <= x1);

    const yTicks = niceTicks(yMin, yMax, 6).filter(t => t >= yMin && t <= yMax);

    const grid = make("g", {{ class: "grid" }});
    yTicks.forEach(t => make("line", {{ x1: margin.left, x2: margin.left + innerW, y1: sy(t), y2: sy(t) }}, grid));

    const axis = make("g", {{ class: "axis" }});
    make("line", {{ x1: margin.left, x2: margin.left + innerW, y1: margin.top + innerH, y2: margin.top + innerH }}, axis);
    make("line", {{ x1: margin.left, x2: margin.left, y1: margin.top, y2: margin.top + innerH }}, axis);
    xTicks.forEach(t => {{
      const x = sx(t);
      make("line", {{ x1: x, x2: x, y1: margin.top + innerH, y2: margin.top + innerH + 5 }}, axis);
      const textAttrs = figure.xTickHours
        ? {{ x, y: margin.top + innerH + 28, "text-anchor": "end", transform: `rotate(-35 ${{x}} ${{margin.top + innerH + 28}})` }}
        : {{ x, y: margin.top + innerH + 24, "text-anchor": "middle" }};
      const txt = make("text", textAttrs, axis);
      txt.textContent = formatXTick(figure, t);
    }});
    yTicks.forEach(t => {{
      const y = sy(t);
      make("line", {{ x1: margin.left - 5, x2: margin.left, y1: y, y2: y }}, axis);
      const txt = make("text", {{ x: margin.left - 9, y: y + 4, "text-anchor": "end" }}, axis);
      txt.textContent = fmtNumber(t);
    }});
    if (hasY2) {{
      const y2Ticks = niceTicks(y2Min, y2Max, 6).filter(t => t >= y2Min && t <= y2Max);
      make("line", {{ x1: margin.left + innerW, x2: margin.left + innerW, y1: margin.top, y2: margin.top + innerH }}, axis);
      y2Ticks.forEach(t => {{
        const y = sy2(t);
        make("line", {{ x1: margin.left + innerW, x2: margin.left + innerW + 5, y1: y, y2: y }}, axis);
        const txt = make("text", {{ x: margin.left + innerW + 9, y: y + 4, "text-anchor": "start" }}, axis);
        txt.textContent = fmtNumber(t);
      }});
    }}
    make("text", {{ class: "axis-label", x: margin.left + innerW / 2, y: height - 18, "text-anchor": "middle" }}).textContent = figure.xLabel;
    make("text", {{ class: "axis-label", transform: `translate(18 ${{margin.top + innerH / 2}}) rotate(-90)`, "text-anchor": "middle" }}).textContent = figure.yLabel;
    if (hasY2 && figure.y2Label) {{
      make("text", {{ class: "axis-label", transform: `translate(${{width - 18}} ${{margin.top + innerH / 2}}) rotate(90)`, "text-anchor": "middle" }}).textContent = figure.y2Label;
    }}

    figure.traces
      .map((trace, traceIndex) => ({{ trace, traceIndex }}))
      .sort((a, b) => renderPriority(a.trace) - renderPriority(b.trace))
      .forEach(({{ trace, traceIndex }}) => {{
      if (hidden.has(traceIndex)) return;
      if (trace.type === "band") {{
        const yScale = syFor(trace);
        const upper = trace.x.map((x, i) => [sx(parseX(figure, x)), yScale(trace.upper[i])]).filter(p => p.every(Number.isFinite));
        const lower = trace.x.map((x, i) => [sx(parseX(figure, x)), yScale(trace.lower[i])]).filter(p => p.every(Number.isFinite)).reverse();
        if (upper.length && lower.length) {{
          make("path", {{ d: linePath(upper.concat(lower)) + " Z", fill: trace.color, opacity: "0.28", stroke: "none" }});
        }}
      }} else if (trace.type === "bar") {{
        const yScale = syFor(trace);
        const zeroY = yScale(0);
        const barWidth = Math.max(2, Math.abs(sx(x0 + (trace.widthMs || 720000)) - sx(x0)));
        trace.x.forEach((xRaw, i) => {{
          const x = sx(parseX(figure, xRaw));
          const y = yScale(trace.y[i]);
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(zeroY)) return;
          const top = Math.min(y, zeroY);
          const height = Math.max(1, Math.abs(zeroY - y));
          const bar = make("rect", {{ x: x - barWidth / 2, y: top, width: barWidth, height, fill: trace.color, opacity: trace.alpha ?? 0.7 }});
          if (trace.hover?.[i]) {{
            bar.addEventListener("mousemove", event => showTooltip(event, trace.hover[i]));
            bar.addEventListener("mouseleave", hideTooltip);
          }}
        }});
      }} else if (trace.type === "line") {{
        const yScale = syFor(trace);
        const pts = trace.x.map((x, i) => [sx(parseX(figure, x)), yScale(trace.y[i])]).filter(p => p.every(Number.isFinite));
        if (pts.length) {{
          make("path", {{ d: linePath(pts), fill: "none", stroke: trace.color, "stroke-width": 2.2, "stroke-dasharray": trace.dash ? "8 6" : "none" }});
          if (trace.marker) {{
            pts.forEach(p => {{
              make("circle", {{ cx: p[0], cy: p[1], r: 4, fill: trace.color, opacity: 0.95, stroke: "#fff", "stroke-width": 1.2 }});
            }});
          }}
        }}
      }} else if (trace.type === "scatter" || trace.type === "error_scatter") {{
        const yScale = syFor(trace);
        trace.x.forEach((xRaw, i) => {{
          const x = sx(parseX(figure, xRaw));
          const y = yScale(trace.y[i]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) return;
          if (trace.type === "error_scatter") {{
            const xl = sx(trace.xLeft[i]);
            const xr = sx(trace.xRight[i]);
            if (Number.isFinite(xl) && Number.isFinite(xr)) {{
              make("line", {{ x1: xl, x2: xr, y1: y, y2: y, stroke: trace.color, "stroke-width": 1.8, opacity: 0.85 }});
              make("line", {{ x1: xl, x2: xl, y1: y - 5, y2: y + 5, stroke: trace.color, "stroke-width": 1.8, opacity: 0.85 }});
              make("line", {{ x1: xr, x2: xr, y1: y - 5, y2: y + 5, stroke: trace.color, "stroke-width": 1.8, opacity: 0.85 }});
            }}
          }}
          const dot = make("circle", {{ cx: x, cy: y, r: trace.size || 6, fill: trace.color, opacity: 0.9, stroke: "#fff", "stroke-width": 1.3 }});
          dot.addEventListener("mousemove", event => showTooltip(event, trace.hover[i]));
          dot.addEventListener("mouseleave", hideTooltip);
        }});
      }}
    }});
    make("rect", {{ x: margin.left, y: margin.top, width: innerW, height: innerH, fill: "none", stroke: "rgba(55, 55, 55, 0.45)", "stroke-width": 1 }});

    if (figure.sharedHover) {{
      const hoverLayer = make("rect", {{
        x: margin.left,
        y: margin.top,
        width: innerW,
        height: innerH,
        fill: "transparent",
        stroke: "none"
      }});

      const distanceToSegment = (px, py, ax, ay, bx, by) => {{
        const dx = bx - ax;
        const dy = by - ay;

        if (dx === 0 && dy === 0) {{
          return Math.hypot(px - ax, py - ay);
        }}

        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)));
        const closestX = ax + t * dx;
        const closestY = ay + t * dy;

        return Math.hypot(px - closestX, py - closestY);
      }};

      const isNearPlottedFeature = (mouseSvgX, mouseSvgY) => {{
        const pointTolerance = 8;
        const lineTolerance = 7;

        for (const [traceIndex, trace] of figure.traces.entries()) {{
          if (hidden.has(traceIndex)) continue;

          if (trace.type === "bar") {{
            const yScale = syFor(trace);
            const zeroY = yScale(0);
            const barWidth = Math.max(2, Math.abs(sx(x0 + (trace.widthMs || 720000)) - sx(x0)));

            for (let i = 0; i < trace.x.length; i++) {{
              if (trace.y[i] === null) continue;

              const x = sx(parseX(figure, trace.x[i]));
              const y = yScale(trace.y[i]);

              if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(zeroY)) continue;

              const left = x - barWidth / 2;
              const right = x + barWidth / 2;
              const top = Math.min(y, zeroY);
              const bottom = Math.max(y, zeroY);

              if (
                mouseSvgX >= left &&
                mouseSvgX <= right &&
                mouseSvgY >= top &&
                mouseSvgY <= bottom
              ) {{
                return true;
              }}
            }}
          }} else if (trace.type === "line") {{
            const yScale = syFor(trace);
            const pts = trace.x
              .map((x, i) => [sx(parseX(figure, x)), yScale(trace.y[i])])
              .filter(p => p.every(Number.isFinite));

            for (let i = 1; i < pts.length; i++) {{
              if (distanceToSegment(mouseSvgX, mouseSvgY, pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1]) <= lineTolerance) {{
                return true;
              }}
            }}

            if (trace.marker) {{
              for (const p of pts) {{
                if (Math.hypot(mouseSvgX - p[0], mouseSvgY - p[1]) <= pointTolerance) {{
                  return true;
                }}
              }}
            }}
          }} else if (trace.type === "band") {{
            const yScale = syFor(trace);
            const upper = trace.x
              .map((x, i) => [sx(parseX(figure, x)), yScale(trace.upper[i])])
              .filter(p => p.every(Number.isFinite));
            const lower = trace.x
              .map((x, i) => [sx(parseX(figure, x)), yScale(trace.lower[i])])
              .filter(p => p.every(Number.isFinite));

            for (let i = 1; i < upper.length; i++) {{
              if (distanceToSegment(mouseSvgX, mouseSvgY, upper[i - 1][0], upper[i - 1][1], upper[i][0], upper[i][1]) <= lineTolerance) {{
                return true;
              }}
            }}

            for (let i = 1; i < lower.length; i++) {{
              if (distanceToSegment(mouseSvgX, mouseSvgY, lower[i - 1][0], lower[i - 1][1], lower[i][0], lower[i][1]) <= lineTolerance) {{
                return true;
              }}
            }}
          }} else if (trace.type === "scatter" || trace.type === "error_scatter") {{
            const yScale = syFor(trace);

            for (let i = 0; i < trace.x.length; i++) {{
              const x = sx(parseX(figure, trace.x[i]));
              const y = yScale(trace.y[i]);

              if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

              if (Math.hypot(mouseSvgX - x, mouseSvgY - y) <= pointTolerance) {{
                return true;
              }}

              if (trace.type === "error_scatter") {{
                const xl = sx(trace.xLeft[i]);
                const xr = sx(trace.xRight[i]);

                if (
                  Number.isFinite(xl) &&
                  Number.isFinite(xr) &&
                  distanceToSegment(mouseSvgX, mouseSvgY, xl, y, xr, y) <= lineTolerance
                ) {{
                  return true;
                }}
              }}
            }}
          }}
        }}

        return false;
      }};

      hoverLayer.addEventListener("mousemove", event => {{
        const svgRect = svg.getBoundingClientRect();
        const mouseSvgX = (event.clientX - svgRect.left) * width / svgRect.width;
        const mouseSvgY = (event.clientY - svgRect.top) * height / svgRect.height;

        if (
          mouseSvgX < margin.left ||
          mouseSvgX > margin.left + innerW ||
          mouseSvgY < margin.top ||
          mouseSvgY > margin.top + innerH ||
          !isNearPlottedFeature(mouseSvgX, mouseSvgY)
        ) {{
          hideTooltip();
          return;
        }}

        const targetX = nearestSharedX(mouseSvgX);

        if (!Number.isFinite(targetX)) {{
          hideTooltip();
          return;
        }}

        showTooltip(event, sharedHoverInfo(targetX));
      }});

      hoverLayer.addEventListener("mouseleave", hideTooltip);
    }}

  }}

  function showTooltip(event, info) {{
    const rows = Object.entries(info || {{}})
      .filter(([k]) => k !== "Day")
      .map(([k, v]) => `<div><strong>${{htmlEscape(k)}}:</strong> ${{htmlEscape(v)}}</div>`)
      .join("");
    tooltip.innerHTML = `<b>${{htmlEscape(info?.Day || "")}}</b>${{rows}}`;
    tooltip.style.display = "block";
    tooltip.style.left = `${{Math.min(window.innerWidth - 300, event.clientX + 14)}}px`;
    tooltip.style.top = `${{Math.min(window.innerHeight - 160, event.clientY + 14)}}px`;
  }}
  function hideTooltip() {{
    tooltip.style.display = "none";
  }}

  draw();
  window.addEventListener("resize", draw);
}}

function renderReportRow(row) {{
  const section = document.createElement("section");
  section.className = "report-row";
  section.innerHTML = `<div class="row-grid"></div>`;
  chartsEl.appendChild(section);

  const grid = section.querySelector(".row-grid");
  row.figures.forEach(figure => renderFigure(figure, grid));
}}

reportRows.forEach(renderReportRow);
</script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def upsert_tracking_csv(path, new_rows, key_cols):
    new_df = pd.DataFrame(new_rows)

    if new_df.empty:
        return new_df

    if path.exists():
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols).reset_index(drop=True)
    combined.to_csv(path, index=False)

    return combined


def upsert_report_data_csv(path, new_rows):
    new_df = pd.DataFrame(new_rows)

    if new_df.empty:
        return pd.DataFrame()

    if path.exists():
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined = combined.drop_duplicates(
        subset=["row_type", "Datetime UTC", "model"],
        keep="last",
    )
    combined = combined.sort_values(["row_type", "Datetime UTC", "model"]).reset_index(drop=True)
    combined.to_csv(path, index=False)

    return combined


def rank_rmse_percent(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(valid) < 3 or valid["x"].nunique() <= 1 or valid["y"].nunique() <= 1:
        return np.nan

    x_rank = valid["x"].rank(method="average", pct=True)
    y_rank = valid["y"].rank(method="average", pct=True)

    return float(np.sqrt(np.mean((x_rank - y_rank) ** 2)) * 100)


def load_tracked_yesterday_prediction(path, model, yesterday_date):
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame()

    target_date = pd.Timestamp(yesterday_date).strftime("%Y-%m-%d")

    df = df[
        (df["model"] == model)
        & (df["forecast_run_date"] == target_date)
        & (df["target_date"] == target_date)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["Datetime UTC"] = pd.to_datetime(df["target_date"], utc=True)
    df["Day"] = df["Datetime UTC"].apply(english_day_label)
    df["Risk score"] = df["predicted_risk_score"].map(lambda v: f"{v:.3f}")
    df["q95 range"] = df.apply(
        lambda row: f"{row['daily_error_q95_left']:.3f} to {row['daily_error_q95_right']:.3f}",
        axis=1,
    )

    return df


# Load data
Folder = str(DATA_DIR)
df_rad = pd.read_csv(os.path.join(Folder, "station_radiation_CH_interp_year_2026.csv"), sep=";")
df_metrics = pd.read_csv(os.path.join(Folder, "station_metrics_CH_interp_year_2026.csv"), sep=";")

# Datetime
df_rad["Datetime UTC"] = pd.to_datetime(df_rad["Datetime"], utc=True)
df_rad = df_rad.drop(columns="Datetime")

df_metrics["Datetime UTC"] = pd.to_datetime(df_metrics["Datetime"], utc=True)
df_metrics = df_metrics.drop(columns="Datetime")


# =========================
# Select date range
# =========================
today_start_utc = pd.Timestamp.now(tz="UTC").normalize()
start_dt_hist = pd.to_datetime("2026-03-21", utc=True)
end_dt_hist = today_start_utc          # exclusive: includes full yesterday
start_dt_future = today_start_utc      # inclusive: first future timestamp is today 00:00 UTC
end_dt_future = pd.to_datetime(END_DATE, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

df_rad_hist = df_rad[(df_rad["Datetime UTC"] >= start_dt_hist) & (df_rad["Datetime UTC"] < end_dt_hist)].copy()
df_metrics_hist = df_metrics[(df_metrics["Datetime UTC"] >= start_dt_hist) & (df_metrics["Datetime UTC"] < end_dt_hist)].copy()

df_rad_future = df_rad[(df_rad["Datetime UTC"] >= start_dt_future) & (df_rad["Datetime UTC"] <= end_dt_future)].copy()
df_metrics_future = df_metrics[(df_metrics["Datetime UTC"] >= start_dt_future) & (df_metrics["Datetime UTC"] <= end_dt_future)].copy()

# =========================
# Figure: MeteoSwiss stations and previous-day model values
# =========================
prev_day_cols = [
    "MeteoSwiss stations",
    f"{MODEL} prev day 0",
    f"{MODEL} prev day 1",
    f"{MODEL} prev day 2",
    f"{MODEL} prev day 3",
]

prev_day_colors = {
    "MeteoSwiss stations": elcom_colors["dark_blue"],
    f"{MODEL} prev day 0": elcom_colors["green"],
    f"{MODEL} prev day 1": elcom_colors["orange"],
    f"{MODEL} prev day 2": elcom_colors["red"],
    f"{MODEL} prev day 3": elcom_colors["lila"],
}


# ==================================
# Fit all historical data
# ==================================

# =========================
# Build features historical data
# =========================
# Historical from 21. march until yesterday = START_DATE
df_features_hist, feature_info = build_risk_feature_dataframe(
    df_rad=df_rad_hist,
    df_metrics=df_metrics_hist,
    model=MODEL,
    start_date=start_dt_hist.strftime("%Y-%m-%d"),
    end_date=end_dt_hist.strftime("%Y-%m-%d"),
    ghi_threshold=GHI_threshold,
    target_horizons=(0,),
)

# =========================
# Fit risk score
# =========================
fit_result, plot_context = fit_risk_score_from_features(
    df_features=df_features_hist,
    feature_cols=feature_info["feature_cols"],
    metric_cols=feature_info["metric_cols"],
    model=feature_info["model"],
    start_date=feature_info["start_date"],
    end_date=feature_info["end_date"],
    ghi_threshold=feature_info["ghi_threshold"],
    figsize=FIGSIZE,
    horizon_colors=feature_info["horizon_colors"],
    use_rank_features=True,
    nonnegative_weights=True,
    verbose=True,
    top_risk_penalty_weight=0.5,
    top_quantile=0.9,
)

feature_cols_hist = fit_result["feature_cols"]
fitted_model_info_hist = fit_result["hourly_model_info"]


# ===================================
# Apply fitted model to future data
# ===================================

df_rad_future_hybrid = fill_model_gaps_with_backup_model(
    df=df_rad_future,
    primary_model="ICON1",
    backup_model="ICON2",
)

# Future from today until END_DATE
df_features_future, feature_info = build_risk_feature_dataframe(
    df_rad=df_rad_future_hybrid,
    df_metrics=df_metrics_future,
    model=MODEL,
    start_date=date.today(),
    end_date=END_DATE,
    ghi_threshold=GHI_threshold,
    target_horizons=(0,),
)

feature_cols = fit_result["feature_cols"]

# Historical forecast error column for day 0
error_col = f"MeteoSwiss - {MODEL}_d0"

# Make sure the historical error exists
df_rad_hist_error = df_rad_hist.copy()
df_rad_hist_error[error_col] = df_rad_hist_error["MeteoSwiss stations"] - df_rad_hist_error[f"{MODEL} prev day 0"]

df_features_future = compute_future_quantile_error_band(
    df_error_hist=df_rad_hist_error,
    df_features_hist=df_features_hist,
    df_features_future=df_features_future,
    fit_result=fit_result,
    error_col=error_col,
    risk_window=risk_window,
    quantile=quantile,
    min_samples=min_samples,
    smooth=True,
    enforce_monotonic=True,
)


# ============================
# Plots
# ============================

now_utc = pd.Timestamp.now(tz="UTC")
today_start_utc = now_utc.normalize()
yesterday_start_utc = today_start_utc - pd.Timedelta(days=1)

future_col = f"{MODEL} prev day 0"

hist_df_until_now = df_rad[(df_rad["Datetime UTC"] >= yesterday_start_utc) & (df_rad["Datetime UTC"] <= now_utc)].copy()
hist_df_yesterday_only = df_rad[(df_rad["Datetime UTC"] >= yesterday_start_utc) & (df_rad["Datetime UTC"] < today_start_utc)].copy()
future_df = df_rad_future[df_rad_future["Datetime UTC"] >= today_start_utc].copy()

future_band_df = df_features_future[df_features_future["Datetime UTC"] >= today_start_utc][["Datetime UTC", "q95_error_band"]].copy()

future_df = future_df.merge(future_band_df, on="Datetime UTC", how="left")
future_df["upper_q95"] = future_df[future_col] + future_df["q95_error_band"]
future_df["lower_q95"] = (future_df[future_col] - future_df["q95_error_band"]).clip(lower=0)

line_traces = []

for col in prev_day_cols:

    if col == "MeteoSwiss stations":
        plot_df = hist_df_until_now
    else:
        plot_df = hist_df_yesterday_only

    line_traces.append(
        time_series_trace(
            plot_df,
            x_col="Datetime UTC",
            y_col=col,
            name=col,
            color=rgb_to_css(prev_day_colors[col]),
        )
    )

line_traces.append(
    time_series_trace(
        future_df,
        x_col="Datetime UTC",
        y_col=future_col,
        name=f"{MODEL} forecast",
        color=rgb_to_css(prev_day_colors[future_col]),
        dash=True,
    )
)

line_traces.append(
    band_trace(
        future_df,
        x_col="Datetime UTC",
        lower_col="lower_q95",
        upper_col="upper_q95",
        name="q95 forecast-error band",
        color=rgb_to_css(prev_day_colors[future_col]),
    )
)

line_y_values = []
for col in prev_day_cols:
    source_df = hist_df_until_now if col == "MeteoSwiss stations" else hist_df_yesterday_only
    line_y_values.extend(source_df[col].dropna().tolist())
line_y_values.extend(future_df[future_col].dropna().tolist())
line_y_values.extend(future_df["upper_q95"].dropna().tolist())
line_y_max = max(1, max(line_y_values) * 1.05) if line_y_values else 1

line_figure = {
    "title": "Previous-Day Model Runs, Actual GHI and Forecasted GHI with errorband",
    "xType": "date",
    "xTickHours": 6,
    "dayOnlyAtMidnight": True,
    "sharedHover": True,
    "xLabel": "Datetime UTC",
    "yLabel": "GHI values [W/m^2]",
    "yRange": [0, clean_float(line_y_max)],
    "traces": line_traces,
}

# =========================
# Get imbalances for yesterday
# =========================

url = "https://www.swissgrid.ch/dam/dataimport/control-area-balance/control-area-balance-2026.csv"

swissgrid_path = DOWNLOADS_DIR / "control-area-balance-2026.csv"

try:
    print(f"Downloading latest Swissgrid control-area balance file to:\n{swissgrid_path}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    swissgrid_path.write_bytes(response.content)

    print("Swissgrid file updated successfully.")

except requests.RequestException as e:
    print(f"Could not download latest Swissgrid file: {e}")

    if not swissgrid_path.exists():
        raise FileNotFoundError("Swissgrid file could not be downloaded and no local fallback file exists.")

    print(f"Using existing local fallback file:\n{swissgrid_path}")

print(f"Loading Swissgrid file:\n{swissgrid_path}")

df_CAB = pd.read_csv(swissgrid_path, sep=";", encoding="utf-8-sig")

df_CAB = df_CAB[["Date Time [UTC]", "Total System Imbalance"]]
df_CAB["Date Time [UTC]"] = pd.to_datetime(df_CAB["Date Time [UTC]"], dayfirst=True, utc=True)

today_start = pd.Timestamp.now(tz="UTC").normalize()
yesterday_start = today_start - pd.Timedelta(days=1)

df_CAB = df_CAB[(df_CAB["Date Time [UTC]"] >= yesterday_start) & (df_CAB["Date Time [UTC]"] < today_start)].copy()

# =========================
# HTML plot: imbalances and GHI metrics
# =========================

df_rad_yesterday = df_rad[(df_rad["Datetime UTC"] >= yesterday_start) & (df_rad["Datetime UTC"] < today_start)].copy()

metric_cols = []

for d in range(4):
    col = f"MeteoSwiss - {MODEL}_d{d}"
    prev_col = f"{MODEL} prev day {d}"
    df_rad_yesterday[col] = df_rad_yesterday["MeteoSwiss stations"] - df_rad_yesterday[prev_col]
    metric_cols.append(col)

horizon_colors = {
    f"MeteoSwiss - {MODEL}_d0": elcom_colors["green"],
    f"MeteoSwiss - {MODEL}_d1": elcom_colors["orange"],
    f"MeteoSwiss - {MODEL}_d2": elcom_colors["red"],
    f"MeteoSwiss - {MODEL}_d3": elcom_colors["lila"],
}

dLmin = df_rad_yesterday[metric_cols].min().min()
dLmax = df_rad_yesterday[metric_cols].max().max()
dRmin = df_CAB["Total System Imbalance"].min()
dRmax = df_CAB["Total System Imbalance"].max()

f0 = choose_f0(dLmin, dLmax, dRmin, dRmax, marginL=1.05, marginR=1.05, wL=0.8, wR=1.2)
yL_min, yL_max, _ = limits_for_f0(dLmin, dLmax, f0, margin=1.05)
yR_min, yR_max, _ = limits_for_f0(dRmin, dRmax, f0, margin=1.05)

imbalance_traces = []

for col in metric_cols:
    imbalance_traces.append(
        time_series_trace(
            df_rad_yesterday,
            x_col="Datetime UTC",
            y_col=col,
            name=col,
            color=rgb_to_css(horizon_colors[col]),
        )
    )

df_CAB["Day"] = df_CAB["Date Time [UTC]"].apply(english_day_label)
df_CAB["Time UTC"] = df_CAB["Date Time [UTC]"].dt.strftime("%H:%M")
df_CAB["Imbalance [MW]"] = df_CAB["Total System Imbalance"].map(lambda v: f"{v:.1f}")

imbalance_traces.append(
    bar_trace(
        df_CAB,
        x_col="Date Time [UTC]",
        y_col="Total System Imbalance",
        name="Imbalance",
        color=rgb_to_css(elcom_colors["grey"]),
        hover_columns=["Day", "Time UTC", "Imbalance [MW]"],
        y_axis="y2",
        width_minutes=12,
        alpha=0.7,
    )
)

plot_date = yesterday_start.strftime("%d.%m.%Y")
imbalance_figure = {
    "title": f"GHI forecast error and Imbalances on {plot_date}",
    "xType": "date",
    "xTickHours": 1,
    "hoursOnlyTicks": True,
    "sharedHover": True,
    "xLabel": "Datetime UTC",
    "yLabel": "GHI forecast error CH",
    "y2Label": "Imbalances [MW]",
    "yRange": [clean_float(yL_min), clean_float(yL_max)],
    "y2Range": [clean_float(yR_min), clean_float(yR_max)],
    "traces": imbalance_traces,
}


# ============================================================
# Hourly historical fitted-risk scatterplot + performance
# ============================================================

hourly_error_col = f"MeteoSwiss - {MODEL}_d0"
hourly_abs_error_col = f"{hourly_error_col} hourly abs error"
hourly_output_col = fit_result["output_col"]
yesterday_date = today_start_utc - pd.Timedelta(days=1)
forecast_run_date = today_start_utc.strftime("%Y-%m-%d")

tmp_hourly_cols = ["Datetime UTC", hourly_error_col, hourly_output_col] + feature_info["feature_cols"]
tmp_hourly = plot_context["df_plot"][tmp_hourly_cols].dropna(subset=["Datetime UTC", hourly_error_col, hourly_output_col]).copy()
tmp_hourly = tmp_hourly[tmp_hourly[hourly_output_col] != 0].copy()
tmp_hourly[hourly_abs_error_col] = tmp_hourly[hourly_error_col].abs()

hourly_x_error = tmp_hourly[hourly_abs_error_col]
hourly_y_risk = tmp_hourly[hourly_output_col].abs()

if len(tmp_hourly) >= 3 and hourly_x_error.nunique() > 1 and hourly_y_risk.nunique() > 1:
    hourly_kendall_tau = hourly_x_error.corr(hourly_y_risk, method="kendall")
else:
    hourly_kendall_tau = np.nan

hourly_kendall_tau_label = f"{100 * hourly_kendall_tau:.1f}%" if np.isfinite(hourly_kendall_tau) else "n/a"

tmp_hourly["Datetime UTC"] = pd.to_datetime(tmp_hourly["Datetime UTC"], utc=True)
tmp_hourly["is_yesterday"] = tmp_hourly["Datetime UTC"].dt.floor("D") == yesterday_date
tmp_hourly["Day"] = tmp_hourly["Datetime UTC"].apply(english_day_label)
tmp_hourly["Time UTC"] = tmp_hourly["Datetime UTC"].dt.strftime("%H:%M")
tmp_hourly["Risk score"] = tmp_hourly[hourly_output_col].map(lambda v: f"{v:.3f}")
tmp_hourly["Forecast error"] = tmp_hourly[hourly_abs_error_col].map(lambda v: f"{v:.3f}")

hourly_y_max = float(hourly_y_risk.max()) if not hourly_y_risk.empty else 1.0

df_rad_pred_hist = df_rad[(df_rad["Datetime UTC"] >= start_dt_hist) & (df_rad["Datetime UTC"] < yesterday_start_utc)].copy()
df_metrics_pred_hist = df_metrics[(df_metrics["Datetime UTC"] >= start_dt_hist) & (df_metrics["Datetime UTC"] < yesterday_start_utc)].copy()
df_metrics_yesterday = df_metrics[(df_metrics["Datetime UTC"] >= yesterday_start_utc) & (df_metrics["Datetime UTC"] < today_start_utc)].copy()

df_features_pred_hist, feature_info_pred = build_risk_feature_dataframe(
    df_rad=df_rad_pred_hist,
    df_metrics=df_metrics_pred_hist,
    model=MODEL,
    start_date=start_dt_hist.strftime("%Y-%m-%d"),
    end_date=yesterday_start_utc.strftime("%Y-%m-%d"),
    ghi_threshold=GHI_threshold,
    target_horizons=(0,),
)

fit_result_pred, _ = fit_risk_score_from_features(
    df_features=df_features_pred_hist,
    feature_cols=feature_info_pred["feature_cols"],
    metric_cols=feature_info_pred["metric_cols"],
    model=feature_info_pred["model"],
    start_date=feature_info_pred["start_date"],
    end_date=feature_info_pred["end_date"],
    ghi_threshold=feature_info_pred["ghi_threshold"],
    figsize=FIGSIZE,
    horizon_colors=feature_info_pred["horizon_colors"],
    use_rank_features=True,
    nonnegative_weights=True,
    verbose=True,
    top_risk_penalty_weight=0.5,
    top_quantile=0.9,
)

df_features_yesterday_pred, _ = build_risk_feature_dataframe(
    df_rad=df_rad_yesterday,
    df_metrics=df_metrics_yesterday,
    model=MODEL,
    start_date=yesterday_start_utc.strftime("%Y-%m-%d"),
    end_date=today_start_utc.strftime("%Y-%m-%d"),
    ghi_threshold=GHI_threshold,
    target_horizons=(0,),
)

df_rad_pred_hist_error = df_rad_pred_hist.copy()
df_rad_pred_hist_error[hourly_error_col] = (
    df_rad_pred_hist_error["MeteoSwiss stations"] - df_rad_pred_hist_error[f"{MODEL} prev day 0"]
)

df_features_yesterday_pred = compute_future_quantile_error_band(
    df_error_hist=df_rad_pred_hist_error,
    df_features_hist=df_features_pred_hist,
    df_features_future=df_features_yesterday_pred,
    fit_result=fit_result_pred,
    error_col=hourly_error_col,
    risk_window=risk_window,
    quantile=quantile,
    min_samples=min_samples,
    smooth=True,
    enforce_monotonic=True,
)

pred_model_info = fit_result_pred["hourly_model_info"]
pred_feature_cols = fit_result_pred["feature_cols"]
pred_weights = np.asarray(pred_model_info["weights"], dtype=float)

X_pred_hist = df_features_pred_hist[pred_feature_cols].copy()

if fit_result_pred.get("use_rank_features", False):
    X_pred_hist = X_pred_hist.rank(method="average", pct=True)

pred_hist_raw_risk = X_pred_hist.values @ pred_weights

if "intercept" in pred_model_info:
    pred_hist_raw_risk += float(pred_model_info["intercept"])

pred_hist_raw_risk = np.sort(pred_hist_raw_risk[np.isfinite(pred_hist_raw_risk)])

X_yesterday_pred = df_features_yesterday_pred[pred_feature_cols].copy()

if fit_result_pred.get("use_rank_features", False):
    X_yesterday_ranked = X_yesterday_pred.copy()

    for col in pred_feature_cols:
        hist_values = df_features_pred_hist[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        hist_values = np.sort(hist_values)
        future_values = X_yesterday_pred[col].to_numpy(dtype=float)

        X_yesterday_ranked[col] = [
            np.searchsorted(hist_values, val, side="right") / len(hist_values)
            if len(hist_values) > 0 and np.isfinite(val)
            else np.nan
            for val in future_values
        ]

    X_yesterday_pred = X_yesterday_ranked

yesterday_raw_risk = X_yesterday_pred.values @ pred_weights

if "intercept" in pred_model_info:
    yesterday_raw_risk += float(pred_model_info["intercept"])

if len(pred_hist_raw_risk) > 0:
    df_features_yesterday_pred[hourly_output_col] = [
        100 * np.searchsorted(pred_hist_raw_risk, val, side="right") / len(pred_hist_raw_risk)
        if np.isfinite(val)
        else np.nan
        for val in yesterday_raw_risk
    ]
else:
    df_features_yesterday_pred[hourly_output_col] = np.nan

df_features_yesterday_pred["hourly_error_q95_left"] = 0.0
df_features_yesterday_pred["hourly_error_q95_right"] = df_features_yesterday_pred["q95_error_band"]
df_features_yesterday_pred["hourly_error_q95_mid"] = 0.5 * df_features_yesterday_pred["q95_error_band"]
df_features_yesterday_pred["Day"] = df_features_yesterday_pred["Datetime UTC"].apply(english_day_label)
df_features_yesterday_pred["Time UTC"] = df_features_yesterday_pred["Datetime UTC"].dt.strftime("%H:%M")
df_features_yesterday_pred["Risk score"] = df_features_yesterday_pred[hourly_output_col].map(lambda v: f"{v:.3f}")
df_features_yesterday_pred["q95 range"] = df_features_yesterday_pred["q95_error_band"].map(lambda v: f"0.000 to {v:.3f}")

predicted_hourly_output_col = f"predicted {hourly_output_col}"

yesterday_actual = tmp_hourly.loc[tmp_hourly["is_yesterday"]].copy()
yesterday_actual = yesterday_actual.drop(columns=["Risk score"], errors="ignore")
yesterday_actual = yesterday_actual.merge(
    df_features_yesterday_pred[["Datetime UTC", hourly_output_col]].rename(
        columns={hourly_output_col: predicted_hourly_output_col}
    ),
    on="Datetime UTC",
    how="inner",
)
yesterday_actual["Risk score"] = yesterday_actual[predicted_hourly_output_col].map(lambda v: f"{v:.3f}")

yesterday_y_values = pd.concat(
    [
        yesterday_actual[predicted_hourly_output_col].abs(),
        df_features_yesterday_pred[hourly_output_col].abs(),
    ],
    ignore_index=True,
).dropna()

yesterday_y_max = float(yesterday_y_values.max()) if not yesterday_y_values.empty else 1.0

yesterday_hourly_scatter_figure = {
    "title": "Yesterday predicted risk scores with error band and actual forecast errors",
    "xType": "linear",
    "xLabel": "Hourly abs. GHI forecast error",
    "yLabel": "Hourly fitted risk score",
    "yRange": [0, clean_float(max(1.0, yesterday_y_max * 1.05))],
    "traces": [
        error_scatter_trace(
            df_features_yesterday_pred,
            x_col="hourly_error_q95_mid",
            y_col=hourly_output_col,
            x_left_col="hourly_error_q95_left",
            x_right_col="hourly_error_q95_right",
            name="Yesterday prediction",
            color=rgb_to_css(elcom_colors["orange"]),
            hover_columns=["Day", "Time UTC", "Risk score", "q95 range"],
            size=5,
        ),
        scatter_trace(
            yesterday_actual,
            x_col=hourly_abs_error_col,
            y_col=predicted_hourly_output_col,
            name="Yesterday actual",
            color=rgb_to_css(elcom_colors["dark_blue"]),
            hover_columns=["Day", "Time UTC", "Risk score", "Forecast error"],
            size=5,
        ),
    ],
}

hourly_scatter_figure = {
    "title": (
        "Historical hourly fitted risk score vs abs. forecast error"
    ),
    "xType": "linear",
    "xLabel": "Hourly abs. GHI forecast error",
    "yLabel": "Hourly fitted risk score",
    "yRange": [0, clean_float(max(1.0, hourly_y_max * 1.05))],
    "traces": [
        scatter_trace(
            tmp_hourly,
            x_col=hourly_abs_error_col,
            y_col=hourly_output_col,
            name="Historical",
            color=rgb_to_css(elcom_colors["green"]),
            hover_columns=["Day", "Time UTC", "Risk score", "Forecast error"],
            size=4,
        ),
    ],
}

current_hourly_rank_rmse = rank_rmse_percent(
    tmp_hourly[hourly_abs_error_col],
    tmp_hourly[hourly_output_col].abs(),
)

historical_performance_percent = (
    100 * (hourly_kendall_tau + 1) / 2
    if np.isfinite(hourly_kendall_tau)
    else np.nan
)

yesterday_ordering_tau = yesterday_actual[hourly_abs_error_col].corr(
    yesterday_actual[predicted_hourly_output_col],
    method="kendall",
) if (
    len(yesterday_actual) >= 3
    and yesterday_actual[hourly_abs_error_col].nunique() > 1
    and yesterday_actual[predicted_hourly_output_col].nunique() > 1
) else np.nan

yesterday_ordering_tau_percent = 100 * yesterday_ordering_tau if np.isfinite(yesterday_ordering_tau) else np.nan

report_rows = []

report_ghi_df = df_rad[
    (df_rad["Datetime UTC"] >= yesterday_start_utc)
    & (df_rad["Datetime UTC"] <= end_dt_future)
].copy()

report_ghi_df = report_ghi_df.merge(
    df_features_future[["Datetime UTC", "q95_error_band"]],
    on="Datetime UTC",
    how="left",
)
report_ghi_df["q95_upper"] = report_ghi_df[f"{MODEL} prev day 0"] + report_ghi_df["q95_error_band"]
report_ghi_df["q95_lower"] = (report_ghi_df[f"{MODEL} prev day 0"] - report_ghi_df["q95_error_band"]).clip(lower=0)

for _, row in report_ghi_df.iterrows():
    out = {
        "row_type": "ghi",
        "Datetime UTC": pd.Timestamp(row["Datetime UTC"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": MODEL,
    }

    for col in [
        "MeteoSwiss stations",
        f"{MODEL} prev day 0",
        f"{MODEL} prev day 1",
        f"{MODEL} prev day 2",
        f"{MODEL} prev day 3",
        "q95_error_band",
        "q95_lower",
        "q95_upper",
    ]:
        out[col] = clean_float(row[col]) if col in row else None

    for d in range(4):
        err_col = f"MeteoSwiss - {MODEL}_d{d}"
        prev_col = f"{MODEL} prev day {d}"
        out[err_col] = clean_float(row["MeteoSwiss stations"] - row[prev_col]) if pd.notna(row.get("MeteoSwiss stations")) and pd.notna(row.get(prev_col)) else None

    report_rows.append(out)

for _, row in df_CAB.iterrows():
    report_rows.append(
        {
            "row_type": "imbalance",
            "Datetime UTC": pd.Timestamp(row["Date Time [UTC]"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": MODEL,
            "Total System Imbalance": clean_float(row["Total System Imbalance"]),
        }
    )

yesterday_prediction_rows = df_features_yesterday_pred[
    ["Datetime UTC", hourly_output_col, "q95_error_band", "hourly_error_q95_left", "hourly_error_q95_mid", "hourly_error_q95_right"] + pred_feature_cols
].merge(
    yesterday_actual[["Datetime UTC", hourly_abs_error_col, predicted_hourly_output_col]],
    on="Datetime UTC",
    how="left",
)

for _, row in yesterday_prediction_rows.iterrows():
    out = {
            "row_type": "hourly_prediction",
            "Datetime UTC": pd.Timestamp(row["Datetime UTC"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": MODEL,
            "predicted_hourly_risk_score": clean_float(row[hourly_output_col]),
            "q95_error_band": clean_float(row["q95_error_band"]),
            "hourly_error_q95_left": clean_float(row["hourly_error_q95_left"]),
            "hourly_error_q95_mid": clean_float(row["hourly_error_q95_mid"]),
            "hourly_error_q95_right": clean_float(row["hourly_error_q95_right"]),
            "actual_abs_hourly_forecast_error": clean_float(row[hourly_abs_error_col]),
            "actual_y_predicted_risk_score": clean_float(row[predicted_hourly_output_col]),
        }

    for col in pred_feature_cols:
        out[col] = clean_float(row[col])

    report_rows.append(out)

for _, row in tmp_hourly.iterrows():
    out = {
            "row_type": "historical_scatter",
            "Datetime UTC": pd.Timestamp(row["Datetime UTC"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": MODEL,
            "actual_abs_hourly_forecast_error": clean_float(row[hourly_abs_error_col]),
            "fitted_hourly_risk_score": clean_float(row[hourly_output_col]),
        }

    for col in feature_info["feature_cols"]:
        out[col] = clean_float(row[col])

    report_rows.append(out)

report_rows.append(
    {
        "row_type": "yesterday_ordering_performance",
        "Datetime UTC": yesterday_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": MODEL,
        "kendall_tau_percent": clean_float(yesterday_ordering_tau_percent),
        "n_hours": int(len(yesterday_actual)),
    }
)

report_rows.append(
    {
        "row_type": "historical_performance",
        "Datetime UTC": yesterday_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": MODEL,
        "kendall_tau_percent": clean_float(historical_performance_percent),
        "rank_rmse_percent": clean_float(current_hourly_rank_rmse),
        "n_hours": int(len(tmp_hourly)),
    }
)

df_report = upsert_report_data_csv(
    REPORT_DATA_PATH,
    report_rows,
)

df_report["Datetime UTC"] = pd.to_datetime(df_report["Datetime UTC"], utc=True)
df_report_model = df_report[df_report["model"] == MODEL].copy()

report_ghi = df_report_model[df_report_model["row_type"] == "ghi"].copy()
report_imbalance = df_report_model[df_report_model["row_type"] == "imbalance"].copy()
report_hourly_prediction = df_report_model[df_report_model["row_type"] == "hourly_prediction"].copy()
report_yesterday_performance = df_report_model[df_report_model["row_type"] == "yesterday_ordering_performance"].copy()
report_historical_scatter = df_report_model[df_report_model["row_type"] == "historical_scatter"].copy()
report_historical_performance = df_report_model[df_report_model["row_type"] == "historical_performance"].copy()

report_ghi_window = report_ghi[
    (report_ghi["Datetime UTC"] >= yesterday_start_utc)
    & (report_ghi["Datetime UTC"] <= end_dt_future)
].copy()

line_traces = []

for col in prev_day_cols:
    if col == "MeteoSwiss stations":
        plot_df = report_ghi_window[(report_ghi_window["Datetime UTC"] >= yesterday_start_utc) & (report_ghi_window["Datetime UTC"] <= now_utc)]
    else:
        plot_df = report_ghi_window[(report_ghi_window["Datetime UTC"] >= yesterday_start_utc) & (report_ghi_window["Datetime UTC"] < today_start_utc)]

    line_traces.append(
        time_series_trace(
            plot_df,
            x_col="Datetime UTC",
            y_col=col,
            name=col,
            color=rgb_to_css(prev_day_colors[col]),
        )
    )

future_plot_df = report_ghi_window[report_ghi_window["Datetime UTC"] >= today_start_utc].copy()

line_traces.append(
    time_series_trace(
        future_plot_df,
        x_col="Datetime UTC",
        y_col=future_col,
        name=f"{MODEL} forecast",
        color=rgb_to_css(prev_day_colors[future_col]),
        dash=True,
    )
)

line_traces.append(
    band_trace(
        future_plot_df,
        x_col="Datetime UTC",
        lower_col="q95_lower",
        upper_col="q95_upper",
        name="q95 forecast-error band",
        color=rgb_to_css(prev_day_colors[future_col]),
    )
)

line_y_values = []
for col in prev_day_cols:
    line_y_values.extend(report_ghi_window[col].dropna().tolist())
line_y_values.extend(report_ghi_window["q95_upper"].dropna().tolist())
line_y_max = max(1, max(line_y_values) * 1.05) if line_y_values else 1

line_figure = {
    "title": "Previous-Day Model Runs, Actual GHI and Forecasted GHI with errorband",
    "xType": "date",
    "xTickHours": 6,
    "dayOnlyAtMidnight": True,
    "sharedHover": True,
    "xLabel": "Datetime UTC",
    "yLabel": "GHI values [W/m^2]",
    "yRange": [0, clean_float(line_y_max)],
    "traces": line_traces,
}

report_yesterday_ghi = report_ghi[
    (report_ghi["Datetime UTC"] >= yesterday_start)
    & (report_ghi["Datetime UTC"] < today_start)
].copy()

imbalance_traces = []

for col in metric_cols:
    imbalance_traces.append(
        time_series_trace(
            report_yesterday_ghi,
            x_col="Datetime UTC",
            y_col=col,
            name=col,
            color=rgb_to_css(horizon_colors[col]),
        )
    )

report_imbalance_yesterday = report_imbalance[
    (report_imbalance["Datetime UTC"] >= yesterday_start)
    & (report_imbalance["Datetime UTC"] < today_start)
].copy()
report_imbalance_yesterday["Day"] = report_imbalance_yesterday["Datetime UTC"].apply(english_day_label)
report_imbalance_yesterday["Time UTC"] = report_imbalance_yesterday["Datetime UTC"].dt.strftime("%H:%M")
report_imbalance_yesterday["Imbalance [MW]"] = report_imbalance_yesterday["Total System Imbalance"].map(lambda v: f"{v:.1f}")

imbalance_traces.append(
    bar_trace(
        report_imbalance_yesterday,
        x_col="Datetime UTC",
        y_col="Total System Imbalance",
        name="Imbalance",
        color=rgb_to_css(elcom_colors["grey"]),
        hover_columns=["Day", "Time UTC", "Imbalance [MW]"],
        y_axis="y2",
        width_minutes=12,
        alpha=0.7,
    )
)

imbalance_figure = {
    "title": f"GHI forecast error and Imbalances on {plot_date}",
    "xType": "date",
    "xTickHours": 1,
    "hoursOnlyTicks": True,
    "sharedHover": True,
    "xLabel": "Datetime UTC",
    "yLabel": "GHI forecast error CH",
    "y2Label": "Imbalances [MW]",
    "yRange": [clean_float(yL_min), clean_float(yL_max)],
    "y2Range": [clean_float(yR_min), clean_float(yR_max)],
    "traces": imbalance_traces,
}

report_hourly_prediction["Day"] = report_hourly_prediction["Datetime UTC"].apply(english_day_label)
report_hourly_prediction["Time UTC"] = report_hourly_prediction["Datetime UTC"].dt.strftime("%H:%M")
report_hourly_prediction["Risk score"] = report_hourly_prediction["predicted_hourly_risk_score"].map(lambda v: f"{v:.3f}")
report_hourly_prediction["Forecast error"] = report_hourly_prediction["actual_abs_hourly_forecast_error"].map(lambda v: f"{v:.3f}")
report_hourly_prediction["q95 range"] = report_hourly_prediction["q95_error_band"].map(lambda v: f"0.000 to {v:.3f}")

yesterday_prediction_window = report_hourly_prediction[
    (report_hourly_prediction["Datetime UTC"] >= yesterday_start_utc)
    & (report_hourly_prediction["Datetime UTC"] < today_start_utc)
].copy()

yesterday_y_values = pd.concat(
    [
        yesterday_prediction_window["predicted_hourly_risk_score"].abs(),
        yesterday_prediction_window["actual_y_predicted_risk_score"].abs(),
    ],
    ignore_index=True,
).dropna()
yesterday_y_max = float(yesterday_y_values.max()) if not yesterday_y_values.empty else 1.0

yesterday_hourly_scatter_figure = {
    "title": "Yesterday predicted risk scores with error band and actual forecast errors",
    "xType": "linear",
    "xLabel": "Hourly abs. GHI forecast error",
    "yLabel": "Hourly predicted risk score",
    "yRange": [0, clean_float(max(1.0, yesterday_y_max * 1.05))],
    "traces": [
        error_scatter_trace(
            yesterday_prediction_window,
            x_col="hourly_error_q95_mid",
            y_col="predicted_hourly_risk_score",
            x_left_col="hourly_error_q95_left",
            x_right_col="hourly_error_q95_right",
            name="Yesterday prediction",
            color=rgb_to_css(elcom_colors["orange"]),
            hover_columns=["Day", "Time UTC", "Risk score", "q95 range"],
            size=5,
        ),
        scatter_trace(
            yesterday_prediction_window,
            x_col="actual_abs_hourly_forecast_error",
            y_col="actual_y_predicted_risk_score",
            name="Yesterday actual",
            color=rgb_to_css(elcom_colors["dark_blue"]),
            hover_columns=["Day", "Time UTC", "Risk score", "Forecast error"],
            size=5,
        ),
    ],
}

report_yesterday_performance = report_yesterday_performance.sort_values("Datetime UTC")
yesterday_performance_figure = {
    "title": "Yesterday's model performance over time",
    "xType": "date",
    "xTickHours": 24,
    "dayOnlyAtMidnight": True,
    "sharedHover": True,
    "xLabel": "Day",
    "yLabel": "Kendall-tau [%]",
    "yRange": [0, 100],
    "traces": [
        time_series_trace(
            report_yesterday_performance,
            x_col="Datetime UTC",
            y_col="kendall_tau_percent",
            name="Yesterday ordering Kendall-tau [%]",
            color="rgba(0, 0, 0, 1)",
            marker=True,
        ),
    ],
}

report_historical_scatter["Day"] = report_historical_scatter["Datetime UTC"].apply(english_day_label)
report_historical_scatter["Time UTC"] = report_historical_scatter["Datetime UTC"].dt.strftime("%H:%M")
report_historical_scatter["Risk score"] = report_historical_scatter["fitted_hourly_risk_score"].map(lambda v: f"{v:.3f}")
report_historical_scatter["Forecast error"] = report_historical_scatter["actual_abs_hourly_forecast_error"].map(lambda v: f"{v:.3f}")
hist_scatter_y_max = float(report_historical_scatter["fitted_hourly_risk_score"].max()) if not report_historical_scatter.empty else 1.0

hourly_scatter_figure = {
    "title": (
        "Historical hourly fitted risk score vs abs. forecast error"
    ),
    "xType": "linear",
    "xLabel": "Hourly abs. GHI forecast error",
    "yLabel": "Hourly fitted risk score",
    "yRange": [0, clean_float(max(1.0, hist_scatter_y_max * 1.05))],
    "traces": [
        scatter_trace(
            report_historical_scatter,
            x_col="actual_abs_hourly_forecast_error",
            y_col="fitted_hourly_risk_score",
            name="Historical",
            color=rgb_to_css(elcom_colors["green"]),
            hover_columns=["Day", "Time UTC", "Risk score", "Forecast error"],
            size=4,
        ),
    ],
}

report_historical_performance = report_historical_performance.sort_values("Datetime UTC")

hourly_performance_figure = {
    "title": "Tracked historical hourly model performance",
    "xType": "date",
    "xTickHours": 24,
    "dayOnlyAtMidnight": True,
    "sharedHover": True,
    "xLabel": "Day",
    "yLabel": "Kendall-tau / Rank RMSE [%]",
    "yRange": [0, 100],
    "traces": [
        time_series_trace(
            report_historical_performance,
            x_col="Datetime UTC",
            y_col="kendall_tau_percent",
            name="Kendall-tau [%]",
            color="rgba(0, 0, 0, 1)",
            marker=True,
        ),
        time_series_trace(
            report_historical_performance,
            x_col="Datetime UTC",
            y_col="rank_rmse_percent",
            name="Rank RMSE [%]",
            color="rgba(80, 80, 80, 1)",
            dash=True,
            marker=True,
        ),
    ],
}

historical_q95_ghi = report_ghi[
    report_ghi["q95_error_band"].notna()
    & report_ghi[f"{MODEL} prev day 0"].notna()
    & (report_ghi["Datetime UTC"] < today_start_utc)
].copy()

historical_q95_ghi = historical_q95_ghi.sort_values("Datetime UTC")

historical_q95_y_values = []
for col in ["MeteoSwiss stations", f"{MODEL} prev day 0", "q95_upper"]:
    historical_q95_y_values.extend(historical_q95_ghi[col].dropna().tolist())
historical_q95_y_max = max(1, max(historical_q95_y_values) * 1.05) if historical_q95_y_values else 1

historical_q95_figure = {
    "title": "Historical prev day 0 forecast with q95 error band",
    "xType": "date",
    "xTickHours": 24,
    "dayOnlyAtMidnight": True,
    "sharedHover": True,
    "xLabel": "Datetime UTC",
    "yLabel": "GHI [W/m^2]",
    "yRange": [0, clean_float(historical_q95_y_max)],
    "traces": [
        dict(
            band_trace(
                historical_q95_ghi,
                x_col="Datetime UTC",
                lower_col="q95_lower",
                upper_col="q95_upper",
                name="q95 forecast-error band",
                color=rgb_to_css(elcom_colors["green"]),
            ),
            legendOrder=3,
        ),
        time_series_trace(
            historical_q95_ghi,
            x_col="Datetime UTC",
            y_col=f"{MODEL} prev day 0",
            name=f"{MODEL} prev day 0 forecast",
            color=rgb_to_css(elcom_colors["green"]),
        ) | {"legendOrder": 2},
        time_series_trace(
            historical_q95_ghi,
            x_col="Datetime UTC",
            y_col="MeteoSwiss stations",
            name="MeteoSwiss stations",
            color=rgb_to_css(elcom_colors["dark_blue"]),
        ) | {"legendOrder": 1},
    ],
}

report_layout = [
    {
        "title": "Prev Day Model Runs and actual GHI from MeteoSwiss stations % forecasted GHI with errorband.",
        "columns": 1,
        "figures": [line_figure],
    },
    {
        "title": f"GHI forecast errors and Imbalances on {plot_date}",
        "columns": 1,
        "figures": [imbalance_figure],
    },
    {
        "title": "Model performance yesterday",
        "columns": 1,
        "figures": [yesterday_hourly_scatter_figure],
    },
    {
        "title": "Model performance yesterday",
        "columns": 1,
        "figures": [yesterday_performance_figure],
    },
    {
        "title": "Model performance historical",
        "columns": 1,
        "figures": [hourly_scatter_figure],
    },
    {
        "title": "Model performance historical",
        "columns": 1,
        "figures": [hourly_performance_figure],
    },
    {
        "title": "Historical GHI forecast with q95 error band",
        "columns": 1,
        "figures": [historical_q95_figure],
    },
]

if not INCLUDE_SCATTERPLOTS:
    report_layout = [
        row
        for row in report_layout
        if not any(
            figure in [yesterday_hourly_scatter_figure, hourly_scatter_figure]
            for figure in row["figures"]
        )
    ]

output_dir = REPORTS_DIR
output_html = output_dir / f"GHI_forecast_risk_{MODEL}_{START_DATE}_to_{END_DATE}{OUTPUT_SUFFIX}.html"
write_interactive_html_report(
    figures=report_layout,
    output_path=output_html,
    page_title=f"GHI Forecast Risk Report - {pd.Timestamp.now(tz='Europe/Zurich').strftime('%d.%m.%Y')}",
)
print(f"Interactive HTML report written to: {output_html}")
if OPEN_BROWSER:
    maybe_open_browser(output_html)

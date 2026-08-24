import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    ("joint_velocity", "Joint velocity", "rad/s", "joint_dof_velocities"),
    ("joint_power", "Instantaneous mechanical power", "W", "joint_instantaneous_power"),
    ("joint_torque", "Joint torque", "N·m", "joint_torques"),
    ("com_position", "Whole-robot center of mass position", "m", "system_com_position"),
    ("com_velocity", "Whole-robot center of mass velocity", "m/s", "system_com_velocity"),
)


def slice_npz(source, start, stop, output_path, terrain_name, checkpoint):
    with np.load(source, allow_pickle=False) as data:
        sample_count = len(data["time"])
        arrays = {}
        for key in data.files:
            value = np.asarray(data[key])
            if value.ndim > 0 and value.shape[0] == sample_count:
                value = value[start:stop].copy()
            else:
                value = value.copy()
            arrays[key] = value
        arrays["time"] = arrays["time"] - arrays["time"][0]
        arrays["terrain_name"] = np.array(terrain_name)
        arrays["source_recording"] = np.array(str(Path(source).resolve()))
        arrays["source_checkpoint"] = np.array(str(Path(checkpoint).resolve()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def first_crossing(values, threshold):
    matches = np.flatnonzero(values >= threshold)
    if matches.size == 0:
        raise ValueError(f"Trajectory never reaches x={threshold:.2f} m")
    return int(matches[0])


def make_segments(flat_path, ramp_path, output_dir, checkpoint):
    with np.load(flat_path, allow_pickle=False) as flat:
        flat_time = np.asarray(flat["time"])
        flat_start = int(np.searchsorted(flat_time, 2.0, side="left"))
        flat_stop = int(np.searchsorted(flat_time, min(12.0, float(flat_time[-1])), side="right"))

    with np.load(ramp_path, allow_pickle=False) as ramp:
        com_x = np.asarray(ramp["system_com_position"])[:, 0]
        uphill_start = first_crossing(com_x, 1.20)
        uphill_stop = first_crossing(com_x, 3.80)
        downhill_start = first_crossing(com_x, 6.20)
        downhill_stop = first_crossing(com_x, 8.80)

    outputs = {
        "flat": output_dir / "flat_model30000.npz",
        "uphill": output_dir / "uphill_12deg_model30000.npz",
        "downhill": output_dir / "downhill_12deg_model30000.npz",
    }
    slice_npz(flat_path, flat_start, flat_stop, outputs["flat"], "flat", checkpoint)
    slice_npz(ramp_path, uphill_start, uphill_stop, outputs["uphill"], "uphill_12deg", checkpoint)
    slice_npz(ramp_path, downhill_start, downhill_stop, outputs["downhill"], "downhill_12deg", checkpoint)
    return outputs


def downsample_indices(length, limit=420):
    if length <= limit:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, limit).round().astype(np.int64))


def round_values(values):
    return np.round(np.asarray(values, dtype=np.float64), 5).tolist()


def build_payload(paths, checkpoint):
    labels = {"flat": "Flat", "uphill": "Uphill 12°", "downhill": "Downhill 12°"}
    conditions = []
    for condition_id, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            indices = downsample_indices(len(data["time"]))
            joint_names = [str(name) for name in data["joint_names"]]
            metrics = {}
            for metric_id, title, unit, key in METRICS:
                values = np.asarray(data[key])[indices]
                channel_names = joint_names if values.shape[1] == len(joint_names) else ["x", "y", "z"]
                metrics[metric_id] = {
                    "title": title,
                    "unit": unit,
                    "channels": channel_names,
                    "values": [round_values(values[:, column]) for column in range(values.shape[1])],
                }
            conditions.append(
                {
                    "id": condition_id,
                    "label": labels[condition_id],
                    "source": str(path.resolve()),
                    "time": round_values(np.asarray(data["time"])[indices]),
                    "metrics": metrics,
                }
            )
    return {
        "title": "RM75 model_30000 dynamics",
        "checkpoint": str(Path(checkpoint).resolve()),
        "conditions": conditions,
    }


def build_fragment(payload):
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f'''<div id="rm75-dynamics-v1">
  <h2>RM75 model_30000 dynamics</h2>
  <div class="viz-controls">
    <label class="form-label" for="rm75-condition">Terrain
      <select class="form-select" id="rm75-condition"></select>
    </label>
    <span class="text-small text-muted" id="rm75-source"></span>
  </div>
  <div id="rm75-plots"></div>
  <div class="tooltip" role="tooltip" id="rm75-tooltip" hidden></div>
</div>
<style>
#rm75-dynamics-v1 {{ color: var(--foreground); width: 100%; }}
#rm75-dynamics-v1 h2 {{ margin: 0 0 12px; font-weight: 500; }}
#rm75-dynamics-v1 .viz-controls {{ margin-bottom: 16px; }}
#rm75-dynamics-v1 #rm75-source {{ min-width: 0; max-width: 100%; overflow-wrap: anywhere; }}
#rm75-dynamics-v1 .metric-block {{ margin: 0 0 24px; }}
#rm75-dynamics-v1 .metric-block h3 {{ margin: 0 0 4px; font-weight: 500; }}
#rm75-dynamics-v1 .legend {{ display: flex; flex-wrap: wrap; gap: 4px 12px; margin: 0 0 4px 64px; }}
#rm75-dynamics-v1 .legend button {{ color: var(--foreground); background: transparent; border: 0; padding: 2px 0; display: inline-flex; align-items: center; gap: 5px; }}
#rm75-dynamics-v1 .legend button[aria-pressed="false"] {{ color: var(--muted-foreground); opacity: .55; }}
#rm75-dynamics-v1 .swatch {{ display: inline-block; width: 18px; border-top: 2px solid var(--swatch); }}
#rm75-dynamics-v1 .plot-host {{ width: 100%; min-height: 270px; }}
#rm75-dynamics-v1 svg {{ display: block; width: 100%; }}
#rm75-dynamics-v1 svg text {{ fill: var(--foreground); font-size: 12px; }}
#rm75-dynamics-v1 .axis path, #rm75-dynamics-v1 .axis line {{ stroke: var(--border); }}
#rm75-dynamics-v1 .grid line {{ stroke: var(--border); stroke-opacity: .45; }}
#rm75-dynamics-v1 .grid path {{ display: none; }}
#rm75-dynamics-v1 rect[data-chart-frame] {{ fill: transparent; stroke: var(--border); }}
#rm75-dynamics-v1 .series {{ fill: none; stroke-width: 1.35; }}
#rm75-dynamics-v1 .hover-guide {{ stroke: var(--muted-foreground); stroke-width: 1; pointer-events: none; }}
#rm75-dynamics-v1 .hover-marker {{ stroke: var(--background); stroke-width: 1.5; pointer-events: none; }}
#rm75-dynamics-v1 .tooltip {{ position: absolute; pointer-events: none; z-index: 20; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 8px 10px; max-width: 300px; }}
#rm75-dynamics-v1 {{ position: relative; }}
@media (max-width: 480px) {{ #rm75-dynamics-v1 .legend {{ margin-left: 0; }} #rm75-dynamics-v1 .plot-host {{ min-height: 250px; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const DATA = {payload_json};
  const root = document.getElementById("rm75-dynamics-v1");
  const conditionSelect = document.getElementById("rm75-condition");
  const sourceText = document.getElementById("rm75-source");
  const plots = document.getElementById("rm75-plots");
  const tooltip = document.getElementById("rm75-tooltip");
  const metricOrder = ["joint_velocity", "joint_power", "joint_torque", "com_position", "com_velocity"];
  const visibility = new Map();
  const dashPatterns = [null, "6 3", "2 3", "8 2 2 2"];

  DATA.conditions.forEach((condition) => {{
    const option = document.createElement("option");
    option.value = condition.id;
    option.textContent = condition.label;
    conditionSelect.appendChild(option);
  }});

  function themeColors() {{
    const styles = getComputedStyle(root);
    return Array.from({{length: 6}}, (_, i) => styles.getPropertyValue(`--viz-series-${{i + 1}}`).trim());
  }}

  function condition() {{ return DATA.conditions.find(d => d.id === conditionSelect.value) || DATA.conditions[0]; }}
  function visibleSet(metric) {{
    if (!visibility.has(metric.title)) visibility.set(metric.title, new Set(metric.channels));
    return visibility.get(metric.title);
  }}

  function interpolate(time, values, xValue) {{
    const index = d3.bisector(d => d).center(time, xValue);
    if (index <= 0 || index >= time.length - 1) return values[index];
    const left = time[index] <= xValue ? index : index - 1;
    const right = Math.min(left + 1, time.length - 1);
    const span = time[right] - time[left];
    const ratio = span > 0 ? (xValue - time[left]) / span : 0;
    return values[left] + ratio * (values[right] - values[left]);
  }}

  function drawChart(block, conditionData, metric, metricIndex) {{
    const host = block.querySelector(".plot-host");
    const legend = block.querySelector(".legend");
    const colors = themeColors();
    const active = visibleSet(metric);
    const width = Math.max(320, Math.floor(host.getBoundingClientRect().width));
    const height = width < 480 ? 250 : 285;
    const margin = {{top: 10, right: 20, bottom: 48, left: 72}};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const allPoints = metric.values.flat().filter(Number.isFinite);
    let domain = d3.extent(allPoints);
    const pad = Math.max((domain[1] - domain[0]) * 0.07, Math.abs(domain[0]) * 0.02, 1e-6);
    domain = [domain[0] - pad, domain[1] + pad];
    const x = d3.scaleLinear().domain(d3.extent(conditionData.time)).range([0, innerWidth]);
    const y = d3.scaleLinear().domain(domain).nice().range([innerHeight, 0]);

    legend.replaceChildren();
    metric.channels.forEach((channel, index) => {{
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-pressed", active.has(channel) ? "true" : "false");
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.setProperty("--swatch", colors[index % (metric.channels.length > 3 ? 3 : 6)]);
      swatch.style.borderTopStyle = index < 12 && dashPatterns[Math.floor(index / 3)] ? "dashed" : "solid";
      const label = document.createElement("span");
      label.textContent = channel.replaceAll("_joint", "").replaceAll("_", " ");
      button.append(swatch, label);
      button.addEventListener("click", () => {{
        active.has(channel) ? active.delete(channel) : active.add(channel);
        button.setAttribute("aria-pressed", active.has(channel) ? "true" : "false");
        drawChart(block, conditionData, metric, metricIndex);
      }});
      legend.appendChild(button);
    }});

    host.replaceChildren();
    const svg = d3.select(host).append("svg")
      .attr("viewBox", `0 0 ${{width}} ${{height}}`)
      .attr("role", "img")
      .attr("aria-label", `${{metric.title}} on ${{conditionData.label}}, in ${{metric.unit}}`);
    svg.append("title").text(`${{metric.title}} — ${{conditionData.label}}`);
    svg.append("desc").text(`Time histories for ${{metric.channels.length}} channels.`);
    const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
    g.append("rect").attr("data-chart-frame", "").attr("width", innerWidth).attr("height", innerHeight);
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(5).tickSize(-innerWidth).tickFormat(""));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${{innerHeight}})`)
      .call(d3.axisBottom(x).ticks(width < 480 ? 4 : 7));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    svg.append("text").attr("class", "axis-title").attr("data-axis", "x")
      .attr("x", margin.left + innerWidth / 2).attr("y", height - 8).attr("text-anchor", "middle").text("Time (s)");
    svg.append("text").attr("class", "axis-title").attr("data-axis", "y")
      .attr("transform", `translate(17,${{margin.top + innerHeight / 2}}) rotate(-90)`).attr("text-anchor", "middle").text(metric.unit);

    const line = d3.line().x((_d, i) => x(conditionData.time[i])).y(d => y(d));
    metric.channels.forEach((channel, index) => {{
      if (!active.has(channel)) return;
      g.append("path").datum(metric.values[index]).attr("class", "series")
        .attr("stroke", colors[index % (metric.channels.length > 3 ? 3 : 6)])
        .attr("stroke-dasharray", metric.channels.length > 3 ? dashPatterns[Math.floor(index / 3)] : null)
        .attr("d", line);
    }});

    const guide = g.append("line").attr("class", "hover-guide").attr("data-chart-hover-guide", "").attr("y1", 0).attr("y2", innerHeight).style("display", "none");
    const markers = g.append("g");
    const overlay = g.append("rect").attr("data-chart-hit", "").attr("data-chart-hover-overlay", "cross-series")
      .attr("width", innerWidth).attr("height", innerHeight).attr("fill", "transparent").style("pointer-events", "all");
    overlay.on("pointermove", (event) => {{
      const [px] = d3.pointer(event, overlay.node());
      const timeValue = x.invert(px);
      guide.attr("x1", px).attr("x2", px).style("display", null);
      markers.selectAll("circle").remove();
      const rows = [];
      metric.channels.forEach((channel, index) => {{
        if (!active.has(channel)) return;
        const value = interpolate(conditionData.time, metric.values[index], timeValue);
        markers.append("circle").attr("class", "hover-marker").attr("data-chart-hover-marker", "")
          .attr("cx", px).attr("cy", y(value)).attr("r", 3.5)
          .attr("fill", colors[index % (metric.channels.length > 3 ? 3 : 6)]);
        rows.push(`${{channel.replaceAll("_joint", "")}}: ${{value.toFixed(3)}} ${{metric.unit}}`);
      }});
      tooltip.innerHTML = `<strong>${{conditionData.label}} · t=${{timeValue.toFixed(2)}} s</strong><br>${{rows.join("<br>")}}`;
      tooltip.hidden = false;
      const rootRect = root.getBoundingClientRect();
      tooltip.style.left = `${{Math.min(event.clientX - rootRect.left + 12, rootRect.width - 310)}}px`;
      tooltip.style.top = `${{event.clientY - rootRect.top + 12}}px`;
    }}).on("pointerleave", () => {{ guide.style("display", "none"); markers.selectAll("circle").remove(); tooltip.hidden = true; }});
  }}

  function render() {{
    const current = condition();
    sourceText.textContent = `checkpoint: model_30000.pt · source: ${{current.source}}`;
    plots.replaceChildren();
    metricOrder.forEach((metricId, metricIndex) => {{
      const metric = current.metrics[metricId];
      const block = document.createElement("section");
      block.className = "metric-block";
      block.innerHTML = `<h3>${{metric.title}}</h3><div class="legend"></div><div class="plot-host"></div>`;
      plots.appendChild(block);
      drawChart(block, current, metric, metricIndex);
    }});
  }}

  conditionSelect.addEventListener("change", render);
  conditionSelect.value = DATA.conditions[0].id;
  render();
  let lastWidth = Math.round(root.getBoundingClientRect().width);
  new ResizeObserver(() => {{
    const nextWidth = Math.round(root.getBoundingClientRect().width);
    if (Math.abs(nextWidth - lastWidth) > 1) {{
      lastWidth = nextWidth;
      render();
    }}
  }}).observe(root);
}})();
</script>'''


def build_standalone(fragment):
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RM75 model_30000 dynamics</title>
<style>
:root {{
  color-scheme: light dark;
  --background: #ffffff;
  --foreground: #17202a;
  --muted-foreground: #5d6b7a;
  --popover: #ffffff;
  --popover-foreground: #17202a;
  --border: #d7dde5;
  --viz-series-1: #0f766e;
  --viz-series-2: #2563eb;
  --viz-series-3: #b42318;
  --viz-series-4: #7c3aed;
  --viz-series-5: #ca8a04;
  --viz-series-6: #0891b2;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --background: #111418;
    --foreground: #edf1f5;
    --muted-foreground: #a8b2bd;
    --popover: #1b2026;
    --popover-foreground: #edf1f5;
    --border: #3a424c;
    --viz-series-1: #5eead4;
    --viz-series-2: #60a5fa;
    --viz-series-3: #fb7185;
    --viz-series-4: #c4b5fd;
    --viz-series-5: #facc15;
    --viz-series-6: #67e8f9;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 20px; overflow-x: hidden; background: var(--background); color: var(--foreground); font-family: system-ui, sans-serif; }}
.viz-controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }}
.form-label {{ display: flex; align-items: center; gap: 8px; }}
.form-select {{ padding: 6px 28px 6px 8px; color: var(--foreground); background: var(--background); border: 1px solid var(--border); }}
.text-small {{ font-size: 12px; }}
.text-muted {{ color: var(--muted-foreground); }}
</style>
</head>
<body>
{fragment}
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", required=True)
    parser.add_argument("--ramp", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fragment", required=True)
    parser.add_argument("--standalone")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    paths = make_segments(
        Path(args.flat).resolve(),
        Path(args.ramp).resolve(),
        output_dir,
        Path(args.checkpoint).resolve(),
    )
    payload = build_payload(paths, args.checkpoint)
    fragment_path = Path(args.fragment).resolve()
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment = build_fragment(payload)
    fragment_path.write_text(fragment, encoding="utf-8")
    result = {"segments": {key: str(value) for key, value in paths.items()}, "fragment": str(fragment_path)}
    if args.standalone:
        standalone_path = Path(args.standalone).resolve()
        standalone_path.parent.mkdir(parents=True, exist_ok=True)
        standalone_path.write_text(build_standalone(fragment), encoding="utf-8")
        result["standalone"] = str(standalone_path)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_gym import LEGGED_GYM_ROOT_DIR


PLOT_GROUPS = [
    ("joint_positions", "Joint Positions"),
    ("joint_dof_velocities", "Joint DOF Velocities"),
    ("joint_dof_accelerations", "Joint DOF Accelerations"),
    ("joint_linear_velocities", "Joint Linear Velocities"),
    ("joint_angular_velocities", "Joint Angular Velocities"),
    ("joint_linear_accelerations", "Joint Linear Accelerations"),
    ("joint_angular_accelerations", "Joint Angular Accelerations"),
    ("joint_torques", "Joint Torques"),
    ("joint_torque_commands", "Joint Torque Commands"),
    ("joint_instantaneous_power", "Joint Instantaneous Mechanical Power"),
    ("total_instantaneous_power", "Total Instantaneous Mechanical Power"),
    ("base_position", "Base Position"),
    ("base_quaternion_wxyz", "Base Quaternion WXYZ"),
    ("base_linear_velocity", "Base Linear Velocity"),
    ("base_linear_acceleration", "Base Linear Acceleration"),
    ("base_angular_velocity", "Base Angular Velocity"),
    ("base_angular_acceleration", "Base Angular Acceleration"),
    ("base_com_position", "Base COM Position"),
    ("base_com_velocity", "Base COM Velocity"),
    ("system_com_position", "System COM Position"),
    ("system_com_velocity", "System COM Velocity"),
]


def resolve_data_path(path_ref):
    path = Path(path_ref)
    if path.is_file():
        return path.resolve()
    candidate = Path(LEGGED_GYM_ROOT_DIR) / "action_data" / path_ref
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not find action data file: {path_ref}")


def newest_data_file(robot_name=None):
    root = Path(LEGGED_GYM_ROOT_DIR) / "action_data"
    search_root = root / robot_name if robot_name else root
    files = sorted(search_root.rglob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No .npz files found under {search_root}")
    return files[-1]


def to_text(value):
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array)


def round_array(values, decimals=6):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.floating):
        values = np.round(values.astype(np.float64), decimals)
    return values.tolist()


def flatten_components(values):
    values = np.asarray(values)
    return values.reshape(values.shape[0], -1)


def component_labels(key, values, joint_names):
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == len(joint_names):
        return list(joint_names)
    if values.ndim == 2 and values.shape[1] == 3:
        if key == "total_instantaneous_power":
            return ["signed_sum", "positive_sum", "absolute_sum"]
        return ["x", "y", "z"]
    if values.ndim == 2 and values.shape[1] == 4 and "quaternion" in key:
        return ["w", "x", "y", "z"]
    if values.ndim == 3 and values.shape[1] == len(joint_names) and values.shape[2] == 3:
        axes = ["x", "y", "z"]
        return [f"{joint}_{axis}" for joint in joint_names for axis in axes]
    return [f"{key}_{index}" for index in range(int(np.prod(values.shape[1:])))]


def build_payload(data_path, decimals=6):
    with np.load(data_path, allow_pickle=False) as data:
        time_values = np.asarray(data["time"])
        step_values = np.asarray(data["step"])
        joint_names = [str(name) for name in data["joint_names"]]
        groups = []
        for key, title in PLOT_GROUPS:
            if key not in data.files:
                continue
            raw_values = np.asarray(data[key])
            flat_values = flatten_components(raw_values)
            labels = component_labels(key, raw_values, joint_names)
            groups.append(
                {
                    "key": key,
                    "title": title,
                    "labels": labels,
                    "values": [
                        round_array(flat_values[:, column], decimals=decimals)
                        for column in range(flat_values.shape[1])
                    ],
                }
            )

        metadata = {
            "source_file": str(data_path),
            "sample_count": int(time_values.shape[0]),
            "robot_name": to_text(data["robot_name"]) if "robot_name" in data.files else "",
            "terrain_name": to_text(data["terrain_name"]) if "terrain_name" in data.files else "",
            "config_path": to_text(data["config_path"]) if "config_path" in data.files else "",
            "xml_path": to_text(data["xml_path"]) if "xml_path" in data.files else "",
            "simulation_dt": float(np.asarray(data["simulation_dt"]).item()) if "simulation_dt" in data.files else None,
        }

    return {
        "metadata": metadata,
        "time": round_array(time_values, decimals=decimals),
        "step": step_values.astype(np.int64).tolist(),
        "groups": groups,
    }


def build_html(payload):
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Action Data Viewer</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #5d6b7a;
  --line: #d7dde5;
  --accent: #0f766e;
  --accent-2: #b42318;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
}}
header {{
  padding: 18px 22px 12px;
  background: var(--panel);
  border-bottom: 1px solid var(--line);
}}
h1 {{
  margin: 0 0 8px;
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 0;
}}
.meta {{
  display: grid;
  grid-template-columns: repeat(4, minmax(160px, 1fr));
  gap: 8px 14px;
  color: var(--muted);
  font-size: 13px;
}}
.meta b {{ color: var(--ink); font-weight: 600; }}
main {{
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
  gap: 16px;
  padding: 16px;
}}
.panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}}
.controls {{
  padding: 14px;
  height: calc(100vh - 130px);
  overflow: auto;
}}
label {{
  display: block;
  font-size: 12px;
  color: var(--muted);
  margin: 12px 0 6px;
}}
select, input[type="search"] {{
  width: 100%;
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0 9px;
  background: #fff;
  color: var(--ink);
}}
.button-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 12px;
}}
button {{
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  cursor: pointer;
}}
button.primary {{
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}}
.channels {{
  margin-top: 10px;
  display: grid;
  gap: 5px;
}}
.channel {{
  display: grid;
  grid-template-columns: 20px minmax(0, 1fr);
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--ink);
}}
.channel span {{
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.chart-panel {{
  display: grid;
  grid-template-rows: minmax(360px, 1fr) auto;
  min-height: calc(100vh - 130px);
  overflow: hidden;
}}
.chart-wrap {{
  position: relative;
  min-height: 360px;
}}
canvas {{
  width: 100%;
  height: 100%;
  display: block;
}}
.playback {{
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr) 130px;
  gap: 12px;
  align-items: center;
  padding: 12px 14px;
  border-top: 1px solid var(--line);
}}
input[type="range"] {{ width: 100%; }}
.stats {{
  margin-top: 14px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}}
.hint {{
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}}
@media (max-width: 900px) {{
  main {{ grid-template-columns: 1fr; }}
  .controls {{ height: auto; max-height: 45vh; }}
  .meta {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>Action Data Viewer</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <section class="panel controls">
    <label for="groupSelect">Signal group</label>
    <select id="groupSelect"></select>
    <label for="channelSearch">Filter channels</label>
    <input id="channelSearch" type="search" placeholder="joint or axis name">
    <div class="button-row">
      <button id="selectAll">Select all</button>
      <button id="selectNone">Select none</button>
    </div>
    <div class="button-row">
      <button class="primary" id="exportCsv">Export CSV</button>
      <button class="primary" id="exportJson">Export JSON</button>
    </div>
    <div class="stats" id="stats"></div>
    <div class="hint">
      CSV and JSON export the full embedded dataset. Plot drawing is downsampled automatically for smoother browser playback.
    </div>
    <div class="channels" id="channels"></div>
  </section>
  <section class="panel chart-panel">
    <div class="chart-wrap"><canvas id="chart"></canvas></div>
    <div class="playback">
      <button id="playPause">Play</button>
      <input id="timeSlider" type="range" min="0" max="0" value="0">
      <div id="timeReadout">t = 0.000 s</div>
    </div>
  </section>
</main>
<script>
const DATA = {data_json};
const COLORS = ["#0f766e", "#2563eb", "#b42318", "#7c3aed", "#ca8a04", "#0891b2", "#be185d", "#15803d", "#ea580c", "#4b5563", "#9333ea", "#0d9488"];
const state = {{
  groupIndex: 0,
  selected: new Set(),
  filter: "",
  timeIndex: 0,
  playing: false,
  rafId: null,
  lastFrameMs: 0
}};

const groupSelect = document.getElementById("groupSelect");
const channelSearch = document.getElementById("channelSearch");
const channelsEl = document.getElementById("channels");
const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("timeSlider");
const playPause = document.getElementById("playPause");
const timeReadout = document.getElementById("timeReadout");

function formatNumber(value, digits = 4) {{
  if (!Number.isFinite(value)) return "";
  return Number(value).toFixed(digits);
}}

function currentGroup() {{
  return DATA.groups[state.groupIndex];
}}

function setupMeta() {{
  const meta = DATA.metadata;
  document.getElementById("meta").innerHTML = [
    ["Robot", meta.robot_name],
    ["Terrain", meta.terrain_name],
    ["Samples", meta.sample_count],
    ["dt", meta.simulation_dt],
    ["Source", meta.source_file],
    ["XML", meta.xml_path]
  ].map(([k, v]) => `<div><b>${{k}}:</b> ${{v ?? ""}}</div>`).join("");
  document.getElementById("stats").innerHTML = `
    Groups: ${{DATA.groups.length}}<br>
    Time: ${{formatNumber(DATA.time[0], 3)}} s to ${{formatNumber(DATA.time[DATA.time.length - 1], 3)}} s<br>
    Full columns: ${{1 + 1 + DATA.groups.reduce((sum, group) => sum + group.labels.length, 0)}}
  `;
}}

function setupGroups() {{
  DATA.groups.forEach((group, index) => {{
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `${{group.title}} (${{group.labels.length}})`;
    groupSelect.appendChild(option);
  }});
  groupSelect.addEventListener("change", () => {{
    state.groupIndex = Number(groupSelect.value);
    selectDefaultChannels();
    renderChannels();
    drawChart();
  }});
}}

function selectDefaultChannels() {{
  state.selected.clear();
  const group = currentGroup();
  const count = Math.min(group.labels.length, group.labels.length > 18 ? 6 : group.labels.length);
  for (let i = 0; i < count; i += 1) state.selected.add(i);
}}

function renderChannels() {{
  const group = currentGroup();
  const filter = state.filter.toLowerCase();
  channelsEl.innerHTML = "";
  group.labels.forEach((label, index) => {{
    if (filter && !label.toLowerCase().includes(filter)) return;
    const row = document.createElement("label");
    row.className = "channel";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.has(index);
    checkbox.addEventListener("change", () => {{
      if (checkbox.checked) state.selected.add(index);
      else state.selected.delete(index);
      drawChart();
    }});
    const text = document.createElement("span");
    text.textContent = label;
    row.appendChild(checkbox);
    row.appendChild(text);
    channelsEl.appendChild(row);
  }});
}}

function resizeCanvas() {{
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}}

function drawChart() {{
  resizeCanvas();
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const margin = {{ left: 64, right: 18, top: 26, bottom: 44 }};
  const plotW = Math.max(1, width - margin.left - margin.right);
  const plotH = Math.max(1, height - margin.top - margin.bottom);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const group = currentGroup();
  const selected = Array.from(state.selected).filter(index => group.values[index]);
  if (selected.length === 0) {{
    ctx.fillStyle = "#5d6b7a";
    ctx.fillText("Select at least one channel.", margin.left, margin.top + 20);
    return;
  }}

  let minY = Infinity;
  let maxY = -Infinity;
  selected.forEach(index => {{
    const values = group.values[index];
    for (let i = 0; i < values.length; i += 1) {{
      const value = values[i];
      if (value < minY) minY = value;
      if (value > maxY) maxY = value;
    }}
  }});
  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {{
    minY = -1;
    maxY = 1;
  }}
  if (Math.abs(maxY - minY) < 1e-9) {{
    minY -= 1;
    maxY += 1;
  }}
  const pad = (maxY - minY) * 0.08;
  minY -= pad;
  maxY += pad;

  const minT = DATA.time[0];
  const maxT = DATA.time[DATA.time.length - 1];
  const xOf = time => margin.left + ((time - minT) / Math.max(maxT - minT, 1e-9)) * plotW;
  const yOf = value => margin.top + (1 - (value - minY) / Math.max(maxY - minY, 1e-9)) * plotH;

  ctx.strokeStyle = "#e3e7ed";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 5; i += 1) {{
    const y = margin.top + (plotH * i) / 5;
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
  }}
  for (let i = 0; i <= 6; i += 1) {{
    const x = margin.left + (plotW * i) / 6;
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + plotH);
  }}
  ctx.stroke();

  ctx.fillStyle = "#5d6b7a";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 5; i += 1) {{
    const value = maxY - ((maxY - minY) * i) / 5;
    const y = margin.top + (plotH * i) / 5;
    ctx.fillText(formatNumber(value, 3), margin.left - 8, y);
  }}
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 6; i += 1) {{
    const value = minT + ((maxT - minT) * i) / 6;
    const x = margin.left + (plotW * i) / 6;
    ctx.fillText(formatNumber(value, 2), x, margin.top + plotH + 10);
  }}

  const stride = Math.max(1, Math.ceil(DATA.time.length / Math.max(plotW * 2, 1)));
  selected.forEach((index, selectedIndex) => {{
    const values = group.values[index];
    ctx.strokeStyle = COLORS[selectedIndex % COLORS.length];
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < DATA.time.length; i += stride) {{
      const x = xOf(DATA.time[i]);
      const y = yOf(values[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }});

  const currentTime = DATA.time[state.timeIndex] ?? DATA.time[0];
  const markerX = xOf(currentTime);
  ctx.strokeStyle = "#b42318";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(markerX, margin.top);
  ctx.lineTo(markerX, margin.top + plotH);
  ctx.stroke();

  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillStyle = "#17202a";
  ctx.font = "600 14px system-ui, sans-serif";
  ctx.fillText(group.title, margin.left, 8);

  ctx.font = "12px system-ui, sans-serif";
  let legendX = margin.left;
  let legendY = margin.top + 8;
  selected.slice(0, 12).forEach((index, selectedIndex) => {{
    ctx.fillStyle = COLORS[selectedIndex % COLORS.length];
    ctx.fillRect(legendX, legendY + 4, 12, 3);
    ctx.fillStyle = "#17202a";
    ctx.fillText(group.labels[index], legendX + 16, legendY);
    legendY += 17;
  }});
}}

function setTimeIndex(index) {{
  state.timeIndex = Math.max(0, Math.min(DATA.time.length - 1, index));
  slider.value = state.timeIndex;
  timeReadout.textContent = `t = ${{formatNumber(DATA.time[state.timeIndex], 3)}} s`;
  drawChart();
}}

function animate(now) {{
  if (!state.playing) return;
  if (!state.lastFrameMs || now - state.lastFrameMs > 33) {{
    setTimeIndex((state.timeIndex + 1) % DATA.time.length);
    state.lastFrameMs = now;
  }}
  state.rafId = requestAnimationFrame(animate);
}}

function togglePlayback() {{
  state.playing = !state.playing;
  playPause.textContent = state.playing ? "Pause" : "Play";
  if (state.playing) {{
    state.lastFrameMs = 0;
    state.rafId = requestAnimationFrame(animate);
  }} else if (state.rafId) {{
    cancelAnimationFrame(state.rafId);
  }}
}}

function csvCell(value) {{
  const text = String(value ?? "");
  if (/[",\\n]/.test(text)) return `"${{text.replaceAll('"', '""')}}"`;
  return text;
}}

function allColumnSpecs() {{
  const columns = [
    {{ name: "time", values: DATA.time }},
    {{ name: "step", values: DATA.step }}
  ];
  DATA.groups.forEach(group => {{
    group.labels.forEach((label, index) => {{
      columns.push({{ name: `${{group.key}}.${{label}}`, values: group.values[index] }});
    }});
  }});
  return columns;
}}

function exportCsv() {{
  const columns = allColumnSpecs();
  const rows = [columns.map(column => csvCell(column.name)).join(",")];
  for (let row = 0; row < DATA.time.length; row += 1) {{
    rows.push(columns.map(column => csvCell(column.values[row])).join(","));
  }}
  downloadBlob(`${{DATA.metadata.robot_name}}_${{DATA.metadata.terrain_name}}.csv`, "text/csv;charset=utf-8", rows.join("\\n"));
}}

function exportJson() {{
  downloadBlob(
    `${{DATA.metadata.robot_name}}_${{DATA.metadata.terrain_name}}.json`,
    "application/json;charset=utf-8",
    JSON.stringify(DATA)
  );
}}

function downloadBlob(filename, type, content) {{
  const blob = new Blob([content], {{ type }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}}

channelSearch.addEventListener("input", () => {{
  state.filter = channelSearch.value;
  renderChannels();
}});
document.getElementById("selectAll").addEventListener("click", () => {{
  currentGroup().labels.forEach((_label, index) => state.selected.add(index));
  renderChannels();
  drawChart();
}});
document.getElementById("selectNone").addEventListener("click", () => {{
  state.selected.clear();
  renderChannels();
  drawChart();
}});
document.getElementById("exportCsv").addEventListener("click", exportCsv);
document.getElementById("exportJson").addEventListener("click", exportJson);
playPause.addEventListener("click", togglePlayback);
slider.addEventListener("input", () => setTimeIndex(Number(slider.value)));
window.addEventListener("resize", drawChart);

setupMeta();
setupGroups();
slider.max = Math.max(DATA.time.length - 1, 0);
selectDefaultChannels();
renderChannels();
setTimeIndex(0);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="Path to a saved .npz action data file.")
    parser.add_argument("--latest", action="store_true", help="Export the newest saved action data file.")
    parser.add_argument("--robot", help="Robot folder to search when using --latest, such as b2, go2, or dog_502.")
    parser.add_argument("--output", help="Output HTML path. Defaults to the .npz path with .html suffix.")
    parser.add_argument("--decimals", type=int, default=6, help="Decimal places kept in embedded numeric data.")
    args = parser.parse_args()

    if args.latest:
        data_path = newest_data_file(args.robot)
    elif args.file:
        data_path = resolve_data_path(args.file)
    else:
        raise SystemExit("Provide a data file or use --latest.")

    output_path = Path(args.output).resolve() if args.output else data_path.with_suffix(".html")
    payload = build_payload(data_path, decimals=max(args.decimals, 0))
    output_path.write_text(build_html(payload), encoding="utf-8")
    print(f"HTML action data viewer saved to: {output_path}")


if __name__ == "__main__":
    main()

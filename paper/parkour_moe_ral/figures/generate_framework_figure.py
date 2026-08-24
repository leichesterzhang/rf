from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path


W = 2400
H = 1500

BG = "#FBF8F2"
BG_GRAD_TOP = "#FFFDF8"
BG_GRAD_BOTTOM = "#F6F1E7"
TEXT = "#18212B"
MUTED = "#566170"
LINE = "#D7CFBF"
WHITE = "#FFFFFF"

INPUT_FILL = "#FFF4E8"
INPUT_STROKE = "#E7B173"
INPUT_ACCENT = "#C65D2C"

FUSION_FILL = "#ECFBF6"
FUSION_STROKE = "#8AC9BA"
FUSION_ACCENT = "#0F766E"

MEM_FILL = "#EEF5FF"
MEM_STROKE = "#8FB3E8"
MEM_ACCENT = "#1D4ED8"

POLICY_FILL = "#FFF9EA"
POLICY_STROKE = "#E5C470"
POLICY_ACCENT = "#B45309"

ACTOR_FILL = "#ECFDF2"
ACTOR_STROKE = "#6AD5A3"
ACTOR_ACCENT = "#166534"

CRITIC_FILL = "#EFF6FF"
CRITIC_STROKE = "#7BB3FF"
CRITIC_ACCENT = "#1D4ED8"

TRAIN_FILL = "#FFFDF8"
TRAIN_STROKE = "#D8C7A2"
TRAIN_ACCENT = "#8C6239"

DASHED = "#8A8F98"

FONT = "Aptos, Segoe UI, Noto Sans, Helvetica Neue, Arial, sans-serif"
MONO = "Aptos Mono, SFMono-Regular, Menlo, Consolas, monospace"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, fragment: str) -> None:
        self.parts.append(fragment)

    def rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        rx: int = 28,
        fill: str = WHITE,
        stroke: str = LINE,
        stroke_width: int = 2,
        dashed: bool = False,
        opacity: float = 1.0,
        filter_id: str | None = "shadow",
    ) -> None:
        dash_attr = ' stroke-dasharray="14 10"' if dashed else ""
        filter_attr = f' filter="url(#{filter_id})"' if filter_id else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" fill-opacity="{opacity:.3f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr}{filter_attr}/>'
        )

    def pill(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        text: str,
        *,
        fill: str,
        stroke: str | None = None,
        color: str = WHITE,
        size: int = 18,
        weight: int = 700,
        filter_id: str | None = None,
    ) -> None:
        self.rect(
            x,
            y,
            w,
            h,
            rx=h // 2,
            fill=fill,
            stroke=stroke or fill,
            stroke_width=1,
            filter_id=filter_id,
        )
        self.text(
            x + w / 2,
            y + h / 2 + 1,
            [text],
            size=size,
            color=color,
            weight=weight,
            anchor="middle",
            baseline="middle",
        )

    def text(
        self,
        x: float,
        y: float,
        lines: list[str],
        *,
        size: int = 22,
        color: str = TEXT,
        weight: int = 400,
        anchor: str = "start",
        baseline: str = "hanging",
        line_gap: int | None = None,
        family: str = FONT,
        italic: bool = False,
        letter_spacing: float | None = None,
    ) -> None:
        gap = line_gap if line_gap is not None else int(size * 1.35)
        style = f'font-style:{"italic" if italic else "normal"};'
        ls = f' letter-spacing="{letter_spacing}"' if letter_spacing is not None else ""
        items = []
        for idx, line in enumerate(lines):
            dy = 0 if idx == 0 else gap
            items.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
        self.add(
            f'<text x="{x}" y="{y}" fill="{color}" font-family="{esc(family)}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'dominant-baseline="{baseline}"{ls} style="{style}">'
            + "".join(items)
            + "</text>"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = MUTED,
        width: int = 3,
        dashed: bool = False,
        marker: str = "arrow",
        opacity: float = 1.0,
    ) -> None:
        dash_attr = ' stroke-dasharray="12 9"' if dashed else ""
        marker_attr = f' marker-end="url(#{marker})"' if marker else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
            f'stroke-linejoin="round" opacity="{opacity:.3f}"{dash_attr}{marker_attr}/>'
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        color: str = MUTED,
        width: int = 3,
        dashed: bool = False,
        marker: str = "arrow",
        fill: str = "none",
        opacity: float = 1.0,
    ) -> None:
        pts = " ".join(f"{x},{y}" for x, y in points)
        dash_attr = ' stroke-dasharray="12 9"' if dashed else ""
        marker_attr = f' marker-end="url(#{marker})"' if marker else ""
        self.add(
            f'<polyline points="{pts}" fill="{fill}" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" '
            f'opacity="{opacity:.3f}"{dash_attr}{marker_attr}/>'
        )

    def circle(self, cx: int, cy: int, r: int, *, fill: str, opacity: float = 1.0) -> None:
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" opacity="{opacity:.3f}"/>')

    def path(
        self,
        d: str,
        *,
        fill: str = "none",
        stroke: str = MUTED,
        width: int = 3,
        dashed: bool = False,
        marker: str = "arrow",
        opacity: float = 1.0,
    ) -> None:
        dash_attr = ' stroke-dasharray="12 9"' if dashed else ""
        marker_attr = f' marker-end="url(#{marker})"' if marker else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity:.3f}"'
            f'{dash_attr}{marker_attr}/>'
        )

    def build(self) -> str:
        defs = f"""
<defs>
  <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="{BG_GRAD_TOP}"/>
    <stop offset="100%" stop-color="{BG_GRAD_BOTTOM}"/>
  </linearGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#C7B89A" flood-opacity="0.18"/>
  </filter>
  <filter id="shadowSoft" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#C7B89A" flood-opacity="0.12"/>
  </filter>
  <marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto">
    <path d="M0,0 L14,7 L0,14 z" fill="{MUTED}"/>
  </marker>
  <marker id="arrowWarm" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto">
    <path d="M0,0 L14,7 L0,14 z" fill="{INPUT_ACCENT}"/>
  </marker>
  <marker id="arrowTrain" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto">
    <path d="M0,0 L14,7 L0,14 z" fill="{DASHED}"/>
  </marker>
</defs>
"""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}">'
            + defs
            + "".join(self.parts)
            + "</svg>"
        )


def add_panel(svg: Svg, x: int, y: int, w: int, h: int, title: str, accent: str, fill: str, stroke: str) -> None:
    svg.rect(x, y, w, h, rx=34, fill=fill, stroke=stroke, stroke_width=2, filter_id="shadowSoft")
    svg.pill(x + 22, y + 18, 190, 34, title.upper(), fill=accent, color=WHITE, size=16, filter_id=None)


def add_card(
    svg: Svg,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    title: str,
    body: list[str],
    fill: str,
    stroke: str,
    title_color: str = TEXT,
    body_color: str = MUTED,
    title_size: int = 24,
    body_size: int = 18,
    tag: str | None = None,
    tag_fill: str | None = None,
    dashed: bool = False,
) -> None:
    svg.rect(x, y, w, h, rx=24, fill=fill, stroke=stroke, stroke_width=2, dashed=dashed, filter_id="shadow")
    ty = y + 20
    if tag and tag_fill:
        svg.pill(x + 18, y + 16, 116, 26, tag, fill=tag_fill, color=WHITE, size=13, filter_id=None)
        ty = y + 52
    svg.text(x + 22, ty, [title], size=title_size, color=title_color, weight=700)
    if body:
        svg.text(x + 22, ty + title_size + 14, body, size=body_size, color=body_color, weight=430)


def add_expert(svg: Svg, x: int, y: int, w: int, h: int, label: str) -> None:
    svg.rect(x, y, w, h, rx=22, fill=WHITE, stroke=MEM_STROKE, stroke_width=2, filter_id="shadowSoft")
    svg.text(x + 18, y + 16, [label], size=20, color=MEM_ACCENT, weight=700)
    svg.text(
        x + 18,
        y + 48,
        ["Read(shared_{t-1}) ->", "GRU(feature_i) -> Write Delta s_i"],
        size=16,
        color=MUTED,
        weight=420,
    )


def main() -> None:
    svg = Svg()

    svg.add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="url(#bgGrad)"/>')
    svg.circle(2140, 180, 190, fill="#F2E7D8", opacity=0.55)
    svg.circle(2210, 350, 120, fill="#E8F4F1", opacity=0.65)
    svg.circle(140, 1320, 170, fill="#E9F0FB", opacity=0.55)
    svg.circle(330, 160, 95, fill="#FFF2E2", opacity=0.80)

    svg.text(92, 58, ["Parkour-MoE: Top-Level Estimator / Memory / Policy Architecture"], size=40, weight=760)
    svg.text(
        92,
        100,
        [
            "Exact dimensions and module wiring follow the active GO2 Parkour code path.",
            "Solid arrows: online inference path. Dashed arrows: training-only supervision.",
        ],
        size=20,
        color=MUTED,
        weight=430,
    )

    svg.rect(60, 130, 2280, 920, rx=42, fill="#FFFEFC", stroke=LINE, stroke_width=2, filter_id="shadowSoft")
    svg.pill(90, 146, 270, 38, "ONLINE INFERENCE / DEPLOYMENT PATH", fill=TEXT, color=WHITE, size=16, filter_id=None)

    svg.rect(60, 1070, 2280, 320, rx=42, fill="#FFFEFC", stroke=TRAIN_STROKE, stroke_width=2, filter_id="shadowSoft")
    svg.pill(90, 1086, 290, 38, "TRAINING-ONLY AUXILIARY LOSSES", fill=TRAIN_ACCENT, color=WHITE, size=16, filter_id=None)

    add_panel(svg, 90, 190, 300, 820, "Inputs", INPUT_ACCENT, WHITE, INPUT_STROKE)
    add_panel(svg, 430, 190, 610, 820, "Fusion", FUSION_ACCENT, WHITE, FUSION_STROKE)
    add_panel(svg, 1080, 190, 620, 820, "Memory", MEM_ACCENT, WHITE, MEM_STROKE)
    add_panel(svg, 1740, 190, 560, 820, "Policy", POLICY_ACCENT, WHITE, POLICY_STROKE)

    add_card(
        svg,
        115,
        255,
        250,
        92,
        title="Current proprio  o_t",
        body=["45-D online observation", "used by router and actor"],
        fill=INPUT_FILL,
        stroke=INPUT_STROKE,
        tag="INPUT",
        tag_fill=INPUT_ACCENT,
    )
    add_card(
        svg,
        115,
        372,
        250,
        112,
        title="History  O_t-9:t",
        body=["10 x 45 proprio frames", "flattened in runner, then reshaped", "to [10, 45] inside estimator"],
        fill=INPUT_FILL,
        stroke=INPUT_STROKE,
    )
    add_card(
        svg,
        115,
        515,
        250,
        148,
        title="Proxy depth stack  D_t",
        body=["2 x 58 x 87 depth frames", "terrain-proxy camera stream", "shared with estimator and depth AE"],
        fill=INPUT_FILL,
        stroke=INPUT_STROKE,
    )
    add_card(
        svg,
        115,
        690,
        250,
        86,
        title="Vision flag  m_vis",
        body=["1-D availability token"],
        fill=INPUT_FILL,
        stroke=INPUT_STROKE,
    )
    add_card(
        svg,
        115,
        803,
        250,
        108,
        title="Privileged state  s_t^priv",
        body=["267-D critic / supervision target", "available only during training"],
        fill="#FFF9F0",
        stroke=INPUT_STROKE,
        dashed=True,
    )
    svg.pill(120, 935, 240, 34, "Estimator refresh: every 5 env steps", fill="#F3E5D2", stroke="#E8C89C", color=INPUT_ACCENT, size=15, filter_id=None)

    svg.pill(602, 215, 265, 34, "sin/cos type + positional encodings", fill="#DDF6F0", stroke="#A9D8CB", color=FUSION_ACCENT, size=15, filter_id=None)
    add_card(
        svg,
        455,
        255,
        255,
        126,
        title="Proprio token encoder",
        body=["MLP: 45 -> 256 -> 128 -> 64", "applied to each of 10 steps", "output: 10 x 64 tokens"],
        fill=FUSION_FILL,
        stroke=FUSION_STROKE,
        tag="TOKENS",
        tag_fill=FUSION_ACCENT,
    )
    add_card(
        svg,
        760,
        255,
        255,
        146,
        title="Depth token encoder",
        body=["Conv: 2 -> 16 -> 32 -> 64", "stride-2 pyramid", "feature map: 64 x 8 x 11", "flatten to 88 x 64 tokens"],
        fill=FUSION_FILL,
        stroke=FUSION_STROKE,
    )
    add_card(
        svg,
        555,
        445,
        355,
        142,
        title="Proprio-to-visual cross-attention",
        body=[
            "queries: proprio tokens + vision token",
            "keys / values: visual tokens",
            "masked when vision is unavailable",
            "output: fused proprio tokens [10 x 64]",
        ],
        fill="#F3FFFB",
        stroke=FUSION_STROKE,
    )
    add_card(
        svg,
        455,
        640,
        255,
        116,
        title="Shared query pooling",
        body=["1 learned query", "context feature [64]"],
        fill="#F3FFFB",
        stroke=FUSION_STROKE,
    )
    add_card(
        svg,
        760,
        640,
        255,
        116,
        title="Expert query pooling",
        body=["4 learned queries", "expert features [4 x 64]"],
        fill="#F3FFFB",
        stroke=FUSION_STROKE,
    )
    add_card(
        svg,
        455,
        805,
        560,
        98,
        title="The estimator core keeps only the fused proprio token stream.",
        body=["No extra transformer stack is active in the current forward path."],
        fill="#FAFFFD",
        stroke="#CDE6E0",
        title_size=22,
        body_size=18,
    )

    add_card(
        svg,
        1115,
        255,
        260,
        116,
        title="Terrain selector head",
        body=["context -> terrain logits", "softmax -> p(terrain) [12]"],
        fill=MEM_FILL,
        stroke=MEM_STROKE,
        tag="SELECTOR",
        tag_fill=MEM_ACCENT,
    )
    add_card(
        svg,
        1405,
        255,
        260,
        136,
        title="Router MLP",
        body=["[o_t, m_vis, context, p(terrain)]", "-> 128 -> 4 -> softmax", "gating weights g_t in R^4"],
        fill=MEM_FILL,
        stroke=MEM_STROKE,
    )
    svg.pill(1136, 410, 208, 34, "shared_{t-1} memory [64]", fill="#DBEAFE", stroke="#A8C6F7", color=MEM_ACCENT, size=15, filter_id=None)
    svg.rect(1120, 450, 545, 285, rx=28, fill="#F8FBFF", stroke=MEM_STROKE, stroke_width=2, filter_id="shadowSoft")
    svg.text(1144, 470, ["Read-Write Expert Bank (4 experts)"], size=24, color=TEXT, weight=700)
    svg.text(1144, 504, ["Each expert reads the shared state, processes its pooled feature, then writes Delta s_i."], size=17, color=MUTED, weight=420)
    add_expert(svg, 1150, 548, 220, 92, "Expert 1")
    add_expert(svg, 1400, 548, 220, 92, "Expert 2")
    add_expert(svg, 1150, 652, 220, 92, "Expert 3")
    add_expert(svg, 1400, 652, 220, 92, "Expert 4")
    add_card(
        svg,
        1120,
        780,
        545,
        138,
        title="Shared memory update",
        body=[
            "Delta s = sum_i g_i Delta s_i",
            "shared_t = GRUCell(Delta s, shared_{t-1})",
            "LayerNorm -> predictive state [64]",
        ],
        fill="#F5FAFF",
        stroke=MEM_STROKE,
    )
    svg.pill(1268, 940, 250, 34, "cached and reused between refreshes", fill="#E4F0FF", stroke="#BAD2F5", color=MEM_ACCENT, size=15, filter_id=None)

    add_card(
        svg,
        1775,
        255,
        490,
        186,
        title="Prediction heads from shared_t",
        body=[
            "v_t [3]     h_tf [4]     z_mu [16]",
            "z_logvar [16]     z_tm [32]",
            "terrain logits -> argmax -> one-hot [12]",
        ],
        fill=POLICY_FILL,
        stroke=POLICY_STROKE,
        tag="MCP",
        tag_fill=POLICY_ACCENT,
    )
    add_card(
        svg,
        1835,
        476,
        370,
        92,
        title="MCP code  c_t  [67]",
        body=["[v_t, h_tf, z_mu, z_tm, onehot(terrain)]"],
        fill="#FFF7DD",
        stroke=POLICY_STROKE,
    )
    add_card(
        svg,
        1880,
        590,
        280,
        60,
        title="append vision flag ->  c_t~ [68]",
        body=[],
        fill="#FFFBEA",
        stroke="#E5D19B",
        title_size=20,
    )
    add_card(
        svg,
        1775,
        675,
        490,
        122,
        title="Actor  (deployment + rollout)",
        body=["input: [c_t~, o_t] [113]", "MLP: 113 -> 512 -> 256 -> 128 -> action [12]"],
        fill=ACTOR_FILL,
        stroke=ACTOR_STROKE,
        title_color=ACTOR_ACCENT,
    )
    add_card(
        svg,
        1775,
        828,
        490,
        126,
        title="Critic  (training / rollout only)",
        body=["input: [s_t^priv, m_vis] [268]", "MLP: 268 -> 512 -> 256 -> 128 -> value"],
        fill=CRITIC_FILL,
        stroke=CRITIC_STROKE,
        title_color=CRITIC_ACCENT,
        dashed=True,
    )
    svg.pill(1845, 968, 350, 34, "Actor reuses latest MCP between estimator refreshes", fill="#F7E9D5", stroke="#EBCB9B", color=POLICY_ACCENT, size=15, filter_id=None)

    add_card(
        svg,
        110,
        1135,
        310,
        132,
        title="State regression heads",
        body=["MSE on v_t and h_tf", "targets from privileged slices"],
        fill=TRAIN_FILL,
        stroke=TRAIN_STROKE,
        tag="LOSS",
        tag_fill=TRAIN_ACCENT,
    )
    add_card(
        svg,
        450,
        1135,
        310,
        132,
        title="Next-observation decoder",
        body=["z_t ~ N(z_mu, z_logvar)", "[v_t, h_tf, z_t] -> o_hat_(t+1) [45]"],
        fill=TRAIN_FILL,
        stroke=TRAIN_STROKE,
    )
    add_card(
        svg,
        790,
        1135,
        310,
        132,
        title="Terrain map decoder",
        body=["z_tm -> m_hat_t [187]", "MSE to privileged terrain map"],
        fill=TRAIN_FILL,
        stroke=TRAIN_STROKE,
    )
    add_card(
        svg,
        1130,
        1135,
        310,
        132,
        title="Terrain ID + latent regularization",
        body=["cross-entropy on terrain logits", "KL(z_mu, z_logvar) with warm-up"],
        fill=TRAIN_FILL,
        stroke=TRAIN_STROKE,
    )
    add_card(
        svg,
        1470,
        1135,
        390,
        132,
        title="Routing regularization",
        body=["load balancing on mean g_t", "cross-view SwAV on gate window vs map window"],
        fill=TRAIN_FILL,
        stroke=TRAIN_STROKE,
    )
    add_card(
        svg,
        1890,
        1135,
        340,
        132,
        title="Depth autoencoder",
        body=["D_t -> D_hat_t", "independent MSE reconstruction branch"],
        fill=TRAIN_FILL,
        stroke=TRAIN_STROKE,
    )

    svg.text(
        110,
        1296,
        [
            "Only the solid path is needed for deployment. Critic, decoders, KL, load-balancing, SwAV, and depth reconstruction are training-only objectives.",
            "The diagram intentionally omits currently inactive placeholders in code (for example terrain_token_encoder, map_unet, and transformer-layer stubs).",
        ],
        size=17,
        color=MUTED,
        weight=420,
    )

    svg.polyline([(365, 430), (405, 430), (405, 320), (455, 320)], color=INPUT_ACCENT, width=3, marker="arrowWarm")
    svg.polyline([(365, 590), (405, 590), (405, 330), (760, 330)], color=INPUT_ACCENT, width=3, marker="arrowWarm")
    svg.polyline([(365, 733), (430, 733), (430, 515), (555, 515)], color=INPUT_ACCENT, width=3, marker="arrowWarm")

    svg.line(710, 516, 760, 516, color=FUSION_ACCENT, width=3)
    svg.line(890, 330, 890, 428, color=FUSION_ACCENT, width=3, marker="")
    svg.line(710, 320, 760, 320, color=FUSION_ACCENT, width=3)
    svg.polyline([(582, 587), (582, 640)], color=FUSION_ACCENT, width=3)
    svg.polyline([(885, 587), (885, 640)], color=FUSION_ACCENT, width=3)

    svg.polyline([(710, 698), (1115, 698)], color=FUSION_ACCENT, width=3)
    svg.polyline([(582, 698), (582, 605), (1115, 605), (1115, 313)], color=FUSION_ACCENT, width=3)
    svg.polyline([(582, 698), (582, 760), (1405, 760), (1405, 323)], color=FUSION_ACCENT, width=3)
    svg.polyline([(1015, 698), (1120, 698)], color=FUSION_ACCENT, width=3)

    svg.polyline([(1375, 313), (1405, 323)], color=MEM_ACCENT, width=3)
    svg.polyline([(1665, 323), (1695, 323), (1695, 850), (1665, 850)], color=MEM_ACCENT, width=3, marker="")
    svg.polyline([(1295, 444), (1295, 548)], color=MEM_ACCENT, width=3, marker="")
    svg.polyline([(1405, 391), (1405, 548)], color=MEM_ACCENT, width=3, marker="")
    svg.polyline([(1535, 391), (1535, 548)], color=MEM_ACCENT, width=3, marker="")
    svg.polyline([(1390, 736), (1390, 780)], color=MEM_ACCENT, width=3)
    svg.line(1665, 850, 1775, 850, color=MEM_ACCENT, width=3)
    svg.polyline([(1665, 323), (1775, 323)], color=MEM_ACCENT, width=3)
    svg.path("M 1470 918 C 1470 980, 1190 980, 1190 444", stroke=MEM_ACCENT, width=3, marker="arrow")

    svg.line(2020, 441, 2020, 476, color=POLICY_ACCENT, width=3)
    svg.line(2020, 568, 2020, 590, color=POLICY_ACCENT, width=3)
    svg.line(2020, 650, 2020, 675, color=ACTOR_ACCENT, width=3)

    svg.polyline([(2018, 441), (2018, 1110), (265, 1110), (265, 1135)], color=DASHED, width=2, dashed=True, marker="arrowTrain")
    svg.polyline([(2018, 441), (2018, 1110), (605, 1110), (605, 1135)], color=DASHED, width=2, dashed=True, marker="arrowTrain")
    svg.polyline([(2018, 441), (2018, 1110), (945, 1110), (945, 1135)], color=DASHED, width=2, dashed=True, marker="arrowTrain")
    svg.polyline([(1245, 313), (1245, 1110), (1285, 1110), (1285, 1135)], color=DASHED, width=2, dashed=True, marker="arrowTrain")
    svg.polyline([(1535, 323), (1535, 1110), (1665, 1110), (1665, 1135)], color=DASHED, width=2, dashed=True, marker="arrowTrain")
    svg.polyline([(365, 590), (2058, 590), (2058, 1135)], color=DASHED, width=2, dashed=True, marker="arrowTrain")

    svg.text(2055, 1380, ["framework.svg / framework.png"], size=16, color="#8A7E68", weight=500, anchor="end", family=MONO)

    out_dir = Path(__file__).resolve().parent
    svg_path = out_dir / "framework.svg"
    png_path = out_dir / "framework.png"
    svg_path.write_text(svg.build(), encoding="utf-8")

    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if chrome:
        url = svg_path.resolve().as_uri()
        commands = [
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                f"--window-size={W},{H}",
                f"--screenshot={png_path}",
                url,
            ],
            [
                chrome,
                "--headless",
                "--disable-gpu",
                f"--window-size={W},{H}",
                f"--screenshot={png_path}",
                url,
            ],
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--window-size={W},{H}",
                f"--screenshot={png_path}",
                url,
            ],
        ]
        for command in commands:
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                break
            except Exception:
                continue

    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    main()

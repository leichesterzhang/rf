from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto import event_pb2
import yaml

from generate_parkour_moe_framework_pptx import (
    ACCENT,
    ACCENT_2,
    ACCENT_3,
    ACCENT_4,
    BG,
    BG_ALT,
    MUTED,
    PANEL,
    PANEL_BLUE,
    PANEL_SAND,
    PANEL_TEAL,
    ROOT_RELS_XML,
    LAYOUT_RELS_XML,
    LAYOUT_XML,
    MASTER_RELS_XML,
    MASTER_XML,
    PRES_PROPS_XML,
    ShapeCounter,
    THEME_XML,
    TEXT,
    VIEW_PROPS_XML,
    WHITE,
    app_xml,
    core_xml,
    esc,
    presentation_rels_xml,
    presentation_xml,
    shape_xml,
)


ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "logs/go2_parkour_moe/ZApr19_16-24-31_GOOD"
EVENT_PATH = RUN_DIR / "events.out.tfevents.1776587072.resetsjx.272002.0"
CONFIG_PATH = RUN_DIR / "config.yaml"
CURVE_IMAGE_PATH = ROOT / "doc/ppo_parkour_training_curves.png"
OUTPUT_PPTX = ROOT / "doc/ppo_algorithm_intro_zh.pptx"


@dataclass
class PictureSpec:
    shape_id: int
    path: Path
    x: int
    y: int
    cx: int
    cy: int
    name: str = "picture"


@dataclass
class SlideSpec:
    title: str
    bg: str
    elements: list[str]
    pictures: list[PictureSpec]


@dataclass
class CurveStats:
    first_step: int
    first_value: float
    last_step: int
    last_value: float
    max_step: int
    max_value: float
    min_step: int
    min_value: float


class SlideBuilder:
    def __init__(self, number: str, kicker: str, title: str, *, bg: str = BG):
        self.number = number
        self.kicker = kicker
        self.title = title
        self.bg = bg
        self.counter = ShapeCounter()
        self.elements: list[str] = []
        self.pictures: list[PictureSpec] = []
        self._add_header()

    def _add_header(self) -> None:
        self._add_shape(
            10_020_000,
            100_000,
            1_780_000,
            1_080_000,
            lines=[self.number],
            size=18_000,
            color="E7DDCE",
            align="ctr",
            anchor="ctr",
            name="watermark",
        )
        self._add_shape(
            720_000,
            250_000,
            2_650_000,
            360_000,
            lines=[self.kicker],
            size=1280,
            color=WHITE,
            fill=ACCENT,
            line=ACCENT,
            bold=True,
            align="ctr",
            anchor="ctr",
            name="kicker",
        )
        self._add_shape(
            720_000,
            640_000,
            10_050_000,
            640_000,
            lines=[self.title],
            size=2600,
            color=TEXT,
            bold=True,
            name="title",
        )

    def _add_shape(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        *,
        lines: Iterable[str],
        size: int = 1800,
        color: str = TEXT,
        fill: str | None = None,
        line: str | None = None,
        bold: bool = False,
        align: str = "l",
        anchor: str = "t",
        rounded: bool = True,
        txbox: bool = True,
        name: str = "shape",
    ) -> None:
        sid = self.counter.alloc()
        self.elements.append(
            shape_xml(
                sid,
                name,
                x,
                y,
                cx,
                cy,
                lines=lines,
                size=size,
                color=color,
                fill=fill,
                line=line,
                bold=bold,
                align=align,
                anchor=anchor,
                rounded=rounded,
                txbox=txbox,
            )
        )

    def add_box(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        *,
        lines: Iterable[str],
        fill: str = PANEL,
        line: str = "D9C4A1",
        size: int = 1800,
        color: str = TEXT,
        bold: bool = False,
        align: str = "l",
        anchor: str = "t",
        rounded: bool = True,
        name: str = "box",
    ) -> None:
        self._add_shape(
            x,
            y,
            cx,
            cy,
            lines=lines,
            size=size,
            color=color,
            fill=fill,
            line=line,
            bold=bold,
            align=align,
            anchor=anchor,
            rounded=rounded,
            name=name,
        )

    def add_text(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        *,
        lines: Iterable[str],
        size: int = 1700,
        color: str = TEXT,
        bold: bool = False,
        align: str = "l",
        anchor: str = "t",
        name: str = "text",
    ) -> None:
        self._add_shape(
            x,
            y,
            cx,
            cy,
            lines=lines,
            size=size,
            color=color,
            bold=bold,
            align=align,
            anchor=anchor,
            name=name,
        )

    def add_arrow(self, x: int, y: int, *, text: str = "→", color: str = ACCENT) -> None:
        self.add_text(
            x,
            y,
            340_000,
            260_000,
            lines=[text],
            size=2600,
            color=color,
            bold=True,
            align="ctr",
            anchor="ctr",
            name="arrow",
        )

    def add_picture(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        *,
        path: Path,
        name: str = "picture",
    ) -> None:
        sid = self.counter.alloc()
        self.pictures.append(
            PictureSpec(
                shape_id=sid,
                path=path,
                x=x,
                y=y,
                cx=cx,
                cy=cy,
                name=name,
            )
        )

    def add_footer(self, text: str) -> None:
        self.add_text(
            740_000,
            6_350_000,
            10_650_000,
            250_000,
            lines=[text],
            size=900,
            color=MUTED,
        )

    def build(self) -> SlideSpec:
        return SlideSpec(
            title=self.title,
            bg=self.bg,
            elements=list(self.elements),
            pictures=list(self.pictures),
        )


def slide_xml(bg: str, elements: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        "<p:cSld>"
        "<p:bg><p:bgPr>"
        f'<a:solidFill><a:srgbClr val="{bg}"/></a:solidFill>'
        "<a:effectLst/>"
        "</p:bgPr></p:bg>"
        "<p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
        "<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/>"
        "<a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
        f"{''.join(elements)}"
        "</p:spTree>"
        "</p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>"
        "</p:sld>"
    )


def picture_xml(pic: PictureSpec, rel_id: str) -> str:
    return (
        "<p:pic>"
        "<p:nvPicPr>"
        f'<p:cNvPr id="{pic.shape_id}" name="{esc(pic.name)}"/>'
        '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>'
        "<p:nvPr/>"
        "</p:nvPicPr>"
        "<p:blipFill>"
        f'<a:blip r:embed="{rel_id}"/>'
        "<a:stretch><a:fillRect/></a:stretch>"
        "</p:blipFill>"
        "<p:spPr>"
        "<a:xfrm>"
        f'<a:off x="{pic.x}" y="{pic.y}"/>'
        f'<a:ext cx="{pic.cx}" cy="{pic.cy}"/>'
        "</a:xfrm>"
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        "<a:ln><a:noFill/></a:ln>"
        "</p:spPr>"
        "</p:pic>"
    )


def slide_rels_xml(media_targets: list[str]) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    rels.extend(
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/>'
        for idx, target in enumerate(media_targets, start=2)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}"
        "</Relationships>"
    )


def content_types_xml(num_slides: int, image_exts: set[str]) -> str:
    defaults = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
    ]
    if ".png" in image_exts:
        defaults.append('<Default Extension="png" ContentType="image/png"/>')
    if ".jpg" in image_exts or ".jpeg" in image_exts:
        defaults.append('<Default Extension="jpg" ContentType="image/jpeg"/>')
        defaults.append('<Default Extension="jpeg" ContentType="image/jpeg"/>')

    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/ppt/slides/slide{idx + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for idx in range(num_slides)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        f"{''.join(defaults)}"
        f"{''.join(overrides)}"
        "</Types>"
    )


def moving_average(values: np.ndarray, window: int = 200) -> np.ndarray:
    if len(values) <= 1:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_run_cfg() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_scalar_series(tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    loader = event_file_loader.RawEventFileLoader(str(EVENT_PATH))
    series: dict[str, list[tuple[int, float]]] = {tag: [] for tag in tags}
    for raw_event in loader.Load():
        event = event_pb2.Event.FromString(raw_event)
        if not event.HasField("summary"):
            continue
        step = int(event.step)
        for value in event.summary.value:
            if value.tag in series:
                series[value.tag].append((step, float(value.simple_value)))

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag, items in series.items():
        if not items:
            raise RuntimeError(f"Missing scalar tag: {tag}")
        steps = np.array([step for step, _ in items], dtype=np.int32)
        values = np.array([val for _, val in items], dtype=np.float64)
        result[tag] = (steps, values)
    return result


def build_curve_stats(steps: np.ndarray, values: np.ndarray) -> CurveStats:
    max_idx = int(np.argmax(values))
    min_idx = int(np.argmin(values))
    return CurveStats(
        first_step=int(steps[0]),
        first_value=float(values[0]),
        last_step=int(steps[-1]),
        last_value=float(values[-1]),
        max_step=int(steps[max_idx]),
        max_value=float(values[max_idx]),
        min_step=int(steps[min_idx]),
        min_value=float(values[min_idx]),
    )


def generate_curves_image(out_path: Path) -> dict[str, CurveStats]:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    tags = [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Episode/terrain_level_all",
        "Termination/no_progress_frac",
    ]
    titles = {
        "Train/mean_reward": "Mean Reward",
        "Train/mean_episode_length": "Mean Episode Length",
        "Episode/terrain_level_all": "Terrain Level All",
        "Termination/no_progress_frac": "No-Progress Fraction",
    }
    colors = {
        "Train/mean_reward": "#C65D2C",
        "Train/mean_episode_length": "#0F766E",
        "Episode/terrain_level_all": "#1D4ED8",
        "Termination/no_progress_frac": "#B45309",
    }

    series = read_scalar_series(tags)
    stats = {tag: build_curve_stats(*series[tag]) for tag in tags}

    def kfmt(value: float, _pos: float) -> str:
        return f"{value / 1000:.0f}k" if value >= 1000 else f"{int(value)}"

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.6), dpi=200)
    fig.patch.set_facecolor("#FCF8F1")

    for ax, tag in zip(axes.flat, tags):
        steps, values = series[tag]
        smooth = moving_average(values, window=200)
        color = colors[tag]

        ax.set_facecolor("#FFFDF8")
        ax.plot(steps, values, color=color, alpha=0.18, linewidth=1.2)
        ax.plot(steps, smooth, color=color, linewidth=2.3)
        ax.set_title(titles[tag], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.tick_params(labelsize=9)
        ax.set_xlabel("iteration", fontsize=9)
        if tag == "Termination/no_progress_frac":
            ax.set_ylim(bottom=0.0)
            ax.set_ylabel("fraction", fontsize=9)
        elif tag == "Episode/terrain_level_all":
            ax.set_ylabel("level", fontsize=9)
        else:
            ax.set_ylabel("value", fontsize=9)

        last_val = stats[tag].last_value
        ax.text(
            0.98,
            0.92,
            f"last={last_val:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=color,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FCF8F1", "edgecolor": color, "alpha": 0.95},
        )

    fig.suptitle("ParkourMoE PPO Training Curves from ZApr19_16-24-31_GOOD", fontsize=15, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return stats


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def build_slides(curves_path: Path, curve_stats: dict[str, CurveStats], cfg: dict) -> list[SlideSpec]:
    train_cfg = cfg["train_cfg"]
    env_cfg = cfg["env_cfg"]
    algorithm = train_cfg["algorithm"]
    policy = train_cfg["policy"]
    runner = train_cfg["runner"]
    estimator = train_cfg["estimator"]

    slides: list[SlideSpec] = []

    obs_dim = env_cfg["env"]["num_observations"]
    privileged_dim = env_cfg["env"]["num_privileged_obs"]
    history_len = estimator["history_length"]
    depth_h = env_cfg["camera"]["output_height"]
    depth_w = env_cfg["camera"]["output_width"]
    depth_buf = env_cfg["camera"]["buffer_len"]
    mcp_dim = policy["mcp_dim"]
    actor_input_dim = obs_dim + mcp_dim + 1
    critic_input_dim = privileged_dim + 1
    total_batch = env_cfg["env"]["num_envs"] * runner["num_steps_per_env"]
    mini_batch_size = total_batch // algorithm["num_mini_batches"]
    total_update_steps = algorithm["num_learning_epochs"] * algorithm["num_mini_batches"]

    s = SlideBuilder("00", "PPO + ParkourMoE", "PPO算法介绍与本仓库中的实际落地")
    s.add_box(
        780_000,
        1_780_000,
        10_500_000,
        1_520_000,
        lines=[
            "内容来源分成三部分：",
            "1. PPO 论文与 OpenAI 发布页，用于背景和基本思想。",
            "2. 本仓库 rsl_rl 的 PPO / ActorCritic 实现，用于网络结构与实现细节。",
            "3. parkour_moe 的训练配置与 ZApr19_16-24-31_GOOD TensorBoard 日志，用于应用与性能曲线。",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1800,
    )
    s.add_box(
        780_000,
        3_700_000,
        3_050_000,
        1_120_000,
        lines=["PPO 论文侧重点", "稳定更新", "一阶优化", "多轮 mini-batch"],
        fill=PANEL_SAND,
        line=ACCENT_4,
        align="ctr",
        anchor="ctr",
        bold=True,
    )
    s.add_box(
        4_080_000,
        3_700_000,
        3_050_000,
        1_120_000,
        lines=["rsl_rl 实现侧重点", "Actor-Critic", "GAE", "clip/value/entropy"],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        align="ctr",
        anchor="ctr",
        bold=True,
    )
    s.add_box(
        7_380_000,
        3_700_000,
        3_050_000,
        1_120_000,
        lines=["parkour_moe 侧重点", "多模态估计器", "terrain-aware MoE", "真实训练曲线"],
        fill=PANEL_BLUE,
        line=ACCENT_3,
        align="ctr",
        anchor="ctr",
        bold=True,
    )
    s.add_box(
        1_280_000,
        5_230_000,
        9_460_000,
        620_000,
        lines=["一句话：PPO 是这套 Go2 parkour 系统的优化骨架，估计器与 MoE 则决定了任务上限。"],
        fill="FFF7EC",
        line=ACCENT,
        size=1750,
        bold=True,
        align="ctr",
        anchor="ctr",
    )
    s.add_footer("参考：Schulman et al. 2017；OpenAI Baselines PPO；rsl_rl 与 parkour_moe 当前代码")
    slides.append(s.build())

    s = SlideBuilder("1", "1. 算法背景介绍", "PPO 的出发点：在不牺牲稳定性的前提下，把 trust-region 思想做成更易用的工程算法", bg=BG_ALT)
    s.add_box(
        720_000,
        1_760_000,
        3_270_000,
        3_030_000,
        lines=[
            "普通 policy gradient / actor-critic",
            "• 更新步长过大时容易崩",
            "• 样本方差高，调参敏感",
            "• 想重复利用 rollout 时更不稳定",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        4_450_000,
        1_760_000,
        3_270_000,
        3_030_000,
        lines=[
            "TRPO 的启发",
            "• 用 trust region 约束策略漂移",
            "• 训练更稳定",
            "• 但实现需要更复杂的二阶近似",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
    )
    s.add_box(
        8_180_000,
        1_760_000,
        3_280_000,
        3_030_000,
        lines=[
            "PPO 的目标",
            "• 保留“不要走太远”的思想",
            "• 改成易实现的一阶优化",
            "• 允许对同一批样本做多轮 mini-batch 更新",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_box(
        720_000,
        5_130_000,
        10_740_000,
        650_000,
        lines=["因此，PPO 的核心不是“求最优闭式解”，而是“让策略每次别走太远，但还能高效地一阶优化”。"],
        fill=PANEL_BLUE,
        line=ACCENT_3,
        size=1750,
        bold=True,
        align="ctr",
        anchor="ctr",
    )
    s.add_footer("参考：PPO 论文摘要与 OpenAI PPO 发布页对 clipped surrogate 的解释")
    slides.append(s.build())

    s = SlideBuilder("2", "2.1 算法基本思想", "从 policy gradient 到 PPO：关键是把“新旧策略差异”显式写进目标函数")
    s.add_box(
        720_000,
        1_760_000,
        5_150_000,
        1_320_000,
        lines=[
            "概率比率",
            "r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)",
            "它表示同一个动作在“新策略”与“旧策略”下的相对偏好变化。",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1680,
    )
    s.add_box(
        6_180_000,
        1_760_000,
        5_250_000,
        1_320_000,
        lines=[
            "未加约束的 surrogate",
            "L^CPI(theta) = E_t[ r_t(theta) * A_t ]",
            "如果 A_t > 0，希望提高该动作概率；如果 A_t < 0，希望降低它。",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1680,
    )
    s.add_box(
        720_000,
        3_450_000,
        3_260_000,
        1_620_000,
        lines=[
            "为什么还不够",
            "• 当 r_t 偏离 1 太多时，单次更新会过猛",
            "• 好动作可能被过度放大，坏动作可能被过度压缩",
            "• 连续控制里这会直接表现为训练震荡或性能塌陷",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
    )
    s.add_box(
        4_320_000,
        3_450_000,
        2_050_000,
        1_620_000,
        lines=[
            "直观理解",
            "A_t > 0 : 该动作比基线更好",
            "A_t < 0 : 该动作比基线更差",
            "r_t > 1 : 新策略更偏爱它",
            "r_t < 1 : 新策略更不偏爱它",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
        align="ctr",
        anchor="ctr",
        bold=True,
    )
    s.add_box(
        6_850_000,
        3_450_000,
        4_610_000,
        1_620_000,
        lines=[
            "PPO 的关键转折",
            "不是直接最大化 r_t * A_t，",
            "而是在“偏离过大”的区域里截断继续获利的空间。",
            "这一步把“稳定性”直接写进了目标函数。",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1650,
    )
    s.add_footer("这一页对应 PPO 论文里从 surrogate objective 走向 clipped surrogate 的动机")
    slides.append(s.build())

    s = SlideBuilder("3", "2.2 算法基本思想", "最重要的公式：clipped surrogate 如何限制策略更新")
    s.add_box(
        720_000,
        1_730_000,
        10_740_000,
        1_120_000,
        lines=[
            "L^CLIP(theta) = E_t [ min( r_t(theta) * A_t, clip(r_t(theta), 1-eps, 1+eps) * A_t ) ]",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1620,
        bold=True,
        align="ctr",
        anchor="ctr",
    )
    s.add_box(
        720_000,
        3_150_000,
        3_270_000,
        2_200_000,
        lines=[
            "当 A_t > 0",
            "希望提高动作概率。",
            "但若 r_t > 1 + eps，",
            "clip 后收益被截住。",
            "也就是：好动作可以多鼓励，",
            "但不能无限制猛增。",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
    )
    s.add_box(
        4_460_000,
        3_150_000,
        3_270_000,
        2_200_000,
        lines=[
            "当 A_t < 0",
            "希望降低动作概率。",
            "但若 r_t < 1 - eps，",
            "clip 后惩罚也被截住。",
            "也就是：坏动作可以压制，",
            "但不能一步压得过头。",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
    )
    s.add_box(
        8_200_000,
        3_150_000,
        3_260_000,
        2_200_000,
        lines=[
            "为什么用 min()",
            "PPO 选择“更保守”的那一项。",
            "一旦 ratio 超出可信区间，",
            "目标值就不再继续奖励",
            "更大的策略漂移。",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_box(
        1_260_000,
        5_680_000,
        9_280_000,
        520_000,
        lines=[
            "一句话记忆：PPO 不是硬约束参数更新，而是让“超出安全区的更新”失去继续变大的收益。",
        ],
        fill="FFF7EC",
        line=ACCENT,
        size=1500,
        bold=True,
        align="ctr",
        anchor="ctr",
    )
    s.add_footer("对应公式：PPO 论文的 clipped surrogate objective")
    slides.append(s.build())

    s = SlideBuilder("4", "2.3 算法基本思想", "优势、回报与完整损失：PPO 不是只有一个 clip 项", bg=BG)
    s.add_box(
        720_000,
        1_760_000,
        3_260_000,
        1_980_000,
        lines=[
            "TD 残差",
            "delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)",
            "",
            "GAE",
            "A_hat_t = sum_l (gamma * lambda)^l * delta_{t+l}",
            "",
            "Return",
            "R_hat_t = A_hat_t + V(s_t)",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1480,
    )
    s.add_box(
        3_520_000,
        1_760_000,
        3_330_000,
        1_980_000,
        lines=[
            "值函数损失",
            "V_clip = V_old + clip(V_theta - V_old, -eps, eps)",
            "L_VF = max( (V_theta - R_hat)^2, (V_clip - R_hat)^2 )",
            "",
            "代码对应",
            "value_clipped = target_values + clamp(value - target_values)",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1450,
    )
    s.add_box(
        6_320_000,
        1_760_000,
        2_650_000,
        1_980_000,
        lines=[
            "熵正则",
            "H[pi_theta](s_t)",
            "鼓励策略保留随机性，避免过早塌缩",
            "",
            "代码里",
            "loss = surrogate + c_v * value - c_e * entropy",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
        size=1480,
    )
    s.add_box(
        9_270_000,
        1_760_000,
        2_190_000,
        1_980_000,
        lines=[
            "优势标准化",
            "A_hat <- (A_hat - mean) / (std + 1e-8)",
            "",
            "作用",
            "让不同 batch 的尺度更稳定",
            "减小优化震荡",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
        size=1450,
    )
    s.add_box(
        720_000,
        4_240_000,
        10_740_000,
        1_040_000,
        lines=[
            "本仓库里的 PPOParkourMoE 保留了以上全部主项：GAE、value clipping、entropy 正则、advantages 标准化，以及 adaptive KL 学习率调度。",
        ],
        fill="FFF7EC",
        line=ACCENT,
        size=1600,
        bold=True,
    )
    s.add_footer("公式视角与代码视角的一个差别：论文常写最大化目标，代码里通常把它改写成最小化 loss")
    slides.append(s.build())

    s = SlideBuilder("5", "2.4 算法基本思想", "把公式落成一次训练迭代：先采样，再算 returns，再做多轮 mini-batch 更新", bg=BG_ALT)
    s.add_box(
        720_000,
        1_760_000,
        6_980_000,
        3_520_000,
        lines=[
            "一轮 PPO / on-policy 训练伪代码",
            "1. 用旧策略 pi_old 在并行环境里采样一整个 rollout。",
            "2. 对每个时刻保存 obs, action, reward, done, value, log_prob, mu, sigma。",
            "3. 用最后一步 critic value 做 bootstrap，反向计算 returns 与 GAE advantages。",
            "4. 把 [T, N] 展平为 [T*N]，随机打乱后切成 mini-batch。",
            f"5. 对同一批数据重复 {algorithm['num_learning_epochs']} 个 epoch、每个 epoch {algorithm['num_mini_batches']} 个 mini-batch。",
            "6. 每个 mini-batch 内重新前向 actor-critic，计算 ratio、surrogate、value、entropy。",
            "7. 做梯度裁剪与 Adam 更新，然后清空 storage，进入下一轮采样。",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1450,
    )
    s.add_box(
        8_020_000,
        1_760_000,
        3_440_000,
        3_520_000,
        lines=[
            "本次运行的关键超参数",
            f"num_envs = {env_cfg['env']['num_envs']}",
            f"num_steps_per_env = {runner['num_steps_per_env']}",
            f"batch = {total_batch}",
            f"num_mini_batches = {algorithm['num_mini_batches']}",
            f"mini_batch_size = {mini_batch_size}",
            f"epochs = {algorithm['num_learning_epochs']}",
            f"每轮总更新步 = {total_update_steps}",
            f"learning_rate = {algorithm['learning_rate']:.1e}",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1450,
    )
    s.add_footer("从公式到代码的桥梁就在这页：PPO 不是单步优化，而是“采样块 + 多轮重用更新”的交替过程")
    slides.append(s.build())

    s = SlideBuilder("6", "3. 算法网络结构", "在本仓库里，PPO 的控制回路是“Estimator -> Actor -> Env -> Storage -> PPO Update”", bg=BG)
    s.add_box(
        720_000,
        1_760_000,
        2_480_000,
        1_900_000,
        lines=[
            "输入侧",
            f"obs_now = {obs_dim}",
            f"obs_history = {history_len} x {obs_dim}",
            f"depth = {depth_buf} x {depth_h} x {depth_w}",
            f"privileged_obs = {privileged_dim}",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_arrow(3_300_000, 2_610_000)
    s.add_box(
        3_660_000,
        1_760_000,
        3_440_000,
        1_900_000,
        lines=[
            "ParkourEstimator",
            "多模态 token 化",
            "Transformer 编码 + terrain selector",
            f"{estimator['expert_num']} 个 read/write experts",
            "输出 mcp_code 与 terrain-aware 中间表征",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_arrow(7_350_000, 2_610_000)
    s.add_box(
        7_760_000,
        1_760_000,
        3_600_000,
        1_270_000,
        lines=[
            "ActorCriticParkourMoE",
            f"actor input = {actor_input_dim}",
            f"critic input = {critic_input_dim}",
            "策略分布 = Normal(mean, std)",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
    )
    s.add_box(
        8_120_000,
        3_260_000,
        2_720_000,
        980_000,
        lines=["RolloutStorage", "缓存整段 rollout", "并支撑后续 mini-batch 重放"],
        fill=PANEL_SAND,
        line=ACCENT_4,
        align="ctr",
        anchor="ctr",
        bold=True,
    )
    s.add_box(
        720_000,
        4_450_000,
        5_420_000,
        1_250_000,
        lines=[
            "mcp_code 组成",
            "mcp_code = [v_t(3), h_tf(4), z_mu(16), z_tm(32), terrain_onehot(12)] = 67",
            "runner 会再 append 一个 vision_flag，因此真正送入 actor 的 context 是 68 维。",
        ],
        fill="FFF7EC",
        line=ACCENT,
        size=1500,
    )
    s.add_box(
        6_520_000,
        4_450_000,
        4_840_000,
        1_250_000,
        lines=[
            "不对称 actor-critic",
            "actor: mcp_code + obs_now",
            "critic: privileged_obs + vision_flag",
            "这让 critic 学得更稳，actor 仍保持部署友好。",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1500,
    )
    s.add_footer("核心文件：actor_critic_parkour_moe.py、ppo_parkour_moe.py、rollout_storage_parkour_moe.py")
    slides.append(s.build())

    s = SlideBuilder("7", "4.1 网络结构解析", "先看模块接口：每个组件在训练时到底负责什么", bg=BG_ALT)
    s.add_box(
        720_000,
        1_760_000,
        3_330_000,
        3_360_000,
        lines=[
            "Estimator 负责“感知压缩”",
            "输入: proprio history + depth + vision flag",
            "输出: mcp_code, terrain_logits, m_hat, o_hat, z_mu, z_logvar",
            "训练目标: vt / ht / mt / terrain_id / reconstruction / z_kl / swav / load_balance",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1460,
    )
    s.add_box(
        4_450_000,
        1_760_000,
        3_330_000,
        3_360_000,
        lines=[
            "ActorCritic 负责“控制优化”",
            "actor.act(): 产生动作采样",
            "critic.evaluate(): 给出 V(s)",
            "get_actions_log_prob(): 提供 PPO ratio 所需的新旧 log_prob",
            "std 是独立可学习参数，不依赖输入。",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1460,
    )
    s.add_box(
        8_180_000,
        1_760_000,
        3_280_000,
        3_360_000,
        lines=[
            "Storage 负责“把一整段 rollout 变成训练批”",
            "add_transitions(): 存 step 级数据",
            "compute_returns(): 反向算 GAE",
            "mini_batch_generator(): 展平 T x N 后随机打乱",
            "同时保留 old_mu / old_sigma 供 adaptive KL 使用。",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
        size=1460,
    )
    s.add_footer("所以这套实现不是单一神经网络，而是 estimator、actor-critic、storage、runner 协同组成的训练系统")
    slides.append(s.build())

    s = SlideBuilder("8", "4.2 网络结构解析", "onpolicyrunner_parkour_moe.learn() 的外层伪代码：先看 iteration 级控制流", bg=BG)
    s.add_box(
        720_000,
        1_700_000,
        7_050_000,
        4_500_000,
        lines=[
            "obs = env.get_observations(); privileged_obs = env.get_privileged_observations()",
            "additional_obs = env.get_additional_observations(); history.zero_(); update_history(obs)",
            "for it in range(current_learning_iteration, tot_iter):",
            "  if easy terrain and it % vision_toggle_interval == 0:",
            "      mask[easy_idx] = ~mask[easy_idx]",
            "  reset estimator metric accumulators, mcp_code = None",
            "  for step in range(num_steps_per_env):",
            "      do rollout collection",
            "  alg.compute_returns(critic_obs, mcp_code)",
            "  mean_value_loss, mean_surrogate_loss, mean_entropy_loss = alg.update()",
            "  log(...); if it % save_interval == 0: save(model_it.pt)",
            "current_learning_iteration += num_learning_iterations; save(last_model)",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1280,
    )
    s.add_box(
        8_080_000,
        1_700_000,
        3_380_000,
        4_500_000,
        lines=[
            "这一层决定了什么",
            f"• 每轮先采 {runner['num_steps_per_env']} 步，再更新",
            "• vision mask 只在 easy terrain 周期翻转",
            "• 一轮 rollout 结束后才统一 compute_returns()",
            "• PPO 主更新与 estimator 更新是交织的，但时序不同",
            "• save / log 都挂在 iteration 粒度上",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1450,
    )
    s.add_footer("如果把 training loop 看成一棵树，这页是最外层 for-it 结构")
    slides.append(s.build())

    s = SlideBuilder("9", "4.3 网络结构解析", "onpolicyrunner_parkour_moe.learn() 的内层伪代码：每一个 env step 里发生了什么", bg=BG_ALT)
    s.add_box(
        720_000,
        1_700_000,
        7_050_000,
        4_620_000,
        lines=[
            "critic_obs = privileged_obs if exists else obs",
            "refresh_estimator = (common_step_counter % camera_update_interval == 0) or (mcp_code is None)",
            "if refresh_estimator:",
            "  obs_now, proprio_hist = extract_obs_and_history(obs)",
            "  gt_vt, gt_ht, gt_mt, gt_terrain_id = get_estimator_targets(critic_obs)",
            "  camera_depth = additional_obs['depth_camera']; mask_snapshot = mask.clone()",
            "  est_out = alg.estimator(proprio_hist, camera_depth, mask_snapshot, gt_mt_step, obs_now)",
            "  mcp_code = append_vision_flag_to_mcp(est_out['mcp_code'], mask_snapshot)",
            "with torch.inference_mode(): actions = alg.act(mcp_code, obs, critic_obs)",
            "obs, privileged_obs, rewards, dones, infos = env.step(actions)",
            "update_history(obs, dones)",
            "if refresh_estimator:",
            "  build_swav_window(...); alg.update_estimator(...); alg.update_image_recon(camera_depth)",
            "alg.process_env_step(rewards, dones, infos); alg.estimator.reset(dones); reset_swav_history(dones)",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1240,
    )
    s.add_box(
        8_080_000,
        1_700_000,
        3_380_000,
        4_620_000,
        lines=[
            "这一层的关键细节",
            f"• estimator 不是每步都刷新，而是按 camera_update_interval={env_cfg['camera']['update_interval']}",
            "• actor 平时可以复用上一帧的 mcp_code",
            "• estimator 更新与 image AE 更新发生在 rollout 收集阶段内部",
            "• 但 PPO 主更新仍在整段 rollout 结束后统一进行",
            "• 因而这是“online data collection + delayed PPO update”的混合节奏",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
        size=1400,
    )
    s.add_footer("这页最像“伪代码”。它对应的是 runner 里 step 级别的真实执行顺序，而不是抽象框图")
    slides.append(s.build())

    s = SlideBuilder("10", "4.4 网络结构解析", "PPOParkourMoE.update() 与 RolloutStorage：真正的 PPO 损失是怎样从缓存数据里算出来的", bg=BG)
    s.add_box(
        720_000,
        1_760_000,
        5_180_000,
        3_760_000,
        lines=[
            "RolloutStorage 伪代码",
            "add_transitions():",
            "  store observations, critic_observations, mcp_code, actions",
            "  store rewards, dones, values, actions_log_prob, mu, sigma",
            "compute_returns(last_values, gamma, lam):",
            "  for step in reversed(range(T)):",
            "      delta = r + gamma * next_value * (1-done) - value",
            "      advantage = delta + gamma * lam * (1-done) * advantage",
            "      returns[step] = advantage + value",
            "  advantages = normalize(returns - values)",
            "mini_batch_generator(): flatten [T, N] -> [T*N], random permute, yield mini-batches",
        ],
        fill=PANEL,
        line=ACCENT,
        size=1240,
    )
    s.add_box(
        6_220_000,
        1_760_000,
        5_240_000,
        3_760_000,
        lines=[
            "PPOParkourMoE.update() 伪代码",
            "for each mini-batch:",
            "  actor_critic.act(mcp_code_batch, obs_batch)",
            "  log_prob_new = get_actions_log_prob(actions_batch)",
            "  value_new = evaluate(critic_obs_batch, mcp_code_batch)",
            "  ratio = exp(log_prob_new - log_prob_old)",
            "  surrogate = max(-A * ratio, -A * clip(ratio, 1-eps, 1+eps))",
            "  value_loss = clipped value loss",
            "  loss = surrogate + c_v * value_loss - c_e * entropy",
            "  if adaptive KL: 根据 old_mu / old_sigma 调 learning_rate",
            "  zero_grad(); backward(); clip_grad_norm(); optimizer.step()",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1240,
    )
    s.add_box(
        1_360_000,
        5_920_000,
        9_380_000,
        420_000,
        lines=[
            f"对应当前配置：每轮共有 {total_batch} 个样本，切成 {algorithm['num_mini_batches']} 个 mini-batch，每轮实际执行 {total_update_steps} 次参数更新。",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
        size=1400,
        bold=True,
        align="ctr",
        anchor="ctr",
    )
    s.add_footer("基础 PPO 看 ppo.py；本项目版本在 update_estimator() 和 update_image_recon() 上额外并行训练感知模块")
    slides.append(s.build())

    s = SlideBuilder("11", "5. 算法的应用", "在 parkour_moe 中，PPO 被用于四足机器人跨多类地形的感知控制联合训练", bg=BG_ALT)
    s.add_box(
        720_000,
        1_760_000,
        4_960_000,
        2_060_000,
        lines=[
            "应用场景",
            "• Go2 四足机器人 parkour",
            "• 视觉 + 本体多模态输入",
            "• 当前观测、10 步历史、2 帧 proxy depth、特权 critic",
            "• 目标不仅是前进，还包括 clearance、targeted foothold、turn-over recovery",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        6_040_000,
        1_760_000,
        5_380_000,
        2_060_000,
        lines=[
            "为什么仍然选 PPO",
            "• 适合大规模并行采样，训练吞吐高",
            "• actor-critic 结构天然兼容 privileged critic",
            "• 可以稳妥叠加 estimator、terrain routing、auxiliary losses",
            "• 在 MuJoCo/真机部署前，先在仿真中形成稳定优化闭环",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    chip_y = 4_050_000
    chip_w = 1_640_000
    chip_h = 560_000
    terrain_rows = [
        ["single_gap", "step_stone", "two_row_stones", "one_row_stones", "single_bridge", "air_beams"],
        ["air_stones", "hurdle", "ramp", "corridor", "stairs_up", "flat"],
    ]
    fills = [PANEL_SAND, PANEL_BLUE, PANEL_TEAL]
    lines = [ACCENT_4, ACCENT_3, ACCENT_2]
    for row_idx, row in enumerate(terrain_rows):
        for col_idx, terrain in enumerate(row):
            s.add_box(
                720_000 + col_idx * 1_780_000,
                chip_y + row_idx * 700_000,
                chip_w,
                chip_h,
                lines=[terrain],
                fill=fills[(row_idx + col_idx) % len(fills)],
                line=lines[(row_idx + col_idx) % len(lines)],
                align="ctr",
                anchor="ctr",
                bold=True,
                size=1200,
            )
    s.add_footer("地形名来自 legged_gym/utils/terrain_mgdp.py；部署路径可见 deploy/deploy_mujoco 与 deploy/deploy_real")
    slides.append(s.build())

    reward_stats = curve_stats["Train/mean_reward"]
    len_stats = curve_stats["Train/mean_episode_length"]
    terrain_stats = curve_stats["Episode/terrain_level_all"]
    no_progress_stats = curve_stats["Termination/no_progress_frac"]

    s = SlideBuilder("12", "6. 算法的性能曲线", "来自 ZApr19_16-24-31_GOOD 的真实训练日志：回报提升、课程推进、停滞率下降", bg=BG)
    s.add_picture(
        760_000,
        1_700_000,
        8_050_000,
        4_390_000,
        path=curves_path,
        name="ppo_training_curves",
    )
    s.add_box(
        9_020_000,
        1_900_000,
        2_430_000,
        3_460_000,
        lines=[
            "关键读图结论",
            f"• Mean reward: final {fmt(reward_stats.last_value)}",
            f"  peak {fmt(reward_stats.max_value)} @ {reward_stats.max_step}",
            f"• Episode length: final {fmt(len_stats.last_value)}",
            f"  peak {fmt(len_stats.max_value)} @ {len_stats.max_step}",
            f"• Terrain level: final {fmt(terrain_stats.last_value)}",
            f"• No-progress frac: final {fmt(100.0 * no_progress_stats.last_value)}%",
            "• 曲线并非单调上升，说明课程难度提升后仍有正常波动",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
        size=1500,
    )
    s.add_footer("数据来源：logs/go2_parkour_moe/ZApr19_16-24-31_GOOD/events.out.tfevents.*")
    slides.append(s.build())

    s = SlideBuilder("13", "7. 结论", "PPO 在本项目中的角色是“稳定优化骨架”，而不是唯一决定性能的模块", bg=BG_ALT)
    s.add_box(
        720_000,
        1_760_000,
        3_260_000,
        2_600_000,
        lines=[
            "从论文视角看",
            "• PPO 把 trust-region 思想改写成更好实现的 clip 目标",
            "• 一阶优化即可得到较强稳定性",
            "• 适合连续控制与大规模并行训练",
        ],
        fill=PANEL,
        line=ACCENT,
    )
    s.add_box(
        4_450_000,
        1_760_000,
        3_260_000,
        2_600_000,
        lines=[
            "从代码视角看",
            "• rsl_rl 实现了 GAE、ratio clipping、value clipping、entropy 与 adaptive KL",
            "• actor 与 critic 明确分工，storage 负责多轮样本复用",
            "• parkour_moe 版本进一步支持 estimator / depth AE 联训",
        ],
        fill=PANEL_TEAL,
        line=ACCENT_2,
    )
    s.add_box(
        8_180_000,
        1_760_000,
        3_280_000,
        2_600_000,
        lines=[
            "从项目视角看",
            "• 真正拉开任务性能的是多模态估计器、terrain-aware MoE 和 reward 设计",
            "• 但这些能力需要一个足够稳的外层优化器来承载",
            "• 本仓库里，这个角色正是 PPO",
        ],
        fill=PANEL_BLUE,
        line=ACCENT_3,
    )
    s.add_box(
        1_360_000,
        4_740_000,
        9_540_000,
        820_000,
        lines=[
            "最后一句：如果把 ParkourMoE 看成一辆车，估计器和 MoE 决定“车能去哪”，PPO 决定“车能否稳定开到那”。",
        ],
        fill=PANEL_SAND,
        line=ACCENT_4,
        size=1700,
        bold=True,
        align="ctr",
        anchor="ctr",
    )
    s.add_footer("局限性：PPO 仍是 on-policy，样本效率不如一些 off-policy 方法；性能也强依赖观测设计与奖励塑形")
    slides.append(s.build())

    return slides


def write_pptx(out_path: Path, slides: list[SlideSpec]) -> None:
    slide_titles = [slide.title for slide in slides]
    media_name_by_path: dict[Path, str] = {}
    image_exts: set[str] = set()

    for slide in slides:
        for picture in slide.pictures:
            path = picture.path.resolve()
            suffix = picture.path.suffix.lower()
            image_exts.add(suffix)
            if path not in media_name_by_path:
                media_name_by_path[path] = f"image{len(media_name_by_path) + 1}{suffix}"

    with ZipFile(out_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(len(slides), image_exts))
        zf.writestr("_rels/.rels", ROOT_RELS_XML)
        zf.writestr("docProps/core.xml", core_xml(slide_titles[0]))
        zf.writestr("docProps/app.xml", app_xml(slide_titles))
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels_xml(len(slides)))
        zf.writestr("ppt/presProps.xml", PRES_PROPS_XML)
        zf.writestr("ppt/viewProps.xml", VIEW_PROPS_XML)
        zf.writestr("ppt/theme/theme1.xml", THEME_XML)
        zf.writestr("ppt/slideMasters/slideMaster1.xml", MASTER_XML)
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", MASTER_RELS_XML)
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT_XML)
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", LAYOUT_RELS_XML)

        for idx, slide in enumerate(slides, start=1):
            media_targets: list[str] = []
            slide_elements = list(slide.elements)
            for rel_offset, picture in enumerate(slide.pictures, start=2):
                media_name = media_name_by_path[picture.path.resolve()]
                media_targets.append(media_name)
                slide_elements.append(picture_xml(picture, f"rId{rel_offset}"))
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide_xml(slide.bg, slide_elements))
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels_xml(media_targets))

        for path, media_name in media_name_by_path.items():
            zf.write(path, f"ppt/media/{media_name}")


def main() -> None:
    if not EVENT_PATH.exists():
        raise FileNotFoundError(f"TensorBoard event file not found: {EVENT_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Run config not found: {CONFIG_PATH}")

    cfg = load_run_cfg()
    curve_stats = generate_curves_image(CURVE_IMAGE_PATH)
    slides = build_slides(CURVE_IMAGE_PATH, curve_stats, cfg)
    write_pptx(OUTPUT_PPTX, slides)
    print(f"Generated curves: {CURVE_IMAGE_PATH}")
    print(f"Generated pptx:   {OUTPUT_PPTX}")


if __name__ == "__main__":
    main()

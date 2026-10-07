"""Render source-checked RM75 control architecture figures.

Run from any directory: python doc/control_architecture/render_diagrams.py
The figures describe registered task configurations, not measured performance.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
BG = "#f7f9fc"
INK = "#17283d"
MUTED = "#536579"
LINE = "#94a4b5"
BLUE = "#25679a"
TEAL = "#207e76"
PURPLE = "#7055a2"
ORANGE = "#b77726"
RED = "#ad5768"
PALE_BLUE = "#e9f2fb"
PALE_TEAL = "#e8f5f1"
PALE_PURPLE = "#f0ecf8"
PALE_ORANGE = "#fff3e5"
PALE_RED = "#faecef"
WHITE = "#ffffff"

plt.rcParams.update(
    {
        "font.family": "Microsoft YaHei",
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)


def canvas(width, height, title, subtitle):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.text(58, 52, title, fontsize=23, fontweight="bold", color=INK, va="center")
    ax.text(59, 88, subtitle, fontsize=11.5, color=MUTED, va="center")
    ax.plot([58, width - 58], [110, 110], color="#d4dee8", linewidth=1.2)
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), accent=BLUE, fill=WHITE, fs=11):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.014,rounding_size=16",
            linewidth=1.15, edgecolor="#c8d4df", facecolor=fill,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 1, y + 1), 7, h - 2,
            boxstyle="round,pad=0,rounding_size=3",
            linewidth=0, facecolor=accent,
        )
    )
    ax.text(x + 23, y + 28, title, fontsize=14, fontweight="bold", color=INK, va="center")
    for i, line in enumerate(lines):
        ax.text(x + 23, y + 57 + i * 24, line, fontsize=fs, color=MUTED, va="center")


def arrow(ax, start, end, label=None, color=LINE, style="-|>", dashed=False, rad=0):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=16,
        linewidth=1.9, linestyle="--" if dashed else "-", color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1,
    )
    ax.add_patch(patch)
    if label:
        xm = (start[0] + end[0]) / 2
        ym = (start[1] + end[1]) / 2 - 10
        ax.text(xm, ym, label, fontsize=10.5, color=color, ha="center", va="center",
                bbox={"facecolor": BG, "edgecolor": "none", "pad": 2})


def section(ax, x, y, number, title, subtitle="", color=BLUE):
    ax.text(x, y, number, fontsize=13, color=WHITE, fontweight="bold", ha="center", va="center",
            bbox={"boxstyle": "circle,pad=0.40", "facecolor": color, "edgecolor": "none"})
    ax.text(x + 27, y, title, fontsize=14, color=INK, fontweight="bold", va="center")
    if subtitle:
        ax.text(x + 27, y + 23, subtitle, fontsize=10.5, color=MUTED, va="center")


def footer(ax, width, height, text):
    ax.plot([58, width - 58], [height - 43, height - 43], color="#d4dee8", linewidth=1)
    ax.text(58, height - 19, text, fontsize=9.5, color=MUTED, va="center")


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", facecolor=BG, bbox_inches="tight", pad_inches=0.09)
    fig.savefig(OUT / f"{name}.png", facecolor=BG, bbox_inches="tight", pad_inches=0.09, dpi=150)
    plt.close(fig)


def overview():
    w, h = 1800, 1080
    fig, ax = canvas(w, h, "RM75 强化控制项目 · 总体架构", "已注册的三个策略分支按任务单独运行；实线为推理与控制数据流，虚线为训练或权重迁移。")

    section(ax, 78, 155, "1", "环境与观测", "详见图 02", BLUE)
    box(ax, 68, 195, 320, 154, "仿真任务与机器人", (
        "RM75 固定臂，12 个腿关节",
        "地球重力；平地 / 坡地 / 多地形",
        "速度命令、接触和奖励",
    ), BLUE, PALE_BLUE)
    box(ax, 68, 400, 320, 189, "可部署输入", (
        "当前本体观测 45 维",
        "视觉任务：历史 10×45",
        "代理深度 1×58×87",
        "视觉可用标志 1 维",
    ), TEAL, PALE_TEAL)
    arrow(ax, (228, 350), (228, 398), color=BLUE)

    section(ax, 482, 155, "2", "任务策略", "从下列三条分支中选择一条；详见图 03", PURPLE)
    box(ax, 470, 195, 810, 154, "A · 平地小跑", (
        "45维本体观测 → MLP Actor [512,256,128] → 12维动作",
        "训练：标准 PPO；Critic 输入 77 维特权观测",
        "目标：速度跟踪、对角小跑、稳定和防滑",
    ), BLUE, WHITE)
    box(ax, 470, 390, 810, 173, "B · 视觉坡地小跑", (
        "深度＋本体历史 → 视觉估计器 → 69维控制码＋视觉标志",
        "平地 Actor ＋ 视觉残差 MLP [256,128] → 12维动作",
        "训练：扩展 PPO 与估计器监督；配置平地权重热启动",
    ), TEAL, WHITE)
    box(ax, 470, 605, 810, 173, "C · 多地形 Parkour MoE", (
        "使用同类估计器结构；各任务独立训练参数",
        "当前观测＋控制码＋标志 → 8专家动作 MoE → 12维动作",
        "训练：扩展 PPO、地形课程及估计器辅助损失",
    ), PURPLE, WHITE)
    arrow(ax, (389, 495), (468, 272), color=BLUE)
    arrow(ax, (389, 495), (468, 477), color=TEAL)
    arrow(ax, (389, 495), (468, 691), color=PURPLE)
    arrow(ax, (1200, 350), (1190, 385), "权重热启动", color=BLUE, dashed=True)

    section(ax, 1390, 155, "3", "关节执行", "详见图 03", ORANGE)
    box(ax, 1365, 345, 350, 240, "共同的动作执行层", (
        "每次仅接收当前任务的 12 维动作",
        "目标角：q* = q0 + 0.20 × a",
        "PD：Kp=500，Kd=18",
        "力矩裁剪 / 可选关节限矩",
        "物理步 0.005 s；策略步 0.020 s",
    ), ORANGE, PALE_ORANGE)
    for yy in (272, 477, 691):
        arrow(ax, (1281, yy), (1363, 465), color=ORANGE)
    ax.plot([1540, 1540, 405, 405], [344, 128, 128, 272], color=ORANGE, linewidth=1.9)
    arrow(ax, (405, 272), (389, 272), color=ORANGE)
    ax.text(1090, 137, "执行后返回下一观测", fontsize=10.5, color=ORANGE, ha="center",
            bbox={"facecolor": BG, "edgecolor": "none", "pad": 2})

    section(ax, 78, 832, "4", "训练、评估与边界", "详见图 04", RED)
    box(ax, 68, 872, 505, 130, "策略优化", (
        "环境奖励＋Critic 价值 → GAE / PPO",
        "平地用基础 PPO；视觉任务用扩展 PPO",
    ), RED, PALE_RED)
    box(ax, 615, 872, 505, 130, "估计器辅助训练", (
        "特权真值监督速度 / 足高 / 地形图等",
        "控制码先 detach，再交给 PPO Actor",
    ), TEAL, PALE_TEAL)
    box(ax, 1162, 872, 553, 130, "验证与部署入口", (
        "Isaac Gym 回放、模型导出、MuJoCo 验证",
        "效果须按任务和 checkpoint 实测；规划项另列",
    ), PURPLE, PALE_PURPLE)
    footer(ax, w, h, "图 01 · 代码级结构总览；图 02–04 展开各模块。配置值随任务变体与命令行覆盖而变化。")
    save(fig, "01_overview")


def estimator():
    w, h = 1900, 1090
    fig, ax = canvas(w, h, "视觉估计器 · 输入到 69 维控制码", "RM75 视觉坡地与 Parkour 使用同类估计器结构，但各任务参数分别训练。")
    section(ax, 78, 154, "2", "视觉与本体输入", "代理深度由仿真地形射线投射得到；不是 SLAM 全局地图。", TEAL)

    box(ax, 65, 210, 310, 142, "深度输入", (
        "1 帧 × 58 × 87",
        "更新间隔 5 个策略步 ≈ 10 Hz",
    ), BLUE, PALE_BLUE)
    box(ax, 425, 210, 350, 142, "深度 CNN 与局部重建", (
        "3层 stride-2 Conv：8×11×64",
        "局部 patch 解码 / MSE 辅助损失",
    ), BLUE, WHITE)
    box(ax, 825, 210, 350, 142, "主动视觉 token", (
        "88 个密集 token → 16 个采样 token",
        "空间遮挡训练；采样点带坐标编码",
    ), BLUE, PALE_BLUE)
    arrow(ax, (376, 280), (423, 280), color=BLUE)
    arrow(ax, (776, 280), (823, 280), color=BLUE)

    box(ax, 65, 420, 310, 142, "本体历史", (
        "10 步 × 每步 45 维",
        "每步由 MLP 编成 64维 token",
    ), TEAL, PALE_TEAL)
    box(ax, 425, 420, 350, 142, "视觉标志", (
        "训练由任务掩码控制",
        "另附 1 个标志 token 参与融合",
    ), TEAL, WHITE)
    box(ax, 825, 420, 350, 142, "跨模态 Transformer", (
        "最多 16 视觉＋10 本体＋1 标志",
        "1 层，4 个注意力头，token宽64",
    ), TEAL, PALE_TEAL)
    ax.plot([376, 401, 401, 805, 805], [490, 490, 385, 385, 490], color=TEAL, linewidth=1.9)
    arrow(ax, (805, 490), (823, 490), color=TEAL)
    arrow(ax, (776, 490), (823, 490), color=TEAL)
    arrow(ax, (1000, 354), (1000, 418), color=BLUE)

    box(ax, 1245, 210, 575, 166, "运动与地形时间表示", (
        "运动 GRU：提取运动历史特征",
        "3 个 ReadWriteExpert：读共享状态、GRUCell、写增量",
        "路由加权增量 → 共享状态门控更新",
    ), PURPLE, PALE_PURPLE)
    arrow(ax, (1176, 490), (1243, 294), color=PURPLE)
    box(ax, 1245, 420, 575, 226, "估计器前向输出 · MCP 码 69 维", (
        "速度 3 ＋ 足高 4 ＋ 潜变量均值 16",
        "地形潜变量 32 ＋ 地形类别 one-hot 13",
        "恢复判别 1 ＝ 69",
        "Runner 再附视觉标志 1 → Actor 接收 70维编码",
        "另有高度图 187维等训练用解码输出",
    ), PURPLE, WHITE)
    arrow(ax, (1530, 377), (1530, 418), color=PURPLE)

    section(ax, 78, 715, "+", "训练监督与部署边界", "训练目标不等同于在线可用传感器。", RED)
    box(ax, 65, 757, 805, 215, "辅助目标 / 独立优化", (
        "速度、足高、187点地形图、下一观测：回归损失",
        "地形类别：交叉熵；恢复标志：二元交叉熵",
        "局部 patch 重建、潜变量 KL、门控/地形 SwAV",
        "估计器独立 Adam；MCP 码送 PPO 前 detach",
    ), RED, PALE_RED)
    box(ax, 930, 757, 890, 215, "视觉可靠性：当前实现的边界", (
        "网络会计算 token 重建误差 / confidence",
        "常规 runner / 当前实机入口的视觉标志由任务或外部输入提供",
        "MuJoCo 脚本另有可选阈值掩码；不等同于论文图9的实机结果",
        "RM75 配置未开启翻身初始化；恢复输出存在不代表已训练该能力",
    ), ORANGE, PALE_ORANGE)
    footer(ax, w, h, "图 02 · 输入维度与输出维度来自当前代码；训练目标的具体权重和模式由任务配置决定。")
    save(fig, "02_estimator")


def policies():
    w, h = 1840, 1050
    fig, ax = canvas(w, h, "三条控制策略 · 不同 Actor，同一关节接口", "同一 RM75 模型与 12 维位置偏置动作；每次运行只实例化当前任务的策略。")
    section(ax, 78, 157, "A", "平地小跑", "无深度与视觉估计器", BLUE)
    box(ax, 65, 200, 355, 145, "输入", ("本体观测 45维", "Critic 特权观测 77维"), BLUE, PALE_BLUE)
    box(ax, 500, 200, 620, 145, "Actor / Critic", (
        "Actor MLP：45→512→256→128→12",
        "Critic MLP：77→512→256→128→1",
    ), BLUE, WHITE)
    box(ax, 1200, 200, 570, 145, "功能与训练", (
        "3 m/s 是目标命令；奖励鼓励对角小跑、速度、稳定",
        "高斯动作策略；基础 PPO",
    ), BLUE, PALE_BLUE)
    arrow(ax, (422, 273), (498, 273), color=BLUE)
    arrow(ax, (1122, 273), (1198, 273), color=BLUE)

    section(ax, 78, 395, "B", "视觉坡地小跑", "平地 Actor 权重热启动；动作端是残差结构", TEAL)
    box(ax, 65, 440, 355, 175, "输入", (
        "当前观测 45维",
        "估计器 MCP 69维＋视觉标志 1维",
        "估计器另见图 02",
    ), TEAL, PALE_TEAL)
    box(ax, 500, 440, 620, 175, "两个 Actor 输出相加", (
        "基础 MLP：45→512→256→128→12",
        "视觉残差 MLP：115→256→128→12",
        "动作均值 = 基础输出＋残差输出",
    ), TEAL, WHITE)
    box(ax, 1200, 440, 570, 175, "功能与训练", (
        "连续坡地 25° / 35° 配置；坡面切向/法向奖励",
        "视觉估计器单独更新；Critic 270维输入",
        "扩展 PPO；基础 Actor 与残差分阶段解冻",
    ), TEAL, PALE_TEAL)
    arrow(ax, (422, 527), (498, 527), color=TEAL)
    arrow(ax, (1122, 527), (1198, 527), color=TEAL)
    arrow(ax, (845, 347), (845, 436), "基础权重热启动", color=BLUE, dashed=True)

    section(ax, 78, 665, "C", "多地形 Parkour MoE", "动作端是真正的 8 专家混合；RM75 关闭翻身初始化", PURPLE)
    box(ax, 65, 710, 355, 175, "输入", (
        "当前观测 45维",
        "估计器 MCP 69维＋视觉标志 1维",
        "Actor 拼接输入 115维",
    ), PURPLE, PALE_PURPLE)
    box(ax, 500, 710, 620, 175, "门控动作专家", (
        "共享 MLP → 分组 Conv1D：8 个专家各输出12维",
        "softmax 路由加权 → 12维动作",
        "Critic 专家值用 Actor 门控权重加权",
    ), PURPLE, WHITE)
    box(ax, 1200, 710, 570, 175, "功能与训练", (
        "沟 / 石块 / 桥 / 梁 / 障碍 / 坡 / 楼梯等地形",
        "地形课程＋针对性落脚、净空等奖励",
        "扩展 PPO＋Actor 专家负载均衡；Critic 270维",
    ), PURPLE, PALE_PURPLE)
    arrow(ax, (422, 797), (498, 797), color=PURPLE)
    arrow(ax, (1122, 797), (1198, 797), color=PURPLE)

    box(ax, 500, 921, 770, 70, "共同输出 12 维 a → q* = q0 + 0.20a → PD / 力矩限制 → RM75", (), ORANGE, PALE_ORANGE)
    footer(ax, w, h, "图 03 · 任务配置变体还会改变速度范围、坡度课程、限矩和学习率；详见随附文档。")
    save(fig, "03_policies")


def training():
    w, h = 1800, 1000
    fig, ax = canvas(w, h, "训练与验证 · 两条参数更新路径", "PPO 优化 Actor/Critic；视觉估计器通过单独的监督损失更新。")
    section(ax, 80, 160, "4", "训练闭环", "策略步 0.020 s；每轮采样 24 步。", RED)
    box(ax, 65, 215, 350, 190, "Isaac Gym 多环境", (
        "RM75、地形与速度命令",
        "输出观测、特权真值、奖励、done",
        "平地 4096 / 坡地 2048 / Parkour 4096",
    ), BLUE, PALE_BLUE)
    box(ax, 475, 215, 350, 190, "当前任务的策略", (
        "平地 MLP / 坡地残差 / Parkour MoE",
        "采样动作，PD 执行，形成轨迹",
        "视觉分支还运行估计器",
    ), PURPLE, PALE_PURPLE)
    box(ax, 885, 215, 350, 190, "Rollout 与 GAE", (
        "24步轨迹，价值估计和奖励",
        "γ=0.99；GAE λ=0.95",
        "计算 return / advantage",
    ), ORANGE, PALE_ORANGE)
    box(ax, 1295, 215, 430, 190, "PPO 更新", (
        "clip=0.2；5 epochs × 4 mini-batches",
        "Actor 策略损失＋Critic 价值损失",
        "Parkour Actor 另加专家负载均衡",
    ), RED, PALE_RED)
    arrow(ax, (416, 307), (473, 307), color=BLUE)
    arrow(ax, (826, 307), (883, 307), color=PURPLE)
    arrow(ax, (1236, 307), (1293, 307), color=ORANGE)
    arrow(ax, (1500, 212), (1500, 180), color=RED)
    arrow(ax, (1500, 180), (645, 180), "更新 Actor / Critic 后继续采样", color=RED)

    section(ax, 80, 500, "5", "视觉估计器独立更新", "仅视觉坡地与 Parkour；MCP 码交给 PPO 前 detach。", TEAL)
    box(ax, 65, 548, 525, 210, "估计器输入", (
        "10×45 本体历史＋1×58×87 代理深度",
        "训练可施加空间块 token dropout / 视觉掩码",
        "输出前向控制码与训练用解码结果",
    ), TEAL, PALE_TEAL)
    box(ax, 645, 548, 525, 210, "监督与正则", (
        "速度、足高、地形图、下一观测：MSE",
        "地形类别 CE；恢复标志 BCE",
        "局部重建＋潜变量 KL＋SwAV",
    ), TEAL, WHITE)
    box(ax, 1225, 548, 500, 210, "独立优化 / 边界", (
        "估计器独立 Adam，学习率通常 1e-3",
        "PPO 不通过 detach 的 MCP 码反传到估计器",
        "RM75 turn_over=False：勿宣称已训练翻身恢复",
    ), TEAL, PALE_TEAL)
    arrow(ax, (592, 652), (643, 652), color=TEAL)
    arrow(ax, (1172, 652), (1223, 652), color=TEAL)
    ax.plot([240, 240, 42, 42], [407, 466, 466, 651], color=TEAL,
            linewidth=1.9, linestyle="--")
    arrow(ax, (42, 651), (63, 651), color=TEAL, dashed=True)

    section(ax, 80, 835, "6", "结果验证", "图中的流程是可用入口，不代表结果已复算。", PURPLE)
    box(ax, 65, 877, 1660, 78, "检查点 → Isaac Gym 回放 / 分地形评估 → 策略导出 → MuJoCo 复核与状态记录；月面低重力、活动臂和自主导航另属后续设计。", (), PURPLE, PALE_PURPLE, fs=10.5)
    footer(ax, w, h, "图 04 · 当前仓库的训练与验证调用链。")
    save(fig, "04_training")


def parkour_network():
    w, h = 2060, 1130
    fig, ax = canvas(w, h, "RM75 Parkour MoE · 神经网络总图", "一条独立注册策略：从可部署观测到 12 维腿部动作；下方训练支路仅在学习阶段使用。")
    section(ax, 78, 156, "1", "输入与编码", "神经网络展开见图 06", BLUE)
    box(ax, 66, 202, 285, 145, "代理深度", ("[B,1,58,87]", "10 Hz；射线投射地形"), BLUE, PALE_BLUE)
    box(ax, 66, 381, 285, 145, "本体历史", ("[B,10,45]", "10步关节/机身/命令"), TEAL, PALE_TEAL)
    box(ax, 66, 560, 285, 145, "当前观测 / 标志", (
        "obs_now [B,45] → 路由 / Actor",
        "flag [B,1] → 融合 / Actor",
    ), TEAL, PALE_TEAL)

    box(ax, 415, 202, 335, 145, "视觉编码", ("3层 Conv2d → [B,88,64]", "主动采样 K=16 → [B,16,64]"), BLUE, WHITE)
    box(ax, 415, 381, 335, 145, "本体编码", ("MLP 45→256→128→64", "输出 [B,10,64]"), TEAL, WHITE)
    box(ax, 800, 281, 330, 205, "跨模态 Transformer", (
        "16视觉＋10本体＋1标志 token",
        "1层，4头，d_model=64",
        "取融合后的10个本体 token",
        "输出 [B,10,64]",
    ), TEAL, PALE_TEAL)
    arrow(ax, (353, 273), (413, 273), color=BLUE)
    arrow(ax, (353, 453), (413, 453), color=TEAL)
    arrow(ax, (752, 273), (798, 355), color=BLUE)
    arrow(ax, (752, 453), (798, 410), color=TEAL)
    arrow(ax, (353, 630), (798, 465), "flag", color=TEAL)

    section(ax, 1190, 156, "2", "时间记忆与估计头", "结构展开见图 07", PURPLE)
    box(ax, 1180, 202, 340, 284, "双路径记忆", (
        "运动 GRU：64维",
        "共享查询/3专家查询注意力池化",
        "3× ReadWriteExpert + softmax路由",
        "写增量混合、共享状态门控",
        "共享状态维度 64",
    ), PURPLE, PALE_PURPLE)
    box(ax, 1570, 202, 410, 284, "状态估计头 / MCP", (
        "速度3＋足高4＋潜变量均值16",
        "地形潜变量32＋地形类别13",
        "恢复判别1 ＝ MCP [B,69]",
        "+ vision_flag1 → [B,70]",
        "另有187点地形图等训练输出",
    ), PURPLE, WHITE)
    arrow(ax, (1132, 385), (1178, 385), color=PURPLE)
    arrow(ax, (1522, 345), (1568, 345), color=PURPLE)

    section(ax, 78, 754, "3", "动作与价值网络", "逐层展开见图 08", ORANGE)
    box(ax, 66, 805, 575, 160, "Actor · 只用可部署输入", (
        "obs_now45＋MCP69＋flag1＝115维",
        "共享MLP / 8组Conv1D专家 / softmax门控",
        "加权成12维关节动作；PD输出力矩",
    ), ORANGE, PALE_ORANGE)
    box(ax, 695, 805, 575, 160, "Critic · 只在训练中评价", (
        "特权观测269＋flag1＝270维",
        "8个价值专家；沿用Actor gate加权",
        "输出 V(s) [B,1]",
    ), RED, PALE_RED)
    box(ax, 1324, 805, 656, 160, "两个优化路径", (
        "环境奖励＋Critic价值 → GAE / PPO → Actor、Critic",
        "估计器：回归/分类/重建/KL/SwAV → 独立Adam",
        "MCP在交给PPO前detach；RM75 turn_over=False",
    ), TEAL, PALE_TEAL)
    arrow(ax, (1780, 488), (1780, 709), color=ORANGE)
    arrow(ax, (1780, 709), (365, 709), color=ORANGE)
    arrow(ax, (365, 709), (365, 803), color=ORANGE)
    footer(ax, w, h, "图 05 · 网络前向与训练职责总览；论文 PRISM 的模块名称/公式不直接代替此实现。")
    save(fig, "05_parkour_network")


def parkour_perception():
    w, h = 2000, 1060
    fig, ax = canvas(w, h, "Parkour MoE · 感知前端逐层结构", "图 05 的模块 1：从代理深度、本体历史和视觉标志构造 Transformer token 序列。")
    section(ax, 78, 155, "a", "视觉分支", "局部重建是辅助目标；常规推理不以该误差自动生成视觉标志。", BLUE)
    box(ax, 65, 205, 305, 185, "代理深度 [B,1,58,87]", (
        "Conv2d 1→16 / k3,s2",
        "Conv2d 16→32 / k3,s2",
        "Conv2d 32→64 / k3,s2",
        "输出 [B,64,8,11]",
    ), BLUE, PALE_BLUE)
    box(ax, 430, 205, 320, 185, "密集视觉 token", (
        "flatten + transpose + LayerNorm",
        "8×11＝88 个位置",
        "形状 [B,88,64]",
        "每个位置有局部 patch 重建头",
    ), BLUE, WHITE)
    box(ax, 810, 205, 380, 185, "采样坐标预测", (
        "有效区域均值池化 [B,64]",
        "Linear 64→64→32 + Sigmoid",
        "reshape [B,16,2]，换算至[-1,1]",
        "训练可对坐标加小扰动",
    ), BLUE, PALE_BLUE)
    box(ax, 1250, 205, 670, 185, "双线性采样 / 可用性", (
        "masked feature map → grid_sample → [B,16,64]",
        "坐标 MLP 2→64→64，加至采样特征",
        "对掩码 grid_sample 得 valid_mass；低于0.05则padding",
        "注意：代码未将采样特征除以 valid_mass",
    ), BLUE, WHITE)
    arrow(ax, (372, 300), (428, 300), color=BLUE)
    arrow(ax, (752, 300), (808, 300), color=BLUE)
    arrow(ax, (1192, 300), (1248, 300), color=BLUE)
    box(ax, 430, 430, 760, 128, "训练辅助：局部 patch 解码", (
        "token 64 → Linear128 → patch 15×15；与深度局部块做 MSE",
        "token_confidence = exp(-reconstruction_error / temperature)",
    ), RED, PALE_RED)
    arrow(ax, (595, 391), (595, 428), color=RED, dashed=True)

    section(ax, 78, 635, "b", "本体与融合分支", "无效视觉位置作为 padding；训练可施加空间块遮挡。", TEAL)
    box(ax, 65, 682, 370, 180, "本体历史 [B,10,45]", (
        "逐步 MLP 45→256→128→64",
        "LayerNorm＋本体类型/时序编码",
        "输出 [B,10,64]",
    ), TEAL, PALE_TEAL)
    box(ax, 495, 682, 370, 180, "视觉 flag [B,1]", (
        "编码为 1 个64维标志 token",
        "控制视觉 token 的 padding",
        "由任务/外部状态提供",
    ), TEAL, WHITE)
    box(ax, 925, 682, 575, 180, "多模态 Transformer", (
        "concat：[16视觉,10本体,1标志] → [B,27,64]",
        "TransformerEncoder：1层、4头、FFN256",
        "仅取本体位置的输出 [B,10,64] 交给记忆模块",
    ), TEAL, PALE_TEAL)
    ax.plot([437, 462, 462, 902, 902], [772, 772, 600, 600, 772],
            color=TEAL, linewidth=1.9)
    arrow(ax, (902, 772), (923, 772), color=TEAL)
    arrow(ax, (867, 772), (923, 772), color=TEAL)
    arrow(ax, (1580, 392), (1580, 635), color=BLUE)
    arrow(ax, (1580, 635), (1502, 772), color=BLUE)
    footer(ax, w, h, "图 06 · 形状对应 RM75_parkour_moe 的当前配置；B 表示 batch 大小。")
    save(fig, "06_parkour_perception")


def parkour_memory():
    w, h = 2050, 1110
    fig, ax = canvas(w, h, "Parkour MoE · 时间记忆、路由与估计头", "图 05 的模块 2：实际代码是 ReadWriteExpert + 共享状态，不等同于 PRISM 式 (9)–(14)。")
    box(ax, 65, 185, 470, 150, "Transformer 融合本体 token", (
        "输入 [B,10,64]",
        "1 个共享查询＋3 个专家查询做注意力池化",
    ), TEAL, PALE_TEAL)
    box(ax, 600, 185, 660, 150, "地形/恢复选择与专家路由", (
        "共享池化 → 地形 logits13、恢复概率1、context64",
        "路由输入：[obs45,flag1,context64,地形32,恢复16]＝158",
        "MLP 158→128→3 + Softmax → 3 个专家权重",
    ), PURPLE, PALE_PURPLE)
    arrow(ax, (537, 260), (598, 260), color=PURPLE)

    section(ax, 78, 415, "c", "双路径时间建模", "两条时间路径产生不同的估计目标。", PURPLE)
    box(ax, 65, 463, 470, 195, "运动路径", (
        "融合本体 token 序列 [B,10,64]",
        "GRU 64→64，取最后隐藏状态",
        "运动 head 64→64",
        "估计速度3、足高4；另有运动潜变量训练头",
    ), BLUE, PALE_BLUE)
    box(ax, 600, 463, 660, 195, "3 个 ReadWriteExpert", (
        "每专家读取上一共享状态 s[t-1] [B,64]",
        "read MLP64→64→64 → GRUCell(feature64,hidden64)",
        "write MLP64→64→64，产生写增量 Δ_i",
        "以 softmax 权重混合专家隐藏状态与 Δ_i",
    ), PURPLE, WHITE)
    box(ax, 1325, 463, 655, 195, "共享状态更新", (
        "拼接 [s[t-1], Σg_iΔ_i] → [B,128]",
        "candidate MLP128→64→64；gate Linear128→64+Sigmoid",
        "s[t] = LayerNorm(gate·candidate + (1-gate)·s[t-1])",
        "地形 head [expert混合隐藏, s[t]]→64",
    ), PURPLE, PALE_PURPLE)
    ax.plot([65, 42, 42], [300, 300, 560], color=BLUE, linewidth=1.9)
    arrow(ax, (42, 560), (63, 560), color=BLUE)
    arrow(ax, (1262, 560), (1323, 560), color=PURPLE)
    arrow(ax, (940, 337), (940, 461), color=PURPLE)

    section(ax, 78, 748, "d", "输出头", "训练解码输出与交给 Actor 的前向控制码分开。", ORANGE)
    box(ax, 65, 793, 920, 215, "Actor 使用 · MCP [B,69]", (
        "运动头：v_t 3＋h_tf 4",
        "地形头：z_mu 16＋z_tm 32",
        "选择头：terrain one-hot 13＋fall_recovery_pred 1",
        "Runner 附 vision_flag 1 → [B,70]；再与 obs_now45 拼成115",
    ), ORANGE, PALE_ORANGE)
    box(ax, 1050, 793, 930, 215, "仅训练/诊断输出", (
        "z/logvar 与重参数化样本 → 潜变量 KL",
        "z_tm 32 → 地形图 decoder → m_hat 187",
        "运动/地形潜变量 → 下一观测 decoder → o_hat 45",
        "地形/恢复分类、门控窗口与地形图 → CE/BCE/SwAV",
    ), RED, PALE_RED)
    arrow(ax, (295, 660), (295, 791), color=BLUE)
    arrow(ax, (1650, 660), (1650, 791), color=PURPLE)
    ax.plot([1520, 1520, 920, 920], [660, 725, 725, 791],
            color=PURPLE, linewidth=1.9)
    footer(ax, w, h, "图 07 · RM75 配置关闭翻身初始化；恢复判别头存在不表示具备已验证的自起立能力。")
    save(fig, "07_parkour_memory")


def parkour_actor_critic():
    w, h = 2050, 1080
    fig, ax = canvas(w, h, "Parkour MoE · Actor、Critic 与优化", "图 05 的模块 3：共享特征、分组 Conv1D 专家与 softmax 门控的实际张量流。")
    section(ax, 78, 155, "e", "部署 Actor", "只接收可部署观测；每个专家都提出一组 12 维动作。", ORANGE)
    box(ax, 65, 202, 360, 188, "拼接输入 [B,115]", (
        "obs_now 45",
        "mcp_code 69",
        "vision_flag 1",
    ), ORANGE, PALE_ORANGE)
    box(ax, 485, 202, 455, 188, "共享主干与专家", (
        "MLP 115→512→256→1024",
        "reshape为 8×128 专家特征",
        "groups=8 的 1×1 Conv1D",
        "专家输出 [B,8,12]",
    ), ORANGE, WHITE)
    box(ax, 1000, 202, 455, 188, "门控网络", (
        "MLP 115→512→256→8",
        "Softmax → gate [B,8]",
        "Σ_i gate_i·expert_action_i",
        "动作均值 μ [B,12]",
    ), ORANGE, PALE_ORANGE)
    box(ax, 1515, 202, 465, 188, "物理动作", (
        "训练：Normal(μ,可学习std)采样 a",
        "推理：用均值 μ",
        "q*=q0+0.20a → PD / 力矩裁剪",
        "RM75 腿关节 ×12",
    ), ORANGE, WHITE)
    arrow(ax, (427, 296), (483, 296), color=ORANGE)
    arrow(ax, (942, 296), (998, 296), color=ORANGE)
    arrow(ax, (1457, 296), (1513, 296), color=ORANGE)
    ax.plot([245, 245, 1220, 1220], [392, 430, 430, 392],
            color=ORANGE, linewidth=1.9)
    ax.text(790, 438, "输入并行送往门控 MLP", fontsize=10.5, color=ORANGE,
            ha="center", bbox={"facecolor": BG, "edgecolor": "none", "pad": 2})

    section(ax, 78, 495, "f", "训练 Critic 与 PPO", "Critic 不参与实机动作前向。", RED)
    box(ax, 65, 545, 390, 225, "Critic 输入 [B,270]", (
        "特权观测 269＋vision_flag 1",
        "训练可见：速度、接触、地形等",
        "部署 Actor 不读取此向量",
    ), RED, PALE_RED)
    box(ax, 515, 545, 485, 225, "8 个价值专家", (
        "共享 MLP 270→512→256→1024",
        "groups=8 Conv1D → [B,8,1]",
        "沿用 Actor 的 gate [B,8]",
        "加权和 → V(s) [B,1]",
    ), RED, WHITE)
    box(ax, 1060, 545, 435, 225, "PPO 策略更新", (
        "rollout24、GAE λ0.95、γ0.99",
        "clip0.2；5 epochs、4 mini-batches",
        "策略损失＋价值损失－熵奖励",
        "Actor gate 负载均衡项参与损失",
    ), RED, PALE_RED)
    box(ax, 1555, 545, 425, 225, "估计器独立优化", (
        "速度/足高/地形图/下一观测",
        "分类＋局部重建＋KL＋SwAV",
        "Adam；MCP 码先 detach",
        "PPO 梯度不回流至估计器",
    ), TEAL, PALE_TEAL)
    arrow(ax, (457, 657), (513, 657), color=RED)
    arrow(ax, (1002, 657), (1058, 657), color=RED)
    arrow(ax, (1260, 392), (760, 543), "Actor gate 同时用于 Critic", color=PURPLE, dashed=True)
    footer(ax, w, h, "图 08 · 本地 Actor MoE 与论文 SCOPE-AC 思路相近，但本图只陈述本地代码实际算子。")
    save(fig, "08_parkour_actor_critic")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    overview()
    estimator()
    policies()
    training()
    parkour_network()
    parkour_perception()
    parkour_memory()
    parkour_actor_critic()
    print(f"Rendered 8 figures to {OUT}")

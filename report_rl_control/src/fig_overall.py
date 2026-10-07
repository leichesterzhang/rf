from figlib import Diagram, C

C["grp"] = ("#E6EBF3", "#E6EBF3")
d = Diagram("整体架构", 1080, 420)
it = dict(italic=True, fs=13)
d.panel(8, 8, 1064, 404, "整体框架", "gry_bg", fs=14)

# ---- inputs
d.text(18, 92, 60, 22, "D_{t}", fs=15, italic=True, bold=True)
d.fat([(22, 128), (78, 128)], "blue", bw=10, hl=14, hw=24)
d.text(18, 236, 60, 22, "O_{t}^{10}", fs=15, italic=True, bold=True)
d.fat([(22, 270), (78, 270)], "org", bw=10, hl=14, hw=24)

# ---- shaded group (perception + fusion + estimator)
d.box(86, 42, 552, 326, "", "grp", arc=18, sw=0.5)
VM = d.box(98, 60, 62, 150, "视觉感知模块", "blue", vertical=True, fs=13, sw=1.8, arc=10)
PM = d.box(98, 228, 62, 90, "本体MLP", "org", vertical=True, fs=13, sw=1.8, arc=10)
d.fat([(164, 135), (194, 135)], "blue", bw=9, hl=12, hw=20)
d.fat([(164, 273), (194, 273)], "org", bw=9, hl=12, hw=20)

# token column
d.box(198, 58, 58, 298, "", "wht", dashed=True, sw=1.4, arc=10)
y = 68
for i in range(3):
    d.box(210, y, 22, 22, "", "blue", arc=5, sw=1.4); y += 28
d.text(206, y - 6, 30, 20, "⋮", fs=14, bold=True)
d.text(232, 96, 26, 16, "×16", fs=10, italic=True)
y = 196
for i in range(3):
    d.box(210, y, 22, 22, "", "org", arc=5, sw=1.4); y += 28
d.text(206, y - 6, 30, 20, "⋮", fs=14, bold=True)
d.text(232, 222, 26, 16, "×10", fs=10, italic=True)
d.box(210, 318, 22, 22, "", "grn", arc=5, sw=1.4)
d.text(186, 372, 70, 20, "b_{t}", **it)
d.edge([(221, 372), (221, 342)], None, None, sw=1.6)

TF = d.box(276, 58, 62, 298, "跨模态Transformer", "blue", vertical=True, fs=13, sw=1.8, arc=10)
d.fat([(258, 207), (272, 207)], "blue", bw=8, hl=8, hw=16)
MG = d.box(364, 58, 62, 298, "读写记忆估计器（MATE-GRU）", "org", vertical=True, fs=13, sw=1.8, arc=10)
d.fat([(340, 207), (360, 207)], "org", bw=8, hl=10, hw=18)
d.text(336, 182, 30, 18, "X_{t}", fs=11, italic=True)
# shared-state loop
d.edge([(426, 330), (446, 330), (446, 384), (395, 384), (395, 358)], None, None, sw=1.6)
d.text(450, 360, 60, 18, "s_{t}", fs=12, italic=True, align="left")

# MCP vector
d.fat([(428, 207), (456, 207)], "org", bw=8, hl=10, hw=18)
d.box(462, 70, 64, 276, "", "wht", dashed=True, sw=1.4, arc=8)
segs = [("v̂_{t}", "gry"), ("ĥ^{f}_{t}", "gry"), ("z^{μ}_{t}", "yel"), ("z^{tm}_{t}", "yel"),
        ("c_{t}", "blue"), ("r_{t}", "blue")]
for i, (lab, col) in enumerate(segs):
    d.box(470, 78 + i * 44, 48, 38, lab, col, rounded=False, fs=13, bold=False, italic=True)
d.text(456, 46, 80, 20, "MCP_{t}", fs=12, bold=True, italic=True)
# training decoders (dashed)
d.edge([(526, 185), (560, 185)], None, None, dashed=True, sw=1.6)
d.text(562, 174, 70, 22, "ô_{t+1}", fs=13, italic=True, align="left")
d.edge([(526, 229), (560, 229)], None, None, dashed=True, sw=1.6)
d.text(562, 218, 70, 22, "m̂_{t}", fs=13, italic=True, align="left")
d.text(544, 250, 90, 34, "辅助解码\n（仅训练）", fs=10)

# ---- actor
d.text(640, 72, 56, 22, "o_{t}, b_{t}", fs=13, italic=True, align="left")
d.fat([(644, 104), (692, 104)], "org", bw=10, hl=14, hw=24)
d.fat([(528, 112), (606, 112), (606, 150), (692, 150)], "blue", bw=10, hl=14, hw=24)
AC = d.box(698, 58, 62, 210, "混合专家Actor", "grn", vertical=True, fs=13, sw=1.8, arc=10)
d.fat([(764, 90), (792, 90)], "blue", bw=10, hl=14, hw=24)
d.text(764, 58, 40, 22, "a_{t}", **it)
PD = d.box(798, 58, 62, 210, "PD控制器", "gry", vertical=True, fs=13, sw=1.8, arc=10)
d.fat([(864, 163), (892, 163)], "gry", bw=10, hl=14, hw=24)
SIM = d.box(898, 62, 160, 92, "仿真环境\n（Isaac Gym，4096并行）", "wht", fs=12, sw=1.6, arc=14)
REAL = d.box(898, 176, 160, 92, "四足机器人实机", "wht", fs=12, sw=1.6, arc=14)

# ---- critic (training only)
CR = d.box(698, 300, 162, 56, "非对称Critic\n（仅训练）", "pur", fs=12, dashed=True, sw=1.6, arc=10)
d.text(898, 296, 160, 24, "x^{priv}_{t}（269维）, b_{t}", fs=12, italic=True)
d.edge([(978, 320), (978, 328), (CR.x + CR.w, 328)], None, CR, sw=1.6, dashed=True)
d.edge([(CR.x + 40, CR.y), (CR.x + 40, 272)], CR, None, sw=1.6, dashed=True)
d.text(742, 274, 110, 20, "V_{t}，用于PPO更新", fs=11, align="left")

d.save("out", "fig1_overall")

from figlib import Diagram

d = Diagram("MATE-GRU 读写记忆估计器", 1240, 700)
OUT = "out"
it = dict(italic=True, fs=12)

# ---------------- panels
d.panel(150, 12, 790, 96, "运动分支", "gry_bg", fs=12)
d.panel(150, 120, 700, 300, "共享查询与专家路由", "blue_bg", fs=12)
d.panel(150, 432, 700, 258, "读写记忆专家 i = 1, 2, 3（参数互不共享）", "org_bg", fs=12)
d.panel(930, 280, 300, 410, "共享状态更新与地形预测", "yel", fs=12)

# ---------------- input + bus
X = d.box(14, 320, 104, 76, "融合本体特征\nX_{t}  (10×64)", "wht", fs=12)
bus = 136
d.edge([X.p("r"), (bus, X.cy)], X, None, arrow=False)
d.edge([(bus, 60), (bus, 547)], arrow=False)

# ---------------- motion branch
G = d.box(176, 34, 110, 52, "运动GRU\n1层, 隐状态64", "gry", fs=12)
d.edge([(bus, 60), G.p("l")], None, G)
MH = d.box(340, 34, 110, 52, "运动预测头\n64→64, ELU", "gry", fs=12)
d.edge([G.p("r"), (MH.x, 60)], G, MH)
d.text(288, 38, 50, 20, "h^{m}_{t}", **it)

# ---------------- MCP bar
bx, bw, y0 = 1110, 116, 31
segs = [("v̂_{t}  (3)", "gry"), ("ĥ^{f}_{t}  (4)", "gry"), ("z^{μ}_{t}  (16)", "yel"),
        ("z^{tm}_{t}  (32)", "yel"), ("onehot(c_{t})  (13)", "blue"), ("𝟙[r_{t}≥0.5]  (1)", "blue")]
d.box(bx - 6, y0 - 6, bw + 12, 6 * 33 + 9, "", "wht", dashed=True, sw=1.4, arc=8)
S = [d.box(bx, y0 + 33 * i, bw, 30, lab, col, fs=11, bold=False, rounded=False)
     for i, (lab, col) in enumerate(segs)]
d.text(bx - 10, 2, bw + 20, 20, "MCP_{t}（69维）", fs=12, bold=True)
d.edge([(MH.x + MH.w, S[0].cy), S[0].p("l")], MH, S[0])
d.edge([(MH.x + MH.w, S[1].cy), S[1].p("l")], MH, S[1])
d.text(470, 30, 90, 16, "Linear 64→3", fs=10)
d.text(470, 82, 90, 16, "Linear 64→4", fs=10)
bot = S[-1].y + 30
d.edge([(bx + bw / 2, bot), (bx + bw / 2, bot + 22)], S[-1], None)
d.text(bx - 6, bot + 22, bw + 12, 18, "送入策略网络", fs=11)

# ---------------- shared query pooling + heads
SP = d.box(176, 230, 110, 54, "注意力池化", "blue")
d.edge([(bus, SP.cy), SP.p("l")], None, SP)
d.text(186, 296, 90, 20, "共享查询 q_{0}", fs=11)
d.edge([(231, 296), SP.p("b")], None, SP)
CL = d.box(340, 148, 110, 44, "地形分类头\n64→64→13", "blue", fs=11)
RC = d.box(340, 235, 110, 44, "恢复判别头\n64→64→1", "blue", fs=11)
CT = d.box(340, 328, 110, 44, "上下文头\n64→64, ELU", "blue", fs=11)
j = 312
d.edge([SP.p("r"), (j, SP.cy)], SP, None, arrow=False)
d.edge([(j, CL.cy), (j, CT.cy)], arrow=False)
for n in (CL, CT):
    d.edge([(j, n.cy), n.p("l")], None, n)
d.edge([(j, RC.cy), RC.p("l")], None, RC)
d.text(286, 236, 26, 18, "p_{0}", **it)

# encoders
E32 = d.box(492, 134, 112, 32, "编码 13→32→32", "blue", fs=11, bold=False)
E64c = d.box(492, 174, 112, 32, "编码 13→64→64", "blue", fs=11, bold=False)
E16 = d.box(492, 221, 112, 32, "编码 1→16→16", "blue", fs=11, bold=False)
E64r = d.box(492, 261, 112, 32, "编码 1→64→64", "blue", fs=11, bold=False)
jc = 475
for head, a, b in ((CL, E32, E64c), (RC, E16, E64r)):
    d.edge([head.p("r"), (jc, head.cy)], head, None, arrow=False)
    d.edge([(jc, a.cy), (jc, b.cy)], arrow=False)
    d.edge([(jc, a.cy), a.p("l")], None, a)
    d.edge([(jc, b.cy), b.p("l")], None, b)
d.text(450, 150, 26, 18, "c_{t}", **it)
d.text(450, 237, 26, 18, "r_{t}", **it)

# 64-d encodings -> expert features (drawn first so later lines hop over)
PL = d.circle(420, 547, "+")
d.edge([E64c.p("r"), (622, E64c.cy), (622, 500), (420, 500), PL.p("t")], E64c, PL)
d.edge([E64r.p("r"), (622, E64r.cy)], E64r, None, arrow=False)

# router
CC = d.box(650, 134, 18, 236, "", "wht", rounded=False)
d.edge([E32.p("r"), (CC.x, E32.cy)], E32, CC)
d.edge([E16.p("r"), (CC.x, E16.cy)], E16, CC)
d.edge([CT.p("r"), (CC.x, CT.cy)], CT, CC)
d.edge([(659, 404), (659, 370)], None, CC)
d.text(560, 394, 92, 20, "o_{t}, b_{t}", **it)
d.text(670, 376, 40, 18, "拼接", fs=11)
RT = d.box(694, 216, 104, 70, "路由MLP\n158→128→3\nSoftmax", "blue", fs=11)
d.edge([(668, RT.cy), RT.p("l")], CC, RT)
d.text(668, 230, 28, 16, "158", fs=10)

# ---------------- experts
EP = d.box(176, 520, 110, 54, "注意力池化", "org")
d.edge([(bus, EP.cy), EP.p("l")], None, EP)
d.text(186, 584, 90, 20, "专家查询 q_{i}", fs=11)
d.edge([(231, 584), EP.p("b")], None, EP)
d.edge([EP.p("r"), PL.p("l")], EP, PL)
GC = d.box(530, 520, 100, 54, "GRUCell\n64→64", "org", fs=12)
d.edge([PL.p("r"), GC.p("l")], PL, GC)
d.text(440, 522, 60, 20, "e_{t,i}", **it)
RM = d.box(510, 612, 100, 44, "读MLP\n64→64→64", "org", fs=11)
d.edge([RM.p("t"), (RM.cx, GC.y + GC.h)], RM, GC)
d.text(500, 582, 50, 20, "h^{0}_{t,i}", fs=11, italic=True)
WM = d.box(694, 520, 100, 54, "写MLP\n64→64→64", "org", fs=11)
d.edge([GC.p("r"), WM.p("l")], GC, WM)

# ---------------- mixing by router weights
MDX = d.circle(880, 547, "×")
MHX = d.circle(880, 630, "×")
d.edge([RT.p("r"), (880, RT.cy), MDX.p("t")], RT, MDX)
d.text(884, 228, 30, 18, "g_{t}", **it)
d.edge([(880, 500), (920, 500), (920, 630), MHX.p("r")], None, MHX)
d.edge([(615, GC.y + GC.h), (615, 630), MHX.p("l")], GC, MHX)
d.text(620, 606, 40, 18, "h_{t,i}", fs=11, italic=True)
d.edge([WM.p("r"), MDX.p("l")], WM, MDX)
d.text(800, 526, 40, 18, "Δ_{t,i}", fs=11, italic=True)
# ---------------- state update
d.edge([MHX.p("b"), (880, 658), (950, 658), (950, 337), (962, 337)], MHX, None)
d.text(884, 640, 40, 18, "h̃_{t}", **it)
CA = d.box(956, 512, 136, 70, "候选 F_{c}: 128→64→64\n写入门 F_{u}: 128→64, σ\n输入 [s_{t-1}, Δ̃_{t}]", "yel", fs=10)
d.edge([MDX.p("r"), CA.p("l")], MDX, CA)
d.text(918, 526, 40, 18, "Δ̃_{t}", **it)
LN = d.box(962, 420, 124, 50, "门控写入\nLayerNorm(64)", "yel", fs=12)
d.edge([CA.p("t"), LN.p("b")], CA, LN)
TP = d.box(962, 312, 124, 50, "地形预测头\n128→64, ELU", "yel", fs=12)
d.edge([LN.p("t"), TP.p("b")], LN, TP)
d.text(1032, 386, 34, 18, "s_{t}", **it)
Z = d.box(1120, 590, 90, 40, "延迟一个\n估计周期", "wht", fs=11, bold=False)
d.edge([LN.p("r"), (Z.cx, LN.cy), Z.p("t")], LN, Z)
d.edge([Z.p("b"), (Z.cx, 676), (RM.cx, 676), RM.p("b")], Z, RM)
d.edge([(CA.cx, 676), CA.p("b")], None, CA)
d.text(1040, 652, 50, 18, "s_{t-1}", **it)
d.edge([TP.p("r", 0.3), (1098, TP.y + 15), (1098, S[2].cy), S[2].p("l")], TP, S[2])
d.edge([TP.p("r", 0.7), (1104, TP.y + 35), (1104, S[3].cy), S[3].p("l")], TP, S[3])

d.save(OUT, "fig3_mate_gru")

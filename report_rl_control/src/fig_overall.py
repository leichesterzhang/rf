from figlib import Diagram

d = Diagram("整体架构", 1240, 580)
it = dict(italic=True, fs=12)

# panels
d.panel(150, 14, 420, 320, "视觉感知与跨模态融合", "blue_bg", fs=13)
d.panel(590, 14, 260, 320, "读写记忆估计器（约10 Hz）", "org_bg", fs=13)
d.panel(870, 14, 196, 320, "混合专家策略（50 Hz）", "grn_bg", fs=13)
d.panel(150, 404, 916, 140, "仅训练阶段使用", "pur_bg", fs=13)

# inputs
D = d.box(14, 50, 112, 56, "深度图 D_{t}\n58×87", "wht", fs=12)
O = d.box(14, 180, 112, 56, "本体历史 O_{t}\n10×45", "wht", fs=12)
OB = d.box(14, 344, 112, 56, "当前观测 o_{t}\n视觉标志 b_{t}", "wht", fs=12)

# perception
CNN = d.box(170, 50, 100, 56, "深度CNN\n3层卷积", "blue", fs=12)
d.edge([D.p("r"), CNN.p("l")], D, CNN)
SM = d.box(304, 50, 106, 56, "主动采样\nK = 16", "blue", fs=12)
d.edge([CNN.p("r"), SM.p("l")], CNN, SM)
d.text(268, 82, 40, 18, "88×64", fs=10)
DEC = d.box(304, 130, 106, 46, "局部解码器", "pur", fs=12, dashed=True)
d.edge([(287, 78), (287, DEC.cy), DEC.p("l")], None, DEC, dashed=True)
d.text(304, 178, 106, 18, "→ L_{patch}", fs=11, italic=True)
PM = d.box(170, 190, 100, 56, "本体MLP", "blue", fs=12)
d.edge([O.p("r", 0.5), (PM.x, O.cy)], O, PM)
TF = d.box(440, 110, 110, 150, "跨模态\nTransformer\n(27个token)", "blue", fs=12)
d.edge([SM.p("r"), (425, 78), (425, 140), (TF.x, 140)], SM, TF)
d.edge([(PM.x + PM.w, O.cy), (TF.x, O.cy)], PM, TF)
d.text(268, O.cy - 20, 60, 18, "10×64", fs=10)
d.text(410, 116, 30, 18, "16", fs=10)

# estimator
MG = d.box(608, 104, 124, 110, "运动GRU\n共享查询与路由\n3个读写专家", "org", fs=12)
d.edge([TF.p("r", 0.4), (MG.x, TF.y + 60)], TF, MG)
d.text(554, 150, 52, 18, "X_{t}", **it)
ST = d.box(608, 256, 92, 40, "共享状态 s_{t}", "yel", fs=11)
d.edge([(640, MG.y + MG.h), (640, ST.y)], MG, ST)
d.edge([(675, ST.y), (675, MG.y + MG.h)], ST, MG)
HD = d.box(756, 120, 80, 78, "输出头", "org", fs=12)
d.edge([MG.p("r", 0.5), (HD.x, MG.cy)], MG, HD)
AUX = d.box(752, 246, 88, 50, "辅助解码头", "pur", fs=11, dashed=True)
d.edge([HD.p("b"), AUX.p("t")], HD, AUX, dashed=True)

# actor
CC = d.box(886, 70, 20, 200, "", "wht", rounded=False)
d.text(874, 276, 44, 18, "115", fs=10)
d.edge([HD.p("r"), (CC.x, HD.cy)], HD, CC)
d.text(838, 138, 48, 18, "MCP_{t}", fs=11, italic=True)
AC = d.box(930, 96, 120, 148, "共享MLP\n\n8个动作专家\n\n门控加权", "grn", fs=12)
d.edge([(906, AC.cy), AC.p("l")], CC, AC)

# execution
PD = d.box(1094, 96, 132, 64, "PD关节控制\nq* = q_{0} + 0.2a_{t}", "gry", fs=12)
d.edge([(AC.x + AC.w, PD.cy), PD.p("l")], AC, PD)
d.text(1052, 106, 40, 18, "a_{t}", **it)
RB = d.box(1094, 220, 132, 64, "仿真环境 /\n四足机器人", "gry", fs=12)
d.edge([PD.p("b"), RB.p("t")], PD, RB)
d.text(1162, 172, 60, 30, "τ", **it)

# o_t, b_t lines (vertical branches first)
yo = 372
d.edge([(896, yo), (896, CC.y + CC.h)], None, CC)
d.edge([(718, yo), (718, MG.y + MG.h)], None, MG)
d.edge([(495, yo), (495, TF.y + TF.h)], None, TF)
d.text(498, 300, 26, 18, "b_{t}", **it)

# training-only row
PR = d.box(170, 440, 132, 52, "特权观测 x^{priv}_{t}\n269维 + b_{t}", "wht", fs=11)
CR = d.box(340, 440, 150, 52, "非对称Critic\n8个价值专家", "pur", fs=12)
d.edge([PR.p("r"), CR.p("l")], PR, CR)
PPO = d.box(900, 440, 150, 52, "GAE + PPO\n更新Actor/Critic", "pur", fs=12)
EL = d.box(720, 486, 140, 48, "估计器辅助损失\n独立Adam更新", "pur", fs=11)
d.edge([(AUX.cx, AUX.y + AUX.h), (AUX.cx, EL.y)], AUX, EL, dashed=True)
d.edge([CR.p("r", 0.5), (PPO.x, CR.cy)], CR, PPO)
d.text(500, 446, 40, 18, "V_{t}", **it)
d.edge([(PPO.cx, PPO.y), (PPO.cx, 334)], PPO, None, dashed=True)
d.edge([RB.p("b", 0.3), (RB.x + RB.w * 0.3, 466), PPO.p("r")], RB, PPO)
d.text(1094, 300, 70, 18, "r_{t}, x^{priv}_{t}", fs=11, italic=True)
d.edge([OB.p("r"), (896, yo)], OB, None, arrow=False)
d.text(140, yo - 22, 110, 18, "o_{t}, b_{t}", **it)

# feedback
d.edge([RB.p("b", 0.7), (RB.x + RB.w * 0.7, 564), (70, 564), OB.p("b")], RB, OB)
d.text(420, 546, 420, 18, "传感器反馈：深度相机、IMU、关节编码器（每20 ms）", fs=11)

d.save("out", "fig1_overall")

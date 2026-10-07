from figlib import Diagram

d = Diagram("训练与推理时序", 1240, 440)
it = dict(italic=True, fs=12)
d.panel(8, 8, 1224, 196, "(a) 一次训练迭代（4096个并行环境）", "gry_bg", fs=12)
d.panel(24, 40, 820, 150, "采样阶段：每个环境推进24个策略步", "org_bg", fs=11)

y, h = 84, 64
B1 = d.box(44, y, 150, h, "估计器前向\n刷新步更新MCP与s", "org", fs=11)
B2 = d.box(244, y, 150, h, "Actor采样 a_{t}\nCritic估计 V_{t}", "grn", fs=11)
B3 = d.box(444, y, 150, h, "环境推进20 ms\n（4个5 ms物理步）", "gry", fs=11)
B4 = d.box(644, y, 170, h, "刷新步：估计器\n辅助损失 + Adam", "pur", fs=11)
for a, b in ((B1, B2), (B2, B3), (B3, B4)):
    d.edge([a.p("r"), b.p("l")], a, b)
d.edge([B4.p("b"), (B4.cx, 170), (B1.cx, 170), B1.p("b")], B4, B1)
d.text(300, 172, 260, 16, "重复24步（深度与估计器每5步刷新一次）", fs=10)
B5 = d.box(880, y, 150, h, "GAE计算\n回报与优势", "pur", fs=11)
B6 = d.box(1064, y, 150, h, "PPO更新\n5轮 × 4个小批量", "pur", fs=11)
d.edge([(844, B5.cy), B5.p("l")], None, B5)
d.edge([B5.p("r"), B6.p("l")], B5, B6)
d.edge([B6.p("t"), (B6.cx, 30), (B1.cx, 30), (B1.cx, 40)], B6, None, dashed=True)
d.text(1050, 34, 170, 16, "进入下一次迭代", fs=10)
d.text(860, 160, 180, 16, "98304条样本", fs=10)

d.panel(8, 216, 1224, 216, "(b) 部署推理时序（单台机器人）", "gry_bg", fs=12)
d.text(20, 262, 120, 20, "估计器（10 Hz）", fs=11, bold=True)
d.text(20, 324, 120, 20, "Actor（50 Hz）", fs=11, bold=True)
x0, dx = 170, 100
d.edge([(150, 384), (1200, 384)])
for i in range(11):
    x = x0 + i * dx
    d.edge([(x, 378), (x, 384)], arrow=False)
    d.text(x - 30, 390, 60, 16, f"{i*20} ms", fs=10)
    d.box(x, 318, 46, 30, "π", "grn", fs=12, bold=False)
    if i % 5 == 0 and i < 10:
        E = d.box(x, 250, 120, 40, "估计器前向", "org", fs=11)
        d.edge([(x + 23, 290), (x + 23, 318)], E, None)
for i in range(11):
    if i % 5:
        x = x0 + i * dx
        d.edge([(x + 23, 300), (x + 23, 318)], None, None, dashed=True)
d.edge([(x0 + 23, 300), (x0 + 4 * dx + 23, 300)], dashed=True, arrow=False)
d.edge([(x0 + 5 * dx + 23, 300), (x0 + 9 * dx + 23, 300)], dashed=True, arrow=False)
d.text(320, 410, 640, 18, "实线：估计器刷新后写入MCP_{t}与s_{t}；虚线：其余周期复用缓存的MCP，只用新的o_{t}计算动作", fs=10)
d.save("out", "fig5_train_infer")

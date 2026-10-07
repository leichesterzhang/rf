from figlib import Diagram

d = Diagram("视觉感知模块", 1240, 500)
it = dict(italic=True, fs=12)
d.panel(8, 8, 1224, 484, "视觉感知模块：深度编码、局部重建与主动采样", "blue_bg", fs=13)


def tokens(x, y, n, color="blue", s=20, gap=6, dots_after=None):
    """vertical token stack; returns list of boxes"""
    out = []
    yy = y
    for i in range(n):
        out.append(d.box(x, yy, s, s, "", color, arc=5, sw=1.4))
        yy += s + gap
        if dots_after is not None and i == dots_after:
            d.text(x - 4, yy - 4, s + 8, 20, "⋮", fs=14, bold=True)
            yy += 18
    return out


# ---- input depth + encoder
DI = d.image(24, 84, 150, 100, "depth.png")
d.text(16, 188, 166, 18, "深度图 D_{t}（58×87）", fs=11, bold=True)
ENC = d.box(198, 56, 94, 158, "Enc\n\n3层卷积\n3×3, 步长2\n+LayerNorm", "blue", shape="trap_l", fs=11)
d.edge([(176, 134), (ENC.x, 134)], None, ENC, sw=1.8)
d.text(186, 216, 120, 16, "通道 16→32→64", fs=10)
TK = tokens(318, 64, 5, "blue", dots_after=2)
d.text(296, 222, 64, 30, "88个token\n(8×11×64)", fs=10)
d.edge([(ENC.x + ENC.w, 134), (314, 134)], ENC, None, sw=1.8)

# ---- upper path: origin token -> pool/MLP/sigmoid
d.fat([(346, 64), (552, 64)], "blue", bw=10, hl=16, hw=24)
d.text(410, 42, 120, 18, "原始token", **it)
P1 = d.box(560, 48, 200, 30, "有效token均值池化", "yel", arc=15, fs=12)
P2 = d.box(560, 84, 200, 30, "MLP  64→64→32", "blue", arc=15, fs=12)
P3 = d.box(560, 120, 200, 30, "Sigmoid", "gry", arc=15, fs=12)
d.text(550, 156, 220, 22, "P_{t} = (u_{1},v_{1}) … (u_{16},v_{16})", fs=12, italic=True)
d.fat([(770, 167), (790, 167), (790, 248), (806, 248)], "blue", bw=9, hl=12, hw=20)

# ---- masked token grid with probes
gx, gy, cs = 812, 190, 15
masked = {(r, c) for r in (5, 6) for c in (2, 3, 4)} | {(r, c) for r in (1, 2) for c in (8, 9)}
for r in range(8):
    for c in range(11):
        col = "msk" if (r, c) in masked else "wht"
        d.box(gx + c * cs, gy + r * cs, cs, cs, "", col, rounded=False, sw=0.6)
probes = [(1.5, 3.2), (2.4, 5.6), (3.3, 1.7), (3.8, 7.4), (4.6, 4.5), (5.2, 9.3), (6.4, 6.1), (2.2, 9.6)]
for r, c in probes:
    d.circle(gx + c * cs, gy + r * cs, "", r=4.5, color="blue")
d.text(852, gy + 124, 140, 16, "掩码后的token图（8×11）", fs=10, align="left")

# ---- sampled view (depth with probes)
SV = d.image(812, 40, 165, 110, "depth.png")
for r, c in probes:
    d.circle(812 + c * 15, 40 + r * 13.75, "", r=5, color="blue")
d.fat([(894, 186), (894, 154)], "blue", bw=9, hl=12, hw=20)
d.text(900, 158, 120, 30, "双线性采样\ngrid_sample", fs=10, align="left")

# ---- coordinate encoding + outputs
CE = d.box(1006, 66, 92, 52, "+坐标编码\n2→64→64", "blue", fs=11)
d.edge([(SV.x + SV.w, 92), (CE.x, 92)], None, CE, sw=1.8)
d.fat([(1100, 92), (1132, 92)], "blue", bw=9, hl=12, hw=20)
d.box(1140, 46, 22, 22, "", "grn", arc=5, sw=1.4)
d.text(1166, 46, 60, 22, "标志 b_{t}", fs=10, align="left")
tokens(1140, 78, 4, "blue", s=22, gap=6, dots_after=1)
d.text(1166, 128, 50, 20, "×16", fs=11, italic=True, align="left")
d.text(1110, 222, 120, 34, "送入跨模态\nTransformer", fs=11, bold=True)
d.edge([(1151, 206), (1151, 222)], None, None, sw=1.5)

# ---- lower path: decoder / MSE / patches (training only)
d.edge([(328, 252), (328, 318), (344, 318)], None, None, sw=1.8, dashed=True)
DEC = d.box(346, 248, 84, 140, "Dec\n\n64→128\n→225", "pur", shape="trap_r", fs=11)
for i in range(4):
    for j in range(3):
        d.image(448 + j * 28, 254 + i * 32, 24, 26, f"tile{i * 3 + j}.png")
d.edge([(DEC.x + DEC.w, 318), (444, 318)], DEC, None, sw=1.8, dashed=True)
d.text(436, 384, 110, 18, "重建块 q̂_{i}", fs=11)
d.edge([(540, 318), (604, 318)], None, None, color="#888888", sw=3, start_arrow=True)
d.text(546, 292, 54, 20, "MSE", fs=12, bold=True, italic=True)
OP = d.image(612, 268, 150, 100, "depth_grid.png")
d.text(586, 372, 204, 18, "原始深度块 q_{i}（unfold 15×15）", fs=11)
d.text(586, 390, 204, 18, "L_{patch}，仅训练使用", fs=10, italic=True)

# ---- token mask (bottom), legend
LG = d.box(24, 404, 150, 70, "", "blue", arc=10, sw=1.0)
d.circle(42, 424, "", r=6, color="blue")
d.text(54, 414, 110, 20, "采样点", fs=11, italic=True, align="left")
d.box(36, 446, 14, 14, "", "msk", arc=3, sw=1.0)
d.text(54, 443, 110, 20, "被屏蔽的token", fs=11, italic=True, align="left")
d.fat([(186, 446), (214, 446)], "grn", bw=9, hl=12, hw=20)
d.text(218, 404, 20, 84, "{", fs=48)
d.text(236, 412, 56, 20, "b_{t}=1", fs=11, italic=True)
d.text(236, 452, 56, 20, "b_{t}=0", fs=11, italic=True)
for j in range(7):
    d.box(296 + j * 24, 412, 20, 20, "", "msk" if j in (2, 3, 6) else "blue", arc=5, sw=1.2)
    d.box(296 + j * 24, 452, 20, 20, "", "msk", arc=5, sw=1.2)
d.text(470, 412, 170, 20, "训练时随机遮挡1～3个连续块", fs=10, align="left")
d.text(470, 452, 170, 20, "视觉不可用，全部屏蔽", fs=10, align="left")
d.text(642, 404, 20, 84, "}", fs=48)
d.fat([(668, 446), (840, 446), (840, 318)], "grn", bw=9, hl=12, hw=20)
d.text(680, 422, 150, 18, "有效掩码 m_{t}", fs=11, italic=True)

d.save("out", "fig2_perception")

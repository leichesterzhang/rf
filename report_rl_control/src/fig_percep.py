from figlib import Diagram

d = Diagram("视觉感知与跨模态融合", 1240, 500)
it = dict(italic=True, fs=12)
d.panel(124, 10, 820, 350, "深度编码与主动采样", "blue_bg", fs=12)
d.panel(124, 370, 820, 120, "本体与标志token", "gry_bg", fs=12)

D = d.box(14, 62, 96, 56, "深度图 D_{t}\n58×87", "wht", fs=12)
cx = 140
convs = ["Conv 3×3/2\n16×29×44", "Conv 3×3/2\n32×15×22", "Conv 3×3/2\n64×8×11"]
prev = D
for i, lab in enumerate(convs):
    b = d.box(cx + i * 108, 62, 92, 56, lab, "blue", fs=11)
    d.edge([prev.p("r"), b.p("l")], prev, b)
    prev = b
TM = d.box(470, 62, 96, 56, "token图\n88×64", "blue", fs=12)
d.edge([prev.p("r"), TM.p("l")], prev, TM)

GS = d.box(720, 62, 110, 56, "双线性采样\ngrid_sample", "blue", fs=12)
CM = d.box(612, 150, 92, 50, "坐标MLP\nSigmoid", "blue", fs=12)
PE = d.box(736, 150, 94, 50, "坐标编码\n2→64", "blue", fs=11)
d.edge([TM.p("r"), GS.p("l")], TM, GS)
d.edge([(592, 90), (592, CM.cy), CM.p("l")], None, CM)
d.text(560, 152, 60, 18, "有效均值", fs=10)
d.edge([CM.p("r"), PE.p("l")], CM, PE)
d.edge([(718, CM.cy), (718, 136), (775, 136), (775, GS.y + GS.h)], None, GS)
d.text(722, 120, 56, 16, "P_{t} 16×2", fs=10, italic=True)
d.text(690, 18, 170, 18, "有效掩码 m_{t}（b_{t}、遮挡增强）", fs=10)
d.edge([(775, 36), (775, GS.y)], None, GS)
PL = d.circle(880, 90, "+")
d.edge([GS.p("r"), PL.p("l")], GS, PL)
d.edge([PE.p("r"), (880, PE.cy), PL.p("b")], PE, PL)

# local reconstruction (training)
LD = d.box(463, 225, 110, 50, "局部解码器\n64→128→225", "pur", fs=11, dashed=True)
d.edge([TM.p("b"), LD.p("t")], TM, LD, dashed=True)
UF = d.box(140, 290, 120, 46, "unfold 15×15\n88个深度块 q_{i}", "pur", fs=11, dashed=True)
d.edge([D.p("b"), (D.cx, UF.cy), UF.p("l")], D, UF, dashed=True)
LP = d.box(660, 225, 110, 50, "L_{patch}\n逐块MSE", "pur", fs=12, dashed=True)
d.edge([LD.p("r"), LP.p("l")], LD, LP, dashed=True)
d.text(580, 227, 60, 18, "q̂_{i}", **it)
d.edge([UF.p("r"), (715, UF.cy), LP.p("b")], UF, LP, dashed=True)
d.text(790, 232, 140, 36, "仅训练使用，\n权重0.05", fs=11)

# proprio
O = d.box(14, 398, 96, 50, "本体历史 O_{t}\n10×45", "wht", fs=11)
PM = d.box(150, 398, 110, 50, "本体MLP\n45→64", "gry", fs=12)
d.edge([O.p("r"), PM.p("l")], O, PM)
B = d.box(330, 440, 96, 40, "标志 b_{t}", "wht", fs=11)
FE = d.box(470, 440, 110, 40, "标志嵌入 1→64", "gry", fs=11)
d.edge([B.p("r"), FE.p("l")], B, FE)

TF = d.box(980, 40, 120, 440, "跨模态\nTransformer\n\n多头自注意力\n4头, d_{k}=16\n\nFFN 64→256→64\n\n输入 27×64", "blue", fs=12)
d.edge([PL.p("r"), (TF.x, 90)], PL, TF)
d.text(900, 70, 70, 18, "16×64", fs=11)
d.edge([PM.p("r"), (TF.x, PM.cy)], PM, TF)
d.text(270, 402, 190, 18, "10×64（+时序、模态编码）", fs=10)
d.edge([FE.p("r"), (TF.x, FE.cy)], FE, TF)
d.text(590, 442, 50, 18, "1×64", fs=10)
X = d.box(1128, 215, 100, 90, "取10个\n本体位置\nX_{t} (10×64)", "org", fs=12)
d.edge([TF.p("r", (X.cy - TF.y) / TF.h), X.p("l")], TF, X)
d.save("out", "fig2_perception")

"""Tiny diagram spec -> (.drawio, .svg). Same geometry for both outputs."""
import html, re

FONT_SVG = "Liberation Sans, Arial, WenQuanYi Zen Hei, Microsoft YaHei, sans-serif"
FONT_DIO = "Arial"

# paper-like palette
C = dict(
    blue=("#DAE8FC", "#6C8EBF"), blue_bg=("#F2F6FC", "#7EA0D0"),
    org=("#FFE6CC", "#D79B00"), org_bg=("#FFF9EC", "#E2A93B"),
    yel=("#FFF2CC", "#D6B656"),
    grn=("#D5E8D4", "#82B366"), grn_bg=("#F3F9F1", "#8DBB72"),
    pur=("#E1D5E7", "#9673A6"), pur_bg=("#F8F4FA", "#A58BB5"),
    red=("#F8CECC", "#B85450"),
    gry=("#F5F5F5", "#666666"), gry_bg=("#FAFAFA", "#9A9A9A"),
    wht=("#FFFFFF", "#555555"),
)
ARROW = "#3A3A3A"


def _md_tokens(s):
    """split 'a_{b}c^{d}' into [(text, mode)] mode in '', 'sub', 'sup'."""
    out, i = [], 0
    for m in re.finditer(r"([_^])\{([^}]*)\}", s):
        if m.start() > i:
            out.append((s[i:m.start()], ""))
        out.append((m.group(2), "sub" if m.group(1) == "_" else "sup"))
        i = m.end()
    if i < len(s):
        out.append((s[i:], ""))
    return out


def to_html(s):
    lines = []
    for line in s.split("\n"):
        parts = []
        for t, mode in _md_tokens(line):
            t = html.escape(t)
            parts.append(f"<{mode}>{t}</{mode}>" if mode else t)
        lines.append("".join(parts))
    return "<br>".join(lines)


class Node:
    def __init__(self, d, nid, x, y, w, h, label, kind, color, **kw):
        self.d, self.id, self.x, self.y, self.w, self.h = d, nid, x, y, w, h
        self.label, self.kind, self.color, self.kw = label, kind, color, kw

    # anchors
    def p(self, side, f=0.5):
        x, y, w, h = self.x, self.y, self.w, self.h
        return {"l": (x, y + h * f), "r": (x + w, y + h * f),
                "t": (x + w * f, y), "b": (x + w * f, y + h)}[side]

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


class Diagram:
    def __init__(self, name, W, H):
        self.name, self.W, self.H = name, W, H
        self.items = []
        self.n = 0

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def box(self, x, y, w, h, label="", color="blue", bold=True, fs=13,
            rounded=True, dashed=False, panel=False, align="center",
            valign="middle", fc="#1A1A1A", sw=1.3, italic=False, shape="rect"):
        nd = Node(self, self._id(), x, y, w, h, label, "box", color, bold=bold,
                  fs=fs, rounded=rounded, dashed=dashed, panel=panel,
                  align=align, valign=valign, fc=fc, sw=sw, italic=italic,
                  shape=shape)
        self.items.append(nd)
        return nd

    def panel(self, x, y, w, h, title, color, fs=14):
        return self.box(x, y, w, h, title, color, bold=True, fs=fs,
                        dashed=True, panel=True, align="left", valign="top",
                        sw=1.6, italic=True)

    def circle(self, cx, cy, sym="+", r=11, color="wht"):
        return self.box(cx - r, cy - r, 2 * r, 2 * r, sym, color, bold=True,
                        fs=15, shape="ellipse")

    def text(self, x, y, w, h, label, fs=12, bold=False, fc="#1A1A1A",
             align="center", italic=False, bg=None):
        nd = Node(self, self._id(), x, y, w, h, label, "text", None, fs=fs,
                  bold=bold, fc=fc, align=align, italic=italic, bg=bg)
        self.items.append(nd)
        return nd

    def edge(self, pts, src=None, dst=None, color=ARROW, sw=1.5,
             dashed=False, arrow=True, start_arrow=False):
        e = dict(kind="edge", id=self._id(), pts=[tuple(p) for p in pts],
                 src=src, dst=dst, color=color, sw=sw, dashed=dashed,
                 arrow=arrow, start_arrow=start_arrow)
        self.items.append(e)
        return e

    # orthogonal helper: from point a, go horizontally then vertically etc.
    def hv(self, a, b, src=None, dst=None, mid=None, **kw):
        """a -> b with one bend; mid='h' (horizontal first) or 'v'."""
        if a[0] == b[0] or a[1] == b[1]:
            return self.edge([a, b], src, dst, **kw)
        if mid == "v":
            return self.edge([a, (a[0], b[1]), b], src, dst, **kw)
        return self.edge([a, (b[0], a[1]), b], src, dst, **kw)

    def hvh(self, a, b, xm, src=None, dst=None, **kw):
        return self.edge([a, (xm, a[1]), (xm, b[1]), b], src, dst, **kw)

    def vhv(self, a, b, ym, src=None, dst=None, **kw):
        return self.edge([a, (a[0], ym), (b[0], ym), b], src, dst, **kw)

    # ------------------------------------------------------------ drawio
    def drawio(self):
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        for it in self.items:
            if isinstance(it, Node):
                cells.append(self._dio_node(it))
            else:
                cells.append(self._dio_edge(it))
        body = "\n".join(cells)
        return (f'<mxfile host="drawio"><diagram name="{html.escape(self.name)}" id="d1">'
                f'<mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" '
                f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
                f'pageWidth="{self.W}" pageHeight="{self.H}" math="0" shadow="0">'
                f'<root>\n{body}\n</root></mxGraphModel></diagram></mxfile>')

    def _dio_node(self, n):
        k = n.kw
        st = ["html=1", "whiteSpace=wrap", f"fontFamily={FONT_DIO}",
              f"fontSize={k['fs']}", f"fontColor={k['fc']}"]
        fstyle = (1 if k.get("bold") else 0) + (2 if k.get("italic") else 0)
        st.append(f"fontStyle={fstyle}")
        st.append(f"align={k.get('align','center')}")
        if n.kind == "text":
            st = ["text"] + st + ["verticalAlign=middle", "strokeColor=none"]
            st.append(f"fillColor={k['bg']}" if k.get("bg") else "fillColor=none")
        else:
            fill, stroke = C[n.color]
            if k["shape"] == "ellipse":
                st = ["ellipse"] + st
            else:
                st.append(f"rounded={1 if k['rounded'] else 0}")
                st.append("arcSize=10" if not k["panel"] else "absoluteArcSize=1;arcSize=24")
            st += [f"fillColor={fill}", f"strokeColor={stroke}",
                   f"strokeWidth={k['sw']}", f"verticalAlign={k['valign']}"]
            if k["dashed"]:
                st.append("dashed=1;dashPattern=6 4")
            if k["panel"]:
                st.append("spacingLeft=10;spacingTop=4")
        val = html.escape(to_html(n.label), quote=True)
        return (f'<mxCell id="{n.id}" value="{val}" style="{";".join(st)};" vertex="1" parent="1">'
                f'<mxGeometry x="{n.x}" y="{n.y}" width="{n.w}" height="{n.h}" as="geometry"/></mxCell>')

    def _dio_edge(self, e):
        st = ["html=1", "rounded=0", "edgeStyle=none", "jumpStyle=arc", "jumpSize=8", f"strokeColor={e['color']}",
              f"strokeWidth={e['sw']}",
              "endArrow=block;endFill=1;endSize=5" if e["arrow"] else "endArrow=none",
              "startArrow=block;startFill=1;startSize=5" if e["start_arrow"] else "startArrow=none"]
        if e["dashed"]:
            st.append("dashed=1;dashPattern=5 4")
        pts = e["pts"]
        attrs = ""
        geo = []
        for key, node, p in (("exit", e["src"], pts[0]), ("entry", e["dst"], pts[-1])):
            if node is not None:
                fx = (p[0] - node.x) / node.w
                fy = (p[1] - node.y) / node.h
                st.append(f"{key}X={fx:.4f};{key}Y={fy:.4f};{key}Dx=0;{key}Dy=0;{key}Perimeter=0")
                attrs += f' {"source" if key=="exit" else "target"}="{node.id}"'
        geo.append(f'<mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>')
        geo.append(f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/>')
        if len(pts) > 2:
            geo.append('<Array as="points">' + "".join(
                f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1]) + "</Array>")
        return (f'<mxCell id="{e["id"]}" style="{";".join(st)};" edge="1" parent="1"{attrs}>'
                f'<mxGeometry relative="1" as="geometry">{"".join(geo)}</mxGeometry></mxCell>')

    # ------------------------------------------------------------ svg
    def svg(self):
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
             f'viewBox="0 0 {self.W} {self.H}" font-family="{FONT_SVG}">',
             '<defs>']
        cols = {it["color"] for it in self.items if isinstance(it, dict)}
        for c in cols:
            cid = c.strip("#")
            o.append(f'<marker id="a{cid}" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="7" '
                     f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
        o.append('</defs><rect width="100%" height="100%" fill="#FFFFFF"/>')
        self._vsegs = []
        for it in self.items:
            o.append(self._svg_node(it) if isinstance(it, Node) else self._svg_edge(it))
        o.append("</svg>")
        return "\n".join(o)

    def _svg_text(self, n, x0, y0, w, h, anchor, valign):
        k = n.kw
        fs = k["fs"]
        lines = n.label.split("\n")
        lh = fs * 1.22
        total = lh * len(lines)
        if valign == "top":
            y = y0 + 6 + fs
        else:
            y = y0 + (h - total) / 2 + fs * 0.88
        if anchor == "start":
            x = x0 + 10
        else:
            x = x0 + w / 2
        weight = "bold" if k.get("bold") else "normal"
        style = "italic" if k.get("italic") else "normal"
        out = []
        for i, line in enumerate(lines):
            spans = []
            for t, mode in _md_tokens(line):
                t = html.escape(t)
                if mode == "sub":
                    spans.append(f'<tspan font-size="{fs*0.72:.1f}" dy="{fs*0.28:.1f}">{t}</tspan><tspan dy="{-fs*0.28:.1f}">​</tspan>')
                elif mode == "sup":
                    spans.append(f'<tspan font-size="{fs*0.72:.1f}" dy="{-fs*0.38:.1f}">{t}</tspan><tspan dy="{fs*0.38:.1f}">​</tspan>')
                else:
                    spans.append(f"<tspan>{t}</tspan>")
            out.append(f'<text x="{x:.1f}" y="{y + i*lh:.1f}" font-size="{fs}" font-weight="{weight}" '
                       f'font-style="{style}" fill="{k["fc"]}" text-anchor="{anchor}">{"".join(spans)}</text>')
        return "".join(out)

    def _svg_node(self, n):
        k = n.kw
        s = ""
        if n.kind == "text":
            if k.get("bg"):
                s += f'<rect x="{n.x}" y="{n.y}" width="{n.w}" height="{n.h}" fill="{k["bg"]}"/>'
            anchor = {"left": "start", "center": "middle"}[k.get("align", "center")]
            return s + self._svg_text(n, n.x, n.y, n.w, n.h, anchor, "middle")
        fill, stroke = C[n.color]
        dash = ' stroke-dasharray="6 4"' if k["dashed"] else ""
        if k["shape"] == "ellipse":
            s += (f'<ellipse cx="{n.cx}" cy="{n.cy}" rx="{n.w/2}" ry="{n.h/2}" fill="{fill}" '
                  f'stroke="{stroke}" stroke-width="{k["sw"]}"/>')
            # center symbol precisely
            fs = k["fs"]
            s += (f'<text x="{n.cx}" y="{n.cy + fs*0.35:.1f}" font-size="{fs}" font-weight="bold" '
                  f'text-anchor="middle" fill="#1A1A1A">{html.escape(n.label)}</text>')
            return s
        if k["panel"]:
            r = 12
        else:
            r = min(n.w, n.h) * 0.10 if k["rounded"] else 0
        s += (f'<rect x="{n.x}" y="{n.y}" width="{n.w}" height="{n.h}" rx="{r:.1f}" fill="{fill}" '
              f'stroke="{stroke}" stroke-width="{k["sw"]}"{dash}/>')
        anchor = "start" if k["align"] == "left" else "middle"
        return s + self._svg_text(n, n.x, n.y, n.w, n.h, anchor, k["valign"])

    def _svg_edge(self, e):
        P = e["pts"]
        d = f"M{P[0][0]},{P[0][1]}"
        R = 5
        for (x1, y1), (x2, y2) in zip(P, P[1:]):
            if y1 == y2 and x1 != x2:
                sgn = 1 if x2 > x1 else -1
                xs = sorted(vx for vx, ya, yb in self._vsegs
                            if min(x1, x2) + R + 1 < vx < max(x1, x2) - R - 1 and ya + 1 < y1 < yb - 1)
                xs = xs if sgn > 0 else xs[::-1]
                for vx in xs:
                    d += f" L{vx - sgn*R},{y1} A{R},{R} 0 0 {1 if sgn > 0 else 0} {vx + sgn*R},{y1}"
            d += f" L{x2},{y2}"
        for (x1, y1), (x2, y2) in zip(P, P[1:]):
            if x1 == x2 and y1 != y2:
                self._vsegs.append((x1, min(y1, y2), max(y1, y2)))
        pts = None
        cid = e["color"].strip("#")
        dash = ' stroke-dasharray="5 4"' if e["dashed"] else ""
        me = f' marker-end="url(#a{cid})"' if e["arrow"] else ""
        ms = f' marker-start="url(#a{cid})"' if e["start_arrow"] else ""
        return (f'<path d="{d}" fill="none" stroke="{e["color"]}" stroke-width="{e["sw"]}"'
                f' stroke-linejoin="round"{dash}{me}{ms}/>')

    def save(self, outdir, stem):
        import os
        os.makedirs(outdir, exist_ok=True)
        open(f"{outdir}/{stem}.drawio", "w", encoding="utf-8").write(self.drawio())
        open(f"{outdir}/{stem}.svg", "w", encoding="utf-8").write(self.svg())
